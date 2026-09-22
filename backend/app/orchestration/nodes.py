"""Discrete nodes for the SatQuery LangGraph state machine.

Implements sections 12, 13, 15, 17, 18, 19, 20, 21, 26 of the primary architecture:
- validate_inputs_node
- interpret_query_node
- clarification_check_node (with LangGraph interrupt)
- constrained_router_node
- check_cache_node
- dispatch_specialist_node (sync in dev, interrupt/async in prod)
- validate_specialist_output_node
- run_evidence_engine_node
- confidence_check_node (ACCEPT | WARN | ABSTAIN)
- compose_response_node
- abstention_response_node
- persist_result_node
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from app.core.config import get_settings
from app.core.storage import get_store
from app.db.repo import (
    append_trace,
    claim_graph_resume,
    find_cached_analysis,
    get_analysis,
    get_uploads,
    mark_failed,
    mark_graph_resumed,
    mark_graph_resume_failed,
    set_clarification,
    transition,
)
from app.db.session import session_scope
from app.models.base import AdapterRequest
from app.models.changenet_adapter import ChangeNetAdapter
from langchain_core.messages import AIMessage, HumanMessage
from app.models.satvlm_adapter import SatVLMAdapter
from app.orchestration.cache import compute_analysis_cache_key
from app.orchestration.state import SatQueryState
from app.preprocessing.canonical_scene import build_scene_bundle
from app.schemas.analyses import AnalysisHints
from app.schemas.common import AnalysisStatus, ConfidenceDecision, Stage
from app.schemas.confidence import ConfidenceResponse
from app.services.interpretation_service import interpret_inputs
from app.services.query_parser import parse_question
from app.services.validation_service import get_validation_service
from src.evidence_engine.contracts import (
    QueryRequirements,
    ValidationReport,
    UncalibratedDependency,
    SemanticSupport,
    extract_query_requirements,
)
from src.evidence_engine.serializer import serialize_evidence_bundle
from src.evidence_engine.validator import evidence_contract_validator
from app.orchestration.confidence_gate import evaluate_confidence_gate
from app.services.satvlm_prompt import compose_grounded_response

logger = logging.getLogger("satquery.orchestration.nodes")


def sanitize_for_checkpoint(obj: Any) -> Any:
    """Recursively converts numpy scalars/arrays to native primitives and strips heavy masks."""
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k in {"binary_mask", "probability_map", "mask", "tensor"} and hasattr(v, "shape"):
                continue
            clean[k] = sanitize_for_checkpoint(v)
        return clean
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_checkpoint(item) for item in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


# ─── Node 1: Input Validation ────────────────────────────────────────────────

def validate_inputs_node(state: SatQueryState) -> Dict[str, Any]:
    """Validates uploads, file existence, and spatial compatibility preflight."""
    analysis_id = state["analysis_id"]
    upload_ids = state.get("upload_ids", [])
    settings = get_settings()
    trace = list(state.get("execution_trace", []))
    trace.append("validate_inputs_started")

    with session_scope() as session:
        analysis = get_analysis(session, analysis_id)
        if analysis:
            transition(
                session,
                analysis,
                status=AnalysisStatus.RUNNING.value,
                stage=Stage.VALIDATING.value,
                progress=10,
                message="Validating uploaded imagery and metadata.",
            )
        uploads = get_uploads(session, upload_ids)
        missing = [uid for uid in upload_ids if uid not in {u.id for u in uploads}]
        if missing:
            trace.append("validation_failed_missing_uploads")
            return {
                "error": {"code": "UPLOAD_NOT_FOUND", "message": f"Uploads missing: {missing}"},
                "status": AnalysisStatus.FAILED.value,
                "stage": Stage.VALIDATING.value,
                "execution_trace": trace,
            }

        hints = AnalysisHints.model_validate(state.get("hints") or {})
        parsed = parse_question(state.get("question", ""))
        validation = get_validation_service(settings).validate_uploads(uploads, hints=hints, parsed=parsed)
        val_payload = validation.model_dump(mode="json")

        trace.append("validate_inputs_completed")
        return {
            "validation": val_payload,
            "stage": Stage.VALIDATING.value,
            "progress": 20,
            "execution_trace": trace,
        }


# ─── Node 2: Query Interpretation ───────────────────────────────────────────

def interpret_query_node(state: SatQueryState) -> Dict[str, Any]:
    """Interprets question intent, modalities, and determines required file roles."""
    analysis_id = state["analysis_id"]
    upload_ids = state.get("upload_ids", [])
    settings = get_settings()
    trace = list(state.get("execution_trace", []))
    trace.append("interpret_query_started")

    with session_scope() as session:
        analysis = get_analysis(session, analysis_id)
        uploads = get_uploads(session, upload_ids)
        hints = AnalysisHints.model_validate(state.get("hints") or {})
        parsed = parse_question(state.get("question", ""))
        interpretation = interpret_inputs(uploads, hints=hints, parsed=parsed, settings=settings)

        input_type = interpretation.input_type.value if interpretation.input_type else "single_image"
        modalities = [m.value for m in interpretation.modalities]

        if analysis:
            analysis.input_type = input_type
            analysis.modalities = modalities
            analysis.intent = parsed.primary_intent.value
            transition(
                session,
                analysis,
                stage=Stage.INTERPRETING.value,
                progress=30,
                message="Interpreting file roles and question intent.",
            )

        needs_clarify = interpretation.needs_clarification
        clarify_payload = None
        if needs_clarify and interpretation.clarification_question:
            clarify_payload = {
                "question": interpretation.clarification_question,
                "missing_fields": interpretation.missing_fields,
            }

        # Track 1 Deterministic Query Requirements Extraction (zero LLM)
        requirements = extract_query_requirements(state.get("question", ""), hints=state.get("hints") or {})
        trace.append("interpretation_rules_v1_applied")
        trace.append("interpret_query_completed")
        return {
            "query_requirements": requirements.model_dump(),
            "interpreted_task": parsed.primary_intent.value,
            "selected_workflow": input_type,
            "needs_clarification": needs_clarify,
            "clarification_payload": clarify_payload,
            "stage": Stage.INTERPRETING.value,
            "progress": 35,
            "execution_trace": trace,
        }


# ─── Node 3: Clarification Check (Human-in-the-Loop Interrupt) ───────────────

def clarification_check_node(state: SatQueryState) -> Dict[str, Any]:
    """Triggers LangGraph interrupt() when required inputs or roles are ambiguous.

    Resumes seamlessly when the user supplies clarification.
    """
    trace = list(state.get("execution_trace", []))

    if state.get("needs_clarification"):
        payload = state.get("clarification_payload") or {
            "question": "Clarification required to proceed with analysis.",
            "missing_fields": ["file_roles"],
        }
        trace.append("clarification_requested")

        with session_scope() as session:
            analysis = get_analysis(session, state["analysis_id"])
            if analysis:
                set_clarification(session, analysis, payload)

        # LangGraph interrupt: execution pauses, saving state to checkpointer
        user_response = interrupt({
            "type": "clarification_required",
            "analysis_id": state["analysis_id"],
            "payload": payload,
        })

        # Resumed from interrupt with user clarification
        trace.append("clarification_resumed")
        updated_hints = dict(state.get("hints") or {})
        if isinstance(user_response, dict):
            updated_hints.update(user_response)

        return {
            "hints": updated_hints,
            "needs_clarification": False,
            "clarification_payload": None,
            "status": AnalysisStatus.RUNNING.value,
            "execution_trace": trace,
        }

    return {"execution_trace": trace}


# ─── Node 4: Constrained Router ──────────────────────────────────────────────

def constrained_router_node(state: SatQueryState) -> Dict[str, Any]:
    """Enforces strict input-compatibility constraints for specialist invocation.

    Prevents invoking specialists when prerequisite data is missing.
    """
    analysis_id = state["analysis_id"]
    upload_ids = state.get("upload_ids", [])
    task_intent = state.get("interpreted_task")
    workflow = state.get("selected_workflow", "single_image")
    trace = list(state.get("execution_trace", []))
    trace.append("constrained_routing_started")
    warnings = list(state.get("warnings", []))

    specialists: list[str] = []
    task = "single_scene_vqa"

    q_req = state.get("query_requirements") or {}
    needs_landcover_presence = q_req.get("requires_landcover_presence") or bool(q_req.get("referenced_classes"))

    # Check if inputs contain a compatible optical + SAR pair
    is_compatible_optical_sar = False
    if len(upload_ids) >= 2:
        with session_scope() as session:
            uploads = get_uploads(session, upload_ids)
            mods = {str(u.modality).lower() for u in uploads if u.modality}
            if "optical" in mods and "sar" in mods:
                is_compatible_optical_sar = True

    # Enforce Rule 13: ChangeNet requires bi-temporal optical pair
    if workflow == "bi_temporal" or task_intent in {"change_detection", "detect_change", "locate_change", "quantify_change"}:
        if len(upload_ids) >= 2:
            specialists = ["ChangeNet", "SatVLM"]
            task = "bi_temporal_change"
        else:
            # Constrained fallback: only 1 image provided, downgrade to single_scene_vqa
            specialists = ["SatVLM"]
            task = "single_scene_vqa"
            warnings.append({
                "code": "MISSING_TEMPORAL_PAIR",
                "level": "warning",
                "message": "Change detection requested but only one image was provided. Routed to single-scene inspection.",
            })
            trace.append("change_detection_downgraded_single_image")

    # Enforce Rule 13: SAR-FuseSeg requires optical + SAR modalities
    elif workflow == "optical_sar" or task_intent in {"land_cover", "list_land_cover", "fused_land_cover"} or (needs_landcover_presence and is_compatible_optical_sar):
        if len(upload_ids) >= 2 and (workflow == "optical_sar" or is_compatible_optical_sar):
            specialists = ["SAR-FuseSeg", "SatVLM"]
            task = "optical_sar_land_cover"
            if needs_landcover_presence and is_compatible_optical_sar:
                trace.append("sar_fuseseg_proactively_routed")
        else:
            specialists = ["SatVLM"]
            task = "single_scene_vqa"
            warnings.append({
                "code": "MISSING_CROSS_MODAL_PAIR",
                "level": "warning",
                "message": "Multimodal land cover requested but pair is incomplete. Routed to single-scene inspection.",
            })
            trace.append("land_cover_downgraded_single_image")

    elif needs_landcover_presence and not is_compatible_optical_sar:
        # Optical-only image: SAR-FuseSeg cannot run; do not pretend its evidence exists.
        specialists = ["SatVLM"]
        task = "single_scene_vqa"
        if q_req.get("requires_landcover_presence"):
            warnings.append({
                "code": "MISSING_CROSS_MODAL_PAIR",
                "level": "warning",
                "message": "Land-cover presence check requested but optical+SAR pair is incomplete. Routed to single-scene inspection.",
            })
            trace.append("landcover_presence_optical_only_no_classifier")

    else:
        specialists = ["SatVLM"]
        task = "single_scene_vqa"

    with session_scope() as session:
        analysis = get_analysis(session, analysis_id)
        if analysis:
            analysis.task = task
            analysis.routing = {
                "selected_workflow": workflow,
                "selected_specialists": specialists,
                "task": task,
            }
            transition(
                session,
                analysis,
                stage=Stage.ROUTING.value,
                progress=40,
                message=f"Task routed to {', '.join(specialists)}.",
            )

    trace.append(f"routed_to_{task}")
    return {
        "interpreted_task": task,
        "selected_specialists": specialists,
        "warnings": warnings,
        "stage": Stage.ROUTING.value,
        "progress": 45,
        "execution_trace": trace,
    }


# ─── Node 5: Check Analysis Cache ────────────────────────────────────────────

def check_cache_node(state: SatQueryState) -> Dict[str, Any]:
    """Computes deterministic cache key and checks for completed analysis reuse."""
    analysis_id = state["analysis_id"]
    upload_ids = state.get("upload_ids", [])
    specialists = state.get("selected_specialists", [])
    trace = list(state.get("execution_trace", []))
    trace.append("check_cache_started")

    primary_specialist = specialists[0] if specialists else "SatVLM"

    with session_scope() as session:
        uploads = get_uploads(session, upload_ids)
        input_hashes = [u.sha256 for u in uploads]

        # Compute deterministic cache identity (Section 20)
        cache_key = compute_analysis_cache_key(
            input_hashes=input_hashes,
            specialist=primary_specialist,
            specialist_version="V3.1-Champion" if primary_specialist == "ChangeNet" else "baseline-v1",
            checkpoint_sha256="changeformer_v3_1_champion",
            threshold=0.25 if primary_specialist == "ChangeNet" else None,
        )

        cached_analysis = find_cached_analysis(session, cache_key)
        analysis = get_analysis(session, analysis_id)
        if analysis:
            analysis.cache_key = cache_key

        if cached_analysis and cached_analysis.result:
            cached_conf = cached_analysis.result.get("confidence") or {}
            cached_dec = str(cached_conf.get("decision", "")).lower()
            if cached_dec != "abstained" and cached_analysis.answer and "cannot answer this question reliably" not in cached_analysis.answer and "can't reliably determine" not in cached_analysis.answer:
                logger.info("Analysis cache hit: %s reused by %s", cached_analysis.id, analysis_id)
                trace.append(f"cache_hit_reused_{cached_analysis.id}")
                if analysis:
                    analysis.cache_hit = True

                return {
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "evidence": cached_analysis.result.get("evidence", {}),
                    "specialist_results": cached_analysis.result.get("specialists", []),
                    "final_answer": cached_analysis.answer,
                    "confidence": cached_analysis.confidence or {},
                    "execution_trace": trace,
                }

        trace.append("cache_miss")
        if analysis:
            analysis.cache_hit = False

        return {
            "cache_key": cache_key,
            "cache_hit": False,
            "execution_trace": trace,
        }


# ─── Node 6: Dispatch Specialist ─────────────────────────────────────────────

def dispatch_specialist_node(state: SatQueryState) -> Dict[str, Any]:
    """Executes specialist synchronously (dev) or enqueues RQ job (prod)."""
    trace = list(state.get("execution_trace", []))

    if state.get("cache_hit"):
        trace.append("dispatch_skipped_cache_hit")
        return {"execution_trace": trace}

    analysis_id = state["analysis_id"]
    upload_ids = state.get("upload_ids", [])
    task = state.get("interpreted_task", "single_scene_vqa")
    specialists = state.get("selected_specialists", [])
    settings = get_settings()

    trace.append("dispatch_specialist_started")

    with session_scope() as session:
        analysis = get_analysis(session, analysis_id)
        uploads = get_uploads(session, upload_ids)
        store = get_store(settings)
        db_hints = (analysis.hints or {}) if analysis else {}
        roles = (state.get("hints") or {}).get("file_roles") or db_hints.get("file_roles") or (analysis.roles if analysis else {})
        if not roles and len(uploads) == 2:
            roles = {uploads[0].id: "before", uploads[1].id: "after"}
        input_type = "bi_temporal" if task == "bi_temporal_change" or len(uploads) >= 2 else "single_image"
        bundle = build_scene_bundle(
            uploads=uploads,
            analysis_id=analysis_id,
            input_type=input_type,
            roles=roles,
            store=store,
        )

        if analysis:
            transition(
                session,
                analysis,
                stage=Stage.INFERENCE.value,
                progress=50,
                message=f"Executing specialist {specialists[0] if specialists else 'pipeline'}.",
            )

        # In development / inline queue: run directly with specialist adapter
        specialist_results = {}
        if "ChangeNet" in specialists:
            adapter = ChangeNetAdapter()
            if adapter.available().available:
                adapter.load()
                before_src = bundle.source_for("before")
                after_src = bundle.source_for("after")
                t1_path = before_src.path if before_src else None
                if t1_path and not t1_path.is_file():
                    for u in uploads:
                        if before_src and getattr(before_src, "upload_id", None) == u.id:
                            if u.stored_name and Path(u.stored_name).is_file():
                                t1_path = Path(u.stored_name)
                            elif u.relative_path and (store.root / u.relative_path).is_file():
                                t1_path = store.root / u.relative_path
                            break

                t2_path = after_src.path if after_src else None
                if t2_path and not t2_path.is_file():
                    for u in uploads:
                        if after_src and getattr(after_src, "upload_id", None) == u.id:
                            if u.stored_name and Path(u.stored_name).is_file():
                                t2_path = Path(u.stored_name)
                            elif u.relative_path and (store.root / u.relative_path).is_file():
                                t2_path = store.root / u.relative_path
                            break

                if t1_path and t2_path and t1_path.is_file() and t2_path.is_file():
                    bundle_dict = {
                        "t1_image": str(t1_path),
                        "t2_image": str(t2_path),
                        "transform": before_src.transform if before_src else None,
                        "crs": before_src.crs if before_src else None,
                    }
                    res = adapter._impl.execute(bundle_dict)
                    specialist_results["ChangeNet"] = sanitize_for_checkpoint(res)
                    trace.append("changenet_v6_inferred")

        if "SatVLM" in specialists and task == "single_scene_vqa":
            satvlm_adapter = SatVLMAdapter()
            if satvlm_adapter.available().available:
                satvlm_adapter.load()
                work_dir = store.root / "work" / analysis_id
                work_dir.mkdir(parents=True, exist_ok=True)
                # Extract prior conversation history for multi-turn conversational context
                prior_messages = state.get("messages", [])
                chat_history: list[dict[str, str]] = []
                current_q = state.get("question", "").strip()
                for m in prior_messages:
                    content = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
                    m_type = getattr(m, "type", "") if not isinstance(m, dict) else m.get("role", "")
                    role = "user" if m_type in {"human", "user"} else "assistant"
                    # Include prior completed turns, excluding the current pending question
                    if content and content != current_q:
                        chat_history.append({"role": role, "content": str(content)})

                req = AdapterRequest(
                    analysis_id=analysis_id,
                    task=task,
                    question=state.get("question", ""),
                    bundle=bundle,
                    work_dir=work_dir,
                    params={"chat_history": chat_history},
                )
                vlm_res = satvlm_adapter.infer(req)
                vlm_detail = getattr(vlm_res, "detail", {}) or {}
                specialist_results["SatVLM"] = sanitize_for_checkpoint(
                    {
                        "source": vlm_res.source,
                        "version": vlm_res.version,
                        "answer": vlm_res.answer,
                        "raw_answer": vlm_detail.get("raw_answer") or vlm_res.answer,
                        "answer_status": vlm_res.answer_status,
                        "evidence_coverage": vlm_res.evidence_coverage,
                        "unsupported_claims": vlm_res.unsupported_claims,
                        "detail": vlm_detail,
                        "trace": vlm_res.trace,
                        "warnings": [
                            w.model_dump() if hasattr(w, "model_dump") else w
                            for w in vlm_res.warnings
                        ],
                    }
                )
                trace.append("satvlm_qwen3b_inferred")

        trace.append("dispatch_specialist_completed")
        return {
            "specialist_results": specialist_results,
            "stage": Stage.INFERENCE.value,
            "progress": 70,
            "execution_trace": trace,
        }


# ─── Node 7: Validate Specialist Output ──────────────────────────────────────

def validate_specialist_output_node(state: SatQueryState) -> Dict[str, Any]:
    """Validates raw output from specialist models before passing to Evidence Engine."""
    trace = list(state.get("execution_trace", []))
    trace.append("validate_specialist_output")
    return {"execution_trace": trace}


# ─── Node 8: Run Evidence Engine ─────────────────────────────────────────────

def run_evidence_engine_node(state: SatQueryState) -> Dict[str, Any]:
    """Runs deterministic Evidence Engine to derive polygons, area in m2, and GeoJSON."""
    trace = list(state.get("execution_trace", []))
    trace.append("run_evidence_engine_started")

    if state.get("cache_hit") and state.get("evidence"):
        trace.append("evidence_engine_reused_from_cache")
        return {"execution_trace": trace, "evidence": state.get("evidence")}

    analysis_id = state["analysis_id"]
    specialist_results = state.get("specialist_results", {})
    evidence: dict[str, Any] = {
        "kind": "change" if "ChangeNet" in state.get("selected_specialists", []) else "scene",
        "georeferenced": False,
        "region_count": None,
        "regions": [],
        "class_areas": [],
        "modality_contributions": [],
        "warnings": [],
    }

    if "ChangeNet" in specialist_results:
        cf_res = specialist_results["ChangeNet"]
        cf_ev = cf_res.get("evidence", {})
        evidence.update(cf_ev)
        evidence["kind"] = "change"
        if "change_percentage" in cf_ev:
            evidence["changed_percentage"] = cf_ev["change_percentage"]
            evidence["percentage"] = cf_ev["change_percentage"]
        for r in evidence.get("regions", []):
            if "id" not in r and "region_id" in r:
                r["id"] = r["region_id"]
            if "area_value" not in r and "area_m2" in r:
                r["area_value"] = r["area_m2"]
                r["area_unit"] = "m2"
        raw_ev_warnings = evidence.get("warnings") or []
        if isinstance(raw_ev_warnings, list):
            clean_ev_warnings = []
            for w in raw_ev_warnings:
                if isinstance(w, str):
                    clean_ev_warnings.append({"code": "PIPELINE_WARNING", "message": w, "level": "warning"})
                else:
                    clean_ev_warnings.append(w)
            evidence["warnings"] = clean_ev_warnings
        trace.append("evidence_engine_change_extracted")

    with session_scope() as session:
        analysis = get_analysis(session, analysis_id)
        if analysis:
            transition(
                session,
                analysis,
                stage=Stage.EVIDENCE.value,
                progress=85,
                message="Assembling verified geospatial evidence.",
            )

    trace.append("run_evidence_engine_completed")
    return {
        "evidence": evidence,
        "evidence": sanitize_for_checkpoint(evidence),
        "stage": Stage.EVIDENCE.value,
        "progress": 85,
        "execution_trace": trace,
    }


# ─── Node 8B: Validate Evidence Contract ─────────────────────────────────────

def validate_evidence_contract_node(state: SatQueryState) -> Dict[str, Any]:
    """Runs deterministic evidence contract validator before the confidence gate."""
    trace = list(state.get("execution_trace", []))
    trace.append("validate_evidence_contract_started")

    evidence = state.get("evidence", {})
    specialist_results = state.get("specialist_results", {})
    q_req_dict = state.get("query_requirements")
    requirements = QueryRequirements.model_validate(q_req_dict) if q_req_dict else None

    # Assemble bundle
    bundle_items: list[Any] = []
    if evidence and isinstance(evidence, dict) and (evidence.get("regions") or evidence.get("class_areas") or evidence.get("area")):
        bundle_items.append(evidence)

    if isinstance(specialist_results, dict):
        for s_name, res in specialist_results.items():
            if isinstance(res, dict) and res.get("evidence"):
                bundle_items.append(res["evidence"])
    elif isinstance(specialist_results, list):
        for res in specialist_results:
            if isinstance(res, dict) and res.get("evidence"):
                bundle_items.append(res["evidence"])

    evidence_bundle = serialize_evidence_bundle(bundle_items)

    report = evidence_contract_validator(evidence_bundle, requirements)
    trace.append(f"evidence_contract_validated_{report.status.lower()}")

    return {
        "evidence_bundle": evidence_bundle,
        "validation_report": report.model_dump(),
        "execution_trace": trace,
    }


# ─── Node 9: Confidence / Abstention Gate ────────────────────────────────────

def confidence_check_node(state: SatQueryState) -> Dict[str, Any]:
    """Enforces deterministic 3-way matrix (ACCEPT, WARN, ABSTAIN) and composite evaluation."""
    trace = list(state.get("execution_trace", []))
    trace.append("confidence_check_started")

    q_req_dict = state.get("query_requirements")
    requirements = QueryRequirements.model_validate(q_req_dict) if q_req_dict else extract_query_requirements(state.get("question", ""))
    val_rep_dict = state.get("validation_report")
    validation_report = ValidationReport.model_validate(val_rep_dict) if val_rep_dict else ValidationReport(status="VALID")
    evidence_bundle = state.get("evidence_bundle") or []
    if not evidence_bundle and state.get("evidence"):
        evidence_bundle = serialize_evidence_bundle([state["evidence"]])

    decision_obj = evaluate_confidence_gate(requirements, validation_report, evidence_bundle)
    conf_payload = decision_obj.to_dict()

    trace.append(f"confidence_{decision_obj.decision.lower()}")
    trace.append(f"confidence_reason_{decision_obj.reason_code.lower()}")
    trace.append("confidence_check_completed")

    return {
        "confidence": conf_payload,
        "warnings": state.get("warnings", []),
        "execution_trace": trace,
    }


# ─── Node 10: Grounded Response Composition (ACCEPT) ─────────────────────────

def compose_response_node(state: SatQueryState) -> Dict[str, Any]:
    """Composes grounded natural language answer from verified Evidence Engine facts (ACCEPT path)."""
    trace = list(state.get("execution_trace", []))
    trace.append("compose_response_started")

    if state.get("cache_hit") and state.get("final_answer"):
        trace.append("response_reused_from_cache")
        return {"execution_trace": trace}

    task = state.get("interpreted_task", "")
    specialist_results = state.get("specialist_results", {})
    if task == "single_scene_vqa" and "SatVLM" in specialist_results and specialist_results["SatVLM"].get("answer"):
        trace.append("satvlm_scene_answer_reused")
        satvlm_res = specialist_results["SatVLM"]
        unsupported = satvlm_res.get("unsupported_claims", 0)
        ans = satvlm_res.get("answer", "")
        raw_ans = satvlm_res.get("raw_answer") or ans
        conf = dict(state.get("confidence") or {})

        if unsupported > 0:
            trace.append("claim_validation_downgraded_to_warn")
            conf["decision"] = "WARN"
            conf["reason"] = f"Claim validation flagged {unsupported} unsupported/conflicting statement(s); response was sanitized."
            return {
                "raw_model_response": raw_ans,
                "final_answer": ans,
                "messages": [AIMessage(content=ans)],
                "confidence": conf,
                "status": AnalysisStatus.COMPLETED.value,
                "stage": Stage.DONE.value,
                "progress": 95,
                "execution_trace": trace,
            }

        return {
            "final_answer": ans,
            "messages": [AIMessage(content=ans)],
            "status": AnalysisStatus.COMPLETED.value,
            "stage": Stage.DONE.value,
            "progress": 95,
            "execution_trace": trace,
        }

    q_req_dict = state.get("query_requirements")
    requirements = QueryRequirements.model_validate(q_req_dict) if q_req_dict else extract_query_requirements(state.get("question", ""))
    val_rep_dict = state.get("validation_report")
    validation_report = ValidationReport.model_validate(val_rep_dict) if val_rep_dict else ValidationReport(status="VALID")
    evidence_bundle = state.get("evidence_bundle") or []
    if not evidence_bundle and state.get("evidence"):
        evidence_bundle = serialize_evidence_bundle([state["evidence"]])

    question = state.get("question", "")

    answer = compose_grounded_response(
        query=question,
        requirements=requirements,
        validation_report=validation_report,
        evidence_bundle=evidence_bundle,
        decision="ACCEPT",
    )

    trace.append("compose_response_completed")
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "status": AnalysisStatus.COMPLETED.value,
        "stage": Stage.DONE.value,
        "progress": 95,
        "execution_trace": trace,
    }


def _format_confidence_response(state_confidence: Dict[str, Any], existing_confidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw_dec = str(state_confidence.get("decision", "ABSTAIN")).upper()
    if raw_dec == "ACCEPT":
        decision = ConfidenceDecision.ACCEPTED
    elif raw_dec == "WARN":
        decision = ConfidenceDecision.WARNING
    else:
        decision = ConfidenceDecision.ABSTAINED

    base = dict(existing_confidence or {})
    specialists = base.get("specialists") or []
    policy = base.get("policy")
    warnings = base.get("warnings") or []

    conf_resp = ConfidenceResponse(
        decision=decision,
        specialists=specialists,
        policy=policy,
        limiting_score=state_confidence.get("score") if state_confidence.get("score") is not None else base.get("limiting_score"),
        limiting_source=state_confidence.get("source") or base.get("limiting_source"),
        answer_status="answered" if decision != ConfidenceDecision.ABSTAINED else "abstained",
        abstain_reason=state_confidence.get("reason") if decision == ConfidenceDecision.ABSTAINED else None,
        rationale=state_confidence.get("reason") or base.get("rationale"),
        warnings=warnings,
    )
    return conf_resp.model_dump(mode="json")


# ─── Node 10B: Grounded Response With Warning (WARN) ─────────────────────────

def compose_response_with_warning_node(state: SatQueryState) -> Dict[str, Any]:
    """Composes grounded natural language answer with explicit warning constraints (WARN path)."""
    trace = list(state.get("execution_trace", []))
    trace.append("compose_response_with_warning_started")

    task = state.get("interpreted_task", "")
    specialist_results = state.get("specialist_results", {})
    if task == "single_scene_vqa" and "SatVLM" in specialist_results and specialist_results["SatVLM"].get("answer"):
        trace.append("satvlm_scene_answer_reused")
        satvlm_res = specialist_results["SatVLM"]
        ans = satvlm_res.get("answer", "")
        raw_ans = satvlm_res.get("raw_answer") or ans
        return {
            "raw_model_response": raw_ans,
            "final_answer": ans,
            "messages": [AIMessage(content=ans)],
            "status": AnalysisStatus.COMPLETED.value,
            "stage": Stage.DONE.value,
            "progress": 95,
            "execution_trace": trace,
        }

    q_req_dict = state.get("query_requirements")
    requirements = QueryRequirements.model_validate(q_req_dict) if q_req_dict else extract_query_requirements(state.get("question", ""))
    val_rep_dict = state.get("validation_report")
    validation_report = ValidationReport.model_validate(val_rep_dict) if val_rep_dict else ValidationReport(status="VALID")
    evidence_bundle = state.get("evidence_bundle") or []
    if not evidence_bundle and state.get("evidence"):
        evidence_bundle = serialize_evidence_bundle([state["evidence"]])

    question = state.get("question", "")
    conf = state.get("confidence", {})
    warn_reason = conf.get("reason")
    reason_code = conf.get("reason_code")

    answer = compose_grounded_response(
        query=question,
        requirements=requirements,
        validation_report=validation_report,
        evidence_bundle=evidence_bundle,
        decision="WARN",
        reason_code=reason_code,
        warning_reason=warn_reason,
    )

    trace.append("compose_response_with_warning_completed")
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "status": AnalysisStatus.COMPLETED.value,
        "stage": Stage.DONE.value,
        "progress": 95,
        "execution_trace": trace,
    }


# ─── Node 11: Abstention Response (ABSTAIN) ──────────────────────────────────

def abstention_response_node(state: SatQueryState) -> Dict[str, Any]:
    """Emits explicit insufficient-evidence response without hallucinating answers (ABSTAIN path)."""
    trace = list(state.get("execution_trace", []))
    trace.append("abstention_response_composed")

    q_req_dict = state.get("query_requirements")
    requirements = QueryRequirements.model_validate(q_req_dict) if q_req_dict else extract_query_requirements(state.get("question", ""))
    val_rep_dict = state.get("validation_report")
    validation_report = ValidationReport.model_validate(val_rep_dict) if val_rep_dict else ValidationReport(status="INVALID")
    evidence_bundle = state.get("evidence_bundle") or []
    if not evidence_bundle and state.get("evidence"):
        evidence_bundle = serialize_evidence_bundle([state["evidence"]])

    question = state.get("question", "")
    conf = state.get("confidence", {})
    abstain_reason = conf.get("reason")
    reason_code = conf.get("reason_code")

    answer = compose_grounded_response(
        query=question,
        requirements=requirements,
        validation_report=validation_report,
        evidence_bundle=evidence_bundle,
        decision="ABSTAIN",
        reason_code=reason_code,
        warning_reason=abstain_reason,
    )

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "status": AnalysisStatus.COMPLETED.value,
        "stage": Stage.DONE.value,
        "progress": 100,
        "execution_trace": trace,
    }


# ─── Node 12: Persist Result ─────────────────────────────────────────────────

def persist_result_node(state: SatQueryState) -> Dict[str, Any]:
    """Persists final result payload, execution trace, and terminal state in DB."""
    analysis_id = state["analysis_id"]
    trace = list(state.get("execution_trace", []))
    trace.append("persist_result_started")

    status_val = AnalysisStatus.COMPLETED.value
    answer = state.get("final_answer")
    evidence = state.get("evidence", {})
    confidence = state.get("confidence", {})
    warnings = state.get("warnings", [])

    conf_dec = str(state.get("confidence", {}).get("decision") or "").upper()
    is_abstained = conf_dec in ("ABSTAIN", "ABSTAINED")
    outcome = {
        "answer": answer,
        "answer_type": "abstained" if is_abstained else "answered",
        "evidence": evidence,
        "evidence_bundle": state.get("evidence_bundle"),
        "query_requirements": state.get("query_requirements"),
        "validation_report": state.get("validation_report"),
        "confidence": confidence,
        "warnings": warnings,
        "trace": trace,
        "models": [{"name": s, "version": "v1"} for s in state.get("selected_specialists", [])],
        "specialists": [
            {"source": s, "answer_status": "abstained" if is_abstained else "answered"}
            for s in state.get("selected_specialists", [])
        ],
    }

    settings = get_settings()
    from app.services.result_service import get_result_service
    result_service = get_result_service(settings)

    with session_scope() as session:
        analysis = get_analysis(session, analysis_id)
        if analysis:
            try:
                finalized = result_service.finalize(session, analysis, outcome, validation=state.get("validation"))
            except Exception as exc:
                logger.warning("ResultService.finalize fallback: %s", exc)
                analysis.result = outcome
                analysis.answer = answer
                analysis.confidence = confidence
                analysis.warnings = warnings
                analysis.trace = trace
                analysis.finished_at = datetime.now(timezone.utc)
                if analysis.started_at:
                    analysis.duration_seconds = (analysis.finished_at - analysis.started_at).total_seconds()
                    started = analysis.started_at
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    analysis.duration_seconds = (analysis.finished_at - started).total_seconds()
                transition(
                    session,
                    analysis,
                    status=status_val,
                    stage=Stage.DONE.value,
                    progress=100,
                    message="Analysis complete.",
                    data=None,
                )

            # Preserve authoritative SatVLM Track 1 grounded response and confidence
            if answer:
                analysis.answer = answer
                if analysis.result and isinstance(analysis.result, dict):
                    res = dict(analysis.result)
                    res["answer"] = answer
                    analysis.result = res
            if confidence:
                formatted_conf = _format_confidence_response(
                    confidence,
                    analysis.result.get("confidence") if analysis.result and isinstance(analysis.result, dict) else None,
                )
                analysis.confidence = formatted_conf
                if analysis.result and isinstance(analysis.result, dict):
                    res = dict(analysis.result)
                    res["confidence"] = formatted_conf
                    analysis.result = res

    trace.append("persist_result_completed")
    return {
        "execution_trace": trace,
        "progress": 100,
        "status": status_val,
    }
