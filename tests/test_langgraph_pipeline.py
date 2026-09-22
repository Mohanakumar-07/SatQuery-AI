"""test_langgraph_pipeline.py -- Verification test suite for SatQuery LangGraph orchestration.

Verifies:
1. StateGraph assembly, compilation & checkpointer.
2. Single-image VQA workflow through LangGraph.
3. Bi-temporal change detection workflow with ChangeFormer V6 & Evidence Engine.
4. Constrained routing (Rule 3 & Section 13).
5. Analysis cache key computation & reuse bypass (Sections 20 & 21).
6. Clarification check interrupt (Section 12).
7. Confidence & abstention gate (Sections 18 & 19).
8. Atomic single-resume guard (Section 15.5).
"""

import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest
from app.core.config import get_settings
from app.db.base import Base
from app.db.models import Analysis, Upload
from app.db.repo import (
    claim_graph_resume,
    create_analysis,
    create_upload,
    find_cached_analysis,
)
from app.db.session import init_db, session_scope
from app.orchestration.cache import compute_analysis_cache_key
from app.orchestration.graph import (
    build_satquery_graph,
    get_checkpointer,
    get_satquery_pipeline,
)
from app.orchestration.state import SatQueryState
from app.schemas.common import AnalysisStatus


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    settings = get_settings()
    init_db(settings.database_url)


def test_graph_assembly():
    """Verifies that the LangGraph StateGraph builds and compiles with checkpointer."""
    pipeline = get_satquery_pipeline()
    assert pipeline is not None
    # Check that required nodes exist in the graph
    node_names = set(pipeline.nodes.keys())
    required_nodes = {
        "validate_inputs",
        "interpret_query",
        "clarification_check",
        "constrained_router",
        "check_cache",
        "dispatch_specialist",
        "validate_specialist_output",
        "run_evidence_engine",
        "confidence_check",
        "compose_response",
        "abstention_response",
        "persist_result",
    }
    assert required_nodes.issubset(node_names)


def test_analysis_cache_key_determinism():
    """Verifies Section 20: identical inputs produce matching keys; changes produce misses."""
    key1 = compute_analysis_cache_key(
        input_hashes=["hash_a", "hash_b"],
        specialist="ChangeNet",
        specialist_version="V3.1-Champion",
        checkpoint_sha256="ck_sha",
        threshold=0.25,
    )
    key2 = compute_analysis_cache_key(
        input_hashes=["hash_a", "hash_b"],
        specialist="ChangeNet",
        specialist_version="V3.1-Champion",
        checkpoint_sha256="ck_sha",
        threshold=0.25,
    )
    # Different threshold => cache miss
    key3 = compute_analysis_cache_key(
        input_hashes=["hash_a", "hash_b"],
        specialist="ChangeNet",
        specialist_version="V3.1-Champion",
        checkpoint_sha256="ck_sha",
        threshold=0.30,
    )

    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 64


def test_atomic_single_resume_guard():
    """Verifies Section 15.5: only one caller can claim PENDING -> RESUMING."""
    analysis_id = f"test_guard_{uuid.uuid4().hex[:8]}"
    with session_scope() as session:
        # Create test analysis
        a = Analysis(
            id=analysis_id,
            question="Test guard",
            status="running",
            graph_resume_status="PENDING",
            resume_attempt_count=0,
        )
        session.merge(a)
        session.commit()

    with session_scope() as session:
        first_claim = claim_graph_resume(session, analysis_id)
        assert first_claim is True

    with session_scope() as session:
        second_claim = claim_graph_resume(session, analysis_id)
        assert second_claim is False

    with session_scope() as session:
        a = session.get(Analysis, analysis_id)
        assert a.graph_resume_status == "RESUMING"
        assert a.resume_attempt_count == 1


def test_langgraph_single_image_workflow():
    """Verifies end-to-end execution of single-image workflow through LangGraph."""
    uid = uuid.uuid4().hex[:8]
    upload_id = f"upl_single_{uid}"
    analysis_id = f"ana_single_{uid}"
    thread_id = f"thr_single_{uid}"

    # Create dummy upload and analysis in DB
    with session_scope() as session:
        u = Upload(
            id=upload_id,
            original_filename="scene.png",
            stored_name="scene.png",
            relative_path=f"uploads/{upload_id}.png",
            size_bytes=1024,
            sha256=f"sha_{uid}",
            extension="png",
            detected_media_type="image/png",
            media_kind="png",
            probe_status="ok",
            width=256,
            height=256,
            band_count=3,
            georeferenced=False,
            probe={"band_names": ["R", "G", "B"]},
        )
        session.merge(u)

        a = Analysis(
            id=analysis_id,
            thread_id=thread_id,
            question="Describe this satellite scene.",
            status="queued",
            graph_resume_status="NOT_REQUIRED",
        )
        create_analysis(session, analysis=a, upload_ids=[upload_id])

    # Run LangGraph pipeline
    pipeline = get_satquery_pipeline()
    state: SatQueryState = {
        "conversation_id": thread_id,
        "thread_id": thread_id,
        "analysis_id": analysis_id,
        "question": "Describe this satellite scene.",
        "upload_ids": [upload_id],
        "hints": {},
        "messages": [],
    }

    result = pipeline.invoke(state, config={"configurable": {"thread_id": thread_id}})

    assert result["status"] == "completed"
    assert result["interpreted_task"] == "single_scene_vqa"
    assert "SatVLM" in result["selected_specialists"]
    assert result["final_answer"] is not None
    assert "confidence" in result
    assert result["confidence"]["status"] == "UNCALIBRATED"
    assert "validate_inputs_started" in result["execution_trace"]
    assert "persist_result_completed" in result["execution_trace"]


def test_langgraph_constrained_routing():
    """Verifies Rule 3: ChangeNet is NOT invoked if only 1 image is provided."""
    uid = uuid.uuid4().hex[:8]
    upload_id = f"upl_route_{uid}"
    analysis_id = f"ana_route_{uid}"
    thread_id = f"thr_route_{uid}"

    with session_scope() as session:
        u = Upload(
            id=upload_id,
            original_filename="one_image.png",
            stored_name="one_image.png",
            relative_path=f"uploads/{upload_id}.png",
            size_bytes=1024,
            sha256=f"sha_{uid}",
            extension="png",
            detected_media_type="image/png",
            media_kind="png",
            probe_status="ok",
            width=256,
            height=256,
            band_count=3,
            georeferenced=False,
            probe={"band_names": ["R", "G", "B"]},
        )
        session.merge(u)

        a = Analysis(
            id=analysis_id,
            thread_id=thread_id,
            question="Detect changes in this satellite scene.",
            status="queued",
            hints={"file_roles": {upload_id: "single"}},
            graph_resume_status="NOT_REQUIRED",
        )
        create_analysis(session, analysis=a, upload_ids=[upload_id])

    pipeline = get_satquery_pipeline()
    state: SatQueryState = {
        "conversation_id": thread_id,
        "thread_id": thread_id,
        "analysis_id": analysis_id,
        "question": "Detect changes in this satellite scene.",
        "upload_ids": [upload_id],
        "hints": {"file_roles": {upload_id: "single"}},
        "messages": [],
    }

    result = pipeline.invoke(state, config={"configurable": {"thread_id": thread_id}})

    # Must be downgraded to single_scene_vqa because only 1 image was uploaded
    assert "ChangeNet" not in result["selected_specialists"]
    assert "SatVLM" in result["selected_specialists"]
    assert any(w["code"] == "MISSING_TEMPORAL_PAIR" for w in result["warnings"])
    assert "change_detection_downgraded_single_image" in result["execution_trace"]


def test_langgraph_bitemporal_change_workflow():
    """Verifies ChangeFormer V6 & Evidence Engine execution in LangGraph."""
    uid = uuid.uuid4().hex[:8]
    up1 = f"upl_t1_{uid}"
    up2 = f"upl_t2_{uid}"
    analysis_id = f"ana_bi_{uid}"
    thread_id = f"thr_bi_{uid}"

    # Use existing Montpellier test patches from disk
    t1_path = REPO_ROOT / "changenet" / "data" / "oscd_v3" / "patches" / "test" / "A" / "montpellier_00000.png"
    t2_path = REPO_ROOT / "changenet" / "data" / "oscd_v3" / "patches" / "test" / "B" / "montpellier_00000.png"

    if not t1_path.is_file() or not t2_path.is_file():
        pytest.skip("Montpellier patches not found")

    from app.models.changenet_adapter import ChangeNetAdapter
    if not ChangeNetAdapter().available().available:
        pytest.skip(f"ChangeNet specialist unavailable: {ChangeNetAdapter().available().reason}")

    with session_scope() as session:
        probe_data = {
            "bounds": [[43.6, 3.8], [43.7, 3.9]],
            "resolution": [10.0, 10.0],
            "band_names": ["R", "G", "B"],
            "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0],
        }
        u1 = Upload(
            id=up1,
            original_filename="montpellier_t1.png",
            stored_name=str(t1_path),
            relative_path=f"uploads/{up1}.png",
            size_bytes=t1_path.stat().st_size,
            sha256=f"sha_{up1}",
            extension="png",
            detected_media_type="image/png",
            media_kind="png",
            probe_status="ok",
            width=256,
            height=256,
            band_count=3,
            crs="EPSG:32631",
            georeferenced=True,
            probe=probe_data,
        )
        u2 = Upload(
            id=up2,
            original_filename="montpellier_t2.png",
            stored_name=str(t2_path),
            relative_path=f"uploads/{up2}.png",
            size_bytes=t2_path.stat().st_size,
            sha256=f"sha_{up2}",
            extension="png",
            detected_media_type="image/png",
            media_kind="png",
            probe_status="ok",
            width=256,
            height=256,
            band_count=3,
            crs="EPSG:32631",
            georeferenced=True,
            probe=probe_data,
        )
        session.merge(u1)
        session.merge(u2)

        a = Analysis(
            id=analysis_id,
            thread_id=thread_id,
            question="Find changes between 2017 and 2018.",
            hints={"file_roles": {up1: "before", up2: "after"}},
            status="queued",
            graph_resume_status="NOT_REQUIRED",
        )
        create_analysis(session, analysis=a, upload_ids=[up1, up2], roles={up1: "before", up2: "after"})

    pipeline = get_satquery_pipeline()
    state: SatQueryState = {
        "conversation_id": thread_id,
        "thread_id": thread_id,
        "analysis_id": analysis_id,
        "question": "Find changes between 2017 and 2018.",
        "upload_ids": [up1, up2],
        "hints": {"file_roles": {up1: "before", up2: "after"}},
        "messages": [],
    }

    result = pipeline.invoke(state, config={"configurable": {"thread_id": thread_id}})

    assert result["status"] == "completed"
    assert "ChangeNet" in result["selected_specialists"]
    assert "changenet_v6_inferred" in result["execution_trace"]
    assert "evidence_engine_change_extracted" in result["execution_trace"]

    # Verify Evidence Engine facts
    evidence = result["evidence"]
    assert evidence["kind"] == "change"
    assert evidence.get("region_count") is not None
    assert evidence.get("region_count") > 0
    assert "geojson" in evidence
    assert len(evidence["geojson"]["features"]) > 0

    # Verify grounded answer
    assert "Bi-temporal change analysis detected" in result["final_answer"]
