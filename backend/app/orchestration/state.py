"""SatQuery AI -- LangGraph State Schema.

Defines the state structure for the SatQuery state machine (plan sections 3, 11, 26).
Adheres strictly to Rule 14 and Rule 29:
- Only lightweight metadata, structured facts, and file references.
- Never stores raw image bytes, GeoTIFF tensors, or large rasters in state.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class SatQueryState(TypedDict, total=False):
    # ---- Conversation & Thread Memory (sections 10, 11) ----
    conversation_id: str
    user_id: Optional[str]
    thread_id: str
    messages: Annotated[list, add_messages]

    # ---- Analysis Context & Request Inputs ----
    analysis_id: str
    question: str
    upload_ids: list[str]
    hints: Optional[dict[str, Any]]
    input_bundle: Optional[dict[str, Any]]

    # ---- Validation & Query Interpretation (sections 8, 13) ----
    validation: Optional[dict[str, Any]]
    needs_clarification: bool
    clarification_payload: Optional[dict[str, Any]]

    # ---- Constrained Routing (sections 8, 13) ----
    interpreted_task: Optional[str]  # single_scene_vqa | bi_temporal_change | optical_sar_land_cover
    selected_workflow: Optional[str]  # single_image | bi_temporal | optical_sar
    selected_specialists: list[str]  # ["SatVLM"] | ["ChangeNet"] | ["SAR-FuseSeg"]

    # ---- Analysis Reuse & Cache Identity (sections 20, 21) ----
    cache_key: Optional[str]
    cache_hit: bool

    # ---- Asynchronous Specialist Job Tracking (sections 15, 16) ----
    job_id: Optional[str]
    job_status: Optional[str]  # QUEUED | RUNNING | COMPLETED | FAILED
    graph_resume_status: Optional[str]  # NOT_REQUIRED | PENDING | RESUMING | RESUMED | RESUME_FAILED

    # ---- Specialist Outputs & Evidence Engine (sections 7, 17) ----
    specialist_results: dict[str, Any]
    evidence: dict[str, Any]
    evidence_bundle: Optional[list[dict[str, Any]]]

    # ---- Contract Validation & Confidence Gate (Track 1) ----
    query_requirements: Optional[dict[str, Any]]
    validation_report: Optional[dict[str, Any]]
    confidence: dict[str, Any]  # {"status": "UNCALIBRATED", "score": None, "level": "UNKNOWN", "decision": "ACCEPT"|"WARN"|"ABSTAIN"}
    warnings: list[dict[str, Any]]

    # ---- Final Composition & Trace (sections 17, 27, 28) ----
    final_answer: Optional[str]
    execution_trace: list[str]
    status: str  # queued | running | needs_clarification | completed | failed | abstained
    stage: str
    progress: int
    error: Optional[dict[str, Any]]
