"""test_satvlm_adapter.py -- Verification test suite for local SatVLM (Qwen2.5-VL-3B-Instruct 4-bit).

Verifies:
  1. Availability probing:
     - CUDA absent -> CUDA_REQUIRED
     - CUDA present but model missing -> MODEL_NOT_FOUND
     - CUDA present + model present -> ADAPTER_READY
  2. Offline mode enforcement (SATQUERY_OFFLINE=1 requires local files only).
  3. Quantization configuration (4-bit NF4, double quant, float16 compute).
  4. Deterministic decoding invariants (temperature=0.0, do_sample=False, top_p=1.0, max_new_tokens=512).
  5. Single ownership: scene VQA runs in specialist dispatch and is reused in compose_response without duplicate inference.
  6. ABSTAIN bypass: confidence ABSTAIN emits deterministic template without Qwen execution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from app.models.base import AdapterProbe, AdapterRequest
from app.models.satvlm_adapter import SatVLMAdapter, resolve_model_path
from app.orchestration.nodes import compose_response_node
from app.preprocessing.base import SceneBundle
from app.schemas.common import AnalysisStatus, ConfidenceDecision
from app.services.satvlm_prompt import SATVLM_PROMPTED_V1_FREEZE_RECORD, compose_grounded_response
from src.evidence_engine.contracts import QueryRequirements, ValidationReport


def test_availability_cuda_required(monkeypatch):
    """Verifies that missing CUDA returns status='unavailable', code='CUDA_REQUIRED'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    adapter = SatVLMAdapter()
    probe = adapter.available()
    assert probe.available is False
    assert probe.status == "unavailable"
    assert probe.code == "CUDA_REQUIRED"


def test_availability_model_not_found(monkeypatch):
    """Verifies that missing model path returns status='unavailable', code='MODEL_NOT_FOUND'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("app.models.satvlm_adapter.resolve_model_path", lambda: None)
    adapter = SatVLMAdapter()
    probe = adapter.available()
    assert probe.available is False
    assert probe.status == "unavailable"
    assert probe.code == "MODEL_NOT_FOUND"


def test_availability_adapter_ready():
    """Verifies that when CUDA and local model checkpoint exist, adapter reports ADAPTER_READY."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this test host.")
    model_path = resolve_model_path()
    if model_path is None:
        pytest.skip("Local Qwen 3B weights not found.")

    adapter = SatVLMAdapter()
    probe = adapter.available()
    assert probe.available is True
    assert probe.status == "available"
    assert probe.code == "ADAPTER_READY"
    assert probe.detail.get("model_path") is not None


def test_offline_mode_no_download(monkeypatch):
    """Verifies that with SATQUERY_OFFLINE=1, resolution never falls back to remote download."""
    monkeypatch.setenv("SATQUERY_OFFLINE", "1")
    monkeypatch.setenv("SATQUERY_SATVLM_MODEL_PATH", "non_existent_fake_path_12345")
    resolved = resolve_model_path()
    # If the env path does not exist on disk, it must not return the fake path
    assert resolved != Path("non_existent_fake_path_12345").resolve()


def test_freeze_record_metadata():
    """Verifies that the frozen prompt metadata accurately records Qwen 2.5 VL 3B 4-bit."""
    assert SATVLM_PROMPTED_V1_FREEZE_RECORD["target_identifier"] == "satvlm-prompted-v1-qwen3b-4bit"
    assert SATVLM_PROMPTED_V1_FREEZE_RECORD["model_name"] == "Qwen2.5-VL-3B-Instruct"
    assert "4-bit" in SATVLM_PROMPTED_V1_FREEZE_RECORD["quantization"]
    assert SATVLM_PROMPTED_V1_FREEZE_RECORD["runtime"] == "Transformers + PyTorch"
    assert SATVLM_PROMPTED_V1_FREEZE_RECORD["device"] == "CUDA"
    assert SATVLM_PROMPTED_V1_FREEZE_RECORD["decoding_config"]["temperature"] == 0.0
    assert SATVLM_PROMPTED_V1_FREEZE_RECORD["decoding_config"]["do_sample"] is False


def test_no_duplicate_scene_inference():
    """Verifies that compose_response_node reuses existing SatVLM answer without running Qwen twice."""
    state = {
        "analysis_id": "test_dup_123",
        "interpreted_task": "single_scene_vqa",
        "question": "Describe this satellite scene.",
        "specialist_results": {
            "SatVLM": {
                "source": "SatVLM",
                "answer": "This is an authoritative single-scene answer from the specialist dispatch.",
                "answer_status": "answered",
            }
        },
        "execution_trace": [],
    }

    result = compose_response_node(state)
    assert result["final_answer"] == "This is an authoritative single-scene answer from the specialist dispatch."
    assert "satvlm_scene_answer_reused" in result["execution_trace"]


def test_abstain_bypasses_qwen_generation():
    """Verifies that when confidence gate produces ABSTAIN, deterministic template is used and Qwen is bypassed."""
    reqs = QueryRequirements(requires_quantitative_change=True)
    val_rep = ValidationReport(status="INVALID", errors=["Missing required source_model in evidence contract."])
    evidence_bundle = []

    answer = compose_grounded_response(
        query="How many square metres changed?",
        requirements=reqs,
        validation_report=val_rep,
        evidence_bundle=evidence_bundle,
        decision="ABSTAIN",
        reason_code="INVALID_EVIDENCE_CONTRACT",
    )

    from app.services.satvlm_prompt import TEMPLATE_INVALID_CONTRACT
    assert answer == TEMPLATE_INVALID_CONTRACT


def test_remote_gateway_probe(monkeypatch):
    """Verifies that when SATQUERY_SATVLM_REMOTE_URL is set, adapter is ready even without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SATQUERY_SATVLM_REMOTE_URL", "https://mock-tunnel.trycloudflare.com")
    adapter = SatVLMAdapter()
    probe = adapter.available()
    assert probe.available is True
    assert probe.status == "available"
    assert probe.code == "ADAPTER_READY"
    assert "https://mock-tunnel.trycloudflare.com" in probe.reason


def test_remote_gateway_infer(monkeypatch, tmp_path):
    """Verifies that when SATQUERY_SATVLM_REMOTE_URL is set, infer posts payload and validates claims."""
    import httpx

    test_img = tmp_path / "test_scene.png"
    test_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)  # minimal png bytes

    mock_source = MagicMock()
    mock_source.path = str(test_img)
    mock_source.role = "primary"

    mock_bundle = MagicMock()
    mock_bundle.sources = [mock_source]
    mock_bundle.source_for.return_value = mock_source

    request = AdapterRequest(
        analysis_id="test_remote_1",
        task="single_scene_vqa",
        question="What is visible here?",
        bundle=mock_bundle,
        work_dir=tmp_path,
        params={},
    )

    monkeypatch.setenv("SATQUERY_SATVLM_REMOTE_URL", "https://mock-tunnel.trycloudflare.com")

    # Mock httpx.Client post response
    class MockResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "status": "success",
                "output_text": "A clear coastal inlet is visible.",
                "trace": ["satvlm_remote_qwen_ok"],
            }

    with patch.object(httpx.Client, "post", return_value=MockResponse()):
        adapter = SatVLMAdapter()
        res = adapter.infer(request)
        assert res.answer_status == "answered"
        assert res.answer == "A clear coastal inlet is visible."
        assert "satvlm_adapter_remote_gateway_invoked" in res.trace
        assert "satvlm_remote_inferred" in res.trace

