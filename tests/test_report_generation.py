"""test_report_generation.py -- Verification of HTML, PDF, and JSON report generation."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest
from starlette.testclient import TestClient

from app.core.config import get_settings
from app.db.models import Analysis
from app.db.repo import create_analysis
from app.db.session import init_db, session_scope
from app.main import create_app
from app.schemas.common import AnalysisStatus
from app.services.report_service import ReportService, get_result_service


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    settings = get_settings()
    init_db(settings.database_url)


def test_report_service_render_formats():
    """Verify that ReportService correctly renders html, pdf, and json payloads."""
    service = ReportService()
    payload = {
        "analysis_id": "test-analysis-12345",
        "task": "bitemporal_change_detection",
        "question": "Where has the built-up area changed?",
        "answer": "Significant change of 12.45 hectares detected in Region 1.",
        "warnings": [],
        "artifacts": [],
        "pipeline": {"mode": "deterministic", "authoritative": True},
        "evidence": {
            "georeferenced": True,
            "measurement_crs": "EPSG:32631",
            "area_value": 12.45,
            "area_unit": "hectares",
            "changed_percentage": 3.82,
            "regions": [
                {"id": 1, "area_value": 12.45, "area_unit": "hectares", "location": "North quadrant"}
            ],
        },
        "confidence": {
            "decision": "ACCEPT",
            "specialists": [
                {"source": "changenet_v3_1", "kind": "heuristic", "calibrated": False, "value": 0.91}
            ],
        },
        "execution_trace": ["interpret_query", "dispatch_specialist", "run_evidence_engine", "compose_response"],
        "models": [{"name": "changenet", "version": "v3.1"}],
    }

    # HTML
    html_rep = service.render(payload, format="html", name="report.html")
    assert html_rep.media_type.startswith("text/html")
    html_text = html_rep.content.decode("utf-8")
    assert "test-analysis-12345" in html_text
    assert "Where has the built-up area changed?" in html_text
    assert "12.45 hectares" in html_text
    assert "ACCEPT" in html_text

    # JSON
    json_rep = service.render(payload, format="json", name="report.json")
    assert json_rep.media_type == "application/json"
    parsed = json.loads(json_rep.content)
    assert parsed["analysis_id"] == "test-analysis-12345"
    assert parsed["confidence"]["decision"] == "ACCEPT"

    # PDF
    pdf_rep = service.render(payload, format="pdf", name="report.pdf")
    assert pdf_rep.media_type == "application/pdf"
    assert pdf_rep.content.startswith(b"%PDF-")
    assert len(pdf_rep.content) > 500


def test_report_api_endpoints():
    """Verify that the FastAPI report endpoint returns valid HTML, PDF, and JSON reports."""
    settings = get_settings()
    app = create_app(settings)

    import uuid
    analysis_id = f"test-rep-{uuid.uuid4().hex[:8]}"

    # Create a completed analysis in the database
    with session_scope() as session:
        analysis_obj = Analysis(
            id=analysis_id,
            question="Where has the built-up area changed?",
            task="bitemporal_change",
            status=AnalysisStatus.COMPLETED,
            answer="Detected 4.2 ha of change.",
            confidence={"decision": "accepted", "specialists": []},
            result={
                "analysis_id": analysis_id,
                "task": "bitemporal_change",
                "question": "Where has the built-up area changed?",
                "answer": "Detected 4.2 ha of change.",
                "answer_type": "change_detection",
                "confidence": {"decision": "accepted", "specialists": []},
                "evidence": {
                    "georeferenced": False,
                    "changed_percentage": 5.1,
                    "area_value": 42000,
                    "area_unit": "pixels",
                    "regions": [],
                },
                "warnings": [],
                "artifacts": [],
                "pipeline": {"mode": "python", "authoritative": True},
                "execution_trace": ["dispatch_specialist"],
                "models": [],
            },
            warnings=[],
            trace=["dispatch_specialist"],
            models=[],
        )
        analysis = create_analysis(
            session,
            analysis=analysis_obj,
            upload_ids=[],
        )
        analysis_id = analysis.id

    with TestClient(app) as client:
        # Test HTML report
        res_html = client.get(f"/api/v1/analyses/{analysis_id}/report?format=html")
        assert res_html.status_code == 200
        assert "text/html" in res_html.headers["content-type"]
        assert f"SatQuery analysis {analysis_id}" in res_html.text

        # Test JSON report
        res_json = client.get(f"/api/v1/analyses/{analysis_id}/report?format=json")
        assert res_json.status_code == 200
        assert res_json.headers["content-type"].startswith("application/json")
        assert res_json.json()["analysis_id"] == analysis_id

        # Test PDF report
        res_pdf = client.get(f"/api/v1/analyses/{analysis_id}/report?format=pdf")
        assert res_pdf.status_code == 200
        assert res_pdf.headers["content-type"] == "application/pdf"
        assert res_pdf.content.startswith(b"%PDF-")
