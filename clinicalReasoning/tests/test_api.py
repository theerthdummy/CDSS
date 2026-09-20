"""Integration and API boundary tests for Agent 5 REST endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture
def valid_agent4_payload():
    """Returns a realistic valid Agent 4 UnifiedClinicalContext dictionary payload (high confidence)."""
    return {
        "merged_findings": [
            {
                "finding": "ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy",
                "sources": ["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            }
        ],
        "normalized_medical_terms": ["STEMI", "Reperfusion Therapy", "Aspirin"],
        "supporting_evidence": [
            {
                "finding": "STEMI requires immediate emergency reperfusion therapy via primary PCI or fibrinolysis",
                "source_attribution": "Agent 2 Biomedical RAG",
            }
        ],
        "conflicting_evidence": [],
        "evidence_priority": [
            {
                "finding": "ST-elevation myocardial infarction",
                "priority_score": 0.95,
                "primary_source": "Agent 3 Clinical Guidelines",
            }
        ],
        "source_traceability": [
            {
                "finding": "ST-elevation myocardial infarction",
                "sources": ["PubMed", "Clinical Guidelines"],
                "upstream_agents": ["Agent 2", "Agent 3"],
            }
        ],
        "fusion_summary": "Patient presentation and evidence indicate acute STEMI requiring emergency reperfusion.",
        "confidence_score": 0.92,
    }


def test_get_health_endpoint():
    """Verify GET /health returns HTTP 200 and expected status object."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "agent": "agent_5",
    }


def test_post_reason_valid_high_confidence_payload(valid_agent4_payload):
    """Verify POST /api/v1/reason with high confidence returns HTTP 200 with ClinicalDecisionSupportResponse."""
    response = client.post("/api/v1/reason", json=valid_agent4_payload)
    assert response.status_code == 200
    data = response.json()
    assert "clinical_assessment" in data
    assert "reasoning" in data
    assert "decision_support" in data
    assert "safety" in data
    assert "uncertainty" in data
    assert "traceability" in data
    assert "agent_metadata" in data
    assert data["agent_metadata"]["agent"] == "agent_5"


def test_post_reason_low_confidence_returns_feedback_request(valid_agent4_payload):
    """Verify POST /api/v1/reason with low confidence returns HTTP 200 with FeedbackRequest."""
    low_conf_payload = dict(valid_agent4_payload, confidence_score=0.45)
    response = client.post("/api/v1/reason?retry_count=0", json=low_conf_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "needs_more_evidence"
    assert data["retry_count"] == 0
    assert data["max_retries"] == 3
    assert data["confidence_score"] == 0.45
    assert len(data["evidence_gaps"]) > 0
    assert "agent_2_request" in data
    assert "agent_3_request" in data


def test_post_reason_max_retries_returns_insufficient_evidence_termination(valid_agent4_payload):
    """Verify POST /api/v1/reason with low confidence and retry_count=3 returns InsufficientEvidenceTermination."""
    low_conf_payload = dict(valid_agent4_payload, confidence_score=0.45)
    response = client.post("/api/v1/reason?retry_count=3", json=low_conf_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "insufficient_evidence"
    assert data["retry_count"] == 3
    assert data["max_retries"] == 3
    assert "termination_message" in data


def test_post_reason_missing_required_fields():
    """Verify POST /api/v1/reason returns HTTP 422 when required fields are missing."""
    incomplete_payload = {
        "merged_findings": [
            {"finding": "Chest pain", "sources": ["Agent 2"]}
        ],
        "normalized_medical_terms": ["Chest Pain"],
    }
    response = client.post("/api/v1/reason", json=incomplete_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_post_reason_invalid_field_type(valid_agent4_payload):
    """Verify POST /api/v1/reason returns HTTP 422 when field types are invalid."""
    invalid_payload = dict(valid_agent4_payload, confidence_score="invalid_string_score")
    response = client.post("/api/v1/reason", json=invalid_payload)
    assert response.status_code == 422


def test_post_reason_confidence_score_out_of_range(valid_agent4_payload):
    """Verify POST /api/v1/reason returns HTTP 422 when confidence_score is out of bounds."""
    invalid_payload_high = dict(valid_agent4_payload, confidence_score=1.5)
    response_high = client.post("/api/v1/reason", json=invalid_payload_high)
    assert response_high.status_code == 422

    invalid_payload_low = dict(valid_agent4_payload, confidence_score=-0.1)
    response_low = client.post("/api/v1/reason", json=invalid_payload_low)
    assert response_low.status_code == 422


def test_post_reason_malformed_json_structure():
    """Verify POST /api/v1/reason returns HTTP 422 when payload is not a JSON object."""
    response = client.post("/api/v1/reason", json=["not", "a", "dictionary"])
    assert response.status_code == 422


def test_post_reason_request_id_header(valid_agent4_payload):
    """Verify POST /api/v1/reason handles X-Request-ID header cleanly without error."""
    custom_req_id = "TRACE-REQ-12345-ABC"
    response = client.post(
        "/api/v1/reason",
        json=valid_agent4_payload,
        headers={"X-Request-ID": custom_req_id},
    )
    assert response.status_code == 200


def test_post_reason_internal_server_error_handling(valid_agent4_payload, monkeypatch):
    """Verify POST /api/v1/reason returns HTTP 500 without leaking stack traces when internal service fails."""
    from app.services.clinical_reasoning import ClinicalReasoningError

    def mock_failing_pipeline(*args, **kwargs):
        raise ClinicalReasoningError("Simulated internal pipeline error.")

    monkeypatch.setattr(
        "app.routers.reasoning.reasoning_service.execute_reasoning_pipeline",
        mock_failing_pipeline,
    )

    response = client.post("/api/v1/reason", json=valid_agent4_payload)
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Simulated internal pipeline error." in data["detail"]

