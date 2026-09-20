"""Integration tests for the FastAPI routes with hybrid fusion orchestration, fallback handling, and correlation tracing."""

import time
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from tests.test_data import VALID_FUSION_REQUEST_DICT, VALID_LLM_FUSION_RESULT

client = TestClient(app)


def test_root_endpoint():
    """Verify that root versioned healthcheck endpoint returns success message."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    assert response.json() == {"message": "Agent 4 Clinical Data Fusion API is running"}
    assert "X-Correlation-ID" in response.headers


@patch("app.fusion.enhance_clinical_context_with_llama")
def test_fuse_success_llama_enhanced(mock_llm):
    """Verify that /api/v1/fuse endpoint returns 200 OK with fusion_mode='llama_enhanced' when LLM succeeds."""
    mock_stats = {"latency_ms": 32.1, "token_usage": {"total_tokens": 150}}
    mock_llm.return_value = (VALID_LLM_FUSION_RESULT, mock_stats)

    custom_correlation_id = "test-correlation-1234"
    start_time = time.perf_counter()
    response = client.post(
        "/api/v1/fuse",
        json=VALID_FUSION_REQUEST_DICT,
        headers={"X-Correlation-ID": custom_correlation_id}
    )
    api_duration_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == custom_correlation_id
    assert api_duration_ms < 5000.0  # Measure overall API response time
    
    data = response.json()
    assert data["status"] == "success"
    assert "unified_context" in data
    assert "metadata" in data
    
    context = data["unified_context"]
    assert "merged_findings" in context
    assert "confidence_score" in context
    
    metadata = data["metadata"]
    assert metadata["fusion_method"] == "llama_enhanced"
    assert metadata["correlation_id"] == custom_correlation_id
    assert metadata["execution_stats"] == mock_stats


@patch("app.fusion.enhance_clinical_context_with_llama")
def test_fuse_automatic_fallback_on_llm_failure(mock_llm):
    """Verify that /api/v1/fuse returns 200 OK with fusion_method='rule_based_fallback' when LLM fails."""
    mock_llm.return_value = None

    custom_correlation_id = "llm-fallback-id"
    start_time = time.perf_counter()
    response = client.post(
        "/api/v1/fuse",
        json=VALID_FUSION_REQUEST_DICT,
        headers={"X-Correlation-ID": custom_correlation_id}
    )
    api_duration_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == custom_correlation_id
    assert api_duration_ms < 1000.0  # Baseline fallback should be fast
    
    data = response.json()
    assert data["status"] == "success"
    assert "unified_context" in data
    assert "metadata" in data
    
    context = data["unified_context"]
    assert "Acute Myocardial Infarction" in context["normalized_medical_terms"]
    
    metadata = data["metadata"]
    assert metadata["fusion_method"] == "rule_based_fallback"
    assert metadata["correlation_id"] == custom_correlation_id


def test_fuse_validation_failure_empty_request():
    """Verify that /api/v1/fuse endpoint returns 422 Unprocessable Entity for an empty request payload."""
    response = client.post("/api/v1/fuse", json={})
    assert response.status_code == 422


def test_fuse_validation_failure_empty_evidence():
    """Verify that /api/v1/fuse endpoint returns 400 Bad Request if both RAG and KG evidence are empty."""
    invalid_payload = {
        "agent2_output": {
            "rag_evidence": [],
            "kg_evidence": []
        },
        "agent3_output": VALID_FUSION_REQUEST_DICT["agent3_output"]
    }
    response = client.post(
        "/api/v1/fuse",
        json=invalid_payload,
        headers={"X-Correlation-ID": "err-corr-id"}
    )
    assert response.status_code == 400
    assert response.headers.get("X-Correlation-ID") == "err-corr-id"
    
    data = response.json()
    assert "detail" in data
    assert "Input validation failed" in data["detail"]
    assert data["correlation_id"] == "err-corr-id"
