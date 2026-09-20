"""Dedicated unit tests for automatic rule-based fallback behavior under all LLM failure modes."""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.config import settings
from app.fusion import fuse_clinical_data, preprocess_fusion_request
from app.schemas import FusionRequest
from tests.test_data import VALID_FUSION_REQUEST_DICT


def _run_fusion_with_mock_llm(mock_post):
    request = FusionRequest.model_validate(VALID_FUSION_REQUEST_DICT)
    preprocessed = preprocess_fusion_request(request)
    context, conflicts, trace_origins, mode, stats, fallback_reason = fuse_clinical_data(preprocessed, correlation_id="test-fallback-suite")
    return context, conflicts, trace_origins, mode, stats, fallback_reason


def test_fallback_on_api_timeout():
    """Verify that network timeout during LLM call falls back automatically to rule-based fusion."""
    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
        
        context, conflicts, origins, mode, stats, reason = _run_fusion_with_mock_llm(None)
        assert mode == "rule_based_fallback"
        assert reason is not None
        assert "Acute Myocardial Infarction" in context.normalized_medical_terms
        assert len(conflicts) > 0


def test_fallback_on_network_error():
    """Verify that network failure during LLM call falls back automatically to rule-based fusion."""
    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", side_effect=httpx.RequestError("DNS Resolution Error")):
        
        context, conflicts, origins, mode, stats, reason = _run_fusion_with_mock_llm(None)
        assert mode == "rule_based_fallback"
        assert reason is not None
        assert "Acute Myocardial Infarction" in context.normalized_medical_terms


def test_fallback_on_invalid_api_key():
    """Verify that unconfigured/invalid API key falls back automatically to rule-based fusion."""
    with patch.object(settings, "llm_api_key", "mock_key_or_set_real_api_key"):
        context, conflicts, origins, mode, stats, reason = _run_fusion_with_mock_llm(None)
        assert mode == "rule_based_fallback"
        assert reason is not None
        assert "Acute Myocardial Infarction" in context.normalized_medical_terms


def test_fallback_on_rate_limit_429():
    """Verify that 429 rate limit response from LLM provider falls back automatically to rule-based fusion."""
    mock_response = MagicMock()
    mock_response.status_code = 429

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        context, conflicts, origins, mode, stats, reason = _run_fusion_with_mock_llm(None)
        assert mode == "rule_based_fallback"
        assert reason is not None
        assert "Acute Myocardial Infarction" in context.normalized_medical_terms


def test_fallback_on_malformed_json():
    """Verify that malformed/invalid JSON returned by LLM falls back automatically to rule-based fusion."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is raw text, not JSON"}}]
    }

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        context, conflicts, origins, mode, stats, reason = _run_fusion_with_mock_llm(None)
        assert mode == "rule_based_fallback"
        assert reason is not None
        assert "Acute Myocardial Infarction" in context.normalized_medical_terms


def test_fallback_on_empty_response():
    """Verify that empty response content from LLM falls back automatically to rule-based fusion."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        context, conflicts, origins, mode, stats, reason = _run_fusion_with_mock_llm(None)
        assert mode == "rule_based_fallback"
        assert reason is not None
        assert "Acute Myocardial Infarction" in context.normalized_medical_terms


def test_fallback_on_schema_validation_error():
    """Verify that LLM output failing Pydantic schema validation falls back automatically to rule-based fusion."""
    invalid_schema_json = {
        "unified_context": {
            "invalid_field": ["Acute Myocardial Infarction"]
            # Missing required fields like merged_findings, normalized_medical_terms, confidence_score, etc.
        }
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": str(invalid_schema_json)}}]
    }

    with patch.object(settings, "llm_api_key", "valid_key_123"), \
         patch("httpx.Client.post", return_value=mock_response):
        
        context, conflicts, origins, mode, stats, reason = _run_fusion_with_mock_llm(None)
        assert mode == "rule_based_fallback"
        assert reason is not None
        assert "Acute Myocardial Infarction" in context.normalized_medical_terms
