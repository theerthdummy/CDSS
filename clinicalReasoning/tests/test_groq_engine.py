"""Unit tests for GroqReasoningEngine using mocked HTTP client.

Tests cover: successful structured reasoning, JSON parsing, authentication failures,
rate limiting, connection failures, timeout, invalid JSON, empty response, and
schema validation errors — all without real Groq API calls.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from app.models.input_models import (
    ConflictEvidence,
    EvidencePriority,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import ClinicalDecisionSupportResponse
from app.services.reasoning_engines import (
    EngineExecutionError,
    EngineUnavailableError,
    GroqReasoningEngine,
)

# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

VALID_GROQ_RESPONSE_JSON = json.dumps({
    "clinical_assessment": {
        "primary_interpretation": "Groq Llama Fallback: Acute STEMI assessment requiring emergency reperfusion.",
        "clinical_significance": "High",
        "differential_considerations": ["NSTEMI", "Unstable Angina"],
    },
    "reasoning": {
        "key_findings": ["ST elevation in leads II, III, aVF"],
        "supporting_factors": ["Troponin elevation consistent with acute MI"],
        "conflicting_factors": [],
        "reasoning_summary": "Groq Llama reasoning from fused clinical context.",
    },
    "decision_support": {
        "recommended_actions": ["Activate STEMI protocol", "Primary PCI within 90 minutes"],
        "additional_information_needed": ["Patient weight for anticoagulation dosing"],
        "priority_level": "Urgent",
    },
    "safety": {
        "safety_flags": [],
        "contraindications_or_concerns": [],
    },
    "uncertainty": {
        "confidence_score": 0.85,
        "uncertainty_factors": ["Evidence sourced from fused RAG context"],
    },
    "traceability": [],
    "agent_metadata": {
        "agent": "agent_5",
        "model": "llama-3.1-8b-instruct",
        "reasoning_mode": "fallback_model",
        "status": "success",
    },
})


@pytest.fixture
def valid_context() -> UnifiedClinicalContext:
    """Returns a valid sample UnifiedClinicalContext instance."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute STEMI requires emergency primary PCI within 90 minutes",
                sources=["Agent 2", "Agent 3"],
            )
        ],
        normalized_medical_terms=["STEMI", "Primary PCI"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Primary PCI improves acute survival in STEMI",
                source_attribution="Agent 3 Guidelines",
            )
        ],
        conflicting_evidence=[
            ConflictEvidence(
                type="Treatment Conflict",
                finding="Fibrinolysis",
                description="Bleeding risk caution",
            )
        ],
        evidence_priority=[
            EvidencePriority(
                finding="STEMI protocol",
                priority_score=0.95,
                primary_source="Agent 3 Clinical Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="STEMI protocol",
                sources=["PubMed", "Clinical Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Acute STEMI requiring emergency reperfusion.",
        confidence_score=0.92,
    )


def _make_mock_http_response(
    status_code: int = 200,
    content: str = VALID_GROQ_RESPONSE_JSON,
) -> MagicMock:
    """Build a mock httpx.Response returning the given status code and content."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = content
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock_resp


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_groq_engine_successful_reasoning(valid_context):
    """Verify that a successful mocked Groq response returns validated ClinicalDecisionSupportResponse."""
    mock_client = MagicMock()
    mock_client.post.return_value = _make_mock_http_response(200)

    engine = GroqReasoningEngine(
        api_key="test_groq_key",
        model_name="llama-3.1-8b-instruct",
        http_client=mock_client,
    )

    result = engine.reason(valid_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "llama-3.1-8b-instruct"
    assert result.agent_metadata.reasoning_mode == "fallback_model"
    assert result.agent_metadata.status == "success"
    mock_client.post.assert_called_once()


def test_groq_engine_sends_authorization_header(valid_context):
    """Verify that GroqReasoningEngine sends Bearer Authorization header with API key."""
    mock_client = MagicMock()
    mock_client.post.return_value = _make_mock_http_response(200)

    engine = GroqReasoningEngine(
        api_key="gsk_test_api_key_value",
        model_name="llama-3.1-8b-instruct",
        http_client=mock_client,
    )
    engine.reason(valid_context)

    call_kwargs = mock_client.post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers") or {}
    assert headers.get("Authorization") == "Bearer gsk_test_api_key_value"


def test_groq_engine_sends_correct_model_in_payload(valid_context):
    """Verify that GroqReasoningEngine sends the configured model name in the request payload."""
    mock_client = MagicMock()
    mock_client.post.return_value = _make_mock_http_response(200)

    engine = GroqReasoningEngine(
        api_key="test_groq_key",
        model_name="llama-3.1-8b-instruct",
        http_client=mock_client,
    )
    engine.reason(valid_context)

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json") or {}
    assert payload.get("model") == "llama-3.1-8b-instruct"


def test_groq_engine_missing_api_key_raises_unavailable(valid_context):
    """Verify that GroqReasoningEngine raises EngineUnavailableError when API key is not set."""
    engine = GroqReasoningEngine(api_key="", model_name="llama-3.1-8b-instruct")

    with pytest.raises(EngineUnavailableError) as exc_info:
        engine.reason(valid_context)

    assert "GROQ_API_KEY" in str(exc_info.value)


def test_groq_engine_authentication_failure_raises_unavailable(valid_context):
    """Verify HTTP 401 raises EngineUnavailableError with authentication failure message."""
    mock_client = MagicMock()
    mock_client.post.return_value = _make_mock_http_response(401, '{"error": "invalid_api_key"}')

    engine = GroqReasoningEngine(api_key="bad_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineUnavailableError) as exc_info:
        engine.reason(valid_context)

    assert "authentication failed" in str(exc_info.value).lower()


def test_groq_engine_rate_limit_raises_unavailable(valid_context):
    """Verify HTTP 429 raises EngineUnavailableError with rate limit message."""
    mock_client = MagicMock()
    mock_client.post.return_value = _make_mock_http_response(429, '{"error": "rate_limit_exceeded"}')

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineUnavailableError) as exc_info:
        engine.reason(valid_context)

    assert "429" in str(exc_info.value)


def test_groq_engine_model_not_found_raises_unavailable(valid_context):
    """Verify HTTP 404 raises EngineUnavailableError with model not found message."""
    mock_client = MagicMock()
    mock_client.post.return_value = _make_mock_http_response(404, '{"error": "model_not_found"}')

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineUnavailableError) as exc_info:
        engine.reason(valid_context)

    assert "404" in str(exc_info.value)


def test_groq_engine_connection_error_raises_unavailable(valid_context):
    """Verify that network connection failure to Groq API raises EngineUnavailableError."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection refused to api.groq.com")

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineUnavailableError) as exc_info:
        engine.reason(valid_context)

    assert "Groq API unreachable" in str(exc_info.value)


def test_groq_engine_timeout_raises_execution_error(valid_context):
    """Verify that request timeout raises EngineExecutionError."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.TimeoutException("Request timed out")

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineExecutionError) as exc_info:
        engine.reason(valid_context)

    assert "timed out" in str(exc_info.value).lower()


def test_groq_engine_http_500_raises_execution_error(valid_context):
    """Verify HTTP 500 server error raises EngineExecutionError."""
    mock_client = MagicMock()
    mock_client.post.return_value = _make_mock_http_response(500, '{"error": "internal_server_error"}')

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineExecutionError) as exc_info:
        engine.reason(valid_context)

    assert "500" in str(exc_info.value)


def test_groq_engine_invalid_json_raises_execution_error(valid_context):
    """Verify that non-JSON response from Groq API raises EngineExecutionError."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "This is not valid JSON { broken"}}]
    }
    mock_client.post.return_value = mock_resp

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineExecutionError) as exc_info:
        engine.reason(valid_context)

    assert "parse" in str(exc_info.value).lower() or "json" in str(exc_info.value).lower()


def test_groq_engine_empty_content_raises_execution_error(valid_context):
    """Verify that an empty response content from Groq API raises EngineExecutionError."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": ""}}]
    }
    mock_client.post.return_value = mock_resp

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    with pytest.raises(EngineExecutionError) as exc_info:
        engine.reason(valid_context)

    assert "empty" in str(exc_info.value).lower()


def test_groq_engine_markdown_wrapped_json_is_parsed(valid_context):
    """Verify that Groq response wrapped in markdown json code blocks is parsed correctly."""
    wrapped_content = f"```json\n{VALID_GROQ_RESPONSE_JSON}\n```"

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": wrapped_content}}]
    }
    mock_client.post.return_value = mock_resp

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    result = engine.reason(valid_context)
    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.reasoning_mode == "fallback_model"


def test_groq_engine_extraneous_prose_before_json_is_parsed(valid_context):
    """Verify that Groq response with leading prose before JSON is parsed correctly."""
    wrapped_content = f"Clinical synthesis follows.\n\n{VALID_GROQ_RESPONSE_JSON}\n\nEnd of response."

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": wrapped_content}}]
    }
    mock_client.post.return_value = mock_resp

    engine = GroqReasoningEngine(api_key="test_groq_key", model_name="llama-3.1-8b-instruct", http_client=mock_client)

    result = engine.reason(valid_context)
    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.reasoning_mode == "fallback_model"
