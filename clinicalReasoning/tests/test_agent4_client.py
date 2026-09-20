"""Unit test suite for Agent4Client and Agent 4 service contract integration.

Operates 100% offline using unittest.mock to verify schema compatibility,
HTTP communication, error handling, timeout handling, confidence propagation,
and traceability preservation.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest
from pydantic import ValidationError

from app.clients.agent4_client import (
    Agent4Client,
    Agent4ClientError,
    Agent4UnavailableError,
    Agent4ValidationError,
)
from app.models.input_models import UnifiedClinicalContext
from app.services.adaptive_optimizer import AdaptiveOptimizer


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def actual_agent4_fusion_response_json():
    """Actual sample FusionResponse JSON dictionary produced by Agent 4 POST /api/v1/fuse."""
    return {
        "status": "success",
        "message": "Clinical data fusion completed successfully via llama_enhanced.",
        "unified_context": {
            "merged_findings": [
                {
                    "finding": "Acute ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy",
                    "sources": ["Agent 2 Biomedical RAG", "Agent 3 PubMed"]
                },
                {
                    "finding": "Primary Percutaneous Coronary Intervention (PCI) within 90 minutes reduces acute mortality",
                    "sources": ["Agent 3 Clinical Guidelines"]
                }
            ],
            "normalized_medical_terms": [
                "STEMI",
                "Primary PCI",
                "Aspirin",
                "Heparin"
            ],
            "supporting_evidence": [
                {
                    "finding": "Early reperfusion within 90 minutes of medical contact improves 30-day survival",
                    "source_attribution": "Agent 3 Clinical Guidelines"
                }
            ],
            "conflicting_evidence": [],
            "evidence_priority": [
                {
                    "finding": "STEMI emergency reperfusion therapy",
                    "priority_score": 0.95,
                    "primary_source": "Agent 3 Clinical Guidelines"
                }
            ],
            "source_traceability": [
                {
                    "finding": "STEMI emergency reperfusion therapy",
                    "sources": ["PubMed", "Clinical Guidelines"],
                    "upstream_agents": ["Agent 2", "Agent 3"]
                }
            ],
            "fusion_summary": "Patient presentation and fused clinical evidence strongly support acute STEMI requiring emergency PCI.",
            "confidence_score": 0.85
        },
        "metadata": {
            "correlation_id": "test-corr-12345",
            "timestamp": "2026-08-10T06:00:00Z",
            "processing_time_ms": 145.2,
            "fusion_mode": "llama_enhanced",
            "fusion_method": "llama_enhanced",
            "llm_provider": "groq",
            "llm_model": "llama-3.1-8b-instruct",
            "llm_used": True,
            "fallback_used": False,
            "fallback_reason": None
        }
    }


@pytest.fixture
def agent2_agent3_fusion_request_payload():
    return {
        "agent2_output": {
            "rag_evidence": [{"text": "STEMI guidelines support primary PCI", "score": 0.95}],
            "kg_evidence": [{"subject": "STEMI", "predicate": "treated_by", "object": "Primary PCI"}]
        },
        "agent3_output": {
            "medical_evidence": [{"finding": "Primary PCI within 90 mins", "source": "PubMed"}]
        }
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_actual_agent4_response_schema_compatibility(actual_agent4_fusion_response_json):
    """Test 1: Agent 4 FusionResponse JSON parses directly into UnifiedClinicalContext with zero translation layer."""
    client = Agent4Client()
    context = client.extract_unified_context_from_response(actual_agent4_fusion_response_json)

    assert isinstance(context, UnifiedClinicalContext)
    assert len(context.merged_findings) == 2
    assert "STEMI" in context.normalized_medical_terms
    assert context.confidence_score == 0.85


def test_successful_mocked_agent4_http_response(agent2_agent3_fusion_request_payload, actual_agent4_fusion_response_json):
    """Test 2: Successful HTTP response from Agent 4 /api/v1/fuse returns validated UnifiedClinicalContext."""
    client = Agent4Client(base_url="http://test-agent4:8004")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = actual_agent4_fusion_response_json

    with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
        context = client.fuse_context(agent2_agent3_fusion_request_payload, correlation_id="corr-999")

        assert isinstance(context, UnifiedClinicalContext)
        assert context.confidence_score == 0.85
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test-agent4:8004/api/v1/fuse"
        assert kwargs["headers"]["X-Correlation-ID"] == "corr-999"


def test_agent4_timeout_raises_unavailable_error(agent2_agent3_fusion_request_payload):
    """Test 3: Timeout communicating with Agent 4 raises Agent4UnavailableError."""
    client = Agent4Client(base_url="http://test-agent4:8004", timeout_seconds=5.0)

    with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(Agent4UnavailableError) as exc_info:
            client.fuse_context(agent2_agent3_fusion_request_payload)

        assert "timed out" in str(exc_info.value)


def test_agent4_http_500_raises_unavailable_error(agent2_agent3_fusion_request_payload):
    """Test 4: Agent 4 returning HTTP 500 error raises Agent4UnavailableError."""
    client = Agent4Client(base_url="http://test-agent4:8004")

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error in Agent 4"

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(Agent4UnavailableError) as exc_info:
            client.fuse_context(agent2_agent3_fusion_request_payload)

        assert "HTTP 500" in str(exc_info.value)


def test_agent4_malformed_json_raises_validation_error(agent2_agent3_fusion_request_payload):
    """Test 5: Non-JSON response from Agent 4 raises Agent4ValidationError."""
    client = Agent4Client(base_url="http://test-agent4:8004")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON string")

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(Agent4ValidationError) as exc_info:
            client.fuse_context(agent2_agent3_fusion_request_payload)

        assert "malformed non-JSON" in str(exc_info.value)


def test_agent4_invalid_unified_context_schema_raises_validation_error(agent2_agent3_fusion_request_payload):
    """Test 6: Agent 4 returning structurally invalid UnifiedClinicalContext raises Agent4ValidationError."""
    client = Agent4Client(base_url="http://test-agent4:8004")

    invalid_response = {
        "status": "success",
        "unified_context": {
            "merged_findings": "invalid_string_not_a_list",  # Invalid type
            "confidence_score": 0.80
        }
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = invalid_response

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(Agent4ValidationError) as exc_info:
            client.fuse_context(agent2_agent3_fusion_request_payload)

        assert "failed schema validation" in str(exc_info.value)


def test_agent4_confidence_score_propagation_to_adaptive_optimizer(actual_agent4_fusion_response_json):
    """Test 7: Agent 4 confidence score propagates unchanged to AdaptiveOptimizer."""
    client = Agent4Client()
    optimizer = AdaptiveOptimizer()
    
    # High confidence context (0.85) -> READY_FOR_REASONING
    high_conf_data = actual_agent4_fusion_response_json.copy()
    high_conf_data["unified_context"]["confidence_score"] = 0.85
    high_context = client.extract_unified_context_from_response(high_conf_data)
    eval_high = optimizer.evaluate_context(high_context)
    assert eval_high.status == "ready_for_reasoning"

    # Low confidence context (0.45) -> NEEDS_MORE_EVIDENCE
    low_conf_data = actual_agent4_fusion_response_json.copy()
    low_conf_data["unified_context"]["confidence_score"] = 0.45
    low_context = client.extract_unified_context_from_response(low_conf_data)
    eval_low = optimizer.evaluate_context(low_context)
    assert eval_low.status == "needs_more_evidence"


def test_agent4_source_traceability_propagation(actual_agent4_fusion_response_json):
    """Test 8: Agent 4 source_traceability mappings propagate unchanged through Agent 5 models."""
    client = Agent4Client()
    context = client.extract_unified_context_from_response(actual_agent4_fusion_response_json)

    assert len(context.source_traceability) == 1
    trace = context.source_traceability[0]
    assert trace.finding == "STEMI emergency reperfusion therapy"
    assert "PubMed" in trace.sources
    assert "Agent 2" in trace.upstream_agents
    assert "Agent 3" in trace.upstream_agents
