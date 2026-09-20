"""End-to-End Integration and Demo Tests for Agent 4 -> Agent 5 Pipeline Boundary.

Validates:
1. High-confidence payload processing & full reasoning cascade execution.
2. Low-confidence gating preventing LLM invocation and emitting structured FeedbackRequest.
3. Maximum retries safe termination (InsufficientEvidenceTermination).
4. Provider fallback cascade (Gemini -> Groq -> Deterministic Python Fallback).
5. Source traceability preservation.
6. Direct schema compatibility without an Agent 5 translation layer.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.models.feedback_models import FeedbackRequest, InsufficientEvidenceTermination
from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import (
    AgentMetadata,
    ClinicalAssessment,
    ClinicalDecisionSupportResponse,
    ClinicalReasoningDetail,
    DecisionSupport,
    SafetyAnalysis,
    TraceabilityLink,
    UncertaintyAssessment,
)
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.reasoning_engines import (
    EngineExecutionError,
    EngineUnavailableError,
    ReasoningEngine,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def agent4_high_confidence_json():
    fixture_path = FIXTURES_DIR / "agent4_high_confidence.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def agent4_low_confidence_json():
    fixture_path = FIXTURES_DIR / "agent4_low_confidence.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_cdss_response():
    return ClinicalDecisionSupportResponse(
        clinical_assessment=ClinicalAssessment(
            primary_interpretation="Acute STEMI requiring emergency primary PCI reperfusion therapy.",
            clinical_significance="Critical",
            differential_considerations=["Acute Inferior STEMI", "Aortic Dissection"],
        ),
        reasoning=ClinicalReasoningDetail(
            key_findings=["ST-elevation in leads II, III, aVF"],
            supporting_factors=["Primary PCI within 90 minutes reduces acute mortality"],
            conflicting_factors=[],
            reasoning_summary="Fused evidence strongly supports acute STEMI emergency protocol.",
        ),
        decision_support=DecisionSupport(
            recommended_actions=["Activate Emergency PCI Protocol", "Administer Aspirin 325 mg"],
            additional_information_needed=["Baseline serum creatinine"],
            priority_level="Urgent",
        ),
        safety=SafetyAnalysis(
            safety_flags=["Monitor for post-reperfusion arrhythmias"],
            contraindications_or_concerns=[],
        ),
        uncertainty=UncertaintyAssessment(
            confidence_score=0.92,
            fusion_confidence=0.92,
            reasoning_confidence=0.92,
            uncertainty_factors=[],
        ),
        traceability=[
            TraceabilityLink(
                reasoning_item="STEMI emergency reperfusion therapy",
                supported_by_findings=["Acute ST-elevation myocardial infarction (STEMI)"],
                upstream_sources=["Agent 2", "Agent 3"],
            )
        ],
        agent_metadata=AgentMetadata(
            agent="agent_5",
            model="gemini-2.5-flash",
            reasoning_mode="primary",
            status="success",
        ),
    )


# -----------------------------------------------------------------------------
# Part 3: High-Confidence End-to-End Test
# -----------------------------------------------------------------------------

def test_e2e_high_confidence_agent4_to_agent5_pipeline(agent4_high_confidence_json, mock_cdss_response):
    """Test Part 3: High-confidence Agent 4 context passes optimizer gating and completes reasoning cascade."""
    context = UnifiedClinicalContext.model_validate(agent4_high_confidence_json)
    assert context.confidence_score >= 0.70

    mock_primary = MagicMock(spec=ReasoningEngine)
    mock_primary.reason.return_value = mock_cdss_response
    mock_fallback = MagicMock(spec=ReasoningEngine)

    svc = ClinicalReasoningService(primary_engine=mock_primary, fallback_engine=mock_fallback)
    result = svc.execute_reasoning_pipeline(context, retry_count=0)

    # Assert final response type, provider call, and metadata
    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "gemini-2.5-flash"
    assert result.agent_metadata.reasoning_mode == "primary"
    assert mock_primary.reason.call_count == 1
    assert mock_fallback.reason.call_count == 0


# -----------------------------------------------------------------------------
# Part 4: Low-Confidence End-to-End Test (Reasoning Blocked)
# -----------------------------------------------------------------------------

def test_e2e_low_confidence_blocks_reasoning_and_emits_feedback(agent4_low_confidence_json):
    """Test Part 4: Low-confidence Agent 4 context blocks LLM invocation and emits FeedbackRequest."""
    context = UnifiedClinicalContext.model_validate(agent4_low_confidence_json)
    assert context.confidence_score < 0.70

    mock_primary = MagicMock(spec=ReasoningEngine)
    mock_fallback = MagicMock(spec=ReasoningEngine)

    svc = ClinicalReasoningService(primary_engine=mock_primary, fallback_engine=mock_fallback)
    result = svc.execute_reasoning_pipeline(context, retry_count=0)

    # Assert structured feedback request returned
    assert isinstance(result, FeedbackRequest)
    assert result.status == "needs_more_evidence"
    assert result.retry_count == 0

    # CRITICAL: Verify NO reasoning engines were invoked
    assert mock_primary.reason.call_count == 0
    assert mock_fallback.reason.call_count == 0


# -----------------------------------------------------------------------------
# Part 5: Maximum Retry Termination Test
# -----------------------------------------------------------------------------

def test_e2e_max_retry_termination_safely_halts_pipeline(agent4_low_confidence_json):
    """Test Part 5: Low-confidence context at retry_count >= max_retries yields InsufficientEvidenceTermination."""
    context = UnifiedClinicalContext.model_validate(agent4_low_confidence_json)

    mock_primary = MagicMock(spec=ReasoningEngine)
    mock_fallback = MagicMock(spec=ReasoningEngine)

    svc = ClinicalReasoningService(primary_engine=mock_primary, fallback_engine=mock_fallback)

    # Retry count 0, 1, 2 return FeedbackRequest
    assert isinstance(svc.execute_reasoning_pipeline(context, retry_count=0), FeedbackRequest)
    assert isinstance(svc.execute_reasoning_pipeline(context, retry_count=1), FeedbackRequest)
    assert isinstance(svc.execute_reasoning_pipeline(context, retry_count=2), FeedbackRequest)

    # Retry count 3 (max_retries=3) terminates safely
    final_result = svc.execute_reasoning_pipeline(context, retry_count=3)
    assert isinstance(final_result, InsufficientEvidenceTermination)
    assert final_result.status == "insufficient_evidence"
    assert final_result.retry_count == 3

    # CRITICAL: Verify zero reasoning engine calls throughout
    assert mock_primary.reason.call_count == 0
    assert mock_fallback.reason.call_count == 0


# -----------------------------------------------------------------------------
# Part 6: Provider Fallback End-to-End Scenarios
# -----------------------------------------------------------------------------

def test_e2e_provider_fallback_scenario_a_gemini_succeeds(agent4_high_confidence_json, mock_cdss_response):
    """Scenario A: Gemini succeeds -> Groq & Deterministic NOT called."""
    context = UnifiedClinicalContext.model_validate(agent4_high_confidence_json)

    mock_primary = MagicMock(spec=ReasoningEngine)
    mock_primary.reason.return_value = mock_cdss_response
    mock_fallback = MagicMock(spec=ReasoningEngine)

    svc = ClinicalReasoningService(primary_engine=mock_primary, fallback_engine=mock_fallback)
    res = svc.execute_reasoning(context)

    assert res.agent_metadata.reasoning_mode == "primary"
    assert mock_primary.reason.call_count == 1
    assert mock_fallback.reason.call_count == 0


def test_e2e_provider_fallback_scenario_b_gemini_fails_groq_succeeds(agent4_high_confidence_json, mock_cdss_response):
    """Scenario B: Gemini fails -> Groq called and succeeds -> Deterministic NOT called."""
    context = UnifiedClinicalContext.model_validate(agent4_high_confidence_json)

    mock_primary = MagicMock(spec=ReasoningEngine)
    mock_primary.reason.side_effect = EngineExecutionError("Gemini API HTTP 500 Internal Server Error")

    mock_cdss_response.agent_metadata.model = "llama-3.1-8b-instruct"
    mock_cdss_response.agent_metadata.reasoning_mode = "fallback_model"
    mock_fallback = MagicMock(spec=ReasoningEngine)
    mock_fallback.reason.return_value = mock_cdss_response

    svc = ClinicalReasoningService(primary_engine=mock_primary, fallback_engine=mock_fallback)
    res = svc.execute_reasoning(context)

    assert res.agent_metadata.reasoning_mode == "fallback_model"
    assert mock_primary.reason.call_count == 1
    assert mock_fallback.reason.call_count == 1


def test_e2e_provider_fallback_scenario_c_both_fail_deterministic_succeeds(agent4_high_confidence_json):
    """Scenario C: Gemini and Groq both fail -> Level 3 Deterministic Python Engine succeeds."""
    context = UnifiedClinicalContext.model_validate(agent4_high_confidence_json)

    mock_primary = MagicMock(spec=ReasoningEngine)
    mock_primary.reason.side_effect = EngineExecutionError("Gemini down")

    mock_fallback = MagicMock(spec=ReasoningEngine)
    mock_fallback.reason.side_effect = EngineUnavailableError("Groq down")

    svc = ClinicalReasoningService(primary_engine=mock_primary, fallback_engine=mock_fallback)
    res = svc.execute_reasoning(context)

    assert res.agent_metadata.model == "deterministic-fallback"
    assert res.agent_metadata.reasoning_mode == "deterministic_fallback"
    assert mock_primary.reason.call_count == 1
    assert mock_fallback.reason.call_count == 1


# -----------------------------------------------------------------------------
# Part 7: Traceability Preservation Test
# -----------------------------------------------------------------------------

def test_e2e_traceability_preservation_from_agent4(agent4_high_confidence_json, mock_cdss_response):
    """Test Part 7: Final response traceability maps directly to Agent 2 and Agent 3 upstream sources."""
    context = UnifiedClinicalContext.model_validate(agent4_high_confidence_json)

    mock_primary = MagicMock(spec=ReasoningEngine)
    mock_primary.reason.return_value = mock_cdss_response
    mock_fallback = MagicMock(spec=ReasoningEngine)

    svc = ClinicalReasoningService(primary_engine=mock_primary, fallback_engine=mock_fallback)
    res = svc.execute_reasoning(context)

    assert len(res.traceability) > 0
    link = res.traceability[0]
    assert "Agent 2" in link.upstream_sources
    assert "Agent 3" in link.upstream_sources


# -----------------------------------------------------------------------------
# Part 10: Contract Compatibility Test
# -----------------------------------------------------------------------------

def test_e2e_agent4_schema_compatibility_no_translation_layer(agent4_high_confidence_json):
    """Test Part 10: Synthetic Agent 4 JSON output loads directly into UnifiedClinicalContext with zero translation layer."""
    json_str = json.dumps(agent4_high_confidence_json)

    # Validate direct parsing from JSON string produced by Agent 4
    context = UnifiedClinicalContext.model_validate_json(json_str)

    assert len(context.merged_findings) == 2
    assert "STEMI" in context.normalized_medical_terms
    assert context.confidence_score == 0.92
