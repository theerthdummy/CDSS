"""Unit tests for Agent 5 Clinical Reasoning Engine, multi-level cascade, and validation gates."""

import copy
from unittest.mock import MagicMock
import pytest

from app.config import settings
from app.models.feedback_models import FeedbackRequest, InsufficientEvidenceTermination
from app.models.input_models import (
    ConflictEvidence,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import ClinicalDecisionSupportResponse, ReasoningPolicy
from app.services.adaptive_optimizer import AdaptiveOptimizer
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.prompts import build_clinical_reasoning_prompt
from app.services.reasoning_engines import (
    DeterministicFallbackEngine,
    EngineExecutionError,
    ReasoningEngine,
)


@pytest.fixture
def high_confidence_context():
    """Valid high confidence context fixture."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute STEMI requires immediate emergency reperfusion therapy",
                sources=["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            )
        ],
        normalized_medical_terms=["STEMI", "Primary PCI"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Primary PCI within 90 minutes reduces acute mortality",
                source_attribution="Agent 3 Guidelines",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[],
        source_traceability=[
            SourceTraceability(
                finding="STEMI reperfusion protocol",
                sources=["PubMed", "Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Acute STEMI requiring primary PCI.",
        confidence_score=0.92,
    )


@pytest.fixture
def low_confidence_context():
    """Valid low confidence context fixture."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(finding="Atypical chest pain", sources=["Agent 2 Biomedical RAG"])
        ],
        normalized_medical_terms=["Chest Pain"],
        supporting_evidence=[],
        conflicting_evidence=[],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Equivocal chest pain presentation.",
        confidence_score=0.45,
    )


@pytest.fixture
def valid_response_payload():
    """Valid response payload dictionary."""
    return {
        "clinical_assessment": {
            "primary_interpretation": "Acute STEMI requiring emergency primary PCI reperfusion therapy.",
            "clinical_significance": "Critical",
            "differential_considerations": ["Acute STEMI", "Aortic Dissection"],
        },
        "reasoning": {
            "key_findings": ["ST elevation on ECG"],
            "supporting_factors": ["Primary PCI guideline recommendation"],
            "conflicting_factors": [],
            "reasoning_summary": "Primary PCI indicated urgently based on STEMI criteria.",
        },
        "decision_support": {
            "recommended_actions": ["Activate Cardiac Cath Lab immediately"],
            "additional_information_needed": [],
            "priority_level": "Urgent",
        },
        "safety": {
            "safety_flags": [],
            "contraindications_or_concerns": [],
        },
        "uncertainty": {
            "confidence_score": 0.92,
            "uncertainty_factors": [],
        },
        "traceability": [
            {
                "reasoning_item": "Primary PCI indication",
                "supported_by_findings": ["Acute STEMI requires immediate emergency reperfusion therapy"],
                "upstream_sources": ["Agent 2", "Agent 3"],
            }
        ],
        "agent_metadata": {
            "agent": "agent_5",
            "model": settings.primary_reasoning_model,
            "reasoning_mode": "primary",
            "status": "success",
        },
    }


def test_gemini_success_path(high_confidence_context, valid_response_payload):
    """Test 1: Gemini succeeds and output passes validation -> returned directly. Groq/Deterministic NOT called."""
    valid_resp = ClinicalDecisionSupportResponse.model_validate(valid_response_payload)
    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.return_value = valid_resp

    fallback = MagicMock(spec=ReasoningEngine)
    deterministic = MagicMock(spec=ReasoningEngine)

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.primary_reasoning_model
    primary.reason.assert_called_once()
    fallback.reason.assert_not_called()
    deterministic.reason.assert_not_called()


def test_gemini_provider_failure_cascades_to_groq(high_confidence_context, valid_response_payload):
    """Test 2: Gemini API fails -> cascades to Groq -> Groq succeeds."""
    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.side_effect = EngineExecutionError("Gemini API connection error")

    groq_payload = copy.deepcopy(valid_response_payload)
    groq_payload["agent_metadata"]["model"] = settings.fallback_reasoning_model
    groq_payload["agent_metadata"]["reasoning_mode"] = "fallback_model"
    groq_resp = ClinicalDecisionSupportResponse.model_validate(groq_payload)

    fallback = MagicMock(spec=ReasoningEngine)
    fallback.reason.return_value = groq_resp

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.fallback_reasoning_model
    assert result.agent_metadata.reasoning_mode == "fallback_model"


def test_gemini_malformed_json_cascades_to_groq(high_confidence_context, valid_response_payload):
    """Test 3: Gemini returns malformed JSON -> cascades to Groq."""
    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.side_effect = EngineExecutionError("Failed to parse Gemini response as JSON")

    groq_payload = copy.deepcopy(valid_response_payload)
    groq_payload["agent_metadata"]["model"] = settings.fallback_reasoning_model
    groq_resp = ClinicalDecisionSupportResponse.model_validate(groq_payload)

    fallback = MagicMock(spec=ReasoningEngine)
    fallback.reason.return_value = groq_resp

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.fallback_reasoning_model


def test_gemini_output_validator_rejection_cascades_to_groq(high_confidence_context, valid_response_payload):
    """Test 4: Gemini output rejected by ClinicalOutputValidator -> cascades to Groq."""
    invalid_gemini_payload = copy.deepcopy(valid_response_payload)
    invalid_gemini_payload["traceability"][0]["upstream_sources"] = ["EVID-999"]
    invalid_gemini_resp = ClinicalDecisionSupportResponse.model_validate(invalid_gemini_payload)

    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.return_value = invalid_gemini_resp

    groq_payload = copy.deepcopy(valid_response_payload)
    groq_payload["agent_metadata"]["model"] = settings.fallback_reasoning_model
    groq_resp = ClinicalDecisionSupportResponse.model_validate(groq_payload)

    fallback = MagicMock(spec=ReasoningEngine)
    fallback.reason.return_value = groq_resp

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.fallback_reasoning_model


def test_groq_provider_failure_cascades_to_deterministic(high_confidence_context):
    """Test 5: Gemini fails + Groq fails -> cascades to Deterministic Fallback Engine."""
    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.side_effect = EngineExecutionError("Gemini API error")

    fallback = MagicMock(spec=ReasoningEngine)
    fallback.reason.side_effect = EngineExecutionError("Groq API unavailable")

    deterministic = DeterministicFallbackEngine()

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "deterministic-fallback"
    assert result.agent_metadata.reasoning_mode == "deterministic_fallback"


def test_both_models_invalid_output_cascades_to_deterministic(high_confidence_context, valid_response_payload):
    """Test 6: Gemini & Groq outputs fail validation -> Deterministic Fallback Engine invoked."""
    invalid_payload = copy.deepcopy(valid_response_payload)
    invalid_payload["traceability"][0]["upstream_sources"] = ["EVID-999"]
    invalid_resp = ClinicalDecisionSupportResponse.model_validate(invalid_payload)

    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.return_value = invalid_resp

    fallback = MagicMock(spec=ReasoningEngine)
    fallback.reason.return_value = invalid_resp

    deterministic = DeterministicFallbackEngine()

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "deterministic-fallback"


def test_adaptive_optimizer_gate_needs_more_evidence(low_confidence_context):
    """Test 7: AdaptiveOptimizer returns NEEDS_MORE_EVIDENCE. Reasoning engines NOT called."""
    primary = MagicMock(spec=ReasoningEngine)
    fallback = MagicMock(spec=ReasoningEngine)
    deterministic = MagicMock(spec=ReasoningEngine)

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(low_confidence_context, retry_count=0)

    assert isinstance(result, FeedbackRequest)
    assert result.status == "needs_more_evidence"

    # Assert NO reasoning engine was invoked
    primary.reason.assert_not_called()
    fallback.reason.assert_not_called()
    deterministic.reason.assert_not_called()


def test_adaptive_optimizer_gate_insufficient_evidence(low_confidence_context):
    """Test 8: AdaptiveOptimizer returns INSUFFICIENT_EVIDENCE. Reasoning engines NOT called."""
    primary = MagicMock(spec=ReasoningEngine)
    fallback = MagicMock(spec=ReasoningEngine)
    deterministic = MagicMock(spec=ReasoningEngine)

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(low_confidence_context, retry_count=3)

    assert isinstance(result, InsufficientEvidenceTermination)
    assert result.status == "insufficient_evidence"

    primary.reason.assert_not_called()
    fallback.reason.assert_not_called()
    deterministic.reason.assert_not_called()


def test_prompt_contains_untrusted_data_boundary_and_policy(high_confidence_context):
    """Test 9: Prompt builder includes untrusted data boundary rules, ReasoningPolicy, and evidence grounding."""
    policy = ReasoningPolicy()
    prompt = build_clinical_reasoning_prompt(high_confidence_context, policy=policy)

    assert "[CLINICAL DATA CONTEXT - FOR REASONING ANALYSIS ONLY]" in prompt
    assert "[REASONING POLICY CONSTRAINTS" in prompt
    assert "Reasoning Scope: CLINICAL_DECISION_SUPPORT" in prompt
    assert "Target Certainty Level: LIKELY" in prompt


def test_no_infinite_fallback_loop(high_confidence_context):
    """Test 10: Ensures each engine is called AT MOST ONCE during a reasoning cycle."""
    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.side_effect = EngineExecutionError("Primary model failed")

    fallback = MagicMock(spec=ReasoningEngine)
    fallback.reason.side_effect = EngineExecutionError("Fallback model failed")

    deterministic = DeterministicFallbackEngine()

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )

    service.execute_reasoning_pipeline(high_confidence_context)

    assert primary.reason.call_count == 1
    assert fallback.reason.call_count == 1
