"""Unit tests for Agent 5 Clinical Reasoning Policy Evaluator and Policy Enforcement.

Verifies deterministic ClinicalReasoningPolicyEvaluator behavior across strong evidence, explicit confirmation,
conflicting evidence, low confidence optimizer boundaries, provider-independent prompts, and ClinicalOutputValidator compliance.
"""

from unittest.mock import MagicMock
import pytest

from app.config import settings
from app.models.feedback_models import FeedbackRequest, InsufficientEvidenceTermination
from app.models.input_models import (
    ConflictEvidence,
    EvidencePriority,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import (
    ClinicalAssessment,
    ClinicalDecisionSupportResponse,
    ReasoningPolicy,
)
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.output_validator import ClinicalOutputValidator
from app.services.policy_evaluator import ClinicalReasoningPolicyEvaluator, REASONING_POLICY_VERSION
from app.services.prompts import build_clinical_reasoning_prompt
from app.services.reasoning_engines import DeterministicFallbackEngine, GeminiReasoningEngine, GroqReasoningEngine


@pytest.fixture
def strong_evidence_context():
    """UnifiedClinicalContext payload with high confidence score and strong supporting evidence, no conflicts."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="ST-elevation myocardial infarction (STEMI) requires immediate emergency reperfusion therapy",
                sources=["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            )
        ],
        normalized_medical_terms=["STEMI", "Reperfusion Therapy", "Aspirin"],
        supporting_evidence=[
            SupportingEvidence(
                finding="STEMI requires immediate emergency reperfusion therapy via primary PCI or fibrinolysis",
                source_attribution="Agent 2 Biomedical RAG",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[
            EvidencePriority(
                finding="ST-elevation myocardial infarction",
                priority_score=0.95,
                primary_source="Agent 3 Clinical Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="ST-elevation myocardial infarction",
                sources=["PubMed", "Clinical Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Patient presentation and evidence indicate acute STEMI requiring emergency reperfusion.",
        confidence_score=0.92,
    )


@pytest.fixture
def explicit_confirmation_context():
    """UnifiedClinicalContext payload explicitly containing confirmation evidence text."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute STEMI confirmed by 12-lead ECG elevation in V1-V4 and serial troponin rise",
                sources=["Agent 2 Biomedical RAG"],
            )
        ],
        normalized_medical_terms=["STEMI", "Troponin"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Laboratory confirmed troponin I level of 4.5 ng/mL",
                source_attribution="Agent 2",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Laboratory confirmed acute STEMI.",
        confidence_score=0.95,
    )


@pytest.fixture
def conflicting_evidence_context():
    """UnifiedClinicalContext payload containing conflicting evidence."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Chest pain suspicious for STEMI",
                sources=["Agent 2"],
            )
        ],
        normalized_medical_terms=["Chest Pain"],
        supporting_evidence=[
            SupportingEvidence(
                finding="ST elevation on ECG",
                source_attribution="Agent 2",
            )
        ],
        conflicting_evidence=[
            ConflictEvidence(
                type="Diagnostic Contradiction",
                finding="Normal Troponin",
                description="Serial troponin I levels remain within normal limits at 3 hours.",
            )
        ],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Acute chest pain with ST elevation but normal cardiac biomarkers.",
        confidence_score=0.75,
    )


def test_policy_evaluator_strong_evidence_default_likely(strong_evidence_context):
    """Verify that high confidence without explicit confirmation keywords evaluates to LIKELY, not CONFIRMED."""
    policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(strong_evidence_context)

    assert isinstance(policy, ReasoningPolicy)
    assert policy.policy_version == REASONING_POLICY_VERSION
    assert policy.reasoning_allowed is True
    assert policy.certainty_level == "LIKELY"
    assert policy.reasoning_scope == "CLINICAL_DECISION_SUPPORT"
    assert policy.uncertainty_required is False


def test_policy_evaluator_explicit_confirmation_evidence(explicit_confirmation_context):
    """Verify that context containing explicit confirmation keywords and high confidence permits CONFIRMED certainty."""
    policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(explicit_confirmation_context)

    assert policy.reasoning_allowed is True
    assert policy.certainty_level == "CONFIRMED"


def test_policy_evaluator_conflicting_evidence_requires_uncertainty(conflicting_evidence_context):
    """Verify that conflicting evidence forces uncertainty_required=True and downgrades certainty level."""
    policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(conflicting_evidence_context)

    assert policy.certainty_level in ["POSSIBLE", "UNCERTAIN"]
    assert policy.uncertainty_required is True
    assert len(policy.policy_warnings) > 0
    assert any("Conflicting evidence" in w for w in policy.policy_warnings)


def test_policy_does_not_override_adaptive_optimizer(strong_evidence_context):
    """Verify that low confidence context returns FeedbackRequest from AdaptiveOptimizer; PolicyEvaluator is NOT invoked."""
    low_conf_context = strong_evidence_context.model_copy(update={"confidence_score": 0.45})

    service = ClinicalReasoningService(use_dev_default_when_unimplemented=False)
    res = service.execute_reasoning_pipeline(low_conf_context, retry_count=0)

    assert isinstance(res, FeedbackRequest)
    assert res.status == "needs_more_evidence"


def test_policy_insufficient_evidence_termination(strong_evidence_context):
    """Verify that context at max retries returns InsufficientEvidenceTermination without reasoning."""
    low_conf_context = strong_evidence_context.model_copy(update={"confidence_score": 0.45})

    service = ClinicalReasoningService(use_dev_default_when_unimplemented=False)
    res = service.execute_reasoning_pipeline(low_conf_context, retry_count=3)

    assert isinstance(res, InsufficientEvidenceTermination)
    assert res.status == "insufficient_evidence"


def test_prompt_builder_includes_provider_independent_policy(strong_evidence_context):
    """Verify that build_clinical_reasoning_prompt formats ReasoningPolicy fields cleanly for both Gemini and Groq Llama."""
    policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(strong_evidence_context)
    prompt = build_clinical_reasoning_prompt(strong_evidence_context, policy=policy)

    assert "[REASONING POLICY CONSTRAINTS" in prompt
    assert f"Target Certainty Level: {policy.certainty_level}" in prompt
    assert f"Uncertainty Requirement: {policy.uncertainty_required}" in prompt
    assert policy.certainty_policy in prompt
    assert policy.recommendation_policy in prompt


def test_deterministic_fallback_engine_respects_policy(conflicting_evidence_context):
    """Verify that DeterministicFallbackEngine respects ReasoningPolicy certainty and uncertainty settings."""
    policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(conflicting_evidence_context)
    engine = DeterministicFallbackEngine()

    response = engine.reason(conflicting_evidence_context, policy=policy)

    assert isinstance(response, ClinicalDecisionSupportResponse)
    assert f"({policy.certainty_level})" in response.clinical_assessment.primary_interpretation
    assert any("ReasoningPolicy" in u for u in response.uncertainty.uncertainty_factors)


def test_output_validator_rejects_unjustified_confirmed_diagnosis(strong_evidence_context):
    """Verify that ClinicalOutputValidator flags outputs claiming 'confirmed diagnosis' when policy is not CONFIRMED."""
    policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(strong_evidence_context)
    assert policy.certainty_level == "LIKELY"

    mock_resp_dict = {
        "clinical_assessment": {
            "primary_interpretation": "Patient has confirmed diagnosis of Acute STEMI",
            "clinical_significance": "Critical",
            "differential_considerations": ["STEMI"],
        },
        "reasoning": {
            "key_findings": ["STEMI"],
            "supporting_factors": ["ST elevation"],
            "conflicting_factors": [],
            "reasoning_summary": "STEMI present.",
        },
        "decision_support": {
            "recommended_actions": ["Reperfusion"],
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
                "reasoning_item": "STEMI",
                "supported_by_findings": ["STEMI finding"],
                "upstream_sources": ["Agent 2"],
            }
        ],
        "agent_metadata": {
            "agent": "agent_5",
            "model": settings.primary_reasoning_model,
            "reasoning_mode": "primary",
            "status": "success",
        },
    }

    response = ClinicalDecisionSupportResponse.model_validate(mock_resp_dict)
    val_result = ClinicalOutputValidator.validate_response(response, strong_evidence_context, policy=policy)

    assert val_result.valid is False
    assert any(e.code == "UNJUSTIFIED_CONFIRMED_DIAGNOSIS" for e in val_result.errors)


def test_output_validator_rejects_missing_required_uncertainty(conflicting_evidence_context):
    """Verify that ClinicalOutputValidator flags outputs missing uncertainty factors when policy.uncertainty_required=True."""
    policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(conflicting_evidence_context)
    assert policy.uncertainty_required is True

    mock_resp_dict = {
        "clinical_assessment": {
            "primary_interpretation": "Possible Acute Coronary Syndrome",
            "clinical_significance": "High",
            "differential_considerations": ["ACS"],
        },
        "reasoning": {
            "key_findings": ["Chest pain"],
            "supporting_factors": ["ST elevation"],
            "conflicting_factors": ["Normal Troponin"],
            "reasoning_summary": "ACS suspected despite normal troponin.",
        },
        "decision_support": {
            "recommended_actions": ["Repeat Troponin"],
            "additional_information_needed": ["Serial ECG"],
            "priority_level": "Urgent",
        },
        "safety": {
            "safety_flags": ["Troponin normal"],
            "contraindications_or_concerns": [],
        },
        "uncertainty": {
            "confidence_score": 0.75,
            "uncertainty_factors": [],  # Violates uncertainty_required=True!
        },
        "traceability": [
            {
                "reasoning_item": "ACS",
                "supported_by_findings": ["Chest pain"],
                "upstream_sources": ["Agent 2"],
            }
        ],
        "agent_metadata": {
            "agent": "agent_5",
            "model": settings.primary_reasoning_model,
            "reasoning_mode": "primary",
            "status": "success",
        },
    }

    response = ClinicalDecisionSupportResponse.model_validate(mock_resp_dict)
    val_result = ClinicalOutputValidator.validate_response(response, conflicting_evidence_context, policy=policy)

    assert val_result.valid is False
    assert any(e.code == "MISSING_REQUIRED_UNCERTAINTY" for e in val_result.errors)
