"""Unit tests for ClinicalOutputValidator and output validation fallback integration."""

import copy
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from app.config import settings
from app.models.feedback_models import FeedbackRequest, InsufficientEvidenceTermination
from app.models.input_models import (
    ConflictEvidence,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import ClinicalDecisionSupportResponse
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.output_validator import ClinicalOutputValidator
from app.services.reasoning_engines import (
    DeterministicFallbackEngine,
    ReasoningEngine,
)


@pytest.fixture
def high_confidence_context():
    """Valid high-confidence context fixture."""
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
                finding="Primary PCI within 90 minutes reduces mortality",
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
        fusion_summary="Acute STEMI requiring emergency PCI.",
        confidence_score=0.92,
    )


@pytest.fixture
def conflicting_context():
    """Context fixture with unmitigated conflicting evidence."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Atypical chest pain with non-specific ECG changes",
                sources=["Agent 2 Biomedical RAG"],
            )
        ],
        normalized_medical_terms=["Chest Pain"],
        supporting_evidence=[],
        conflicting_evidence=[
            ConflictEvidence(
                type="Diagnostic Conflict",
                finding="Ischemia vs Musculoskeletal Pain",
                description="Equivocal ECG changes with normal initial cardiac biomarkers.",
            )
        ],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Equivocal chest pain presentation.",
        confidence_score=0.75,
    )


@pytest.fixture
def valid_response_payload():
    """Valid ClinicalDecisionSupportResponse dictionary."""
    return {
        "clinical_assessment": {
            "primary_interpretation": "Acute STEMI requiring primary PCI reperfusion therapy.",
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


def test_valid_final_response_validation_passes(high_confidence_context, valid_response_payload):
    """Verify that a valid response matching context passes ClinicalOutputValidator."""
    response = ClinicalDecisionSupportResponse.model_validate(valid_response_payload)
    val_result = ClinicalOutputValidator.validate_response(response, high_confidence_context)

    assert val_result.valid is True
    assert len(val_result.errors) == 0


def test_invalid_confidence_score_out_of_range(high_confidence_context, valid_response_payload):
    """Verify validator rejects confidence_score out of [0.0, 1.0] range."""
    payload = copy.deepcopy(valid_response_payload)
    payload["uncertainty"]["confidence_score"] = 1.5
    response = ClinicalDecisionSupportResponse.model_validate(payload)
    val_result = ClinicalOutputValidator.validate_response(response, high_confidence_context)
    assert val_result.valid is False
    assert any(e.code == "INVALID_CONFIDENCE" for e in val_result.errors)


def test_overconfident_output_with_conflicting_evidence(conflicting_context, valid_response_payload):
    """Verify validator flags confidence score > 0.90 when unmitigated conflicting evidence exists."""
    payload = copy.deepcopy(valid_response_payload)
    payload["uncertainty"]["confidence_score"] = 0.95
    response = ClinicalDecisionSupportResponse.model_validate(payload)
    val_result = ClinicalOutputValidator.validate_response(response, conflicting_context)

    assert val_result.valid is False
    assert any(e.code == "OVERCONFIDENT_OUTPUT" for e in val_result.errors)


def test_unjustified_confirmed_diagnosis_with_conflicting_evidence(conflicting_context, valid_response_payload):
    """Verify validator flags 'confirmed diagnosis' claims under conflicting evidence."""
    payload = copy.deepcopy(valid_response_payload)
    payload["clinical_assessment"]["primary_interpretation"] = "Confirmed diagnosis of Acute Myocardial Infarction."
    payload["uncertainty"]["confidence_score"] = 0.80
    response = ClinicalDecisionSupportResponse.model_validate(payload)
    val_result = ClinicalOutputValidator.validate_response(response, conflicting_context)

    assert val_result.valid is False
    assert any(e.code == "UNJUSTIFIED_CONFIRMED_DIAGNOSIS" for e in val_result.errors)


def test_unsupported_invented_source_identifier(high_confidence_context, valid_response_payload):
    """Verify validator flags invented source identifiers like EVID-999 or Agent 9."""
    payload = copy.deepcopy(valid_response_payload)
    payload["traceability"][0]["upstream_sources"] = ["EVID-999", "Agent 9"]
    response = ClinicalDecisionSupportResponse.model_validate(payload)
    val_result = ClinicalOutputValidator.validate_response(response, high_confidence_context)

    assert val_result.valid is False
    assert any(e.code == "UNSUPPORTED_SOURCE" for e in val_result.errors)


def test_gemini_invalid_output_validation_triggers_groq(high_confidence_context, valid_response_payload):
    """Gemini returns output with invented source EVID-999 -> fails validation -> triggers Groq fallback."""
    invalid_gemini_payload = copy.deepcopy(valid_response_payload)
    invalid_gemini_payload["traceability"][0]["upstream_sources"] = ["EVID-999"]
    invalid_gemini_resp = ClinicalDecisionSupportResponse.model_validate(invalid_gemini_payload)

    valid_groq_payload = copy.deepcopy(valid_response_payload)
    valid_groq_payload["agent_metadata"]["model"] = "llama-3.1-8b-instruct"
    valid_groq_payload["agent_metadata"]["reasoning_mode"] = "fallback_model"
    valid_groq_resp = ClinicalDecisionSupportResponse.model_validate(valid_groq_payload)

    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.return_value = invalid_gemini_resp

    fallback = MagicMock(spec=ReasoningEngine)
    fallback.reason.return_value = valid_groq_resp

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "llama-3.1-8b-instruct"
    assert result.agent_metadata.reasoning_mode == "fallback_model"


def test_gemini_and_groq_invalid_output_triggers_deterministic_fallback(high_confidence_context, valid_response_payload):
    """Gemini & Groq return output with invented sources -> both fail validation -> Deterministic Fallback used."""
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

    result = service.execute_reasoning(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "deterministic-fallback"
    assert result.agent_metadata.reasoning_mode == "deterministic_fallback"


def test_confidence_breakdown_populated_in_response(high_confidence_context, valid_response_payload):
    """Verify fusion_confidence and reasoning_confidence are populated in UncertaintyAssessment."""
    response = ClinicalDecisionSupportResponse.model_validate(valid_response_payload)
    primary = MagicMock(spec=ReasoningEngine)
    primary.reason.return_value = response

    service = ClinicalReasoningService(primary_engine=primary, use_dev_default_when_unimplemented=False)
    result = service.execute_reasoning(high_confidence_context)

    assert result.uncertainty.fusion_confidence == 0.92
    assert result.uncertainty.reasoning_confidence == 0.92
