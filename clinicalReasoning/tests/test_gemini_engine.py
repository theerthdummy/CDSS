"""Unit tests for GeminiReasoningEngine using client mocks."""

import json
from unittest.mock import MagicMock
import pytest

from app.config import settings
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
    GeminiReasoningEngine,
)


@pytest.fixture
def valid_context():
    """Returns a valid sample UnifiedClinicalContext instance."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute myocardial infarction requires emergency PCI",
                sources=["Agent 2", "Agent 3"],
            )
        ],
        normalized_medical_terms=["AMI", "PCI"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Primary PCI improves acute survival",
                source_attribution="PubMed",
            )
        ],
        conflicting_evidence=[
            ConflictEvidence(
                type="Treatment Conflict",
                finding="Thrombolytics",
                description="Bleeding risk caution",
            )
        ],
        evidence_priority=[
            EvidencePriority(
                finding="AMI protocol",
                priority_score=0.9,
                primary_source="Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="AMI protocol",
                sources=["Guidelines"],
                upstream_agents=["Agent 2"],
            )
        ],
        fusion_summary="AMI requiring immediate intervention.",
        confidence_score=0.9,
    )


@pytest.fixture
def valid_gemini_json_response_dict():
    """Returns a dictionary matching ClinicalDecisionSupportResponse schema."""
    return {
        "clinical_assessment": {
            "primary_interpretation": "Acute STEMI presentation.",
            "clinical_significance": "Critical",
            "differential_considerations": ["Acute STEMI", "Pericarditis"],
        },
        "reasoning": {
            "key_findings": ["Acute myocardial infarction requires emergency PCI"],
            "supporting_factors": ["Primary PCI improves acute survival"],
            "conflicting_factors": ["Bleeding risk caution"],
            "reasoning_summary": "Reperfusion therapy is urgent; bleeding risk must be screened.",
        },
        "decision_support": {
            "recommended_actions": ["Immediate PCI Lab Activation"],
            "additional_information_needed": ["Baseline Troponin"],
            "priority_level": "Urgent",
        },
        "safety": {
            "safety_flags": ["Bleeding risk warning"],
            "contraindications_or_concerns": ["Thrombolytics contraindicated if active bleeding"],
        },
        "uncertainty": {
            "confidence_score": 0.88,
            "uncertainty_factors": ["Pending baseline troponin values"],
        },
        "traceability": [
            {
                "reasoning_item": "Immediate PCI Lab Activation",
                "supported_by_findings": ["Acute myocardial infarction requires emergency PCI"],
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


def test_gemini_engine_missing_api_key(valid_context):
    """Verify that missing API key raises EngineUnavailableError."""
    engine = GeminiReasoningEngine(api_key="", model_name=settings.primary_reasoning_model)
    with pytest.raises(EngineUnavailableError) as exc_info:
        engine.reason(valid_context)
    assert "GEMINI_API_KEY environment variable is not configured" in str(exc_info.value)


def test_gemini_engine_successful_mocked_response(valid_context, valid_gemini_json_response_dict):
    """Verify that a successful mocked Gemini response returns validated ClinicalDecisionSupportResponse."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(valid_gemini_json_response_dict)
    mock_client.models.generate_content.return_value = mock_response

    engine = GeminiReasoningEngine(
        api_key="test_dummy_key",
        model_name=settings.primary_reasoning_model,
        client=mock_client,
    )

    response = engine.reason(valid_context)
    assert isinstance(response, ClinicalDecisionSupportResponse)
    assert response.agent_metadata.model == settings.primary_reasoning_model
    assert response.agent_metadata.reasoning_mode == "primary"
    assert response.clinical_assessment.clinical_significance == "Critical"
    mock_client.models.generate_content.assert_called_once()


def test_gemini_engine_markdown_wrapped_json(valid_context, valid_gemini_json_response_dict):
    """Verify that JSON text wrapped in markdown backticks is parsed correctly."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    json_raw = json.dumps(valid_gemini_json_response_dict)
    mock_response.text = f"```json\n{json_raw}\n```"
    mock_client.models.generate_content.return_value = mock_response

    engine = GeminiReasoningEngine(
        api_key="test_dummy_key",
        model_name=settings.primary_reasoning_model,
        client=mock_client,
    )

    response = engine.reason(valid_context)
    assert response.agent_metadata.status == "success"


def test_gemini_engine_api_exception(valid_context):
    """Verify that API exceptions raise EngineExecutionError."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Connection Timeout")

    engine = GeminiReasoningEngine(
        api_key="test_dummy_key",
        model_name=settings.primary_reasoning_model,
        client=mock_client,
    )

    with pytest.raises(EngineExecutionError) as exc_info:
        engine.reason(valid_context)
    assert "Gemini API request failed" in str(exc_info.value)


def test_gemini_engine_invalid_json_text(valid_context):
    """Verify that malformed JSON response text raises EngineExecutionError."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "NOT_VALID_JSON_STRING {"
    mock_client.models.generate_content.return_value = mock_response

    engine = GeminiReasoningEngine(
        api_key="test_dummy_key",
        model_name=settings.primary_reasoning_model,
        client=mock_client,
    )

    with pytest.raises(EngineExecutionError) as exc_info:
        engine.reason(valid_context)
    assert "Failed to parse Gemini response as JSON" in str(exc_info.value)


def test_gemini_engine_invalid_pydantic_schema(valid_context):
    """Verify that JSON missing required fields raises EngineExecutionError."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Missing clinical_assessment, reasoning, etc.
    mock_response.text = json.dumps({"incomplete": "dict"})
    mock_client.models.generate_content.return_value = mock_response

    engine = GeminiReasoningEngine(
        api_key="test_dummy_key",
        model_name=settings.primary_reasoning_model,
        client=mock_client,
    )

    with pytest.raises(EngineExecutionError) as exc_info:
        engine.reason(valid_context)
    assert "Gemini response failed Pydantic schema validation" in str(exc_info.value)
