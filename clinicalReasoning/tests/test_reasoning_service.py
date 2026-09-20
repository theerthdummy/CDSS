"""Unit tests for ClinicalReasoningService, AdaptiveOptimizer integration, and LLM fallback cascade."""

import json
from typing import Optional
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
from app.models.output_models import ClinicalDecisionSupportResponse, ReasoningPolicy
from app.services.adaptive_optimizer import AdaptiveOptimizer
from app.services.clinical_reasoning import ClinicalReasoningError, ClinicalReasoningService
from app.services.reasoning_engines import (
    DeterministicFallbackEngine,
    EngineExecutionError,
    GeminiReasoningEngine,
    GroqReasoningEngine,
    OllamaReasoningEngine,
    ReasoningEngine,
)


@pytest.fixture
def high_confidence_context():
    """Returns a valid high confidence UnifiedClinicalContext model instance."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute STEMI requires immediate emergency reperfusion therapy",
                sources=["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            )
        ],
        normalized_medical_terms=["STEMI", "Reperfusion Therapy"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Early primary PCI within 90 minutes reduces mortality in STEMI",
                source_attribution="Agent 3 Clinical Guidelines",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[
            EvidencePriority(
                finding="Acute STEMI reperfusion protocol",
                priority_score=0.95,
                primary_source="Agent 3 Clinical Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="Acute STEMI reperfusion protocol",
                sources=["PubMed", "Clinical Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Patient presentation strongly consistent with acute STEMI requiring emergency reperfusion.",
        confidence_score=0.92,
    )


@pytest.fixture
def low_confidence_context():
    """Returns a valid low confidence UnifiedClinicalContext model instance."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(finding="Atypical chest pain", sources=["Agent 2"])
        ],
        normalized_medical_terms=["Chest Pain"],
        supporting_evidence=[],
        conflicting_evidence=[],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Equivocal chest pain presentation.",
        confidence_score=0.45,
    )


class FailingMockEngine(ReasoningEngine):
    """Mock engine that always raises EngineExecutionError."""

    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        raise EngineExecutionError("Simulated mock engine failure.")


def test_case_1_optimizer_needs_more_evidence_bypasses_all_reasoning_models(low_confidence_context):
    """CASE 1: Optimizer returns NEEDS_MORE_EVIDENCE. Gemini, Groq, and Deterministic engines MUST NOT be called."""
    mock_gemini = MagicMock(spec=ReasoningEngine)
    mock_groq = MagicMock(spec=ReasoningEngine)
    mock_deterministic = MagicMock(spec=ReasoningEngine)

    service = ClinicalReasoningService(
        primary_engine=mock_gemini,
        fallback_engine=mock_groq,
        deterministic_engine=mock_deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(low_confidence_context, retry_count=0)

    assert isinstance(result, FeedbackRequest)
    assert result.status == "needs_more_evidence"
    assert result.retry_count == 0

    # Assert NO reasoning engine was invoked
    mock_gemini.reason.assert_not_called()
    mock_groq.reason.assert_not_called()
    mock_deterministic.reason.assert_not_called()


def test_case_2_optimizer_max_retries_insufficient_evidence_bypasses_all_models(low_confidence_context):
    """CASE 2: Optimizer returns INSUFFICIENT_EVIDENCE (retry_count >= max_retries). Engines MUST NOT be called."""
    mock_gemini = MagicMock(spec=ReasoningEngine)
    mock_groq = MagicMock(spec=ReasoningEngine)
    mock_deterministic = MagicMock(spec=ReasoningEngine)

    service = ClinicalReasoningService(
        primary_engine=mock_gemini,
        fallback_engine=mock_groq,
        deterministic_engine=mock_deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(low_confidence_context, retry_count=3)

    assert isinstance(result, InsufficientEvidenceTermination)
    assert result.status == "insufficient_evidence"
    assert result.retry_count == 3

    # Assert NO reasoning engine was invoked
    mock_gemini.reason.assert_not_called()
    mock_groq.reason.assert_not_called()
    mock_deterministic.reason.assert_not_called()


def test_case_3_optimizer_ready_calls_gemini(high_confidence_context):
    """CASE 3: Optimizer returns READY_FOR_REASONING. Primary engine (Gemini) is called."""
    mock_gemini_client = MagicMock()
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = '{"clinical_assessment": {"primary_interpretation": "Gemini STEMI Assessment", "clinical_significance": "High", "differential_considerations": []}, "reasoning": {"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""}, "decision_support": {"recommended_actions": [], "additional_information_needed": [], "priority_level": "Urgent"}, "safety": {"safety_flags": [], "contraindications_or_concerns": []}, "uncertainty": {"confidence_score": 0.9, "uncertainty_factors": []}, "traceability": [], "agent_metadata": {"agent": "agent_5", "model": "gemini-2.5-flash", "reasoning_mode": "primary", "status": "success"}}'
    mock_gemini_client.models.generate_content.return_value = mock_gemini_resp

    mock_groq_http_client = MagicMock()

    primary = GeminiReasoningEngine(api_key="test_key", client=mock_gemini_client)
    fallback = GroqReasoningEngine(api_key="test_groq_key", http_client=mock_groq_http_client)

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context, retry_count=0)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.primary_reasoning_model
    assert result.agent_metadata.reasoning_mode == "primary"
    mock_groq_http_client.post.assert_not_called()


def test_case_4_optimizer_ready_gemini_fails_calls_groq(high_confidence_context):
    """CASE 4: Optimizer returns READY_FOR_REASONING, primary fails, Groq fallback is called."""
    primary = FailingMockEngine()

    groq_content = json.dumps({
        "clinical_assessment": {
            "primary_interpretation": "Groq Llama Fallback STEMI Assessment",
            "clinical_significance": "High",
            "differential_considerations": [],
        },
        "reasoning": {
            "key_findings": [],
            "supporting_factors": [],
            "conflicting_factors": [],
            "reasoning_summary": "",
        },
        "decision_support": {
            "recommended_actions": [],
            "additional_information_needed": [],
            "priority_level": "Urgent",
        },
        "safety": {
            "safety_flags": [],
            "contraindications_or_concerns": [],
        },
        "uncertainty": {
            "confidence_score": 0.85,
            "uncertainty_factors": [],
        },
        "traceability": [],
        "agent_metadata": {
            "agent": "agent_5",
            "model": settings.groq_model,
            "reasoning_mode": "fallback_model",
            "status": "success",
        },
    })

    mock_groq_http_client = MagicMock()
    mock_groq_resp = MagicMock()
    mock_groq_resp.status_code = 200
    mock_groq_resp.json.return_value = {
        "choices": [{"message": {"content": groq_content}}]
    }
    mock_groq_http_client.post.return_value = mock_groq_resp

    fallback = GroqReasoningEngine(api_key="test_groq_key", http_client=mock_groq_http_client)

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context, retry_count=0)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.groq_model
    assert result.agent_metadata.reasoning_mode == "fallback_model"
    mock_groq_http_client.post.assert_called_once()


def test_case_5_optimizer_ready_gemini_and_groq_fail_calls_deterministic(high_confidence_context):
    """CASE 5: Optimizer returns READY_FOR_REASONING, Gemini and Groq fail, Deterministic engine is called."""
    primary = FailingMockEngine()
    fallback = FailingMockEngine()
    deterministic = DeterministicFallbackEngine()

    service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )

    result = service.execute_reasoning_pipeline(high_confidence_context, retry_count=0)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "deterministic-fallback"
    assert result.agent_metadata.reasoning_mode == "deterministic_fallback"


def test_case_6_adaptive_loop_simulation(low_confidence_context, high_confidence_context):
    """CASE 6: Simulate multi-attempt adaptive feedback loop (Attempt 1: low confidence -> FeedbackRequest; Attempt 2: re-fused context -> Gemini called)."""
    mock_gemini_client = MagicMock()
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = '{"clinical_assessment": {"primary_interpretation": "Re-fused Context Reasoning", "clinical_significance": "High", "differential_considerations": []}, "reasoning": {"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""}, "decision_support": {"recommended_actions": [], "additional_information_needed": [], "priority_level": "Urgent"}, "safety": {"safety_flags": [], "contraindications_or_concerns": []}, "uncertainty": {"confidence_score": 0.92, "uncertainty_factors": []}, "traceability": [], "agent_metadata": {"agent": "agent_5", "model": "gemini-2.5-flash", "reasoning_mode": "primary", "status": "success"}}'
    mock_gemini_client.models.generate_content.return_value = mock_gemini_resp

    primary = GeminiReasoningEngine(api_key="test_key", client=mock_gemini_client)
    service = ClinicalReasoningService(
        primary_engine=primary,
        use_dev_default_when_unimplemented=False,
    )

    # Attempt 1: Context is low confidence (0.45), retry_count = 0
    res1 = service.execute_reasoning_pipeline(low_confidence_context, retry_count=0)
    assert isinstance(res1, FeedbackRequest)
    assert res1.status == "needs_more_evidence"

    # Attempt 2: Re-fused context with improved confidence (0.92), retry_count = 1
    res2 = service.execute_reasoning_pipeline(high_confidence_context, retry_count=1)
    assert isinstance(res2, ClinicalDecisionSupportResponse)
    assert res2.agent_metadata.model == settings.primary_reasoning_model


def test_default_service_uses_groq_primary_and_ollama_fallback(monkeypatch):
    """Verify the default Agent 5 service wiring prefers Groq first and Ollama second."""
    monkeypatch.setattr("app.services.clinical_reasoning.settings.primary_reasoning_provider", "groq")
    monkeypatch.setattr("app.services.clinical_reasoning.settings.fallback_reasoning_provider", "ollama")

    service = ClinicalReasoningService(use_dev_default_when_unimplemented=False)

    assert isinstance(service.primary_engine, GroqReasoningEngine)
    assert isinstance(service.fallback_engine, OllamaReasoningEngine)
