"""End-to-end integration and contract hardening tests for Agent 5 pipeline.

Covers all 5 core CDSS pipeline scenarios and contract validation tests using realistic fixtures and mocked external agents.
"""

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
from app.orchestration.adaptive_orchestrator import AdaptiveOrchestrator
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.reasoning_engines import (
    DeterministicFallbackEngine,
    EngineExecutionError,
    GeminiReasoningEngine,
    GroqReasoningEngine,
    ReasoningEngine,
)


@pytest.fixture
def high_confidence_context():
    """Realistic high-confidence UnifiedClinicalContext model fixture (0.92)."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute ST-elevation myocardial infarction (STEMI) requires emergency reperfusion therapy",
                sources=["Agent 2 Biomedical RAG", "Agent 3 PubMed"],
            )
        ],
        normalized_medical_terms=["STEMI", "Reperfusion Therapy", "Aspirin"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Primary PCI within 90 minutes reduces acute 30-day mortality in STEMI",
                source_attribution="Agent 3 Clinical Guidelines",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[
            EvidencePriority(
                finding="Acute STEMI emergency reperfusion protocol",
                priority_score=0.95,
                primary_source="Agent 3 Clinical Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="Acute STEMI emergency reperfusion protocol",
                sources=["PubMed", "Clinical Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Patient presentation strongly consistent with acute STEMI requiring emergency reperfusion.",
        confidence_score=0.92,
    )


@pytest.fixture
def low_confidence_context():
    """Realistic low-confidence UnifiedClinicalContext model fixture (0.45)."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Atypical chest pain with non-specific ECG ST-T wave changes",
                sources=["Agent 2 Biomedical RAG"],
            )
        ],
        normalized_medical_terms=["Chest Pain", "ECG"],
        supporting_evidence=[],
        conflicting_evidence=[
            ConflictEvidence(
                type="Diagnostic Conflict",
                finding="Ischemia vs Musculoskeletal Pain",
                description="Equivocal ECG changes with normal initial cardiac troponin I.",
            )
        ],
        evidence_priority=[],
        source_traceability=[
            SourceTraceability(
                finding="Chest Pain",
                sources=["EHR"],
                upstream_agents=["Agent 2"],
            )
        ],
        fusion_summary="Equivocal chest pain presentation with low evidence confidence.",
        confidence_score=0.45,
    )


class FailingMockEngine(ReasoningEngine):
    """Mock reasoning engine that always raises EngineExecutionError."""

    def reason(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        raise EngineExecutionError("Simulated LLM engine failure.")


# -----------------------------------------------------------------------------
# SCENARIO A: HIGH CONFIDENCE CONTEXT -> DIRECT REASONING VIA GEMINI
# -----------------------------------------------------------------------------

def test_scenario_a_high_confidence_direct_gemini_reasoning(high_confidence_context):
    """SCENARIO A: Agent 4 confidence = 0.92. Agent 5 approves context. Reasoning via Gemini succeeds. Agent 2/3/4/Groq NOT called."""
    mock_http = MagicMock()
    mock_gemini_client = MagicMock()
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = '{"clinical_assessment": {"primary_interpretation": "Acute STEMI confirmed.", "clinical_significance": "Critical", "differential_considerations": ["Acute STEMI"]}, "reasoning": {"key_findings": ["STEMI"], "supporting_factors": ["Primary PCI"], "conflicting_factors": [], "reasoning_summary": "Primary PCI indicated urgently."}, "decision_support": {"recommended_actions": ["Activate Cath Lab"], "additional_information_needed": [], "priority_level": "Urgent"}, "safety": {"safety_flags": [], "contraindications_or_concerns": []}, "uncertainty": {"confidence_score": 0.92, "uncertainty_factors": []}, "traceability": [], "agent_metadata": {"agent": "agent_5", "model": "gemini-2.5-flash", "reasoning_mode": "primary", "status": "success"}}'
    mock_gemini_client.models.generate_content.return_value = mock_gemini_resp

    primary = GeminiReasoningEngine(api_key="test_key", client=mock_gemini_client)
    reasoning_service = ClinicalReasoningService(primary_engine=primary, use_dev_default_when_unimplemented=False)
    orchestrator = AdaptiveOrchestrator(reasoning_service=reasoning_service, max_retries=3, http_client=mock_http)

    result = orchestrator.run(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.primary_reasoning_model
    assert result.agent_metadata.reasoning_mode == "primary"
    assert result.clinical_assessment.clinical_significance == "Critical"
    mock_http.post.assert_not_called()


# -----------------------------------------------------------------------------
# SCENARIO B: LOW CONFIDENCE -> RETRIEVAL (AGENT 2 + 3) -> RE-FUSION -> GEMINI
# -----------------------------------------------------------------------------

def test_scenario_b_low_confidence_retrieval_refusion_gemini_success(low_confidence_context, high_confidence_context):
    """SCENARIO B: Initial confidence 0.45 -> Feedback requests Agent 2+3 -> Re-fusion confidence 0.92 -> Gemini succeeds in 1 adaptive cycle."""
    mock_http = MagicMock()

    resp_agent2 = MagicMock(status_code=200, json=lambda: {"biomedical_findings": ["Troponin I serial rise detected"]})
    resp_agent3 = MagicMock(status_code=200, json=lambda: {"clinical_evidence": ["Guidelines recommend serial cardiac biomarkers"]})
    resp_agent4 = MagicMock(status_code=200, json=lambda: high_confidence_context.model_dump())

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8003" in url:
            return resp_agent3
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected URL: {url}")

    mock_http.post.side_effect = mock_post

    mock_gemini_client = MagicMock()
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = '{"clinical_assessment": {"primary_interpretation": "NSTEMI confirmed after re-fusion.", "clinical_significance": "High", "differential_considerations": []}, "reasoning": {"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""}, "decision_support": {"recommended_actions": [], "additional_information_needed": [], "priority_level": "Urgent"}, "safety": {"safety_flags": [], "contraindications_or_concerns": []}, "uncertainty": {"confidence_score": 0.92, "uncertainty_factors": []}, "traceability": [], "agent_metadata": {"agent": "agent_5", "model": "gemini-2.5-flash", "reasoning_mode": "primary", "status": "success"}}'
    mock_gemini_client.models.generate_content.return_value = mock_gemini_resp

    primary = GeminiReasoningEngine(api_key="test_key", client=mock_gemini_client)
    reasoning_service = ClinicalReasoningService(primary_engine=primary, use_dev_default_when_unimplemented=False)
    orchestrator = AdaptiveOrchestrator(reasoning_service=reasoning_service, max_retries=3, http_client=mock_http)

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.primary_reasoning_model
    # Verify Agent 2, Agent 3, and Agent 4 re-fusion were invoked exactly once
    assert mock_http.post.call_count == 3


# -----------------------------------------------------------------------------
# SCENARIO C: PERSISTENT LOW CONFIDENCE -> SAFE TERMINATION AT MAX RETRIES
# -----------------------------------------------------------------------------

def test_scenario_c_persistent_low_confidence_safe_termination(low_confidence_context):
    """SCENARIO C: Persistent low confidence across retries. Loop halts safely at MAX_RETRIES (3). No LLMs or extra retrieval calls."""
    mock_http = MagicMock()

    resp_agent2 = MagicMock(status_code=200, json=lambda: {"biomedical_findings": []})
    resp_agent4 = MagicMock(status_code=200, json=lambda: low_confidence_context.model_dump())

    mock_http.post.side_effect = lambda url, json=None, **kwargs: resp_agent2 if "8002" in url else resp_agent4

    mock_gemini = MagicMock(spec=ReasoningEngine)
    mock_groq = MagicMock(spec=ReasoningEngine)
    mock_deterministic = MagicMock(spec=ReasoningEngine)

    reasoning_service = ClinicalReasoningService(
        primary_engine=mock_gemini,
        fallback_engine=mock_groq,
        deterministic_engine=mock_deterministic,
        use_dev_default_when_unimplemented=False,
    )
    orchestrator = AdaptiveOrchestrator(reasoning_service=reasoning_service, max_retries=3, http_client=mock_http)

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, InsufficientEvidenceTermination)
    assert result.status == "insufficient_evidence"
    assert result.retry_count == 3
    assert result.max_retries == 3

    # Assert NO reasoning models were invoked during low-confidence feedback iterations
    mock_gemini.reason.assert_not_called()
    mock_groq.reason.assert_not_called()
    mock_deterministic.reason.assert_not_called()


# -----------------------------------------------------------------------------
# SCENARIO D: READY FOR REASONING -> GEMINI FAILS -> GROQ SUCCESS
# -----------------------------------------------------------------------------

def test_scenario_d_ready_context_gemini_fails_groq_succeeds(high_confidence_context):
    """SCENARIO D: Context ready for reasoning. Gemini API fails -> Groq fallback succeeds. No adaptive retrieval calls made."""
    mock_http = MagicMock()

    primary = FailingMockEngine()

    mock_groq_http = MagicMock()
    mock_groq_resp = MagicMock(status_code=200)
    mock_groq_resp.json.return_value = {
        "choices": [{"message": {"content": '{"clinical_assessment": {"primary_interpretation": "Groq Llama STEMI Reasoning", "clinical_significance": "High", "differential_considerations": []}, "reasoning": {"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""}, "decision_support": {"recommended_actions": [], "additional_information_needed": [], "priority_level": "Urgent"}, "safety": {"safety_flags": [], "contraindications_or_concerns": []}, "uncertainty": {"confidence_score": 0.88, "uncertainty_factors": []}, "traceability": [], "agent_metadata": {"agent": "agent_5", "model": "llama-3.1-8b-instruct", "reasoning_mode": "fallback_model", "status": "success"}}'}}]
    }
    mock_groq_http.post.return_value = mock_groq_resp

    fallback = GroqReasoningEngine(api_key="test_groq_key", http_client=mock_groq_http)
    reasoning_service = ClinicalReasoningService(primary_engine=primary, fallback_engine=fallback, use_dev_default_when_unimplemented=False)
    orchestrator = AdaptiveOrchestrator(reasoning_service=reasoning_service, max_retries=3, http_client=mock_http)

    result = orchestrator.run(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "llama-3.1-8b-instruct"
    assert result.agent_metadata.reasoning_mode == "fallback_model"
    mock_http.post.assert_not_called()


# -----------------------------------------------------------------------------
# SCENARIO E: READY FOR REASONING -> GEMINI + GROQ FAIL -> DETERMINISTIC FALLBACK
# -----------------------------------------------------------------------------

def test_scenario_e_ready_context_gemini_and_groq_fail_deterministic_fallback(high_confidence_context):
    """SCENARIO E: Context ready for reasoning. Gemini & Groq fail -> Deterministic Fallback Engine succeeds. No adaptive retrieval calls made."""
    mock_http = MagicMock()

    primary = FailingMockEngine()
    fallback = FailingMockEngine()
    deterministic = DeterministicFallbackEngine()

    reasoning_service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        deterministic_engine=deterministic,
        use_dev_default_when_unimplemented=False,
    )
    orchestrator = AdaptiveOrchestrator(reasoning_service=reasoning_service, max_retries=3, http_client=mock_http)

    result = orchestrator.run(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "deterministic-fallback"
    assert result.agent_metadata.reasoning_mode == "deterministic_fallback"
    assert result.agent_metadata.status == "fallback_applied"
    mock_http.post.assert_not_called()


# -----------------------------------------------------------------------------
# CONTRACT HARDENING & SCHEMA COMPATIBILITY TESTS
# -----------------------------------------------------------------------------

def test_contract_agent4_dict_to_agent5_unified_context_schema():
    """Verify that an Agent 4 JSON dict payload parses 100% cleanly into Agent 5 UnifiedClinicalContext."""
    raw_agent4_output = {
        "merged_findings": [
            {"finding": "Acute myocardial infarction", "sources": ["Agent 2", "Agent 3"]}
        ],
        "normalized_medical_terms": ["AMI", "PCI"],
        "supporting_evidence": [
            {"finding": "Primary PCI within 90 mins", "source_attribution": "PubMed"}
        ],
        "conflicting_evidence": [
            {"type": "Treatment Conflict", "finding": "Thrombolysis", "description": "Bleeding risk caution"}
        ],
        "evidence_priority": [
            {"finding": "AMI protocol", "priority_score": 0.95, "primary_source": "Guidelines"}
        ],
        "source_traceability": [
            {"finding": "AMI protocol", "sources": ["Guidelines"], "upstream_agents": ["Agent 2", "Agent 3"]}
        ],
        "fusion_summary": "AMI requiring PCI.",
        "confidence_score": 0.91,
    }

    parsed_context = UnifiedClinicalContext.model_validate(raw_agent4_output)
    assert parsed_context.confidence_score == 0.91
    assert parsed_context.merged_findings[0].finding == "Acute myocardial infarction"
    assert parsed_context.conflicting_evidence[0].type == "Treatment Conflict"


def test_contract_feedback_request_to_upstream_requirements():
    """Verify that FeedbackRequest produces valid Agent 2 and Agent 3 retrieval payloads."""
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.50,
        reason="conflicting_evidence",
        evidence_gaps=["Unresolved treatment conflict regarding thrombolysis"],
        requested_sources=["agent_2", "agent_3"],
        agent_2_request={
            "query_requirements": ["Search graph for thrombolytic contraindications"],
            "focus_terms": ["Thrombolysis", "Bleeding"],
            "knowledge_graph_areas": ["Drug Contraindications"],
        },
        agent_3_request={
            "evidence_requirements": ["Search PubMed for thrombolysis guidelines"],
            "publication_recency": "Last 5 years",
            "conflict_verification_requirements": ["GI bleeding risk"],
        },
    )

    assert fb_req.agent_2_request.focus_terms == ["Thrombolysis", "Bleeding"]
    assert fb_req.agent_3_request.publication_recency == "Last 5 years"
