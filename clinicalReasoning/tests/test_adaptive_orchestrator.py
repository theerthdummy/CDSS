"""Unit tests for AdaptiveOrchestrator feedback loop execution."""

from typing import Optional
from unittest.mock import MagicMock
import httpx
import pytest

from app.config import settings
from app.models.feedback_models import (
    Agent2RetrievalRequirement,
    Agent3RetrievalRequirement,
    FeedbackRequest,
    InsufficientEvidenceTermination,
)
from app.models.input_models import (
    ConflictEvidence,
    EvidencePriority,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.models.output_models import ClinicalDecisionSupportResponse, ReasoningPolicy
from app.orchestration.adaptive_orchestrator import (
    AdaptiveOrchestrator,
    OrchestrationFailureError,
)
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.reasoning_engines import (
    EngineExecutionError,
    GeminiReasoningEngine,
    GroqReasoningEngine,
    ReasoningEngine,
)


@pytest.fixture
def high_confidence_context():
    """Valid high-confidence UnifiedClinicalContext model (0.92)."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Acute STEMI requires immediate emergency reperfusion therapy",
                sources=["Agent 2", "Agent 3"],
            )
        ],
        normalized_medical_terms=["STEMI", "Reperfusion Therapy"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Primary PCI within 90 minutes reduces mortality",
                source_attribution="Agent 3 Guidelines",
            )
        ],
        conflicting_evidence=[],
        evidence_priority=[
            EvidencePriority(
                finding="STEMI reperfusion protocol",
                priority_score=0.95,
                primary_source="Guidelines",
            )
        ],
        source_traceability=[
            SourceTraceability(
                finding="STEMI reperfusion protocol",
                sources=["Guidelines"],
                upstream_agents=["Agent 2", "Agent 3"],
            )
        ],
        fusion_summary="Acute STEMI presentation requiring immediate PCI.",
        confidence_score=0.92,
    )


@pytest.fixture
def low_confidence_context():
    """Valid low-confidence UnifiedClinicalContext model (0.45)."""
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


def test_orchestrator_scenario_1_initial_context_ready(high_confidence_context):
    """SCENARIO 1: Initial context is ready. Clinical reasoning proceeds, no Agent 2/3/4 calls."""
    mock_http_client = MagicMock()
    mock_gemini_client = MagicMock()
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = f'{{"clinical_assessment": {{"primary_interpretation": "STEMI Assessment", "clinical_significance": "High", "differential_considerations": []}}, "reasoning": {{"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""}}, "decision_support": {{"recommended_actions": [], "additional_information_needed": [], "priority_level": "Urgent"}}, "safety": {{"safety_flags": [], "contraindications_or_concerns": []}}, "uncertainty": {{"confidence_score": 0.9, "uncertainty_factors": []}}, "traceability": [], "agent_metadata": {{"agent": "agent_5", "model": "{settings.primary_reasoning_model}", "reasoning_mode": "primary", "status": "success"}}}}'
    mock_gemini_client.models.generate_content.return_value = mock_gemini_resp

    primary = GeminiReasoningEngine(api_key="test_key", client=mock_gemini_client)
    reasoning_service = ClinicalReasoningService(primary_engine=primary, use_dev_default_when_unimplemented=False)

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=reasoning_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.primary_reasoning_model
    mock_http_client.post.assert_not_called()


def test_orchestrator_scenario_2_feedback_requests_agent_2_only(low_confidence_context, high_confidence_context):
    """SCENARIO 2: Feedback requests Agent 2 only. Agent 2 called, Agent 3 NOT called, Agent 4 re-fuses context."""
    mock_http_client = MagicMock()

    # Agent 2 response
    resp_agent2 = MagicMock()
    resp_agent2.status_code = 200
    resp_agent2.json.return_value = {"biomedical_findings": ["Troponin I negative"]}

    # Agent 4 response returning updated high confidence context
    resp_agent4 = MagicMock()
    resp_agent4.status_code = 200
    resp_agent4.json.return_value = high_confidence_context.model_dump()

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    # Iteration 0: return FeedbackRequest requesting agent_2
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="low_confidence",
        evidence_gaps=["Missing biomedical findings"],
        requested_sources=["agent_2"],
        agent_2_request=Agent2RetrievalRequirement(focus_terms=["Chest Pain"]),
        agent_3_request=None,
    )
    # Iteration 1: return reasoning success
    reasoning_resp = ClinicalDecisionSupportResponse(
        clinical_assessment={"primary_interpretation": "Success", "clinical_significance": "High", "differential_considerations": []},
        reasoning={"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""},
        decision_support={"recommended_actions": [], "additional_information_needed": [], "priority_level": "Routine"},
        safety={"safety_flags": [], "contraindications_or_concerns": []},
        uncertainty={"confidence_score": 0.92, "uncertainty_factors": []},
        traceability=[],
        agent_metadata={"agent": "agent_5", "model": settings.primary_reasoning_model, "reasoning_mode": "primary", "status": "success"},
    )
    mock_service.execute_reasoning_pipeline.side_effect = [fb_req, reasoning_resp]

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == settings.primary_reasoning_model
    assert mock_http_client.post.call_count == 2  # Agent 2 + Agent 4


def test_orchestrator_scenario_3_feedback_requests_agent_3_only(low_confidence_context, high_confidence_context):
    """SCENARIO 3: Feedback requests Agent 3 only. Agent 3 called, Agent 2 NOT called, Agent 4 re-fuses context."""
    mock_http_client = MagicMock()

    resp_agent3 = MagicMock()
    resp_agent3.status_code = 200
    resp_agent3.json.return_value = {"clinical_evidence": ["Guidelines suggest observation"]}

    resp_agent4 = MagicMock()
    resp_agent4.status_code = 200
    resp_agent4.json.return_value = high_confidence_context.model_dump()

    def mock_post(url, json=None, **kwargs):
        if "8003" in url:
            return resp_agent3
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="low_confidence",
        evidence_gaps=["Missing literature guidelines"],
        requested_sources=["agent_3"],
        agent_2_request=None,
        agent_3_request=Agent3RetrievalRequirement(evidence_requirements=["Search Chest Pain"]),
    )
    reasoning_resp = ClinicalDecisionSupportResponse(
        clinical_assessment={"primary_interpretation": "Success", "clinical_significance": "High", "differential_considerations": []},
        reasoning={"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""},
        decision_support={"recommended_actions": [], "additional_information_needed": [], "priority_level": "Routine"},
        safety={"safety_flags": [], "contraindications_or_concerns": []},
        uncertainty={"confidence_score": 0.92, "uncertainty_factors": []},
        traceability=[],
        agent_metadata={"agent": "agent_5", "model": settings.primary_reasoning_model, "reasoning_mode": "primary", "status": "success"},
    )
    mock_service.execute_reasoning_pipeline.side_effect = [fb_req, reasoning_resp]

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert mock_http_client.post.call_count == 2  # Agent 3 + Agent 4


def test_orchestrator_scenario_4_feedback_requests_agent_2_and_3(low_confidence_context, high_confidence_context):
    """SCENARIO 4: Feedback requests Agent 2 AND Agent 3. Both called, Agent 4 receives both outputs."""
    mock_http_client = MagicMock()

    resp_agent2 = MagicMock(status_code=200)
    resp_agent2.json.return_value = {"biomedical_findings": ["Finding 1"]}
    resp_agent3 = MagicMock(status_code=200)
    resp_agent3.json.return_value = {"clinical_evidence": ["Evidence 1"]}
    resp_agent4 = MagicMock(status_code=200)
    resp_agent4.json.return_value = high_confidence_context.model_dump()

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8003" in url:
            return resp_agent3
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="missing_information",
        evidence_gaps=["Gap 1"],
        requested_sources=["agent_2", "agent_3"],
        agent_2_request=Agent2RetrievalRequirement(focus_terms=["Chest Pain"]),
        agent_3_request=Agent3RetrievalRequirement(evidence_requirements=["Search 1"]),
    )
    reasoning_resp = ClinicalDecisionSupportResponse(
        clinical_assessment={"primary_interpretation": "Success", "clinical_significance": "High", "differential_considerations": []},
        reasoning={"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""},
        decision_support={"recommended_actions": [], "additional_information_needed": [], "priority_level": "Routine"},
        safety={"safety_flags": [], "contraindications_or_concerns": []},
        uncertainty={"confidence_score": 0.92, "uncertainty_factors": []},
        traceability=[],
        agent_metadata={"agent": "agent_5", "model": settings.primary_reasoning_model, "reasoning_mode": "primary", "status": "success"},
    )
    mock_service.execute_reasoning_pipeline.side_effect = [fb_req, reasoning_resp]

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert mock_http_client.post.call_count == 3  # Agent 2 + Agent 3 + Agent 4


def test_orchestrator_scenario_5_adaptive_improvement(low_confidence_context, high_confidence_context):
    """SCENARIO 5: Iteration 0 low confidence -> FeedbackRequest; Iteration 1 high confidence -> Reasoning success."""
    mock_http_client = MagicMock()
    resp_agent2 = MagicMock(status_code=200)
    resp_agent2.json.return_value = {"biomedical_findings": ["Finding 1"]}
    resp_agent4 = MagicMock(status_code=200)
    resp_agent4.json.return_value = high_confidence_context.model_dump()

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="low_confidence",
        evidence_gaps=["Gap 1"],
        requested_sources=["agent_2"],
        agent_2_request=Agent2RetrievalRequirement(focus_terms=["Chest Pain"]),
        agent_3_request=None,
    )
    reasoning_resp = ClinicalDecisionSupportResponse(
        clinical_assessment={"primary_interpretation": "Success", "clinical_significance": "High", "differential_considerations": []},
        reasoning={"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""},
        decision_support={"recommended_actions": [], "additional_information_needed": [], "priority_level": "Routine"},
        safety={"safety_flags": [], "contraindications_or_concerns": []},
        uncertainty={"confidence_score": 0.92, "uncertainty_factors": []},
        traceability=[],
        agent_metadata={"agent": "agent_5", "model": settings.primary_reasoning_model, "reasoning_mode": "primary", "status": "success"},
    )
    mock_service.execute_reasoning_pipeline.side_effect = [fb_req, reasoning_resp]

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert mock_service.execute_reasoning_pipeline.call_count == 2


def test_orchestrator_scenario_6_persistent_low_confidence_terminates_at_max_retries(low_confidence_context):
    """SCENARIO 6: Persistent low confidence terminates loop at max retries with InsufficientEvidenceTermination."""
    mock_http_client = MagicMock()
    resp_agent2 = MagicMock(status_code=200)
    resp_agent2.json.return_value = {"biomedical_findings": ["Finding 1"]}
    resp_agent4 = MagicMock(status_code=200)
    resp_agent4.json.return_value = low_confidence_context.model_dump()

    mock_http_client.post.side_effect = lambda url, json=None, **kwargs: resp_agent2 if "8002" in url else resp_agent4

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req_0 = FeedbackRequest(status="needs_more_evidence", retry_count=0, max_retries=3, confidence_score=0.45, reason="low_confidence", evidence_gaps=[], requested_sources=["agent_2"], agent_2_request=Agent2RetrievalRequirement())
    fb_req_1 = FeedbackRequest(status="needs_more_evidence", retry_count=1, max_retries=3, confidence_score=0.45, reason="low_confidence", evidence_gaps=[], requested_sources=["agent_2"], agent_2_request=Agent2RetrievalRequirement())
    fb_req_2 = FeedbackRequest(status="needs_more_evidence", retry_count=2, max_retries=3, confidence_score=0.45, reason="low_confidence", evidence_gaps=[], requested_sources=["agent_2"], agent_2_request=Agent2RetrievalRequirement())
    termination_3 = InsufficientEvidenceTermination(status="insufficient_evidence", retry_count=3, max_retries=3, confidence_score=0.45, reason="low_confidence")

    mock_service.execute_reasoning_pipeline.side_effect = [fb_req_0, fb_req_1, fb_req_2, termination_3]

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, InsufficientEvidenceTermination)
    assert result.status == "insufficient_evidence"
    assert result.retry_count == 3


def test_orchestrator_scenario_7_agent_2_fails_agent_3_succeeds(low_confidence_context, high_confidence_context):
    """SCENARIO 7: Agent 2 fails (non-200), Agent 3 succeeds. Re-fusion proceeds with Agent 3 evidence."""
    mock_http_client = MagicMock()
    resp_agent2 = MagicMock(status_code=500)
    resp_agent3 = MagicMock(status_code=200)
    resp_agent3.json.return_value = {"clinical_evidence": ["Evidence 1"]}
    resp_agent4 = MagicMock(status_code=200)
    resp_agent4.json.return_value = high_confidence_context.model_dump()

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8003" in url:
            return resp_agent3
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="missing_information",
        evidence_gaps=["Gap 1"],
        requested_sources=["agent_2", "agent_3"],
        agent_2_request=Agent2RetrievalRequirement(),
        agent_3_request=Agent3RetrievalRequirement(),
    )
    reasoning_resp = ClinicalDecisionSupportResponse(
        clinical_assessment={"primary_interpretation": "Success", "clinical_significance": "High", "differential_considerations": []},
        reasoning={"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""},
        decision_support={"recommended_actions": [], "additional_information_needed": [], "priority_level": "Routine"},
        safety={"safety_flags": [], "contraindications_or_concerns": []},
        uncertainty={"confidence_score": 0.92, "uncertainty_factors": []},
        traceability=[],
        agent_metadata={"agent": "agent_5", "model": settings.primary_reasoning_model, "reasoning_mode": "primary", "status": "success"},
    )
    mock_service.execute_reasoning_pipeline.side_effect = [fb_req, reasoning_resp]

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)


def test_orchestrator_scenario_8_agent_3_fails_agent_2_succeeds(low_confidence_context, high_confidence_context):
    """SCENARIO 8: Agent 3 fails, Agent 2 succeeds. Re-fusion proceeds with Agent 2 evidence."""
    mock_http_client = MagicMock()
    resp_agent2 = MagicMock(status_code=200)
    resp_agent2.json.return_value = {"biomedical_findings": ["Finding 1"]}
    resp_agent3 = MagicMock(status_code=500)
    resp_agent4 = MagicMock(status_code=200)
    resp_agent4.json.return_value = high_confidence_context.model_dump()

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8003" in url:
            return resp_agent3
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="missing_information",
        evidence_gaps=["Gap 1"],
        requested_sources=["agent_2", "agent_3"],
        agent_2_request=Agent2RetrievalRequirement(),
        agent_3_request=Agent3RetrievalRequirement(),
    )
    reasoning_resp = ClinicalDecisionSupportResponse(
        clinical_assessment={"primary_interpretation": "Success", "clinical_significance": "High", "differential_considerations": []},
        reasoning={"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""},
        decision_support={"recommended_actions": [], "additional_information_needed": [], "priority_level": "Routine"},
        safety={"safety_flags": [], "contraindications_or_concerns": []},
        uncertainty={"confidence_score": 0.92, "uncertainty_factors": []},
        traceability=[],
        agent_metadata={"agent": "agent_5", "model": settings.primary_reasoning_model, "reasoning_mode": "primary", "status": "success"},
    )
    mock_service.execute_reasoning_pipeline.side_effect = [fb_req, reasoning_resp]

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)


def test_orchestrator_scenario_9_agent_2_and_3_both_fail(low_confidence_context):
    """SCENARIO 9: Both requested upstream agents fail. Orchestrator raises OrchestrationFailureError."""
    mock_http_client = MagicMock()
    resp_agent2 = MagicMock(status_code=500)
    resp_agent3 = MagicMock(status_code=500)

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8003" in url:
            return resp_agent3
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="missing_information",
        evidence_gaps=["Gap 1"],
        requested_sources=["agent_2", "agent_3"],
        agent_2_request=Agent2RetrievalRequirement(),
        agent_3_request=Agent3RetrievalRequirement(),
    )
    mock_service.execute_reasoning_pipeline.return_value = fb_req

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    with pytest.raises(OrchestrationFailureError) as exc_info:
        orchestrator.run(low_confidence_context)
    assert "Both requested upstream agents" in str(exc_info.value)


def test_orchestrator_scenario_10_agent_4_refusion_fails(low_confidence_context):
    """SCENARIO 10: Agent 4 re-fusion fails. Orchestrator raises OrchestrationFailureError."""
    mock_http_client = MagicMock()
    resp_agent2 = MagicMock(status_code=200, json=lambda: {"biomedical_findings": []})
    resp_agent4 = MagicMock(status_code=500)

    def mock_post(url, json=None, **kwargs):
        if "8002" in url:
            return resp_agent2
        elif "8004" in url:
            return resp_agent4
        raise ValueError(f"Unexpected POST url: {url}")

    mock_http_client.post.side_effect = mock_post

    mock_service = MagicMock(spec=ClinicalReasoningService)
    fb_req = FeedbackRequest(
        status="needs_more_evidence",
        retry_count=0,
        max_retries=3,
        confidence_score=0.45,
        reason="low_confidence",
        evidence_gaps=["Gap 1"],
        requested_sources=["agent_2"],
        agent_2_request=Agent2RetrievalRequirement(),
    )
    mock_service.execute_reasoning_pipeline.return_value = fb_req

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    with pytest.raises(OrchestrationFailureError) as exc_info:
        orchestrator.run(low_confidence_context)
    assert "Agent 4 re-fusion failed" in str(exc_info.value)


def test_orchestrator_scenario_11_agent_5_returns_insufficient_evidence(low_confidence_context):
    """SCENARIO 11: Agent 5 returns InsufficientEvidenceTermination directly. No retrieval, safe response returned."""
    mock_http_client = MagicMock()
    mock_service = MagicMock(spec=ClinicalReasoningService)

    termination = InsufficientEvidenceTermination(
        status="insufficient_evidence",
        retry_count=3,
        max_retries=3,
        confidence_score=0.45,
        reason="low_confidence",
    )
    mock_service.execute_reasoning_pipeline.return_value = termination

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=mock_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(low_confidence_context)

    assert isinstance(result, InsufficientEvidenceTermination)
    assert result.status == "insufficient_evidence"
    mock_http_client.post.assert_not_called()

def test_orchestrator_scenario_12_ready_for_reasoning_gemini_fails_groq_invoked(high_confidence_context):
    """SCENARIO 12: Context ready for reasoning, Gemini fails -> Groq fallback invoked. Demonstrates mechanism separation."""
    mock_http_client = MagicMock()

    primary = FailingMockEngine()

    mock_groq_http = MagicMock()
    mock_groq_resp = MagicMock(status_code=200)
    mock_groq_resp.json.return_value = {
        "choices": [{"message": {"content": '{"clinical_assessment": {"primary_interpretation": "Groq Llama Reasoning", "clinical_significance": "High", "differential_considerations": []}, "reasoning": {"key_findings": [], "supporting_factors": [], "conflicting_factors": [], "reasoning_summary": ""}, "decision_support": {"recommended_actions": [], "additional_information_needed": [], "priority_level": "Urgent"}, "safety": {"safety_flags": [], "contraindications_or_concerns": []}, "uncertainty": {"confidence_score": 0.85, "uncertainty_factors": []}, "traceability": [], "agent_metadata": {"agent": "agent_5", "model": "llama-3.1-8b-instruct", "reasoning_mode": "fallback_model", "status": "success"}}'}}]
    }
    mock_groq_http.post.return_value = mock_groq_resp

    fallback = GroqReasoningEngine(api_key="test_groq_key", http_client=mock_groq_http)

    reasoning_service = ClinicalReasoningService(
        primary_engine=primary,
        fallback_engine=fallback,
        use_dev_default_when_unimplemented=False,
    )

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=reasoning_service,
        max_retries=3,
        http_client=mock_http_client,
    )

    result = orchestrator.run(high_confidence_context)

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert result.agent_metadata.model == "llama-3.1-8b-instruct"
    assert result.agent_metadata.reasoning_mode == "fallback_model"
    # Verify no Agent 2/3/4 HTTP call was made
    mock_http_client.post.assert_not_called()
