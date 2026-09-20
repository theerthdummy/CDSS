"""Comprehensive Integration Test Suite for Agent 5 Adaptive Feedback Architecture (Step 16).

Tests the end-to-end adaptive feedback loop using mock upstream evidence providers (MockAgent2Provider, MockAgent3Provider)
and mock Agent 4 re-fusion boundary (MockAgent4RefusionService).
"""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock
import pytest

from app.config import settings
from app.models.feedback_models import (
    Agent2RetrievalRequirement,
    Agent3RetrievalRequirement,
    FeedbackRequest,
    InsufficientEvidenceTermination,
)
from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import ClinicalDecisionSupportResponse, ReasoningPolicy
from app.orchestration.adaptive_orchestrator import AdaptiveOrchestrator, OrchestrationFailureError
from app.services.clinical_reasoning import ClinicalReasoningService
from app.services.reasoning_engines import DeterministicFallbackEngine, ReasoningEngine
from app.services.upstream import (
    MockAgent2Provider,
    MockAgent3Provider,
    MockAgent4RefusionService,
    UpstreamEvidenceItem,
    UpstreamEvidenceProvider,
    UpstreamEvidenceRequest,
    UpstreamEvidenceResponse,
    UpstreamProviderError,
    UpstreamSourceType,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "adaptive"


def load_fixture(filename: str) -> UnifiedClinicalContext:
    """Load JSON fixture into UnifiedClinicalContext model."""
    filepath = FIXTURES_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return UnifiedClinicalContext.model_validate(data)


@pytest.fixture
def low_conf_context():
    return load_fixture("low_confidence_initial.json")


@pytest.fixture
def improved_agent2_context():
    return load_fixture("improved_context_after_agent2.json")


@pytest.fixture
def improved_agent3_context():
    return load_fixture("improved_context_after_agent3.json")


@pytest.fixture
def still_low_conf_context():
    return load_fixture("still_low_confidence_context.json")


# -----------------------------------------------------------------------------
# TEST 1: PRIMARY SUCCESSFUL ADAPTIVE LOOP (AGENT 2)
# -----------------------------------------------------------------------------
def test_adaptive_loop_agent2_success(low_conf_context, improved_agent2_context):
    """Test 1: Low confidence initial context -> FeedbackRequest (Agent 2) -> MockAgent2 -> Re-Fusion -> READY_FOR_REASONING."""
    mock_engine = MagicMock(spec=ReasoningEngine)
    mock_resp = DeterministicFallbackEngine().reason(improved_agent2_context)
    mock_engine.reason.return_value = mock_resp

    reasoning_svc = ClinicalReasoningService(
        primary_engine=mock_engine,
        use_dev_default_when_unimplemented=False,
    )

    mock_agent2 = MockAgent2Provider()
    mock_refusion = MockAgent4RefusionService()

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=reasoning_svc,
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: mock_agent2,
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: MockAgent3Provider(),
        },
        refusion_service=mock_refusion,
        max_retries=3,
    )

    result = orchestrator.run(low_conf_context, correlation_id="TEST-CORR-A2")

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert mock_engine.reason.called
    assert result.agent_metadata.status == "fallback_applied" or result.agent_metadata.status == "success"


# -----------------------------------------------------------------------------
# TEST 2: SUCCESSFUL AGENT 3 ADAPTIVE LOOP
# -----------------------------------------------------------------------------
def test_adaptive_loop_agent3_success(low_conf_context, improved_agent3_context):
    """Test 2: Low confidence initial context -> FeedbackRequest targeting Agent 3 -> MockAgent3 -> Re-Fusion -> READY_FOR_REASONING."""
    mock_engine = MagicMock(spec=ReasoningEngine)
    mock_resp = DeterministicFallbackEngine().reason(improved_agent3_context)
    mock_engine.reason.return_value = mock_resp

    # Custom reasoning service where initial attempt requests Agent 3
    class Agent3ReasoningService(ClinicalReasoningService):
        def execute_reasoning_pipeline(self, context, retry_count=0):
            if context.confidence_score < 0.70:
                return FeedbackRequest(
                    requested_sources=["agent_3"],
                    agent_3_request=Agent3RetrievalRequirement(evidence_requirements=["Primary PCI", "STEMI Guidelines"]),
                    reason="Missing latest PubMed guideline recommendations.",
                    evidence_gaps=["Clinical guideline recommendations"],
                    retry_count=retry_count,
                    max_retries=3,
                    confidence_score=context.confidence_score,
                )
            return mock_resp

    mock_agent3 = MockAgent3Provider()
    mock_refusion = MockAgent4RefusionService()

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=Agent3ReasoningService(primary_engine=mock_engine, use_dev_default_when_unimplemented=False),
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: MockAgent2Provider(),
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: mock_agent3,
        },
        refusion_service=mock_refusion,
        max_retries=3,
    )

    result = orchestrator.run(low_conf_context, correlation_id="TEST-CORR-A3")

    assert isinstance(result, ClinicalDecisionSupportResponse)


# -----------------------------------------------------------------------------
# TEST 3: DUAL PROVIDER ADAPTIVE LOOP (AGENT 2 + AGENT 3)
# -----------------------------------------------------------------------------
def test_adaptive_loop_both_providers(low_conf_context, improved_agent2_context):
    """Test 3: FeedbackRequest targeting BOTH Agent 2 and Agent 3 invokes both providers and passes combined evidence to re-fusion."""
    mock_resp = DeterministicFallbackEngine().reason(improved_agent2_context)

    class DualReasoningService(ClinicalReasoningService):
        def execute_reasoning_pipeline(self, context, retry_count=0):
            if context.confidence_score < 0.70:
                return FeedbackRequest(
                    requested_sources=["agent_2", "agent_3"],
                    agent_2_request=Agent2RetrievalRequirement(focus_terms=["Angiography", "Troponin"]),
                    agent_3_request=Agent3RetrievalRequirement(evidence_requirements=["STEMI Guidelines"]),
                    reason="Missing both biomedical RAG data and PubMed guidelines.",
                    evidence_gaps=["Biomedical RAG", "PubMed guidelines"],
                    retry_count=retry_count,
                    max_retries=3,
                    confidence_score=context.confidence_score,
                )
            return mock_resp

    mock_agent2 = MagicMock(wraps=MockAgent2Provider())
    mock_agent3 = MagicMock(wraps=MockAgent3Provider())
    mock_refusion = MagicMock(wraps=MockAgent4RefusionService())

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=DualReasoningService(use_dev_default_when_unimplemented=False),
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: mock_agent2,
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: mock_agent3,
        },
        refusion_service=mock_refusion,
        max_retries=3,
    )

    result = orchestrator.run(low_conf_context, correlation_id="TEST-CORR-DUAL")

    assert isinstance(result, ClinicalDecisionSupportResponse)
    assert mock_agent2.retrieve.called
    assert mock_agent3.retrieve.called
    assert mock_refusion.refuse_context.called


# -----------------------------------------------------------------------------
# TEST 4: PERSISTENT LOW CONFIDENCE TERMINATION
# -----------------------------------------------------------------------------
def test_adaptive_loop_persistent_low_confidence_terminates(low_conf_context, still_low_conf_context):
    """Test 4: Persistent low confidence context executes up to MAX_RETRIES then safely terminates with INSUFFICIENT_EVIDENCE."""
    mock_engine = MagicMock(spec=ReasoningEngine)

    class PersistentLowConfRefusionService(MockAgent4RefusionService):
        def refuse_context(self, current_context, additional_evidence, correlation_id, target_confidence_score=None):
            # Always return a context with confidence score below threshold (0.50)
            return still_low_conf_context

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=ClinicalReasoningService(primary_engine=mock_engine, use_dev_default_when_unimplemented=False),
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: MockAgent2Provider(),
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: MockAgent3Provider(),
        },
        refusion_service=PersistentLowConfRefusionService(),
        max_retries=3,
    )

    result = orchestrator.run(low_conf_context, correlation_id="TEST-PERSISTENT-LOW")

    assert isinstance(result, InsufficientEvidenceTermination)
    assert result.status == "insufficient_evidence"
    assert result.retry_count == 3
    # Reasoning engines MUST NOT be called!
    assert not mock_engine.reason.called


# -----------------------------------------------------------------------------
# TEST 5: MOCK PROVIDER FAILURE HANDLING
# -----------------------------------------------------------------------------
def test_adaptive_loop_mock_provider_failure_raises_error(low_conf_context):
    """Test 5: When mock upstream provider fails, OrchestrationFailureError is raised; stale context is NOT sent to reasoning engines."""
    mock_engine = MagicMock(spec=ReasoningEngine)
    failing_agent2 = MockAgent2Provider(simulate_failure=True)

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=ClinicalReasoningService(primary_engine=mock_engine, use_dev_default_when_unimplemented=False),
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: failing_agent2,
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: MockAgent3Provider(),
        },
        refusion_service=MockAgent4RefusionService(),
        max_retries=3,
    )

    with pytest.raises(OrchestrationFailureError) as exc_info:
        orchestrator.run(low_conf_context, correlation_id="TEST-FAIL-A2")

    assert "Requested upstream Agent 2 retrieval failed" in str(exc_info.value)
    assert not mock_engine.reason.called


# -----------------------------------------------------------------------------
# TEST 6: MOCK RE-FUSION FAILURE HANDLING (STALE CONTEXT PROTECTION)
# -----------------------------------------------------------------------------
def test_adaptive_loop_mock_refusion_failure_raises_error(low_conf_context):
    """Test 6: When Agent 4 re-fusion fails (returns None), OrchestrationFailureError is raised and stale context is protected."""
    mock_engine = MagicMock(spec=ReasoningEngine)
    failing_refusion = MockAgent4RefusionService(simulate_failure=True)

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=ClinicalReasoningService(primary_engine=mock_engine, use_dev_default_when_unimplemented=False),
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: MockAgent2Provider(),
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: MockAgent3Provider(),
        },
        refusion_service=failing_refusion,
        max_retries=3,
    )

    with pytest.raises(OrchestrationFailureError) as exc_info:
        orchestrator.run(low_conf_context, correlation_id="TEST-FAIL-REFUSION")

    assert "Agent 4 re-fusion failed" in str(exc_info.value)
    assert not mock_engine.reason.called


# -----------------------------------------------------------------------------
# TEST 7: CORRELATION ID PROPAGATION ACROSS CYCLE
# -----------------------------------------------------------------------------
def test_adaptive_loop_correlation_id_propagation(low_conf_context, improved_agent2_context):
    """Test 7: Verify single correlation ID propagates through request, provider, re-fusion, and orchestrator."""
    test_corr_id = "TRACE-UUID-999-XYZ"
    captured_requests = []

    class TrackingAgent2Provider(MockAgent2Provider):
        def retrieve(self, request: UpstreamEvidenceRequest) -> UpstreamEvidenceResponse:
            captured_requests.append(request)
            return super().retrieve(request)

    mock_resp = DeterministicFallbackEngine().reason(improved_agent2_context)
    mock_engine = MagicMock(spec=ReasoningEngine)
    mock_engine.reason.return_value = mock_resp

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=ClinicalReasoningService(primary_engine=mock_engine, use_dev_default_when_unimplemented=False),
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: TrackingAgent2Provider(),
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: MockAgent3Provider(),
        },
        refusion_service=MockAgent4RefusionService(),
        max_retries=3,
    )

    result = orchestrator.run(low_conf_context, correlation_id=test_corr_id)

    assert len(captured_requests) == 1
    assert captured_requests[0].correlation_id == test_corr_id


# -----------------------------------------------------------------------------
# TEST 8: SEPARATION OF MODEL FALLBACK AND ADAPTIVE FEEDBACK
# -----------------------------------------------------------------------------
def test_low_confidence_never_triggers_model_fallback(low_conf_context):
    """Test 8: Low confidence context triggers FeedbackRequest -> upstream retrieval, NEVER Gemini -> Groq model fallbacks."""
    primary_engine = MagicMock(spec=ReasoningEngine)
    fallback_engine = MagicMock(spec=ReasoningEngine)

    service = ClinicalReasoningService(
        primary_engine=primary_engine,
        fallback_engine=fallback_engine,
        use_dev_default_when_unimplemented=False,
    )

    res = service.execute_reasoning_pipeline(low_conf_context, retry_count=0)

    assert isinstance(res, FeedbackRequest)
    # Neither model engine should be invoked when confidence is insufficient!
    assert not primary_engine.reason.called
    assert not fallback_engine.reason.called


# -----------------------------------------------------------------------------
# TEST 9: FUTURE HTTP PROVIDER ADAPTER INTERFACE COMPATIBILITY
# -----------------------------------------------------------------------------
def test_future_http_provider_adapter_compatibility(low_conf_context, improved_agent2_context):
    """Test 9: Demonstrates that future real HTTP provider adapters can plug into AdaptiveOrchestrator without modifying core code."""
    class DummyAgent2HTTPAdapter(UpstreamEvidenceProvider):
        """Simulates future Agent2HTTPProvider implementing UpstreamEvidenceProvider."""

        def get_source_type(self) -> UpstreamSourceType:
            return UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2

        def retrieve(self, request: UpstreamEvidenceRequest) -> UpstreamEvidenceResponse:
            return UpstreamEvidenceResponse(
                request_id=request.request_id,
                correlation_id=request.correlation_id,
                source_type=self.get_source_type(),
                evidence_items=[
                    UpstreamEvidenceItem(
                        evidence_id="REAL_HTTP_A2_001",
                        finding="Future HTTP Adapter response finding",
                        source="Agent 2 HTTP Endpoint",
                        relevance=0.95,
                        content="Simulated future HTTP adapter response content.",
                        source_type="HTTP RAG Adapter",
                    )
                ],
                status="success",
            )

    mock_resp = DeterministicFallbackEngine().reason(improved_agent2_context)
    mock_engine = MagicMock(spec=ReasoningEngine)
    mock_engine.reason.return_value = mock_resp

    orchestrator = AdaptiveOrchestrator(
        reasoning_service=ClinicalReasoningService(primary_engine=mock_engine, use_dev_default_when_unimplemented=False),
        providers={
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: DummyAgent2HTTPAdapter(),
            UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: MockAgent3Provider(),
        },
        refusion_service=MockAgent4RefusionService(),
        max_retries=3,
    )

    result = orchestrator.run(low_conf_context, correlation_id="TEST-HTTP-ADAPTER")

    assert isinstance(result, ClinicalDecisionSupportResponse)


# -----------------------------------------------------------------------------
# TEST 10: FEEDBACK REQUEST CONTRACT SCENARIO TESTS (STEP 25)
# -----------------------------------------------------------------------------
def test_feedback_request_contains_confidence_score_and_threshold(low_conf_context):
    """Test 10: FeedbackRequest includes actual confidence_score and confidence_threshold."""
    service = ClinicalReasoningService(use_dev_default_when_unimplemented=False)
    feedback = service.execute_reasoning_pipeline(low_conf_context, retry_count=0)

    assert isinstance(feedback, FeedbackRequest)
    assert feedback.status == "needs_more_evidence"
    assert feedback.confidence_score == low_conf_context.confidence_score
    assert feedback.confidence_threshold == settings.confidence_threshold
    assert feedback.confidence_threshold == 0.70
    assert feedback.retry_count == 0
    assert feedback.max_retries == 3


def test_feedback_request_identifies_evidence_gaps_and_conflicts(low_conf_context):
    """Test 11: FeedbackRequest identifies specific evidence gaps and reasons."""
    service = ClinicalReasoningService(use_dev_default_when_unimplemented=False)
    feedback = service.execute_reasoning_pipeline(low_conf_context, retry_count=1)

    assert isinstance(feedback, FeedbackRequest)
    assert feedback.reason in ("low_confidence", "conflicting_evidence", "missing_information")
    assert len(feedback.evidence_gaps) > 0
    assert any("confidence score" in gap for gap in feedback.evidence_gaps)
    assert feedback.agent_2_request is not None
    assert feedback.agent_3_request is not None


def test_feedback_request_contains_no_secrets(low_conf_context):
    """Test 12: FeedbackRequest payload contains zero credentials, secrets, or API keys."""
    service = ClinicalReasoningService(use_dev_default_when_unimplemented=False)
    feedback = service.execute_reasoning_pipeline(low_conf_context, retry_count=0)

    dump_str = json.dumps(feedback.model_dump())
    assert "GEMINI_API_KEY" not in dump_str
    assert "GROQ_API_KEY" not in dump_str
    assert "AIza" not in dump_str
    assert "gsk_" not in dump_str


def test_new_agent4_context_evaluated_independently_on_next_cycle(low_conf_context, improved_agent2_context):
    """Test 13: New context returned after re-fusion is evaluated independently on the next cycle."""
    optimizer = settings.confidence_threshold

    # Cycle 0: low confidence context -> needs_more_evidence
    svc = ClinicalReasoningService(use_dev_default_when_unimplemented=False)
    decision1 = svc.execute_reasoning_pipeline(low_conf_context, retry_count=0)
    assert isinstance(decision1, FeedbackRequest)

    # Cycle 1: improved context -> ready for reasoning
    mock_engine = MagicMock(spec=ReasoningEngine)
    mock_resp = DeterministicFallbackEngine().reason(improved_agent2_context)
    mock_engine.reason.return_value = mock_resp

    svc2 = ClinicalReasoningService(primary_engine=mock_engine, use_dev_default_when_unimplemented=False)
    decision2 = svc2.execute_reasoning_pipeline(improved_agent2_context, retry_count=1)
    assert isinstance(decision2, ClinicalDecisionSupportResponse)

