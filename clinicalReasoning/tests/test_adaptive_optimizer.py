"""Unit tests for AdaptiveOptimizer service and feedback requests."""

import pytest

from app.models.feedback_models import FeedbackRequest, InsufficientEvidenceTermination
from app.models.input_models import (
    ConflictEvidence,
    EvidencePriority,
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.services.adaptive_optimizer import AdaptiveOptimizer


@pytest.fixture
def high_confidence_context():
    """Valid UnifiedClinicalContext with high confidence score (0.92)."""
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
    """Valid UnifiedClinicalContext with low confidence score (0.48)."""
    return UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(
                finding="Atypical chest pain with non-specific ECG changes",
                sources=["Agent 2"],
            )
        ],
        normalized_medical_terms=["Chest Pain", "ECG"],
        supporting_evidence=[
            SupportingEvidence(
                finding="Initial troponin I within normal limits",
                source_attribution="Agent 2 Labs",
            )
        ],
        conflicting_evidence=[
            ConflictEvidence(
                type="Diagnostic Conflict",
                finding="Ischemia vs Musculoskeletal Pain",
                description="Equivocal ECG findings with normal cardiac enzymes.",
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
        fusion_summary="Equivocal presentation of chest pain requiring further evidence.",
        confidence_score=0.48,
    )


def test_high_confidence_context_ready_for_reasoning(high_confidence_context):
    """Verify high confidence context produces READY_FOR_REASONING status."""
    optimizer = AdaptiveOptimizer(confidence_threshold=0.70, max_retries=3)
    decision = optimizer.evaluate_context(high_confidence_context, retry_count=0)

    assert decision.status == "ready_for_reasoning"
    assert decision.reason == "context_sufficient"
    assert len(decision.evidence_gaps) == 0
    assert decision.feedback_request is None
    assert decision.termination_payload is None


def test_low_confidence_context_needs_more_evidence(low_confidence_context):
    """Verify low confidence context produces NEEDS_MORE_EVIDENCE with FeedbackRequest."""
    optimizer = AdaptiveOptimizer(confidence_threshold=0.70, max_retries=3)
    decision = optimizer.evaluate_context(low_confidence_context, retry_count=0)

    assert decision.status == "needs_more_evidence"
    assert decision.feedback_request is not None
    assert isinstance(decision.feedback_request, FeedbackRequest)

    fb = decision.feedback_request
    assert fb.status == "needs_more_evidence"
    assert fb.retry_count == 0
    assert fb.max_retries == 3
    assert fb.confidence_score == 0.48
    assert len(fb.evidence_gaps) > 0

    # Verify structured Agent 2 and Agent 3 retrieval requests
    assert "agent_2" in fb.requested_sources
    assert "agent_3" in fb.requested_sources
    assert fb.agent_2_request is not None
    assert len(fb.agent_2_request.focus_terms) > 0
    assert fb.agent_3_request is not None
    assert len(fb.agent_3_request.conflict_verification_requirements) > 0


def test_conflicting_evidence_triggers_feedback():
    """Verify conflicting evidence with moderate confidence triggers NEEDS_MORE_EVIDENCE."""
    context = UnifiedClinicalContext(
        merged_findings=[
            MergedFinding(finding="Possible pulmonary embolism", sources=["Agent 2"])
        ],
        normalized_medical_terms=["PE", "D-dimer"],
        supporting_evidence=[
            SupportingEvidence(finding="Elevated D-dimer", source_attribution="Labs")
        ],
        conflicting_evidence=[
            ConflictEvidence(
                type="Treatment Contraindication",
                finding="Anticoagulation",
                description="Active gastrointestinal bleeding reported in nursing notes.",
            )
        ],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Possible PE with active GI bleeding conflict.",
        confidence_score=0.75,  # Below 0.80 conflict threshold
    )

    optimizer = AdaptiveOptimizer(confidence_threshold=0.70, max_retries=3)
    decision = optimizer.evaluate_context(context, retry_count=0)

    assert decision.status == "needs_more_evidence"
    assert decision.reason == "conflicting_evidence"
    assert any("Unresolved Treatment Contraindication" in gap for gap in decision.evidence_gaps)


def test_missing_findings_triggers_feedback():
    """Verify context missing merged findings triggers NEEDS_MORE_EVIDENCE."""
    context = UnifiedClinicalContext(
        merged_findings=[],
        normalized_medical_terms=[],
        supporting_evidence=[],
        conflicting_evidence=[],
        evidence_priority=[],
        source_traceability=[],
        fusion_summary="Empty context payload.",
        confidence_score=0.90,
    )

    optimizer = AdaptiveOptimizer(confidence_threshold=0.70, max_retries=3)
    decision = optimizer.evaluate_context(context, retry_count=0)

    assert decision.status == "needs_more_evidence"
    assert decision.reason == "missing_information"


def test_retry_count_progression(low_confidence_context):
    """Verify retry_count progression from 0 to max_retries."""
    optimizer = AdaptiveOptimizer(confidence_threshold=0.70, max_retries=3)

    # Retry 0: Needs more evidence
    d0 = optimizer.evaluate_context(low_confidence_context, retry_count=0)
    assert d0.status == "needs_more_evidence"
    assert d0.feedback_request.retry_count == 0

    # Retry 2: Still needs more evidence
    d2 = optimizer.evaluate_context(low_confidence_context, retry_count=2)
    assert d2.status == "needs_more_evidence"
    assert d2.feedback_request.retry_count == 2

    # Retry 3 (>= max_retries): Safe termination
    d3 = optimizer.evaluate_context(low_confidence_context, retry_count=3)
    assert d3.status == "insufficient_evidence"
    assert d3.feedback_request is None
    assert isinstance(d3.termination_payload, InsufficientEvidenceTermination)
    assert d3.termination_payload.retry_count == 3
    assert d3.termination_payload.max_retries == 3


def test_deterministic_evaluation(low_confidence_context):
    """Verify optimizer produces deterministic evaluation outputs for identical input."""
    optimizer = AdaptiveOptimizer(confidence_threshold=0.70, max_retries=3)
    d1 = optimizer.evaluate_context(low_confidence_context, retry_count=1)
    d2 = optimizer.evaluate_context(low_confidence_context, retry_count=1)

    assert d1.status == d2.status
    assert d1.reason == d2.reason
    assert d1.evidence_gaps == d2.evidence_gaps
    assert d1.feedback_request.model_dump() == d2.feedback_request.model_dump()



def test_original_context_confidence_never_modified(low_confidence_context):
    """Verify evaluating context does not mutate original Agent 4 confidence score."""
    original_score = low_confidence_context.confidence_score
    optimizer = AdaptiveOptimizer(confidence_threshold=0.70, max_retries=3)
    optimizer.evaluate_context(low_confidence_context, retry_count=1)

    assert low_confidence_context.confidence_score == original_score
