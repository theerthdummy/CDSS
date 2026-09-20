"""Adaptive Optimizer Service for Agent 5.

Evaluates upstream Agent 4 UnifiedClinicalContext completeness, consistency,
relevance, and confidence against configurable thresholds before permitting clinical reasoning.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel

from app.config import settings
from app.models.feedback_models import (
    Agent2RetrievalRequirement,
    Agent3RetrievalRequirement,
    FeedbackRequest,
    InsufficientEvidenceTermination,
)
from app.models.input_models import UnifiedClinicalContext

logger = logging.getLogger("agent5.adaptive_optimizer")


class OptimizationDecision(BaseModel):
    """Result of AdaptiveOptimizer context evaluation."""

    status: str  # "ready_for_reasoning", "needs_more_evidence", or "insufficient_evidence"
    confidence_score: float
    reason: str
    evidence_gaps: List[str]
    feedback_request: Optional[FeedbackRequest] = None
    termination_payload: Optional[InsufficientEvidenceTermination] = None


class AdaptiveOptimizer:
    """Evaluates whether UnifiedClinicalContext contains sufficient, consistent evidence for reasoning.

    Maintains complete separation between adaptive evidence feedback and LLM model fallbacks.
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        """Initialize optimizer with configurable settings.

        Args:
            confidence_threshold: Minimum confidence required (default from settings: 0.70).
            max_retries: Maximum adaptive feedback retries (default from settings: 3).
        """
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.confidence_threshold
        )
        self.max_retries = (
            max_retries if max_retries is not None else settings.max_retries
        )

    def evaluate_context(
        self, context: UnifiedClinicalContext, retry_count: int = 0
    ) -> OptimizationDecision:
        """Deterministically evaluate UnifiedClinicalContext for evidence readiness.

        Args:
            context: The upstream Agent 4 UnifiedClinicalContext model.
            retry_count: Current adaptive retry iteration (default: 0).

        Returns:
            OptimizationDecision indicating "ready_for_reasoning", "needs_more_evidence", or "insufficient_evidence".
        """
        logger.info(
            f"Evaluating clinical context (confidence: {context.confidence_score:.2f}, "
            f"threshold: {self.confidence_threshold:.2f}, retry: {retry_count}/{self.max_retries})..."
        )

        evidence_gaps: List[str] = []
        low_confidence = context.confidence_score < self.confidence_threshold
        missing_findings = len(context.merged_findings) == 0
        missing_supporting = len(context.supporting_evidence) == 0
        has_conflicts = len(context.conflicting_evidence) > 0

        # Build explicit evidence gaps
        if missing_findings:
            evidence_gaps.append("No merged findings present in supplied clinical context.")

        if missing_supporting:
            evidence_gaps.append("No supporting evidence items present for clinical correlation.")

        if low_confidence:
            evidence_gaps.append(
                f"Overall context confidence score ({context.confidence_score:.2f}) "
                f"is below target threshold ({self.confidence_threshold:.2f})."
            )

        if has_conflicts:
            for ce in context.conflicting_evidence:
                evidence_gaps.append(
                    f"Unresolved {ce.type} regarding '{ce.finding}': {ce.description}"
                )

        # Context is READY if confidence is sufficient, findings exist, supporting evidence exists, and no severe unmitigated conflicts
        is_insufficient = low_confidence or missing_findings or missing_supporting or (has_conflicts and context.confidence_score < 0.80)

        if not is_insufficient:
            logger.info("Context evaluation PASSED: READY_FOR_REASONING.")
            return OptimizationDecision(
                status="ready_for_reasoning",
                confidence_score=context.confidence_score,
                reason="context_sufficient",
                evidence_gaps=[],
            )

        # Context is INSUFFICIENT -> Determine primary reason
        if missing_findings or missing_supporting:
            primary_reason = "missing_information"
        elif has_conflicts:
            primary_reason = "conflicting_evidence"
        else:
            primary_reason = "low_confidence"

        # Check if retries remain
        if retry_count < self.max_retries:
            logger.warning(
                f"Context evaluation FAILED (retry {retry_count} < {self.max_retries}): NEEDS_MORE_EVIDENCE. "
                f"Primary reason: {primary_reason}."
            )

            feedback_request = self.build_feedback_request(
                context=context,
                confidence_score=context.confidence_score,
                reason=primary_reason,
                evidence_gaps=evidence_gaps,
                retry_count=retry_count,
            )

            return OptimizationDecision(
                status="needs_more_evidence",
                confidence_score=context.confidence_score,
                reason=primary_reason,
                evidence_gaps=evidence_gaps,
                feedback_request=feedback_request,
            )
        else:
            logger.error(
                f"Context evaluation FAILED (retry {retry_count} >= max {self.max_retries}): INSUFFICIENT_EVIDENCE. "
                "Safe execution termination enforced."
            )

            termination_payload = self.build_termination_payload(
                context=context,
                confidence_score=context.confidence_score,
                reason=primary_reason,
                evidence_gaps=evidence_gaps,
                retry_count=retry_count,
            )

            return OptimizationDecision(
                status="insufficient_evidence",
                confidence_score=context.confidence_score,
                reason=primary_reason,
                evidence_gaps=evidence_gaps,
                termination_payload=termination_payload,
            )

    def build_feedback_request(
        self,
        context: UnifiedClinicalContext,
        confidence_score: float,
        reason: str,
        evidence_gaps: List[str],
        retry_count: int = 0,
    ) -> FeedbackRequest:
        """Construct structured FeedbackRequest to re-query upstream retrieval agents."""
        agent_2_req = Agent2RetrievalRequirement(
            query_requirements=[
                f"Retrieve biomedical RAG context and knowledge graph relationships for target terms: {', '.join(context.normalized_medical_terms[:5])}"
            ] if context.normalized_medical_terms else ["Retrieve comprehensive biomedical knowledge graph context."],
            focus_terms=context.normalized_medical_terms,
            knowledge_graph_areas=[ce.finding for ce in context.conflicting_evidence],
        )

        agent_3_req = Agent3RetrievalRequirement(
            evidence_requirements=[
                f"Search latest clinical guidelines and PubMed literature for evidence clarifying gap: {gap}"
                for gap in evidence_gaps[:3]
            ] if evidence_gaps else ["Search for peer-reviewed evidence to resolve diagnostic uncertainty."],
            publication_recency="Last 5 years",
            conflict_verification_requirements=[ce.description for ce in context.conflicting_evidence],
        )

        return FeedbackRequest(
            status="needs_more_evidence",
            retry_count=retry_count,
            max_retries=self.max_retries,
            confidence_score=round(confidence_score, 2),
            confidence_threshold=self.confidence_threshold,
            reason=reason,
            evidence_gaps=evidence_gaps or ["Evidence is insufficient to reach the target clinical confidence threshold."],
            requested_sources=["agent_2", "agent_3"],
            agent_2_request=agent_2_req,
            agent_3_request=agent_3_req,
        )

    def build_termination_payload(
        self,
        context: UnifiedClinicalContext,
        confidence_score: float,
        reason: str,
        evidence_gaps: List[str],
        retry_count: int = 0,
    ) -> InsufficientEvidenceTermination:
        """Construct InsufficientEvidenceTermination when max retries are exhausted."""
        return InsufficientEvidenceTermination(
            status="insufficient_evidence",
            retry_count=retry_count,
            max_retries=self.max_retries,
            confidence_score=round(confidence_score, 2),
            reason=reason,
            evidence_gaps=evidence_gaps or ["Evidence remains insufficient after maximum adaptive retries."],
            conflicting_evidence_summary=[ce.description for ce in context.conflicting_evidence],
            termination_message=(
                f"Maximum adaptive retrieval retries ({self.max_retries}) reached without "
                f"satisfying target confidence threshold ({self.confidence_threshold:.2f}). "
                "Safe termination enforced to prevent clinical hallucination."
            ),
        )
