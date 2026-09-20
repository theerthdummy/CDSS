"""Mock Agent 4 Re-Fusion Boundary Service for CDSS Agent 5 Adaptive Testing.

Simulates Agent 4's cumulative data fusion boundary when supplied with additional
upstream evidence items retrieved during adaptive feedback cycles.

IMPORTANT:
This service does NOT implement Agent 4's real semantic data fusion logic (normalization, graph resolution).
It is a controlled mock boundary designed specifically for Agent 5 adaptive loop simulation and integration testing.
"""

import logging
from typing import List, Optional
from app.models.input_models import (
    MergedFinding,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)
from app.services.upstream.base_provider import UpstreamEvidenceItem

logger = logging.getLogger("agent5.mock_refusion")


class MockAgent4RefusionError(Exception):
    """Exception raised when MockAgent4RefusionService encounters a simulated re-fusion failure."""

    pass


class MockAgent4RefusionService:
    """Mock boundary representing Agent 4 re-fusion capabilities (`POST /fuse`)."""

    def __init__(self, simulate_failure: bool = False) -> None:
        """Initialize MockAgent4RefusionService.
        
        Args:
            simulate_failure: If True, simulate a re-fusion failure.
        """
        self.simulate_failure = simulate_failure

    def refuse_context(
        self,
        current_context: UnifiedClinicalContext,
        additional_evidence: List[UpstreamEvidenceItem],
        correlation_id: str,
        target_confidence_score: Optional[float] = None,
    ) -> Optional[UnifiedClinicalContext]:
        """Perform simulated cumulative evidence re-fusion over UnifiedClinicalContext.
        
        Args:
            current_context: Original UnifiedClinicalContext before adaptive retrieval.
            additional_evidence: List of UpstreamEvidenceItem objects retrieved from upstream providers.
            correlation_id: Traceable correlation ID.
            target_confidence_score: Optional explicit confidence score override for the updated context.
            
        Returns:
            Updated UnifiedClinicalContext model instance, or None if re-fusion fails.
        """
        logger.info(
            f"MockAgent4RefusionService executing re-fusion for correlation ID '{correlation_id}' "
            f"with {len(additional_evidence)} new evidence items..."
        )

        if self.simulate_failure:
            logger.error("MockAgent4RefusionService simulated re-fusion failure!")
            return None

        if not additional_evidence:
            logger.warning("No additional evidence provided to MockAgent4RefusionService; context unchanged.")
            return current_context

        # Deep copy existing context fields
        merged_findings = list(current_context.merged_findings)
        supporting_evidence = list(current_context.supporting_evidence)
        source_traceability = list(current_context.source_traceability)
        medical_terms = set(current_context.normalized_medical_terms)

        for item in additional_evidence:
            merged_findings.append(
                MergedFinding(
                    finding=item.finding,
                    sources=[item.source],
                )
            )
            supporting_evidence.append(
                SupportingEvidence(
                    finding=item.finding,
                    source_attribution=item.source,
                )
            )
            source_traceability.append(
                SourceTraceability(
                    finding=item.finding,
                    sources=[item.source],
                    upstream_agents=[item.source_type],
                )
            )
            # Extract basic terms for normalization list
            terms = [t for t in item.finding.split() if len(t) > 3]
            medical_terms.update(terms[:2])

        # Compute updated confidence score
        if target_confidence_score is not None:
            new_confidence = max(0.0, min(1.0, target_confidence_score))
        else:
            # Default mock boost: if initial context was low confidence (e.g. 0.45), boost to 0.85
            new_confidence = max(current_context.confidence_score, 0.85)

        updated_summary = (
            f"{current_context.fusion_summary} "
            f"[Re-Fused Context with {len(additional_evidence)} additional upstream evidence items. "
            f"Correlation ID: {correlation_id}]"
        )

        updated_context = UnifiedClinicalContext(
            merged_findings=merged_findings,
            normalized_medical_terms=list(medical_terms),
            supporting_evidence=supporting_evidence,
            conflicting_evidence=list(current_context.conflicting_evidence),
            evidence_priority=list(current_context.evidence_priority),
            source_traceability=source_traceability,
            fusion_summary=updated_summary,
            confidence_score=new_confidence,
        )

        logger.info(
            f"MockAgent4RefusionService re-fusion complete. "
            f"Updated Confidence Score: {new_confidence:.2f}."
        )
        return updated_context
