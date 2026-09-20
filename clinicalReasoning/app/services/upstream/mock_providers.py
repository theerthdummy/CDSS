"""Mock Upstream Evidence Providers for Agent 5 Adaptive Loop Simulation.

Provides deterministic synthetic implementations for simulating future Agent 2 (Biomedical RAG)
and Agent 3 (PubMed / Latest Medical Evidence) evidence retrieval.

IMPORTANT:
These are development/test simulation boundaries ONLY.
They do NOT call LLMs or external network endpoints and are NOT actual implementations of Agent 2 or Agent 3.
"""

import logging
from typing import List, Optional
from app.services.upstream.base_provider import (
    UpstreamEvidenceItem,
    UpstreamEvidenceProvider,
    UpstreamEvidenceRequest,
    UpstreamEvidenceResponse,
    UpstreamProviderError,
    UpstreamSourceType,
)

logger = logging.getLogger("agent5.mock_providers")


class MockAgent2Provider(UpstreamEvidenceProvider):
    """Mock implementation representing future Agent 2 (Biomedical RAG & Knowledge Graph).
    
    Returns synthetic biomedical knowledge items deterministically.
    """

    def __init__(
        self,
        custom_evidence: Optional[List[UpstreamEvidenceItem]] = None,
        simulate_failure: bool = False,
    ) -> None:
        """Initialize MockAgent2Provider.
        
        Args:
            custom_evidence: Optional predefined list of synthetic evidence items.
            simulate_failure: If True, simulate an upstream retrieval failure.
        """
        self.custom_evidence = custom_evidence
        self.simulate_failure = simulate_failure

    def get_source_type(self) -> UpstreamSourceType:
        return UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2

    def retrieve(self, request: UpstreamEvidenceRequest) -> UpstreamEvidenceResponse:
        logger.info(
            f"MockAgent2Provider handling retrieval request '{request.request_id}' "
            f"(Correlation ID: {request.correlation_id}, Retry: {request.retry_count})..."
        )

        if self.simulate_failure:
            logger.error("MockAgent2Provider simulated upstream retrieval failure!")
            raise UpstreamProviderError("Simulated failure in MockAgent2Provider.")

        if self.custom_evidence is not None:
            evidence_items = self.custom_evidence
        else:
            # Predefined deterministic synthetic biomedical RAG evidence
            evidence_items = [
                UpstreamEvidenceItem(
                    evidence_id="MOCK_A2_BIO_001",
                    finding="Coronary angiography findings confirm culprit 95% stenosis in LAD coronary artery",
                    source="Agent 2 Biomedical RAG Mock",
                    relevance=0.95,
                    content="Biomedical knowledge graph association links ST elevation in V1-V4 with anterior wall ischemia and LAD occlusion.",
                    source_type="Biomedical Knowledge Graph",
                ),
                UpstreamEvidenceItem(
                    evidence_id="MOCK_A2_BIO_002",
                    finding="Troponin I elevation (>4.5 ng/mL) confirms acute myocardial injury",
                    source="Agent 2 Lab Knowledge Base",
                    relevance=0.92,
                    content="Biomedical laboratory threshold database indicates cardiac troponin I > 0.04 ng/mL diagnostic for acute myocardial injury.",
                    source_type="Laboratory Knowledge Base",
                ),
            ]

        logger.info(f"MockAgent2Provider returning {len(evidence_items)} synthetic evidence items.")
        return UpstreamEvidenceResponse(
            request_id=request.request_id,
            correlation_id=request.correlation_id,
            source_type=self.get_source_type(),
            evidence_items=evidence_items,
            status="success",
        )


class MockAgent3Provider(UpstreamEvidenceProvider):
    """Mock implementation representing future Agent 3 (Latest Medical Evidence / PubMed).
    
    Returns synthetic PubMed / guideline clinical evidence items deterministically.
    """

    def __init__(
        self,
        custom_evidence: Optional[List[UpstreamEvidenceItem]] = None,
        simulate_failure: bool = False,
    ) -> None:
        """Initialize MockAgent3Provider.
        
        Args:
            custom_evidence: Optional predefined list of synthetic evidence items.
            simulate_failure: If True, simulate an upstream retrieval failure.
        """
        self.custom_evidence = custom_evidence
        self.simulate_failure = simulate_failure

    def get_source_type(self) -> UpstreamSourceType:
        return UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3

    def retrieve(self, request: UpstreamEvidenceRequest) -> UpstreamEvidenceResponse:
        logger.info(
            f"MockAgent3Provider handling search request '{request.request_id}' "
            f"(Correlation ID: {request.correlation_id}, Retry: {request.retry_count})..."
        )

        if self.simulate_failure:
            logger.error("MockAgent3Provider simulated upstream search failure!")
            raise UpstreamProviderError("Simulated failure in MockAgent3Provider.")

        if self.custom_evidence is not None:
            evidence_items = self.custom_evidence
        else:
            # Predefined deterministic synthetic PubMed clinical evidence
            evidence_items = [
                UpstreamEvidenceItem(
                    evidence_id="MOCK_A3_PUBMED_001",
                    finding="ACC/AHA Guidelines: Emergency primary PCI recommended within 90 minutes of first medical contact for STEMI",
                    source="Agent 3 PubMed Search Mock",
                    relevance=0.96,
                    content="Class I guideline recommendation: Primary PCI is preferred reperfusion strategy over fibrinolysis when performed within 90 minutes.",
                    source_type="Clinical Practice Guidelines",
                ),
                UpstreamEvidenceItem(
                    evidence_id="MOCK_A3_PUBMED_002",
                    finding="Dual antiplatelet therapy (Aspirin + P2Y12 inhibitor) indicated immediately upon STEMI diagnosis",
                    source="Agent 3 PubMed Search Mock",
                    relevance=0.90,
                    content="Systematic review demonstrates significant 30-day mortality reduction with prompt dual antiplatelet initiation.",
                    source_type="Systematic Review",
                ),
            ]

        logger.info(f"MockAgent3Provider returning {len(evidence_items)} synthetic evidence items.")
        return UpstreamEvidenceResponse(
            request_id=request.request_id,
            correlation_id=request.correlation_id,
            source_type=self.get_source_type(),
            evidence_items=evidence_items,
            status="success",
        )
