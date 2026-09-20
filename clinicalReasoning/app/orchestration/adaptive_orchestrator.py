"""External Adaptive Orchestrator for CDSS.

Manages the adaptive feedback loop between Agent 5, upstream evidence providers, and Agent 4 re-fusion.
Supports both future HTTP adapters and pluggable UpstreamEvidenceProvider abstractions.

Flow:
Agent 4 Context -> Agent 5 Optimizer -> FeedbackRequest -> Upstream Provider (Mock/HTTP) -> Agent 4 Re-fusion -> Agent 5
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union
import httpx

from app.config import settings
from app.models.feedback_models import (
    Agent2RetrievalRequirement,
    Agent3RetrievalRequirement,
    FeedbackRequest,
    InsufficientEvidenceTermination,
)
from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import ClinicalDecisionSupportResponse
from app.services.clinical_reasoning import ClinicalReasoningService
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

logger = logging.getLogger("agent5.adaptive_orchestrator")


class OrchestrationFailureError(Exception):
    """Raised when an unrecoverable failure occurs in an upstream agent during orchestration."""

    pass


class AdaptiveOrchestrator:
    """Executes the adaptive feedback loop controlling evidence retrieval and re-fusion cycles."""

    def __init__(
        self,
        reasoning_service: Optional[ClinicalReasoningService] = None,
        max_retries: Optional[int] = None,
        http_client: Optional[httpx.Client] = None,
        providers: Optional[Dict[UpstreamSourceType, UpstreamEvidenceProvider]] = None,
        refusion_service: Optional[MockAgent4RefusionService] = None,
        use_mock_providers_by_default: bool = True,
    ) -> None:
        """Initialize the AdaptiveOrchestrator.

        Args:
            reasoning_service: ClinicalReasoningService instance. Defaults to ClinicalReasoningService().
            max_retries: Maximum allowed adaptive iterations. Defaults to settings.max_retries (3).
            http_client: Optional httpx.Client for external agent HTTP calls.
            providers: Optional dictionary mapping UpstreamSourceType to UpstreamEvidenceProvider instances.
            refusion_service: Optional MockAgent4RefusionService instance for simulated re-fusion.
            use_mock_providers_by_default: If True and providers is None and http_client is None, initialize mock providers.
        """
        self.reasoning_service = reasoning_service or ClinicalReasoningService()
        self.max_retries = max_retries if max_retries is not None else settings.max_retries
        self._http_client = http_client
        self.refusion_service = refusion_service

        if providers is not None:
            self.providers = providers
        elif use_mock_providers_by_default and http_client is None:
            self.providers = {
                UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2: MockAgent2Provider(),
                UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3: MockAgent3Provider(),
            }
            if self.refusion_service is None:
                self.refusion_service = MockAgent4RefusionService()
        else:
            self.providers = {}

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return httpx.Client(timeout=30.0)

    def _call_agent_2_http(self, req: Agent2RetrievalRequirement) -> Optional[Dict[str, Any]]:
        """Invoke Agent 2 via HTTP (legacy/live client mode)."""
        client = self._get_http_client()
        url = f"{settings.agent_2_base_url}/api/v1/retrieve"
        logger.info(f"Invoking Agent 2 at '{url}' for biomedical retrieval...")

        try:
            resp = client.post(url, json=req.model_dump())
            if resp.status_code == 200:
                logger.info("Agent 2 retrieval successful.")
                return resp.json()
            logger.warning(f"Agent 2 returned non-200 status code: {resp.status_code}")
            return None
        except Exception as exc:
            logger.error(f"Agent 2 HTTP invocation error: {str(exc)}")
            return None

    def _call_agent_3_http(self, req: Agent3RetrievalRequirement) -> Optional[Dict[str, Any]]:
        """Invoke Agent 3 via HTTP (legacy/live client mode)."""
        client = self._get_http_client()
        url = f"{settings.agent_3_base_url}/api/v1/search"
        logger.info(f"Invoking Agent 3 at '{url}' for medical evidence search...")

        try:
            resp = client.post(url, json=req.model_dump())
            if resp.status_code == 200:
                logger.info("Agent 3 evidence search successful.")
                return resp.json()
            logger.warning(f"Agent 3 returned non-200 status code: {resp.status_code}")
            return None
        except Exception as exc:
            logger.error(f"Agent 3 HTTP invocation error: {str(exc)}")
            return None

    def _call_agent_4_refusion_http(
        self,
        current_context: UnifiedClinicalContext,
        agent_2_data: Optional[Dict[str, Any]] = None,
        agent_3_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[UnifiedClinicalContext]:
        """Invoke Agent 4 re-fusion via HTTP (legacy/live client mode)."""
        client = self._get_http_client()
        url = f"{settings.agent_4_base_url}/fuse"
        logger.info(f"Invoking Agent 4 at '{url}' for cumulative evidence re-fusion...")

        payload = {
            "current_context": current_context.model_dump(),
            "new_agent_2_evidence": agent_2_data,
            "new_agent_3_evidence": agent_3_data,
        }

        try:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info("Agent 4 cumulative re-fusion successful.")
                return UnifiedClinicalContext.model_validate(resp.json())
            logger.warning(f"Agent 4 returned non-200 status code: {resp.status_code}")
            return None
        except Exception as exc:
            logger.error(f"Agent 4 HTTP re-fusion error: {str(exc)}")
            return None

    def run(
        self,
        initial_context: UnifiedClinicalContext,
        correlation_id: Optional[str] = None,
    ) -> Union[ClinicalDecisionSupportResponse, InsufficientEvidenceTermination]:
        """Execute the external adaptive orchestration loop.

        Args:
            initial_context: Upstream Agent 4 UnifiedClinicalContext model instance.
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            Union[ClinicalDecisionSupportResponse, InsufficientEvidenceTermination]

        Raises:
            OrchestrationFailureError: If an unrecoverable failure occurs in upstream agent communication.
        """
        current_context = initial_context
        retry_count = 0
        cycle_id = correlation_id or str(uuid.uuid4())

        logger.info(f"adaptive_cycle_started (Correlation ID: {cycle_id})")

        while retry_count <= self.max_retries:
            logger.info(
                f"Adaptive Orchestrator iteration {retry_count}/{self.max_retries} executing "
                f"(Correlation ID: {cycle_id})..."
            )

            # Step 1: Send current context to Agent 5 pipeline
            pipeline_result = self.reasoning_service.execute_reasoning_pipeline(
                current_context, retry_count=retry_count
            )

            # Case A: READY_FOR_REASONING -> Clinical reasoning completed
            if isinstance(pipeline_result, ClinicalDecisionSupportResponse):
                logger.info(
                    f"adaptive_cycle_completed on iteration {retry_count}. "
                    f"Model: {pipeline_result.agent_metadata.model}."
                )
                return pipeline_result

            # Case B: Safe termination returned by Agent 5
            if isinstance(pipeline_result, InsufficientEvidenceTermination):
                logger.warning(
                    f"adaptive_cycle_completed safely on iteration {retry_count}: INSUFFICIENT_EVIDENCE."
                )
                return pipeline_result

            # Case C: NEEDS_MORE_EVIDENCE -> Process FeedbackRequest
            if isinstance(pipeline_result, FeedbackRequest):
                logger.info(
                    f"feedback_request_created on iteration {retry_count}. "
                    f"Reason: '{pipeline_result.reason}'. Targets: {pipeline_result.requested_sources}."
                )

                if retry_count >= self.max_retries:
                    logger.warning(
                        f"Retry limit ({self.max_retries}) reached on FeedbackRequest; enforcing safe termination."
                    )
                    return InsufficientEvidenceTermination(
                        status="insufficient_evidence",
                        retry_count=retry_count,
                        max_retries=self.max_retries,
                        confidence_score=current_context.confidence_score,
                        reason=pipeline_result.reason,
                        evidence_gaps=pipeline_result.evidence_gaps,
                        conflicting_evidence_summary=[
                            ce.description for ce in current_context.conflicting_evidence
                        ],
                        termination_message=f"Maximum retries ({self.max_retries}) reached without meeting confidence threshold.",
                    )

                requested_sources = pipeline_result.requested_sources
                req_agent_2 = "agent_2" in requested_sources and pipeline_result.agent_2_request is not None
                req_agent_3 = "agent_3" in requested_sources and pipeline_result.agent_3_request is not None

                # Branch A: UpstreamProvider Abstraction Mode
                if self.providers:
                    collected_evidence: List[UpstreamEvidenceItem] = []

                    if req_agent_2:
                        p2 = self.providers.get(UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2)
                        if p2 is None:
                            logger.error("adaptive_cycle_failed: Requested FUTURE_UPSTREAM_AGENT_2 provider not registered.")
                            raise OrchestrationFailureError("Requested upstream Agent 2 provider not registered.")

                        logger.info("upstream_provider_selected: FUTURE_UPSTREAM_AGENT_2")
                        req_p2 = UpstreamEvidenceRequest(
                            correlation_id=cycle_id,
                            source_type=UpstreamSourceType.FUTURE_UPSTREAM_AGENT_2,
                            query_or_terms=(pipeline_result.agent_2_request.focus_terms or pipeline_result.agent_2_request.query_requirements) if pipeline_result.agent_2_request else [],
                            evidence_gaps=pipeline_result.evidence_gaps,
                            retry_count=retry_count,
                            agent_2_details=pipeline_result.agent_2_request.model_dump() if pipeline_result.agent_2_request else None,
                        )
                        try:
                            resp_p2 = p2.retrieve(req_p2)
                            collected_evidence.extend(resp_p2.evidence_items)
                            logger.info(f"mock_retrieval_completed: {len(resp_p2.evidence_items)} items from Agent 2.")
                        except UpstreamProviderError as exc:
                            logger.error(f"adaptive_cycle_failed: Mock Agent 2 retrieval failed: {str(exc)}")
                            raise OrchestrationFailureError(f"Requested upstream Agent 2 retrieval failed: {str(exc)}") from exc

                    if req_agent_3:
                        p3 = self.providers.get(UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3)
                        if p3 is None:
                            logger.error("adaptive_cycle_failed: Requested FUTURE_UPSTREAM_AGENT_3 provider not registered.")
                            raise OrchestrationFailureError("Requested upstream Agent 3 provider not registered.")

                        logger.info("upstream_provider_selected: FUTURE_UPSTREAM_AGENT_3")
                        req_p3 = UpstreamEvidenceRequest(
                            correlation_id=cycle_id,
                            source_type=UpstreamSourceType.FUTURE_UPSTREAM_AGENT_3,
                            query_or_terms=pipeline_result.agent_3_request.evidence_requirements if pipeline_result.agent_3_request else [],
                            evidence_gaps=pipeline_result.evidence_gaps,
                            retry_count=retry_count,
                            agent_3_details=pipeline_result.agent_3_request.model_dump() if pipeline_result.agent_3_request else None,
                        )
                        try:
                            resp_p3 = p3.retrieve(req_p3)
                            collected_evidence.extend(resp_p3.evidence_items)
                            logger.info(f"mock_retrieval_completed: {len(resp_p3.evidence_items)} items from Agent 3.")
                        except UpstreamProviderError as exc:
                            logger.error(f"adaptive_cycle_failed: Mock Agent 3 search failed: {str(exc)}")
                            raise OrchestrationFailureError(f"Requested upstream Agent 3 search failed: {str(exc)}") from exc

                    refusion_svc = self.refusion_service or MockAgent4RefusionService()
                    logger.info("refusion_started")
                    updated_context = refusion_svc.refuse_context(
                        current_context=current_context,
                        additional_evidence=collected_evidence,
                        correlation_id=cycle_id,
                    )

                    if updated_context is None:
                        logger.error("adaptive_cycle_failed: Agent 4 re-fusion returned None.")
                        raise OrchestrationFailureError("Agent 4 re-fusion failed or returned an invalid payload.")

                    logger.info(f"refusion_completed: context_updated (Confidence: {updated_context.confidence_score:.2f})")
                    current_context = updated_context
                    retry_count += 1
                    logger.info(f"adaptive_retry_incremented: {retry_count}")

                # Branch B: HTTP Client Mode (Legacy/Live Endpoint Invocation)
                else:
                    agent_2_data: Optional[Dict[str, Any]] = None
                    agent_3_data: Optional[Dict[str, Any]] = None

                    if req_agent_2:
                        agent_2_data = self._call_agent_2_http(pipeline_result.agent_2_request)

                    if req_agent_3:
                        agent_3_data = self._call_agent_3_http(pipeline_result.agent_3_request)

                    if req_agent_2 and req_agent_3 and agent_2_data is None and agent_3_data is None:
                        raise OrchestrationFailureError("Both requested upstream agents (Agent 2 & Agent 3) failed to respond.")
                    elif req_agent_2 and agent_2_data is None and not req_agent_3:
                        raise OrchestrationFailureError("Requested upstream Agent 2 retrieval failed.")
                    elif req_agent_3 and agent_3_data is None and not req_agent_2:
                        raise OrchestrationFailureError("Requested upstream Agent 3 retrieval failed.")

                    updated_context = self._call_agent_4_refusion_http(
                        current_context,
                        agent_2_data=agent_2_data,
                        agent_3_data=agent_3_data,
                    )

                    if not updated_context:
                        raise OrchestrationFailureError("Agent 4 re-fusion failed or returned an invalid payload.")

                    current_context = updated_context
                    retry_count += 1
                    logger.info(f"adaptive_retry_incremented: {retry_count}")

        logger.error("adaptive_cycle_failed: Exceeded maximum adaptive iteration boundary.")
        raise OrchestrationFailureError(
            f"Adaptive orchestration loop exceeded maximum iteration boundary ({self.max_retries})."
        )
