"""FastAPI Router for Agent 5 Clinical Reasoning API."""

from typing import Union
from fastapi import APIRouter, HTTPException, Query, status

from app.models.feedback_models import FeedbackRequest, InsufficientEvidenceTermination
from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import ClinicalDecisionSupportResponse
from app.services.clinical_reasoning import ClinicalReasoningError, ClinicalReasoningService

router = APIRouter(prefix="/api/v1")
reasoning_service = ClinicalReasoningService()


@router.post(
    "/reason",
    response_model=Union[ClinicalDecisionSupportResponse, FeedbackRequest, InsufficientEvidenceTermination],
    status_code=status.HTTP_200_OK,
    summary="Process Unified Clinical Context via Adaptive Optimization & Multi-Level Clinical Reasoning",
    description=(
        "Agent 5 Clinical Reasoning Endpoint. Evaluates Agent 4's UnifiedClinicalContext payload via AdaptiveOptimizer. "
        "If evidence is insufficient (retry_count < max_retries), returns a structured FeedbackRequest ('needs_more_evidence'). "
        "If maximum retries are reached, returns safe InsufficientEvidenceTermination ('insufficient_evidence'). "
        "If evidence is sufficient ('ready_for_reasoning'), executes the multi-level clinical reasoning cascade "
        "(Level 1: Local Mistral via Ollama -> Level 2: Fallback -> Level 3: Deterministic Fallback)."
    ),
    responses={
        200: {
            "description": (
                "Successful processing. May return ClinicalDecisionSupportResponse (if context ready), "
                "FeedbackRequest (if evidence insufficient and retries remain), or "
                "InsufficientEvidenceTermination (if max retries reached)."
            ),
        },
        422: {
            "description": "Validation Error - Malformed or schema-invalid request body.",
        },
        500: {
            "description": "Internal Server Error - All clinical reasoning engine levels failed.",
        },
    },
)
def reason_over_clinical_context(
    context: UnifiedClinicalContext,
    retry_count: int = Query(default=0, ge=0, description="Current adaptive retry iteration count."),
) -> Union[ClinicalDecisionSupportResponse, FeedbackRequest, InsufficientEvidenceTermination]:
    """Receive UnifiedClinicalContext and execute adaptive optimization and clinical reasoning pipeline."""
    try:
        return reasoning_service.execute_reasoning_pipeline(context, retry_count=retry_count)
    except ClinicalReasoningError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
