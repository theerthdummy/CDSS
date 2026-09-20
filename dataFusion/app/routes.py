import time
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Header, Response, status
from fastapi.responses import JSONResponse

from app.adapters import adapt_agent2_output, adapt_agent3_output
from app.config import settings
from app.fusion import (
    fuse_clinical_data,
    preprocess_fusion_request,
    validate_agent2_output,
    validate_agent3_output,
)
from app.schemas import ErrorResponse, FusionMetadata, FusionRequest, FusionResponse
from app.utils import generate_correlation_id, logger

# Versioned router prefix /api/v1
router = APIRouter(prefix="/api/v1")


@router.get("/", responses={200: {"description": "Health check success"}})
def root(
    response: Response,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> dict[str, str]:
    """Basic health check endpoint for the Agent 4 service with correlation tracking."""
    correlation_id = generate_correlation_id(x_correlation_id)
    response.headers[settings.correlation_header] = correlation_id
    logger.info(f"[{correlation_id}] Health check endpoint hit.")
    return {"message": "Agent 4 Clinical Data Fusion API is running"}


@router.post(
    "/fuse",
    response_model=FusionResponse,
    responses={
        200: {"model": FusionResponse, "description": "Successful clinical data fusion (Llama enhanced or rule-based fallback)"},
        400: {"model": ErrorResponse, "description": "Invalid input payload or validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def fuse_context(
    payload: FusionRequest,
    response: Response,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> Response:
    """Receive evidence-oriented outputs from upstream retrieval agents and return a unified clinical context.

    Hybrid Orchestration Sequence:
      Receive Request -> Adapt Upstream Payloads -> Validate Inputs -> Preprocess Inputs -> Baseline Rule-Based Fusion -> Llama 3.1 8B Instruct Enhancement -> Validate LLM Output -> Return Response (or automatic Rule-Based Fallback)
    """
    correlation_id = generate_correlation_id(x_correlation_id)
    response.headers[settings.correlation_header] = correlation_id
    
    logger.info(f"[{correlation_id}] Received clinical data fusion request.")

    # 1. Adapter & Validation Stage
    try:
        logger.info(f"[{correlation_id}] Stage 0: Received clinical data fusion request.")
        
        logger.info(f"[{correlation_id}] Stage 1: Running Agent 2 adapter (RawAgent2Payload -> Agent2Output)...")
        adapted_agent2 = adapt_agent2_output(payload.agent2_output)
        logger.info(
            f"[{correlation_id}] Stage 1: Agent 2 adapter complete. "
            f"RAG Entries: {len(adapted_agent2.rag_evidence)}, KG Entries: {len(adapted_agent2.kg_evidence)}."
        )

        logger.info(f"[{correlation_id}] Stage 2: Running Agent 3 adapter (RawAgent3Payload -> Agent3Output)...")
        adapted_agent3 = adapt_agent3_output(payload.agent3_output)
        logger.info(
            f"[{correlation_id}] Stage 2: Agent 3 adapter complete. "
            f"Medical Evidence Entries: {len(adapted_agent3.medical_evidence)}."
        )

        logger.info(f"[{correlation_id}] Stage 3: Running schema validation on normalized inputs...")
        validate_agent2_output(adapted_agent2)
        validate_agent3_output(adapted_agent3)
        logger.info(f"[{correlation_id}] Stage 3: Input validation SUCCESS.")

        normalized_payload = FusionRequest(
            agent2_output=adapted_agent2,
            agent3_output=adapted_agent3,
        )
    except ValueError as val_err:
        error_msg = f"Input validation failed: {str(val_err)}"
        logger.warning(f"[{correlation_id}] {error_msg}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            headers={settings.correlation_header: correlation_id},
            content={
                "detail": error_msg,
                "correlation_id": correlation_id,
            },
        )
    except Exception as exc:
        error_msg = "Unexpected error during request adapter normalization or validation."
        logger.error(f"[{correlation_id}] {error_msg} details: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={settings.correlation_header: correlation_id},
            content={
                "detail": error_msg,
                "correlation_id": correlation_id,
            },
        )

    # 2. Preprocessing & Hybrid Fusion Stage
    try:
        start_time = time.perf_counter()
        logger.debug(f"[{correlation_id}] Preprocessing fusion inputs...")
        preprocessed_payload = preprocess_fusion_request(normalized_payload)

        fused_context, conflicts, trace_origins, fusion_mode, stats, fallback_reason = fuse_clinical_data(
            preprocessed_payload,
            correlation_id=correlation_id
        )
        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        num_a2 = len(preprocessed_payload.agent2_output.rag_evidence) + len(preprocessed_payload.agent2_output.kg_evidence)
        num_a3 = len(preprocessed_payload.agent3_output.medical_evidence)
        
        logger.info(
            f"[{correlation_id}] Structured Evaluation Metrics: "
            f"correlation_id={correlation_id}, "
            f"agent4_confidence={fused_context.confidence_score}, "
            f"conflict_detected={len(conflicts) > 0}, "
            f"number_of_agent2_findings={num_a2}, "
            f"number_of_agent3_findings={num_a3}, "
            f"number_of_merged_findings={len(fused_context.merged_findings)}, "
            f"number_of_conflicts={len(conflicts)}, "
            f"fallback_used={(fusion_mode == 'rule_based_fallback')}, "
            f"response_type=FusionResponse, "
            f"processing_time_ms={processing_time_ms}"
        )

        metadata = FusionMetadata(
            correlation_id=correlation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=processing_time_ms,
            fusion_mode=fusion_mode,
            fusion_method=fusion_mode,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model_name,
            llm_used=(fusion_mode == "llama_enhanced"),
            fallback_used=(fusion_mode == "rule_based_fallback"),
            fallback_reason=fallback_reason,
            token_usage=stats.get("token_usage") if stats else None,
            latency_ms=stats.get("latency_ms") if stats else None,
            execution_stats=stats or None,
            conflicts_detected=conflicts,
            findings_sources={k: list(v) for k, v in trace_origins.items()},
        )

        success_response = FusionResponse(
            status="success",
            message=f"Clinical data fusion completed successfully via {fusion_mode}.",
            unified_context=fused_context,
            metadata=metadata,
        )
        return success_response

    except Exception as exc:
        error_msg = "An unexpected error occurred during the clinical data fusion process."
        logger.error(f"[{correlation_id}] {error_msg} details: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={settings.correlation_header: correlation_id},
            content={
                "detail": error_msg,
                "correlation_id": correlation_id,
            },
        )