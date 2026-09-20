"""Clinical Reasoning Service Module for Agent 5.

Orchestrates the two-phase reasoning execution:
Phase 1: Adaptive Optimizer evaluation (Completeness, relevance, consistency & confidence thresholds).
Phase 2: If READY_FOR_REASONING, evaluate ReasoningPolicy via ClinicalReasoningPolicyEvaluator,
         then execute multi-level Reasoning Engine cascade (Local Mistral via Ollama -> Deterministic Fallback)
         with mandatory deterministic output validation via ClinicalOutputValidator.
"""

import logging
from typing import Optional, Union

from app.config import settings
from app.models.feedback_models import FeedbackRequest, InsufficientEvidenceTermination
from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import ClinicalDecisionSupportResponse, ReasoningPolicy
from app.services.adaptive_optimizer import AdaptiveOptimizer
from app.services.output_validator import ClinicalOutputValidator
from app.services.policy_evaluator import ClinicalReasoningPolicyEvaluator
from app.services.reasoning_engines import (
    DevelopmentPlaceholderEngine,
    DeterministicFallbackEngine,
    EngineExecutionError,
    GeminiReasoningEngine,
    GroqReasoningEngine,
    OllamaReasoningEngine,
    ReasoningEngine,
)

logger = logging.getLogger("agent5.clinical_reasoning")


class ClinicalReasoningError(Exception):
    """Raised when clinical reasoning fails catastrophically across all engine levels."""

    pass


class ClinicalReasoningService:
    """Service orchestrating context validation, ReasoningPolicy evaluation, and multi-level reasoning cascade."""

    @staticmethod
    def _build_engine_for_provider(provider: str):
        provider_name = provider.lower().strip()
        if provider_name == "openai":
            return GroqReasoningEngine()  # Groq operates as the hosted OpenAI GPT endpoint
        if provider_name == "groq":
            return GroqReasoningEngine()
        if provider_name == "ollama":
            return OllamaReasoningEngine()
        if provider_name == "gemini":
            return GeminiReasoningEngine()
        if provider_name == "deterministic":
            return DeterministicFallbackEngine()
        raise ClinicalReasoningError(f"Unsupported reasoning provider '{provider}'.")

    @staticmethod
    def _engine_source_label(engine: ReasoningEngine) -> str:
        if isinstance(engine, GroqReasoningEngine):
            return "openai_hosted_gpt"
        if isinstance(engine, OllamaReasoningEngine):
            return "ollama_mistral"
        if isinstance(engine, GeminiReasoningEngine):
            return "gemini_primary"
        if isinstance(engine, DeterministicFallbackEngine):
            return "deterministic_fallback"
        return engine.__class__.__name__.lower()

    @staticmethod
    def _engine_model_label(engine: ReasoningEngine) -> str:
        return getattr(engine, "model_name", "deterministic-fallback")

    def __init__(
        self,
        primary_engine: Optional[ReasoningEngine] = None,
        fallback_engine: Optional[ReasoningEngine] = None,
        deterministic_engine: Optional[ReasoningEngine] = None,
        optimizer: Optional[AdaptiveOptimizer] = None,
        validator: Optional[ClinicalOutputValidator] = None,
        use_dev_default_when_unimplemented: bool = False,
    ) -> None:
        if primary_engine is not None:
            self.primary_engine = primary_engine
        elif use_dev_default_when_unimplemented:
            self.primary_engine = DevelopmentPlaceholderEngine()
        else:
            self.primary_engine = self._build_engine_for_provider(settings.primary_reasoning_provider)

        if fallback_engine is not None:
            self.fallback_engine = fallback_engine
        elif use_dev_default_when_unimplemented:
            self.fallback_engine = DevelopmentPlaceholderEngine()
        else:
            self.fallback_engine = self._build_engine_for_provider(settings.fallback_reasoning_provider)

        self.deterministic_engine = deterministic_engine or DeterministicFallbackEngine()
        self.optimizer = optimizer or AdaptiveOptimizer()
        self.validator = validator or ClinicalOutputValidator()

    def execute_reasoning_pipeline(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
        retry_count: int = 0,
    ) -> Union[ClinicalDecisionSupportResponse, FeedbackRequest, InsufficientEvidenceTermination]:
        """Execute complete CDSS reasoning pipeline: AdaptiveOptimizer -> ReasoningPolicy -> LLM Cascade -> OutputValidator."""
        logger.info(
            f"Executing ClinicalReasoningService pipeline for context (confidence: {context.confidence_score}, retry: {retry_count})..."
        )

        # Phase 1: Adaptive Optimizer Evaluation
        decision = self.optimizer.evaluate_context(context, retry_count=retry_count)

        if decision.status == "needs_more_evidence":
            logger.info(
                f"Adaptive Optimizer decision: NEEDS_MORE_EVIDENCE (retry {retry_count}/{self.optimizer.max_retries}). "
                "Returning FeedbackRequest. Clinical reasoning models will NOT be invoked."
            )
            assert decision.feedback_request is not None
            return decision.feedback_request

        if decision.status == "insufficient_evidence":
            logger.info(
                f"Adaptive Optimizer decision: INSUFFICIENT_EVIDENCE (max retries {retry_count} reached). "
                "Safe execution termination. Clinical reasoning models will NOT be invoked."
            )
            assert decision.termination_payload is not None
            return decision.termination_payload

        # Phase 2: Context READY_FOR_REASONING -> Evaluate ReasoningPolicy & Proceed to LLM Reasoning Cascade
        logger.info(
            "Adaptive Optimizer decision: READY_FOR_REASONING. Evaluating ReasoningPolicy & proceeding to reasoning cascade..."
        )
        evaluated_policy = policy or ClinicalReasoningPolicyEvaluator.evaluate_policy(context)
        response = self.execute_reasoning(context, policy=evaluated_policy)

        # Phase 3: Post-Reasoning Confidence Threshold Enforcement
        final_confidence = response.uncertainty.confidence_score
        logger.info(
            f"Clinical reasoning completed. Final confidence: {final_confidence:.2f}, "
            f"Threshold: {self.optimizer.confidence_threshold:.2f}, Retry: {retry_count}/{self.optimizer.max_retries}."
        )

        if final_confidence < self.optimizer.confidence_threshold:
            primary_reason = "conflicting_evidence" if context.conflicting_evidence else "low_confidence"
            evidence_gaps = list(response.uncertainty.uncertainty_factors) if response.uncertainty.uncertainty_factors else [
                f"Clinical reasoning synthesis confidence ({final_confidence:.2f}) is below the required clinical threshold ({self.optimizer.confidence_threshold:.2f})."
            ]

            if retry_count < self.optimizer.max_retries:
                logger.warning(
                    f"Post-reasoning confidence {final_confidence:.2f} < threshold {self.optimizer.confidence_threshold:.2f}. "
                    f"Converting output to FeedbackRequest (retry {retry_count}/{self.optimizer.max_retries})."
                )
                feedback_req = self.optimizer.build_feedback_request(
                    context=context,
                    confidence_score=final_confidence,
                    reason=primary_reason,
                    evidence_gaps=evidence_gaps,
                    retry_count=retry_count,
                )
                logger.info(
                    f"Structured Evaluation Metrics: "
                    f"agent4_confidence={context.confidence_score}, "
                    f"agent5_confidence={final_confidence}, "
                    f"confidence_threshold={self.optimizer.confidence_threshold}, "
                    f"conflict_detected={len(context.conflicting_evidence) > 0}, "
                    f"number_of_merged_findings={len(context.merged_findings)}, "
                    f"number_of_conflicts={len(context.conflicting_evidence)}, "
                    f"retry_count={retry_count}, "
                    f"response_type=FeedbackRequest"
                )
                return feedback_req
            else:
                logger.error(
                    f"Post-reasoning confidence {final_confidence:.2f} < threshold {self.optimizer.confidence_threshold:.2f}. "
                    f"Max retries ({self.optimizer.max_retries}) reached. Enforcing safe termination."
                )
                termination = self.optimizer.build_termination_payload(
                    context=context,
                    confidence_score=final_confidence,
                    reason=primary_reason,
                    evidence_gaps=evidence_gaps,
                    retry_count=retry_count,
                )
                logger.info(
                    f"Structured Evaluation Metrics: "
                    f"agent4_confidence={context.confidence_score}, "
                    f"agent5_confidence={final_confidence}, "
                    f"confidence_threshold={self.optimizer.confidence_threshold}, "
                    f"conflict_detected={len(context.conflicting_evidence) > 0}, "
                    f"number_of_merged_findings={len(context.merged_findings)}, "
                    f"number_of_conflicts={len(context.conflicting_evidence)}, "
                    f"retry_count={retry_count}, "
                    f"response_type=InsufficientEvidenceTermination"
                )
                return termination

        logger.info(
            f"Structured Evaluation Metrics: "
            f"agent4_confidence={context.confidence_score}, "
            f"agent5_confidence={final_confidence}, "
            f"confidence_threshold={self.optimizer.confidence_threshold}, "
            f"conflict_detected={len(context.conflicting_evidence) > 0}, "
            f"number_of_merged_findings={len(context.merged_findings)}, "
            f"number_of_conflicts={len(context.conflicting_evidence)}, "
            f"retry_count={retry_count}, "
            f"response_type=ClinicalDecisionSupportResponse"
        )
        return response

    def execute_reasoning(
        self,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> ClinicalDecisionSupportResponse:
        """Execute reasoning cascade across Level 1 (Local Ollama Mistral) -> Level 2 (Fallback) -> Level 3 (Deterministic)."""
        evaluated_policy = policy or ClinicalReasoningPolicyEvaluator.evaluate_policy(context)
        primary_error: Optional[str] = None
        fallback_error: Optional[str] = None

        logger.info("AGENT5_START")
        logger.info(f"AGENT5_PRIMARY_MODEL: {self._engine_model_label(self.primary_engine)}")
        logger.info(f"AGENT5_FALLBACK_MODEL: {self._engine_model_label(self.fallback_engine)}")

        # Level 1: Primary Reasoning Engine (Local Mistral via Ollama)
        try:
            logger.info(f"Executing Level 1 Primary Reasoning Engine ({self.primary_engine.__class__.__name__})...")
            response = self.primary_engine.reason(context, policy=evaluated_policy)
            self._populate_confidence_breakdown(response, context)
            val_result = self.validator.validate_response(response, context, policy=evaluated_policy)

            if val_result.valid:
                logger.info("Level 1 Primary Engine output validated successfully.")
                logger.info("AGENT5_LLM_SUCCESS")
                logger.info(f"AGENT5_REASONING_SOURCE: {self._engine_source_label(self.primary_engine)}")
                return response
            else:
                err_msg = "; ".join([e.message for e in val_result.errors])
                primary_error = f"Output validation failed: {err_msg}"
                logger.warning(
                    f"Level 1 Primary Engine output failed deterministic validation: {primary_error}. Falling back to Level 2..."
                )
        except EngineExecutionError as primary_err:
            primary_error = str(primary_err)
            logger.warning(
                f"Level 1 Primary Engine model execution failed: {primary_error}. Falling back to Level 2..."
            )
            logger.warning(f"AGENT5_PRIMARY_LLM_FAILED: {primary_error}")
        except Exception as exc:
            primary_error = str(exc)
            logger.error(
                f"Unexpected error in Level 1 Primary Engine: {primary_error}. Falling back to Level 2...",
                exc_info=True,
            )
            logger.warning(f"AGENT5_PRIMARY_LLM_FAILED: {primary_error}")

        # Level 2: Secondary Fallback Model Engine
        try:
            logger.info(f"Executing Level 2 Fallback Model Engine ({self.fallback_engine.__class__.__name__})...")
            response = self.fallback_engine.reason(context, policy=evaluated_policy)
            self._populate_confidence_breakdown(response, context)
            val_result = self.validator.validate_response(response, context, policy=evaluated_policy)

            if val_result.valid:
                logger.info("Level 2 Secondary Engine output validated successfully.")
                logger.info("AGENT5_LLM_SUCCESS")
                logger.info(f"AGENT5_REASONING_SOURCE: {self._engine_source_label(self.fallback_engine)}")
                return response
            else:
                err_msg = "; ".join([e.message for e in val_result.errors])
                fallback_error = f"Output validation failed: {err_msg}"
                logger.warning(
                    f"Level 2 Secondary Engine output failed deterministic validation: {fallback_error}. Falling back to Level 3 (Deterministic Engine)..."
                )
        except EngineExecutionError as fb_err:
            fallback_error = str(fb_err)
            logger.warning(
                f"Level 2 Fallback Model Engine model execution failed: {fallback_error}. Falling back to Level 3 (Deterministic Engine)..."
            )
            logger.warning(f"AGENT5_LLM_FALLBACK_FAILED: {fallback_error}")
        except Exception as exc:
            fallback_error = str(exc)
            logger.error(
                f"Unexpected error in Level 2 Fallback Engine: {fallback_error}. Falling back to Level 3...",
                exc_info=True,
            )
            logger.warning(f"AGENT5_LLM_FALLBACK_FAILED: {fallback_error}")

        # Level 3: Final Safety Mechanism (Deterministic Python Engine)
        try:
            logger.info("Executing Level 3 Deterministic Python Fallback Engine...")
            response = self.deterministic_engine.reason(context, policy=evaluated_policy)
            self._populate_confidence_breakdown(response, context)
            val_result = self.validator.validate_response(response, context, policy=evaluated_policy)

            if val_result.valid:
                logger.info("Level 3 Deterministic Fallback Engine output validated successfully.")
                logger.info(f"AGENT5_REASONING_SOURCE: {self._engine_source_label(self.deterministic_engine)}")
                return response
            else:
                err_msg = "; ".join([e.message for e in val_result.errors])
                logger.critical(f"Deterministic Fallback Engine failed validation: {err_msg}")
                logger.info(f"AGENT5_REASONING_SOURCE: {self._engine_source_label(self.deterministic_engine)}")
                return response
        except Exception as exc:
            sanitized_primary = self._sanitize_error_message(primary_error)
            sanitized_fallback = self._sanitize_error_message(fallback_error)
            logger.critical(f"Level 3 Deterministic Engine execution failed: {self._sanitize_error_message(str(exc))}", exc_info=True)
            raise ClinicalReasoningError(
                f"All reasoning engine levels failed execution. Primary error: '{sanitized_primary}'; Fallback error: '{sanitized_fallback}'"
            ) from exc

    @staticmethod
    def _sanitize_error_message(msg: Optional[str]) -> str:
        """Sanitize error messages to prevent leaking API keys, internal paths, or raw credentials."""
        if not msg:
            return "None"
        cleaned = str(msg)
        if settings.gemini_api_key and settings.gemini_api_key in cleaned:
            cleaned = cleaned.replace(settings.gemini_api_key, "<REDACTED_KEY>")
        if settings.groq_api_key and settings.groq_api_key in cleaned:
            cleaned = cleaned.replace(settings.groq_api_key, "<REDACTED_KEY>")
        import re
        cleaned = re.sub(r'AIzaSy[A-Za-z0-9_-]{30,}', '<REDACTED_KEY>', cleaned)
        cleaned = re.sub(r'gsk_[A-Za-z0-9_-]{30,}', '<REDACTED_KEY>', cleaned)
        cleaned = re.sub(r'[A-Za-z]:\\[^"\'\s:]+', '<REDACTED_PATH>', cleaned)
        return cleaned

    def _populate_confidence_breakdown(
        self,
        response: ClinicalDecisionSupportResponse,
        context: UnifiedClinicalContext,
    ) -> None:
        """Populate fusion_confidence and reasoning_confidence breakdown fields."""
        if response.uncertainty.fusion_confidence is None:
            response.uncertainty.fusion_confidence = context.confidence_score

        if response.uncertainty.reasoning_confidence is None:
            response.uncertainty.reasoning_confidence = response.uncertainty.confidence_score
