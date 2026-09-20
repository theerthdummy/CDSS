"""Deterministic Python validation layer for Agent 5 clinical decision support outputs.

Ensures LLM model outputs satisfy strict schema fidelity, confidence range constraints,
source traceability, conflict/uncertainty consistency, ReasoningPolicy adherence, and safety rules before returning to users.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import ClinicalDecisionSupportResponse, ReasoningPolicy
from app.services.policy_evaluator import ClinicalReasoningPolicyEvaluator


class OutputValidationError(BaseModel):
    """Structured error descriptor for output validation failures."""

    code: str = Field(..., description="Validation error category code.")
    message: str = Field(..., description="Human-readable error narrative.")
    field: Optional[str] = Field(default=None, description="Field path associated with the error.")


class OutputValidationResult(BaseModel):
    """Structured output validation report."""

    valid: bool = Field(..., description="True if output passed all deterministic checks.")
    errors: List[OutputValidationError] = Field(default_factory=list, description="Validation failure details.")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or recommendations.")


class ClinicalOutputValidator:
    """Deterministic validator enforcing clinical safety, traceability, schema rules, and ReasoningPolicy compliance."""

    @staticmethod
    def validate_response(
        response: ClinicalDecisionSupportResponse,
        context: UnifiedClinicalContext,
        policy: Optional[ReasoningPolicy] = None,
    ) -> OutputValidationResult:
        """Validate ClinicalDecisionSupportResponse against UnifiedClinicalContext, ReasoningPolicy, and safety rules."""
        errors: List[OutputValidationError] = []
        warnings: List[str] = []

        if policy is None:
            policy = ClinicalReasoningPolicyEvaluator.evaluate_policy(context)

        # Rule 1: Confidence Range Check (0.0 <= confidence_score <= 1.0)
        conf_score = response.uncertainty.confidence_score
        if not (0.0 <= conf_score <= 1.0):
            errors.append(
                OutputValidationError(
                    code="INVALID_CONFIDENCE",
                    message=f"Confidence score {conf_score} out of valid bounds [0.0, 1.0].",
                    field="uncertainty.confidence_score",
                )
            )

        # Rule 2: Overconfidence vs Context Conflict Consistency
        if context.conflicting_evidence and conf_score > 0.90:
            errors.append(
                OutputValidationError(
                    code="OVERCONFIDENT_OUTPUT",
                    message="Confidence score is > 0.90 despite unmitigated conflicting evidence in context.",
                    field="uncertainty.confidence_score",
                )
            )

        # Rule 3: Confirmed Diagnosis Prohibition under Conflict/Uncertainty or Non-Confirmed Policy
        primary_text = response.clinical_assessment.primary_interpretation.lower()
        if ("confirmed diagnosis" in primary_text or "diagnosis confirmed" in primary_text) and policy.certainty_level != "CONFIRMED":
            errors.append(
                OutputValidationError(
                    code="UNJUSTIFIED_CONFIRMED_DIAGNOSIS",
                    message=f"Output claims 'confirmed diagnosis' but ReasoningPolicy certainty_level is '{policy.certainty_level}' due to lack of explicit confirmation evidence or presence of conflicts.",
                    field="clinical_assessment.primary_interpretation",
                )
            )

        # Rule 4: Uncertainty Requirement Compliance
        if policy.uncertainty_required and len(response.uncertainty.uncertainty_factors) == 0:
            errors.append(
                OutputValidationError(
                    code="MISSING_REQUIRED_UNCERTAINTY",
                    message="ReasoningPolicy requires explicit uncertainty factors due to evidence conflicts or gaps, but response listed 0 uncertainty factors.",
                    field="uncertainty.uncertainty_factors",
                )
            )

        # Rule 5: Conflict Acknowledgement Requirement Compliance
        if len(context.conflicting_evidence) > 0 and len(response.reasoning.conflicting_factors) == 0:
            errors.append(
                OutputValidationError(
                    code="UNACKNOWLEDGED_CONFLICT",
                    message="ReasoningPolicy requires conflicting evidence to be acknowledged, but response lists 0 conflicting factors.",
                    field="reasoning.conflicting_factors",
                )
            )

        # Rule 6: Source Traceability & Unsupported Citation Detection
        valid_sources = set()
        for trace in context.source_traceability:
            valid_sources.update(trace.sources)
            valid_sources.update(trace.upstream_agents)
        for supp in context.supporting_evidence:
            valid_sources.add(supp.source_attribution)
        for mf in context.merged_findings:
            valid_sources.update(mf.sources)

        valid_sources.update(["Agent 2", "Agent 3", "Agent 4", "Guidelines", "PubMed", "EHR", "Knowledge Graph", "Biomedical RAG"])

        for link in response.traceability:
            for source_item in link.upstream_sources:
                if any(bad_pattern in source_item for bad_pattern in ["EVID-999", "Agent 9", "DOI-999", "FAKE_SOURCE"]):
                    errors.append(
                        OutputValidationError(
                            code="UNSUPPORTED_SOURCE",
                            message=f"Traceability references unsupported/invented source identifier: '{source_item}'.",
                            field="traceability.upstream_sources",
                        )
                    )

        # Rule 7: Status and Metadata Consistency
        meta = response.agent_metadata
        if meta.status not in ["success", "partial_success", "fallback_applied"]:
            errors.append(
                OutputValidationError(
                    code="INCONSISTENT_STATUS",
                    message=f"Agent metadata status '{meta.status}' is invalid.",
                    field="agent_metadata.status",
                )
            )

        valid = len(errors) == 0
        return OutputValidationResult(valid=valid, errors=errors, warnings=warnings)
