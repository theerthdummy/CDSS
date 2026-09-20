"""Clinical Reasoning Policy Evaluator Module for Agent 5.

Provides deterministic policy evaluation over validated UnifiedClinicalContext payloads.
Controls reasoning scope, certainty language boundaries, conflict preservation, evidence provenance,
and safety limits before invoking LLM reasoning engines.

Crucial Rule: PolicyEvaluator does NOT override AdaptiveOptimizer (cannot turn NEEDS_MORE_EVIDENCE
into READY_FOR_REASONING). High confidence_score permits reasoning but does NOT equal diagnostic confirmation.
"""

import logging
from typing import List

from app.models.input_models import UnifiedClinicalContext
from app.models.output_models import ReasoningPolicy

logger = logging.getLogger("agent5.policy_evaluator")

REASONING_POLICY_VERSION = "1.0"

# Explicit confirmation keywords required to justify "CONFIRMED" certainty
EXPLICIT_CONFIRMATION_KEYWORDS = [
    "confirmed by",
    "biopsy confirmed",
    "laboratory confirmed",
    "pathology confirmed",
    "confirmed diagnosis",
    "histologically confirmed",
    "serologically confirmed",
    "microbiologically confirmed",
    "radiologically confirmed stemi",
]


class ClinicalReasoningPolicyEvaluator:
    """Deterministic policy evaluator establishing reasoning constraints for Agent 5."""

    @staticmethod
    def evaluate_policy(context: UnifiedClinicalContext) -> ReasoningPolicy:
        """Evaluate input context and construct a deterministic ReasoningPolicy instance.

        Args:
            context: Validated UnifiedClinicalContext payload from Agent 4.

        Returns:
            ReasoningPolicy: Structured policy parameters guiding LLM reasoning & validation.
        """
        policy_warnings: List[str] = []

        # 1. Inspect context text for explicit confirmation evidence
        text_corpus = " ".join(
            [mf.finding for mf in context.merged_findings]
            + [se.finding for se in context.supporting_evidence]
            + [context.fusion_summary]
        ).lower()

        has_explicit_confirmation = any(
            kw in text_corpus for kw in EXPLICIT_CONFIRMATION_KEYWORDS
        )

        has_conflicts = len(context.conflicting_evidence) > 0

        # 2. Determine Certainty Level
        # Part 5 Rule: High confidence alone does NOT equal "CONFIRMED"
        if has_explicit_confirmation and context.confidence_score >= 0.85 and not has_conflicts:
            certainty_level = "CONFIRMED"
        elif context.confidence_score >= 0.70 and not has_conflicts:
            certainty_level = "LIKELY"
        elif context.confidence_score >= 0.70 and has_conflicts:
            certainty_level = "POSSIBLE"
            policy_warnings.append(
                "Conflicting evidence detected in input context. Certainty restricted to POSSIBLE/LIKELY with uncertainty."
            )
        else:
            certainty_level = "UNCERTAIN"
            policy_warnings.append(
                "Moderate or low context confidence score. High uncertainty preservation required."
            )

        # 3. Determine Uncertainty Requirement
        uncertainty_required = has_conflicts or context.confidence_score < 0.70
        if has_conflicts:
            policy_warnings.append(
                "Conflicting evidence present; reasoning engine must explicitly address conflicting factors."
            )

        policy = ReasoningPolicy(
            policy_version=REASONING_POLICY_VERSION,
            reasoning_allowed=True,
            certainty_level=certainty_level,
            reasoning_scope="CLINICAL_DECISION_SUPPORT",
            certainty_policy=(
                "Distinguish likely interpretation from confirmed diagnosis. "
                "Do NOT claim 'confirmed diagnosis' unless the context explicitly contains confirmation evidence."
            ),
            evidence_policy=(
                "Prioritize Agent 4 ranked evidence. "
                "Do NOT alter Agent 4 evidence priority ranking or invent unstated findings."
            ),
            conflict_policy=(
                "Evaluate conflicting evidence thoroughly. "
                "Express and preserve uncertainty when contradictions are present."
            ),
            uncertainty_policy=(
                "Identify specific evidence gaps and conflicts rather than vague generic statements."
            ),
            traceability_policy=(
                "All source references must originate strictly from Agent 4 context. "
                "Do NOT invent PMIDs, DOIs, URLs, or external study citations."
            ),
            follow_up_policy=(
                "Identify necessary follow-up considerations supported by context without fabricating unperformed patient tests or labs."
            ),
            recommendation_policy=(
                "Provide clinical decision support considerations. "
                "Do NOT issue unsupported medication prescriptions, dosages, or emergency procedures."
            ),
            uncertainty_required=uncertainty_required,
            policy_warnings=policy_warnings,
            # Legacy boolean flags for backward compatibility
            reason_only_from_context=True,
            distinguish_interpretation_from_diagnosis=True,
            preserve_uncertainty=True,
            forbid_unsupported_citations=True,
        )

        logger.info(
            f"Evaluated ReasoningPolicy (v{REASONING_POLICY_VERSION}): certainty_level='{certainty_level}', "
            f"uncertainty_required={uncertainty_required}, warnings={len(policy_warnings)}"
        )
        return policy
