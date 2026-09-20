"""Pydantic output models representing Agent 5's Clinical Decision Support Response.

These models define the strongly-typed data contract for clinical decision support output.
No reasoning logic or LLM calls are contained within these data models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ClinicalAssessment(BaseModel):
    """Synthesized clinical interpretation and assessed severity."""

    primary_interpretation: str = Field(
        ...,
        description="High-level clinical summary and pathophysiological interpretation (most likely condition).",
    )
    clinical_significance: str = Field(
        ...,
        description="Assessed severity level (e.g. 'Critical', 'High', 'Moderate', 'Low').",
    )
    differential_considerations: List[str] = Field(
        ...,
        description="Plausible differential considerations derived strictly from findings.",
    )


class ClinicalReasoningDetail(BaseModel):
    """Detailed breakdown of key findings, supporting factors, and conflicting factors."""

    key_findings: List[str] = Field(
        ...,
        description="Most critical clinical findings prioritized during reasoning.",
    )
    supporting_factors: List[str] = Field(
        ...,
        description="Clinical evidence supporting the primary interpretation.",
    )
    conflicting_factors: List[str] = Field(
        ...,
        description="Contradictory findings or contraindications evaluated.",
    )
    reasoning_summary: str = Field(
        ...,
        description="Synthesized narrative explaining the reasoning chain.",
    )


class DecisionSupport(BaseModel):
    """Actionable decision support recommendations, required information, and triage priority."""

    recommended_actions: List[str] = Field(
        ...,
        description="Prioritized, actionable clinical next steps.",
    )
    additional_information_needed: List[str] = Field(
        ...,
        description="Missing diagnostic parameters, lab tests, or history required.",
    )
    priority_level: str = Field(
        ...,
        description="Clinical urgency priority ('Urgent', 'Routine', 'Elective').",
    )


class SafetyAnalysis(BaseModel):
    """Safety analysis including flags, contraindications, or clinical concerns."""

    safety_flags: List[str] = Field(
        ...,
        description="Safety warnings or critical alerts derived from context.",
    )
    contraindications_or_concerns: List[str] = Field(
        ...,
        description="Specific contraindications or clinical concerns identified.",
    )


class UncertaintyAssessment(BaseModel):
    """Confidence scoring breakdown and clinical uncertainty factor documentation."""

    confidence_score: float = Field(
        ...,
        description="Overall clinical confidence score bounded in [0.0, 1.0].",
    )
    fusion_confidence: Optional[float] = Field(
        default=None,
        description="Agent 4 UnifiedClinicalContext confidence score.",
    )
    reasoning_confidence: Optional[float] = Field(
        default=None,
        description="Agent 5 clinical reasoning synthesis confidence score.",
    )
    uncertainty_factors: List[str] = Field(
        ...,
        description="Explicit documentation of clinical ambiguity, evidence gaps, or conflicts.",
    )


class TraceabilityLink(BaseModel):
    """Provenance tracking connecting reasoning conclusions to Agent 4 findings."""

    reasoning_item: str = Field(
        ...,
        description="Reasoning conclusion or clinical assessment item.",
    )
    supported_by_findings: List[str] = Field(
        ...,
        description="Agent 4 merged finding descriptions supporting this item.",
    )
    upstream_sources: List[str] = Field(
        ...,
        description="Original upstream source attributions ('Agent 2', 'Agent 3', etc.).",
    )


class AgentMetadata(BaseModel):
    """Metadata regarding Agent 5 execution, model identification, and operational mode."""

    agent: str = Field(
        default="agent_5",
        description="Agent identification identifier.",
    )
    model: str = Field(
        ...,
        description="Model identifier used ('gemini-2.5-flash', 'llama-3.1-8b-instruct', or 'deterministic-fallback').",
    )
    reasoning_mode: str = Field(
        ...,
        description="Execution mode ('primary', 'fallback_model', or 'deterministic_fallback').",
    )
    status: str = Field(
        ...,
        description="Processing status ('success', 'partial_success', 'fallback_applied').",
    )


class ReasoningPolicy(BaseModel):
    """Structured policy parameters guiding LLM clinical reasoning and deterministic validation."""

    policy_version: str = Field(
        default="1.0",
        description="Reasoning policy specification version.",
    )
    reasoning_allowed: bool = Field(
        default=True,
        description="Indicates whether clinical reasoning is permitted based on evidence sufficiency.",
    )
    certainty_level: str = Field(
        default="LIKELY",
        description="Assessed evidence certainty level ('CONFIRMED', 'LIKELY', 'POSSIBLE', 'UNCERTAIN').",
    )
    reasoning_scope: str = Field(
        default="CLINICAL_DECISION_SUPPORT",
        description="Operational boundary ('CLINICAL_DECISION_SUPPORT' vs autonomous medical authority).",
    )
    certainty_policy: str = Field(
        default="Distinguish likely interpretation from confirmed diagnosis. Do not claim confirmed diagnosis unless context explicitly contains confirmation evidence.",
        description="Policy governing diagnostic terminology and confirmation boundaries.",
    )
    evidence_policy: str = Field(
        default="Prioritize Agent 4 ranked evidence. Do not alter Agent 4 evidence priority ranking or invent findings.",
        description="Policy governing evidence usage and Agent 4 priority preservation.",
    )
    conflict_policy: str = Field(
        default="Evaluate conflicting evidence. Express and preserve uncertainty when conflicts are present.",
        description="Policy governing conflicting evidence evaluation and preservation.",
    )
    uncertainty_policy: str = Field(
        default="Identify specific evidence gaps and conflicts rather than vague generic statements.",
        description="Policy governing uncertainty identification and reporting.",
    )
    traceability_policy: str = Field(
        default="All source references must originate strictly from Agent 4 context. Do not invent PMIDs, DOIs, or URLs.",
        description="Policy governing source attribution and citation provenance.",
    )
    follow_up_policy: str = Field(
        default="Identify necessary follow-up considerations supported by context without fabricating unperformed patient tests or labs.",
        description="Policy governing follow-up recommendations and diagnostic boundaries.",
    )
    recommendation_policy: str = Field(
        default="Provide clinical decision support considerations. Do not issue unsupported medication prescriptions, dosages, or emergency procedures.",
        description="Policy governing actionable recommendation safety limits.",
    )
    uncertainty_required: bool = Field(
        default=False,
        description="Indicates whether explicit uncertainty factors are required due to conflicts or evidence gaps.",
    )
    policy_warnings: List[str] = Field(
        default_factory=list,
        description="List of deterministic warnings generated during policy evaluation.",
    )

    # Legacy boolean flags for backward compatibility
    reason_only_from_context: bool = Field(
        default=True,
        description="Strict instruction to reason solely from provided clinical evidence.",
    )
    distinguish_interpretation_from_diagnosis: bool = Field(
        default=True,
        description="Require distinguishing clinical interpretation/most likely condition from confirmed diagnosis.",
    )
    preserve_uncertainty: bool = Field(
        default=True,
        description="Ensure unmitigated evidence conflicts and gaps are expressed in uncertainty output.",
    )
    forbid_unsupported_citations: bool = Field(
        default=True,
        description="Prohibit inventing source identifiers or external citations not present in context.",
    )


class ClinicalDecisionSupportResponse(BaseModel):
    """Top-level structured output payload produced by Agent 5."""

    clinical_assessment: ClinicalAssessment = Field(
        ...,
        description="Clinical interpretation, significance, and differential considerations.",
    )
    reasoning: ClinicalReasoningDetail = Field(
        ...,
        description="Detailed evidence reasoning breakdown and narrative summary.",
    )
    decision_support: DecisionSupport = Field(
        ...,
        description="Actionable next steps, missing information, and triage priority.",
    )
    safety: SafetyAnalysis = Field(
        ...,
        description="Safety warnings and treatment contraindications.",
    )
    uncertainty: UncertaintyAssessment = Field(
        ...,
        description="Confidence scoring and clinical uncertainty factors.",
    )
    traceability: List[TraceabilityLink] = Field(
        ...,
        description="Provenance links linking reasoning items back to Agent 4 findings.",
    )
    agent_metadata: AgentMetadata = Field(
        ...,
        description="Agent execution metadata and operational status.",
    )


class Step3AcknowledgementResponse(BaseModel):
    """Temporary structured acknowledgment response for Step 3 API verification."""

    status: str = Field(
        default="accepted",
        description="Status of context validation.",
    )
    message: str = Field(
        default="Unified clinical context validated successfully.",
        description="Acknowledgment message.",
    )
    agent: str = Field(
        default="agent_5",
        description="Agent identifier.",
    )
