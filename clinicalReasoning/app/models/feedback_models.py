"""Feedback Pydantic models for Agent 5 Adaptive Evidence Feedback Mechanism."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Agent2RetrievalRequirement(BaseModel):
    """Structured evidence retrieval requirements targeting Agent 2 (Biomedical RAG + Knowledge Graph)."""

    query_requirements: List[str] = Field(
        default_factory=list,
        description="Biomedical knowledge queries to search in graph and vector store.",
    )
    focus_terms: List[str] = Field(
        default_factory=list,
        description="Target medical terms, entities, or concepts requiring biomedical context.",
    )
    knowledge_graph_areas: List[str] = Field(
        default_factory=list,
        description="Specific medical entity types or relationship categories to inspect.",
    )


class Agent3RetrievalRequirement(BaseModel):
    """Structured evidence retrieval requirements targeting Agent 3 (Latest Medical Evidence / PubMed)."""

    evidence_requirements: List[str] = Field(
        default_factory=list,
        description="Clinical guideline or literature search topics.",
    )
    publication_recency: Optional[str] = Field(
        default="Last 5 years",
        description="Desired publication timeframe constraint for literature evidence.",
    )
    conflict_verification_requirements: List[str] = Field(
        default_factory=list,
        description="Specific conflicting findings or treatment contraindications requiring verification.",
    )


class FeedbackRequest(BaseModel):
    """Structured FeedbackRequest produced by Agent 5 when UnifiedClinicalContext is insufficient."""

    status: str = Field(
        default="needs_more_evidence",
        description="Discriminator status indicating additional retrieval is required.",
    )
    retry_count: int = Field(
        ...,
        ge=0,
        description="Current adaptive retrieval iteration count.",
    )
    max_retries: int = Field(
        ...,
        ge=1,
        description="Maximum allowed adaptive retrieval iterations.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Upstream Agent 4 clinical context confidence score evaluated.",
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Target clinical context confidence threshold required for reasoning.",
    )
    reason: str = Field(
        ...,
        description="Derived primary reason (e.g., 'low_confidence', 'conflicting_evidence', 'missing_information').",
    )
    evidence_gaps: List[str] = Field(
        default_factory=list,
        description="Detailed list of specific clinical evidence gaps identified.",
    )
    requested_sources: List[str] = Field(
        default_factory=list,
        description="Target upstream retrieval agents (e.g., ['agent_2', 'agent_3']).",
    )
    agent_2_request: Optional[Agent2RetrievalRequirement] = Field(
        default=None,
        description="Structured retrieval request for Agent 2.",
    )
    agent_3_request: Optional[Agent3RetrievalRequirement] = Field(
        default=None,
        description="Structured retrieval request for Agent 3.",
    )


class InsufficientEvidenceTermination(BaseModel):
    """Safe termination payload returned when maximum adaptive retries are reached without sufficient confidence."""

    status: str = Field(
        default="insufficient_evidence",
        description="Discriminator status indicating safe termination without clinical reasoning.",
    )
    retry_count: int = Field(
        ...,
        description="Final adaptive retry count reached.",
    )
    max_retries: int = Field(
        ...,
        description="Configured maximum retry limit.",
    )
    confidence_score: float = Field(
        ...,
        description="Final context confidence score.",
    )
    reason: str = Field(
        ...,
        description="Final reason for evidence insufficiency.",
    )
    evidence_gaps: List[str] = Field(
        default_factory=list,
        description="Summary of unresolved evidence gaps.",
    )
    conflicting_evidence_summary: List[str] = Field(
        default_factory=list,
        description="Summary of unresolved clinical conflicts.",
    )
    termination_message: str = Field(
        default="Maximum adaptive retrieval retries reached without satisfying minimum confidence threshold.",
        description="Explanation of safe execution termination.",
    )
