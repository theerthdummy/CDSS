"""Pydantic input models representing Agent 4's Unified Clinical Context output.

This module defines the local representation of the Agent 4 -> Agent 5 contract.
All field names, types, nested structures, and validation constraints match Agent 4.
"""

from typing import List
from pydantic import BaseModel, Field


class MergedFinding(BaseModel):
    """Represents a consolidated clinical finding extracted across evidence sources."""

    finding: str = Field(..., description="Normalized clinical finding statement.")
    sources: List[str] = Field(..., description="List of evidence sources contributing this finding.")


class SupportingEvidence(BaseModel):
    """Represents a supporting clinical evidence entry with explicit source attribution."""

    finding: str = Field(..., description="Clinical finding or concept statement.")
    source_attribution: str = Field(..., description="Attributed evidence source origin.")


class ConflictEvidence(BaseModel):
    """Represents a contradictory or conflicting finding detected across evidence sources."""

    type: str = Field(..., description="Classification of evidence conflict (e.g., 'Treatment Conflict').")
    finding: str = Field(..., description="Name or summary of the conflicting concept.")
    description: str = Field(..., description="Detailed narrative explaining the opposing statements.")


class EvidencePriority(BaseModel):
    """Represents a clinical finding ranked according to source priority weights."""

    finding: str = Field(..., description="Clinical finding string.")
    priority_score: float = Field(..., description="Calculated priority score based on source weighting.")
    primary_source: str = Field(..., description="Primary originating evidence source.")


class SourceTraceability(BaseModel):
    """Maps a fused finding back to its originating upstream agents and sources."""

    finding: str = Field(..., description="Fused clinical finding string.")
    sources: List[str] = Field(..., description="List of specific evidence source names.")
    upstream_agents: List[str] = Field(..., description="Upstream agent identifiers (e.g., Agent 2, Agent 3).")


class UnifiedClinicalContext(BaseModel):
    """Represents the unified clinical context received from Agent 4."""

    merged_findings: List[MergedFinding] = Field(
        ...,
        description="Consolidated clinical findings extracted and aggregated from all evidence sources.",
    )
    normalized_medical_terms: List[str] = Field(
        ...,
        description="Medical concepts and terms standardized by the clinical data fusion engine.",
    )
    supporting_evidence: List[SupportingEvidence] = Field(
        ...,
        description="Supporting evidence entries with source attribution.",
    )
    conflicting_evidence: List[ConflictEvidence] = Field(
        ...,
        description="Any contradictory or conflicting findings detected across evidence sources.",
    )
    evidence_priority: List[EvidencePriority] = Field(
        ...,
        description="Clinical findings categorized and ordered according to evidence priority.",
    )
    source_traceability: List[SourceTraceability] = Field(
        ...,
        description="Traceability mapping linking each fused finding back to Agent 2 or Agent 3.",
    )
    fusion_summary: str = Field(
        ...,
        description="Concise semantic summary produced by Llama semantic fusion or rule engine fallback.",
    )
    confidence_score: float = Field(
        ...,
        description="Overall confidence score in the fused clinical output.",
        ge=0.0,
        le=1.0,
    )
