"""Pydantic models for clinical text processing."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Agent1Response(BaseModel):
    """Structured clinical extraction output."""
    session_id: Optional[str] = Field(None, description="Session identifier for multi-turn clarification")

    # Demographics
    age: Optional[str] = Field(None, description="Patient age if provided")
    gender: Optional[str] = Field(None, description="Patient gender if provided")

    # Symptoms
    symptoms: List[str] = Field(default_factory=list, description="Confirmed positive symptoms")
    denied_symptoms: List[str] = Field(default_factory=list, description="Explicitly denied symptoms")

    # Measurements
    measurements: dict = Field(default_factory=dict, description="Vitals and measurements, e.g. {'temperature': '102°F'}")
    duration: Optional[str] = Field(None, description="Duration of complaint")
    severity: Optional[str] = Field(None, description="Severity of chief complaint")

    # Medical background
    medical_history: List[str] = Field(default_factory=list, description="Previous diagnoses and conditions")
    medical_history_status: str = Field("UNKNOWN", description="UNKNOWN | NONE_REPORTED | PRESENT")
    medications: List[str] = Field(default_factory=list, description="Current or past medications")
    medications_status: str = Field("UNKNOWN", description="UNKNOWN | NONE_REPORTED | PRESENT")

    # Other clinical context and relations
    other_information: List[str] = Field(default_factory=list, description="Other clinical observations or context")
    relationships: List[str] = Field(default_factory=list, description="Explicit relationships between clinical entities")

    # Clarification
    missing_information: List[str] = Field(default_factory=list, description="Still-missing required fields")
    requires_clarification: bool = Field(False, description="Whether a follow-up question is needed")
    clarification_question: Optional[str] = Field(None, description="Next question to ask the clinician")

    # SNOMED CT
    terminology_mappings: List[dict] = Field(default_factory=list, description="SNOMED CT concept codes for clinical terms")


class Agent1Request(BaseModel):
    """Request to Agent 1 for clinical text processing."""
    text: str = Field(..., description="The clinical text input")
    session_id: Optional[str] = Field(None, description="Optional session ID for multi-turn conversation")
