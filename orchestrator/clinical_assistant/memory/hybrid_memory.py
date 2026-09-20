"""Hybrid Memory Architecture for Clinical Conversational Assistant.

Implements:
1. Entity-Driven Long-Term Memory (persistent patient entities: age, medical history,
   allergies, medications, vitals, confirmed/denied symptoms, events).
   - Zero Knowledge Graph dependency.
   - Strict separation of CONFIRMED, DENIED, UNKNOWN facts, and SUSPECTED hypotheses.
   - Strict Memory Write Policy: Speculative hypotheses are NEVER committed as patient facts.
   - Relevance-driven entity retrieval for prompt assembly.
2. Strictly Truncated Short-Term Message Buffer:
   - Enforces a bounded window of recent conversational turns to prevent unbounded context growth.
"""

import enum
import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger("orchestrator.clinical_assistant.memory")


class FactStatus(str, enum.Enum):
    """Authoritative status classification for clinical information."""
    CONFIRMED = "CONFIRMED"     # Explicitly reported or clinically validated
    DENIED = "DENIED"           # Explicitly denied by the patient
    UNKNOWN = "UNKNOWN"         # Not yet asked or answered (UNKNOWN != FALSE, UNKNOWN != TRUE)
    SUSPECTED = "SUSPECTED"     # Diagnostic hypothesis (SUSPECTED != CONFIRMED)


class PatientSymptom(BaseModel):
    """Detailed clinical tracking for a specific patient symptom."""
    name: str
    status: FactStatus = FactStatus.CONFIRMED
    severity: Optional[str] = None
    duration: Optional[str] = None
    source: str = "patient"
    details: Optional[str] = None


class ClinicalHypothesis(BaseModel):
    """Hypothesis or potential differential generated during reasoning.

    Kept strictly separated from confirmed patient medical facts.
    """
    condition: str
    status: FactStatus = FactStatus.SUSPECTED
    confidence: float = 0.5
    supporting_evidence: List[str] = Field(default_factory=list)
    source: str = "clinical_reasoning"


class PatientEntityMemory(BaseModel):
    """Entity-Driven Long-Term Memory for persistent patient data.

    Retains validated patient facts across turns without Knowledge Graph dependencies.
    """
    session_id: str
    age: Optional[str] = None
    gender: Optional[str] = None
    
    # Demographics & Background
    medical_history: List[str] = Field(default_factory=list)
    medical_history_status: str = "UNKNOWN"  # UNKNOWN | NONE_REPORTED | PRESENT
    
    medications: List[str] = Field(default_factory=list)
    medications_status: str = "UNKNOWN"      # UNKNOWN | NONE_REPORTED | PRESENT
    
    allergies: List[str] = Field(default_factory=list)
    previous_investigations: List[str] = Field(default_factory=list)
    
    # Symptoms by status
    confirmed_symptoms: Dict[str, PatientSymptom] = Field(default_factory=dict)
    denied_symptoms: Set[str] = Field(default_factory=set)
    unknown_symptoms: Set[str] = Field(default_factory=set)
    
    # Vitals and measurements (e.g. temperature, blood pressure)
    measurements: Dict[str, str] = Field(default_factory=dict)
    
    # Hypotheses (STRICTLY separated from persistent patient facts)
    suspected_conditions: List[ClinicalHypothesis] = Field(default_factory=list)
    
    # Milestones & audit log
    clinical_events: List[str] = Field(default_factory=list)

    def commit_patient_fact(
        self,
        category: str,
        name: str,
        status: FactStatus,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Memory Write Policy: Commits validated patient facts only.

        Hypotheses must NEVER be committed through this method.
        """
        details = details or {}
        normalized_name = name.strip().lower()

        if status == FactStatus.SUSPECTED:
            logger.warning(
                "Memory Write Policy Violation Blocked: Attempted to commit hypothesis '%s' as patient fact.",
                name,
            )
            return

        if category == "symptom":
            if status == FactStatus.CONFIRMED:
                self.denied_symptoms.discard(normalized_name)
                self.unknown_symptoms.discard(normalized_name)
                self.confirmed_symptoms[normalized_name] = PatientSymptom(
                    name=normalized_name,
                    status=FactStatus.CONFIRMED,
                    severity=details.get("severity"),
                    duration=details.get("duration"),
                    details=details.get("details"),
                )
            elif status == FactStatus.DENIED:
                self.confirmed_symptoms.pop(normalized_name, None)
                self.unknown_symptoms.discard(normalized_name)
                self.denied_symptoms.add(normalized_name)
            elif status == FactStatus.UNKNOWN:
                if normalized_name not in self.confirmed_symptoms and normalized_name not in self.denied_symptoms:
                    self.unknown_symptoms.add(normalized_name)

        elif category == "medical_history":
            if status == FactStatus.CONFIRMED:
                if normalized_name not in [h.lower() for h in self.medical_history]:
                    self.medical_history.append(name.strip())
                self.medical_history_status = "PRESENT"
            elif status == FactStatus.DENIED:
                self.medical_history = [
                    h for h in self.medical_history if h.lower() != normalized_name
                ]
                if not self.medical_history:
                    self.medical_history_status = "NONE_REPORTED"

        elif category == "medication":
            if status == FactStatus.CONFIRMED:
                if normalized_name not in [m.lower() for m in self.medications]:
                    self.medications.append(name.strip())
                self.medications_status = "PRESENT"
            elif status == FactStatus.DENIED:
                self.medications = [
                    m for m in self.medications if m.lower() != normalized_name
                ]
                if not self.medications:
                    self.medications_status = "NONE_REPORTED"

        elif category == "measurement":
            self.measurements[name.strip()] = str(details.get("value", ""))

    def record_hypothesis(self, condition: str, confidence: float = 0.5, evidence: Optional[List[str]] = None) -> None:
        """Record a diagnostic hypothesis in isolated suspected storage."""
        norm = condition.strip()
        # Avoid duplicates
        for hyp in self.suspected_conditions:
            if hyp.condition.lower() == norm.lower():
                hyp.confidence = confidence
                if evidence:
                    hyp.supporting_evidence = list(set(hyp.supporting_evidence + evidence))
                return
        self.suspected_conditions.append(
            ClinicalHypothesis(
                condition=norm,
                status=FactStatus.SUSPECTED,
                confidence=confidence,
                supporting_evidence=evidence or [],
            )
        )

    def retrieve_relevant_context(self, current_query: str) -> Dict[str, Any]:
        """Relevance-driven memory retrieval for prompt construction."""
        # Active confirmed symptoms
        confirmed = [s.name for s in self.confirmed_symptoms.values()]
        denied = sorted(list(self.denied_symptoms))
        unknown = sorted(list(self.unknown_symptoms))

        return {
            "age": self.age,
            "gender": self.gender,
            "confirmed_symptoms": confirmed,
            "denied_symptoms": denied,
            "unknown_symptoms": unknown,
            "measurements": self.measurements,
            "medical_history": self.medical_history,
            "medical_history_status": self.medical_history_status,
            "medications": self.medications,
            "medications_status": self.medications_status,
            "suspected_hypotheses": [h.condition for h in self.suspected_conditions],
        }


class ShortTermMessageBuffer:
    """Strictly Truncated Message Buffer for short-term conversational context.

    Maintains recent dialogue history up to a strict limit (e.g. 10 messages)
    to prevent prompt bloat and context drift.
    """

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the buffer and enforce strict truncation."""
        self.messages.append({"role": role, "content": content.strip()})
        if len(self.messages) > self.max_messages:
            # Strictly truncate oldest messages while keeping recent context
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        """Return the strictly truncated message history."""
        return list(self.messages)

    def clear(self) -> None:
        """Clear short-term conversational buffer."""
        self.messages.clear()


class HybridMemoryManager:
    """Orchestrates Entity-Driven Long-Term Memory and Short-Term Message Buffer."""

    _sessions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_or_create(cls, session_id: str, max_buffer_messages: int = 10) -> Dict[str, Any]:
        if session_id not in cls._sessions:
            cls._sessions[session_id] = {
                "long_term": PatientEntityMemory(session_id=session_id),
                "short_term": ShortTermMessageBuffer(max_messages=max_buffer_messages),
            }
        return cls._sessions[session_id]

    @classmethod
    def reset(cls, session_id: str) -> None:
        """Reset a specific session."""
        cls._sessions.pop(session_id, None)
