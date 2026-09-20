"""Test Case: Multi-Turn Conversation and Patient State Evolution.

Sequence:
Turn 1: "I have a fever."
Turn 2: "It has been 2 days."
Turn 3: "My temperature is 102°F."
Turn 4: "I started hallucinating."
Turn 5: "No headache."
Turn 6: "No neck stiffness."

Verification:
- The system correctly updates and accumulates state turn-by-turn.
- State at end of Turn 6:
  * fever = CONFIRMED
  * hallucinations = CONFIRMED
  * headache = DENIED
  * neck stiffness = DENIED
- If any future model response attempts to assert headache or neck stiffness,
  the validator flags and blocks it as a contradiction.
"""

import pytest
from orchestrator.clinical_assistant.memory.hybrid_memory import (
    FactStatus,
    HybridMemoryManager,
    PatientEntityMemory,
)
from orchestrator.clinical_assistant.extraction.state_extractor import ClinicalStateExtractor
from orchestrator.clinical_assistant.validation.response_validator import ClinicalResponseValidator


def test_six_turn_conversational_state_evolution():
    session_id = "test-session-multiturn-evolution"
    HybridMemoryManager.reset(session_id)
    session = HybridMemoryManager.get_or_create(session_id)
    memory: PatientEntityMemory = session["long_term"]

    # Turn 1: "I have a fever."
    ClinicalStateExtractor.extract_from_turn("I have a fever.", memory)
    assert "fever" in memory.confirmed_symptoms
    assert memory.confirmed_symptoms["fever"].status == FactStatus.CONFIRMED

    # Turn 2: "It has been 2 days."
    ClinicalStateExtractor.extract_from_turn("It has been 2 days.", memory)
    # fever persists
    assert "fever" in memory.confirmed_symptoms

    # Turn 3: "My temperature is 102°F."
    ClinicalStateExtractor.extract_from_turn("My temperature is 102°F.", memory)
    assert "temperature" in memory.measurements
    assert "102" in memory.measurements["temperature"]

    # Turn 4: "I started hallucinating."
    ClinicalStateExtractor.extract_from_turn("I started hallucinating.", memory)
    assert "hallucinations" in memory.confirmed_symptoms
    assert "fever" in memory.confirmed_symptoms

    # Turn 5: "No headache."
    ClinicalStateExtractor.extract_from_turn("No headache.", memory)
    assert "headache" in memory.denied_symptoms
    assert "headache" not in memory.confirmed_symptoms

    # Turn 6: "No neck stiffness."
    ClinicalStateExtractor.extract_from_turn("No neck stiffness.", memory)
    assert "neck stiffness" in memory.denied_symptoms
    assert "neck stiffness" not in memory.confirmed_symptoms

    # Verify Final Authoritative State
    assert "fever" in memory.confirmed_symptoms
    assert "hallucinations" in memory.confirmed_symptoms
    assert "headache" in memory.denied_symptoms
    assert "neck stiffness" in memory.denied_symptoms

    # Contradiction Test: If model says patient has headache after denial, validator catches it
    contradictory_response = "Experiencing your headache alongside fever requires immediate pain relief."
    val_result = ClinicalResponseValidator.validate_response(contradictory_response, memory)
    assert val_result.is_valid is False
    categories = [i.category for i in val_result.issues]
    assert "CONTRADICTION_WITH_DENIED_SYMPTOM" in categories
