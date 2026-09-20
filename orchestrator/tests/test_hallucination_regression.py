"""Test Case: Original Hallucination Bug Regression Test.

Scenario:
Patient: 55 years old
Medical history: No previous medical history reported.
Current symptoms: Fever: 102°F, Duration: 2 days, Hallucinations: yes.
NO other symptoms provided (no headache, neck stiffness, seizures, rash, vomiting, etc.).

Verification:
- The generated response must NOT state or imply that the patient has unreported symptoms.
- It may ask questions about those symptoms or list them as unknown.
- The validator must explicitly detect any fabricated patient symptoms and constrain/correct the response.
"""

import pytest
from orchestrator.clinical_assistant.memory.hybrid_memory import (
    FactStatus,
    PatientEntityMemory,
)
from orchestrator.clinical_assistant.extraction.state_extractor import ClinicalStateExtractor
from orchestrator.clinical_assistant.progressive_reasoning.reasoning_controller import (
    ProgressiveReasoningController,
)
from orchestrator.clinical_assistant.validation.response_validator import (
    ClinicalResponseValidator,
    ValidationResult,
)


def test_state_extractor_no_hallucination():
    """Verify state extractor only extracts reported facts."""
    memory = PatientEntityMemory(session_id="test-hallucination-reg")
    raw_text = (
        "I'm 55 years old with no previous medical history and I've had a fever of "
        "102°F for 2 days. I started hallucinating."
    )

    extracted = ClinicalStateExtractor.extract_from_turn(raw_text, memory)

    # 1. Demographics & Duration
    assert memory.age == "55"
    assert memory.medical_history_status == "NONE_REPORTED"
    assert "temperature" in memory.measurements
    assert "102" in memory.measurements["temperature"]

    # 2. Confirmed symptoms: ONLY fever and hallucinations
    confirmed = set(memory.confirmed_symptoms.keys())
    assert "fever" in confirmed
    assert "hallucinations" in confirmed

    # 3. Must NOT have fabricated headache, neck stiffness, vomiting, rash, seizures
    for unconfirmed in ["headache", "neck stiffness", "vomiting", "rash", "seizures", "cough"]:
        assert unconfirmed not in confirmed, f"Fabrication bug: '{unconfirmed}' should not be confirmed!"

    # 4. Unasked symptoms should be in unknown_symptoms
    assert "headache" in memory.unknown_symptoms
    assert "neck stiffness" in memory.unknown_symptoms


def test_triage_detects_urgent_red_flags():
    """Verify progressive reasoning identifies acute fever + hallucinations as an emergency triage."""
    memory = PatientEntityMemory(session_id="test-hallucination-triage")
    ClinicalStateExtractor.extract_from_turn(
        "I'm 55 years old with no previous medical history and I've had a fever of 102°F for 2 days. I started hallucinating.",
        memory,
    )

    triage = ProgressiveReasoningController.evaluate(memory)

    assert triage.is_urgent is True
    assert triage.priority_level == "Emergency"
    assert "fever + hallucinations" in triage.red_flags_detected or "hallucinations + fever" in triage.red_flags_detected
    assert len(triage.suggested_follow_up_questions) > 0


def test_validator_detects_and_remedies_fabricated_symptoms():
    """Verify the validator catches hallucinated symptoms if an LLM asserts them."""
    memory = PatientEntityMemory(session_id="test-validator-block")
    ClinicalStateExtractor.extract_from_turn(
        "I'm 55 years old with no previous medical history and I've had a fever of 102°F for 2 days. I started hallucinating.",
        memory,
    )

    # Bad hallucinated text (similar to what the unconstrained model produced)
    hallucinated_text = (
        "Your fever, headache, neck stiffness, and vomiting indicate severe meningitis. "
        "Because of your neck stiffness, we need a lumbar puncture."
    )

    result: ValidationResult = ClinicalResponseValidator.validate_response(
        response_text=hallucinated_text,
        memory=memory,
    )

    # Must fail validation and be constrained
    assert result.is_valid is False
    assert result.action_taken == "CONSTRAINED"
    assert len(result.issues) >= 1

    # Check issue categories
    categories = [issue.category for issue in result.issues]
    assert "SYMPTOM_FABRICATION" in categories

    # Verify the sanitized response is safe and does NOT assert headache/neck stiffness as confirmed
    sanitized = result.sanitized_response.lower()
    assert "confirmed **fever, hallucinations**" in sanitized or ("fever" in sanitized and "hallucinations" in sanitized)
    assert "urgent advice" in sanitized
    assert "emergency" in sanitized
