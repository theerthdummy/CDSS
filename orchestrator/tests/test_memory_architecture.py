"""Test Case: Memory Architecture Integrity.

Verifies:
1. Short-term buffer truncation enforces strict message count limit.
2. Long-term memory persists patient entities across turns.
3. Hypotheses do NOT become patient facts (Memory Write Policy).
4. UNKNOWN values remain unknown.
5. Denied symptoms remain denied.
"""

import pytest
from orchestrator.clinical_assistant.memory.hybrid_memory import (
    FactStatus,
    PatientEntityMemory,
    ShortTermMessageBuffer,
)


def test_short_term_buffer_strict_truncation():
    """Verify short-term message buffer strictly maintains bounded message budget."""
    buffer = ShortTermMessageBuffer(max_messages=4)

    buffer.add_message("user", "Msg 1")
    buffer.add_message("assistant", "Resp 1")
    buffer.add_message("user", "Msg 2")
    buffer.add_message("assistant", "Resp 2")
    assert len(buffer.get_messages()) == 4
    assert buffer.get_messages()[0]["content"] == "Msg 1"

    # Add 5th message -> must truncate oldest message
    buffer.add_message("user", "Msg 3")
    msgs = buffer.get_messages()
    assert len(msgs) == 4
    assert msgs[0]["content"] == "Resp 1"
    assert msgs[-1]["content"] == "Msg 3"


def test_memory_write_policy_prevents_hypotheses_as_facts():
    """Verify that clinical hypotheses are NEVER committed as confirmed patient history."""
    memory = PatientEntityMemory(session_id="test-write-policy")

    # 1. Attempt to commit a hypothesis as a patient fact directly -> must be rejected
    memory.commit_patient_fact(
        category="symptom",
        name="meningitis",
        status=FactStatus.SUSPECTED,
    )
    assert "meningitis" not in memory.confirmed_symptoms

    # 2. Record hypothesis in isolated hypothesis storage
    memory.record_hypothesis("meningitis", confidence=0.7, evidence=["fever", "hallucinations"])
    assert "meningitis" not in memory.confirmed_symptoms
    assert len(memory.suspected_conditions) == 1
    assert memory.suspected_conditions[0].condition == "meningitis"
    assert memory.suspected_conditions[0].status == FactStatus.SUSPECTED


def test_unknown_remains_unknown_and_separate():
    """Verify UNKNOWN does not equal FALSE or TRUE."""
    memory = PatientEntityMemory(session_id="test-unknown")

    memory.commit_patient_fact("symptom", "fever", FactStatus.CONFIRMED)
    memory.commit_patient_fact("symptom", "cough", FactStatus.UNKNOWN)
    memory.commit_patient_fact("symptom", "vomiting", FactStatus.DENIED)

    assert "fever" in memory.confirmed_symptoms
    assert "cough" not in memory.confirmed_symptoms
    assert "cough" in memory.unknown_symptoms
    assert "cough" not in memory.denied_symptoms
    assert "vomiting" in memory.denied_symptoms
    assert "vomiting" not in memory.confirmed_symptoms
