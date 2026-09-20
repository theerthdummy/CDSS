"""Tests for Clinical Domain Guardrail and Out-of-Domain Query Rejection."""

import asyncio
import pytest
from orchestrator.clinical_assistant.validation.domain_guardrail import (
    DomainGuardrail,
    OUT_OF_DOMAIN_RESPONSE,
)
from orchestrator.clinical_assistant.memory.hybrid_memory import PatientEntityMemory
from orchestrator.clinical_assistant.assistant import ClinicalConversationalAssistant


def test_distance_query_is_rejected():
    """Verify that geographical distance query is strictly rejected."""
    is_ood, reason = DomainGuardrail.is_out_of_domain("distance between newyork and london")
    assert is_ood is True
    assert "pattern" in reason.lower() or "out-of-domain" in reason.lower()


def test_various_out_of_domain_queries_rejected():
    """Verify that trivia, coding, recipes, and sports queries are rejected."""
    ood_queries = [
        "capital of france",
        "how far is tokyo from paris",
        "write a python script to sort a list",
        "who won the world cup in 2022",
        "what is the recipe for chocolate cake",
        "forecast for new york tomorrow",
        "what is the stock price of apple",
        "who is the prime minister of the uk",
    ]
    for q in ood_queries:
        is_ood, reason = DomainGuardrail.is_out_of_domain(q)
        assert is_ood is True, f"Expected '{q}' to be flagged as out-of-domain, got reason: {reason}"


def test_in_domain_clinical_queries_accepted():
    """Verify that genuine clinical and symptom queries are accepted."""
    in_domain_queries = [
        "Patient has a fever of 102 degrees for 2 days",
        "I have severe chest pain radiating to left arm",
        "My child has a rash and high temperature",
        "What are the side effects of ibuprofen?",
        "Patient presents with headache, stiff neck, and photophobia",
        "Blood pressure is 160/100 and feeling dizzy",
    ]
    for q in in_domain_queries:
        is_ood, reason = DomainGuardrail.is_out_of_domain(q)
        assert is_ood is False, f"Expected '{q}' to be accepted as in-domain, got reason: {reason}"


def test_conversational_turns_in_active_session_accepted():
    """Verify that follow-up answers during an ongoing case are not rejected."""
    from orchestrator.clinical_assistant.memory.hybrid_memory import FactStatus
    memory = PatientEntityMemory(session_id="test_session_guardrail")
    memory.commit_patient_fact("symptom", "fever", FactStatus.CONFIRMED)
    memory.measurements["temperature"] = "102F"

    # Short follow-up replies
    follow_ups = ["no", "neither", "for 2 days", "yes since yesterday", "none"]
    for reply in follow_ups:
        is_ood, reason = DomainGuardrail.is_out_of_domain(reply, memory=memory)
        assert is_ood is False, f"Expected active case reply '{reply}' to be accepted, got: {reason}"


def test_greetings_accepted():
    """Verify greetings are accepted as clinical conversation entry points."""
    greetings = ["hello", "hi", "good morning", "can you help me"]
    for g in greetings:
        is_ood, reason = DomainGuardrail.is_out_of_domain(g)
        assert is_ood is False, f"Expected greeting '{g}' to be accepted"


def test_assistant_chat_rejects_out_of_domain():
    """Verify assistant.chat immediately intercepts distance queries without invoking LLM."""
    async def _run():
        assistant = ClinicalConversationalAssistant()
        res = await assistant.chat("distance between newyork and london")

        assert res["response"] == OUT_OF_DOMAIN_RESPONSE
        assert res["clinical_assessment"]["stage"] == "OUT_OF_DOMAIN"
        assert res["validation"]["action_taken"] == "domain_guardrail_intercepted"
        assert "geography, trivia, math, or coding" in res["response"]

    asyncio.run(_run())
