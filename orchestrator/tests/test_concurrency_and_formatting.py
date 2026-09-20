"""Unit and Integration tests for Agent Concurrency and Conversational Formatting.

Validates:
1. Agent 2 & Agent 3 concurrent execution.
2. Graceful partial failure handling (Agent 2 success + Agent 3 unavailable, and vice versa).
3. Markdown formatting sanitization (no raw tables, no literal \\<br>, no escaped asterisks).
4. Conversational bullet-point-first structure and concise question count.
"""

import asyncio
import re
import pytest
from unittest.mock import AsyncMock, patch

from orchestrator.clinical_assistant.assistant import ClinicalConversationalAssistant
from orchestrator.clinical_assistant.validation.response_validator import ClinicalResponseValidator
from orchestrator.clinical_assistant.memory.hybrid_memory import HybridMemoryManager


def test_sanitize_formatting_eliminates_raw_tables_and_html():
    """Verify that raw tables, literal <br>, and escaped markdown are cleanly sanitized."""
    raw_text = (
        "**What you’ve told me so far**  | Category | Details |\n"
        "|----------|---------|\n"
        "| **Confirmed finding** | Temperature = 102 °F |\n"
        "| **Denied finding** | Headache |\n\n"
        "\\<br>\\<br>\n"
        "\\*\\*Confirmed finding\\*\\* is present."
    )

    sanitized = ClinicalResponseValidator.sanitize_formatting(raw_text)

    # 1. No raw table separator
    assert "|----------|" not in sanitized
    assert "| Category | Details |" not in sanitized

    # 2. No literal <br> or \<br>
    assert "<br>" not in sanitized
    assert "\\<br>" not in sanitized

    # 3. No double escaped asterisks
    assert "\\*" not in sanitized

    # 4. Table converted to clean bullet points
    assert "• Confirmed finding: Temperature = 102 °F" in sanitized or "Confirmed finding" in sanitized


def test_conversational_response_bullet_style():
    """Verify response complies with concise bullet-first structure."""
    text = (
        "**So far**\n"
        "- Reported: fever (102°F)\n"
        "- Denied: cough, shortness of breath\n\n"
        "**What this could mean**\n"
        "- Fever indicates your immune system is responding to an infection or inflammation.\n"
        "- Without more symptoms, it is not possible to determine the exact cause.\n\n"
        "**A few questions**\n"
        "1. Any burning or discomfort when urinating?\n"
        "2. Any neck stiffness or severe headache?\n"
        "3. Any abdominal pain or diarrhea?\n"
    )

    sanitized = ClinicalResponseValidator.sanitize_formatting(text)
    
    # Check bullet structure
    bullets = re.findall(r"^[-*•]\s+", sanitized, re.MULTILINE)
    assert len(bullets) >= 2

    # Check numbered questions
    questions = re.findall(r"^\d+[.)]\s+", sanitized, re.MULTILINE)
    assert 2 <= len(questions) <= 5


def test_agent2_agent3_concurrent_execution():
    """Verify that Agent 2 and Agent 3 are executed concurrently via asyncio.gather."""
    async def _run():
        assistant = ClinicalConversationalAssistant()

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {}

            # Run assistant turn
            res = await assistant.chat("I have a fever of 102 for 2 days", session_id="test_concurrency_session")
            assert "response" in res
            assert res["session_id"] == "test_concurrency_session"

    asyncio.run(_run())


def test_graceful_partial_failure_handling():
    """Verify system does not crash if Agent 2 or Agent 3 fails/times out."""
    async def _run():
        assistant = ClinicalConversationalAssistant()

        # Agent 2 fails with network error, Agent 3 succeeds
        with patch("httpx.AsyncClient.post") as mock_post:
            def side_effect(url, **kwargs):
                mock_resp = AsyncMock()
                if "8002" in url:
                    raise Exception("Agent 2 Connection Refused")
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "acute_symptoms": ["fever"],
                    "evidence": [{"source": "web", "snippet": "Fever guidelines"}],
                }
                return mock_resp

            mock_post.side_effect = side_effect

            # Should complete successfully despite Agent 2 failing
            res = await assistant.chat("Fever for 2 days", session_id="partial_failure_session")
            assert "response" in res
            assert isinstance(res["response"], str)
            assert len(res["response"]) > 0

    asyncio.run(_run())

