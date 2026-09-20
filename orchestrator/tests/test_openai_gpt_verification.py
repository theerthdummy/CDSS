"""Test Case: OpenAI GPT Model Integration and Verification.

Verifies:
1. The primary conversational assistant explicitly uses LLMProvider.OPENAI.
2. OpenAIGPTClient initializes the official openai SDK client.
3. Runtime logging explicitly records:
   - LLM Provider: OpenAI
   - LLM Model: <configured GPT model>
   - Purpose: Clinical Conversational Assistant
4. Verification that the API call executes through the OpenAI client.
"""

import logging
import pytest
from unittest.mock import MagicMock, patch

from orchestrator.clinical_assistant.llm_client import (
    LLMProvider,
    OpenAIGPTClient,
)
import orchestrator.config as config


def test_openai_provider_and_model_configuration():
    """Verify provider abstraction and configured model."""
    client = OpenAIGPTClient(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        model_name=config.OPENAI_MODEL,
    )

    assert client.provider == LLMProvider.OPENAI
    assert client.model_name in ["openai/gpt-oss-120b", "gpt-4o", "gpt-4o-mini"]
    assert client.is_available is True


def test_openai_gpt_observable_logging(caplog):
    """Verify that every generation call emits mandatory observable logs."""
    client = OpenAIGPTClient(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o",
    )

    # Mock the internal OpenAI client
    mock_openai_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Test response."
    mock_completion = MagicMock(choices=[mock_choice], model="gpt-4o")
    mock_openai_client.chat.completions.create.return_value = mock_completion

    client._client = mock_openai_client

    with caplog.at_level(logging.INFO):
        result = client.generate(
            messages=[{"role": "user", "content": "Hello"}]
        )

    assert result["content"] == "Test response."
    assert result["provider"] == "OpenAI"
    assert result["model"] == "gpt-4o"

    # Verify observable logging
    log_text = caplog.text
    assert "LLM Provider: OpenAI" in log_text
    assert "LLM Model: gpt-4o" in log_text
    assert "Purpose: Clinical Conversational Assistant" in log_text


@pytest.mark.live
def test_live_openai_gpt_hosted_call():
    """Live integration test verifying the hosted OpenAI GPT endpoint is accessible."""
    if not config.OPENAI_API_KEY:
        pytest.skip("No OPENAI_API_KEY configured for live test.")

    client = OpenAIGPTClient(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        model_name=config.OPENAI_MODEL,
    )

    result = client.generate(
        messages=[{"role": "user", "content": "Respond with the single word: VERIFIED."}],
        max_tokens=50,
    )

    assert result["status"] == "success"
    assert result["provider"] == "OpenAI"
    assert len(result["content"]) > 0
