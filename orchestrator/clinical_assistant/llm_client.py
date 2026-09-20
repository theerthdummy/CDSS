"""LLM Client Abstraction for Clinical Conversational Assistant.

Strictly manages model provider selection, explicit observable logging, and
asynchronous non-blocking integration with the hosted OpenAI GPT model via
the official openai SDK (supporting both AsyncOpenAI and OpenAI).
"""

import enum
import logging
import os
import time
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger("orchestrator.clinical_assistant.llm_client")


class LLMProvider(str, enum.Enum):
    """Supported LLM Providers for the Clinical Assistant."""
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
    DETERMINISTIC = "Deterministic"


class OpenAIGPTClient:
    """Client for the hosted OpenAI GPT model using the official OpenAI SDK.

    Provides explicit routing to OpenAI, observable runtime logging, and
    both asynchronous and synchronous invocation methods without blocking the event loop.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.provider = LLMProvider.OPENAI
        
        # Priority resolution for API Key and Base URL
        env_openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        env_groq_key = os.getenv("GROQ_API_KEY", "").strip()
        
        if api_key:
            self.api_key = api_key
        elif env_openai_key:
            self.api_key = env_openai_key
        elif env_groq_key:
            self.api_key = env_groq_key
        else:
            self.api_key = ""

        # Determine if key is a direct OpenAI key or hosted OpenAI endpoint (e.g. Groq)
        if base_url:
            self.base_url = base_url
        elif os.getenv("OPENAI_BASE_URL"):
            self.base_url = os.getenv("OPENAI_BASE_URL")
        elif self.api_key.startswith("gsk_") or not env_openai_key:
            self.base_url = "https://api.groq.com/openai/v1"
        else:
            self.base_url = "https://api.openai.com/v1"

        # Model configuration
        configured_model = os.getenv("OPENAI_MODEL") or os.getenv("GROQ_MODEL")
        if model_name:
            self.model_name = model_name
        elif configured_model:
            self.model_name = configured_model
        elif "groq.com" in self.base_url:
            self.model_name = "openai/gpt-oss-120b"
        else:
            self.model_name = "gpt-4o"

        self.timeout = timeout
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

        if self.api_key:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    @property
    def is_available(self) -> bool:
        """Check if client is configured with credentials."""
        return bool(self._client and self.api_key)

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = 1200,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously execute chat completion with mandatory observable logging."""
        t_start = time.perf_counter()
        logger.info("[TIMESTAMP] GPT request START: %.4f", t_start)
        logger.info("=" * 60)
        logger.info("LLM Provider: %s", self.provider.value)
        logger.info("LLM Model: %s", self.model_name)
        logger.info("Purpose: Clinical Conversational Assistant")
        logger.info("Endpoint Base: %s", self.base_url)
        logger.info("Mode: ASYNCHRONOUS NON-BLOCKING")
        logger.info("=" * 60)

        if not self.is_available or not self._async_client:
            raise RuntimeError(
                "OpenAI GPT Client not configured with an API key. "
                "Please configure OPENAI_API_KEY or GROQ_API_KEY."
            )

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        try:
            completion = await self._async_client.chat.completions.create(**kwargs)
            t_end = time.perf_counter()
            logger.info("[TIMESTAMP] GPT response RECEIVED: %.4f (duration: %.3fs)", t_end, t_end - t_start)
            message = completion.choices[0].message
            content = message.content or ""
            return {
                "content": content,
                "provider": self.provider.value,
                "model": completion.model or self.model_name,
                "usage": getattr(completion, "usage", None),
                "status": "success",
                "duration_seconds": t_end - t_start,
            }
        except Exception as exc:
            logger.error(
                "OpenAI GPT async model invocation failed on model '%s': %s",
                self.model_name,
                exc,
                exc_info=True,
            )
            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = 1200,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronously execute chat completion (kept for testing/scripts)."""
        t_start = time.perf_counter()
        logger.info("[TIMESTAMP] GPT request START: %.4f", t_start)
        logger.info("=" * 60)
        logger.info("LLM Provider: %s", self.provider.value)
        logger.info("LLM Model: %s", self.model_name)
        logger.info("Purpose: Clinical Conversational Assistant")
        logger.info("Endpoint Base: %s", self.base_url)
        logger.info("Mode: SYNCHRONOUS")
        logger.info("=" * 60)

        if not self.is_available or not self._client:
            raise RuntimeError(
                "OpenAI GPT Client not configured with an API key. "
                "Please configure OPENAI_API_KEY or GROQ_API_KEY."
            )

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        try:
            completion = self._client.chat.completions.create(**kwargs)
            t_end = time.perf_counter()
            logger.info("[TIMESTAMP] GPT response RECEIVED: %.4f (duration: %.3fs)", t_end, t_end - t_start)
            message = completion.choices[0].message
            content = message.content or ""
            return {
                "content": content,
                "provider": self.provider.value,
                "model": completion.model or self.model_name,
                "usage": getattr(completion, "usage", None),
                "status": "success",
                "duration_seconds": t_end - t_start,
            }
        except Exception as exc:
            logger.error(
                "OpenAI GPT model invocation failed on model '%s': %s",
                self.model_name,
                exc,
                exc_info=True,
            )
            raise
