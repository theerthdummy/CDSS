"""Authoritative Configuration Management Module for Agent 5.

Centralizes environment configuration, settings validation, safety masking,
and default baseline values for Agent 5 Clinical Reasoning System.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Automatically load environment variables from local .env file if present
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger("agent5.config")

ALLOWED_PROVIDERS = {"gemini", "groq", "ollama", "deterministic", "openai"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
ALLOWED_ENVIRONMENTS = {"development", "test", "production"}


class Settings(BaseModel):
    """Pydantic-validated Authoritative Application Settings for Agent 5."""

    # System & Environment Settings
    app_env: str = Field(default="development", description="Application execution environment.")
    log_level: str = Field(default="INFO", description="Logging output verbosity level.")

    # API Host & Port Settings
    api_host: str = Field(default="0.0.0.0", description="API server bind host.")
    api_port: int = Field(default=8005, description="API server bind port.")

    # OpenAI Settings (Hosted or Direct OpenAI GPT Integration)
    openai_api_key: str = Field(default="", description="OpenAI API authentication key.")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model identifier.")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL.")
    openai_timeout_seconds: float = Field(default=30.0, description="OpenAI API HTTP timeout in seconds.")

    # Gemini Settings (kept for compatibility with non-orchestration paths)
    gemini_api_key: str = Field(default="", description="Google Gemini API authentication key.")
    gemini_model: str = Field(default="gemini-3.5-flash", description="Gemini model identifier.")
    gemini_timeout_seconds: float = Field(default=15.0, description="Gemini API HTTP timeout in seconds.")

    # Ollama Local Provider Settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Local Ollama base URL.")
    ollama_model: str = Field(default="mistral:latest", description="Local Ollama model identifier.")
    ollama_timeout_seconds: float = Field(default=120.0, description="Ollama HTTP timeout in seconds.")

    # Groq Fallback Provider Settings (Cloud API, no local deps)
    groq_api_key: str = Field(default="", description="Groq Cloud API authentication key.")
    groq_api_key_pool: list = Field(default_factory=list, description="Pool of Groq API keys for round-robin fallback.")
    groq_model: str = Field(default="gpt-oss-120b", description="Groq model identifier.")
    groq_timeout_seconds: float = Field(default=15.0, description="Groq API HTTP timeout in seconds.")

    # Provider & Model Selection
    primary_reasoning_provider: str = Field(default="groq", description="Primary reasoning engine provider.")
    primary_reasoning_model: str = Field(default="openai/gpt-oss-120b", description="Primary reasoning model name.")
    fallback_reasoning_provider: str = Field(default="deterministic", description="Fallback reasoning engine provider.")
    fallback_reasoning_model: str = Field(default="deterministic-fallback", description="Fallback reasoning model name.")

    # Adaptive Optimizer Settings
    confidence_threshold: float = Field(
        default=0.70,
        description="Development baseline confidence score threshold for evidence sufficiency.",
    )
    max_adaptive_retries: int = Field(
        default=3,
        description="Maximum allowed adaptive evidence feedback iterations.",
    )

    # Upstream Multi-Agent System URLs & Timeouts
    upstream_timeout_seconds: float = Field(default=30.0, description="Upstream agent HTTP request timeout.")
    agent_2_base_url: str = Field(default="http://localhost:8002", description="Agent 2 Biomedical RAG base URL.")
    agent_3_base_url: str = Field(default="http://localhost:8003", description="Agent 3 PubMed Search base URL.")
    agent_4_base_url: str = Field(default="http://localhost:8004", description="Agent 4 Data Fusion base URL.")
    agent_5_base_url: str = Field(default="http://localhost:8005", description="Agent 5 Reasoning base URL.")

    # Pydantic Field Validators
    @field_validator("app_env")
    def validate_app_env(cls, v: str) -> str:
        val = v.lower().strip()
        if val not in ALLOWED_ENVIRONMENTS:
            raise ValueError(f"Invalid APP_ENV '{v}'. Allowed values: {sorted(ALLOWED_ENVIRONMENTS)}")
        return val

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        val = v.upper().strip()
        if val not in ALLOWED_LOG_LEVELS:
            raise ValueError(f"Invalid LOG_LEVEL '{v}'. Allowed values: {sorted(ALLOWED_LOG_LEVELS)}")
        return val

    @field_validator("primary_reasoning_provider", "fallback_reasoning_provider")
    def validate_provider(cls, v: str) -> str:
        val = v.lower().strip()
        if val not in ALLOWED_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{v}'. Allowed reasoning providers: {sorted(ALLOWED_PROVIDERS)}."
            )
        return val

    @field_validator("gemini_model", "primary_reasoning_model", "fallback_reasoning_model")
    def validate_model_name(cls, v: str) -> str:
        val = v.strip()
        if not val:
            raise ValueError("Model name must be a non-empty string.")
        return val

    @field_validator("confidence_threshold")
    def validate_confidence_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0 (got {v}).")
        return v

    @field_validator("max_adaptive_retries")
    def validate_max_adaptive_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"MAX_ADAPTIVE_RETRIES must be non-negative (got {v}).")
        return v

    @field_validator("gemini_timeout_seconds", "ollama_timeout_seconds", "groq_timeout_seconds", "upstream_timeout_seconds")
    def validate_timeouts(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"Timeout values must be positive floats (got {v}).")
        return v

    @field_validator("api_port")
    def validate_api_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"API_PORT must be between 1 and 65535 (got {v}).")
        return v

    @field_validator("agent_2_base_url", "agent_3_base_url", "agent_4_base_url", "agent_5_base_url")
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def max_retries(self) -> int:
        """Alias for max_adaptive_retries for backward compatibility."""
        return self.max_adaptive_retries

    @property
    def masked_gemini_api_key(self) -> str:
        """Return masked representation of GEMINI_API_KEY for safe logging."""
        if not self.gemini_api_key:
            return "<NOT_SET>"
        if len(self.gemini_api_key) <= 8:
            return "***MASKED***"
        return f"{self.gemini_api_key[:4]}...{self.gemini_api_key[-4:]}"

    @property
    def masked_groq_api_key(self) -> str:
        """Return masked representation of GROQ_API_KEY for safe logging."""
        if not self.groq_api_key:
            return "<NOT_SET>"
        if len(self.groq_api_key) <= 8:
            return "***MASKED***"
        return f"{self.groq_api_key[:4]}...{self.groq_api_key[-4:]}"

    @classmethod
    def from_env(cls, overrides: Optional[Dict[str, Any]] = None) -> "Settings":
        """Instantiate Settings from current process environment variables with optional overrides."""
        env_dict: Dict[str, Any] = {
            "app_env": os.getenv("APP_ENV", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "api_host": os.getenv("API_HOST", "0.0.0.0"),
            "api_port": int(os.getenv("API_PORT", "8005")),
            "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
            "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip(),
            "gemini_timeout_seconds": float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15.0")),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            "ollama_model": os.getenv("OLLAMA_MODEL", "mistral:latest").strip(),
            "ollama_timeout_seconds": float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120.0")),
            "groq_api_key": os.getenv("GROQ_API_KEY", "").strip(),
            "groq_model": os.getenv("GROQ_MODEL", "gpt-oss-120b").strip(),
            "groq_api_key_pool": [
                k.strip() for k in [
                    os.getenv("GROQ_API_KEY_1", ""),
                    os.getenv("GROQ_API_KEY_2", ""),
                    os.getenv("GROQ_API_KEY_3", ""),
                    os.getenv("GROQ_API_KEY_4", ""),
                ] if k.strip()
            ],
            "groq_timeout_seconds": float(os.getenv("GROQ_TIMEOUT_SECONDS", "15.0")),
            "primary_reasoning_provider": os.getenv("PRIMARY_REASONING_PROVIDER", "ollama").strip(),
            "primary_reasoning_model": os.getenv("PRIMARY_REASONING_MODEL", "mistral:latest").strip(),
            "fallback_reasoning_provider": os.getenv("FALLBACK_REASONING_PROVIDER", "deterministic").strip(),
            "fallback_reasoning_model": os.getenv("FALLBACK_REASONING_MODEL", "deterministic-fallback").strip(),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),
            "max_adaptive_retries": int(os.getenv("MAX_RETRIES", os.getenv("MAX_ADAPTIVE_RETRIES", "3"))),
            "upstream_timeout_seconds": float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30.0")),
            "agent_2_base_url": os.getenv("AGENT_2_BASE_URL", "http://localhost:8002").strip(),
            "agent_3_base_url": os.getenv("AGENT_3_BASE_URL", "http://localhost:8003").strip(),
            "agent_4_base_url": os.getenv("AGENT_4_BASE_URL", "http://localhost:8004").strip(),
            "agent_5_base_url": os.getenv("AGENT_5_BASE_URL", "http://localhost:8005").strip(),
        }
        if overrides:
            env_dict.update(overrides)
        return cls.model_validate(env_dict)


# Global authoritative singleton settings instance
settings = Settings.from_env()
