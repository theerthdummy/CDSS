"""FastAPI application for Agent 5 Clinical Reasoning Service."""

import logging
from typing import Dict
from fastapi import FastAPI

from app.config import settings
from app.routers.reasoning import router as reasoning_router

# Configure root logger level from settings
logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger("agent5.main")

app = FastAPI(
    title="Agent 5 - Clinical Decision Support & Reasoning Engine",
    description="Agent 5 service for clinical semantic reasoning and decision support.",
    version="1.0.0",
)

# Register versioned API routers
app.include_router(reasoning_router)


def _provider_model_label(provider: str) -> str:
    provider_name = provider.lower().strip()
    if provider_name == "groq":
        return settings.groq_model
    if provider_name == "ollama":
        return settings.ollama_model
    if provider_name == "gemini":
        return settings.gemini_model
    if provider_name == "deterministic":
        return "deterministic-fallback"
    return "unknown"


@app.on_event("startup")
def startup_event() -> None:
    """Log safe provider/model configuration summary on application startup (no API keys)."""
    logger.info(
        f"Starting Agent 5 Service (Environment: '{settings.app_env}', Log Level: '{settings.log_level}'). "
        f"Primary: '{settings.primary_reasoning_provider}:{_provider_model_label(settings.primary_reasoning_provider)}', "
        f"Fallback: '{settings.fallback_reasoning_provider}:{_provider_model_label(settings.fallback_reasoning_provider)}', "
        f"Confidence Threshold: {settings.confidence_threshold:.2f}, Max Retries: {settings.max_adaptive_retries}."
    )


@app.get("/")
@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint confirming Agent 5 service availability."""
    return {
        "status": "ok",
        "agent": "agent_5",
    }
