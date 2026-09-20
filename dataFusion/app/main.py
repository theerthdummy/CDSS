"""FastAPI application entrypoint with lifecycle logging."""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.routes import router
from app.utils import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for server startup and shutdown logging."""
    logger.info(f"Starting up {settings.app_name} (v{settings.app_version}) in {settings.env} environment...")
    
    # LLM Startup Diagnostic Logging (PART 1 Requirement)
    api_key = settings.llm_api_key.strip()
    if api_key and api_key != "mock_key_or_set_real_api_key":
        masked_key = f"{api_key[:8]}...{api_key[-6:]}" if len(api_key) > 14 else "***DETECTED***"
        key_status = f"Detected ({masked_key})"
    else:
        key_status = "Unconfigured / Default Mock Key"

    logger.info(
        f"LLM Integration Configuration -> Provider: '{settings.llm_provider}', "
        f"Model: '{settings.llm_model_name}', Status: {key_status}, Base URL: '{settings.llm_base_url}'"
    )
    yield
    logger.info(f"Shutting down {settings.app_name}...")


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(router)

@app.get("/")
async def root_health_check():
    """Simple root health check endpoint for the orchestrator."""
    return {"status": "ok", "service": settings.app_name}