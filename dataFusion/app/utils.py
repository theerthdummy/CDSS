"""Utility helpers and reusable logging configuration for the API."""

import logging
import sys
import uuid
from typing import Optional
from app.config import settings


def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("agent4_datafusion")
    
    if logger.handlers:
        return logger

    log_level = getattr(logging, settings.log_level, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logging()


def normalize_text(value: str) -> str:
    """Return a trimmed string for consistent handling."""
    return value.strip()


def generate_correlation_id(incoming_id: Optional[str] = None) -> str:
    """Return incoming_id if present, otherwise generate a new UUID4 string."""
    if incoming_id and incoming_id.strip():
        return incoming_id.strip()
    return str(uuid.uuid4())