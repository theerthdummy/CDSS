# =============================================================================
# Agent 5 Clinical Reasoning Service - Production Dockerfile
# =============================================================================

FROM python:3.10-slim as base

# Set environment defaults for Python runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Create a non-root runtime user for container security
RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose production service port
EXPOSE 8000

# Healthcheck targeting GET /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server execution command (No --reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
