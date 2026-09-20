FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system utilities and supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    supervisor \
    build-essential \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Copy all project code
COPY . /app

# Install dependencies for all agents and orchestrator
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    groq \
    google-generativeai \
    openai \
    tavily-python \
    qdrant-client \
    sentence-transformers \
    biopython \
    tenacity \
    beautifulsoup4 \
    python-dotenv \
    requests

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Start all internal microservices and orchestrator via supervisord
CMD ["supervisord", "-c", "/app/supervisord.conf"]
