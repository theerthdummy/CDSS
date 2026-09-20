# Use an official lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files to disc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies (curl for health checks / network utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install all Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire agent code structure into the container
COPY . .

# Set PYTHONPATH so internal module imports (e.g. from src.abbrevs) work seamlessly
ENV PYTHONPATH=/app

# Default command: Run PyTest suite to ensure all unit tests pass on container startup
CMD ["pytest", "-v"]