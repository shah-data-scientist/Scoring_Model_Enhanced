# Dockerfile for Credit Scoring API - Lean Multi-stage Build

# Stage 1: Export dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install Poetry and export dependencies in one layer
RUN pip install --no-cache-dir poetry poetry-plugin-export

COPY pyproject.toml poetry.lock ./

RUN poetry export --only main --without-hashes --format=requirements.txt > requirements.txt

# Stage 2: Runtime environment
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and Python packages in one layer
COPY --from=builder /app/requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# Create user and directories first (better layer caching)
RUN addgroup --system --gid 1001 appgroup \
    && adduser --system --uid 1001 --gid 1001 appuser \
    && mkdir -p logs mlruns models data \
    && chown -R appuser:appgroup /app

# Copy only necessary application code
COPY --chown=appuser:appgroup api ./api
COPY --chown=appuser:appgroup src ./src
COPY --chown=appuser:appgroup backend ./backend
COPY --chown=appuser:appgroup config ./config
COPY --chown=appuser:appgroup config.yaml .

USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
