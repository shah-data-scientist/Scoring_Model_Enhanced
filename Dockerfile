# Dockerfile for Credit Scoring API - Optimized Multi-stage Build

# Stage 1: Export dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install Poetry
RUN pip install poetry poetry-plugin-export

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Export dependencies to requirements.txt
RUN poetry export --only main --without-hashes --format=requirements.txt > requirements.txt

# Stage 2: Runtime environment
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# libgomp1 is required for LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from builder
COPY --from=builder /app/requirements.txt .

# Install dependencies using pip (faster and lighter than poetry)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api ./api
COPY src ./src
COPY backend ./backend
COPY config ./config
COPY config.yaml .

# Create required directories
RUN mkdir -p logs mlruns models data

# Create a non-root user and switch to it
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 appuser && \
    chown -R appuser:appgroup /app

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
