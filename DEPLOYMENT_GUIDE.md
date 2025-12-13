# Deployment Guide - Credit Scoring API

## Overview

This guide covers deploying the Credit Scoring API in production using Docker and CI/CD.

## Prerequisites

- Docker & Docker Compose installed
- Git repository with GitHub Actions enabled
- 2GB RAM minimum, 4GB recommended
- Port 8000 available

---

## Local Deployment

### Option 1: Docker Compose with Full Stack (Recommended)

The full stack includes FastAPI, Streamlit Dashboard, and PostgreSQL database.

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Scoring_Model_Enhanced

# 2. Setup environment variables
cp .env.example .env

# Edit .env and set REQUIRED variables:
# - POSTGRES_PASSWORD (change from default!)
# - SECRET_KEY (generate with: openssl rand -hex 32)

# 3. Build and start all services
docker-compose up --build -d

# 4. Verify services are running
docker-compose ps

# Expected output:
# NAME                        STATUS         PORTS
# credit-scoring-postgres     Up (healthy)   0.0.0.0:5432->5432/tcp
# credit-scoring-api          Up (healthy)   0.0.0.0:8000->8000/tcp
# credit-scoring-streamlit    Up (healthy)   0.0.0.0:8501->8501/tcp

# 5. Check health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/database

# 6. Access services
# API Documentation: http://localhost:8000/docs
# Streamlit Dashboard: http://localhost:8501
# PostgreSQL: localhost:5432

# 7. View logs
docker-compose logs -f api
docker-compose logs -f streamlit
docker-compose logs -f postgres

# 8. Stop services
docker-compose down

# 9. Stop and remove volumes (WARNING: deletes database!)
docker-compose down -v
```

**What gets deployed:**

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| PostgreSQL | `credit-scoring-postgres` | 5432 | Production database |
| FastAPI | `credit-scoring-api` | 8000 | REST API |
| Streamlit | `credit-scoring-streamlit` | 8501 | Web dashboard |

**Persistent data:**
- `postgres_data` volume - Database persists across restarts
- `./logs` - Application logs
- `./data` - Model and features
- `./models` - ML model files

### Option 1b: Individual Services

#### API Only
```bash
docker-compose up api -d
```

#### Streamlit Only (requires API running)
```bash
docker-compose up streamlit -d
```

#### Database Only
```bash
docker-compose up postgres -d
```

### Option 2: Manual Docker

```bash
# Build image
docker build -t credit-scoring-api:latest .

# Run container
docker run -d \
  --name credit-api \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  credit-scoring-api:latest

# Check status
docker ps
curl http://localhost:8000/health
```

### Option 3: Local Python

```bash
# Install dependencies
poetry install

# Start API
poetry run uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## Production Deployment

### GitHub Container Registry

1. **Enable GitHub Container Registry**
   - Repository Settings → Packages
   - Enable "Improved container support"

2. **Push triggers automatic build**
   ```bash
   git push origin main
   ```

3. **Pull and run the image**
   ```bash
   docker pull ghcr.io/<username>/<repo>:latest
   docker run -d -p 8000:8000 ghcr.io/<username>/<repo>:latest
   ```

### Cloud Platforms

#### Heroku

```bash
# Login
heroku login
heroku container:login

# Create app
heroku create credit-scoring-api

# Deploy
heroku container:push web
heroku container:release web

# Open
heroku open
```

#### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/credit-api

# Deploy
gcloud run deploy credit-api \
  --image gcr.io/PROJECT_ID/credit-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS ECS

```bash
# Create ECR repository
aws ecr create-repository --repository-name credit-scoring-api

# Build and push
docker tag credit-scoring-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/credit-scoring-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/credit-scoring-api:latest

# Deploy to ECS (use console or CLI)
```

---

## Monitoring Setup

### 1. Enable Logging

Logs are automatically written to `logs/predictions.jsonl`

```bash
# Mount logs volume
docker run -v $(pwd)/logs:/app/logs credit-scoring-api
```

### 2. Monitor Performance

```bash
# Inside container or locally
poetry run python scripts/monitoring/dashboard.py
poetry run python scripts/monitoring/profile_performance.py
```

### 3. Data Drift Detection

```bash
# Install evidently
poetry add evidently

# Run drift detection
poetry run python scripts/monitoring/detect_drift.py

# View report
open reports/drift/drift_report_*.html
```

---

## Configuration

### Environment Variables

```bash
# Optional environment variables
export MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
export API_PORT=8000
export LOG_LEVEL=INFO
```

### Model Updates

To update the production model:

1. Train new model and save to `models/production_model.pkl`
2. Rebuild Docker image
3. Redeploy

```bash
docker build -t credit-scoring-api:v2 .
docker stop credit-api
docker rm credit-api
docker run -d --name credit-api -p 8000:8000 credit-scoring-api:v2
```

---

## Health Checks

### API Health

```bash
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "unknown",
  "model_version": null,
  "timestamp": "2025-12-12T..."
}
```

### Docker Health

```bash
docker ps  # Check if container is healthy
docker logs credit-api --tail 100
```

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs credit-api

# Common issues:
# - Port 8000 already in use
# - Missing precomputed features file
# - Insufficient memory
```

### High memory usage

```bash
# Check memory
docker stats credit-api

# Solution: Limit memory
docker run -d -m 1g credit-scoring-api
```

### Slow predictions

```bash
# Profile performance
poetry run python scripts/monitoring/profile_performance.py

# Check if precomputed features loaded
docker logs credit-api | grep "Loaded.*precomputed"
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

Automatically runs on push to `main`:

1. **Test**: Run pytest (67 tests)
2. **Build**: Create Docker image
3. **Push**: Upload to GitHub Container Registry

### Manual Trigger

```bash
# Via GitHub UI
Actions → CI/CD Pipeline → Run workflow

# Via API
curl -X POST \
  -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/repos/<user>/<repo>/actions/workflows/ci-cd.yml/dispatches \
  -d '{"ref":"main"}'
```

---

## Security

### Best Practices

1. **Don't commit secrets** (.env files, credentials)
2. **Use environment variables** for sensitive config
3. **Enable HTTPS** in production
4. **Implement rate limiting** to prevent abuse
5. **Regular security updates** (rebuild Docker images monthly)

### Secrets Management

```bash
# GitHub Secrets (for CI/CD)
Settings → Secrets → Actions

# Docker secrets (for sensitive data)
docker secret create db_password ./db_password.txt
```

---

## Scaling

### Horizontal Scaling

```bash
# Docker Compose (multiple replicas)
docker-compose up --scale api=3

# Kubernetes
kubectl scale deployment credit-api --replicas=3
```

### Load Balancing

Use nginx or cloud load balancer:

```nginx
upstream credit_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://credit_api;
    }
}
```

---

## Maintenance

### Regular Tasks

**Daily**:
- Check logs for errors
- Monitor prediction volume
- Verify API health

**Weekly**:
- Run drift detection
- Review performance metrics
- Check disk space

**Monthly**:
- Update dependencies
- Rebuild Docker images
- Review and archive old logs

### Backup

```bash
# Backup database
docker cp credit-api:/app/data/credit_scoring.db ./backup/

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

---

## Support

For issues or questions:
- Check logs: `docker logs credit-api`
- Review monitoring: `scripts/monitoring/dashboard.py`
- GitHub Issues: Create an issue with logs and error messages

