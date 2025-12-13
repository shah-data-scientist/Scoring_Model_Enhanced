# Setup Guide

## Prerequisites

- Docker Desktop
- Docker Compose
- Git

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Scoring_Model_Enhanced
```

### 2. Build Docker Images
```bash
docker-compose build
```

### 3. Start Services
```bash
docker-compose up -d
```

### 4. Verify Health
```bash
docker ps
```

All three containers should show "healthy" status:
- postgres
- api
- streamlit

## Configuration

### Environment Variables
Set in `docker-compose.yml`:
- `DATABASE_URL`: PostgreSQL connection string
- `API_BASE_URL`: API service URL
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `PYTHONPATH`: /app

### Database Credentials
Default PostgreSQL credentials (defined in docker-compose.yml):
- User: postgres
- Password: postgres
- Database: credit_scoring_db

### Application Users
Defined in `backend/init_db.sql`:
- Admin: admin/admin123 (ADMIN role)
- Analyst: analyst/analyst123 (ANALYST role)

## MLflow Setup

### Local MLflow Server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Register Model
1. Train model (see notebooks/)
2. Log to MLflow
3. Register as "credit_scoring_model"
4. Promote to "Production" stage

## Troubleshooting

### Database Connection Issues
```bash
docker logs postgres
```

### API Not Loading Model
Check MLflow tracking URI and model registration:
```bash
docker logs api
```

### Streamlit Import Errors
Verify PYTHONPATH is set correctly in `docker-compose.yml`
