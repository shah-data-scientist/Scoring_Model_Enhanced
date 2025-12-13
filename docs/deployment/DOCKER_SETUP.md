# Docker Setup

## Services

### postgres
- **Image**: postgres:15-alpine
- **Port**: 5432
- **Health Check**: pg_isready
- **Volume**: postgres_data

### api
- **Build**: Dockerfile
- **Port**: 8000
- **Depends On**: postgres
- **Health Check**: GET /health
- **Environment**:
  - DATABASE_URL
  - MLFLOW_TRACKING_URI
  - PYTHONPATH=/app

### streamlit
- **Build**: Dockerfile.streamlit
- **Port**: 8501
- **Depends On**: api
- **Health Check**: GET /
- **Environment**:
  - API_BASE_URL
  - PYTHONPATH=/app

## Docker Compose Commands

### Build
```bash
docker-compose build
```

### Start
```bash
docker-compose up -d
```

### Stop
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f <service>
```

### Restart Service
```bash
docker-compose restart <service>
```

### Clean Rebuild
```bash
docker-compose down --rmi local
docker-compose build --no-cache
docker-compose up -d
```

## Volume Management

### List Volumes
```bash
docker volume ls
```

### Remove Volumes
```bash
docker-compose down -v
```

**Warning:** This deletes all database data.

## Troubleshooting

### Service Won't Start
```bash
docker-compose logs <service>
```

### Port Already in Use
Change port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Host:Container
```

### Database Connection Failed
Verify service order and health checks:
```bash
docker ps
```
