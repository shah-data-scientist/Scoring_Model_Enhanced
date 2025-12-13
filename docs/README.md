# Credit Scoring Model - Enhanced

Enterprise-grade credit risk prediction system with MLflow integration, Docker deployment, and secure user management.

## Overview

This system provides real-time credit risk assessment using machine learning models, with features including:
- RESTful API for predictions
- Interactive Streamlit dashboard
- MLflow for model versioning
- PostgreSQL database with role-based access
- Docker containerization
- SHAP explainability

## Quick Start

See [QUICK_START.md](../QUICK_START.md) for deployment instructions.

## Documentation

- [Setup Guide](SETUP.md) - Installation and configuration
- [API Documentation](API.md) - Endpoint reference
- [Architecture](architecture/) - System design and database schema
- [Deployment](deployment/) - Docker and MLflow setup

## Credentials

**Default users:**
- Admin: `admin` / `admin123`
- Analyst: `analyst` / `analyst123`

## Services

- API: http://localhost:8000
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
- PostgreSQL: localhost:5432

## Project Status

✅ Fully operational
- All 3 Docker containers healthy
- Database initialized with proper schema
- Data anonymized (SK_ID_CURR)
- MLflow integrated
