# System Architecture

## Components

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│             │       │             │       │             │
│  Streamlit  │──────▶│   FastAPI   │──────▶│ PostgreSQL  │
│  Dashboard  │       │     API     │       │  Database   │
│             │       │             │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   MLflow    │
                      │   Server    │
                      └─────────────┘
```

## Technology Stack

### Frontend
- **Streamlit**: Interactive web dashboard
- **Plotly**: Data visualization

### Backend
- **FastAPI**: RESTful API framework
- **SQLAlchemy**: ORM for database operations
- **Pydantic**: Data validation

### Database
- **PostgreSQL 15**: Relational database
- **pgcrypto**: Password hashing

### ML/MLOps
- **MLflow**: Model versioning and tracking
- **scikit-learn**: ML models
- **SHAP**: Model explainability

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

## Data Flow

1. **User Request**: User logs in via Streamlit or API
2. **Authentication**: JWT token verification
3. **Prediction**: API loads model from MLflow, runs inference
4. **Storage**: Results saved to PostgreSQL
5. **Response**: Prediction + SHAP values returned
6. **Visualization**: Streamlit renders charts and explanations

## Security

- JWT-based authentication
- Role-based access control (ADMIN, ANALYST)
- Bcrypt password hashing
- Environment variable configuration
- Docker network isolation
