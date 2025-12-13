# API Documentation

Base URL: `http://localhost:8000`

## Authentication

All prediction endpoints require authentication via Bearer token.

### Login
```http
POST /login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "username": "admin",
  "role": "ADMIN"
}
```

## Endpoints

### Health Check
```http
GET /health
```

### Predict
```http
POST /predict
Authorization: Bearer <token>
Content-Type: application/json

{
  "client_id": 100001,
  "features": {...}
}
```

**Response:**
```json
{
  "client_id": 100001,
  "prediction": 0,
  "probability": 0.15,
  "shap_values": {...},
  "top_features": [...]
}
```

### Get Client Info
```http
GET /client/{client_id}
Authorization: Bearer <token>
```

### Global Statistics
```http
GET /global-statistics
Authorization: Bearer <token>
```

### Batch Predictions
```http
POST /batch-predict
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <CSV file>
```

## Error Codes

- `400`: Bad Request - Invalid input data
- `401`: Unauthorized - Missing or invalid token
- `404`: Not Found - Client ID not found
- `500`: Internal Server Error - Model or database error

## Rate Limiting

No rate limiting currently implemented.

## Data Format

Features must match the model's training schema. See `config/model_features.txt` for required features.
