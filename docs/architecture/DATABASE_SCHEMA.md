# Database Schema

## Tables

### users
Stores application user credentials and roles.

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Roles:**
- `ADMIN`: Full access to all features
- `ANALYST`: Read-only access

### predictions
Stores all prediction requests and results.

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL,
    prediction INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    shap_values JSONB,
    top_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Fields:**
- `client_id`: SK_ID_CURR (anonymized)
- `prediction`: 0 (approved) or 1 (rejected)
- `probability`: Risk probability [0-1]
- `shap_values`: SHAP explanation (JSON)
- `top_features`: Top contributing features (JSON)

## Indexes

```sql
CREATE INDEX idx_predictions_client_id ON predictions(client_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
```

## Initial Data

Default users created via `backend/init_db.sql`:
- admin (ADMIN role)
- analyst (ANALYST role)

Passwords hashed using bcrypt (`crypt()` function).

## NaN Handling

Python `NaN` values converted to SQL `NULL` before insertion to prevent JSON serialization errors.
