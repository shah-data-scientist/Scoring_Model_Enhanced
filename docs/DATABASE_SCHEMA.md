# Database Schema Documentation

## Overview

The application uses a PostgreSQL database to store user information, predictions, batch upload metadata, model performance metrics, data drift reports, and audit logs. The schema is initialized automatically by the `backend/init_db.sql` script.

## Tables

### 1. `users`
Manages user accounts for the application.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Unique identifier for the user. |
| `username` | VARCHAR(255) | UNIQUE, NOT NULL | User's login name. |
| `email` | VARCHAR(255) | UNIQUE, NOT NULL | User's email address. |
| `hashed_password` | VARCHAR(255) | NOT NULL | Bcrypt hashed password. |
| `full_name` | VARCHAR(255) | | User's full name. |
| `is_active` | BOOLEAN | DEFAULT TRUE | Whether the account is active. |
| `is_superuser` | BOOLEAN | DEFAULT FALSE | Whether the user has administrative privileges. |
| `role` | VARCHAR(50) | DEFAULT 'viewer' | User role (e.g., 'admin', 'viewer'). |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Account creation timestamp. |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Last update timestamp (auto-updated via trigger). |
| `last_login` | TIMESTAMP | | Timestamp of the last successful login. |

### 2. `predictions`
Stores individual credit scoring predictions.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Unique identifier for the prediction record. |
| `sk_id_curr` | INTEGER | NOT NULL | Customer ID from the application data. |
| `probability` | FLOAT | NOT NULL | Predicted probability of default. |
| `risk_level` | VARCHAR(50) | NOT NULL | Categorized risk level (e.g., 'Low', 'Medium', 'High'). |
| `prediction_date` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When the prediction was made. |
| `model_version` | VARCHAR(50) | | Version of the model used for prediction. |
| `user_id` | INTEGER | FK -> users(id) | User who requested the prediction (NULL if system). |
| `batch_id` | INTEGER | | ID of the batch if part of a batch prediction. |
| `processing_time_ms` | FLOAT | | Time taken to process the prediction in milliseconds. |
| `metadata` | JSONB | | Additional metadata about the prediction. |

### 3. `batch_uploads`
Tracks files uploaded for batch predictions.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Unique identifier for the batch upload. |
| `user_id` | INTEGER | FK -> users(id) | User who uploaded the file. |
| `filename` | VARCHAR(255) | NOT NULL | Original name of the uploaded file. |
| `file_size` | BIGINT | | Size of the file in bytes. |
| `num_applications` | INTEGER | | Number of applications (rows) in the file. |
| `status` | VARCHAR(50) | DEFAULT 'pending' | Processing status ('pending', 'processing', 'completed', 'failed'). |
| `upload_date` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When the file was uploaded. |
| `processing_started_at` | TIMESTAMP | | When processing began. |
| `processing_completed_at` | TIMESTAMP | | When processing finished. |
| `error_message` | TEXT | | Error message if processing failed. |
| `results_file_path` | TEXT | | Path to the generated results file. |
| `metadata` | JSONB | | Additional metadata. |

### 4. `model_performance`
Stores metrics to track model performance over time.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Unique identifier for the metric record. |
| `model_version` | VARCHAR(50) | NOT NULL | Version of the model being evaluated. |
| `metric_name` | VARCHAR(100) | NOT NULL | Name of the metric (e.g., 'ROC-AUC', 'Precision'). |
| `metric_value` | FLOAT | NOT NULL | Value of the metric. |
| `dataset` | VARCHAR(50) | | Dataset used for evaluation (e.g., 'validation', 'production_batch'). |
| `recorded_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When the metric was recorded. |
| `metadata` | JSONB | | Additional metadata. |

### 5. `data_drift_reports`
Records reports on data drift detection.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Unique identifier for the drift report. |
| `report_date` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Date of the report. |
| `drift_detected` | BOOLEAN | DEFAULT FALSE | Whether drift was detected. |
| `drift_score` | FLOAT | | Overall drift score. |
| `drifted_features` | JSONB | | Details of features that showed drift. |
| `report_file_path` | TEXT | | Path to the full drift report file. |
| `metadata` | JSONB | | Additional metadata. |

### 6. `audit_log`
Logs system events and user actions for auditing purposes.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Unique identifier for the log entry. |
| `user_id` | INTEGER | FK -> users(id) | User who performed the action. |
| `action` | VARCHAR(100) | NOT NULL | Description of the action (e.g., 'LOGIN', 'PREDICT'). |
| `resource_type` | VARCHAR(50) | | Type of resource affected. |
| `resource_id` | INTEGER | | ID of the resource affected. |
| `details` | JSONB | | structured details about the event. |
| `ip_address` | INET | | IP address of the requester. |
| `user_agent` | TEXT | | User agent string of the client. |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When the event occurred. |

## Indexes

To ensure performance, the following indexes are created:

- **`predictions`**: `sk_id_curr`, `prediction_date`, `user_id`, `batch_id`
- **`batch_uploads`**: `user_id`, `status`, `upload_date`
- **`model_performance`**: `model_version`, `recorded_at`
- **`data_drift_reports`**: `report_date`
- **`audit_log`**: `user_id`, `created_at`, `action`

## Triggers

- **`update_users_updated_at`**: Automatically updates the `updated_at` column in the `users` table whenever a row is modified.

## Initialization

The schema is initialized by `backend/init_db.sql`. On first run, it also creates default users:
- **Admin**: `admin` / `admin123` (Change in production!)
- **Viewer**: `viewer` / `viewer123` (Change in production!)
