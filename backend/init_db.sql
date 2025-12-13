-- =============================================================================
-- PostgreSQL Database Initialization Script
-- =============================================================================
-- This script initializes the credit scoring database schema
-- It is automatically run when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    role VARCHAR(50) DEFAULT 'viewer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    sk_id_curr INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    risk_level VARCHAR(50) NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    batch_id INTEGER,
    processing_time_ms FLOAT,
    metadata JSONB
);

-- Create batch_uploads table
CREATE TABLE IF NOT EXISTS batch_uploads (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT,
    num_applications INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    error_message TEXT,
    results_file_path TEXT,
    metadata JSONB
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    dataset VARCHAR(50),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create data_drift_reports table
CREATE TABLE IF NOT EXISTS data_drift_reports (
    id SERIAL PRIMARY KEY,
    report_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_detected BOOLEAN DEFAULT FALSE,
    drift_score FLOAT,
    drifted_features JSONB,
    report_file_path TEXT,
    metadata JSONB
);

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id INTEGER,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_sk_id_curr ON predictions(sk_id_curr);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_batch_id ON predictions(batch_id);
CREATE INDEX IF NOT EXISTS idx_batch_uploads_user_id ON batch_uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_batch_uploads_status ON batch_uploads(status);
CREATE INDEX IF NOT EXISTS idx_batch_uploads_upload_date ON batch_uploads(upload_date);
CREATE INDEX IF NOT EXISTS idx_model_performance_model_version ON model_performance(model_version);
CREATE INDEX IF NOT EXISTS idx_model_performance_recorded_at ON model_performance(recorded_at);
CREATE INDEX IF NOT EXISTS idx_data_drift_reports_report_date ON data_drift_reports(report_date);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);

-- Create trigger function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user (password: admin123 - CHANGE IN PRODUCTION!)
-- Password hash generated with bcrypt for 'admin123'
INSERT INTO users (username, email, hashed_password, full_name, is_active, is_superuser, role)
VALUES (
    'admin',
    'admin@creditscore.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS3MKM5jm',
    'System Administrator',
    TRUE,
    TRUE,
    'admin'
)
ON CONFLICT (username) DO NOTHING;

-- Insert default viewer user (password: viewer123 - CHANGE IN PRODUCTION!)
INSERT INTO users (username, email, hashed_password, full_name, is_active, is_superuser, role)
VALUES (
    'viewer',
    'viewer@creditscore.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS3MKM5jm',
    'Default Viewer',
    TRUE,
    FALSE,
    'viewer'
)
ON CONFLICT (username) DO NOTHING;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'Default users created:';
    RAISE NOTICE '  - admin (password: admin123) - CHANGE IN PRODUCTION!';
    RAISE NOTICE '  - viewer (password: viewer123) - CHANGE IN PRODUCTION!';
END $$;
