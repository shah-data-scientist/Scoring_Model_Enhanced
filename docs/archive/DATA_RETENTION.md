# Data Retention & Privacy

## Storage
- Predictions stored in PostgreSQL (`predictions` table)
- Raw applications stored (`raw_applications`) for audit

## Retention Policy
- Default: 180 days for raw applications
- Predictions retained for analytics (configurable)

## Anonymization
- `SK_ID_CURR` anonymized for end-user tests
- PII is not stored

## Access Control
- Roles: ADMIN, ANALYST
- JWT-based auth, database users with bcrypt

## Deletion
- Admin tools to purge old batches
- Backups handled via Docker volumes
