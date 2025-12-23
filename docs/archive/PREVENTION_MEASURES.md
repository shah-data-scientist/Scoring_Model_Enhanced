# Error Prevention Measures

This document outlines the safeguards implemented to prevent recurring errors in the application.

## Overview

After encountering several critical issues (model injection bug, JSON serialization errors, pandas API deprecation), comprehensive prevention measures have been implemented.

## 1. Centralized JSON Serialization (`api/json_utils.py`)

**Purpose**: Handle NaN, Inf, and other non-JSON-compliant values consistently across the application.

**Features**:
- `sanitize_for_json()`: Recursively sanitize any Python object for JSON serialization
- `dataframe_to_json_safe()`: Convert pandas DataFrames with automatic NaN/Inf handling
- `validate_numeric_value()`: Validate numeric values and raise errors for NaN/Inf

**Usage**: All API endpoints returning data now use these utilities instead of direct `.to_dict()` calls.

**Tests**: `tests/test_json_utils.py` with 100% coverage of edge cases

## 2. Model Validation (`api/model_validator.py`)

**Purpose**: Ensure model is loaded and has required methods before use.

**Features**:
- `ModelValidator.check_model_loaded()`: Raises 503 if model is None
- `ModelValidator.validate_model_attributes()`: Verifies model has required methods

**Usage**:
- Applied in `api/app.py` at startup to validate loaded model
- Applied in `api/batch_predictions.py` before making predictions
- Can be used in any endpoint that requires model access

**Tests**: `tests/test_model_validator.py` covering all validation scenarios

## 3. Startup Validation

**Location**: `api/app.py` startup event

**Checks**:
1. Model loading status
2. Model has `predict` method
3. Model has `predict_proba` method
4. Logs validation results

**Benefit**: Catches configuration issues immediately instead of failing on first request.

## 4. Comprehensive Testing

**New Test Coverage**:
- `tests/test_json_utils.py`: 15+ test cases for NaN/Inf handling
- `tests/test_model_validator.py`: 7+ test cases for model validation
- Tests run automatically in CI pipeline

**CI Integration**: Tests explicitly added to `.github/workflows/ci-cd.yml` with verbose output.

## 5. Defensive Programming Patterns

**Applied Throughout**:
- Always validate before use (model, data, parameters)
- Use centralized utilities for common operations (JSON, validation)
- Log validation steps for debugging
- Return specific error messages with context

**Example Pattern**:
```python
# Bad: Direct use without validation
predictions = model.predict(data)

# Good: Validate then use
ModelValidator.check_model_loaded(model, "prediction endpoint")
ModelValidator.validate_model_attributes(model, ['predict'])
predictions = model.predict(data)
```

## 6. Error Propagation

**Strategy**: Fail fast with clear error messages

**HTTP Status Codes**:
- `503 Service Unavailable`: Model not loaded (transient issue)
- `500 Internal Server Error`: Model misconfigured (requires intervention)
- `422 Unprocessable Entity`: Invalid input data

**Benefits**:
- Users get actionable error messages
- Monitoring can alert on specific error types
- Logs contain detailed context for debugging

## 7. Import Path Safety

**Issue Addressed**: Streamlit pages couldn't import from `api/` directory

**Solution**: Added explicit path manipulation in monitoring.py:
```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'api'))
```

**Benefit**: Consistent imports across all application components.

## Testing Checklist

Before deploying changes, verify:
- [ ] All tests pass: `pytest tests/ -v`
- [ ] JSON utils tests pass: `pytest tests/test_json_utils.py -v`
- [ ] Model validator tests pass: `pytest tests/test_model_validator.py -v`
- [ ] Coverage ≥ 75%: Check CI output
- [ ] Docker build succeeds: `docker-compose build`
- [ ] Containers start healthy: `docker-compose up -d && docker-compose ps`
- [ ] API loads model: Check logs for "Model validation successful"
- [ ] Batch predictions work: Upload test CSV
- [ ] Monitoring dashboard loads: No 500 errors on any tab
- [ ] Data quality check works: Click "Check Quality" button

## Maintenance Guidelines

**When Adding New Endpoints**:
1. Use `ModelValidator.check_model_loaded()` if endpoint needs model
2. Use `dataframe_to_json_safe()` for any DataFrame responses
3. Use `sanitize_for_json()` for complex nested structures
4. Add tests for new functionality

**When Updating Dependencies**:
1. Check for pandas API changes (e.g., `.fillna()` deprecations)
2. Verify numpy compatibility with current patterns
3. Run full test suite after updates

**When Debugging Issues**:
1. Check startup logs for model validation
2. Look for JSON serialization errors in API logs
3. Verify model methods exist: `hasattr(model, 'predict_proba')`
4. Test with edge case data (NaN, Inf, missing values)

## Related Documentation

- [API Documentation](docs/api.md)
- [Testing Guide](tests/README.md) (if exists)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
