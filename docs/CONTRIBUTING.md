# Contributing Guidelines

## Setup
- Install Poetry: https://python-poetry.org/
- Install deps: `poetry install`
- Run tests: `poetry run pytest -v`

## Code Style
- Ruff for linting: `poetry run ruff check .`
- Black for formatting: `poetry run black .`
- Mypy for type checks: `poetry run mypy`

## Pull Requests
- Include tests for new features (target ≥80% coverage)
- Update docs in `docs/`
- Verify Docker build: `docker-compose build`
- Keep changes focused; avoid unrelated refactors
