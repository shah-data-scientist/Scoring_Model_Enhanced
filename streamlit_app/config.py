import os

# Base URL for API; defaults to localhost for local runs
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
