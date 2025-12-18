"""Interaction Logger for recording user activities in Streamlit.
Logs widget changes to logs/interaction_trace.log.
"""

import logging
from pathlib import Path
import streamlit as st
import json
from datetime import datetime

# Set up dedicated logger
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRACE_FILE = LOG_DIR / "interaction_trace.log"

def log_interaction(widget_type, label, value, key=None):
    """Log a single user interaction."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "widget": widget_type,
        "label": label,
        "key": key,
        "value": str(value)
    }
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def track_state_changes():
    """Monitor session state for changes and log them."""
    if "last_logged_state" not in st.session_state:
        st.session_state.last_logged_state = {}

    current_state = {k: v for k, v in st.session_state.items() if k != "last_logged_state" and not k.startswith("_")}
    
    for key, value in current_state.items():
        if key not in st.session_state.last_logged_state or st.session_state.last_logged_state[key] != value:
            # Log the change
            log_interaction("session_state", key, value, key=key)
    
    st.session_state.last_logged_state = current_state.copy()

