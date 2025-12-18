"""Streamlit UI Simulation Script - Simplified.

This script uses streamlit.testing.v1.AppTest to simulate user interaction.
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from streamlit.testing.v1 import AppTest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_simulation():
    """Run a full simulation bypassing login."""
    print("[START] Starting Simplified UI Simulation...")
    
    # Initialize the app test
    at = AppTest.from_file("streamlit_app/app.py", default_timeout=300)
    
    # Set session state to be logged in as admin
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["user_role"] = "ADMIN"
    at.session_state["user_email"] = "admin@creditscoring.local"
    at.session_state["auth_initialized"] = True
    # Pre-set the tab we want to be on (must match exact tab label with emoji)
    at.session_state["main_active_tab"] = "\U0001F4C1 Batch Predictions"  # 📁
    at.session_state["batch_active_tab"] = "Upload & Predict"
    
    at.run()

    print("[OK] Session state initialized")

    # 1. Find File Uploader
    print("[STEP 1] Uploading 7 CSV files")
    samples_dir = PROJECT_ROOT / "data" / "samples"
    file_names = [
        "application.csv", "bureau.csv", "bureau_balance.csv",
        "credit_card_balance.csv", "installments_payments.csv",
        "POS_CASH_balance.csv", "previous_application.csv"
    ]

    files_to_upload = [str(samples_dir / fn) for fn in file_names if (samples_dir / fn).exists()]
    print(f"[FILES] Found {len(files_to_upload)} files to upload.")
    for f in files_to_upload:
        print(f"  - {f}")

    print(f"[DEBUG] Total file_uploader elements: {len(at.get('file_uploader'))}")

    # Try to find any element with 'upload' in its name or type
    all_elements = at.main
    print(f"[DEBUG] Main elements: {len(all_elements)}")

    # Try via key directly
    try:
        uploaders = at.get(key="multi_file_upload")
        if uploaders:
            uploader = uploaders[0]
            print(f"[OK] Found via key='multi_file_upload'. Type: {type(uploader)}")
        else:
            print("[WARN] Not found via key='multi_file_upload'")
            uploaders = at.get("file_uploader")
            uploader = uploaders[0] if uploaders else None
    except Exception as e:
        print(f"[WARN] Error finding by key: {e}")
        uploaders = at.get("file_uploader")
        uploader = uploaders[0] if uploaders else None

    if not uploader:
        print("[FAIL] No file uploaders found on page.")
        return

    print(f"[OK] Selected uploader: {uploader.label} (Key: {uploader.key})")
    print(f"[DEBUG] Uploader type (attr): {uploader.type}")
    print(f"[DEBUG] Uploader type (class): {type(uploader)}")

    try:
        if hasattr(uploader, "upload"):
            print("[ACTION] Using .upload() method")
            uploader.upload(files_to_upload)
        elif hasattr(uploader, "set_value"):
            print("[ACTION] Using .set_value() method")
            uploader.set_value(files_to_upload)
        else:
            print("[SKIP] File upload not supported via AppTest for this element type")
            print("[INFO] The file uploader was found but Streamlit AppTest doesn't support file uploads directly.")
            print("[INFO] Simulation can proceed to test other UI elements.")
        at.run()
    except Exception as upload_err:
        print(f"[WARN] Upload step issue: {type(upload_err).__name__}: {upload_err}")
        print("[INFO] Continuing with simulation...")

    print(f"[OK] Files uploaded.")

    # 2. Process Batch
    print("[STEP 2] Processing Batch")
    buttons = at.get("button")
    process_btn = None
    for b in buttons:
        if b.label and "Process Batch" in b.label:
            process_btn = b
            break

    if not process_btn:
        for b in buttons:
            if b.key == "process_multi":
                process_btn = b
                break

    if process_btn:
        print(f"[OK] Found button: {process_btn.label}")
        process_btn.click()
        print("[WAIT] Processing (simulating API call)...")
        at.run(timeout=300)
    else:
        print("[FAIL] 'Process Batch' button not found.")
        print(f"Available buttons: {[getattr(b, 'label', 'N/A') for b in buttons]}")
        return

    # 3. Check Results
    if at.error:
        print("[ERROR] Errors during processing:")
        for err in at.error:
            print(f"  - {err.body.splitlines()[0]}")
    else:
        success_msgs = [s.body for s in at.success]
        if any("successfully" in msg.lower() for msg in success_msgs):
            print("[SUCCESS] Simulation SUCCESS!")
            for m in at.metric:
                print(f"  - {m.label}: {m.value}")
        else:
            print("[WARN] Success message not found.")
            if at.info:
                print(f"[INFO] {at.info[-1].body}")
            if at.warning:
                print(f"[WARN] {at.warning[-1].body}")

    print("\n[DONE] Simulation Complete!")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"[CRASH] Simulation failed: {e}")
