"""Comprehensive UI Simulation Test - Data Quality Focus.

Tests:
1. Model Performance: Slider interaction.
2. Batch Predictions: Upload 7 files and process.
3. Download Reports: Verify Excel and HTML report availability.
4. Monitoring:
   - Overview
   - Model Monitoring
   - Performance Monitoring
5. Data Quality:
   - Feature Drift (against Training Data)
   - Data Quality Checks
   - Drift History
"""

import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_full_test():
    print("🚀 Starting Comprehensive UI Test Simulation...")
    
    at = AppTest.from_file("streamlit_app/app.py", default_timeout=180)
    
    # Initialize session state for Admin
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["user_role"] = "ADMIN"
    at.session_state["user_email"] = "admin@creditscoring.local"
    at.session_state["auth_initialized"] = True
    
    # --- TEST 1: Model Performance ---
    print("\n📝 TEST 1: Model Performance")
    at.session_state["main_active_tab"] = "📈 Model Performance"
    at.run()
    try:
        sliders = at.get("slider")
        slider = next((s for s in sliders if "Decision" in s.label), None)
        if slider:
            slider.set_value(0.40).run()
            print("  ✅ Slider set to 0.40, metrics updated.")
    except Exception as e:
        print(f"  ❌ Slider test error: {e}")

    # --- TEST 2: Batch Predictions ---
    print("\n📝 TEST 2: Batch Predictions - Upload & Process")
    at.session_state["main_active_tab"] = "📁 Batch Predictions"
    at.session_state["batch_active_tab"] = "Upload & Predict"
    at.run()
    
    samples_dir = PROJECT_ROOT / "data" / "samples"
    file_names = ["application.csv", "bureau.csv", "bureau_balance.csv", "credit_card_balance.csv", 
                  "installments_payments.csv", "POS_CASH_balance.csv", "previous_application.csv"]
    files = [str(samples_dir / fn) for fn in file_names]
    
    try:
        uploaders = at.get("file_uploader")
        uploader = next((u for u in uploaders if "select all 7" in u.label.lower() or u.key == "multi_file_upload"), None)
        if uploader:
            uploader.upload(files).run()
            print("  ✅ 7 Files uploaded.")
            
            btn = next((b for b in at.get("button") if "Process Batch" in b.label or b.key == "process_multi"), None)
            if btn:
                print("  ⏳ Processing batch (this may take a minute)...")
                btn.click().run(timeout=300)
                if any("successfully" in s.body.lower() for s in at.get("success")):
                    print("  ✅ Batch processed successfully.")
                else:
                    print("  ❌ Success message not found.")
    except Exception as e:
        print(f"  ❌ Batch processing error: {e}")

    # --- TEST 3: Download Reports ---
    print("\n📝 TEST 3: Download Reports")
    at.session_state["batch_active_tab"] = "Download Reports"
    at.run()
    dls = at.get("download_button")
    has_excel = any("Excel" in d.label for d in dls)
    has_html = any("HTML" in d.label for d in dls)
    if has_excel and has_html:
        print("  ✅ Excel and HTML reports are available for the new batch.")
    else:
        print(f"  ❌ Missing reports. Found: {[d.label for d in dls]}")

    # --- TEST 4: Monitoring ---
    print("\n📝 TEST 4: Monitoring Sub-tabs")
    at.session_state["main_active_tab"] = "📉 Monitoring"
    
    monitoring_subs = [
        ("📊 Overview", "Overview"),
        ("🧠 Model Monitoring", "Model"),
        ("⚡ Performance Monitoring", "Performance")
    ]
    
    for tab_val, name in monitoring_subs:
        at.session_state["monitoring_active_tab"] = tab_val
        at.run()
        if at.get("error"):
            print(f"  ❌ Error in Monitoring -> {name}: {at.get('error')[0].body.splitlines()[0]}")
        else:
            print(f"  ✅ Monitoring -> {name} tab is healthy.")

    # --- TEST 5: Data Quality ---
    print("\n📝 TEST 5: Data Quality Sub-tabs")
    at.session_state["monitoring_active_tab"] = "🔍 Data Quality"
    
    # 5.1 Feature Drift
    at.session_state["quality_active_tab"] = "📊 Feature Drift"
    at.run()
    print("  - Checking Feature Drift...")
    # Verify Reference Data label
    text_inputs = at.get("text_input")
    ref_input = next((ti for ti in text_inputs if "Reference Data" in ti.label), None)
    if ref_input and "Training" in ref_input.value:
        print("    ✅ Reference data correctly set to Training Data.")
    
    # Trigger Analyze Drift
    drift_btn = next((b for b in at.get("button") if "Analyze Drift" in b.label), None)
    if drift_btn:
        print("    ⏳ Running drift analysis...")
        drift_btn.click().run(timeout=120)
        if at.get("dataframe"):
            print("    ✅ Drift analysis results displayed.")
        if at.get("error"):
            print(f"    ❌ Drift analysis error: {at.get('error')[0].body.splitlines()[0]}")

    # 5.2 Data Quality Checks
    at.session_state["quality_active_tab"] = "✔️ Data Quality"
    at.run()
    print("  - Checking Data Quality Checks...")
    dq_btn = next((b for b in at.get("button") if "Analyze Data Quality" in b.label), None)
    if dq_btn:
        dq_btn.click().run()
        if not at.get("error"):
            print("    ✅ Data quality analysis completed.")
        else:
            print(f"    ❌ Data quality error: {at.get('error')[0].body.splitlines()[0]}")

    # 5.3 Drift History
    at.session_state["quality_active_tab"] = "📈 Drift History"
    at.run()
    print("  - Checking Drift History...")
    if at.get("selectbox") and not at.get("error"):
        print("    ✅ Drift history tab rendered correctly.")

    print("\n🏁 Comprehensive UI Simulation Complete!")

if __name__ == "__main__":
    try:
        run_full_test()
    except Exception as e:
        print(f"💥 Top level failure: {e}")
        import traceback
        traceback.print_exc()