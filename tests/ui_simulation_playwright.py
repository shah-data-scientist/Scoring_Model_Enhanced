"Playwright UI Simulation with comprehensive tab/button testing and API monitoring."

import asyncio
import httpx
import re
from playwright.async_api import async_playwright, expect
from pathlib import Path
import os
import sys

# Ensure UTF-8 output for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Project configuration
BASE_URL = "http://localhost:8501"
API_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "admin123"

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)


async def check_api_health():
    """Check if API is running and healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                status = "healthy" if data.get("status") == "healthy" else "unhealthy"
                return True, f"API {status.upper()} - Model Loaded: {data.get('model_loaded')}"
            return False, f"API returned {response.status_code}"
    except Exception as e:
        return False, f"API unreachable: {str(e)[:50]}"


async def run_simulation():
    # Check API before starting
    api_ok, api_msg = await check_api_health()
    print(f"[API CHECK] {api_msg}")
    if not api_ok:
        print("[FATAL] API is not running. Start it with: poetry run uvicorn api.app:app")
        return False

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1600, 'height': 1000},
            accept_downloads=True
        )
        page = await context.new_page()

        success = True
        step_counter = 0

        async def screenshot(name):
            nonlocal step_counter
            step_counter += 1
            path = SCREENSHOTS_DIR / f"{step_counter:02d}_{name}.png"
            # Disable full_page to avoid Streamlit sticky element artifacts
            await page.screenshot(path=str(path), full_page=False)
            print(f"  [SCREENSHOT] {path.name}")

        async def api_check(label):
            ok, msg = await check_api_health()
            status = "[OK]" if ok else "[FAIL]"
            print(f"  [API {label}] {status} {msg}")
            return ok

        async def click_tab(name_regex):
            """Robustly click a Streamlit tab and wait for content to be visible."""
            tab = page.locator('button[role="tab"]').filter(has_text=name_regex).first
            if await tab.count() > 0:
                print(f"  [ACTION] Clicking tab: {name_regex.pattern if hasattr(name_regex, 'pattern') else name_regex}...")
                await tab.scroll_into_view_if_needed()
                await tab.click(force=True)
                # Wait for Streamlit to switch tabs and render content
                await page.wait_for_timeout(5000)
                return True
            return False

        async def check_results(marker_text, screenshot_name):
            """Check if results are present on the page."""
            print(f"  [CHECK] Verifying results for {marker_text}...")
            # Look for success messages or dataframes
            success_indicators = [
                page.get_by_text(re.compile(marker_text, re.I)),
                page.locator(".stDataFrame"),
                page.locator(".js-plotly-plot")
            ]
            
            found = False
            for indicator in success_indicators:
                if await indicator.first.is_visible():
                    found = True
                    break
            
            if found:
                print(f"    [OK] Results detected")
            else:
                print(f"    [WARN] No immediate results detected, waiting a bit longer...")
                await page.wait_for_timeout(3000)
            
            await screenshot(screenshot_name)
            return found

        try:
            # Step 1: Navigate to app
            print(f"\n[STEP] Navigating to {BASE_URL}...")
            # Increased timeout for Docker cold start
            await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
            await page.wait_for_timeout(5000)
            await screenshot("01_01_initial_load")

            # Step 2: Login
            print(f"\n[STEP] Logging in as {USERNAME}...")
            await page.get_by_role("textbox", name="Username").fill(USERNAME)
            await page.get_by_role("textbox", name="Password").fill(PASSWORD)
            await page.get_by_role("textbox", name="Password").press("Enter")
            # Increase wait for dashboard load
            await page.wait_for_timeout(15000) 
            await screenshot("02_02_after_login")

            # Step 3: Threshold Interaction
            print(f"\n[STEP] Testing Threshold Interaction...")
            if await click_tab(re.compile(r"Model Performance", re.I)):
                threshold_indicator = page.locator("div").filter(has_text=re.compile(r"^0\.[0-9]+$")).first
                if await threshold_indicator.count() > 0:
                    print(f"  [ACTION] Clicking threshold indicator...")
                    await threshold_indicator.click()
                    await page.wait_for_timeout(2000)
                    await check_results(r"Confusion Matrix", "03_threshold_result")
                else:
                    print("  [WARN] Threshold indicator not found")

            # Step 4: Batch Predictions
            print(f"\n[STEP] Testing Batch Predictions...")
            if await click_tab(re.compile(r"Batch Predictions", re.I)):
                await click_tab(re.compile(r"Upload & Predict", re.I))

                file_names = ["application.csv", "bureau.csv", "bureau_balance.csv", "credit_card_balance.csv", "installments_payments.csv", "POS_CASH_balance.csv", "previous_application.csv"]
                files_to_upload = [str(SAMPLES_DIR / fn) for fn in file_names if (SAMPLES_DIR / fn).exists()]
                
                if len(files_to_upload) >= 7:
                    print(f"  [ACTION] Uploading {len(files_to_upload)} files...")
                    await page.set_input_files("input[type='file']", files_to_upload)
                    print("  [WAIT] Processing uploads...")
                    await page.wait_for_timeout(15000)
                    
                    process_btn = page.locator("button").filter(has_text=re.compile(r"Process Batch", re.I)).first
                    if await process_btn.is_visible():
                        print("  [ACTION] Clicking Process Batch...")
                        await process_btn.click(force=True)
                        print("  [WAIT] Processing (max 60s)...")
                        await page.wait_for_timeout(20000) # Initial wait
                        await check_results(r"successfully|Predictions", "04_batch_result")
                    else:
                        print("  [ERROR] Process Batch button not visible after upload")
                        await screenshot("04_upload_error")

            # Step 5: Monitoring
            print(f"\n[STEP] Testing Monitoring...")
            if await click_tab(re.compile(r"Monitoring", re.I)):
                m_tabs = ["📊 Overview", "🧠 Model Monitoring", "🔍 Data Quality"]
                for m_tab in m_tabs:
                    if await click_tab(m_tab):
                        if "Data Quality" in m_tab:
                            # Nested tabs in Data Quality
                            for sub in ["📊 Feature Drift", "✔️ Data Quality"]:
                                if await click_tab(sub):
                                    analyze_btn = page.locator("button").filter(has_text=re.compile(r"Analyze", re.I)).first
                                    if await analyze_btn.is_visible():
                                        print(f"      [ACTION] Clicking Analyze in {sub}...")
                                        await analyze_btn.click(force=True)
                                        await page.wait_for_timeout(8000)
                                        
                                        # Verification for anomaly plots if in Data Quality tab
                                        if "Data Quality" in sub:
                                            # Switch to Out-of-Range tab if it's there
                                            await click_tab(re.compile(r"Out-of-Range", re.I))
                                            await check_results(r"Anomaly Visualizations", f"05_monitoring_anomaly_check")
                                        else:
                                            await check_results(r"Analysis Complete|Missing Values", f"05_monitoring_{sub[2:].strip()}")
                        else:
                            await check_results(r"Status|History", f"05_monitoring_{m_tab[2:].strip()}")

            print(f"\n[STEP] Final State Check...")
            await api_check("final")

        except Exception as e:
            print(f"\n[FATAL ERROR] {e}")
            await screenshot("99_fatal_error")
            success = False


            # Final API Health Check
            print(f"\n[STEP] Final State Check...")
            api_ok, api_msg = await check_api_health()
            print(f"  [API CHECK] {api_msg}")
            if not api_ok:
                success = False

        except Exception as e:
            print(f"\n[FATAL ERROR] {e}")
            await screenshot("fatal_error")
            success = False

        finally:
            await browser.close()

        return success


async def main():
    print("=" * 60)
    print("FINAL REFINED PLAYWRIGHT UI SIMULATION")
    print("=" * 60)

    success = await run_simulation()

    print("\n" + "=" * 60)
    if success:
        print("SIMULATION COMPLETED")
    else:
        print("SIMULATION FAILED")
    print("=" * 60)
    print(f"Screenshots saved to: {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
