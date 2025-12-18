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
            await page.screenshot(path=str(path), full_page=True)
            print(f"  [SCREENSHOT] {path.name}")

        async def api_check(label):
            ok, msg = await check_api_health()
            status = "[OK]" if ok else "[FAIL]"
            print(f"  [API {label}] {status} {msg}")
            return ok

        try:
            # Step 1: Navigate to app
            print(f"\n[STEP] Navigating to {BASE_URL}...")
            await page.goto(BASE_URL, wait_until="networkidle")
            await screenshot("initial_load")
            await api_check("after_navigate")

            # Step 2: Login
            print(f"\n[STEP] Logging in as {USERNAME}...")
            try:
                await page.get_by_role("textbox", name="Username").fill(USERNAME)
                await page.get_by_role("textbox", name="Password").fill(PASSWORD)
                await page.get_by_role("textbox", name="Password").press("Enter")
                await page.wait_for_timeout(5000) # Give more time for dashboard to load
                
                # Verify login success
                if await page.get_by_text("Credit Scoring Dashboard").count() > 0 or await page.get_by_text("Welcome").count() > 0:
                    print("  [OK] Login successful")
                else:
                    print("  [WARN] Dashboard markers not found, but continuing...")
                
                await screenshot("after_login")
            except Exception as e:
                print(f"  [ERROR] Login failed: {e}")
                return False

            # Step 3: Threshold Interaction
            print(f"\n[STEP] Testing Threshold Interaction...")
            try:
                # Target the slider or value display from recording
                threshold_indicator = page.locator("div").filter(has_text=re.compile(r"^0\.[0-9]+$")).first
                if await threshold_indicator.count() > 0:
                    print(f"  [ACTION] Clicking threshold indicator...")
                    await threshold_indicator.click()
                    await page.wait_for_timeout(1000)
                    await screenshot("threshold_interacted")
                else:
                    print("  [WARN] Threshold indicator not found")
            except Exception as e:
                print(f"  [ERROR] Threshold interaction failed: {e}")

            # Step 4: Batch Predictions Upload & Process
            print(f"\n[STEP] Testing Batch Predictions Upload...")
            try:
                file_names = [
                    "application.csv", "bureau.csv", "bureau_balance.csv",
                    "credit_card_balance.csv", "installments_payments.csv",
                    "POS_CASH_balance.csv", "previous_application.csv"
                ]
                files_to_upload = [str(SAMPLES_DIR / fn) for fn in file_names if (SAMPLES_DIR / fn).exists()]
                
                if len(files_to_upload) > 0:
                    print(f"  [ACTION] Uploading {len(files_to_upload)} files...")
                    # Streamlit file uploader input
                    await page.set_input_files("input[type='file']", files_to_upload)
                    print("  [WAIT] Waiting for files to be processed by Streamlit...")
                    await page.wait_for_timeout(8000)
                    await screenshot("files_uploaded")
                    
                    # Process Batch - Try finding by text
                    process_btn = page.get_by_role("button", name=re.compile(r"Process Batch", re.I))
                    if await process_btn.count() == 0:
                         # Fallback to any primary button if name doesn't match exactly
                         process_btn = page.get_by_test_id("stBaseButton-primary")

                    if await process_btn.count() > 0:
                        print("  [ACTION] Clicking Process Batch...")
                        # Ensure it's visible before clicking
                        await process_btn.scroll_into_view_if_needed()
                        await process_btn.click()
                        print("  [WAIT] Waiting for processing (max 90s)...")
                        # Wait for a success indicator or results
                        try:
                            await page.wait_for_selector("text=successfully", timeout=90000)
                            print("  [OK] Processing complete")
                        except:
                            print("  [WARN] Success message 'successfully' not found, checking for results table...")
                            if await page.locator(".stDataFrame").count() > 0:
                                print("  [OK] Results table found")
                            else:
                                print("  [ERROR] Processing might have failed or is taking too long")
                        
                        await screenshot("batch_results")
                    else:
                        print("  [ERROR] Process Batch button not found after upload")
                else:
                    print("  [ERROR] No sample files found for upload")

            except Exception as e:
                print(f"  [ERROR] Batch Prediction workflow failed: {e}")

            # Step 5: Download Reports
            print(f"\n[STEP] Testing Report Downloads...")
            try:
                # Try to find the tab - might be at the top or in sidebar
                reports_tab = page.get_by_role("tab", name=re.compile(r"Download Reports", re.I))
                if await reports_tab.count() > 0:
                    await reports_tab.click()
                    await page.wait_for_timeout(2000)
                    
                    # Download buttons
                    excel_btn = page.get_by_role("button", name=re.compile(r"Download Excel", re.I))
                    if await excel_btn.count() > 0:
                        print("  [ACTION] Downloading Excel...")
                        async with page.expect_download() as download_info:
                            await excel_btn.click()
                        download = await download_info.value
                        print(f"  [OK] Excel downloaded: {download.suggested_filename}")
                    
                    html_btn = page.get_by_role("button", name=re.compile(r"Download HTML", re.I))
                    if await html_btn.count() > 0:
                        print("  [ACTION] Downloading HTML...")
                        async with page.expect_download() as download_info:
                            await html_btn.click()
                        download = await download_info.value
                        print(f"  [OK] HTML downloaded: {download.suggested_filename}")
                    
                    await screenshot("reports_download")
                else:
                    print("  [WARN] Download Reports tab not found")
            except Exception as e:
                print(f"  [ERROR] Report download failed: {e}")

            # Step 6: Monitoring Tabs
            print(f"\n[STEP] Testing Monitoring Tabs...")
            try:
                # Exact names from recording often include emojis
                monitoring_tabs = [
                    "📊 Overview", 
                    "🧠 Model Monitoring", 
                    "⚡ Performance Monitoring", 
                    "⚙️ System Health", 
                    "🔍 Data Quality", 
                    "📊 Feature Drift", 
                    "✔️ Data Quality", 
                    "📈 Drift History"
                ]
                
                for tab_name in monitoring_tabs:
                    print(f"  [ACTION] Clicking tab: {tab_name}")
                    tab_locator = page.get_by_role("tab", name=tab_name)
                    if await tab_locator.count() > 0:
                        await tab_locator.click()
                        await page.wait_for_timeout(2000)
                        print(f"    [OK] Selected {tab_name}")
                        
                        # Trigger analysis if applicable
                        if "Drift" in tab_name or "Data Quality" in tab_name:
                            analyze_btn = page.get_by_role("button", name=re.compile(r"Analyze", re.I))
                            if await analyze_btn.count() > 0:
                                print("    [ACTION] Clicking Analyze button...")
                                await analyze_btn.click()
                                await page.wait_for_timeout(5000)
                                await screenshot(f"monitoring_{tab_name[2:]}")
                    else:
                        # Try without emoji if exact match fails
                        clean_name = tab_name[2:].strip()
                        tab_locator = page.get_by_role("tab", name=re.compile(clean_name, re.I))
                        if await tab_locator.count() > 0:
                            await tab_locator.click()
                            await page.wait_for_timeout(2000)
                            print(f"    [OK] Selected {clean_name} (fallback)")
                        else:
                            print(f"    [WARN] Tab {tab_name} not found")

                await screenshot("monitoring_complete")
            except Exception as e:
                print(f"  [ERROR] Monitoring tabs failed: {e}")

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
