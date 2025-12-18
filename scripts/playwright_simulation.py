"""Playwright UI Simulation with comprehensive tab/button testing and API monitoring."""

import asyncio
import httpx
from playwright.async_api import async_playwright
from pathlib import Path

# Project configuration
BASE_URL = "http://127.0.0.1:8501"
API_URL = "http://127.0.0.1:8000"
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
                return True, f"API OK - Model: {data.get('model_status', 'unknown')}"
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
        context = await browser.new_context(viewport={'width': 1400, 'height': 900})
        page = await context.new_page()

        success = True
        step = 0

        async def screenshot(name):
            path = SCREENSHOTS_DIR / f"{step:02d}_{name}.png"
            await page.screenshot(path=str(path))
            print(f"  [SCREENSHOT] {path.name}")

        async def api_check(label):
            ok, msg = await check_api_health()
            status = "[OK]" if ok else "[FAIL]"
            print(f"  [API {label}] {status} {msg}")
            return ok

        try:
            # Step 1: Navigate to app
            step = 1
            print(f"\n[STEP {step}] Navigating to {BASE_URL}...")
            await page.goto(BASE_URL, wait_until="networkidle")
            await screenshot("01_initial_page")
            await api_check("after_navigate")

            # Step 2: Login
            step = 2
            print(f"\n[STEP {step}] Logging in as {USERNAME}...")
            try:
                await page.get_by_label("Username").fill(USERNAME)
                await page.get_by_label("Password").first.fill(PASSWORD)
                await screenshot("02_login_filled")
                await page.get_by_role("button", name="Login").click()
                await page.wait_for_timeout(3000)
                await screenshot("02_after_login")
            except Exception as e:
                print(f"  [ERROR] Login failed: {e}")
                await screenshot("02_login_error")
                return False

            # Wait for dashboard
            try:
                await page.wait_for_selector("text=Credit Scoring Dashboard", timeout=30000)
                print("  [OK] Dashboard loaded")
            except:
                print("  [WARN] Dashboard text not found, continuing...")

            await api_check("after_login")

            # Step 3: Click Model Performance tab
            step = 3
            print(f"\n[STEP {step}] Testing Model Performance tab...")
            try:
                tab = page.locator("div[data-baseweb='tab']").filter(has_text="Model Performance")
                if await tab.count() > 0:
                    await tab.first.click()
                    await page.wait_for_timeout(3000)
                    await screenshot("03_model_performance")
                    print("  [OK] Model Performance tab clicked")
                else:
                    print("  [SKIP] Model Performance tab not found")
            except Exception as e:
                print(f"  [ERROR] {e}")

            await api_check("after_model_perf")

            # Step 4: Click Batch Predictions tab
            step = 4
            print(f"\n[STEP {step}] Testing Batch Predictions tab...")
            try:
                tab = page.locator("div[data-baseweb='tab']").filter(has_text="Batch Predictions")
                if await tab.count() > 0:
                    await tab.first.click()
                    await page.wait_for_timeout(3000)
                    await screenshot("04_batch_predictions")
                    print("  [OK] Batch Predictions tab clicked")
                else:
                    print("  [SKIP] Batch Predictions tab not found")
            except Exception as e:
                print(f"  [ERROR] {e}")

            await api_check("after_batch_tab")

            # Step 5: Click Monitoring tab (if exists)
            step = 5
            print(f"\n[STEP {step}] Testing Monitoring tab...")
            try:
                tab = page.locator("div[data-baseweb='tab']").filter(has_text="Monitoring")
                if await tab.count() > 0:
                    await tab.first.click()
                    await page.wait_for_timeout(3000)
                    await screenshot("05_monitoring")
                    print("  [OK] Monitoring tab clicked")
                else:
                    print("  [SKIP] Monitoring tab not found")
            except Exception as e:
                print(f"  [ERROR] {e}")

            await api_check("after_monitoring")

            # Step 6: Go back to Batch Predictions and upload files
            step = 6
            print(f"\n[STEP {step}] Uploading CSV files...")
            try:
                tab = page.locator("div[data-baseweb='tab']").filter(has_text="Batch Predictions")
                await tab.first.click()
                await page.wait_for_timeout(2000)

                file_names = [
                    "application.csv", "bureau.csv", "bureau_balance.csv",
                    "credit_card_balance.csv", "installments_payments.csv",
                    "POS_CASH_balance.csv", "previous_application.csv"
                ]
                files_to_upload = [str(SAMPLES_DIR / fn) for fn in file_names if (SAMPLES_DIR / fn).exists()]

                if len(files_to_upload) < 7:
                    print(f"  [WARN] Only {len(files_to_upload)}/7 files found")

                # Try to find file uploader
                try:
                    async with page.expect_file_chooser(timeout=10000) as fc_info:
                        browse_btn = page.get_by_test_id("stFileUploader").get_by_text("Browse files")
                        if await browse_btn.count() > 0:
                            await browse_btn.click()
                        else:
                            # Fallback
                            await page.locator("input[type='file']").first.set_input_files(files_to_upload)
                            raise Exception("Used fallback input method")
                    file_chooser = await fc_info.value
                    await file_chooser.set_files(files_to_upload)
                    print(f"  [OK] Uploaded {len(files_to_upload)} files")
                except Exception as e:
                    print(f"  [WARN] File chooser method: {e}")
                    # Try direct input method
                    try:
                        await page.locator("input[type='file']").first.set_input_files(files_to_upload)
                        print(f"  [OK] Uploaded via direct input")
                    except:
                        print(f"  [ERROR] Could not upload files")

                await page.wait_for_timeout(5000)
                await screenshot("06_files_uploaded")

            except Exception as e:
                print(f"  [ERROR] Upload step: {e}")
                await screenshot("06_upload_error")

            await api_check("after_upload")

            # Step 7: Click Process Batch button
            step = 7
            print(f"\n[STEP {step}] Processing batch...")
            try:
                # Find Process Batch button
                clicked = await page.evaluate("""
                    () => {
                        const buttons = Array.from(document.querySelectorAll('button'));
                        const btn = buttons.find(b => b.innerText.includes('Process Batch'));
                        if (btn) { btn.click(); return true; }
                        return false;
                    }
                """)
                if clicked:
                    print("  [OK] Process Batch button clicked")
                    await page.wait_for_timeout(2000)
                    await screenshot("07_processing_started")

                    # Wait for results (up to 5 minutes)
                    print("  [WAIT] Waiting for batch processing (up to 5 min)...")
                    try:
                        await page.wait_for_selector("text=successfully", timeout=300000)
                        print("  [OK] Batch processed successfully!")
                        await screenshot("07_batch_success")
                    except:
                        print("  [TIMEOUT] Batch processing timed out or no success message")
                        await screenshot("07_batch_timeout")
                else:
                    print("  [SKIP] Process Batch button not found (no files uploaded?)")

            except Exception as e:
                print(f"  [ERROR] Processing: {e}")
                await screenshot("07_process_error")

            await api_check("after_process")

            # Step 8: Check all visible buttons
            step = 8
            print(f"\n[STEP {step}] Listing all visible buttons...")
            try:
                buttons = await page.evaluate("""
                    () => Array.from(document.querySelectorAll('button'))
                           .map(b => b.innerText.trim())
                           .filter(t => t.length > 0 && t.length < 50)
                """)
                print(f"  [INFO] Found {len(buttons)} buttons: {buttons[:10]}...")
            except Exception as e:
                print(f"  [ERROR] {e}")

            # Step 9: Try clicking History tab if exists
            step = 9
            print(f"\n[STEP {step}] Testing History sub-tab...")
            try:
                history_tab = page.locator("div[data-baseweb='tab']").filter(has_text="History")
                if await history_tab.count() > 0:
                    await history_tab.first.click()
                    await page.wait_for_timeout(2000)
                    await screenshot("09_history_tab")
                    print("  [OK] History tab clicked")
                else:
                    print("  [SKIP] History tab not found")
            except Exception as e:
                print(f"  [ERROR] {e}")

            await api_check("after_history")

            # Step 10: Final screenshot and summary
            step = 10
            print(f"\n[STEP {step}] Final state...")
            await screenshot("10_final_state")

            # Final API check
            api_ok, api_msg = await check_api_health()
            print(f"\n[FINAL API CHECK] {api_msg}")

        except Exception as e:
            print(f"\n[FATAL ERROR] {e}")
            await screenshot("fatal_error")
            success = False

        finally:
            await browser.close()

        return success


async def main():
    print("=" * 60)
    print("PLAYWRIGHT UI SIMULATION WITH API MONITORING")
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
