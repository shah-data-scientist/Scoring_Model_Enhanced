import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("http://localhost:8501/")
    page.get_by_role("textbox", name="Username").click()
    page.get_by_role("textbox", name="Username").fill("admin")
    page.get_by_role("textbox", name="Username").press("Tab")
    page.get_by_role("textbox", name="Password").fill("admin123")
    page.get_by_role("textbox", name="Password").press("Enter")
    page.locator("div").filter(has_text=re.compile(r"^0\.33$")).first.click()
    page.get_by_role("img", name="0").first.click()
    page.get_by_test_id("stFileUploaderDropzone").get_by_test_id("stBaseButton-secondary").click()
    page.get_by_test_id("stFileUploaderDropzone").get_by_test_id("stBaseButton-secondary").set_input_files(["application.csv", "bureau.csv", "bureau_balance.csv", "credit_card_balance.csv", "installments_payments.csv", "POS_CASH_balance.csv", "previous_application.csv"])
    page.get_by_test_id("stBaseButton-primary").click()
    page.get_by_role("heading", name="Predictions").click()
    page.get_by_role("heading", name="Predictions").click()
    page.locator(".dvn-scroller").click()
    page.get_by_role("tab", name="Download Reports").click()
    page.get_by_text("[OK] Batch_185 2025-12-18 17:39:").click()
    page.get_by_text("keyboard_arrow_down").click()
    page.get_by_role("group").filter(has_text="keyboard_arrow_right[OK] Batch_185 2025-12-18 17:39:56Status: completedTotal").get_by_test_id("stIconMaterial").click()
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Download Excel").click()
    download = download_info.value
    with page.expect_download() as download1_info:
        page.get_by_role("button", name="Download HTML Report").click()
    download1 = download1_info.value
    page.get_by_role("tab", name="📊 Overview").click()
    page.get_by_role("tab", name="🧠 Model Monitoring").click()
    page.locator(".dvn-scroller").click()
    page.get_by_role("tab", name="⚡ Performance Monitoring").click()
    page.get_by_role("tab", name="⚙️ System Health").click()
    page.get_by_role("tab", name="🔍 Data Quality").click()
    page.get_by_role("tab", name="📊 Feature Drift").click()
    page.get_by_role("img", name="open").click()
    page.get_by_test_id("stSelectboxVirtualDropdown").get_by_text("Batch 185 -").click()
    page.get_by_role("button", name="🔍 Analyze Drift").click()
    page.locator(".dvn-scroller").click()
    page.get_by_role("tab", name="✔️ Data Quality").click()
    page.locator("div").filter(has_text=re.compile(r"^Batch 185 - batch_20251218_230956$")).nth(2).click()
    page.get_by_test_id("stSelectboxVirtualDropdown").get_by_text("Batch 185 -").click()
    page.get_by_role("button", name="🔍 Analyze Data Quality").click()
    page.locator(".dvn-scroller").click()
    page.get_by_role("tab", name="📈 Drift History").click()
    page.locator("div").filter(has_text=re.compile(r"^EXT_SOURCE_1$")).first.click()
    page.get_by_text("EXT_SOURCE_2").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
