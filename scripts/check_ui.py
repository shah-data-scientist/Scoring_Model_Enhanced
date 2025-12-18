
import asyncio
from playwright.async_api import async_playwright
import os
from pathlib import Path

# Project configuration
BASE_URL = "http://127.0.0.1:8501"
USERNAME = "admin"
PASSWORD = "admin123"

async def run_simulation():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print(f"🚀 Navigating to {BASE_URL}...")
        await page.goto(BASE_URL)

        print("🔐 Logging in...")
        await page.get_by_label("Username").fill(USERNAME)
        await page.get_by_label("Password").first.fill(PASSWORD)
        await page.get_by_role("button", name="Login").click()

        print("⏳ Waiting for dashboard...")
        await page.wait_for_timeout(10000) # Wait 10s for everything to settle
        
        await page.screenshot(path="dashboard_check.png")
        print("📸 Dashboard screenshot saved to dashboard_check.png")
        
        # List all tabs found
        tabs = await page.get_by_role("tab").all()
        print(f"🔍 Found {len(tabs)} tabs:")
        for i, tab in enumerate(tabs):
            print(f"  [{i}] {await tab.get_attribute('aria-label')} | Text: {await tab.inner_text()}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_simulation())
