"""Save a screenshot of every dashboard page to docs/reports/screenshots, as a demo backup.

Starts the real server on the real database. Pages are saved in full.

Usage: python src/scripts/capture_screenshots.py
"""
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "reports" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 8003
PAGES = {"Main": "home", "Upload": "upload", "Alerts": "alerts", "Analytics": "analytics", "Accuracy": "accuracy"}

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.server:app", "--port", str(PORT), "--log-level", "warning"], cwd=ROOT)
failed = False
try:
    time.sleep(6)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        problems = []
        page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: problems.append(str(e)))
        for name, label in PAGES.items():
            page.goto(f"http://localhost:{PORT}/{name}.dc.html")
            page.wait_for_function("() => document.querySelectorAll('.skel').length === 0", timeout=30000)
            page.wait_for_timeout(2500)
            path = OUT / f"dashboard_{label}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"{path.name}: {path.stat().st_size // 1024} KB, page problems so far: {len(problems)}")
        failed = bool(problems)
        for problem in problems[:5]:
            print("PROBLEM:", problem[:200])
        browser.close()
finally:
    server.terminate()
    server.wait(timeout=20)
print("SCREENSHOTS_DONE" if not failed else "SCREENSHOTS_HAD_PROBLEMS")
sys.exit(1 if failed else 0)
