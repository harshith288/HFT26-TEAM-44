"""Real-browser upload test: 3 real golden images plus 1 corrupted real file.

Starts the actual dashboard on a clean temporary database, because every golden image is
already in the main database and would only return "duplicate". Uploads through Edge,
never reloads the page, and waits for results to appear. Writes docs/reports/upload_e2e.json.

Usage: python src/scripts/e2e_upload.py
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
PORT = 8502
expected = json.loads((ROOT / "tests" / "golden_expected.json").read_text())
picks = [next(g for g in expected if g["labels"] == [s]) for s in ("bobcat", "coyote", "empty")]
results = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": 4,
           "hardware": "RTX 4050 pipeline, Edge browser", "checks": {}}

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    files = []
    for g in picks:
        files.append(folder / g["file_name"])
        shutil.copy(ROOT / "data" / "golden" / g["file_name"], files[-1])
    data = files[0].read_bytes()
    broken = folder / "corrupted_photo.jpg"
    broken.write_bytes(data[:len(data) // 3])
    files.append(broken)

    db = folder / "upload_test.db"
    sqlite3.connect(db).close()
    env = {**os.environ, "WILDWATCH_DB": str(db)}
    # Create the tables with the real code, in a process without torch.
    subprocess.run([sys.executable, "-c", f"from src.db import init_db; init_db(r'{db}').close()"], cwd=ROOT, check=True, env=env)
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py", "--server.port", str(PORT),
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(6)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 1500})
            navigations = []
            page.on("framenavigated", lambda f: navigations.append(f.url))
            page.goto(f"http://localhost:{PORT}")
            page.get_by_role("tab", name="Upload").wait_for(timeout=30000)
            loads_after_start = len(navigations)
            page.get_by_role("tab", name="Upload").click()
            page.locator('input[type="file"]').set_input_files([str(f) for f in files])
            page.get_by_role("button", name="Analyse photos").click(timeout=20000)
            started = time.time()
            page.get_by_text("Analysis finished").wait_for(timeout=280000)
            seconds = time.time() - started
            page.wait_for_timeout(1500)
            text = page.inner_text("body")
            page.screenshot(path=str(ROOT / "docs" / "reports" / "screenshots" / "upload_result.png"))
            page.get_by_role("tab", name="Live alerts").click()
            page.wait_for_timeout(1500)
            alerts_text = page.inner_text("body")
            page.screenshot(path=str(ROOT / "docs" / "reports" / "screenshots" / "upload_alert_in_live_panel.png"))
            results["checks"]["no_page_reload"] = len(navigations) == loads_after_start
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=20)

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT file_path, status, error, triage FROM images").fetchall()
    n_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    species = sorted(r[0] for r in conn.execute("SELECT species FROM alerts"))
    messages = [{"species": r[0], "level": r[1], "source": r[2], "message": r[3]}
                for r in conn.execute("SELECT species, level, message_source, message FROM alerts")]
    conn.close()

results["seconds_until_result"] = round(seconds, 1)
results["database_rows"] = [{"file": Path(r[0]).name, "status": r[1], "error": (r[2] or "")[:90], "triage": r[3]} for r in rows]
results["alerts_in_database"] = n_alerts
results["alert_species"] = species
results["alert_messages"] = messages
results["checks"].update({
    "four_files_stored": len(rows) == 4,
    "corrupted_marked_unreadable": any(r[1] == "unreadable" and "corrupted_photo" in r[0] for r in rows),
    "corrupted_shown_to_user": "corrupted_photo.jpg: could not be read" in text,
    "two_alerts_expected_species": species == ["bobcat", "coyote"],
    "empty_image_raised_no_alert": n_alerts == 2,
    "alert_visible_in_upload_tab": "bobcat" in text.lower() and "coyote" in text.lower(),
    "alert_visible_in_live_panel": "bobcat" in alerts_text.lower() and "coyote" in alerts_text.lower(),
})
(ROOT / "docs" / "reports" / "upload_e2e.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
print("E2E_PASSED" if all(results["checks"].values()) else "E2E_FAILED")
sys.exit(0 if all(results["checks"].values()) else 1)
