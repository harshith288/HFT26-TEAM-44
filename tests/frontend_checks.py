"""Real-browser checks of the dashboard: the real server, real data and installed Edge.

Runs on a temporary copy of the database with LLM calls off. tests/test_frontend.py runs
this file in its own process.
"""
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
PAGES = ["Main", "Upload", "Alerts", "Analytics", "Accuracy"]


def start(db, port):
    env = {**os.environ, "WILDWATCH_DB": str(db), "FEATHERLESS_API_KEY": ""}
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.server:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/summary", timeout=2).status_code == 200:
                return process
        except httpx.HTTPError:
            time.sleep(0.3)
    process.kill()
    raise AssertionError("server did not start")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    db = tmp_path_factory.mktemp("frontend") / "copy.db"
    shutil.copy(ROOT / CONFIG["paths"]["database"], db)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    process = start(db, port)
    yield {"url": f"http://127.0.0.1:{port}", "db": db, "port": port, "process": process}
    process.terminate()
    process.wait(timeout=20)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        instance = p.chromium.launch(channel="msedge", headless=True)
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1400, "height": 1000})
    tab = context.new_page()
    tab.problems = []
    tab.on("console", lambda m: tab.problems.append(f"console: {m.text}") if m.type == "error" else None)
    tab.on("pageerror", lambda e: tab.problems.append(f"pageerror: {e}"))
    tab.on("response", lambda r: tab.problems.append(f"{r.status}: {r.url}") if r.status >= 400 else None)
    yield tab
    context.close()


def api(site, path):
    return httpx.get(site["url"] + path, timeout=30).json()


def open_page(page, site, name):
    page.goto(f"{site['url']}/{name}.dc.html")
    page.wait_for_function("() => document.querySelectorAll('.skel').length === 0", timeout=30000)
    page.wait_for_timeout(1800)
    return page.inner_text("body")


@pytest.mark.parametrize("name", PAGES)
def test_page_has_no_errors_and_no_unfilled_values(page, site, name):
    text = open_page(page, site, name)
    assert page.problems == []
    assert "{{" not in text
    assert "undefined" not in text and "NaN" not in text and "[object" not in text
    for label in ("Sample data", "sample data", "Sample alert", "Sample result", "layout review"):
        assert label not in text
    assert page.locator('[role="alert"]').count() == 0


def test_root_opens_the_home_page(page, site):
    page.goto(site["url"] + "/")
    expect(page.locator("h1")).to_contain_text("Ranger")


def test_home_shows_real_three_tier_triage(page, site):
    text = open_page(page, site, "Main")
    s = api(site, "/api/summary")
    assert f"{s['images']:,}" in text
    assert "3-tier triage" in text and "empty frames auto-removed" not in text.lower()
    assert f"{s['triage']['kept']:,}" in text
    assert f"{s['triage']['needs_review']} needs review" in text
    assert f"{s['triage']['auto_discarded']} auto-discarded" in text
    assert "88.3" not in text, "the Caltech accuracy belongs only on the Accuracy page"
    assert "3,682" in text and "22,446" in text and "13,874" in text


def test_home_daily_report_shows_a_real_summary(page, site):
    open_page(page, site, "Main")
    page.get_by_role("button", name="Daily report").click()
    expect(page.get_by_label("Daily report")).to_contain_text("Latest daily report")


def test_nav_badge_and_pipeline_box_are_real(page, site):
    text = open_page(page, site, "Analytics")
    s = api(site, "/api/summary")
    assert str(s["open_alerts"]) == page.locator('nav a[href="Alerts.dc.html"] span').inner_text()
    assert ("Watching incoming folder" if s["watcher"]["running"] else "Folder watcher is off") in text
    assert "Last image" in text


def test_alert_shows_the_real_photo_box_and_message(page, site):
    text = open_page(page, site, "Alerts")
    first = api(site, "/api/alerts")["alerts"][0]
    assert first["message"] in text
    assert "[Camera frame" not in text
    image = page.locator('img[src^="/api/images/"]').first
    expect(image).to_be_visible()
    page.wait_for_function("() => { const i = document.querySelector('img[src^=\"/api/images/\"]'); return i.complete && i.naturalWidth > 0; }")
    assert "written by LLM" in text
    assert f"Medium {CONFIG['risk_levels']['medium_min']}" in text and f"High {CONFIG['risk_levels']['high_min']}" in text
    assert f"Critical {CONFIG['risk_levels']['critical_min']}" in text


def test_template_message_is_not_labeled_as_llm(page, site):
    alert = next(a for a in api(site, "/api/alerts")["alerts"] if a["message_source"] == "template")
    page.goto(f"{site['url']}/Alerts.dc.html?alert={alert['alert_id']}")
    expect(page.get_by_label("Alert detail")).to_contain_text("from the template, not the LLM", timeout=20000)
    expect(page.get_by_label("Alert detail")).to_contain_text(alert["message"])


def test_alert_filters_match_the_api(page, site):
    open_page(page, site, "Alerts")
    high = api(site, "/api/alerts?level=HIGH")["total"]
    poachers = sum(a["possible_poacher"] for a in api(site, "/api/alerts")["alerts"])
    assert page.get_by_role("button", name=re.compile(r"^All · ")).count() == 1
    assert poachers == int(page.get_by_role("button", name=re.compile("^Poacher · ")).inner_text().split("· ")[1])
    page.get_by_role("button", name=re.compile("^High · ")).click()
    label = page.get_by_role("button", name=re.compile("^High · ")).inner_text()
    assert int(label.split("· ")[1]) <= high
    assert page.get_by_label("Alert list").locator("button").count() >= 1


def test_needs_review_queue_is_visible(page, site):
    text = open_page(page, site, "Alerts")
    total = api(site, "/api/review")["total"]
    assert f"of {total} bursts" in text


def test_dispatch_flow_persists_through_a_reload(page, site):
    open_page(page, site, "Alerts")
    alert = next(a for a in api(site, "/api/alerts")["alerts"] if a["status"] == "new")
    page.goto(f"{site['url']}/Alerts.dc.html?alert={alert['alert_id']}")
    detail = page.get_by_label("Alert detail")
    expect(detail).to_contain_text("Awaiting action", timeout=20000)
    page.get_by_role("button", name="Dispatch team").click()
    expect(detail).to_contain_text("Team dispatched", timeout=15000)
    page.reload()
    expect(page.get_by_label("Alert detail")).to_contain_text("Team dispatched", timeout=20000)
    conn = sqlite3.connect(site["db"])
    row = conn.execute("SELECT status FROM alert_status WHERE burst_id = ? AND species = ?", (alert["burst_id"], alert["species"])).fetchone()
    conn.close()
    assert row[0] == "dispatched"
    page.get_by_role("button", name="Reopen").click()
    expect(page.get_by_label("Alert detail")).to_contain_text("Awaiting action", timeout=15000)


def test_home_dispatch_moves_on_to_the_next_alert(page, site):
    open_page(page, site, "Main")
    before = api(site, "/api/summary")["open_alerts"]
    page.get_by_role("button", name="Dispatch team").click()
    page.wait_for_function(f"() => document.querySelector('nav a[href=\"Alerts.dc.html\"] span').innerText == '{before - 1}'", timeout=15000)
    assert api(site, "/api/summary")["open_alerts"] == before - 1


def test_analytics_ranges_and_real_cameras(page, site):
    text = open_page(page, site, "Analytics")
    data = api(site, "/api/analytics?range=all")
    assert f"{data['night_share_pct']:.1f}%" in text
    assert len(page.get_by_role("row").all()) == len(data["cameras"]) + 1
    for label in ("24h", "14d", "30d"):
        page.get_by_role("button", name=label, exact=True).click()
        expect(page.get_by_label("Sightings over time")).to_contain_text("the newest photo", timeout=15000)
    assert page.problems == []


def test_rangers_brief_keeps_the_design_and_answers(page, site):
    text = open_page(page, site, "Accuracy")
    assert "Ranger's Brief" in text and "AI field report, updated with every upload" in text and "Model accuracy" not in text
    b = api(site, "/api/brief")
    expect(page.get_by_label("Overall brief")).to_contain_text(b["brief"][:40], timeout=15000)
    assert page.locator('section[aria-label="Species notes"] .lift').count() == len(b["notes"])
    page.get_by_role("button", name="Explain this detection").click()
    expect(page.get_by_label("Explain a detection")).to_contain_text("confidence", timeout=15000)
    page.get_by_label("Your question").fill("How many bursts are there?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.get_by_label("Ask the data")).to_contain_text("our numbers", timeout=15000)
    assert page.problems == []


def test_accuracy_page_shows_results_failures_and_real_matrix(page, site):
    text = open_page(page, site, "Accuracy")
    a = api(site, "/api/accuracy")
    assert f"{a['headline']['accuracy']:.1f}%" in text
    assert "accuracy target was missed" in text
    assert f"Test-set accuracy on Caltech species: {a['headline']['accuracy']:.1f}%" in text
    assert "falls from 0.94 by day to 0.43 at night" in text
    assert a["frozen_at_commit"] in text
    assert "Not measured" not in text
    assert page.locator('section[aria-label="Confusion matrix"] [title*="predicted as"]').count() == len(a["confusion"]["rows"]) * len(a["confusion"]["columns"])


def test_upload_flow_real_photo_and_corrupted_file(page, site):
    known = {r[0] for r in sqlite3.connect(site["db"]).execute("SELECT file_path FROM images")}
    photos = [p for p in sorted((ROOT / "data" / "extra_val" / "images").glob("*.jpg")) if f"data/extra_val/images/{p.name}" not in known]
    photo = photos[0]
    broken = Path(os.environ.get("TEMP", ".")) / "wildwatch_broken_upload.jpg"
    broken.write_bytes(photo.read_bytes()[:1500])
    open_page(page, site, "Upload")
    assert page.locator("#cam option").count() == 21
    page.locator('input[type="file"]').set_input_files([str(photo), str(broken)])
    expect(page.get_by_text("Looking for animals").first).to_be_visible(timeout=60000)
    expect(page.get_by_label("Processing queue")).to_contain_text("2 of 2 done", timeout=200000)
    text = page.get_by_label("Processing queue").inner_text()
    assert broken.name in text
    assert page.locator('[aria-label="Processing queue"] [role="alert"]').count() == 1
    assert "truncated" in text.lower() or "cannot" in text.lower() or "error" in text.lower()
    assert "Sample result" not in text
    broken.unlink()


def test_a_new_upload_shows_up_in_alerts_and_analytics(page, site):
    """A real photo through the real pipeline, with no camera chosen, then both pages."""
    known = {r[0] for r in sqlite3.connect(site["db"]).execute("SELECT file_path FROM images")}
    photo = next(p for p in sorted((ROOT / "data" / "global_photos" / "test").glob("*tiger*.jpg")) if not any(p.name in k for k in known))
    with open(photo, "rb") as f:
        job = httpx.post(site["url"] + "/api/upload", files={"files": (photo.name, f, "image/jpeg")}, timeout=60).json()["job_id"]
    for _ in range(200):
        state = api(site, f"/api/jobs/{job}")
        if state["state"] in ("done", "failed"):
            break
        time.sleep(1)
    assert state["state"] == "done", state
    label = state["results"][0]["alert"]["species"]
    open_page(page, site, "Alerts")
    first_row = page.get_by_label("Alert list").locator("button").first
    expect(first_row).to_contain_text(label)
    expect(first_row).to_contain_text("Unknown camera")
    page.get_by_role("button", name=re.compile("^Uploads · ")).click()
    expect(page.get_by_label("Alert list")).to_contain_text(label)
    text = open_page(page, site, "Analytics")
    assert label.capitalize() in text
    assert "Uploads, no camera" in text
    assert "Cam null" not in text and "Cam None" not in text
    assert page.problems == []


def test_loading_shimmer_shows_while_data_is_on_its_way(page, site):
    page.add_init_script("""const realFetch = window.fetch;
        window.fetch = (...args) => new Promise(done => setTimeout(done, 1500)).then(() => realFetch(...args));""")
    page.goto(f"{site['url']}/Main.dc.html")
    assert page.locator(".skel").count() > 0
    page.wait_for_function("() => document.querySelectorAll('.skel').length === 0", timeout=30000)


def test_clear_error_when_the_api_is_down(page, site):
    page.route("**/api/**", lambda route: route.abort())
    page.goto(f"{site['url']}/Alerts.dc.html")
    expect(page.get_by_role("alert")).to_contain_text("Cannot reach the WildWatch server", timeout=15000)


def test_clear_error_when_the_database_is_missing(browser, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "WILDWATCH_DB": str(tmp_path / "none.db")}
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.server:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{port}/", timeout=2)
                break
            except httpx.HTTPError:
                time.sleep(0.3)
        tab = browser.new_page()
        tab.goto(f"http://127.0.0.1:{port}/Main.dc.html")
        expect(tab.get_by_role("alert")).to_contain_text("run.ps1 pipeline", timeout=15000)
        tab.close()
    finally:
        process.terminate()
        process.wait(timeout=20)
