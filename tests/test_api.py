"""API tests against direct SQL, on a temporary copy of the real database.

Uploads run the real pipeline in a background process on real images. LLM calls are
turned off here, so a test run never spends credit.
"""
import csv
import io
import json
import os
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
from fastapi.testclient import TestClient
from PIL import Image

from src import server
from src.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
DATABASE = ROOT / CONFIG["paths"]["database"]
FINAL = json.loads((ROOT / "docs" / "reports" / "final_eval.json").read_text())


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("api") / "copy.db"
    shutil.copy(DATABASE, path)
    previous = {k: os.environ.get(k) for k in ("WILDWATCH_DB", "FEATHERLESS_API_KEY")}
    os.environ["WILDWATCH_DB"] = str(path)
    os.environ["FEATHERLESS_API_KEY"] = ""  # no credit is spent by tests
    yield path
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="module")
def client(db):
    with TestClient(server.app) as test_client:
        yield test_client


def sql(db, text, *args):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(text, args).fetchall()
    finally:
        conn.close()


# --- summary ---

def test_summary_matches_sql(client, db):
    s = client.get("/api/summary").json()
    assert s["images"] == sql(db, "SELECT COUNT(*) FROM images WHERE status = 'ok'")[0][0]
    assert s["bursts"] == sql(db, "SELECT COUNT(DISTINCT seq_id) FROM images WHERE status = 'ok'")[0][0]
    tiers = dict(sql(db, "SELECT triage, COUNT(DISTINCT seq_id) FROM images WHERE status = 'ok' GROUP BY triage"))
    assert s["triage"] == {"auto_discarded": tiers["discard"], "needs_review": tiers["review"], "kept": tiers["keep"]}
    assert sum(s["triage"].values()) == s["bursts"]
    assert s["species_count"] == sql(db, "SELECT COUNT(DISTINCT species) FROM bursts WHERE species NOT IN ('vehicle', 'person', 'unknown')")[0][0]
    assert s["last_image"]["processed_at"] == sql(db, "SELECT MAX(processed_at) FROM images WHERE status = 'ok'")[0][0]
    assert s["watcher"]["running"] in (True, False)


def test_open_alerts_counts_new_alerts_above_low(client, db):
    expected = sql(db, """SELECT COUNT(*) FROM alerts a LEFT JOIN alert_status s ON a.burst_id = s.burst_id
                          AND a.species = s.species WHERE a.level != 'LOW' AND COALESCE(s.status, 'new') = 'new'""")[0][0]
    assert client.get("/api/summary").json()["open_alerts"] == expected


def test_watcher_status_reads_a_real_heartbeat(client, tmp_path, monkeypatch):
    path = tmp_path / "watcher.json"
    monkeypatch.setattr(server, "WATCHER_FILE", path)
    assert client.get("/api/summary").json()["watcher"]["running"] is False
    path.write_text(json.dumps({"last_seen": time.time()}))
    assert client.get("/api/summary").json()["watcher"]["running"] is True
    path.write_text(json.dumps({"last_seen": time.time() - 3600}))
    assert client.get("/api/summary").json()["watcher"]["running"] is False


# --- alerts ---

def test_alerts_match_sql(client, db):
    alerts = client.get("/api/alerts").json()["alerts"]
    rows = {r[0]: r for r in sql(db, "SELECT alert_id, species, level, village, danger_score, message, message_source FROM alerts")}
    assert len(alerts) == len(rows)
    for a in alerts:
        row = rows[a["alert_id"]]
        assert (a["species"], a["level"], a["village"], a["danger_score"], a["message"], a["message_source"]) == row[1:]


@pytest.mark.parametrize("level", ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
def test_level_filter_matches_sql(client, db, level):
    got = client.get("/api/alerts", params={"level": level}).json()["alerts"]
    assert len(got) == sql(db, "SELECT COUNT(*) FROM alerts WHERE level = ?", level)[0][0]
    assert {a["level"] for a in got} == {level}


def test_village_filter_matches_sql(client, db):
    got = client.get("/api/alerts", params={"village": "Larkmere"}).json()["alerts"]
    assert len(got) == sql(db, "SELECT COUNT(*) FROM alerts WHERE village = 'Larkmere'")[0][0]


def test_score_parts_multiply_to_the_stored_score(client):
    for a in client.get("/api/alerts").json()["alerts"]:
        product = a["species_weight"] * a["min_observed_count"] * a["night_factor"] * a["zone_factor"]
        assert product == pytest.approx(a["danger_score"], abs=0.0001)


def test_alert_scale_comes_from_config(client):
    assert client.get("/api/alerts").json()["levels"] == CONFIG["risk_levels"]


def test_alert_photo_and_boxes_are_real(client):
    alert = next(a for a in client.get("/api/alerts").json()["alerts"] if a["boxes"])
    response = client.get(f"/api/images/{alert['image_id']}")
    assert response.headers["content-type"] == "image/jpeg"
    image = Image.open(io.BytesIO(response.content))
    assert max(image.size) <= 720
    assert all(0 <= b["x"] <= 1 and 0 <= b["y"] <= 1 for b in alert["boxes"])


def test_status_changes_are_stored_and_validated(client, db):
    alert = client.get("/api/alerts", params={"level": "HIGH"}).json()["alerts"][0]
    url = f"/api/alerts/{alert['alert_id']}/status"
    for status in ("dispatched", "false_alarm", "new"):
        assert client.post(url, json={"status": status}).status_code == 200
        assert sql(db, "SELECT status FROM alert_status WHERE burst_id = ? AND species = ?", alert["burst_id"], alert["species"])[0][0] == status
    assert client.post(url, json={"status": "closed"}).status_code == 422
    assert client.post("/api/alerts/99999999/status", json={"status": "new"}).status_code == 404


def test_status_survives_a_burst_rebuild_that_replaces_the_alert(client, db):
    alert = client.get("/api/alerts", params={"level": "HIGH"}).json()["alerts"][1]
    assert client.post(f"/api/alerts/{alert['alert_id']}/status", json={"status": "dispatched"}).status_code == 200
    pipeline = Pipeline(CONFIG, db_path=db, detector=object())
    pipeline.rebuild_burst(alert["burst_id"])
    pipeline.conn.commit()
    pipeline.conn.close()
    after = next(a for a in client.get("/api/alerts").json()["alerts"]
                 if a["burst_id"] == alert["burst_id"] and a["species"] == alert["species"])
    assert after["alert_id"] != alert["alert_id"]
    assert after["status"] == "dispatched"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(db, port):
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.server:app", "--port", str(port), "--log-level", "warning"],
                               cwd=ROOT, env={**os.environ, "WILDWATCH_DB": str(db)})
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/summary", timeout=2).status_code == 200:
                return process
        except httpx.HTTPError:
            time.sleep(0.3)
    process.kill()
    raise AssertionError("server did not start")


def test_status_persists_after_a_real_server_restart(db):
    port = free_port()
    first = start_server(db, port)
    base = f"http://127.0.0.1:{port}"
    try:
        alert = httpx.get(f"{base}/api/alerts", params={"level": "MEDIUM"}, timeout=30).json()["alerts"][0]
        assert httpx.post(f"{base}/api/alerts/{alert['alert_id']}/status", json={"status": "false_alarm"}, timeout=30).status_code == 200
    finally:
        first.terminate()
        first.wait(timeout=20)
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{base}/api/summary", timeout=2)
    second = start_server(db, port)
    try:
        again = httpx.get(f"{base}/api/alerts", params={"level": "MEDIUM"}, timeout=30).json()["alerts"]
        assert next(a for a in again if a["alert_id"] == alert["alert_id"])["status"] == "false_alarm"
    finally:
        second.terminate()
        second.wait(timeout=20)


def test_the_server_process_never_loads_torch():
    code = ("import sys\nfrom src import server\nfrom fastapi.testclient import TestClient\n"
            "c = TestClient(server.app)\nc.get('/api/summary'); c.get('/api/alerts'); c.get('/api/accuracy')\n"
            "assert not {'torch', 'ultralytics', 'speciesnet'} & set(sys.modules), 'heavy module loaded'")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]


# --- analytics ---

@pytest.mark.parametrize("range_name,buckets", [("24h", 24), ("14d", 14), ("30d", 30)])
def test_windows_have_the_right_size_and_end_at_the_newest_photo(client, db, range_name, buckets):
    s = client.get("/api/analytics", params={"range": range_name}).json()["sightings"]
    assert len(s["buckets"]) == buckets
    assert s["sightings_in_window"] == sum(b["day"] + b["night"] for b in s["buckets"])
    newest = sql(db, """SELECT MAX(COALESCE(b.start_time, i.processed_at)) FROM bursts b JOIN images i ON i.seq_id = b.burst_id
                        WHERE b.species NOT IN ('vehicle', 'person', 'unknown')""")[0][0]
    assert s["window_end"] == newest


def test_all_range_counts_every_dated_sighting(client, db):
    s = client.get("/api/analytics", params={"range": "all"}).json()["sightings"]
    expected = sql(db, "SELECT COUNT(*) FROM bursts WHERE species NOT IN ('vehicle', 'person', 'unknown')")[0][0]
    assert sum(b["day"] + b["night"] for b in s["buckets"]) == expected, "every sighting, including undated uploads"


def add_upload(db, seq, species):
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO images (image_id, file_path, source, seq_id, status, processed_at) VALUES (?, 'x.jpg', 'incoming', ?, 'ok', '2030-01-02 03:04:05')", (seq, seq))
    conn.execute("INSERT INTO bursts (burst_id, species, n_frames, min_observed_count, is_night) VALUES (?, ?, 1, 1, 0)", (seq, species))
    conn.execute("INSERT INTO alerts (burst_id, species, min_observed_count, danger_score, level, rangers, vehicle, possible_poacher, created_at) "
                 "VALUES (?, ?, 1, 0.1, 'LOW', 0, 0, 0, '2030-01-02 03:04:05')", (seq, species))
    conn.commit()
    conn.close()


def remove_upload(db, seq):
    conn = sqlite3.connect(db)
    for table, column in (("alerts", "burst_id"), ("bursts", "burst_id"), ("images", "seq_id")):
        conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (seq,))
    conn.commit()
    conn.close()


def test_a_new_low_upload_comes_first_and_counts_in_analytics(client, db):
    add_upload(db, "new-upload", "lion")
    try:
        first = client.get("/api/alerts").json()["alerts"][0]
        assert (first["burst_id"], first["level"], first["uploaded"]) == ("new-upload", "LOW", True)
        data = client.get("/api/analytics", params={"range": "24h"}).json()
        assert data["sightings"]["sightings_in_window"] >= 1
        assert data["sightings"]["window_end"] == "2030-01-02 03:04:05"
        assert "lion" in [s["species"] for s in data["species"]]
        assert None in [c["camera_id"] for c in data["cameras"]]
    finally:
        remove_upload(db, "new-upload")


def test_camera_top_risk_is_still_the_most_serious_alert(client, db):
    cameras = {c["camera_id"]: c for c in client.get("/api/analytics").json()["cameras"]}
    assert cameras["88"]["top_risk"] == sql(db, """SELECT level FROM alerts WHERE camera_id = '88' ORDER BY
        CASE level WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END LIMIT 1""")[0][0]


def test_bad_range_is_rejected(client):
    assert client.get("/api/analytics", params={"range": "7d"}).status_code == 422


def test_species_split_matches_sql(client, db):
    data = client.get("/api/analytics").json()
    for s in data["species"]:
        rows = dict(sql(db, "SELECT is_night, COUNT(*) FROM bursts WHERE species = ? GROUP BY is_night", s["species"]))
        assert (s["day"], s["night"]) == (rows.get(0, 0), rows.get(1, 0))


def test_hourly_activity_leaves_out_camera_120(client, db):
    data = client.get("/api/analytics").json()
    total = sum(sum(h["counts"]) for h in data["hourly"])
    expected = sql(db, """SELECT COUNT(*) FROM bursts WHERE species NOT IN ('vehicle', 'person', 'unknown')
                          AND start_time IS NOT NULL AND camera_id != '120'""")[0][0]
    assert total == expected
    assert all(len(h["counts"]) == 24 for h in data["hourly"])


def test_camera_stats_match_sql(client, db):
    cameras = client.get("/api/analytics").json()["cameras"]
    rows = dict(sql(db, "SELECT camera_id, COUNT(*) FROM images WHERE status = 'ok' GROUP BY camera_id"))
    assert {c["camera_id"]: c["images"] for c in cameras} == rows


# --- accuracy ---

def test_accuracy_reports_the_frozen_results_and_the_missed_target(client):
    a = client.get("/api/accuracy").json()
    groups = FINAL["groups"]
    assert a["headline"]["accuracy"] == groups["all"]["species"]["accuracy"]
    assert a["headline"]["macro_precision"] == groups["all"]["species"]["macro_precision"]
    assert a["target"]["accuracy_reached"] is False
    assert a["target"]["precision_reached"] is True
    assert a["frozen_at_commit"] == FINAL["frozen_at_commit"]


def test_day_and_night_failures_are_included(client):
    rows = {r["metric"]: r for r in client.get("/api/accuracy").json()["day_vs_ir_night"]}
    assert rows["Empty precision (val, frames)"]["night"] < rows["Empty precision (val, frames)"]["day"]
    assert all(r["day"] is not None and r["night"] is not None for r in rows.values())


def test_confusion_matrix_gives_the_frozen_accuracy(client):
    c = client.get("/api/accuracy").json()["confusion"]
    right = sum(row[c["columns"].index(name)] for name, row in zip(c["rows"], c["matrix"]))
    total = sum(sum(row) for row in c["matrix"])
    assert 100 * right / total == pytest.approx(FINAL["groups"]["all"]["species"]["accuracy"])
    assert total == FINAL["n_scored"]


def test_missing_reports_are_null_not_invented(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPORTS", tmp_path)
    a = client.get("/api/accuracy").json()
    assert a["evaluated"] is False
    assert a["headline"]["accuracy"] is None and a["target"]["accuracy_reached"] is None
    assert a["confusion"] is None and a["models"] == []
    assert a["speed"]["detection_ms"] is None
    assert all(r["day"] is None for r in a["day_vs_ir_night"])


def test_missing_database_is_a_clear_503(client, tmp_path, monkeypatch):
    monkeypatch.setenv("WILDWATCH_DB", str(tmp_path / "none.db"))
    response = client.get("/api/summary")
    assert response.status_code == 503
    assert "run.ps1 pipeline" in response.json()["detail"]


# --- rare, ticker, review, images ---

def test_rare_figures_come_from_config_and_sightings_are_real(client, db):
    rows = {r["name"]: r for r in client.get("/api/rare").json()["species"]}
    assert rows["Bengal tiger"]["count"] == 3682
    assert rows["Asian elephant"]["count"] == 22446
    assert rows["Indian leopard"]["count"] == 13874
    assert "All India Tiger Estimation, 2022" == rows["Bengal tiger"]["source"]
    for name, item in ((i["name"], i) for i in CONFIG["rare_species"]):
        marks = ",".join("?" for _ in item["sightings_species"])
        expected = sql(db, f"SELECT COUNT(*) FROM bursts WHERE species IN ({marks})", *item["sightings_species"])[0][0]
        assert rows[name]["camera_sightings"] == expected, name
    assert "asian elephant" in next(i for i in CONFIG["rare_species"] if i["name"] == "Asian elephant")["sightings_species"]


def test_ticker_gives_the_latest_ten_real_detections(client, db):
    rows = client.get("/api/ticker").json()["detections"]
    assert len(rows) == 10
    for r in rows:
        assert sql(db, "SELECT COUNT(*) FROM detections WHERE image_id = ?", r["image_id"])[0][0] > 0


def test_review_queue_matches_sql(client, db):
    data = client.get("/api/review").json()
    assert data["total"] == sql(db, "SELECT COUNT(DISTINCT seq_id) FROM images WHERE triage = 'review'")[0][0]
    scores = [b["top_confidence"] for b in data["bursts"]]
    assert scores == sorted(scores, reverse=True)


def test_unknown_image_and_job_are_404(client):
    assert client.get("/api/images/nothing").status_code == 404
    assert client.get("/api/jobs/nothing").status_code == 404
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404


def test_dashboard_files_are_served(client):
    assert "WildWatch home" in client.get("/").text
    for page in ("Main", "Upload", "Alerts", "Analytics", "Accuracy"):
        assert client.get(f"/{page}.dc.html").status_code == 200


# --- uploads: the real background pipeline on real files ---

def wait(client, job_id, seen=None, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if seen is not None:
            seen.add(job["stage"])
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job did not finish: {job}")


def fresh_photo():
    labeled = sql(ROOT / "data" / "extra_val" / "extra.db", """SELECT i.file_path FROM images i JOIN detections d ON d.image_id = i.image_id
                  WHERE d.category = 'animal' AND d.confidence > 0.9 ORDER BY i.image_id LIMIT 1""")[0][0]
    return ROOT / labeled


def test_uploading_a_fresh_real_photo_gives_a_real_result(client, db):
    photo = fresh_photo()
    response = client.post("/api/upload", files=[("files", (photo.name, photo.read_bytes(), "image/jpeg"))],
                           data={"camera_id": "33", "captured_at": "2026-01-02T03:04", "prefer_exif": "true"})
    assert response.status_code == 200
    seen = set()
    job = wait(client, response.json()["job_id"], seen)
    assert job["state"] == "done" and job["stage"] == "done"
    assert {"detect", "done"} <= seen
    assert seen <= {"upload", "detect", "species", "count", "risk", "brief", "done"}
    result = job["results"][0]
    assert result["stage"] == "done" and result["status"] == "ok"
    assert result["camera_id"] == "33" and result["captured_at"] == "2026-01-02 03:04:00"
    stored = sql(db, "SELECT category, species FROM detections WHERE image_id = ? ORDER BY confidence DESC", result["image_id"])
    assert stored and stored[0][0] == "animal" and stored[0][1]
    assert result["alert"]["village"] == "Ambernook"
    assert result["alert"]["species"] == stored[0][1]
    assert client.post(f"/api/alerts/{result['alert']['alert_id']}/status", json={"status": "dispatched"}).status_code == 200
    assert client.get(f"/api/images/{result['image_id']}").status_code == 200


@pytest.mark.parametrize("name,data", [("corrupted.jpg", None), ("empty.jpg", b""), ("notes.txt", b"this is not an image")])
def test_bad_uploads_end_in_a_clean_error(client, db, name, data):
    if data is None:
        data = fresh_photo().read_bytes()[:2000]
    response = client.post("/api/upload", files=[("files", (name, data, "application/octet-stream"))])
    assert response.status_code == 200
    job = wait(client, response.json()["job_id"])
    assert job["state"] == "done", job
    result = job["results"][0]
    assert result["stage"] == "error" and result["status"] == "unreadable"
    assert result["error"]
    assert "Traceback" not in result["error"]
    assert result["alert"] is None
    assert sql(db, "SELECT status FROM images WHERE image_id = ?", result["image_id"])[0][0] == "unreadable"


def test_upload_rejects_a_bad_camera_or_time(client):
    files = [("files", ("a.jpg", b"x", "image/jpeg"))]
    assert client.post("/api/upload", files=files, data={"camera_id": "99999"}).status_code == 422
    assert client.post("/api/upload", files=files, data={"captured_at": "yesterday"}).status_code == 422


def test_uploaded_names_cannot_leave_the_upload_folder(client):
    response = client.post("/api/upload", files=[("files", ("../../evil.jpg", b"not an image", "image/jpeg"))])
    job = wait(client, response.json()["job_id"])
    assert job["files"] == ["evil.jpg"]
    assert not (ROOT / "data" / "evil.jpg").exists()


def test_demo_upload_uses_a_real_unseen_photo(client):
    known = {r[0] for r in sql(os.environ["WILDWATCH_DB"], "SELECT file_path FROM images")}
    response = client.post("/api/upload/demo")
    assert response.status_code == 200
    job = wait(client, response.json()["job_id"])
    assert job["state"] == "done"
    assert job["results"][0]["file_path"] not in known


def test_no_sample_data_is_left_in_the_pages():
    for page in ("Main", "Upload", "Alerts", "Analytics", "Accuracy"):
        text = (ROOT / "src" / "frontend" / f"{page}.dc.html").read_text(encoding="utf-8")
        for stale in ("Sample data", "Sample alert", "Sample result", "Rampur", "Kondapur", "[—", "[N]", "1,506", "Phase 1 tests"):
            assert stale not in text, f"{page}: {stale}"


def test_manifest_images_are_still_all_in_the_database(db):
    ids = {r["image_id"] for r in csv.DictReader(open(ROOT / CONFIG["paths"]["manifest"]))}
    assert ids <= {r[0] for r in sql(db, "SELECT image_id FROM images")}
