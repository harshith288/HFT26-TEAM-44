"""WildWatch API and dashboard server. Serves src/frontend/ and reads SQLite, docs/reports/ and src/config.yaml.

Torch never loads here: uploads run src.pipeline in a separate process, one job at a time.

Run: python -m uvicorn server:app --port 8000
"""
import collections
import io
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.activity import corrected_hour
from src.classify import apply_burst_vote, burst_vote
from src.db import init_db
from src.pipeline import read_job, retry_while_busy
from src.risk import species_weight, zone_factor

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "src" / "frontend"
REPORTS = ROOT / "docs" / "reports"
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
UPLOADS = ROOT / CONFIG["paths"]["incoming"] / "uploads"
WATCHER_FILE = ROOT / "data" / "cache" / "watcher.json"
WATCHER_MAX_AGE_SECONDS = 20
JOB_TIMEOUT_SECONDS = 600
TARGET_PERCENT = 90.0
NOT_ANIMALS = ("vehicle", "person", "unknown")
LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
STATUSES = ("new", "dispatched", "false_alarm")

def db_path():
    return ROOT / (os.environ.get("WILDWATCH_DB") or CONFIG["paths"]["database"])


@asynccontextmanager
async def lifespan(_):
    if db_path().exists():
        init_db(db_path()).close()  # upgrades an older schema
    yield


app = FastAPI(title="WildWatch", lifespan=lifespan)


def query(sql, args=()):
    """Rows as dicts. A missing database is a clear 503, not a crash."""
    if not db_path().exists():
        raise HTTPException(503, "The database does not exist yet. Run src/scripts/run.ps1 pipeline first.")
    conn = sqlite3.connect(db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def report(name):
    """A report as a dict, or None if it has not been produced."""
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def dig(data, *keys):
    """Nested lookup that gives None for anything missing, which the UI shows as Not measured."""
    for key in keys:
        if data is None:
            return None
        data = data.get(key) if isinstance(data, dict) else None
    return data


def village_of(camera_id):
    for name, village in CONFIG["villages"].items():
        if str(camera_id) in village["cameras"]:
            return name
    return None


def alert_rows():
    """Every alert with its status, night flag, photo and detection boxes."""
    alerts = query("""SELECT a.*, b.start_time, b.is_night, COALESCE(s.status, 'new') AS status
                      FROM alerts a JOIN bursts b ON a.burst_id = b.burst_id AND a.species = b.species
                      LEFT JOIN alert_status s ON a.burst_id = s.burst_id AND a.species = s.species""")
    frames = collections.defaultdict(list)
    for r in query("SELECT image_id, seq_id, is_ir, width, height, source, processed_at FROM images WHERE status = 'ok' AND seq_id IS NOT NULL"):
        frames[r["seq_id"]].append(r)
    boxes = collections.defaultdict(list)
    for d in query("SELECT * FROM detections"):
        boxes[d["image_id"]].append(d)
    threshold = CONFIG["detection"]["count_threshold"]
    gate = CONFIG["classification"]["burst_vote_min_confidence"]
    out = []
    for a in alerts:
        burst = frames[a["burst_id"]]
        per_frame = [boxes[f["image_id"]] for f in burst]
        voted = apply_burst_vote(per_frame, burst_vote(per_frame, gate))
        best, best_score, best_box = None, -1.0, None
        for frame, frame_boxes in zip(burst, voted):
            for d in frame_boxes:
                label = d["species"] if d["category"] == "animal" else d["category"]
                if label == a["species"] and d["confidence"] > best_score:
                    best, best_score, best_box = (frame, frame_boxes), d["confidence"], d
        frame, frame_boxes = best if best else (burst[0] if burst else None, [])
        night = bool(a["is_night"])
        out.append({
            **{k: a[k] for k in ("alert_id", "burst_id", "species", "camera_id", "village", "min_observed_count",
                                 "danger_score", "level", "rangers", "message", "message_source", "status", "start_time")},
            "is_guess": bool(a["is_guess"]), "uploaded": bool(burst) and burst[0]["source"] == "incoming",
            "uploaded_at": burst[0]["processed_at"] if burst and burst[0]["source"] == "incoming" else None,
            "vehicle": bool(a["vehicle"]), "possible_poacher": bool(a["possible_poacher"]), "is_night": night,
            "hour": corrected_hour(a["start_time"], night),
            "label_confidence": best_box["species_confidence"] if best_box else None,
            "identified_by": best_box["classifier"] if best_box else None,
            "light": ("IR night" if frame and frame["is_ir"] else "Color night" if night else "Day"),
            "species_weight": species_weight(a["species"], CONFIG), "night_factor": CONFIG["night_factor"] if night else 1.0,
            "zone_factor": zone_factor(a["camera_id"], CONFIG), "action": CONFIG["response"][a["level"]]["action"],
            "image_id": frame["image_id"] if frame else None, "width": frame["width"] if frame else None,
            "height": frame["height"] if frame else None,
            "boxes": [{"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                       "label": d["species"] if d["category"] == "animal" else d["category"], "confidence": d["confidence"]}
                      for d in frame_boxes if d["confidence"] >= threshold],
        })
    # Newest uploads first, so a new photo is always on screen; then the most serious.
    out.sort(key=lambda a: (LEVELS.index(a["level"]), -a["danger_score"], a["alert_id"]))
    uploads = sorted((a for a in out if a["uploaded"]), key=lambda a: a["uploaded_at"], reverse=True)
    return uploads + [a for a in out if not a["uploaded"]]


def watcher_status():
    if not WATCHER_FILE.exists():
        return {"running": False, "last_seen": None}
    last_seen = json.loads(WATCHER_FILE.read_text())["last_seen"]
    return {"running": time.time() - last_seen < WATCHER_MAX_AGE_SECONDS,
            "last_seen": datetime.fromtimestamp(last_seen).isoformat(timespec="seconds")}


@app.get("/api/summary")
def summary():
    images = query("SELECT status, COUNT(*) AS n FROM images GROUP BY status")
    counts = {r["status"]: r["n"] for r in images}
    tiers = {r["triage"]: r["n"] for r in query(
        "SELECT triage, COUNT(DISTINCT seq_id) AS n FROM images WHERE status = 'ok' GROUP BY triage")}
    bursts = query("SELECT COUNT(DISTINCT seq_id) AS n FROM images WHERE status = 'ok'")[0]["n"]
    marks = ",".join("?" for _ in NOT_ANIMALS)
    species = query(f"SELECT COUNT(DISTINCT species) AS n FROM bursts WHERE species NOT IN ({marks})", NOT_ANIMALS)[0]["n"]
    open_alerts = query("""SELECT COUNT(*) AS n FROM alerts a LEFT JOIN alert_status s
                           ON a.burst_id = s.burst_id AND a.species = s.species
                           WHERE a.level != 'LOW' AND COALESCE(s.status, 'new') = 'new'""")[0]["n"]
    last = query("SELECT image_id, processed_at, camera_id FROM images WHERE status = 'ok' ORDER BY processed_at DESC LIMIT 1")
    daily = query("SELECT summary_date, summary, summary_source FROM daily_summaries ORDER BY summary_date DESC LIMIT 1")
    return {"images": counts.get("ok", 0), "unreadable": counts.get("unreadable", 0), "bursts": bursts,
            "triage": {"auto_discarded": tiers.get("discard", 0), "needs_review": tiers.get("review", 0),
                       "kept": tiers.get("keep", 0)},
            "species_count": species, "open_alerts": open_alerts, "last_image": last[0] if last else None,
            "watcher": watcher_status(), "daily_summary": daily[0] if daily else None,
            "server_time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/alerts")
def alerts(level: str | None = None, village: str | None = None):
    rows = alert_rows()
    if level:
        rows = [a for a in rows if a["level"] == level.upper()]
    if village:
        rows = [a for a in rows if a["village"] == village]
    return {"alerts": rows, "total": len(rows), "levels": {"medium_min": CONFIG["risk_levels"]["medium_min"],
                                                          "high_min": CONFIG["risk_levels"]["high_min"],
                                                          "critical_min": CONFIG["risk_levels"]["critical_min"]}}


class StatusChange(BaseModel):
    status: str


@app.post("/api/alerts/{alert_id}/status")
def set_status(alert_id: int, change: StatusChange):
    if change.status not in STATUSES:
        raise HTTPException(422, f"status must be one of {', '.join(STATUSES)}")
    found = query("SELECT burst_id, species FROM alerts WHERE alert_id = ?", (alert_id,))
    if not found:
        raise HTTPException(404, "No such alert")
    conn = sqlite3.connect(db_path(), timeout=10)
    try:
        conn.execute("INSERT OR REPLACE INTO alert_status (burst_id, species, status, updated_at) VALUES (?, ?, ?, ?)",
                     (found[0]["burst_id"], found[0]["species"], change.status,
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    finally:
        conn.close()
    return {"alert_id": alert_id, "status": change.status}


def sighting_time(row, uploaded_at):
    """When a sighting happened, or for an upload with no capture time, when it was uploaded."""
    return row["start_time"] or uploaded_at.get(row["burst_id"])


def sightings_over_time(rows, range_name, uploaded_at):
    """Windows end at the newest sighting. Uploads with no capture time count by upload date.
    The stored capture time stays empty."""
    dated = [(datetime.strptime(sighting_time(r, uploaded_at), "%Y-%m-%d %H:%M:%S"), r) for r in rows if sighting_time(r, uploaded_at)]
    if not dated:
        return {"range": range_name, "buckets": [], "window_end": None, "sightings_in_window": 0}
    end = max(d for d, _ in dated)
    if range_name == "all":
        keys = sorted({d.strftime("%Y-%m") for d, _ in dated})
        months, year, month = [], int(keys[0][:4]), int(keys[0][5:7])
        while f"{year:04d}-{month:02d}" <= keys[-1]:
            months.append(f"{year:04d}-{month:02d}")
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        keys, key_of, start = months, (lambda d: d.strftime("%Y-%m")), datetime.min
    elif range_name == "24h":
        start = end - timedelta(hours=24)
        keys = [(start + timedelta(hours=i + 1)).strftime("%m-%d %H") for i in range(24)]
        key_of = lambda d: d.strftime("%m-%d %H")  # noqa: E731
    else:
        days = 14 if range_name == "14d" else 30
        start = end - timedelta(days=days)
        keys = [(start + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(days)]
        key_of = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731
    table = {k: {"label": k, "day": 0, "night": 0} for k in keys}
    total = 0
    for d, r in dated:
        if d > start and key_of(d) in table:
            table[key_of(d)]["night" if r["is_night"] else "day"] += 1
            total += 1
    return {"range": range_name, "buckets": [table[k] for k in keys], "window_end": end.isoformat(sep=" "),
            "sightings_in_window": total}


@app.get("/api/analytics")
def analytics(range: str = Query("14d", pattern="^(24h|14d|30d|all)$")):
    marks = ",".join("?" for _ in NOT_ANIMALS)
    sightings = query(f"SELECT * FROM bursts WHERE species NOT IN ({marks})", NOT_ANIMALS)
    uploaded_at = {r["seq_id"]: r["t"] for r in query("SELECT seq_id, MAX(processed_at) AS t FROM images WHERE source = 'incoming' GROUP BY seq_id")}
    species = collections.defaultdict(lambda: {"day": 0, "night": 0})
    hours = collections.defaultdict(lambda: [0] * 24)
    for r in sightings:
        species[r["species"]]["night" if r["is_night"] else "day"] += 1
        hour = corrected_hour(r["start_time"], bool(r["is_night"]))
        if hour is not None and r["camera_id"] != "120":  # camera 120's clock is unreliable
            hours[r["species"]][hour] += 1
    split = sorted(({"species": k, **v, "total": v["day"] + v["night"]} for k, v in species.items()),
                   key=lambda s: (-s["total"], s["species"]))

    images = query("SELECT camera_id, COUNT(*) AS images, SUM(is_empty) AS empty FROM images "
                   "WHERE status = 'ok' GROUP BY camera_id")
    top = {}
    for r in query(f"SELECT camera_id, species, COUNT(*) AS n FROM bursts WHERE species NOT IN ({marks}) "
                   "GROUP BY camera_id, species ORDER BY n DESC", NOT_ANIMALS):
        top.setdefault(r["camera_id"], r["species"])
    animals = {r["camera_id"]: r["n"] for r in query(
        f"SELECT camera_id, SUM(min_observed_count) AS n FROM bursts WHERE species NOT IN ({marks}) GROUP BY camera_id", NOT_ANIMALS)}
    risk, open_by_camera = {}, collections.Counter()
    for a in sorted(alert_rows(), key=lambda x: LEVELS.index(x["level"])):
        if a["camera_id"] not in risk:
            risk[a["camera_id"]] = a["level"]
        open_by_camera[a["camera_id"]] += a["status"] == "new" and a["level"] != "LOW"
    cameras = [{"camera_id": r["camera_id"], "village": village_of(r["camera_id"]), "images": r["images"],
                "animals": animals.get(r["camera_id"], 0), "empty_pct": 100 * (r["empty"] or 0) / r["images"],
                "top_species": top.get(r["camera_id"]), "top_risk": risk.get(r["camera_id"]),
                "open_alerts": open_by_camera[r["camera_id"]]} for r in images]
    cameras.sort(key=lambda c: -c["images"])
    night = sum(s["night"] for s in split)
    total = sum(s["total"] for s in split)
    return {"sightings": sightings_over_time(sightings, range, uploaded_at), "species": split,
            "hourly": [{"species": s["species"], "counts": hours[s["species"]]} for s in split],
            "hourly_note": "Left out: camera 120, whose clock is unreliable, and photos with no capture time.",
            "cameras": cameras, "night_share_pct": 100 * night / total if total else None,
            "note": "The sample was balanced by day and night, so these describe the sample, not natural behaviour."}


@app.get("/api/accuracy")
def accuracy():
    final = report("final_eval.json")
    species = report("phase2_species.json")
    day_night = report("phase3_day_night.json")
    discard = report("discard_tuning.json")
    improvement = report("improvement.json")
    confusion = report("confusion_matrix_test.json")
    groups = dig(final, "groups")
    speed, classifier_speed = report("detection_speed.json"), report("speciesnet_speed.json")
    accuracy_value = dig(groups, "all", "species", "accuracy")
    precision_value = dig(groups, "all", "species", "macro_precision")

    models = []
    kinds = {"speciesnet_top_mapped": ("SpeciesNet top guess", "Pretrained, global"),
             "speciesnet_processed": ("SpeciesNet processed", "Pretrained, global"),
             "probe_dinov2": ("DINOv2 linear probe", "Trained on CCT"), "probe_bioclip": ("BioCLIP linear probe", "Trained on CCT"),
             "bioclip_zero_shot": ("BioCLIP", "Zero-shot")}
    for key, scores in sorted((dig(species, "candidates") or {}).items(), key=lambda kv: -kv[1]["all"]["macro_f1"]):
        name, kind = kinds.get(key, (key, ""))
        probe = dig(species, "overfitting", key.replace("probe_", ""))
        models.append({"key": key, "name": name, "kind": kind, "macro_f1": scores["all"]["macro_f1"],
                       "accuracy": scores["all"]["top1_accuracy"], "winner": key == dig(species, "best_by_val_macro_f1"),
                       "overfit": bool(probe and probe["overfit_flag"]),
                       "train_val_gap": probe["accuracy_gap_points"] if probe else None})

    def night_pair(section, key, kind):
        if kind == "val":
            pair = [dig(day_night, "groups", g, section, key) for g in ("day", "ir_night")]
        else:
            pair = [dig(groups, g, section, key) for g in ("day", "ir_night")]
        return pair

    splits = query("SELECT split, COUNT(*) AS n FROM images WHERE split IS NOT NULL GROUP BY split")
    order = ["train", "cis_val", "trans_val", "cis_test", "trans_test"]
    splits.sort(key=lambda s: order.index(s["split"]))
    totals = query("SELECT COUNT(*) AS images, COUNT(DISTINCT seq_id) AS bursts, SUM(gt_n_boxes > 0) AS with_boxes "
                   "FROM images WHERE split IS NOT NULL")[0]

    return {
        "evaluated": final is not None, "frozen_at_commit": dig(final, "frozen_at_commit"),
        "test_images": dig(final, "n_real_images"), "scored_images": dig(final, "n_scored"),
        "hardware": dig(final, "hardware"),
        "target": {"percent": TARGET_PERCENT, "accuracy": accuracy_value, "macro_precision": precision_value,
                   "accuracy_reached": None if accuracy_value is None else accuracy_value >= TARGET_PERCENT,
                   "precision_reached": None if precision_value is None else precision_value >= TARGET_PERCENT},
        "headline": {"accuracy": accuracy_value, "accuracy_95_range": dig(groups, "all", "species", "accuracy_95_range"),
                     "macro_precision": precision_value, "macro_f1": dig(groups, "all", "species", "macro_f1"),
                     "accuracy_confirmed_by_box": dig(groups, "all", "species_confirmed_by_box", "accuracy"),
                     "missed_images": dig(final, "missed_images"),
                     "missed_with_ground_truth_box": dig(final, "missed_with_a_ground_truth_box")},
        "triage_test": dig(groups, "all", "triage"), "discard_fresh_val": dig(discard, "fresh_val_after"),
        "day_vs_ir_night": [
            {"metric": "Empty filter F1 (val, frames)", "day": night_pair("filter", "f1_empty", "val")[0],
             "night": night_pair("filter", "f1_empty", "val")[1]},
            {"metric": "Empty precision (val, frames)", "day": night_pair("filter", "precision_empty", "val")[0],
             "night": night_pair("filter", "precision_empty", "val")[1]},
            {"metric": "Species accuracy % (test)", "day": night_pair("species", "accuracy", "test")[0],
             "night": night_pair("species", "accuracy", "test")[1]},
            {"metric": "Species macro-F1 % (test)", "day": night_pair("species", "macro_f1", "test")[0],
             "night": night_pair("species", "macro_f1", "test")[1]},
            {"metric": "Count error, MAE (test)", "day": night_pair("count", "mae", "test")[0],
             "night": night_pair("count", "mae", "test")[1]}],
        "models": models, "confusion": None if confusion is None else {k: confusion[k] for k in ("rows", "columns", "matrix")},
        "generalization": {"seen_macro_f1": dig(groups, "cis_test", "species", "macro_f1"),
                           "new_macro_f1": dig(groups, "trans_test", "species", "macro_f1"),
                           "seen_accuracy": dig(groups, "cis_test", "species", "accuracy"),
                           "new_accuracy": dig(groups, "trans_test", "species", "accuracy"),
                           "gap_points": dig(final, "cis_minus_trans_accuracy_points")},
        "speed": {"detection_ms": dig(speed, "ms_per_image_including_load"),
                  "classification_ms": dig(classifier_speed, "ms_per_crop_classifier_only"), "hardware": dig(speed, "hardware")},
        "fresh_val": {"accuracy": dig(improvement, "fresh_val", "all", "accuracy"),
                      "macro_precision": dig(improvement, "fresh_val", "all", "macro_precision")},
        "integrity": {"splits": splits, **totals},
    }


@app.get("/api/rare")
def rare():
    rows = []
    for item in CONFIG["rare_species"]:
        marks = ",".join("?" for _ in item["sightings_species"])
        seen = query(f"SELECT COUNT(*) AS n FROM bursts WHERE species IN ({marks})", item["sightings_species"])[0]["n"]
        rows.append({k: item[k] for k in ("name", "scientific", "count", "status", "trend", "trend_up", "source")}
                    | {"camera_sightings": seen})
    return {"species": rows, "note": "This data set covers US wildlife, so these Indian species are not expected in it."}


@app.get("/api/ticker")
def ticker():
    rows = query("""SELECT i.image_id, i.seq_id, i.camera_id, i.captured_at, i.processed_at, i.is_night FROM images i
                    WHERE i.status = 'ok' AND EXISTS (SELECT 1 FROM detections d WHERE d.image_id = i.image_id
                                                      AND d.confidence >= ?)
                    ORDER BY i.processed_at DESC, i.rowid DESC LIMIT 10""", (CONFIG["detection"]["count_threshold"],))
    out = []
    threshold = CONFIG["detection"]["count_threshold"]
    for r in rows:
        boxes = query("SELECT category, species, confidence FROM detections WHERE image_id = ? AND confidence >= ?",
                      (r["image_id"], threshold))
        labels = collections.Counter(b["species"] if b["category"] == "animal" else b["category"] for b in boxes)
        species, count = labels.most_common(1)[0]
        level = query("SELECT level FROM alerts WHERE burst_id = ? ORDER BY danger_score DESC LIMIT 1", (r["seq_id"],))
        out.append({"image_id": r["image_id"], "camera_id": r["camera_id"], "species": species, "count": count,
                    "level": level[0]["level"] if level else None, "hour": corrected_hour(r["captured_at"], bool(r["is_night"])),
                    "captured_at": r["captured_at"]})
    return {"detections": out}


@app.get("/api/review")
def review():
    rows = query("""SELECT image_id, seq_id, camera_id, captured_at, top_confidence FROM images
                    WHERE triage = 'review' AND status = 'ok' ORDER BY top_confidence DESC""")
    seen, out = set(), []
    for r in rows:
        if r["seq_id"] not in seen:
            seen.add(r["seq_id"])
            out.append({**r, "village": village_of(r["camera_id"])})
    return {"bursts": out, "total": len(out)}


@lru_cache(maxsize=256)
def thumbnail(path, modified):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((720, 720))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=82)
    return buffer.getvalue()


@app.get("/api/images/{image_id}")
def image_file(image_id: str):
    rows = query("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
    if not rows:
        raise HTTPException(404, "No such image")
    path = Path(rows[0]["file_path"])
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        raise HTTPException(404, "The image file is missing")
    try:
        return Response(thumbnail(str(path), path.stat().st_mtime), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})
    except OSError:
        raise HTTPException(422, "The image file cannot be read") from None


# --- uploads: one background job at a time, because the GPU has only 6 GB ---

job_queue = queue.Queue()


def write_job(folder, state, stage, error=None):
    temporary = folder / "job.tmp"
    temporary.write_text(json.dumps({"state": state, "stage": stage, "results": [], "error": error}))
    retry_while_busy(lambda: temporary.replace(folder / "job.json"))


def run_jobs():
    while True:
        folder, command = job_queue.get()
        try:
            with open(folder / "job.log", "w") as log:
                done = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=JOB_TIMEOUT_SECONDS)
            state = read_job(folder)["state"]
            if done.returncode != 0 or state not in ("done", "failed"):
                write_job(folder, "failed", "error", f"The pipeline process stopped with exit code {done.returncode}.")
        except subprocess.TimeoutExpired:
            write_job(folder, "failed", "error", f"The pipeline took longer than {JOB_TIMEOUT_SECONDS} seconds.")
        except OSError as e:
            write_job(folder, "failed", "error", f"{type(e).__name__}: {e}")
        finally:
            job_queue.task_done()


threading.Thread(target=run_jobs, daemon=True).start()


def clean_time(value):
    """Turn the form's YYYY-MM-DDTHH:MM into the database format, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise HTTPException(422, "captured_at is not a valid date and time") from None


def start_job(folder, camera_id, captured_at, prefer_exif):
    write_job(folder, "queued", "upload")
    command = [sys.executable, "-m", "src.pipeline", "--job-dir", str(folder)]
    if camera_id:
        command += ["--camera", camera_id]
    if captured_at:
        command += ["--captured-at", captured_at]
    if not prefer_exif:
        command.append("--ignore-exif")
    job_queue.put((folder, command))
    return {"job_id": folder.name}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), camera_id: str | None = Form(None),
                 captured_at: str | None = Form(None), prefer_exif: bool = Form(True)):
    if camera_id and village_of(camera_id) is None:
        raise HTTPException(422, f"Unknown camera {camera_id}")
    when = clean_time(captured_at)
    folder = UPLOADS / uuid.uuid4().hex
    folder.mkdir(parents=True)
    for index, file in enumerate(files):
        name = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "").name) or f"upload_{index}"
        if name.startswith("job."):
            name = f"upload_{name}"
        (folder / name).write_bytes(await file.read())
    return start_job(folder, camera_id, when, prefer_exif)


@app.post("/api/upload/demo")
def upload_demo():
    """Run the pipeline on a real fresh val photo that is not in the database yet."""
    source = ROOT / "data" / "extra_val" / "images"
    known = {Path(r["file_path"]).name for r in query("SELECT file_path FROM images")}
    picks = sorted(p for p in source.glob("*.jpg") if p.name not in known) if source.exists() else []
    if not picks:
        raise HTTPException(404, "No unused real photo is available in data/extra_val/images.")
    folder = UPLOADS / uuid.uuid4().hex
    folder.mkdir(parents=True)
    (folder / picks[0].name).write_bytes(picks[0].read_bytes())
    return start_job(folder, None, None, True)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "No such job")
    folder = UPLOADS / job_id
    path = folder / "job.json"
    if not path.exists():
        raise HTTPException(404, "No such job")
    data = read_job(folder)
    files = sorted(p.name for p in folder.iterdir() if not p.name.startswith("job."))
    results = data["results"]
    for r in results:
        r["stage"] = "error" if r["status"] == "unreadable" else "done"
    return {"job_id": job_id, "state": data["state"], "stage": "risk" if data.get("stage") == "brief" else data.get("stage"), "error": data["error"],
            "files": files, "results": results}


class BriefQuestion(BaseModel):
    question: str


class BriefPick(BaseModel):
    detection_id: int


def brief_frames():
    """Every table, through the one reader. pandas loads here and not at import, because the
    tests import this file in a process that holds torch, and the two crash together."""
    from src.db import load_data
    data = load_data({})
    if data is None:
        raise HTTPException(503, "The database does not exist yet. Run src/scripts/run.ps1 pipeline first.")
    return data


@lru_cache(maxsize=1)
def brief_facts_for(version):
    from src.analytics import brief_facts
    return brief_facts(brief_frames())


def llm_answer(ask):
    """Run one LLM question with the cache connection and client, closing them afterwards."""
    from src.brief import load_cfg
    from src.llm import make_client
    cfg = load_cfg()
    conn = init_db(db_path())
    try:
        text, source = ask(cfg, conn, make_client(cfg))
        conn.commit()
    finally:
        conn.close()
    return {"text": text, "source": source}


@app.get("/api/brief")
def brief():
    from src.brief import template_brief, template_note, updated_label
    from src.llm import species_name
    data = brief_frames()
    facts = brief_facts_for(data["data_version"])
    saved = data["briefs"]
    latest = saved.iloc[0] if len(saved) else None
    notes = json.loads(latest["species_notes"]) if latest is not None else {}
    boxes = data["detections"]
    boxes = boxes[boxes["category"] != "vehicle"].sort_values("confidence", ascending=False).head(100)
    return {"brief": latest["brief"] if latest is not None else template_brief(facts),
            "source": latest["source"] if latest is not None else "template",
            "updated": updated_label(latest["created_at"]) if latest is not None else "Written after the next upload",
            "data_version": int(data["data_version"]),
            "notes": [{"species": species_name(r["species"]), **notes.get(r["species"], {"note": template_note(r), "source": "template"})}
                      for r in facts["species"]],
            "detections": [{"detection_id": int(r["detection_id"]), "image_id": r["image_id"],
                            "label": f"{r['species'] or r['category']} · {r['confidence']:.2f} · {camera_label(r['camera_id'])}"}
                           for r in boxes.to_dict("records")]}


def camera_label(camera_id):
    return f"camera {camera_id}" if isinstance(camera_id, str) else "unknown camera"


@app.post("/api/brief/explain")
def brief_explain(pick: BriefPick):
    from src.brief import detection_facts, explain_detection
    data = brief_frames()
    found = data["detections"][data["detections"]["detection_id"] == pick.detection_id]
    if found.empty:
        raise HTTPException(404, "No such detection")
    box = found.iloc[0]
    image = data["images"][data["images"]["image_id"] == box["image_id"]].iloc[0]
    details = detection_facts(box.where(box.notna(), None).to_dict(), image.where(image.notna(), None).to_dict())
    return llm_answer(lambda cfg, conn, client: explain_detection(details, cfg, conn, client))


@app.post("/api/brief/ask")
def brief_ask(body: BriefQuestion):
    from src.brief import ask_data
    question = body.question.strip()
    if not question or len(question) > 300:
        raise HTTPException(422, "Ask a question of up to 300 characters.")
    facts = brief_facts_for(brief_frames()["data_version"])
    return llm_answer(lambda cfg, conn, client: ask_data(question, facts, cfg, conn, client))


@app.get("/api/cameras")
def cameras():
    return {"cameras": [{"camera_id": c, "village": name} for name, v in CONFIG["villages"].items() for c in v["cameras"]]}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND / "Main.dc.html")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
