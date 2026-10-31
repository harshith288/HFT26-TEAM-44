"""Full pipeline: image -> detections -> species -> counts -> risk -> SQLite.

Usage:
  python -m src.pipeline --sample              train and val images from the manifest
  python -m src.pipeline --sample --include-test   all splits (Phase 5 only)
  python -m src.pipeline --once                everything new in data/incoming
  python -m src.pipeline --watch               keep checking data/incoming
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from PIL import ExifTags

from src.activity import is_dark_scene, is_ir_image, is_night
from src.classify import apply_burst_vote, burst_vote, classify_boxes, classify_global
from src.count import burst_counts, detection_label
from src.db import SCHEMA_VERSION, bump_data_version, init_db
from src.llm import fill_alert_messages, fill_daily_summaries
from src.risk import assess

ROOT = Path(__file__).resolve().parent.parent
WATCHER_FILE = ROOT / "data" / "cache" / "watcher.json"  # heartbeat the dashboard reads
MIN_CROP_PIXELS = 2  # smaller boxes cannot be classified
CHUNK = 200
UPLOAD_TIMEOUT_SECONDS = 600
JOB_FILE_RETRIES = 200  # 10 seconds in all


def load_config():
    return yaml.safe_load((ROOT / "src" / "config.yaml").read_text())


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def triage(top_confidence, cfg):
    """Burst decision: keep, send to human review, or discard as empty."""
    if top_confidence >= cfg["keep_min"]:
        return "keep"
    return "review" if top_confidence >= cfg["discard_below"] else "discard"


def exif_time(image):
    """EXIF capture time as 'YYYY-MM-DD HH:MM:SS', or None."""
    try:
        exif = image.getexif()
        value = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
        return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


class Pipeline:
    def __init__(self, cfg=None, db_path=None, detector=None, use_llm=False):
        self.cfg = cfg or load_config()
        self.use_llm = use_llm
        self.db_path = ROOT / (db_path or os.environ.get("WILDWATCH_DB") or self.cfg["paths"]["database"])
        self.conn = init_db(self.db_path)
        if detector is None:  # torch loads here, so importing this file from Streamlit stays safe
            from src.detect import Detector
            detector = Detector(self.cfg, ROOT)
        self.detector = detector
        self.filter_thr = self.cfg["detection"]["confidence_threshold"]
        self.count_thr = self.cfg["detection"]["count_threshold"]
        self.classifier_error = None

    def existing_by_hash(self, sha):
        return self.conn.execute("SELECT image_id FROM images WHERE sha256 = ?", (sha,)).fetchone()

    def prepare(self, item):
        """Read and detect one file. item: path, plus optional metadata fields."""
        path = Path(item["path"])
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        image_id = item.get("image_id") or sha[:32]
        record = {
            "image_id": image_id, "file_path": relative(path), "source": item.get("source", "incoming"),
            "split": item.get("split"), "seq_id": item.get("seq_id"), "frame_num": item.get("frame_num"),
            "camera_id": item.get("camera_id"), "captured_at": item.get("captured_at"), "width": None,
            "height": None, "sha256": sha, "is_ir": None, "is_night": None, "is_empty": None,
            "gt_labels": json.dumps(item["labels"]) if item.get("labels") is not None else None,
            "gt_n_boxes": item.get("n_boxes"), "status": "ok", "error": None, "processed_at": now(),
            "top_confidence": None, "triage": None,
        }
        from src.detect import load_image
        image, error = load_image(path)
        if image is None:
            record.update(status="unreadable", error=error, seq_id=None)
            return record, []
        activity = self.cfg["activity"]
        ir = is_ir_image(image, activity["ir_channel_tolerance"])
        record.update(width=image.width, height=image.height, is_ir=int(ir),
                      is_night=int(is_night(ir, is_dark_scene(image, activity))))
        record["seq_id"] = record["seq_id"] or f"incoming-{image_id}"
        record["captured_at"] = record["captured_at"] or exif_time(image) or item.get("fallback_time")
        from src.detect import filter_detections
        raw = self.detector.detect_image(image)
        record["top_confidence"] = max((d["confidence"] for d in raw), default=0.0)
        detections = filter_detections(raw, self.filter_thr)
        record["is_empty"] = int(not detections)
        return record, detections

    def add_species(self, batch):
        """Fill species on animal boxes across a batch of (record, detections)."""
        items, world_items = [], []
        for record, detections in batch:
            boxes = []
            for d in detections:
                if d["category"] != "animal":
                    continue
                d["species"], d["species_confidence"], d["classifier"] = "unknown", None, None
                big_enough = (d["w"] * record["width"] >= MIN_CROP_PIXELS and
                              d["h"] * record["height"] >= MIN_CROP_PIXELS)
                if big_enough:
                    boxes.append(d)
            if boxes:
                path = str(ROOT / record["file_path"]) if not Path(record["file_path"]).is_absolute() else record["file_path"]
                # Only uploads use the global path. CCT images always keep the CCT rule.
                is_upload = record["source"] == "incoming" and self.cfg.get("upload_mode") == "global"
                (world_items if is_upload else items).append((path, boxes))
        payload = [{"path": p, "boxes": b} for p, b in items]
        results, self.classifier_error = classify_boxes(payload, ROOT)
        if self.classifier_error:
            print("WARNING classifier failed, using unknown:", self.classifier_error, flush=True)
        method = self.cfg["classification"]["method"]
        for path, boxes in items:
            for box, result in zip(boxes, results.get(path, [])):
                box["species"], box["species_confidence"] = result["species"], result["score"]
                box["classifier"] = method if not self.classifier_error else None
        if world_items:
            found, error = classify_global([{"path": p, "boxes": b} for p, b in world_items], ROOT, self.cfg)
            self.classifier_error = self.classifier_error or error
            if error:
                print("WARNING global identification failed in part:", error, flush=True)
            for path, boxes in world_items:
                for box, result in zip(boxes, found[path]):
                    box["species"], box["species_confidence"], box["classifier"] = result["species"], result["score"], result["model"]

    def save_image(self, record, detections):
        c = self.conn
        c.execute("DELETE FROM detections WHERE image_id = ?", (record["image_id"],))
        columns = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        c.execute(f"INSERT OR REPLACE INTO images ({columns}) VALUES ({marks})", tuple(record.values()))
        for d in detections:
            c.execute("INSERT INTO detections (image_id, category, confidence, x, y, w, h, species, "
                      "species_confidence, classifier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (record["image_id"], d["category"], d["confidence"], d["x"], d["y"], d["w"], d["h"],
                       d.get("species"), d.get("species_confidence"), d.get("classifier")))

    def rebuild_burst(self, seq_id):
        """Recompute one burst's rows from the database, so re-runs stay correct."""
        c = self.conn
        frames = c.execute("SELECT * FROM images WHERE seq_id = ? AND status = 'ok'", (seq_id,)).fetchall()
        c.execute("DELETE FROM alerts WHERE burst_id = ?", (seq_id,))
        c.execute("DELETE FROM bursts WHERE burst_id = ?", (seq_id,))
        if not frames:
            return
        all_boxes = [[dict(r) for r in c.execute("SELECT * FROM detections WHERE image_id = ?", (f["image_id"],))]
                     for f in frames]
        votes = burst_vote(all_boxes, self.cfg["classification"]["burst_vote_min_confidence"])
        voted = apply_burst_vote(all_boxes, votes)
        kept = [[d for d in boxes if d["confidence"] >= self.count_thr] for boxes in voted]
        times = sorted(f["captured_at"] for f in frames if f["captured_at"])
        night = any(f["is_night"] for f in frames)
        camera = frames[0]["camera_id"]
        decision = triage(max((f["top_confidence"] or 0) for f in frames), self.cfg["triage"])
        c.execute("UPDATE images SET triage = ? WHERE seq_id = ?", (decision, seq_id))
        for label, count in burst_counts(kept).items():
            c.execute("INSERT INTO bursts (burst_id, species, camera_id, start_time, end_time, n_frames, "
                      "min_observed_count, is_night) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (seq_id, label, camera, times[0] if times else None, times[-1] if times else None,
                       len(frames), count, int(night)))
            boxes = [d for frame_boxes in kept for d in frame_boxes if detection_label(d) == label]
            guess = bool(boxes) and all(str(d.get("classifier") or "").endswith(" guess") for d in boxes)
            alert = assess(label, count, night, camera, self.cfg, guess)
            if alert["danger_score"] > 0 or alert["possible_poacher"]:
                c.execute("INSERT INTO alerts (burst_id, species, camera_id, village, min_observed_count, "
                          "danger_score, level, rangers, vehicle, possible_poacher, created_at, is_guess) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (seq_id, label, camera, alert["village"], count, alert["danger_score"], alert["level"],
                           alert["rangers"], int(alert["vehicle"]), int(alert["possible_poacher"]), now(), int(guess)))

    def result_json(self, image_id):
        """The per-image JSON documented in docs/SCHEMA.md."""
        c = self.conn
        row = dict(c.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone())
        detections = [dict(d) for d in c.execute("SELECT * FROM detections WHERE image_id = ?", (image_id,))]
        for d in detections:
            d.pop("detection_id"), d.pop("image_id")
        alert = c.execute("SELECT * FROM alerts WHERE burst_id = ? ORDER BY danger_score DESC LIMIT 1",
                          (row["seq_id"],)).fetchone()
        if alert:
            alert = {k: alert[k] for k in ("alert_id", "burst_id", "species", "min_observed_count", "danger_score", "level",
                                           "rangers", "vehicle", "possible_poacher", "village", "message",
                                           "message_source", "is_guess")}
            alert["vehicle"], alert["possible_poacher"] = bool(alert["vehicle"]), bool(alert["possible_poacher"])
            alert["is_guess"] = bool(alert["is_guess"])
        flags = {k: None if row[k] is None else bool(row[k]) for k in ("is_ir", "is_night", "is_empty")}
        return {"schema_version": SCHEMA_VERSION, "image_id": image_id, "file_path": row["file_path"],
                "status": row["status"], "error": row["error"], "camera_id": row["camera_id"],
                "captured_at": row["captured_at"], **flags, "triage": row["triage"], "detections": detections,
                "alert": alert}

    def process(self, items, progress=None):
        """Run a list of items. Returns one result dict per item. Bad files never raise.
        progress, if given, is called with each stage name as the work reaches it."""
        report = progress or (lambda stage: None)
        results = {}
        for start in range(0, len(items), CHUNK):
            chunk = items[start:start + CHUNK]
            batch, duplicates = [], []
            seen = {}
            report("detect")
            for item in chunk:
                sha = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
                earlier = self.existing_by_hash(sha)
                if (earlier and earlier["image_id"] != item.get("image_id")) or sha in seen:
                    duplicates.append((item["path"], earlier["image_id"] if earlier else seen[sha]))
                    continue
                record, detections = self.prepare(item)
                seen[sha] = record["image_id"]
                batch.append((record, detections))
            report("species")
            self.add_species([b for b in batch if b[0]["status"] == "ok"])
            report("count")
            for record, detections in batch:
                self.save_image(record, detections)
            for seq_id in {r["seq_id"] for r, _ in batch if r["seq_id"]}:
                self.rebuild_burst(seq_id)
            self.conn.commit()
            report("risk")
            fill_alert_messages(self.conn, self.cfg, self.use_llm)
            for record, _ in batch:
                results[record["file_path"]] = self.result_json(record["image_id"])
            for path, original in duplicates:
                results[relative(path)] = {**self.result_json(original), "duplicate_of": original}
            print(f"{min(start + CHUNK, len(items))}/{len(items)}", flush=True)
        return results


def publish_update(pipeline):
    """Stats, then the Ranger's Brief, then data_version + 1. The brief runs in its own process
    because it uses pandas, which crashes beside torch. If it fails, pages must still refresh."""
    command = [sys.executable, "-m", "src.brief", "--db", str(pipeline.db_path)]
    if not pipeline.use_llm:
        command.append("--no-llm")
    try:
        done = subprocess.run(command, cwd=ROOT, timeout=180)
        if done.returncode == 0:
            return
        print("WARNING brief was not refreshed: exit", done.returncode, flush=True)
    except subprocess.TimeoutExpired:
        print("WARNING brief was not refreshed: timed out", flush=True)
    bump_data_version(pipeline.conn)


def sample_items(cfg, include_test):
    splits = {"train", "cis_val", "trans_val"} | ({"cis_test", "trans_test"} if include_test else set())
    items = []
    for r in csv.DictReader(open(ROOT / cfg["paths"]["manifest"])):
        if r["split"] in splits:
            n_boxes = int(r["n_boxes"])
            items.append({"path": str(ROOT / cfg["paths"]["images"] / r["file_name"]), "image_id": r["image_id"],
                          "source": "cct20_sample", "split": r["split"], "seq_id": r["seq_id"],
                          "frame_num": int(r["frame_num"]), "camera_id": r["location"],
                          "captured_at": r["date_captured"], "labels": r["labels"].split("|"),
                          "n_boxes": n_boxes or None})
    return items


def incoming_items(cfg, conn):
    """New files in data/incoming. An optional <name>.json beside a file gives labels,
    n_boxes, camera_id, captured_at or seq_id. A broken sidecar is ignored."""
    folder = ROOT / cfg["paths"]["incoming"]
    items = []
    for path in sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() != ".json"):
        if conn.execute("SELECT 1 FROM images WHERE sha256 = ?",
                        (hashlib.sha256(path.read_bytes()).hexdigest(),)).fetchone():
            continue
        item = {"path": str(path), "source": "incoming"}
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                item.update({k: meta[k] for k in ("labels", "n_boxes", "camera_id", "captured_at", "seq_id")
                             if k in meta})
            except (ValueError, AttributeError) as e:
                print(f"WARNING ignoring bad sidecar {sidecar.name}: {e}", flush=True)
        items.append(item)
    return items


def run_incoming(pipeline):
    """Process new files. If any had labels, re-run the regression check."""
    items = incoming_items(pipeline.cfg, pipeline.conn)
    if not items:
        return {}
    results = pipeline.process(items)
    publish_update(pipeline)
    for key, value in results.items():
        print(json.dumps({"file": key, "status": value["status"], "is_empty": value["is_empty"],
                          "alert": value["alert"]}), flush=True)
    if any(i.get("labels") is not None for i in items):
        print("Labeled image added: re-running regression", flush=True)
        code = subprocess.run([sys.executable, str(ROOT / "src" / "scripts" / "regression.py")]).returncode
        print("REGRESSION PASSED" if code == 0 else "REGRESSION FAILED: METRICS DROPPED", flush=True)
    return results


def retry_while_busy(action):
    """On Windows a file cannot be replaced while another process has it open, and the worker
    and the dashboard use job.json at the same moment, so wait for the other side to finish."""
    for attempt in range(JOB_FILE_RETRIES):
        try:
            return action()
        except PermissionError:
            if attempt == JOB_FILE_RETRIES - 1:
                raise
            time.sleep(0.05)


def write_job(folder, state, results=None, error=None, stage=None):
    """Status file the dashboard polls. Written to a temp name first so it is never half-read."""
    temporary = Path(folder) / "job.tmp"
    temporary.write_text(json.dumps({"state": state, "stage": stage, "results": results or [], "error": error}))
    retry_while_busy(lambda: temporary.replace(Path(folder) / "job.json"))


def read_job(folder):
    return json.loads(retry_while_busy((Path(folder) / "job.json").read_text))


def run_upload_job(folder, camera_id, use_llm, captured_at=None, prefer_exif=True):
    """Process every file in an upload folder, for the dashboard's background process.
    captured_at is the time typed on the upload form. With prefer_exif, a time stored in
    the photo wins and captured_at only fills in when the photo has none."""
    write_job(folder, "running", stage="upload")
    try:
        pipeline = Pipeline(use_llm=use_llm)
        paths = sorted(p for p in Path(folder).iterdir() if p.is_file() and not p.name.startswith("job."))
        time_field = "fallback_time" if prefer_exif else "captured_at"
        items = [{"path": str(p), "camera_id": camera_id, "source": "incoming", time_field: captured_at} for p in paths]
        results = pipeline.process(items, progress=lambda stage: write_job(folder, "running", stage=stage))
        by_name = {Path(k).name: v for k, v in results.items()}
        write_job(folder, "running", stage="brief")
        publish_update(pipeline)
        write_job(folder, "done", [{"file": p.name, **by_name[p.name]} for p in paths], stage="done")
    except Exception as e:  # the dashboard must always learn how the job ended
        write_job(folder, "failed", error=f"{type(e).__name__}: {e}", stage="error")


def process_upload(files, camera_id=None, progress=None, use_llm=True, timeout=UPLOAD_TIMEOUT_SECONDS):
    """The one upload path: save images -> detection -> species -> alert rules -> stats ->
    Ranger's Brief -> save -> data_version + 1. files is a list of (name, bytes). The work runs
    in run_upload_job in a separate process, because torch crashes beside Streamlit's pyarrow.
    Returns {"state", "error", "results", "new_alerts"}. Never raises for a failed job."""
    cfg = load_config()
    folder = ROOT / cfg["paths"]["incoming"] / "uploads" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    for name, data in files:
        (folder / Path(name).name).write_bytes(data)
    write_job(folder, "queued")
    command = [sys.executable, "-m", "src.pipeline", "--job-dir", str(folder)]
    if camera_id:
        command += ["--camera", str(camera_id)]
    if not use_llm:
        command.append("--no-llm")
    with open(folder / "job.log", "w") as log:
        worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    started, stage = time.time(), None
    while True:
        job = read_job(folder)
        if job["stage"] != stage and progress:
            stage = job["stage"]
            progress(stage)
        if job["state"] in ("done", "failed"):
            break
        if worker.poll() is not None:  # one more read, in case it finished just now
            job = read_job(folder)
            if job["state"] not in ("done", "failed"):
                job = {**job, "state": "failed", "error": f"The worker stopped with exit code {worker.returncode}"}
            break
        if time.time() - started > timeout:
            worker.kill()
            job = {**job, "state": "failed", "error": f"No result after {timeout} seconds. See {folder / 'job.log'}"}
            break
        time.sleep(0.5)
    worker.wait(timeout=30)  # it has written its result, and is only exiting now
    fresh = [r for r in job["results"] if not r.get("duplicate_of")]
    alerts = {r["alert"]["alert_id"]: r["alert"] for r in fresh if r.get("alert")}
    return {"state": job["state"], "error": job["error"], "results": job["results"], "new_alerts": list(alerts.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", help="process an upload folder for the dashboard")
    parser.add_argument("--camera", help="camera id for an upload job")
    parser.add_argument("--captured-at", help="capture time for an upload job")
    parser.add_argument("--ignore-exif", action="store_true", help="use --captured-at over the time in the photo")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="only cis_test and trans_test (Phase 5, once)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="recompute bursts and alerts from stored boxes")
    parser.add_argument("--no-llm", action="store_true", help="use template messages only")
    args = parser.parse_args()
    if args.job_dir:
        run_upload_job(args.job_dir, args.camera, not args.no_llm, args.captured_at, not args.ignore_exif)
        return
    pipeline = Pipeline(use_llm=not args.no_llm)
    if args.rebuild:
        seqs = [r[0] for r in pipeline.conn.execute("SELECT DISTINCT seq_id FROM images WHERE seq_id IS NOT NULL")]
        for seq_id in seqs:
            pipeline.rebuild_burst(seq_id)
        pipeline.conn.commit()
        print("REBUILD_DONE bursts:", len(seqs), flush=True)
        print("messages:", fill_alert_messages(pipeline.conn, pipeline.cfg, pipeline.use_llm),
              "summaries:", fill_daily_summaries(pipeline.conn, pipeline.cfg, pipeline.use_llm), flush=True)
        publish_update(pipeline)
    if args.sample:
        items = sample_items(pipeline.cfg, args.include_test or args.test_only)
        if args.test_only:
            items = [i for i in items if i["split"].endswith("_test")]
        start =time.perf_counter()
        pipeline.process(items)
        publish_update(pipeline)
        print(f"PIPELINE_DONE images={len(items)} seconds={time.perf_counter() - start:.1f}", flush=True)
    if args.once or args.watch:
        run_incoming(pipeline)
    while args.watch:
        WATCHER_FILE.write_text(json.dumps({"last_seen": time.time()}))
        time.sleep(5)
        run_incoming(pipeline)


if __name__ == "__main__":
    main()
