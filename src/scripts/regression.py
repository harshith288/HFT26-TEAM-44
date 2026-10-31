"""Regression check on the golden set plus any labeled incoming images.

Fails loudly (exit 1) if a metric is more than 2 points worse than docs/reports/baseline.json.

Usage:
  python src/scripts/regression.py                    compare with the baseline
  python src/scripts/regression.py --update-baseline  save a new baseline
"""
import argparse
import json
import os
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.db import connect  # noqa: E402
from src.pipeline import Pipeline, load_config  # noqa: E402

MAX_DROP = 0.02
BASELINE = ROOT / "docs" / "reports" / "baseline.json"
HIGHER_IS_BETTER = ("empty_accuracy", "animal_found_rate", "species_accuracy", "person_found_rate")
LOWER_IS_BETTER = ("count_mae",)


def evaluate(expected_rows):
    """expected_rows: dicts with labels, n_boxes, is_empty, result (pipeline JSON)."""
    empty_ok = found = found_of = species_ok = species_of = 0
    count_errors = []
    people = [e for e in expected_rows if e.get("expect_person")]
    people_found = sum(any(d["category"] == "person" for d in e["result"]["detections"]) for e in people)
    for e in expected_rows:
        r = e["result"]
        empty_ok += bool(r["is_empty"]) == e["is_empty"]
        if e["is_empty"]:
            continue
        animals = [d for d in r["detections"] if d["category"] == "animal"]
        wants_animal = e["labels"] != ["car"]
        if wants_animal:
            found_of += 1
            found += bool(animals)
            species_of += 1
            if animals:
                top = max(animals, key=lambda d: d["confidence"])
                species_ok += top["species"] in e["labels"]
            threshold = load_config()["detection"]["count_threshold"]
            if e["n_boxes"]:
                count_errors.append(abs(sum(d["confidence"] >= threshold for d in animals) - e["n_boxes"]))
    return {"n_images": len(expected_rows), "empty_accuracy": empty_ok / len(expected_rows),
            "animal_found_rate": found / found_of if found_of else None,
            "species_accuracy": species_ok / species_of if species_of else None,
            "person_found_rate": people_found / len(people) if people else None,
            "count_mae": sum(count_errors) / len(count_errors) if count_errors else None}


def golden_rows(pipeline):
    expected = json.loads((ROOT / "tests" / "golden_expected.json").read_text())
    items = [{"path": str(ROOT / pipeline.cfg["paths"]["golden"] / e["file_name"]), "image_id": e["image_id"],
              "seq_id": e["image_id"], "camera_id": e["camera_id"], "captured_at": e["captured_at"],
              "source": "cct20_sample"} for e in expected]
    results = pipeline.process(items)
    by_id = {v["image_id"]: v for v in results.values()}
    return [{**e, "result": by_id[e["image_id"]]} for e in expected]


def incoming_rows(pipeline):
    """Labeled files already processed from data/incoming, read from the main database."""
    main = ROOT / pipeline.cfg["paths"]["database"]
    if not main.exists():
        return []
    conn = connect(main)
    rows = []
    for image in conn.execute("SELECT * FROM images WHERE source = 'incoming' AND gt_labels IS NOT NULL "
                              "AND status = 'ok'").fetchall():
        labels = json.loads(image["gt_labels"])
        detections = [dict(d) for d in conn.execute("SELECT * FROM detections WHERE image_id = ?",
                                                    (image["image_id"],))]
        rows.append({"labels": labels, "n_boxes": image["gt_n_boxes"], "is_empty": labels == ["empty"],
                     "result": {"is_empty": bool(image["is_empty"]), "detections": detections}})
    conn.close()
    return rows


def compare(metrics, baseline):
    """Returns a list of failure messages."""
    failures = []
    for key in HIGHER_IS_BETTER:
        if metrics[key] is not None and baseline[key] is not None and metrics[key] < baseline[key] - MAX_DROP:
            failures.append(f"{key} fell from {baseline[key]:.3f} to {metrics[key]:.3f}")
    for key in LOWER_IS_BETTER:
        if metrics[key] is not None and baseline[key] is not None and metrics[key] > baseline[key] + MAX_DROP:
            failures.append(f"{key} rose from {baseline[key]:.3f} to {metrics[key]:.3f}")
    return failures


def run(detector=None):
    with tempfile.TemporaryDirectory() as folder:
        cfg = load_config()
        pipeline = Pipeline(cfg, db_path=Path(folder) / "golden.db", detector=detector)
        rows = golden_rows(pipeline)
        rows += incoming_rows(pipeline)
        pipeline.conn.close()
    metrics = evaluate(rows)
    return metrics, pipeline.classifier_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    metrics, classifier_error = run()
    if classifier_error:
        print("REGRESSION FAILED: classifier did not run:", classifier_error)
        sys.exit(1)
    report = {**metrics, "command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(),
              "n_real_images": metrics["n_images"],
              "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()}
    print(json.dumps(report, indent=1))
    if args.update_baseline:
        BASELINE.write_text(json.dumps(report, indent=1))
        print("Baseline saved")
        return
    failures = compare(metrics, json.loads(BASELINE.read_text()))
    (ROOT / "docs" / "reports" / "regression_latest.json").write_text(json.dumps({**report, "failures": failures}, indent=1))
    if failures:
        print("\n!!! REGRESSION FAILED !!!")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)
    print("Regression passed")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    main()
