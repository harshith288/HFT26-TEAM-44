"""Features 4, 6, 7: evaluate the pipeline's database on VAL and summarize alerts.

Day/night groups come from the pixels: day, ir_night, color_night.
Writes docs/reports/phase3_day_night.json and docs/reports/phase3_activity.json.

Usage: python src/scripts/eval_phase3.py
"""
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.activity import corrected_hour  # noqa: E402
from src.classify import CLASSES  # noqa: E402
from src.db import connect  # noqa: E402

VAL = ("cis_val", "trans_val")
cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
conn = connect(ROOT / cfg["paths"]["database"])
count_thr = cfg["detection"]["count_threshold"]
hardware = torch.cuda.get_device_name(0)


def group(row):
    if row["is_ir"]:
        return "ir_night"
    return "color_night" if row["is_night"] else "day"


images = [dict(r) for r in conn.execute("SELECT * FROM images WHERE status = 'ok'")]
detections = collections.defaultdict(list)
for d in conn.execute("SELECT * FROM detections"):
    detections[d["image_id"]].append(dict(d))
for r in images:
    r["labels"] = json.loads(r["gt_labels"])
    r["group"] = group(r)
    r["dets"] = detections[r["image_id"]]
    r["boxed"] = bool(r["gt_n_boxes"])
val = [r for r in images if r["split"] in VAL]


def safe(a, b):
    return a / b if b else None


def filter_metrics(rows):
    empty_gt = [r["labels"] == ["empty"] for r in rows]
    called = [bool(r["is_empty"]) for r in rows]
    tp = sum(a and b for a, b in zip(empty_gt, called))
    fp = sum(b and not a for a, b in zip(empty_gt, called))
    fn = sum(a and not b for a, b in zip(empty_gt, called))
    boxed = [r for r in rows if r["boxed"]]
    lost = sum(bool(r["is_empty"]) for r in boxed)
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    return {"n_images": len(rows), "n_empty": tp + fn, "precision_empty": precision, "recall_empty": recall,
            "f1_empty": safe(2 * precision * recall, precision + recall) if precision and recall else None,
            "precision_empty_vs_confirmed_animals": safe(tp, tp + lost),
            "pct_images_removed": 100 * safe(tp + fp, len(rows)) if rows else None,
            "pct_animal_frames_discarded_all_labels": 100 * safe(fp, len(rows) - tp - fn) if len(rows) - tp - fn else None,
            "n_boxed_frames": len(boxed), "n_boxed_frames_lost": lost}


def species_metrics(rows):
    rows = [r for r in rows if len(r["labels"]) == 1 and r["labels"][0] in CLASSES]
    truth, predicted = [], []
    for r in rows:
        animals = [d for d in r["dets"] if d["category"] == "animal"]
        top = max(animals, key=lambda d: d["confidence"])["species"] if animals else "no_detection"
        truth.append(r["labels"][0])
        predicted.append(top)
    if not truth:
        return {"n_images": 0}
    labels = sorted(set(truth))
    return {"n_images": len(truth), "n_classes": len(labels), "top1_accuracy": 100 * accuracy_score(truth, predicted),
            "macro_f1": 100 * f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)}


def count_metrics(rows):
    bursts = collections.defaultdict(list)
    for r in rows:
        bursts[r["seq_id"]].append(r)
    errors = []
    for frames in bursts.values():
        labels = {s for f in frames for s in f["labels"]}
        if "car" in labels or not any(f["boxed"] for f in frames):
            continue
        if not all(f["boxed"] or f["labels"] == ["empty"] for f in frames):
            continue
        truth = max(f["gt_n_boxes"] or 0 for f in frames)
        seen = max(sum(d["category"] == "animal" and d["confidence"] >= count_thr for d in f["dets"]) for f in frames)
        errors.append(abs(seen - truth))
    return {"n_bursts": len(errors), "mae_max_rule": safe(sum(errors), len(errors)),
            "pct_exact": 100 * safe(sum(e == 0 for e in errors), len(errors)) if errors else None}


groups = {"all": val}
for name in ("day", "ir_night", "color_night"):
    groups[name] = [r for r in val if r["group"] == name]
breakdown = {name: {"filter": filter_metrics(rows), "species": species_metrics(rows), "count": count_metrics(rows)}
             for name, rows in groups.items()}
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(val),
          "hardware": hardware, "split": "val only", "source": "pipeline database", "groups": breakdown}
(ROOT / "docs" / "reports" / "phase3_day_night.json").write_text(json.dumps(report, indent=1))

# Feature 4. True labels are burst-level; predictions come from the bursts table.
bursts = collections.defaultdict(list)
for r in images:
    bursts[r["seq_id"]].append(r)
truth_counts = collections.defaultdict(collections.Counter)
for frames in bursts.values():
    period = "night" if any(f["is_night"] for f in frames) else "day"
    for s in {s for f in frames for s in f["labels"]} - {"empty", "car"}:
        truth_counts[s][period] += 1
predicted_counts = collections.defaultdict(collections.Counter)
for b in conn.execute("SELECT species, is_night FROM bursts"):
    predicted_counts[b["species"]]["night" if b["is_night"] else "day"] += 1
hours = collections.Counter()
for r in images:
    hour = corrected_hour(r["captured_at"], bool(r["is_night"]))
    if hour is not None and r["camera_id"] != "120":
        hours[hour] += 1
all_clock_night = sum(1 for r in images if r["is_night"] and 6 <= (corrected_hour(r["captured_at"], True) or 0) < 18)

levels = collections.Counter(a["level"] for a in conn.execute("SELECT level FROM alerts"))
by_species = collections.Counter((a["species"], a["level"]) for a in conn.execute("SELECT species, level FROM alerts"))
activity = {
    "command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(images),
    "hardware": hardware, "splits": "train + val",
    "warning": "The sample was stratified by day/night, so these ratios show the sample, not natural behaviour.",
    "bursts_by_true_species": {k: dict(v) for k, v in sorted(truth_counts.items())},
    "bursts_by_predicted_species": {k: dict(v) for k, v in sorted(predicted_counts.items())},
    "images_by_hour_excluding_camera_120": dict(sorted(hours.items())),
    "night_images_with_a_daytime_clock": all_clock_night,
    "alerts_by_level": dict(levels), "alerts_by_species_level": {f"{s}/{l}": n for (s, l), n in sorted(by_species.items())},
    "n_alerts": sum(levels.values()), "n_bursts": len(bursts),
}
(ROOT / "docs" / "reports" / "phase3_activity.json").write_text(json.dumps(activity, indent=1))


def show(v):
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else v)


for name, m in breakdown.items():
    print(f"\n[{name}]")
    for section, values in m.items():
        print(" ", section, {k: show(v) for k, v in values.items()})
print("\nbursts true:", activity["bursts_by_true_species"])
print("bursts predicted:", activity["bursts_by_predicted_species"])
print("night images, daytime clock:", all_clock_night)
print("alerts:", activity["alerts_by_level"], "of", activity["n_alerts"], "| bursts", activity["n_bursts"])
print("alerts by species/level:", activity["alerts_by_species_level"])
print("PHASE3_EVAL_DONE")
