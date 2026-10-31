"""Features 1 and 3 on VAL: tune thresholds, measure the empty filter, boxes and counts.

Threshold rules, fixed before looking at results:
- Filter: highest threshold losing at most MAX_MISS of confirmed animal frames.
- Count: threshold with the lowest count MAE on val.

Usage: python src/scripts/eval_detection.py
"""
import collections
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.count import burst_animal_sum, burst_animal_total  # noqa: E402
from src.detect import filter_detections, is_empty  # noqa: E402

MAX_MISS = 0.01
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VAL = ("cis_val", "trans_val")

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
cache = json.loads((ROOT / "data" / "cache" / "detections.json").read_text())
rows = [r for r in csv.DictReader(open(ROOT / cfg["paths"]["manifest"])) if r["split"] in VAL]
channel_diff = json.loads((ROOT / "data" / "cache" / "ir.json").read_text())
tolerance = cfg["activity"]["ir_channel_tolerance"]
for r in rows:
    r["dets"] = cache[r["image_id"]]["detections"]
    r["empty_gt"] = r["labels"] == "empty"
    r["boxed"] = int(r["n_boxes"]) > 0
    # Split by the IR signal: raw timestamps put midnight at hour 12 (see src/activity.py).
    r["period"] = "night" if channel_diff[r["image_id"]] < tolerance else "day"


def safe(a, b):
    return a / b if b else None


def filter_metrics(subset, threshold):
    tp = sum(r["empty_gt"] and is_empty(r["dets"], threshold) for r in subset)
    fp = sum(not r["empty_gt"] and is_empty(r["dets"], threshold) for r in subset)
    fn = sum(r["empty_gt"] and not is_empty(r["dets"], threshold) for r in subset)
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    f1 = safe(2 * precision * recall, precision + recall) if precision and recall else 0.0
    boxed = [r for r in subset if r["boxed"]]
    lost_boxed = sum(is_empty(r["dets"], threshold) for r in boxed)
    bursts = collections.defaultdict(list)
    for r in subset:
        if not r["empty_gt"]:
            bursts[r["seq_id"]].append(r)
    lost_bursts = sum(all(is_empty(x["dets"], threshold) for x in v) for v in bursts.values())
    return {
        "threshold": threshold, "n_images": len(subset), "n_empty": tp + fn,
        "tp_empty": tp, "fp_animal_called_empty": fp, "fn_empty_kept": fn,
        "precision_empty": precision, "recall_empty": recall, "f1_empty": f1,
        # Labels are burst-level, so a frame labeled 'animal' may really be empty.
        # This version counts as errors only frames where a box proves an animal.
        "precision_empty_vs_confirmed_animals": safe(tp, tp + lost_boxed),
        "pct_images_removed": 100 * (tp + fp) / len(subset),
        "pct_animal_frames_discarded_all_labels": 100 * safe(fp, len(subset) - tp - fn),
        "n_boxed_frames": len(boxed), "n_boxed_frames_lost": lost_boxed,
        "pct_confirmed_animal_frames_lost": 100 * safe(lost_boxed, len(boxed)),
        "n_animal_bursts": len(bursts), "n_animal_bursts_lost_entirely": lost_bursts,
    }


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def load_gt_boxes():
    """Normalized ground-truth boxes per image, from the official annotation files."""
    boxes = collections.defaultdict(list)
    sizes = {r["image_id"]: (float(r["width"]), float(r["height"])) for r in rows}
    for split in VAL:
        data = json.loads((ROOT / "data" / "raw" / "eccv_18_annotation_files" /
                           f"{split}_annotations.json").read_text())
        for a in data["annotations"]:
            if a["image_id"] in sizes and "bbox" in a:
                w, h = sizes[a["image_id"]]
                x, y, bw, bh = a["bbox"]
                boxes[a["image_id"]].append((x / w, y / h, bw / w, bh / h))
    return boxes


def box_metrics(subset, threshold, gt_boxes):
    """Greedy one-to-one matching at IoU 0.5, on boxed frames only."""
    tp = fp = fn = 0
    for r in [x for x in subset if x["boxed"]]:
        gt = list(gt_boxes[r["image_id"]])
        kept = sorted(filter_detections(r["dets"], threshold), key=lambda d: -d["confidence"])
        for d in kept:
            box = (d["x"], d["y"], d["w"], d["h"])
            best = max(gt, key=lambda g: iou(box, g), default=None)
            if best is not None and iou(box, best) >= 0.5:
                gt.remove(best)
                tp += 1
            else:
                fp += 1
        fn += len(gt)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision_iou50": safe(tp, tp + fp),
            "recall_iou50": safe(tp, tp + fn)}


def count_bursts(subset):
    """Bursts usable for count MAE: animals only, every frame is boxed or labeled empty."""
    bursts = collections.defaultdict(list)
    for r in subset:
        bursts[r["seq_id"]].append(r)
    usable = []
    for frames in bursts.values():
        labels = {s for f in frames for s in f["labels"].split("|")}
        if "car" in labels or not any(f["boxed"] for f in frames):
            continue
        if all(f["boxed"] or f["empty_gt"] for f in frames):
            usable.append(frames)
    return usable


def count_metrics(bursts, threshold):
    errors, sum_errors, signed = [], [], []
    for frames in bursts:
        truth = max(int(f["n_boxes"]) for f in frames)
        kept = [filter_detections(f["dets"], threshold) for f in frames]
        errors.append(abs(burst_animal_total(kept) - truth))
        signed.append(burst_animal_total(kept) - truth)
        sum_errors.append(abs(burst_animal_sum(kept) - truth))
    n = len(bursts)
    return {"threshold": threshold, "n_bursts": n, "mae_max_rule": safe(sum(errors), n),
            "mean_signed_error": safe(sum(signed), n), "mae_sum_baseline": safe(sum(sum_errors), n),
            "pct_exact": 100 * safe(sum(e == 0 for e in errors), n)}


sweep = [filter_metrics(rows, t) for t in THRESHOLDS]
ok = [m for m in sweep if m["pct_confirmed_animal_frames_lost"] <= 100 * MAX_MISS]
chosen = max(ok, key=lambda m: m["threshold"])["threshold"] if ok else min(THRESHOLDS)

bursts = count_bursts(rows)
count_sweep = [count_metrics(bursts, t) for t in THRESHOLDS]
count_chosen = min(count_sweep, key=lambda m: (m["mae_max_rule"], -m["threshold"]))["threshold"]

gt_boxes = load_gt_boxes()
groups = {"all": rows, "day": [r for r in rows if r["period"] == "day"],
          "night": [r for r in rows if r["period"] == "night"],
          "cis_val": [r for r in rows if r["split"] == "cis_val"],
          "trans_val": [r for r in rows if r["split"] == "trans_val"]}
result = {
    "command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(),
    "n_real_images": len(rows), "hardware": torch.cuda.get_device_name(0),
    "split": "val only (cis_val + trans_val)", "day_night_source": "IR signal (night = IR image)",
    "note_natural_rate": "Empty bursts are oversampled, so pct_images_removed is not the natural CCT rate.",
    "rule_met": bool(ok), "chosen_filter_threshold": chosen, "chosen_count_threshold": count_chosen,
    "sweep": sweep, "count_sweep": count_sweep,
    "filter_by_group": {k: filter_metrics(v, chosen) for k, v in groups.items()},
    "boxes_by_group": {k: box_metrics(v, count_chosen, gt_boxes) for k, v in groups.items()},
    "count_by_group": {k: count_metrics(count_bursts(v), count_chosen) for k, v in groups.items()},
}
(ROOT / "docs" / "reports" / "phase2_detection.json").write_text(json.dumps(result, indent=1))


def fmt(v, digits=3):
    return "n/a" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


print("rule met:", result["rule_met"], "| filter threshold:", chosen, "| count threshold:", count_chosen)
print("\nthr   prec  rec   f1    %removed %lost(all labels) lost/boxed  bursts lost")
for m in sweep:
    print(f"{m['threshold']:<5} {fmt(m['precision_empty'])} {fmt(m['recall_empty'])} {fmt(m['f1_empty'])} "
          f"{m['pct_images_removed']:6.1f}   {m['pct_animal_frames_discarded_all_labels']:6.1f}"
          f"            {m['n_boxed_frames_lost']}/{m['n_boxed_frames']}     "
          f"{m['n_animal_bursts_lost_entirely']}/{m['n_animal_bursts']}")
print("\ncount sweep: thr MAE(max) signed MAE(sum) %exact n")
for m in count_sweep:
    print(f"{m['threshold']:<5} {fmt(m['mae_max_rule'])} {fmt(m['mean_signed_error'])} "
          f"{fmt(m['mae_sum_baseline'])} {m['pct_exact']:.1f} {m['n_bursts']}")
for section in ("filter_by_group", "boxes_by_group", "count_by_group"):
    print("\n" + section)
    for k, v in result[section].items():
        print(" ", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print("EVAL_DONE")
