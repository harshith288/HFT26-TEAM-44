"""Phase 5: final evaluation. The test splits are read once, with everything frozen.

Refuses to run on test if docs/reports/final_eval.json already exists.
Check the code first on val, where the answers are known: --dry-run-val prints only.

Usage:
  python src/scripts/final_eval.py --dry-run-val
  python src/scripts/final_eval.py
"""
import collections
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, precision_score  # noqa: E402

from src.classify import CLASSES, apply_burst_vote, burst_vote  # noqa: E402
from src.db import connect  # noqa: E402

SEED = 42
BOOTSTRAPS = 2000
GAP_FLAG_POINTS = 10
OUT = ROOT / "docs" / "reports" / "final_eval.json"
dry_run = "--dry-run-val" in sys.argv
SPLITS = ("cis_val", "trans_val") if dry_run else ("cis_test", "trans_test")
if not dry_run and OUT.exists():
    sys.exit("REFUSED: the test splits have already been evaluated once.")

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
gate = cfg["classification"]["burst_vote_min_confidence"]
count_thr = cfg["detection"]["count_threshold"]
conn = connect(ROOT / cfg["paths"]["database"])
marks = ",".join("?" for _ in SPLITS)
images = [dict(r) for r in conn.execute(f"SELECT * FROM images WHERE split IN ({marks}) AND status = 'ok'", SPLITS)]
if not images:
    sys.exit("No images for these splits. Run the pipeline first.")
boxes = collections.defaultdict(list)
for d in conn.execute("SELECT * FROM detections"):
    boxes[d["image_id"]].append(dict(d))
bursts = collections.defaultdict(list)
for r in images:
    r["labels"] = json.loads(r["gt_labels"])
    r["group"] = "ir_night" if r["is_ir"] else ("color_night" if r["is_night"] else "day")
    bursts[r["seq_id"]].append(r)


def voted_frames(seq):
    frames = [boxes[f["image_id"]] for f in bursts[seq]]
    return apply_burst_vote(frames, burst_vote(frames, gate))


def burst_prediction(seq):
    votes = burst_vote([boxes[f["image_id"]] for f in bursts[seq]], gate)
    return votes[0] if votes else "no_detection"


scored = [r for r in images if len(r["labels"]) == 1 and r["labels"][0] in CLASSES]
for r in scored:
    r["truth"], r["prediction"] = r["labels"][0], burst_prediction(r["seq_id"])
    r["boxed_burst"] = any(f["gt_n_boxes"] for f in bursts[r["seq_id"]])


def bootstrap_accuracy(rows):
    """95% range for accuracy, resampling whole bursts because frames of a burst are alike."""
    if not rows:
        return None
    by_burst = collections.defaultdict(list)
    for r in rows:
        by_burst[r["seq_id"]].append(r["truth"] == r["prediction"])
    groups = list(by_burst.values())
    rng = random.Random(SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        picks = [groups[rng.randrange(len(groups))] for _ in groups]
        right = sum(sum(g) for g in picks)
        values.append(100 * right / sum(len(g) for g in picks))
    values.sort()
    return [round(values[int(0.025 * BOOTSTRAPS)], 1), round(values[int(0.975 * BOOTSTRAPS)], 1)]


def species_metrics(rows):
    if not rows:
        return {"n": 0}
    truth = [r["truth"] for r in rows]
    pred = [r["prediction"] for r in rows]
    labels = sorted(set(truth))
    return {"n": len(rows), "n_bursts": len({r["seq_id"] for r in rows}), "n_classes": len(labels),
            "accuracy": 100 * sum(t == p for t, p in zip(truth, pred)) / len(rows),
            "accuracy_95_range": bootstrap_accuracy(rows),
            "macro_precision": 100 * precision_score(truth, pred, labels=labels, average="macro", zero_division=0),
            "macro_f1": 100 * f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
            "missed": sum(p == "no_detection" for p in pred),
            "wrong_species": sum(p not in (t, "no_detection") for t, p in zip(truth, pred))}


def triage_metrics(seqs):
    rows = []
    for seq in seqs:
        frames = bursts[seq]
        labels = {s for f in frames for s in f["labels"]}
        rows.append({"tier": frames[0]["triage"], "empty": labels == {"empty"}, "boxed": any(f["gt_n_boxes"] for f in frames)})
    animal = [r for r in rows if not r["empty"]]
    empty = [r for r in rows if r["empty"]]
    confirmed = [r for r in animal if r["boxed"]]
    tiers = collections.Counter(r["tier"] for r in rows)
    discarded = [r for r in rows if r["tier"] == "discard"]
    lost = sum(r["tier"] == "discard" for r in animal)
    return {"n_bursts": len(rows), "n_animal_bursts": len(animal), "n_empty_bursts": len(empty), "auto_discarded": tiers["discard"],
            "needs_review": tiers["review"], "kept": tiers["keep"],
            "pct_auto_discarded": 100 * tiers["discard"] / len(rows), "pct_needs_review": 100 * tiers["review"] / len(rows),
            "pct_kept": 100 * tiers["keep"] / len(rows), "animal_bursts_lost": lost,
            "pct_animal_bursts_lost": 100 * lost / len(animal) if animal else None,
            "confirmed_animal_bursts_lost": sum(r["tier"] == "discard" for r in confirmed),
            "n_confirmed_animal_bursts": len(confirmed),
            "discard_precision": sum(r["empty"] for r in discarded) / len(discarded) if discarded else None,
            "pct_empty_bursts_discarded": 100 * sum(r["tier"] == "discard" for r in empty) / len(empty) if empty else None}


def count_metrics(seqs):
    errors = []
    for seq in seqs:
        frames = bursts[seq]
        labels = {s for f in frames for s in f["labels"]}
        if "car" in labels or not any(f["gt_n_boxes"] for f in frames):
            continue
        if not all(f["gt_n_boxes"] or f["labels"] == ["empty"] for f in frames):
            continue
        truth = max(f["gt_n_boxes"] or 0 for f in frames)
        seen = max(sum(d["category"] == "animal" and d["confidence"] >= count_thr for d in f) for f in voted_frames(seq))
        errors.append(abs(seen - truth))
    return {"n_bursts": len(errors), "mae": sum(errors) / len(errors) if errors else None,
            "pct_exact": 100 * sum(e == 0 for e in errors) / len(errors) if errors else None}


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def box_metrics(rows):
    sizes = {r["image_id"]: (r["width"], r["height"]) for r in rows}
    truth = collections.defaultdict(list)
    for split in SPLITS:
        data = json.loads((ROOT / "data" / "raw" / "eccv_18_annotation_files" / f"{split}_annotations.json").read_text())
        for a in data["annotations"]:
            if a["image_id"] in sizes and "bbox" in a:
                w, h = sizes[a["image_id"]]
                truth[a["image_id"]].append((a["bbox"][0] / w, a["bbox"][1] / h, a["bbox"][2] / w, a["bbox"][3] / h))
    tp = fp = fn = 0
    for r in rows:
        if not r["gt_n_boxes"]:
            continue
        remaining = list(truth[r["image_id"]])
        for d in sorted((d for d in boxes[r["image_id"]] if d["confidence"] >= count_thr), key=lambda d: -d["confidence"]):
            box = (d["x"], d["y"], d["w"], d["h"])
            best = max(remaining, key=lambda g: iou(box, g), default=None)
            if best is not None and iou(box, best) >= 0.5:
                remaining.remove(best)
                tp += 1
            else:
                fp += 1
        fn += len(remaining)
    return {"tp": tp, "fp": fp, "fn": fn, "precision_iou50": tp / (tp + fp) if tp + fp else None,
            "recall_iou50": tp / (tp + fn) if tp + fn else None}


def filtered(key, value):
    return [r for r in scored if r[key] == value]


def seqs_where(test):
    return [s for s, frames in bursts.items() if test(frames)]


groups = {"all": (scored, list(bursts), images)}
for split in SPLITS:
    groups[split] = (filtered("split", split), seqs_where(lambda f: f[0]["split"] == split), [r for r in images if r["split"] == split])
for period in ("day", "ir_night", "color_night"):
    groups[period] = (filtered("group", period), seqs_where(lambda f: f[0]["group"] == period), [r for r in images if r["group"] == period])
results = {name: {"species": species_metrics(s), "triage": triage_metrics(q), "count": count_metrics(q), "boxes": box_metrics(i)}
           for name, (s, q, i) in groups.items()}
results["all"]["species_confirmed_by_box"] = species_metrics([r for r in scored if r["boxed_burst"]])
results["all"]["species_no_box_in_burst"] = species_metrics([r for r in scored if not r["boxed_burst"]])
cis, trans = SPLITS
gap = results[cis]["species"]["accuracy"] - results[trans]["species"]["accuracy"]
missed = [r for r in scored if r["prediction"] == "no_detection"]

print("images", len(images), "| bursts", len(bursts), "| scored", len(scored))
for name in results:
    s, t, c, b = (results[name][k] for k in ("species", "triage", "count", "boxes"))
    if not s["n"]:
        print(f"[{name}] no scored images, triage {t['auto_discarded']}/{t['needs_review']}/{t['kept']}")
        continue
    print(f"[{name}] acc {s['accuracy']:.1f} {s['accuracy_95_range']} prec {s['macro_precision']:.1f} f1 {s['macro_f1']:.1f} n {s['n']} "
          f"classes {s['n_classes']} missed {s['missed']} wrong {s['wrong_species']} | discard/review/keep {t['auto_discarded']}/"
          f"{t['needs_review']}/{t['kept']} lost {t['animal_bursts_lost']}/{t['n_animal_bursts']} confirmed lost "
          f"{t['confirmed_animal_bursts_lost']}/{t['n_confirmed_animal_bursts']} | count MAE {c['mae']} n {c['n_bursts']} | "
          f"boxes P {b['precision_iou50']} R {b['recall_iou50']}")
print("confirmed by box:", {k: round(v, 1) if isinstance(v, float) else v for k, v in results["all"]["species_confirmed_by_box"].items()})
print("missed with a ground-truth box:", sum(bool(r["gt_n_boxes"]) for r in missed), "of", len(missed))
print(f"cis minus trans accuracy gap: {gap:.1f} points")
if dry_run:
    print("DRY_RUN_DONE")
    sys.exit()

truth = [r["truth"] for r in scored]
pred = [r["prediction"] for r in scored]
labels = sorted(set(truth))
precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
per_class = {c: {"precision": float(p), "recall": float(r), "f1": float(f), "n": int(n)}
             for c, p, r, f, n in zip(labels, precision, recall, f1, support)}
columns = labels + sorted(set(pred) - set(labels))
matrix = confusion_matrix(truth, pred, labels=columns)[:len(labels)]
shares = matrix / matrix.sum(1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(0.62 * len(columns) + 3, 0.5 * len(labels) + 2.5))
ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True label")
ax.set_title(f"Final test: burst vote, {len(truth)} real images")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if matrix[i, j]:
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=9, color="white" if shares[i, j] > 0.55 else "#1a1a1a")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "reports" / "confusion_matrix_test.png", dpi=150)

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(images),
          "hardware": torch.cuda.get_device_name(0), "frozen_at_commit": commit, "splits": SPLITS, "n_bursts": len(bursts),
          "n_scored": len(scored), "groups": results, "per_class": per_class,
          "cis_minus_trans_accuracy_points": gap, "generalization_flag": abs(gap) > GAP_FLAG_POINTS,
          "missed_images": len(missed), "missed_with_a_ground_truth_box": sum(bool(r["gt_n_boxes"]) for r in missed),
          "note": "Empty bursts are oversampled and classes are balanced in this sample, so triage shares are not natural rates."}
OUT.write_text(json.dumps(report, indent=1))
print("per class:", {c: (round(v["precision"], 2), round(v["recall"], 2), v["n"]) for c, v in per_class.items()})
print("FINAL_EVAL_DONE")
