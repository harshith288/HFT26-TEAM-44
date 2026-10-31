"""Where do the remaining errors come from, under the chosen burst rule?

Also measures the empty filter per burst: a burst is removed only if nothing in it passes
the threshold. Writes docs/reports/improvement.json with one section per database.

Usage: python src/scripts/error_breakdown.py
"""
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from sklearn.metrics import accuracy_score, precision_score  # noqa: E402

from src.classify import CLASSES, burst_vote  # noqa: E402
from src.db import connect  # noqa: E402
from src.pipeline import load_config  # noqa: E402

cfg = load_config()
GATE = cfg["classification"]["burst_vote_min_confidence"]


def analyse(db):
    conn = connect(db)
    images = [dict(r) for r in conn.execute("SELECT * FROM images WHERE split IN ('cis_val','trans_val') AND status='ok'")]
    dets = collections.defaultdict(list)
    for d in conn.execute("SELECT * FROM detections"):
        dets[d["image_id"]].append(dict(d))
    bursts = collections.defaultdict(list)
    for r in images:
        r["labels"] = json.loads(r["gt_labels"])
        r["group"] = "ir_night" if r["is_ir"] else ("color_night" if r["is_night"] else "day")
        bursts[r["seq_id"]].append(r)

    def vote(seq):
        return burst_vote([dets[f["image_id"]] for f in bursts[seq]], GATE)

    scored = [r for r in images if len(r["labels"]) == 1 and r["labels"][0] in CLASSES]
    truth = [r["labels"][0] for r in scored]
    pred = [(vote(r["seq_id"]) or ["no_detection"])[0] for r in scored]
    labels = sorted(set(truth))

    def metrics(keep):
        t = [a for a, k in zip(truth, keep) if k]
        p = [b for b, k in zip(pred, keep) if k]
        if not t:
            return {"n": 0}
        names = sorted(set(t))
        return {"n": len(t), "accuracy": 100 * accuracy_score(t, p),
                "macro_precision": 100 * precision_score(t, p, labels=names, average="macro", zero_division=0)}

    boxed_burst = {s: any(f["gt_n_boxes"] for f in v) for s, v in bursts.items()}
    result = {"n_real_images": len(images), "n_scored": len(scored), "n_classes": len(labels),
              "all": metrics([True] * len(scored)),
              "bursts_with_a_ground_truth_box": metrics([boxed_burst[r["seq_id"]] for r in scored]),
              "bursts_with_no_ground_truth_box": metrics([not boxed_burst[r["seq_id"]] for r in scored])}
    for g in ("day", "ir_night", "color_night"):
        result[g] = metrics([r["group"] == g for r in scored])
    for s in ("cis_val", "trans_val"):
        result[s] = metrics([r["split"] == s for r in scored])
    missed = [r for r, p in zip(scored, pred) if p == "no_detection"]
    result["missed_images"] = len(missed)
    result["missed_with_a_ground_truth_box"] = sum(bool(r["gt_n_boxes"]) for r in missed)
    result["missed_by_class"] = dict(collections.Counter(r["labels"][0] for r in missed).most_common())
    wrong = collections.Counter((t, p) for t, p in zip(truth, pred) if p not in (t, "no_detection"))
    result["wrong_species"] = {f"{t}->{p}": n for (t, p), n in wrong.most_common(10)}

    # Per-burst empty filter: confidence_threshold on any category, plus the stricter gate.
    for name, threshold in (("filter_burst_at_filter_threshold", cfg["detection"]["confidence_threshold"]),
                            ("filter_burst_at_vote_gate", GATE)):
        tp = fp = fn = tn = 0
        for frames in bursts.values():
            empty_truth = all(f["labels"] == ["empty"] for f in frames)
            has = any(d["confidence"] >= threshold for f in frames for d in dets[f["image_id"]])
            tp += empty_truth and not has
            fp += not empty_truth and not has
            fn += empty_truth and has
            tn += not empty_truth and has
        result[name] = {"threshold": threshold, "n_bursts": len(bursts), "empty_kept": fn, "animal_bursts_lost": fp,
                        "precision_empty": tp / (tp + fp) if tp + fp else None,
                        "recall_empty": tp / (tp + fn) if tp + fn else None,
                        "pct_bursts_removed": 100 * (tp + fp) / len(bursts),
                        "pct_animal_bursts_lost": 100 * fp / (fp + tn) if fp + tn else None}
    return result


output = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(),
          "hardware": torch.cuda.get_device_name(0), "n_real_images": 0, "rule": f"burst vote, confidence >= {GATE}",
          "target": "species accuracy and macro precision both >= 90% on all single-species images"}
for name, db in (("tuning_val", ROOT / cfg["paths"]["database"]), ("fresh_val", ROOT / "data" / "extra_val" / "extra.db")):
    output[name] = analyse(db)
    output["n_real_images"] += output[name]["n_real_images"]
(ROOT / "docs" / "reports" / "improvement.json").write_text(json.dumps(output, indent=1))
print(json.dumps({k: v for k, v in output.items() if k in ("tuning_val", "fresh_val")}, indent=0, default=float))
