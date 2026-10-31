"""Compare species decision rules using the fixed target.

Target, set before any experiment: species accuracy and macro precision, both at least
90%, on all single-species images, with detector misses counted as wrong.
Reads the pipeline database. Test splits are never read.

Usage: python src/scripts/improve_eval.py [path to database]
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, precision_score, recall_score  # noqa: E402

from src.classify import CLASSES  # noqa: E402
from src.db import connect  # noqa: E402

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
db = sys.argv[1] if len(sys.argv) > 1 else ROOT / cfg["paths"]["database"]
conn = connect(db)
images = [dict(r) for r in conn.execute(
    "SELECT * FROM images WHERE split IN ('cis_val', 'trans_val') AND status = 'ok'")]
detections = collections.defaultdict(list)
for d in conn.execute("SELECT * FROM detections WHERE category = 'animal'"):
    detections[d["image_id"]].append(dict(d))
bursts = collections.defaultdict(list)
for r in images:
    r["labels"] = json.loads(r["gt_labels"])
    bursts[r["seq_id"]].append(r)
scored = [r for r in images if len(r["labels"]) == 1 and r["labels"][0] in CLASSES]


def image_rule(r):
    animals = detections[r["image_id"]]
    return max(animals, key=lambda d: d["confidence"])["species"] if animals else "no_detection"


def burst_rule(min_conf, weight):
    """Every frame takes the burst's vote. weight: 'count', 'detector', 'score'."""
    def rule(r):
        votes = collections.Counter()
        for frame in bursts[r["seq_id"]]:
            for d in detections[frame["image_id"]]:
                if d["confidence"] >= min_conf and d["species"] != "unknown":
                    votes[d["species"]] += {"count": 1, "detector": d["confidence"],
                                            "score": d["confidence"] * (d["species_confidence"] or 0)}[weight]
        return votes.most_common(1)[0][0] if votes else "no_detection"
    return rule


def score(rule, rows):
    truth = [r["labels"][0] for r in rows]
    pred = [rule(r) for r in rows]
    labels = sorted(set(truth))
    return {"n": len(rows), "accuracy": 100 * accuracy_score(truth, pred),
            "macro_precision": 100 * precision_score(truth, pred, labels=labels, average="macro", zero_division=0),
            "macro_recall": 100 * recall_score(truth, pred, labels=labels, average="macro", zero_division=0),
            "missed": sum(p == "no_detection" for p in pred), "wrong_species": sum(
                p not in (t, "no_detection") for p, t in zip(pred, truth))}


rules = {"image_top_box (current)": image_rule}
for min_conf in (0.05, 0.1, 0.2, 0.3):
    for weight in ("count", "detector", "score"):
        rules[f"burst conf>={min_conf} {weight}"] = burst_rule(min_conf, weight)

print(f"{'rule':<32}{'acc':>7}{'prec':>7}{'recall':>8}{'missed':>8}{'wrong':>7}")
for name, rule in rules.items():
    s = score(rule, scored)
    print(f"{name:<32}{s['accuracy']:7.1f}{s['macro_precision']:7.1f}{s['macro_recall']:8.1f}"
          f"{s['missed']:8d}{s['wrong_species']:7d}")
print("images scored:", len(scored))
