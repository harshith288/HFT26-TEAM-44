"""Tune the discard threshold on tuning val ONLY, then report before/after on fresh val.

Rule, fixed before the sweep: the highest threshold losing under 1% of animal bursts on
tuning val, counting every burst labeled as animal. Fresh val never affects the choice.

Usage: python src/scripts/tune_discard.py
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

from src.db import connect  # noqa: E402

MAX_LOST_PCT = 1.0
THRESHOLDS = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
keep_min = cfg["triage"]["keep_min"]


def load(db):
    conn = connect(db)
    rows = [dict(r) for r in conn.execute("SELECT * FROM images WHERE split IN ('cis_val','trans_val') AND status='ok'")]
    conn.close()
    bursts = collections.defaultdict(list)
    for r in rows:
        bursts[r["seq_id"]].append(r)
    out = []
    for frames in bursts.values():
        labels = {s for f in frames for s in json.loads(f["gt_labels"])}
        out.append({"top": max(f["top_confidence"] for f in frames), "empty": labels == {"empty"},
                    "boxed": any(f["gt_n_boxes"] for f in frames), "frames": len(frames)})
    return out, len(rows)


def measure(bursts, threshold):
    animal = [b for b in bursts if not b["empty"]]
    empty = [b for b in bursts if b["empty"]]
    boxed = [b for b in animal if b["boxed"]]
    discard = [b for b in bursts if b["top"] < threshold]
    review = [b for b in bursts if threshold <= b["top"] < keep_min]
    lost = sum(b["top"] < threshold for b in animal)
    true_discard = sum(b["top"] < threshold for b in empty)
    return {"discard_below": threshold, "n_bursts": len(bursts), "n_animal_bursts": len(animal), "n_empty_bursts": len(empty),
            "discarded_bursts": len(discard), "review_bursts": len(review),
            "kept_bursts": len(bursts) - len(discard) - len(review),
            "pct_bursts_auto_discarded": 100 * len(discard) / len(bursts),
            "pct_bursts_needing_review": 100 * len(review) / len(bursts),
            "animal_bursts_lost": lost, "pct_animal_bursts_lost": 100 * lost / len(animal),
            "confirmed_animal_bursts_lost": sum(b["top"] < threshold for b in boxed), "n_confirmed_animal_bursts": len(boxed),
            "precision_of_discard": true_discard / len(discard) if discard else None,
            "pct_empty_bursts_discarded": 100 * true_discard / len(empty) if empty else None}


tuning, n_tuning = load(ROOT / cfg["paths"]["database"])
sweep = [measure(tuning, t) for t in THRESHOLDS]
passing = [m for m in sweep if m["pct_animal_bursts_lost"] < MAX_LOST_PCT]
chosen = max(m["discard_below"] for m in passing) if passing else None
print(f"{'thr':<8}{'discard':>8}{'review':>8}{'lost':>6}{'lost%':>7}{'conf.lost':>10}{'disc.prec':>10}")
for m in sweep:
    p = m["precision_of_discard"]
    print(f"{m['discard_below']:<8}{m['discarded_bursts']:>8}{m['review_bursts']:>8}{m['animal_bursts_lost']:>6}"
          f"{m['pct_animal_bursts_lost']:>7.1f}{m['confirmed_animal_bursts_lost']:>10}{'n/a' if p is None else round(p, 3):>10}")
print("chosen on tuning val:", chosen, "| rule met:", bool(passing))

fresh, n_fresh = load(ROOT / "data" / "extra_val" / "extra.db")
old_threshold = cfg["detection"]["confidence_threshold"]
result = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": n_tuning + n_fresh,
          "hardware": torch.cuda.get_device_name(0), "rule": "highest threshold losing under 1% of animal bursts on tuning val",
          "rule_met": bool(passing), "chosen_discard_below": chosen, "keep_min": keep_min, "tuning_sweep": sweep,
          "fresh_val_before": measure(fresh, old_threshold), "fresh_val_after": measure(fresh, chosen) if chosen else None,
          "tuning_val_before": measure(tuning, old_threshold)}
(ROOT / "docs" / "reports" / "discard_tuning.json").write_text(json.dumps(result, indent=1))
print("\nfresh before (single threshold, no review tier was in use):", json.dumps(result["fresh_val_before"], default=float))
print("\nfresh after:", json.dumps(result["fresh_val_after"], default=float))
print("TUNE_DONE")
