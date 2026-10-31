"""Save the test confusion matrix as JSON, for the dashboard's grid.

docs/reports/final_eval.json keeps only per-class results and the PNG. This reads the same
stored predictions with the same burst rule, so it adds no new evaluation and changes
nothing. A test checks that its diagonal gives the frozen final accuracy.

Usage: python src/scripts/export_confusion.py
"""
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.classify import CLASSES, burst_vote  # noqa: E402
from src.db import connect  # noqa: E402

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
gate = cfg["classification"]["burst_vote_min_confidence"]
conn = connect(ROOT / cfg["paths"]["database"])
images = [dict(r) for r in conn.execute("SELECT * FROM images WHERE split IN ('cis_test', 'trans_test') AND status = 'ok'")]
boxes = collections.defaultdict(list)
for d in conn.execute("SELECT * FROM detections"):
    boxes[d["image_id"]].append(dict(d))
bursts = collections.defaultdict(list)
for r in images:
    bursts[r["seq_id"]].append(r)

pairs = collections.Counter()
for r in images:
    labels = json.loads(r["gt_labels"])
    if len(labels) == 1 and labels[0] in CLASSES:
        votes = burst_vote([boxes[f["image_id"]] for f in bursts[r["seq_id"]]], gate)
        pairs[labels[0], votes[0] if votes else "no_detection"] += 1

rows = sorted({t for t, _ in pairs})
columns = rows + sorted({p for _, p in pairs} - set(rows))
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": sum(pairs.values()),
          "hardware": "not applicable (reads stored predictions)", "rows": rows, "columns": columns,
          "matrix": [[pairs[t, c] for c in columns] for t in rows]}
(ROOT / "docs" / "reports" / "confusion_matrix_test.json").write_text(json.dumps(report, indent=1))
print("images", report["n_real_images"], "correct", sum(pairs[c, c] for c in rows), "columns", columns)
