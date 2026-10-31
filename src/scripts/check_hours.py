"""Compare metadata hours with the IR signal, on train and val images only.
Writes data/cache/ir.json, used for the day/night split.

Usage: python src/scripts/check_hours.py
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.activity import channel_difference  # noqa: E402
from src.detect import load_image  # noqa: E402

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
rows = [r for r in csv.DictReader(open(ROOT / cfg["paths"]["manifest"]))
        if r["split"] in ("train", "cis_val", "trans_val")]
diffs = {}
for r in rows:
    image, error = load_image(ROOT / cfg["paths"]["images"] / r["file_name"])
    diffs[r["image_id"]] = channel_difference(image)
(ROOT / "data" / "cache" / "ir.json").write_text(json.dumps(diffs))

values = np.array(list(diffs.values()))
print("channel difference percentiles 1,5,25,50,75,95,99:",
      np.round(np.percentile(values, [1, 5, 25, 50, 75, 95, 99]), 2).tolist())
print("histogram:", [(float(e), int(c)) for c, e in zip(*np.histogram(values, bins=[0, .5, 1, 2, 3, 4, 6, 8, 12, 20, 200]))])
tolerance = cfg["activity"]["ir_channel_tolerance"]
hours = collections.defaultdict(lambda: [0, 0])
for r in rows:
    hour = int(r["date_captured"][11:13])
    hours[hour][0] += 1
    hours[hour][1] += diffs[r["image_id"]] < tolerance
print("hour images IR")
for hour in sorted(hours):
    print(f"{hour:>4} {hours[hour][0]:>6} {hours[hour][1]:>4}")
print("HOURS_DONE")
