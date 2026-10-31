"""Run MDv6 on the sample and cache raw boxes in data/cache/detections.json.
Test splits are refused unless --include-test is given (Phase 5 only).

Usage: python src/scripts/run_detection.py [--include-test]
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.detect import Detector  # noqa: E402

CACHE = ROOT / "data" / "cache" / "detections.json"
parser = argparse.ArgumentParser()
parser.add_argument("--include-test", action="store_true")
args = parser.parse_args()

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
rows = list(csv.DictReader(open(ROOT / cfg["paths"]["manifest"])))
splits = {"train", "cis_val", "trans_val"} | ({"cis_test", "trans_test"} if args.include_test else set())
rows = [r for r in rows if r["split"] in splits]

CACHE.parent.mkdir(parents=True, exist_ok=True)
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
todo = [r for r in rows if r["image_id"] not in cache]
detector = Detector(cfg, ROOT)
# First call builds the model on the GPU, so it is left out of the timing.
detector.detect_file(ROOT / cfg["paths"]["images"] / todo[0]["file_name"])

start = time.perf_counter()
for i, r in enumerate(todo, 1):
    cache[r["image_id"]] = detector.detect_file(ROOT / cfg["paths"]["images"] / r["file_name"])
    if i % 200 == 0:
        print(f"{i}/{len(todo)}", flush=True)
elapsed = time.perf_counter() - start
CACHE.write_text(json.dumps(cache))

report = {
    "command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(),
    "n_real_images": len(todo), "hardware": torch.cuda.get_device_name(0),
    "model": cfg["detection"]["weights"], "image_size": cfg["detection"]["image_size"],
    "ms_per_image_including_load": round(elapsed / len(todo) * 1000, 1),
    "unreadable": sum(cache[r["image_id"]]["status"] != "ok" for r in todo),
}
(ROOT / "docs" / "reports" / "detection_speed.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
print("DETECTION_DONE")
