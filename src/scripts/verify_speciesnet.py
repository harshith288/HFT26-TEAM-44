"""Step 0 check: run SpeciesNet on the same real image as verify_models.py.

SpeciesNet must run in its own process: inside a process that already ran the
other models it fails with WinError 6714 on every filesystem call (root cause
not found; it does not happen in a fresh process).

Usage: python src/scripts/verify_speciesnet.py
"""
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["HF_HOME"] = str(ROOT / "models" / "hf")
os.environ["KAGGLEHUB_CACHE"] = str(ROOT / "models" / "kaggle")

import torch  # noqa: E402
from speciesnet import DEFAULT_MODEL, SpeciesNet  # noqa: E402

rows = list(csv.DictReader(open(ROOT / "data" / "sample" / "manifest.csv")))
target = next(r for r in rows if r["split"] == "train" and r["labels"] == "bobcat" and r["n_boxes"] == "1")
path = str(ROOT / "data" / "sample" / "images" / target["file_name"])
print("Real test image:", target["file_name"], "| ground truth:", target["labels"])

model = SpeciesNet(DEFAULT_MODEL, components="all", geofence=True)
timings = []
for _ in range(3):  # first call includes warm-up, so the last is the fair one
    start = time.perf_counter()
    out = model.predict(instances_dict={"instances": [{"filepath": path, "country": "USA"}]},
                        run_mode="single_thread")
    timings.append(round((time.perf_counter() - start) * 1000, 1))
pred = out["predictions"][0]
summary = {
    "command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": 1,
    "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "model": DEFAULT_MODEL, "ground_truth": target["labels"],
    "prediction": pred.get("prediction"), "prediction_score": pred.get("prediction_score"),
    "prediction_source": pred.get("prediction_source"),
    "top3_classes": pred.get("classifications", {}).get("classes", [])[:3],
    "top3_scores": pred.get("classifications", {}).get("scores", [])[:3],
    "detections": pred.get("detections"), "failures": pred.get("failures"), "ms_per_call": timings,
}
print(json.dumps(summary, indent=1))
(ROOT / "docs" / "reports" / "step0_speciesnet.json").write_text(json.dumps(summary, indent=1))
print("SPECIESNET_DONE")
