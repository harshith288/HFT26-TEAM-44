"""Run SpeciesNet (USA geofence) on the top MDv6 box of each val image.

Runs in its own process because SpeciesNet fails with WinError 6714 when it shares
a process with the other models. Writes data/cache/speciesnet.json.

Usage: python src/scripts/run_speciesnet.py
"""
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["KAGGLEHUB_CACHE"] = str(ROOT / "models" / "kaggle")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from speciesnet import DEFAULT_MODEL, SpeciesNet  # noqa: E402

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
threshold = cfg["detection"]["confidence_threshold"]
cache = json.loads((ROOT / "data" / "cache" / "detections.json").read_text())
rows = [r for r in csv.DictReader(open(ROOT / cfg["paths"]["manifest"])) if r["split"] in ("cis_val", "trans_val")]

instances, boxes, ids = [], {}, {}
for r in rows:
    animals = [d for d in cache[r["image_id"]]["detections"]
               if d["category"] == "animal" and d["confidence"] >= threshold]
    if not animals:
        continue
    top = max(animals, key=lambda d: d["confidence"])
    path = (ROOT / cfg["paths"]["images"] / r["file_name"]).as_posix()
    instances.append({"filepath": path, "country": "USA"})
    boxes[path] = {"filepath": path, "detections": [
        {"category": "1", "label": "animal", "conf": top["confidence"],
         "bbox": [top["x"], top["y"], top["w"], top["h"]]}]}
    ids[path] = r["image_id"]

classifier = SpeciesNet(DEFAULT_MODEL, components="classifier")
start = time.perf_counter()
# Threshold 0 so SpeciesNet classifies the same crops as the other candidates.
raw = classifier.classify(instances_dict={"instances": instances}, detections_dict=boxes, run_mode="multi_thread",
                          detection_threshold_for_classification=0.0)
elapsed = time.perf_counter() - start
del classifier
torch.cuda.empty_cache()

raw_by_path = {p["filepath"]: p for p in raw["predictions"]}
raw_copy = copy.deepcopy(raw_by_path)  # the ensemble step changes its input in place
ensemble =SpeciesNet(DEFAULT_MODEL, components="ensemble", geofence=True)
final = ensemble.ensemble_from_past_runs(instances_dict={"instances": instances},
                                         classifications_dict=raw_by_path, detections_dict=boxes)
out, failures = {}, 0
for p in final["predictions"]:
    dets = p.get("detections") or []
    raw_dets = raw_copy[p["filepath"]].get("detections") or [{}]
    if (p.get("failures") or not dets or "processed_classification" not in dets[0]
            or "raw_classifications" not in raw_dets[0]):
        failures += 1
        print("FAILED", json.dumps({"failures": p.get("failures"), "raw_keys": sorted(raw_copy[p["filepath"]]),
                                    "raw_det_keys": sorted(raw_dets[0]), "det_keys": sorted(dets[0]) if dets else None,
                                    "bbox": raw_dets[0].get("bbox")}))
        continue
    raw_det = raw_dets[0]["raw_classifications"]
    out[ids[p["filepath"]]] = {
        "processed": dets[0]["processed_classification"]["class"],
        "processed_score": dets[0]["processed_classification"]["score"],
        "raw_classes": raw_det["classes"], "raw_scores": raw_det["scores"]}
(ROOT / "data" / "cache" / "speciesnet.json").write_text(json.dumps(out))
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(instances),
          "hardware": torch.cuda.get_device_name(0), "model": DEFAULT_MODEL, "failures": failures,
          "ms_per_crop_classifier_only": round(elapsed / len(instances) * 1000, 1)}
(ROOT / "docs" / "reports" / "speciesnet_speed.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
print("SPECIESNET_DONE")
