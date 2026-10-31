"""Classify boxes with SpeciesNet in its own process. Called by src.classify.classify_boxes.

Input JSON: [{"path": str, "boxes": [{"x", "y", "w", "h", "confidence"}]}]
Output JSON: {path: [{"species": str, "score": float or null}, one per input box]}
With "global" as a third argument: no country, and {"classes": [...], "scores": [...]} per box.

Usage: python src/scripts/speciesnet_worker.py input.json output.json [global]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["KAGGLEHUB_CACHE"] = str(ROOT / "models" / "kaggle")
sys.path.insert(0, str(ROOT))

from speciesnet import DEFAULT_MODEL, SpeciesNet  # noqa: E402

from src.classify import pick_species  # noqa: E402


def key(bbox):
    return tuple(round(v, 5) for v in bbox)


items = json.loads(Path(sys.argv[1]).read_text())
world = len(sys.argv) > 3 and sys.argv[3] == "global"
instances, detections = [], {}
for item in items:
    instances.append({"filepath": item["path"]} if world else {"filepath": item["path"], "country": "USA"})
    detections[item["path"]] = {"filepath": item["path"], "detections": [
        {"category": "1", "label": "animal", "conf": b["confidence"], "bbox": [b["x"], b["y"], b["w"], b["h"]]}
        for b in item["boxes"]]}

model = SpeciesNet(DEFAULT_MODEL, components="classifier")
# Threshold 0: the pipeline has already decided which boxes to keep.
raw = model.classify(instances_dict={"instances": instances}, detections_dict=detections,
                     run_mode="multi_thread", detection_threshold_for_classification=0.0)
by_path = {p["filepath"]: p.get("detections") or [] for p in raw["predictions"]}

out = {}
for item in items:
    found = {key(d["bbox"]): d.get("raw_classifications") for d in by_path.get(item["path"], [])}
    results = []
    for b in item["boxes"]:
        result = found.get(key([b["x"], b["y"], b["w"], b["h"]])) or {}
        if world:
            results.append({"classes": result.get("classes", []), "scores": result.get("scores", [])})
            continue
        species, score = pick_species(result.get("classes", []), result.get("scores", []))
        results.append({"species": species, "score": score})
    out[item["path"]] = results
Path(sys.argv[2]).write_text(json.dumps(out))
