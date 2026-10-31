"""Tune the upload thresholds on real non-CCT photos. Never uses the 20 test photos or CCT.

Runs each tuning photo's top box through SpeciesNet (no country) and BioCLIP, then sweeps both
thresholds. A label is right if its genus matches the verified genus.

Rule, fixed before the sweep: most right labels with wrong labels at or under 10% of photos.
Ties go to the higher thresholds. Writes docs/reports/global_tuning.json.

Usage: python src/scripts/global_tune.py
"""
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.classify import global_decision, parse_label, run_worker  # noqa: E402
from src.species import load_species  # noqa: E402

MAX_WRONG = 0.10
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
OUT = ROOT / "data" / "global_photos"
CACHE = OUT / "tune_scores.json"
cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
rows = list(csv.DictReader(open(OUT / "tune.csv", encoding="utf-8")))

if not CACHE.exists():
    detect = """
import json, sys, yaml
from pathlib import Path
sys.path.insert(0, '.')
from src.detect import Detector
cfg = yaml.safe_load(open('src/config.yaml'))
detector = Detector(cfg, '.')
rows = json.load(open(sys.argv[1]))
out = {}
for path in rows:
    found = [d for d in detector.detect_file(path)['detections'] if d['category'] == 'animal' and d['confidence'] >= cfg['detection']['confidence_threshold']]
    out[path] = max(found, key=lambda d: d['confidence']) if found else None
json.dump(out, open(sys.argv[2], 'w'))
"""
    paths = [str(OUT / "tune" / r["file"]) for r in rows]
    (OUT / "paths.json").write_text(json.dumps(paths))
    subprocess.run([sys.executable, "-c", detect, str(OUT / "paths.json"), str(OUT / "boxes.json")], cwd=ROOT, check=True)
    boxes = json.loads((OUT / "boxes.json").read_text())
    items = [{"path": p, "boxes": [b]} for p, b in boxes.items() if b]
    raw, error = run_worker("speciesnet_worker.py", items, ROOT, extra=("global",))
    assert not error, error
    named, error = run_worker("bioclip_worker.py", items, ROOT)
    assert not error, error
    CACHE.write_text(json.dumps({"boxes": boxes, "raw": raw, "named": named}))

data = json.loads(CACHE.read_text())
species = load_species()


def genus_of(name, classes):
    for label in classes:
        common, taxonomy, _ = parse_label(label)
        if common == name:
            return taxonomy[3]
    return species[name]["taxonomy"][3] if name in species else None


def score_setting(sn_min, bc_min):
    settings = {"speciesnet_min_score": sn_min, "bioclip_min_score": bc_min}
    right = wrong = unknown = no_box = 0
    by_model = {"SpeciesNet": [0, 0], "BioCLIP": [0, 0]}
    for r in rows:
        path = str(OUT / "tune" / r["file"])
        if not data["boxes"].get(path):
            no_box += 1
            continue
        result = data["raw"][path][0]
        name, _, model = global_decision(result["classes"], result["scores"], data["named"][path][0], settings)
        if model is None:
            unknown += 1
            continue
        correct = genus_of(name, result["classes"]) == r["scientific"].split()[0].lower()
        right += correct
        wrong += not correct
        by_model[model][0 if correct else 1] += 1
    return {"speciesnet_min_score": sn_min, "bioclip_min_score": bc_min, "right": right, "wrong": wrong, "unknown": unknown,
            "no_detection": no_box, "by_model_right_wrong": by_model}


sweep = [score_setting(a, b) for a in GRID for b in GRID]
allowed = [s for s in sweep if s["wrong"] <= MAX_WRONG * len(rows)]
best = max(allowed, key=lambda s: (s["right"], s["speciesnet_min_score"], s["bioclip_min_score"]))
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(rows), "hardware": "local GPU",
          "rule": "most right labels with wrong labels at or under 10% of photos; ties go to higher thresholds",
          "photos": "iNaturalist research grade, not CCT, not the 20 test photos", "chosen": best, "sweep": sweep}
(ROOT / "docs" / "reports" / "global_tuning.json").write_text(json.dumps(report, indent=1))
print("photos", len(rows), "| chosen", best)
for a in (0.3, 0.5, 0.7, 0.9):
    print({k: v for k, v in score_setting(a, best["bioclip_min_score"]).items() if k != "by_model_right_wrong"})
print("TUNE_GLOBAL_DONE")
