"""Choose how uploads make a best guess when neither model passes its threshold.

Uses only the cached tuning photos, not the check photos. A guess is right if its genus
matches the verified genus. Writes docs/reports/global_guess.json.

Usage: python src/scripts/global_guess_tune.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.classify import needs_fallback, parse_label  # noqa: E402
from src.species import load_species  # noqa: E402

OUT = ROOT / "data" / "global_photos"
cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())["global_identification"]
rows = list(csv.DictReader(open(OUT / "tune.csv", encoding="utf-8")))
data = json.loads((OUT / "tune_scores.json").read_text())
species = load_species()


def options(path):
    """(SpeciesNet guess or None, BioCLIP guess), each (name, score, genus)."""
    r = data["raw"][path][0]
    sn = None
    if r["classes"]:
        common, taxonomy, vague = parse_label(r["classes"][0])
        if not vague:
            sn = (common, r["scores"][0], taxonomy[3])
    b = data["named"][path][0]
    return sn, (b["species"], b["score"], species[b["species"]]["taxonomy"][3])


def pick(method, sn, bc):
    if method == "speciesnet_first":
        return sn or bc
    if method == "bioclip_first":
        return bc
    return max([x for x in (sn, bc) if x], key=lambda x: x[1])


results = {}
for method in ("speciesnet_first", "bioclip_first", "higher_score"):
    right = wrong = 0
    for r in rows:
        path = str(OUT / "tune" / r["file"])
        if not data["boxes"].get(path):
            continue
        result = data["raw"][path][0]
        b = data["named"][path][0]
        if not needs_fallback(result["classes"], result["scores"], cfg) or b["score"] >= cfg["bioclip_min_score"]:
            continue  # already named with confidence
        sn, bc = options(path)
        guess = pick(method, sn, bc)
        if guess[2] == r["scientific"].split()[0].lower():
            right += 1
        else:
            wrong += 1
    results[method] = {"right": right, "wrong": wrong, "pct_right": round(100 * right / max(1, right + wrong), 1)}
best = max(results, key=lambda k: results[k]["right"])
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(rows), "hardware": "not applicable (cached scores)",
          "scope": "tuning photos that were unknown under the thresholds", "methods": results, "chosen": best}
(ROOT / "docs" / "reports" / "global_guess.json").write_text(json.dumps(report, indent=1))
print(json.dumps({"methods": results, "chosen": best}, indent=1))
