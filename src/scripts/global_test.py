"""Run the 20 real non-CCT photos through the real upload pipeline, on a temporary database.

A hit needs the expected words in the final label, or for person a person box, and for the
empty field no animal alert. Writes docs/reports/global_test.json.

Usage: python src/scripts/global_test.py [world]
"""
import csv
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.pipeline import Pipeline, load_config  # noqa: E402

OUT = ROOT / "data" / "global_photos"
SET = "world" if "world" in sys.argv else "test"
rows = list(csv.DictReader(open(OUT / f"{SET}.csv", encoding="utf-8")))
cfg = load_config()
items = [{"path": str(OUT / SET / r["file"]), "source": "incoming"} for r in rows]
with tempfile.TemporaryDirectory() as folder:
    pipeline = Pipeline(cfg, db_path=Path(folder) / "global.db", use_llm=False)
    start = time.perf_counter()
    results = pipeline.process(items)
    seconds = time.perf_counter() - start
    by_name = {Path(k).name: v for k, v in results.items()}
    pipeline.conn.close()

report_rows, hits = [], 0
for r in rows:
    result = by_name[r["file"]]
    boxes = sorted(result["detections"], key=lambda d: -d["confidence"])
    animals = [d for d in boxes if d["category"] == "animal"]
    top = animals[0] if animals else None
    people = [d for d in boxes if d["category"] == "person"]
    label = top["species"] if top else ("person" if people else "nothing")
    if r["hit_words"] == "PERSON":
        hit = bool(people) and result["alert"] is not None and result["alert"]["possible_poacher"]
    elif r["hit_words"] == "EMPTY":
        hit = not [d for d in animals if d["confidence"] >= cfg["detection"]["count_threshold"]]
    else:
        hit = bool(top) and re.search(r["hit_words"], label) is not None
    hits += hit
    alert = result["alert"]
    report_rows.append({"file": r["file"], "subject": r["subject"], "hit": bool(hit), "label": label,
                        "confidence": top["species_confidence"] if top else None, "model": top["classifier"] if top else None,
                        "detector_confidence": boxes[0]["confidence"] if boxes else None, "triage": result["triage"],
                        "alert_level": alert["level"] if alert else None, "danger": alert["danger_score"] if alert else None,
                        "n_boxes": len(boxes)})
confident = sum(p["hit"] and not str(p["model"]).endswith("guess") for p in report_rows)
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(rows),
          "hardware": torch.cuda.get_device_name(0), "thresholds": cfg["global_identification"], "hits": hits, "hits_with_confidence": confident, "hit_rate_pct": 100 * hits / len(rows),
          "guesses": sum(str(p["model"]).endswith("guess") for p in report_rows), "unknown": sum(p["label"] == "unknown" for p in report_rows),
          "seconds_total": round(seconds, 1), "photos": report_rows,
          "note": "20 photos is a small check, not an accuracy figure. The 95% range is wide."}
(ROOT / "docs" / "reports" / f"global_{SET}.json").write_text(json.dumps(report, indent=1))
print(f"{'photo':<22}{'hit':<5}{'label':<28}{'conf':<7}{'model':<17}{'det':<6}{'triage':<9}{'alert':<9}")
for p in report_rows:
    conf = "-" if p["confidence"] is None else f"{p['confidence']:.3f}"
    det = "-" if p["detector_confidence"] is None else f"{p['detector_confidence']:.2f}"
    print(f"{p['file']:<22}{'YES' if p['hit'] else 'no':<5}{p['label'][:27]:<28}{conf:<7}{str(p['model']):<17}{det:<6}{str(p['triage']):<9}{str(p['alert_level']):<9}")
print(f"HIT RATE: {hits} of {len(rows)} ({report['hit_rate_pct']:.0f}%), {confident} of those confident, {report['guesses']} guesses, {report['unknown']} unknown, in {seconds:.0f}s")
print("GLOBAL_TEST_DONE")
