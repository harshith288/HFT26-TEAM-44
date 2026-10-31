"""Experiment: does a second detector pass find animals in bursts the first pass missed?

Reruns on every val burst with no animal box at confidence 0.2, using enhanced copies of
each frame. Reports how many labeled-animal bursts are recovered and how many truly
empty bursts are wrongly triggered, because recovering animals by flagging everything
would be useless. Saves results to data/cache/second_look.json.

Usage: python src/scripts/second_look.py [database]
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from PIL import ImageOps  # noqa: E402

from src.db import connect  # noqa: E402
from src.detect import Detector, load_image  # noqa: E402
from src.pipeline import load_config  # noqa: E402

GATE = 0.2
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
cfg = load_config()
db = sys.argv[1] if len(sys.argv) > 1 else ROOT / cfg["paths"]["database"]
conn = connect(db)
rows = [dict(r) for r in conn.execute("SELECT * FROM images WHERE split IN ('cis_val','trans_val') AND status='ok'")]
bursts = collections.defaultdict(list)
for r in rows:
    r["labels"] = json.loads(r["gt_labels"])
    bursts[r["seq_id"]].append(r)
found = {r[0] for r in conn.execute("SELECT DISTINCT image_id FROM detections WHERE category='animal' AND confidence>=?", (GATE,))}
todo = {k: v for k, v in bursts.items() if not any(f["image_id"] in found for f in v)}
print("bursts with no animal box:", len(todo), "of", len(bursts), flush=True)

detector = Detector(cfg, ROOT)


def views(image):
    yield "autocontrast", ImageOps.autocontrast(image, cutoff=1)
    yield "equalize", ImageOps.equalize(image)
    w, h = image.size
    for name, box in (("left", (0, 0, w // 2 + 100, h)), ("right", (w // 2 - 100, 0, w, h))):
        yield "zoom_" + name, ImageOps.autocontrast(image.crop(box), cutoff=1)


best = {}
for seq, frames in todo.items():
    scores = collections.defaultdict(float)
    for f in frames:
        path = Path(f["file_path"])
        image, _ = load_image(path if path.is_absolute() else ROOT / path)
        for name, view in views(image):
            for d in detector.detect_image(view):
                if d["category"] == "animal":
                    scores[name] = max(scores[name], d["confidence"])
    labels = {s for f in frames for s in f["labels"]}
    best[seq] = {"empty": labels == {"empty"}, "labels": sorted(labels), "scores": dict(scores)}
(ROOT / "data" / "cache" / "second_look.json").write_text(json.dumps(best))

animal = [b for b in best.values() if not b["empty"]]
empty = [b for b in best.values() if b["empty"]]
print(f"labeled-animal bursts {len(animal)} | empty bursts {len(empty)}")
for view in ("autocontrast", "equalize", "zoom_left", "zoom_right", "any"):
    for t in THRESHOLDS:
        hit = (lambda b: max(b["scores"].values(), default=0) >= t) if view == "any" \
            else (lambda b: b["scores"].get(view, 0) >= t)
        print(f"{view:<13} >= {t}: animal recovered {sum(map(hit, animal))}/{len(animal)}, "
              f"empty falsely hit {sum(map(hit, empty))}/{len(empty)}")
print("SECOND_LOOK_DONE")
