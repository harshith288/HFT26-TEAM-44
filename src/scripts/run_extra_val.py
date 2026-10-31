"""Fresh real images: official val bursts that are not in the main sample.

They are picked at random, so classes appear in natural proportions and empties are
not oversampled. Used only to confirm changes tuned on the main val images.
Test splits are never read. Results go to data/extra_val/extra.db.

Usage: python src/scripts/run_extra_val.py
"""
import collections
import concurrent.futures as cf
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from src.detect import load_image  # noqa: E402
from src.pipeline import Pipeline, load_config  # noqa: E402

SEED = 42
TARGET_IMAGES = 1500
BASE = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"
OUT = ROOT / "data" / "extra_val"
IMAGES = OUT / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

cfg = load_config()
used = {r["seq_id"] for r in csv.DictReader(open(ROOT / cfg["paths"]["manifest"]))}
bursts = []
for split in ("cis_val", "trans_val"):
    data = json.loads((ROOT / "data" / "raw" / "eccv_18_annotation_files" / f"{split}_annotations.json").read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    labels, boxes = collections.defaultdict(set), collections.Counter()
    for a in data["annotations"]:
        labels[a["image_id"]].add(cats[a["category_id"]])
        boxes[a["image_id"]] += "bbox" in a
    seqs = collections.defaultdict(list)
    for img in data["images"]:
        seqs[img["seq_id"]].append(img)
    for seq_id, imgs in sorted(seqs.items()):
        if seq_id not in used and len(imgs) == imgs[0]["seq_num_frames"]:
            bursts.append([{"id": i["id"], "file": i["file_name"], "split": split, "seq": seq_id,
                            "frame": i.get("frame_num"), "camera": str(i["location"]), "time": i["date_captured"],
                            "labels": sorted(labels[i["id"]]), "n_boxes": boxes[i["id"]]} for i in imgs])

random.Random(SEED).shuffle(bursts)
rows = []
for burst in bursts:
    if len(rows) >= TARGET_IMAGES:
        break
    rows.extend(burst)
print("fresh images:", len(rows), "bursts:", len({r["seq"] for r in rows}),
      "labels:", collections.Counter(s for r in rows for s in r["labels"]).most_common(), flush=True)


def download(row):
    path = IMAGES / row["file"]
    try:
        if not path.exists():
            response = requests.get(BASE + row["file"], timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        return None if load_image(path)[0] is not None else row["file"]
    except requests.RequestException:
        return row["file"]


with cf.ThreadPoolExecutor(max_workers=8) as pool:
    failed = {f for f in pool.map(download, rows) if f}
print("download failures:", len(failed), flush=True)

items = [{"path": str(IMAGES / r["file"]), "image_id": r["id"], "source": "cct20_sample", "split": r["split"],
          "seq_id": r["seq"], "frame_num": r["frame"], "camera_id": r["camera"], "captured_at": r["time"],
          "labels": r["labels"], "n_boxes": r["n_boxes"] or None} for r in rows if r["file"] not in failed]
pipeline = Pipeline(cfg, db_path=OUT / "extra.db")
pipeline.process(items)
print("EXTRA_DONE images:", len(items), "classifier error:", pipeline.classifier_error, flush=True)
