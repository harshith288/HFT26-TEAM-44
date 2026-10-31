"""Download 20 real photos for trying the dashboard by hand, with a separate answer key.

They come from official CCT20 val bursts that are in neither the main database nor the
fresh val set, and never from the test splits. Each comes from a different burst.
Animal photos have a ground-truth box, so the label is confirmed for that frame.

Usage: python src/scripts/get_test_photos.py
"""
import collections
import csv
import hashlib
import json
import random
import sqlite3
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "my_test_photos"
RAW = ROOT / "data" / "raw" / "eccv_18_annotation_files"
BASE = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"
WANTED = {"bobcat": 2, "coyote": 2, "opossum": 2, "raccoon": 2, "rabbit": 1, "cat": 1, "dog": 1, "skunk": 1, "squirrel": 1,
          "bird": 1, "rodent": 1, "car": 1, "empty": 4}

used = set()
hashes = set()
for db in (ROOT / "data" / "wildwatch.db", ROOT / "data" / "extra_val" / "extra.db"):
    conn = sqlite3.connect(db)
    used |= {r[0] for r in conn.execute("SELECT seq_id FROM images WHERE seq_id IS NOT NULL")}
    hashes |= {r[0] for r in conn.execute("SELECT sha256 FROM images")}
    conn.close()

candidates = collections.defaultdict(list)
for split in ("cis_val", "trans_val"):
    data = json.loads((RAW / f"{split}_annotations.json").read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    labels, boxes = collections.defaultdict(set), collections.Counter()
    for a in data["annotations"]:
        labels[a["image_id"]].add(names[a["category_id"]])
        boxes[a["image_id"]] += "bbox" in a
    for image in data["images"]:
        label = labels[image["id"]]
        if image["seq_id"] in used or len(label) != 1:
            continue
        name = next(iter(label))
        if name in WANTED and (name == "empty" or boxes[image["id"]] == 1):
            candidates[name].append({"image": image, "label": name, "split": split})

rng = random.Random(42)
OUT.mkdir(parents=True, exist_ok=True)
picks = []
for name, n in WANTED.items():
    pool = sorted(candidates[name], key=lambda c: c["image"]["id"])
    rng.shuffle(pool)
    bursts = set()
    chosen = []
    for item in pool:
        if item["image"]["seq_id"] not in bursts and len(chosen) < n:
            bursts.add(item["image"]["seq_id"])
            chosen.append(item)
    if len(chosen) < n:
        print(f"only {len(chosen)} of {n} available for {name}")
    picks += chosen
rng.shuffle(picks)

rows = []
for i, item in enumerate(picks, 1):
    file_name = f"photo_{i:02d}.jpg"
    response = requests.get(BASE + item["image"]["file_name"], timeout=60)
    response.raise_for_status()
    path = OUT / file_name
    path.write_bytes(response.content)
    with Image.open(path) as image:
        image.verify()
    assert hashlib.sha256(response.content).hexdigest() not in hashes, "already in the database"
    rows.append({"file": file_name, "true_label": item["label"], "camera_in_cct": item["image"]["location"],
                 "captured_at": item["image"]["date_captured"], "cct_split": item["split"]})
with open(OUT / "answers.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("downloaded", len(rows), "to", OUT)
print(collections.Counter(r["true_label"] for r in rows).most_common())
