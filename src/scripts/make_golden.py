"""Build the 30-image golden set from VAL images and copy it to data/golden.

Picks are seeded and use only ground truth, never model output. Animal frames must
have ground-truth boxes, so their labels are confirmed for that frame. Expected values
come from CCT annotations and are saved in tests/golden_expected.json.

CCT has no person label, but real people do appear in its photos. One train image of
two hikers with a dog (labeled "dog"), confirmed by eye, is added with expect_person.

Usage: python src/scripts/make_golden.py
"""
import csv
import json
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SEED = 42
HIKER_IMAGE_ID = "58b6673d-23d2-11e8-a6a3-ec086b02610b"
cfg =yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
manifest = list(csv.DictReader(open(ROOT / cfg["paths"]["manifest"])))
rows = [r for r in manifest if r["split"] in ("cis_val", "trans_val")]
hiker = next(r for r in manifest if r["image_id"] == HIKER_IMAGE_ID)
rng = random.Random(SEED)
rng.shuffle(rows)

picked, used_bursts = [], set()


def take(condition, n):
    for r in rows:
        if n and r["seq_id"] not in used_bursts and condition(r):
            picked.append(r)
            used_bursts.add(r["seq_id"])
            n -= 1


boxed_single = lambda r: int(r["n_boxes"]) == 1 and "|" not in r["labels"]  # noqa: E731
take(lambda r: int(r["n_boxes"]) >= 2, 4)
take(lambda r: r["labels"] == "car" and int(r["n_boxes"]) > 0, 2)
take(lambda r: r["labels"] == "empty", 6)
species = sorted({r["labels"] for r in rows if boxed_single(r) and r["labels"] != "car"})
for s in species:
    take(lambda r: boxed_single(r) and r["labels"] == s, 1)
while len(picked) < 29:
    before = len(picked)
    for s in species:
        if len(picked) < 29:
            take(lambda r: boxed_single(r) and r["labels"] == s, 1)
    if len(picked) == before:
        raise SystemExit(f"Only {before} golden images available")
picked.append(hiker)

folder = ROOT / cfg["paths"]["golden"]
folder.mkdir(parents=True, exist_ok=True)
expected = []
for r in picked:
    shutil.copy(ROOT / cfg["paths"]["images"] / r["file_name"], folder / r["file_name"])
    labels = r["labels"].split("|")
    expected.append({"image_id": r["image_id"], "file_name": r["file_name"], "split": r["split"],
                     "camera_id": r["location"], "captured_at": r["date_captured"], "labels": labels,
                     "n_boxes": int(r["n_boxes"]), "is_empty": labels == ["empty"],
                     "expect_person": r["image_id"] == HIKER_IMAGE_ID})
(ROOT / "tests" / "golden_expected.json").write_text(json.dumps(expected, indent=1))
print("golden images:", len(expected), "|", sorted(r["labels"] for r in picked))
