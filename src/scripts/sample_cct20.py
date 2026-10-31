"""Sample ~1,500 CCT20 images and download only those, one file at a time.

Whole sequences are sampled (never single frames) so burst counting (feature 3)
stays valid. Within each official split, sequences are picked round-robin over
(species, day/night) strata, so rare classes are kept rather than swamped.

Usage: python src/scripts/sample_cct20.py [--dry-run]
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import json
import random
from pathlib import Path

import requests
from PIL import Image

SEED = 42
ROOT = Path(__file__).resolve().parent.parent.parent
META = ROOT / "data" / "raw" / "eccv_18_annotation_files"
OUT = ROOT / "data" / "sample"
IMAGES = OUT / "images"
BASES = [
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/",
]
# Image budget per official split; total 1,500.
BUDGET = {"train": 600, "cis_val": 200, "trans_val": 200, "cis_test": 250, "trans_test": 250}
EMPTY_DRAWS = 6  # sequences drawn from each empty stratum per round-robin pass


def is_day(date_captured):
    """Timestamp-only day/night (06:00-17:59 = day). IR check comes later, in activity.py."""
    return 6 <= int(date_captured[11:13]) < 18


def load_split(split):
    data = json.loads((META / f"{split}_annotations.json").read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    labels = collections.defaultdict(set)
    n_boxes = collections.Counter()
    for a in data["annotations"]:
        labels[a["image_id"]].add(cats[a["category_id"]])
        if "bbox" in a:
            n_boxes[a["image_id"]] += 1
    seqs = collections.defaultdict(list)
    for img in data["images"]:
        seqs[img["seq_id"]].append(img)
    return seqs, labels, n_boxes


def sample_split(split, rng):
    seqs, labels, n_boxes = load_split(split)
    freq = collections.Counter(s for v in labels.values() for s in v)
    strata = collections.defaultdict(list)
    for seq_id in sorted(seqs):
        imgs = seqs[seq_id]
        # Skip bursts with frames elsewhere: 224 official sequences span train and
        # cis_val (leakage), and 430 have frames in no split (count unknowable).
        if len(imgs) != imgs[0]["seq_num_frames"]:
            continue
        species ={s for i in imgs for s in labels[i["id"]]}
        non_empty = species - {"empty"} or species
        # A multi-species burst is filed under its rarest species.
        rarest = min(sorted(non_empty), key=lambda s: freq[s])
        strata[(rarest, is_day(imgs[0]["date_captured"]))].append(seq_id)
    for key in sorted(strata):
        rng.shuffle(strata[key])

    rows, keys = [], sorted(strata)
    while len(rows) < BUDGET[split] and any(strata.values()):
        for key in keys:
            # Empty is oversampled so the threshold is tuned on enough empties.
            # This means % removed on this sample is NOT the natural CCT rate.
            draws = EMPTY_DRAWS if key[0] == "empty" else 1
            for _ in range(draws):
                if strata[key] and len(rows) < BUDGET[split]:
                    seq_id = strata[key].pop()
                    for img in seqs[seq_id]:
                        rows.append({
                            "image_id": img["id"], "file_name": img["file_name"], "split": split,
                            "seq_id": seq_id, "frame_num": img.get("frame_num"),
                            "seq_num_frames": img.get("seq_num_frames"), "location": img["location"],
                            "date_captured": img["date_captured"], "width": img["width"],
                            "height": img["height"], "day_by_timestamp": is_day(img["date_captured"]),
                            "labels": "|".join(sorted(labels[img["id"]])), "n_boxes": n_boxes[img["id"]],
                        })
    return rows


def download(row):
    """Returns (file_name, error or None). Skips files already present and valid."""
    path = IMAGES / row["file_name"]
    errors = []
    for attempt_base in BASES:
        try:
            if not path.exists():
                r = requests.get(attempt_base + row["file_name"], timeout=60)
                r.raise_for_status()
                path.write_bytes(r.content)
            with Image.open(path) as im:
                im.verify()
            return row["file_name"], None
        except Exception as e:  # keep trying the mirror, then report the real error
            errors.append(f"{attempt_base}: {e}")
            path.unlink(missing_ok=True)
    return row["file_name"], " | ".join(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(SEED)
    rows = [r for split in BUDGET for r in sample_split(split, rng)]
    OUT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("sampled images:", len(rows), "sequences:", len({r["seq_id"] for r in rows}))
    print("per split:", dict(collections.Counter(r["split"] for r in rows)))
    print("day/night:", dict(collections.Counter(r["day_by_timestamp"] for r in rows)))
    print("with boxes:", sum(1 for r in rows if r["n_boxes"]))
    print("labels:", collections.Counter(s for r in rows for s in r["labels"].split("|")).most_common())
    if args.dry_run:
        return

    # Remove images from any earlier sample so the folder always matches the manifest.
    wanted = {r["file_name"] for r in rows}
    removed = [p for p in IMAGES.iterdir() if p.name not in wanted]
    for p in removed:
        p.unlink()
    print("removed stale images:", len(removed))

    failed = {}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, (name, err) in enumerate(pool.map(download, rows), 1):
            if err:
                failed[name] = err
            if i % 100 == 0:
                print(f"{i}/{len(rows)} failed so far: {len(failed)}", flush=True)
    (OUT / "download_failures.json").write_text(json.dumps(failed, indent=1))
    size = sum(p.stat().st_size for p in IMAGES.iterdir())
    print(f"DOWNLOAD_DONE ok={len(rows) - len(failed)} failed={len(failed)} size_gb={size / 1e9:.2f}")


if __name__ == "__main__":
    main()
