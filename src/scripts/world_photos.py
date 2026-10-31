"""Download a worldwide check set: real wild animals from every continent and ocean.

Uses iNaturalist research-grade observations, none in the tuning or 20-photo sets.
Lists 3 candidates per animal, so the first that really shows it can be kept by eye.

Usage: python src/scripts/world_photos.py
"""
import csv
import random
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "global_photos"
HEADERS = {"User-Agent": "WildWatch-hackathon-test/1.0 (student project)"}
# continent, subject, scientific name, words that count as a hit (set before any model ran)
WORLD = [("Africa", "hippopotamus", "Hippopotamus amphibius", "hippo"), ("Africa", "ring-tailed lemur", "Lemur catta", "lemur"),
         ("Africa", "black rhinoceros", "Diceros bicornis", "rhino"), ("Asia", "giant panda", "Ailuropoda melanoleuca", "panda"),
         ("Asia", "komodo dragon", "Varanus komodoensis", "komodo"), ("Asia", "orangutan", "Pongo pygmaeus", "orangutan"),
         ("Europe", "european badger", "Meles meles", "badger"), ("Europe", "eurasian lynx", "Lynx lynx", "lynx"),
         ("Europe", "red deer", "Cervus elaphus", "red deer|elk|deer"), ("North America", "polar bear", "Ursus maritimus", "polar bear"),
         ("North America", "american bison", "Bison bison", "bison"), ("North America", "moose", "Alces alces", "moose|elk"),
         ("South America", "capybara", "Hydrochoerus hydrochaeris", "capybara"), ("South America", "llama", "Lama glama", "llama"),
         ("South America", "jaguar", "Panthera onca", "jaguar"), ("Oceania", "red kangaroo", "Osphranter rufus", "kangaroo"),
         ("Oceania", "koala", "Phascolarctos cinereus", "koala"), ("Oceania", "platypus", "Ornithorhynchus anatinus", "platypus"),
         ("Antarctica", "emperor penguin", "Aptenodytes forsteri", "penguin"), ("Oceans", "walrus", "Odobenus rosmarus", "walrus")]

folder = OUT / "world"
folder.mkdir(exist_ok=True)
tuned = {r["observation"] for r in csv.DictReader(open(OUT / "tune.csv", encoding="utf-8"))}
rows = []
for i, (continent, subject, scientific, words) in enumerate(WORLD, 1):
    found = requests.get("https://api.inaturalist.org/v1/observations", headers=HEADERS, timeout=60, params={
        "taxon_name": scientific, "quality_grade": "research", "photos": "true", "per_page": 40, "order_by": "votes",
        "photo_license": "cc0,cc-by,cc-by-nc"}).json()["results"]
    pool = []
    for o in found:
        photo = next((p for p in o["photos"] if p.get("license_code") in ("cc0", "cc-by", "cc-by-nc") and p.get("url")), None)
        if photo and str(o["id"]) not in tuned:
            pool.append((o["id"], photo["url"].replace("square", "large"), photo.get("attribution", "")))
    random.Random(42).shuffle(pool)
    for k, (oid, url, credit) in enumerate(pool[:3]):
        name = f"{i:02d}_{subject.replace(' ', '_')}_{k}.jpg"
        (folder / name).write_bytes(requests.get(url, headers=HEADERS, timeout=60).content)
        with Image.open(folder / name) as image:
            image.verify()
        rows.append({"file": name, "subject": subject, "continent": continent, "scientific": scientific, "hit_words": words,
                     "source": f"iNaturalist {oid}", "attribution": credit, "candidate": k})
    time.sleep(1.2)
with open(OUT / "world_candidates.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("candidates", len(rows))
print("WORLD_DONE")
