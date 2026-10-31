"""Download real non-CCT photos: a tuning set and a separate 20-photo test set.

Animals come from iNaturalist research-grade observations with open licences. Person and
empty field come from Wikimedia Commons. No observation is in both sets.

Usage: python src/scripts/global_photos.py
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
API = "https://api.inaturalist.org/v1/observations"
TUNE_EACH = 3
TUNE = ["Panthera leo", "Panthera tigris", "Panthera pardus", "Elephas maximus", "Melursus ursinus", "Sus scrofa", "Macaca mulatta",
        "Semnopithecus entellus", "Pavo cristatus", "Naja naja", "Daboia russelii", "Python molurus", "Crocodylus palustris", "Axis axis",
        "Rusa unicolor", "Boselaphus tragocamelus", "Antilope cervicapra", "Bos gaurus", "Bos taurus", "Bubalus bubalis", "Capra hircus",
        "Ovis aries", "Canis familiaris", "Felis catus", "Equus caballus", "Camelus dromedarius", "Giraffa camelopardalis", "Equus quagga",
        "Hyaena hyaena", "Canis lupus", "Ursus arctos", "Rhinoceros unicornis", "Varanus bengalensis", "Gavialis gangeticus", "Bubo bubo",
        "Milvus migrans", "Acridotheres tristis", "Ailurus fulgens", "Gorilla gorilla", "Osphranter rufus", "Phascolarctos cinereus",
        "Vulpes vulpes", "Hippopotamus amphibius", "Struthio camelus"]
# subject, scientific name, words any of which count as a hit (chosen before any model ran)
TEST = [("lion", "Panthera leo", "lion"), ("tiger", "Panthera tigris", "tiger"), ("leopard", "Panthera pardus", "leopard"),
        ("elephant", "Elephas maximus", "elephant"), ("wild boar", "Sus scrofa", "boar|wild pig"), ("sloth bear", "Melursus ursinus", "sloth bear"),
        ("monkey", "Macaca mulatta", "macaque|monkey|langur"), ("peacock", "Pavo cristatus", "peafowl|peacock"),
        ("cobra", "Naja naja", "cobra"), ("crocodile", "Crocodylus palustris", "crocodile|mugger"), ("deer", "Axis axis", "chital|deer"),
        ("cow", "Bos taurus", "cattle|cow"), ("dog", "Canis familiaris", "dog"), ("cat", "Felis catus", "domestic cat|\\bcat\\b"),
        ("horse", "Equus caballus", "horse"), ("camel", "Camelus dromedarius", "camel"), ("giraffe", "Giraffa camelopardalis", "giraffe"),
        ("zebra", "Equus quagga", "zebra")]
COMMONS = [("person", "hiker walking on forest trail", "PERSON"), ("empty field", "empty grass field landscape", "EMPTY")]


def observations(scientific):
    response = requests.get(API, headers=HEADERS, timeout=60, params={
        "taxon_name": scientific, "quality_grade": "research", "photos": "true", "per_page": 60, "order_by": "votes",
        "photo_license": "cc0,cc-by,cc-by-nc"})
    response.raise_for_status()
    found = []
    for o in response.json()["results"]:
        photo = next((p for p in o["photos"] if p.get("license_code") in ("cc0", "cc-by", "cc-by-nc") and p.get("url")), None)
        if photo:
            found.append((o["id"], photo["url"].replace("square", "large"), photo.get("attribution", "")))
    random.Random(42).shuffle(found)
    return found


def fetch(url, path):
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            with Image.open(path) as image:
                image.verify()
            return True
        except (requests.RequestException, OSError):
            time.sleep(2 * (attempt + 1))
    path.unlink(missing_ok=True)
    return False


def commons_url(search):
    response = requests.get("https://commons.wikimedia.org/w/api.php", headers=HEADERS, timeout=60, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrsearch": f"{search} filetype:bitmap",
        "gsrlimit": 10, "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1024})
    response.raise_for_status()
    pages = sorted(response.json()["query"]["pages"].values(), key=lambda p: p["index"])
    return [(p["imageinfo"][0]["thumburl"], p["title"]) for p in pages if p["title"].lower().endswith((".jpg", ".jpeg"))]


(OUT / "tune").mkdir(parents=True, exist_ok=True)
(OUT / "test").mkdir(parents=True, exist_ok=True)
tune_rows, test_rows, used = [], [], set()
cache = {}
for scientific in TUNE:
    cache[scientific] = observations(scientific)
    time.sleep(1.2)
    for oid, url, credit in cache[scientific][:TUNE_EACH]:
        name = f"{scientific.replace(' ', '_')}_{oid}.jpg"
        used.add(oid)
        if fetch(url, OUT / "tune" / name):
            tune_rows.append({"file": name, "scientific": scientific, "observation": oid, "attribution": credit})
for i, (subject, scientific, words) in enumerate(TEST, 1):
    pool = cache.get(scientific) or observations(scientific)
    oid, url, credit = next(o for o in pool if o[0] not in used)
    used.add(oid)
    name = f"{i:02d}_{subject.replace(' ', '_')}.jpg"
    if fetch(url, OUT / "test" / name):
        test_rows.append({"file": name, "subject": subject, "scientific": scientific, "hit_words": words, "source": f"iNaturalist {oid}", "attribution": credit})
    time.sleep(0.6)
for j, (subject, search, words) in enumerate(COMMONS, len(TEST) + 1):
    url, title = commons_url(search)[0]
    name = f"{j:02d}_{subject.replace(' ', '_')}.jpg"
    if fetch(url, OUT / "test" / name):
        test_rows.append({"file": name, "subject": subject, "scientific": "", "hit_words": words, "source": title, "attribution": "Wikimedia Commons"})
for label, rows in (("tune", tune_rows), ("test", test_rows)):
    with open(OUT / f"{label}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print("tune photos:", len(tune_rows), "| test photos:", len(test_rows))
print("PHOTOS_DONE")
