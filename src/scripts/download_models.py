"""Download every model into models/ and verify checksums. Safe to re-run.

Usage: python src/scripts/download_models.py
"""
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
# Keep all caches inside the project so setup is self-contained.
os.environ["HF_HOME"] = str(MODELS / "hf")
os.environ["KAGGLEHUB_CACHE"] = str(MODELS / "kaggle")

import requests  # noqa: E402

ZENODO = "https://zenodo.org/api/records/15398270/files/{}/content"
WEIGHTS = {
    "MDV6-yolov9-e-1280.pt": "8e54811c26225304a3f3c71b4f20f0bb",
    "MDV6-yolov9-c.pt": "4ad433595f3f06e62dd67debb9329398",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_megadetector():
    for name, expected in WEIGHTS.items():
        path = MODELS / name
        if not (path.exists() and md5(path) == expected):
            print("downloading", name, flush=True)
            with requests.get(ZENODO.format(name), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        actual = md5(path)
        if actual != expected:
            raise RuntimeError(f"{name}: md5 {actual} != {expected}")
        print(f"OK {name} {path.stat().st_size / 1e6:.1f} MB md5 verified", flush=True)


def fetch_bioclip():
    import open_clip
    open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    print("OK BioCLIP", flush=True)


def fetch_dinov2():
    import timm
    timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    print("OK DINOv2 ViT-S/14", flush=True)


def fetch_speciesnet():
    from speciesnet import DEFAULT_MODEL
    from speciesnet.utils import ModelInfo
    info = ModelInfo(DEFAULT_MODEL)
    print("OK SpeciesNet", DEFAULT_MODEL, info.version if hasattr(info, "version") else "", flush=True)


if __name__ == "__main__":
    failed = []
    for step in (fetch_megadetector, fetch_bioclip, fetch_dinov2, fetch_speciesnet):
        try:
            step()
        except Exception as e:
            failed.append(step.__name__)
            print(f"FAILED {step.__name__}: {type(e).__name__}: {e}", flush=True)
    print("MODELS_DONE failed:", failed or "none", flush=True)
    sys.exit(1 if failed else 0)
