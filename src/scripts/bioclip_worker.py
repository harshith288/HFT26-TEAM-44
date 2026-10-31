"""Name animal boxes with BioCLIP zero-shot against src/species_list.csv, in its own process.

Input JSON: [{"path": str, "boxes": [{"x", "y", "w", "h"}]}]
Output JSON: {path: [{"species": str, "score": float, "top": [[name, score], ...]}, one per box]}

Usage: python src/scripts/bioclip_worker.py input.json output.json
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["HF_HOME"] = str(ROOT / "models" / "hf")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.classify import Embedder  # noqa: E402
from src.detect import crop_box, load_image  # noqa: E402
from src.species import LIST_PATH, load_species  # noqa: E402

species = load_species()
names = list(species)
embedder = Embedder("bioclip")
digest = hashlib.sha256(LIST_PATH.read_bytes()).hexdigest()[:16]
cache = ROOT / "data" / "cache" / f"bioclip_species_{digest}.npy"
if cache.exists():
    text = np.load(cache)
else:
    text = embedder.text_embeddings_for([(n, species[n]["scientific"]) for n in names])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, text)

items = json.loads(Path(sys.argv[1]).read_text())
out = {}
for item in items:
    image, _ = load_image(item["path"])
    crops = [crop_box(image, b) for b in item["boxes"]]
    vectors = embedder.embed(crops)
    logits = 100 * vectors @ text.T
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    results = []
    for row in probabilities:
        best = np.argsort(-row)[:3]
        results.append({"species": names[best[0]], "score": float(row[best[0]]),
                        "top": [[names[i], float(row[i])] for i in best]})
    out[item["path"]] = results
Path(sys.argv[2]).write_text(json.dumps(out))
