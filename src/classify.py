"""Feature 2: species identification on detector crops.

SpeciesNet is not imported here. It must run in its own process (see
src/scripts/run_speciesnet.py), so this file maps its labels to CCT classes and calls that worker.
"""
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

CLASSES = ["badger", "bird", "bobcat", "cat", "coyote", "deer", "dog", "fox", "opossum",
           "rabbit", "raccoon", "rodent", "skunk", "squirrel"]

# (common name, scientific name) used for the BioCLIP text prompts.
PROMPT_NAMES = {
    "badger": ("American badger", "Taxidea taxus"),
    "bird": ("bird", "Aves"),
    "bobcat": ("bobcat", "Lynx rufus"),
    "cat": ("domestic cat", "Felis catus"),
    "coyote": ("coyote", "Canis latrans"),
    "deer": ("deer", "Odocoileus hemionus"),
    "dog": ("domestic dog", "Canis familiaris"),
    "fox": ("gray fox", "Urocyon cinereoargenteus"),
    "opossum": ("Virginia opossum", "Didelphis virginiana"),
    "rabbit": ("cottontail rabbit", "Sylvilagus audubonii"),
    "raccoon": ("raccoon", "Procyon lotor"),
    "rodent": ("mouse or rat", "Rodentia"),
    "skunk": ("striped skunk", "Mephitis mephitis"),
    "squirrel": ("ground squirrel", "Otospermophilus beecheyi"),
}


SECOND_SPECIES_SHARE = 0.5  # keeps a real second species in a mixed burst


def map_speciesnet_label(label):
    """Map 'uuid;class;order;family;genus;species;common' to a CCT class, or None."""
    fields = (label or "").split(";")
    if len(fields) < 7:
        return None
    _, cls, order, family, genus, species, _ = fields[:7]
    if cls == "aves":
        return "bird"
    if family == "felidae":
        return {"lynx": "bobcat", "felis": "cat", "puma": "mountain_lion"}.get(genus)
    if family == "canidae":
        if genus == "canis":
            return {"latrans": "coyote", "familiaris": "dog"}.get(species)
        return "fox" if genus in ("vulpes", "urocyon") else None
    if family == "sciuridae":
        return "squirrel"
    if order == "rodentia":
        return "rodent"
    if order == "lagomorpha":
        return "rabbit"
    families = {"procyonidae": "raccoon", "didelphidae": "opossum", "mephitidae": "skunk",
                "cervidae": "deer"}
    if family in families:
        return families[family] if family != "procyonidae" or genus == "procyon" else None
    if genus == "taxidea":
        return "badger"
    return None


def pick_species(classes, scores):
    """The val-selected rule: first of SpeciesNet's raw top guesses that maps to a known class."""
    for label, score in zip(classes, scores):
        mapped = map_speciesnet_label(label)
        if mapped:
            return mapped, score
    return "unknown", None


def burst_vote(frames, min_confidence):
    """Species for a burst, ranked by summed detector confidence. frames: detections per
    image. Returns a list, best first. Only animal boxes with a known species vote, and a
    species needs at least SECOND_SPECIES_SHARE of the winner's votes to stay in the list."""
    votes = Counter()
    for detections in frames:
        for d in detections:
            if d["category"] == "animal" and d["confidence"] >= min_confidence and d.get("species") not in (None, "unknown"):
                votes[d["species"]] += d["confidence"]
    ranked = votes.most_common()
    return [species for species, v in ranked if v >= SECOND_SPECIES_SHARE * ranked[0][1]]


def apply_burst_vote(frames, species):
    """Replace minor or unknown guesses on animal boxes with the burst's winning species."""
    if not species:
        return frames
    return [[{**d, "species": species[0]} if d["category"] == "animal" and d.get("species") not in species else d
             for d in detections] for detections in frames]


def run_worker(script, items, root, extra=(), timeout=1800):
    """Run a worker process. Returns (result dict or None, error text or None)."""
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder) / "in.json", Path(folder) / "out.json"
        source.write_text(json.dumps(items))
        try:
            done = subprocess.run([sys.executable, str(Path(root) / "src" / "scripts" / script), str(source), str(target), *extra],
                                  capture_output=True, text=True, timeout=timeout)
            if done.returncode != 0:
                return None, done.stderr[-800:]
            return json.loads(target.read_text()), None
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return None, f"{type(e).__name__}: {e}"


def classify_boxes(items, root, timeout=1800):
    """Run the SpeciesNet worker with the CCT rule. On any failure every box comes back
    unknown, so the pipeline keeps going. Returns ({path: [{"species", "score"}]}, error or None)."""
    unknown = {i["path"]: [{"species": "unknown", "score": None} for _ in i["boxes"]] for i in items}
    if not items:
        return unknown, None
    results, error = run_worker("speciesnet_worker.py", items, root, timeout=timeout)
    return (unknown, error) if error else (results, None)


def parse_label(label):
    """(common name, taxonomy list, vague) for a raw SpeciesNet label. A label is vague
    unless it names a species, so "mammal", "felidae family" and "blank" are all vague."""
    fields = (label or "").split(";")
    if len(fields) < 7:
        return "", [], True
    common = "person" if fields[5] == "sapiens" else fields[6]
    return common, fields[1:6], not fields[5]


def needs_fallback(classes, scores, cfg):
    if not classes:
        return True
    return parse_label(classes[0])[2] or scores[0] < cfg["speciesnet_min_score"]


def global_decision(classes, scores, bioclip, cfg):
    """Upload rule: SpeciesNet's own label, else BioCLIP, else a labeled best guess, else
    unknown. Never a CCT class. Returns (species, score, model). A guess has a model name
    ending in "guess". docs/reports/global_guess.json shows why SpeciesNet's guess comes first."""
    if not needs_fallback(classes, scores, cfg):
        return parse_label(classes[0])[0], scores[0], "SpeciesNet"
    if bioclip and bioclip["score"] >= cfg["bioclip_min_score"]:
        return bioclip["species"], bioclip["score"], "BioCLIP"
    if cfg.get("show_best_guess"):
        if classes and not parse_label(classes[0])[2]:
            return parse_label(classes[0])[0], scores[0], "SpeciesNet guess"
        if bioclip:
            return bioclip["species"], bioclip["score"], "BioCLIP guess"
    return "unknown", None, None


def classify_global(items, root, cfg, timeout=1800):
    """Name boxes anywhere in the world. Returns ({path: [{"species", "score", "model"}]}, error or None)."""
    settings = cfg["global_identification"]
    results = {i["path"]: [{"species": "unknown", "score": None, "model": None} for _ in i["boxes"]] for i in items}
    if not items:
        return results, None
    raw, error = run_worker("speciesnet_worker.py", items, root, extra=("global",), timeout=timeout)
    raw = raw or {i["path"]: [{"classes": [], "scores": []} for _ in i["boxes"]] for i in items}
    doubtful = []
    for item in items:
        boxes = [b for b, r in zip(item["boxes"], raw[item["path"]]) if needs_fallback(r["classes"], r["scores"], settings)]
        if boxes:
            doubtful.append({"path": item["path"], "boxes": boxes})
    named, bioclip_error = run_worker("bioclip_worker.py", doubtful, root, timeout=timeout) if doubtful else ({}, None)
    named = named or {}
    for item in items:
        queue = list(named.get(item["path"], []))
        for i, r in enumerate(raw[item["path"]]):
            fallback = queue.pop(0) if needs_fallback(r["classes"], r["scores"], settings) and queue else None
            species, score, model = global_decision(r["classes"], r["scores"], fallback, settings)
            results[item["path"]][i] = {"species": species, "score": score, "model": model}
    return results, error or bioclip_error


class Embedder:
    """Unit-length image embeddings from BioCLIP or DINOv2."""

    def __init__(self, name):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name
        if name == "bioclip":
            import open_clip
            self.model, _, self.transform = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            self.tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
        elif name == "dinov2":
            import timm
            self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
            self.transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.model))
        else:
            raise ValueError(name)
        self.model = self.model.to(self.device).eval()

    def embed(self, crops, batch_size=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = self.torch.stack([self.transform(c) for c in crops[i:i + batch_size]]).to(self.device)
                features = self.model.encode_image(batch) if self.name == "bioclip" else self.model(batch)
                out.append((features / features.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 0))

    def text_embeddings_for(self, names, batch_size=256):
        """One unit vector per (common, scientific) name, from both prompts."""
        rows = []
        with self.torch.no_grad():
            for i in range(0, len(names), batch_size):
                part = names[i:i + batch_size]
                prompts = [f"a photo of a {common}" for common, _ in part] + [f"a photo of {sci}" for _, sci in part]
                t = self.model.encode_text(self.tokenizer(prompts).to(self.device))
                t = t / t.norm(dim=-1, keepdim=True)
                t = t[:len(part)] + t[len(part):]
                rows.append((t / t.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(rows)

    def text_embeddings(self):
        """One unit vector per class, averaged over common and scientific name prompts."""
        rows = []
        with self.torch.no_grad():
            for cls in CLASSES:
                prompts = [f"a photo of {n}" for n in PROMPT_NAMES[cls]]
                t = self.model.encode_text(self.tokenizer(prompts).to(self.device))
                t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
                rows.append((t / t.norm()).cpu().numpy())
        return np.stack(rows)


def zero_shot_predict(image_embeddings, text_embeddings):
    """Returns (class names, scores) for BioCLIP zero-shot."""
    logits = 100 * image_embeddings @ text_embeddings.T
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    best = probs.argmax(1)
    return [CLASSES[i] for i in best], probs.max(1)
