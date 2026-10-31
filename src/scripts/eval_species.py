"""Feature 2 on VAL: compare species candidates and keep the best by macro-F1.

All candidates classify the same crop: the top-confidence MDv6 animal box in each image.
Images where the detector found no animal count as wrong for every candidate.
Test splits are not read here.

Usage: python src/scripts/eval_species.py
"""
import collections
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.classify import CLASSES, Embedder, map_speciesnet_label, zero_shot_predict  # noqa: E402
from src.detect import crop_box, load_image  # noqa: E402

SEED = 42
GAP_FLAG_POINTS = 10
MISSED = "no_detection"
np.random.seed(SEED)
torch.manual_seed(SEED)

cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
filter_thr = cfg["detection"]["confidence_threshold"]
count_thr = cfg["detection"]["count_threshold"]
cache = json.loads((ROOT / "data" / "cache" / "detections.json").read_text())
channel_diff = json.loads((ROOT / "data" / "cache" / "ir.json").read_text())
all_rows = list(csv.DictReader(open(ROOT / cfg["paths"]["manifest"])))


def top_animal(row, threshold):
    animals = [d for d in cache[row["image_id"]]["detections"]
               if d["category"] == "animal" and d["confidence"] >= threshold]
    return max(animals, key=lambda d: d["confidence"]) if animals else None


def usable(row):
    """Single-species animal images only: empty, car and mixed-species are excluded."""
    return row["labels"] in CLASSES


# Train crops use the stricter count threshold, so weak boxes do not pollute training.
train = [r for r in all_rows if r["split"] == "train" and usable(r) and top_animal(r, count_thr)]
val_all = [r for r in all_rows if r["split"] in ("cis_val", "trans_val")]
val = [r for r in val_all if usable(r)]
excluded = collections.Counter(r["labels"] for r in val_all if not usable(r))
val_crops = [r for r in val if top_animal(r, filter_thr)]
print(f"train crops {len(train)} | val images {len(val)} | val crops {len(val_crops)} | excluded {dict(excluded)}")


def embed(name, rows, threshold):
    path = ROOT / "data" / "cache" / f"embeddings_{name}_{rows[0]['split']}.npy"
    if path.exists():
        return np.load(path), None
    embedder = EMBEDDERS[name]
    crops = []
    for r in rows:
        image, _ = load_image(ROOT / cfg["paths"]["images"] / r["file_name"])
        crops.append(crop_box(image, top_animal(r, threshold)))
    start = time.perf_counter()
    result = embedder.embed(crops)
    ms = (time.perf_counter() - start) / len(crops) * 1000
    np.save(path, result)
    return result, ms


EMBEDDERS = {n: Embedder(n) for n in ("bioclip", "dinov2")}
features, speed = {}, {}
for name in EMBEDDERS:
    features[name, "train"], _ = embed(name, train, count_thr)
    features[name, "val"], speed[name] = embed(name, val_crops, filter_thr)
text = EMBEDDERS["bioclip"].text_embeddings()
del EMBEDDERS

y_train = np.array([r["labels"] for r in train])
groups = np.array([r["location"] for r in train])
print("train class counts:", collections.Counter(y_train.tolist()).most_common())


def fit_probe(name):
    """Choose C by cross-validation across cameras, using the train split only."""
    X = features[name, "train"]
    scores = {}
    for C in (0.1, 1, 10, 100, 1000):
        fold_scores = []
        for fit, hold in GroupKFold(n_splits=5).split(X, y_train, groups):
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED)
            clf.fit(X[fit], y_train[fit])
            fold_scores.append(f1_score(y_train[hold], clf.predict(X[hold]), average="macro", zero_division=0))
        scores[C] = float(np.mean(fold_scores))
    best = max(scores, key=scores.get)
    clf = LogisticRegression(C=best, max_iter=2000, class_weight="balanced", random_state=SEED).fit(X, y_train)
    return clf, best, scores


probes, probe_info = {}, {}
for name in ("bioclip", "dinov2"):
    probes[name], best_c, cv = fit_probe(name)
    probe_info[name] = {"best_C": best_c, "train_cv_macro_f1_by_C": cv}
    joblib.dump(probes[name], ROOT / "models" / f"linear_probe_{name}.joblib")

sn = json.loads((ROOT / "data" / "cache" / "speciesnet.json").read_text())


def sn_processed(image_id):
    return map_speciesnet_label(sn[image_id]["processed"]) or "unknown" if image_id in sn else "unknown"


def sn_top_mapped(image_id):
    """First of the raw top guesses that maps to a CCT class; ignores rollup and geofence."""
    for label in sn.get(image_id, {}).get("raw_classes", []):
        mapped = map_speciesnet_label(label)
        if mapped:
            return mapped
    return "unknown"


crop_ids = [r["image_id"] for r in val_crops]
zero_shot, _ = zero_shot_predict(features["bioclip", "val"], text)
candidate_crop_preds = {
    "speciesnet_processed": [sn_processed(i) for i in crop_ids],
    "speciesnet_top_mapped": [sn_top_mapped(i) for i in crop_ids],
    "bioclip_zero_shot": zero_shot,
    "probe_bioclip": probes["bioclip"].predict(features["bioclip", "val"]).tolist(),
    "probe_dinov2": probes["dinov2"].predict(features["dinov2", "val"]).tolist(),
}


def full_predictions(crop_preds):
    by_id = dict(zip(crop_ids, crop_preds))
    return [by_id.get(r["image_id"], MISSED) for r in val]


y_val = [r["labels"] for r in val]
eval_labels = sorted(set(y_val))


def score(y_true, y_pred):
    if not y_true:
        return None
    labels = sorted(set(y_true))
    return {"n": len(y_true), "top1_accuracy": 100 * accuracy_score(y_true, y_pred),
            "macro_f1": 100 * f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)}


def pick(predictions, condition):
    keep = [i for i, r in enumerate(val) if condition(r)]
    return [y_val[i] for i in keep], [predictions[i] for i in keep]


is_night = lambda r: channel_diff[r["image_id"]] < cfg["activity"]["ir_channel_tolerance"]  # noqa: E731
GROUPS = {
    "all": lambda r: True, "day": lambda r: not is_night(r), "night": is_night,
    "cis_val": lambda r: r["split"] == "cis_val", "trans_val": lambda r: r["split"] == "trans_val",
    "confirmed_boxed": lambda r: int(r["n_boxes"]) > 0,
}
results = {}
for name, crop_preds in candidate_crop_preds.items():
    preds = full_predictions(crop_preds)
    results[name] = {g: score(*pick(preds, c)) for g, c in GROUPS.items()}

best = max(results, key=lambda n: results[n]["all"]["macro_f1"])
best_preds = full_predictions(candidate_crop_preds[best])

gaps = {}
for name in ("bioclip", "dinov2"):
    train_preds = probes[name].predict(features[name, "train"])
    train_acc = 100 * accuracy_score(y_train, train_preds)
    train_f1 = 100 * f1_score(y_train, train_preds, average="macro", zero_division=0)
    val_score = results[f"probe_{name}"]["all"]
    gaps[name] = {"train_accuracy": train_acc, "train_macro_f1": train_f1,
                  "val_accuracy": val_score["top1_accuracy"], "val_macro_f1": val_score["macro_f1"],
                  "accuracy_gap_points": train_acc - val_score["top1_accuracy"],
                  "macro_f1_gap_points": train_f1 - val_score["macro_f1"]}
    gaps[name]["overfit_flag"] = bool(gaps[name]["accuracy_gap_points"] > GAP_FLAG_POINTS
                                      or gaps[name]["macro_f1_gap_points"] > GAP_FLAG_POINTS)

precision, recall, f1, support = precision_recall_fscore_support(
    y_val, best_preds, labels=eval_labels, zero_division=0)
per_class = {c: {"precision": float(p), "recall": float(r), "f1": float(f), "n": int(s)}
             for c, p, r, f, s in zip(eval_labels, precision, recall, f1, support)}

# Confusion matrix: rows are true classes; cells show counts, shade shows the row share.
columns = eval_labels + sorted(set(best_preds) - set(eval_labels))
matrix = confusion_matrix(y_val, best_preds, labels=columns)[:len(eval_labels)]
shares = matrix / matrix.sum(1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(0.62 * len(columns) + 3, 0.5 * len(eval_labels) + 2.5))
ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
ax.set_yticks(range(len(eval_labels)), eval_labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True label")
ax.set_title(f"{best} on val: {len(y_val)} real images")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if matrix[i, j]:
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=9,
                    color="white" if shares[i, j] > 0.55 else "#1a1a1a")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "reports" / "confusion_matrix_val.png", dpi=150)

output = {
    "command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(),
    "n_real_images_val": len(val), "n_real_crops_train": len(train), "hardware": torch.cuda.get_device_name(0),
    "split": "val only (cis_val + trans_val)", "excluded_val_labels": dict(excluded),
    "val_images_without_detection": len(val) - len(val_crops), "classes_scored": eval_labels,
    "candidates": results, "best_by_val_macro_f1": best, "per_class_best": per_class, "probe_info": probe_info,
    "overfitting": gaps, "ms_per_crop_embedding": speed,
    "day_night_source": "IR signal (night = IR image)",
}
(ROOT / "docs" / "reports" / "phase2_species.json").write_text(json.dumps(output, indent=1, default=float))

print(f"\n{'candidate':<24}{'macroF1':>8}{'acc':>7} | {'day':>6}{'night':>7}{'cis':>7}{'trans':>7}{'boxed':>7}  (macro-F1)")
for name, r in sorted(results.items(), key=lambda kv: -kv[1]["all"]["macro_f1"]):
    row = "".join(f"{r[g]['macro_f1']:7.1f}" for g in ("day", "night", "cis_val", "trans_val", "confirmed_boxed"))
    print(f"{name:<24}{r['all']['macro_f1']:8.1f}{r['all']['top1_accuracy']:7.1f} |{row}")
print("\nBEST:", best)
print("group sizes:", {g: results[best][g]["n"] for g in GROUPS})
print("overfitting:", json.dumps(gaps, indent=1, default=float))
print("per class (best):")
for c, m in per_class.items():
    print(f"  {c:<10} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} n {m['n']}")
print("embedding ms/crop:", speed)
print("SPECIES_DONE")
