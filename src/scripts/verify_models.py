"""Step 0 check: run one real CCT image through every model and print real output.
Also measures MDv6 compact vs extra on real images at several batch sizes, which
is what the detector and batch-size choice is based on. Writes docs/reports/step0_verify.json.

Usage: python src/scripts/verify_models.py
"""
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["HF_HOME"] = str(ROOT / "models" / "hf")
os.environ["KAGGLEHUB_CACHE"] = str(ROOT / "models" / "kaggle")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

SEED = 42
torch.manual_seed(SEED)
IMAGES = ROOT / "data" / "sample" / "images"
results = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(),
           "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()}


def pick_images():
    """A real, single-species, boxed bobcat from train, plus 16 other real images for timing."""
    rows = list(csv.DictReader(open(ROOT / "data" / "sample" / "manifest.csv")))
    target = next(r for r in rows if r["split"] == "train" and r["labels"] == "bobcat" and r["n_boxes"] == "1")
    return target, [str(IMAGES / r["file_name"]) for r in rows[:16]]


def largest_box(image_path, model):
    res = model.predict(image_path, imgsz=1280, conf=0.2, verbose=False, device=0)[0]
    if len(res.boxes) == 0:
        return res, None
    i = int(res.boxes.conf.argmax())
    return res, [float(v) for v in res.boxes.xyxy[i]]


def run_megadetector(target_path, timing_paths):
    from ultralytics import YOLO
    out, keep, box = {}, None, None
    for name in ["MDV6-yolov9-c.pt", "MDV6-yolov9-e-1280.pt"]:
        model = YOLO(str(ROOT / "models" / name))
        res, b = largest_box(target_path, model)
        dets = [(model.names[int(c)], round(float(s), 3)) for c, s in zip(res.boxes.cls, res.boxes.conf)]
        print(f"{name}: classes={model.names} detections={dets} box={b}")
        entry = {"classes": model.names, "detections": dets, "batch": {}}
        # Extra at batch 8 was measured to OOM on the 6 GB RTX 4050, so it is not re-run.
        for batch in [1, 2, 4] if "yolov9-e" in name else [1, 2, 4, 8]:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for i in range(0, len(timing_paths), batch):
                    model.predict(timing_paths[i:i + batch], imgsz=1280, conf=0.05, verbose=False,
                                  device=0, batch=batch)
                ms = (time.perf_counter() - start) / len(timing_paths) * 1000
                peak = torch.cuda.max_memory_allocated() / 1e9
                entry["batch"][batch] = {"ms_per_image": round(ms, 1), "peak_vram_gb": round(peak, 2)}
                print(f"  batch {batch}: {ms:.1f} ms/image, peak VRAM {peak:.2f} GB ({len(timing_paths)} real images)")
            except torch.cuda.OutOfMemoryError:
                entry["batch"][batch] = "OOM"
                print(f"  batch {batch}: OOM")
                torch.cuda.empty_cache()
        out[name] = entry
        if name.startswith("MDV6-yolov9-e"):
            box = b
        del model
        torch.cuda.empty_cache()
    results["megadetector"] = out
    return box


def run_bioclip(crop):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model = model.cuda().eval()
    names = ["a photo of a bobcat", "a photo of a coyote", "a photo of a raccoon", "a photo of a domestic cat",
             "a photo of an opossum", "a photo of a deer"]
    with torch.no_grad():
        image = model.encode_image(preprocess(crop).unsqueeze(0).cuda())
        text = model.encode_text(tokenizer(names).cuda())
        image = image / image.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        probs = (100 * image @ text.T).softmax(-1)[0].cpu().tolist()
    top = sorted(zip(names, probs), key=lambda t: -t[1])
    print("BioCLIP:", [(n, round(p, 3)) for n, p in top])
    results["bioclip"] = [(n, round(p, 3)) for n, p in top]
    del model
    torch.cuda.empty_cache()


def run_dinov2(crop):
    import timm
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).cuda().eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    with torch.no_grad():
        emb = model(transform(crop).unsqueeze(0).cuda())[0].cpu()
    print(f"DINOv2: embedding shape {tuple(emb.shape)}, norm {emb.norm():.3f}, first3 {emb[:3].tolist()}")
    results["dinov2"] = {"shape": list(emb.shape), "norm": round(float(emb.norm()), 3)}
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    target, timing_paths = pick_images()
    target_path = str(IMAGES / target["file_name"])
    print("Real test image:", target["file_name"], "| ground truth:", target["labels"],
          "| split:", target["split"], "| captured:", target["date_captured"])
    results["image"] = {k: target[k] for k in ("file_name", "labels", "split")}
    results["n_timing_images"] = len(timing_paths)

    box = run_megadetector(target_path, timing_paths)
    image = Image.open(target_path).convert("RGB")
    if box is None:
        print("WARNING: extra model found no box on this image; using the full frame for the embedders")
        crop = image
    else:
        crop = image.crop(box)
    # SpeciesNet runs separately: see src/scripts/verify_speciesnet.py
    for step, arg in ((run_bioclip, crop), (run_dinov2, crop)):
        try:
            step(arg)
        except Exception as e:
            print(f"FAILED {step.__name__}: {type(e).__name__}: {e}")
            results[step.__name__] = f"FAILED: {e}"
    (ROOT / "docs" / "reports").mkdir(exist_ok=True)
    (ROOT / "docs" / "reports" / "step0_verify.json").write_text(json.dumps(results, indent=1, default=str))
    print("VERIFY_DONE")
