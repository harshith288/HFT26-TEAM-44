"""Feature 1: MegaDetector v6 detection and empty-frame filtering."""
import os
from pathlib import Path

from PIL import Image, ImageOps

# ultralytics must never pip-install packages in the middle of a run.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# MDv6 class ids, as printed by the model in Step 0.
CATEGORIES = {0: "animal", 1: "person", 2: "vehicle"}


def load_image(path):
    """Open any image as upright RGB. Returns (image, None) or (None, error text)."""
    try:
        with Image.open(path) as im:
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB"), None
    except Exception as e:  # a bad file can raise almost anything, and must never stop the pipeline
        return None, f"{type(e).__name__}: {e}"


class Detector:
    def __init__(self, cfg, root="."):
        from ultralytics import YOLO
        d = cfg["detection"]
        self.model = YOLO(str(Path(root) / d["weights"]))
        self.image_size = d["image_size"]
        self.iou = d["nms_iou"]
        self.raw_confidence = d["raw_confidence"]

    def detect_image(self, image):
        """Run on one PIL image. Boxes are normalized x, y, w, h (top-left corner)."""
        result = self.model.predict(image, imgsz=self.image_size, conf=self.raw_confidence,
                                    iou=self.iou, verbose=False, batch=1)[0]
        detections = []
        for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                  result.boxes.xywhn.tolist()):
            cx, cy, w, h = box
            detections.append({"category": CATEGORIES[int(cls)], "confidence": conf,
                               "x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h})
        return detections

    def detect_file(self, path):
        """Never raises for a bad file: unreadable files come back with status 'unreadable'."""
        image, error = load_image(path)
        if image is None:
            return {"status": "unreadable", "error": error, "detections": []}
        return {"status": "ok", "error": None, "width": image.width, "height": image.height,
                "detections": self.detect_image(image)}


def filter_detections(detections, threshold):
    return [d for d in detections if d["confidence"] >= threshold]


def is_empty(detections, threshold):
    """Empty means no animal, person or vehicle at or above the threshold."""
    return not filter_detections(detections, threshold)


def crop_box(image, detection):
    """Cut a detection out of a PIL image, keeping at least 1 pixel in each direction."""
    left = max(0, int(detection["x"] * image.width))
    top = max(0, int(detection["y"] * image.height))
    right = min(image.width, max(left + 1, int((detection["x"] + detection["w"]) * image.width)))
    bottom = min(image.height, max(top + 1, int((detection["y"] + detection["h"]) * image.height)))
    return image.crop((left, top, right, bottom))
