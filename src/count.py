"""Feature 3: animal count per burst.

The count for a burst is the MAX in any single frame, never a sum, because the same
animal usually appears in several frames. It is a "minimum observed count": more
animals may have been present out of view.
"""
from collections import Counter

MIN_COUNT_LABEL = "minimum observed count"


def detection_label(detection):
    """Species for animals; 'person' or 'vehicle' otherwise. Unclassified animals are 'unknown'."""
    if detection["category"] == "animal":
        return detection.get("species") or "unknown"
    return detection["category"]


def burst_counts(frames):
    """frames: one list of kept detections per image. Returns {label: max in any one frame}."""
    best = Counter()
    for detections in frames:
        for label, n in Counter(detection_label(d) for d in detections).items():
            best[label] = max(best[label], n)
    return dict(best)


def burst_animal_total(frames):
    """Most animals of any species visible in a single frame of the burst."""
    return max((sum(d["category"] == "animal" for d in f) for f in frames), default=0)


def burst_animal_sum(frames):
    """Sum across frames. Kept ONLY as a baseline to show why summing is wrong."""
    return sum(sum(d["category"] == "animal" for d in f) for f in frames)
