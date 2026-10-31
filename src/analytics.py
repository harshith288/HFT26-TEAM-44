"""Shared analytics. Every page and the Ranger's Brief get their numbers from here, so no two
screens can disagree. Uses pandas, so never import this where torch is loaded."""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
UNMAPPED = "Unmapped"
NOT_A_SPECIES = ["vehicle", "person", "unknown"]
ACCURACY_TARGET = 90.0
CAMERA_VILLAGE = {c: name for name, v in CONFIG["villages"].items() for c in v["cameras"]}


def village_of(cameras):
    return cameras.map(CAMERA_VILLAGE).fillna(UNMAPPED)


def in_dates(frame, start, end):
    days = frame["seen_at"].fillna("").str[:10]
    return frame[(days >= start) & (days <= end)]


def apply_filters(frames, filters):
    """filters: optional dates (start, end as YYYY-MM-DD), locations, species, min_confidence.
    All of them are applied to every table, so one filter means the same thing on each page."""
    images, bursts, alerts = frames["images"], frames["bursts"], frames["alerts"]
    if filters.get("dates"):
        start, end = filters["dates"]
        images, bursts, alerts = in_dates(images, start, end), in_dates(bursts, start, end), in_dates(alerts, start, end)
    if filters.get("locations"):
        chosen = list(filters["locations"])
        images = images[village_of(images["camera_id"]).isin(chosen)]
        bursts = bursts[village_of(bursts["camera_id"]).isin(chosen)]
        alerts = alerts[village_of(alerts["camera_id"]).isin(chosen)]
    if filters.get("species"):
        chosen = list(filters["species"])
        bursts = bursts[bursts["species"].isin(chosen)]
        alerts = alerts[alerts["species"].isin(chosen)]
        images = images[images["seq_id"].isin(bursts["burst_id"])]
    if filters.get("min_confidence"):
        floor = filters["min_confidence"]
        images = images[images["top_confidence"] >= floor]
        bursts = bursts[bursts["confidence"] >= floor]
        alerts = alerts[alerts["confidence"] >= floor]
    detections = frames["detections"]
    return {**frames, "images": images, "bursts": bursts, "alerts": alerts,
            "detections": detections[detections["image_id"].isin(images["image_id"])]}


def filter_options(frames):
    """Choices for the sidebar, from the unfiltered data."""
    days = pd.concat([frames["images"]["seen_at"], frames["bursts"]["seen_at"]]).dropna().astype(str).str[:10]
    days = days[days.str.fullmatch(r"\d{4}-\d{2}-\d{2}")]
    species = frames["bursts"]["species"]
    return {"locations": sorted(CONFIG["villages"]) + [UNMAPPED], "species": sorted(species[species != "vehicle"].unique()),
            "first_day": days.min() if len(days) else None, "last_day": days.max() if len(days) else None}


def overview(frames):
    readable = frames["images"][frames["images"]["status"] == "ok"]
    tiers = readable.groupby("triage")["seq_id"].nunique()
    alerts = frames["alerts"]
    return {"images": len(readable), "bursts": int(readable["seq_id"].nunique()),
            "discard": int(tiers.get("discard", 0)), "review": int(tiers.get("review", 0)), "keep": int(tiers.get("keep", 0)),
            "serious_alerts": int(alerts["level"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum()),
            "open_alerts": int(((alerts["level"] != "LOW") & (alerts["status"] == "new")).sum())}


def review_bursts(frames):
    """Best frame of each burst waiting for a person, most likely animals first."""
    images = frames["images"]
    review = images[(images["status"] == "ok") & (images["triage"] == "review")]
    if review.empty:
        return review
    best = review.sort_values("top_confidence", ascending=False).groupby("seq_id", as_index=False).first()
    return best.sort_values("top_confidence", ascending=False)


def species_counts(frames):
    bursts = frames["bursts"]
    counts = bursts[bursts["species"] != "vehicle"].groupby("species").size()
    return counts.sort_values().reset_index(name="bursts")


def monthly_animals(frames):
    bursts = frames["bursts"]
    dated = bursts[bursts["species"] != "vehicle"].dropna(subset=["seen_at"]).copy()
    dated["month"] = dated["seen_at"].str[:7]
    return dated.groupby("month")["min_observed_count"].sum().reset_index()


def day_night(frames):
    animals = frames["bursts"]
    animals = animals[~animals["species"].isin(NOT_A_SPECIES)].copy()
    animals["period"] = animals["is_night"].map({1: "Night", 0: "Day"})
    return animals.groupby(["species", "period"]).size().reset_index(name="bursts")


def camera_table(frames):
    readable = frames["images"][frames["images"]["status"] == "ok"]
    if readable.empty:
        return readable
    stats = readable.groupby("camera_id").agg(images=("image_id", "size"), removed=("is_empty", "sum"),
                                              night=("is_night", "sum")).reset_index()
    stats["bursts"] = stats["camera_id"].map(frames["bursts"].groupby("camera_id")["burst_id"].nunique()).fillna(0).astype(int)
    stats["alerts"] = stats["camera_id"].map(frames["alerts"].groupby("camera_id").size()).fillna(0).astype(int)
    stats["village"] = village_of(stats["camera_id"])
    stats["zone factor"] = stats["village"].map(lambda v: CONFIG["villages"].get(v, {}).get("zone_factor", CONFIG["default_zone_factor"]))
    stats["% removed"] = (100 * stats["removed"] / stats["images"]).round(1)
    return stats.sort_values("alerts", ascending=False)


def species_table(frames):
    """One record per species, most active first."""
    bursts = frames["bursts"]
    animals = bursts[~bursts["species"].isin(NOT_A_SPECIES)].copy()
    animals["place"] = village_of(animals["camera_id"])
    rows = []
    for species, group in animals.groupby("species"):
        places = group["place"].value_counts()
        rows.append({"species": species, "bursts": len(group), "animals": int(group["min_observed_count"].sum()),
                     "night_pct": round(100 * group["is_night"].fillna(0).mean(), 1), "top_place": places.index[0],
                     "top_place_bursts": int(places.iloc[0]), "last_seen": str(group["seen_at"].dropna().max())[:10],
                     "alerts": int((frames["alerts"]["species"] == species).sum())})
    return sorted(rows, key=lambda r: (-r["bursts"], r["species"]))


def uncertainty(frames):
    boxes = frames["detections"]
    boxes = boxes[boxes["category"] == "animal"]
    classifier = boxes["classifier"].fillna("").astype(str)
    return {"bursts_needing_review": overview(frames)["review"], "low_confidence_guess_boxes": int(classifier.str.endswith(" guess").sum()),
            "unknown_animal_boxes": int((boxes["species"] == "unknown").sum()), "animal_boxes": len(boxes)}


def notable_change(frames):
    """The biggest rise or fall per species between the two latest months with data."""
    bursts = frames["bursts"]
    dated = bursts[~bursts["species"].isin(NOT_A_SPECIES)].dropna(subset=["seen_at"]).copy()
    dated["month"] = dated["seen_at"].str[:7]
    months = sorted(dated["month"].unique())
    if len(months) < 2:
        return None
    before, latest = months[-2:]
    table = dated[dated["month"].isin([before, latest])].groupby(["species", "month"]).size().unstack(fill_value=0)
    change = (table[latest] - table[before]).sort_values(key=abs, ascending=False)
    return {"species": change.index[0], "earlier_month": before, "latest_month": latest,
            "earlier_bursts": int(table[before][change.index[0]]), "latest_bursts": int(table[latest][change.index[0]])}


def test_accuracy():
    path = ROOT / "docs" / "reports" / "final_eval.json"
    if not path.exists():
        return None
    return round(json.loads(path.read_text())["groups"]["all"]["species"]["accuracy"], 1)


def brief_facts(frames):
    """Everything the LLM may see. Aggregated numbers only: no images, paths or ids."""
    table, totals = species_table(frames), overview(frames)
    seen = frames["bursts"]["seen_at"].dropna().astype(str).str[:10]
    return {"images": totals["images"], "bursts": totals["bursts"], "species_seen": len(table), "open_alerts": totals["open_alerts"],
            "first_day": seen.min() if len(seen) else None, "last_day": seen.max() if len(seen) else None,
            "uploaded_bursts": int((frames["bursts"]["source"] == "incoming").sum()), "species": table,
            "notable_change": notable_change(frames), "uncertain": uncertainty(frames),
            "test_accuracy_pct": test_accuracy(), "accuracy_target_pct": ACCURACY_TARGET}
