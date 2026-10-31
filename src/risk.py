"""Feature 7: intrusion risk. danger = species_weight x count x night_factor x zone_factor."""
from src.species import group_for

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def species_weight(label, cfg):
    """An exact CCT label first, then the danger group from taxonomy, then unknown."""
    weights = cfg["species_weights"]
    if label in weights:
        return weights[label]
    group = group_for(label, cfg)
    return cfg["danger_groups"][group]["weight"] if group else weights["unknown"]


def village_for(camera_id, cfg):
    for name, village in cfg["villages"].items():
        if str(camera_id) in village["cameras"]:
            return name
    return None


def zone_factor(camera_id, cfg):
    village = village_for(camera_id, cfg)
    return cfg["villages"][village]["zone_factor"] if village else cfg["default_zone_factor"]


def danger_score(label, count, night, camera_id, cfg, guess=False):
    """If night is unknown (None) the higher night factor is used: a false alarm beats a miss.
    A low-confidence guess uses the unknown weight, so it cannot raise or lower the risk."""
    night_factor = 1.0 if night is False else cfg["night_factor"]
    weight = cfg["species_weights"]["unknown"] if guess else species_weight(label, cfg)
    return round(weight * max(count, 0) * night_factor * zone_factor(camera_id, cfg), 4)


def level_for(score, cfg):
    r = cfg["risk_levels"]
    if score >= r["critical_min"]:
        return "CRITICAL"
    if score >= r["high_min"]:
        return "HIGH"
    return "MEDIUM" if score >= r["medium_min"] else "LOW"


def assess(label, count, night, camera_id, cfg, guess=False):
    """Full alert decision for one species in one burst."""
    score = danger_score(label, count, night, camera_id, cfg, guess)
    level = level_for(score, cfg)
    poacher = label == "person" and count > 0
    if poacher and LEVELS.index(level) < LEVELS.index(cfg["person_min_level"]):
        level = cfg["person_min_level"]
    response = cfg["response"][level]
    return {"species": label, "min_observed_count": count, "danger_score": score, "level": level,
            "rangers": response["rangers"], "vehicle": response["vehicle"], "action": response["action"],
            "possible_poacher": poacher, "village": village_for(camera_id, cfg), "camera_id": camera_id, "is_guess": guess}
