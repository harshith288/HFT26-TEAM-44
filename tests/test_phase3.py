"""Phase 3: risk scoring, activity, the full pipeline, and regression.

Pipeline tests use the real detector and the real SpeciesNet worker on real CCT
photos. Only the all-black and all-white images are synthetic, as those cases require.
The classifier-failure test makes the real worker fail; nothing is mocked.
"""
import csv
import itertools
import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.scripts import regression
from src.activity import corrected_hour, is_dark_scene, is_ir_image, is_night, period
from src.classify import apply_burst_vote, burst_vote, classify_boxes, pick_species
from src.detect import Detector, crop_box
from src.pipeline import Pipeline, incoming_items, triage
from src.risk import assess, danger_score, level_for, species_weight, village_for, zone_factor

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
IMAGES = ROOT / CONFIG["paths"]["images"]
ROWS = list(csv.DictReader(open(ROOT / CONFIG["paths"]["manifest"])))
TRAIN = [r for r in ROWS if r["split"] == "train"]
SPECIES = list(CONFIG["species_weights"]) + ["badger", "fox", "dragon"]
CAMERAS = ["33", "61", "125", "0", "78", "not-a-camera", None]


def real(label):
    row = next(r for r in TRAIN if r["labels"] == label and r["n_boxes"] == "1")
    return IMAGES / row["file_name"]


# --- risk: the four required properties ---

@pytest.mark.parametrize("label,count,night,camera", [("coyote", 3, True, "33"), ("person", 1, False, None)])
def test_same_input_gives_same_output(label, count, night, camera):
    assert assess(label, count, night, camera, CONFIG) == assess(label, count, night, camera, CONFIG)


def test_night_is_never_below_day():
    for label, count, camera in itertools.product(SPECIES, range(0, 13), CAMERAS):
        assert danger_score(label, count, True, camera, CONFIG) >= danger_score(label, count, False, camera, CONFIG)


def test_more_animals_never_lowers_the_score():
    for label, night, camera in itertools.product(SPECIES, (True, False), CAMERAS):
        scores = [danger_score(label, n, night, camera, CONFIG) for n in range(0, 15)]
        assert scores == sorted(scores)


@pytest.mark.parametrize("label", ["dragon", "", "unknown", None])
def test_unknown_species_uses_safe_default(label):
    alert = assess(label, 2, False, "33", CONFIG)
    assert alert["danger_score"] == pytest.approx(0.5 * 2 * 1.0 * 1.5)
    assert alert["level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# --- risk: formula and rules ---

def test_formula_uses_all_four_factors():
    assert danger_score("bobcat", 2, True, "33", CONFIG) == pytest.approx(0.7 * 2 * 1.5 * 1.5)
    assert danger_score("bobcat", 2, False, "78", CONFIG) == pytest.approx(0.7 * 2 * 1.0 * 0.8)


def test_level_boundaries_are_inclusive():
    r = CONFIG["risk_levels"]
    assert level_for(r["medium_min"] - 0.001, CONFIG) == "LOW"
    assert level_for(r["medium_min"], CONFIG) == "MEDIUM"
    assert level_for(r["high_min"], CONFIG) == "HIGH"
    assert level_for(r["critical_min"], CONFIG) == "CRITICAL"


@pytest.mark.parametrize("night,camera", [(False, "78"), (True, "33")])
def test_person_is_always_a_poacher_alert_at_least_high(night, camera):
    alert = assess("person", 1, night, camera, CONFIG)
    assert alert["possible_poacher"] is True
    assert alert["level"] in ("HIGH", "CRITICAL")


def test_recommendation_follows_the_level():
    for score in (0.1, 0.6, 1.5, 5.0):
        level = level_for(score, CONFIG)
        response = CONFIG["response"][level]
        assert response["rangers"] >= 0 and isinstance(response["vehicle"], bool)
    alert = assess("coyote", 4, True, "33", CONFIG)
    assert alert["level"] == "CRITICAL"
    assert (alert["rangers"], alert["vehicle"]) == (CONFIG["response"]["CRITICAL"]["rangers"], True)


def test_vehicle_and_zero_count_score_zero():
    assert danger_score("vehicle", 3, True, "33", CONFIG) == 0
    assert danger_score("coyote", 0, True, "33", CONFIG) == 0
    assert danger_score("coyote", -1, True, "33", CONFIG) == 0


def test_unknown_time_uses_night_factor():
    assert danger_score("coyote", 1, None, "33", CONFIG) == danger_score("coyote", 1, True, "33", CONFIG)


def test_unmapped_camera_is_safe():
    assert village_for("not-a-camera", CONFIG) is None
    assert zone_factor(None, CONFIG) == CONFIG["default_zone_factor"]
    assert species_weight("nothing", CONFIG) == CONFIG["species_weights"]["unknown"]


# --- triage and the low-confidence count rule ---

TRIAGE = {"keep_min": 0.2, "discard_below": 0.01}


def test_triage_boundaries():
    assert triage(0.2, TRIAGE) == "keep"
    assert triage(0.19, TRIAGE) == "review"
    assert triage(0.01, TRIAGE) == "review"
    assert triage(0.009, TRIAGE) == "discard"
    assert triage(0.0, TRIAGE) == "discard"


def low_confidence_burst(detector, tmp_path, confidence):
    """A real image whose stored box is set to a chosen score, to test the rule on it."""
    pipeline = Pipeline(CONFIG, db_path=tmp_path / "low.db", detector=detector)
    pipeline.process([{"path": str(real("bobcat")), "image_id": "x", "seq_id": "b", "camera_id": "33"}])
    pipeline.conn.execute("UPDATE detections SET confidence = ?, species = 'bobcat'", (confidence,))
    pipeline.conn.execute("UPDATE images SET top_confidence = ?", (confidence,))
    pipeline.rebuild_burst("b")
    return pipeline


def test_voted_burst_below_count_threshold_gets_no_count(detector, tmp_path):
    """The max(1, ...) rule was removed: most alerts it added named the wrong species."""
    pipeline = low_confidence_burst(detector, tmp_path, 0.25)
    assert pipeline.conn.execute("SELECT COUNT(*) FROM bursts").fetchone()[0] == 0
    assert pipeline.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0
    assert pipeline.conn.execute("SELECT triage FROM images").fetchone()[0] == "keep"
    pipeline.conn.close()


def test_box_below_the_vote_gate_stays_uncounted_and_goes_to_review(detector, tmp_path):
    pipeline = low_confidence_burst(detector, tmp_path, 0.1)
    assert pipeline.conn.execute("SELECT COUNT(*) FROM bursts").fetchone()[0] == 0
    assert pipeline.conn.execute("SELECT triage FROM images").fetchone()[0] == "review"
    pipeline.conn.close()


def test_upload_job_reports_good_and_bad_files(detector, tmp_path, monkeypatch):
    import src.pipeline as module
    folder = tmp_path / "job"
    folder.mkdir()
    shutil.copy(real("bobcat"), folder / "good.jpg")
    (folder / "bad.jpg").write_text("not an image")
    monkeypatch.setenv("WILDWATCH_DB", str(tmp_path / "job.db"))
    monkeypatch.setattr(module, "Pipeline", lambda use_llm: Pipeline(CONFIG, detector=detector))
    module.run_upload_job(folder, "33", False)
    job = json.loads((folder / "job.json").read_text())
    assert job["state"] == "done"
    by_file = {r["file"]: r for r in job["results"]}
    assert by_file["bad.jpg"]["status"] == "unreadable"
    assert by_file["good.jpg"]["alert"]["village"] == "Ambernook"


def test_upload_job_reports_a_crash(tmp_path, monkeypatch):
    import src.pipeline as module

    def broken(use_llm):
        raise RuntimeError("no GPU")
    monkeypatch.setattr(module, "Pipeline", broken)
    module.run_upload_job(tmp_path, None, False)
    job = json.loads((tmp_path / "job.json").read_text())
    assert job["state"] == "failed" and "no GPU" in job["error"]


# --- activity ---

def test_period_groups():
    assert period(True, False) == period(True, True) == "ir_night"
    assert period(False, True) == "color_night"
    assert period(False, False) == "day"


def test_real_daylight_image_is_not_a_dark_scene():
    row = next(r for r in TRAIN if r["date_captured"][11:13] == "14" and r["location"] != "120")
    assert not is_dark_scene(Image.open(IMAGES / row["file_name"]), CONFIG["activity"])


def test_hour_12_correction_needs_night():
    assert corrected_hour("2011-01-01 12:00:00", True) == 0
    assert corrected_hour("2011-01-01 12:00:00", False) == 12
    assert corrected_hour("2011-01-01 13:00:00", True) == 13


# --- classifier ---

def test_pick_species_skips_labels_outside_cct():
    classes = ["f1856211;;;;;;blank", "ba76d46e;mammalia;carnivora;felidae;lynx;rufus;bobcat"]
    assert pick_species(classes, [0.8, 0.1]) == ("bobcat", 0.1)
    assert pick_species(["f1856211;;;;;;blank"], [0.8]) == ("unknown", None)
    assert pick_species([], []) == ("unknown", None)


def box(species, confidence=0.9, category="animal"):
    return {"category": category, "species": species, "confidence": confidence}


def test_burst_vote_corrects_one_wrong_frame():
    frames = [[box("bobcat")], [box("deer", 0.4)], [box("bobcat")]]
    votes = burst_vote(frames, 0.2)
    assert votes == ["bobcat"]
    fixed = apply_burst_vote(frames, votes)
    assert [f[0]["species"] for f in fixed] == ["bobcat"] * 3


def test_burst_vote_keeps_a_real_second_species():
    frames = [[box("bobcat"), box("rabbit", 0.8)], [box("bobcat"), box("rabbit", 0.8)]]
    votes = burst_vote(frames, 0.2)
    assert votes == ["bobcat", "rabbit"]
    assert {d["species"] for f in apply_burst_vote(frames, votes) for d in f} == {"bobcat", "rabbit"}


def test_burst_vote_ignores_weak_boxes_unknowns_and_people():
    assert burst_vote([[box("deer", 0.1), box("unknown"), box(None, category="person")]], 0.2) == []
    frames = [[box(None, category="person"), box("unknown")], [box("coyote")]]
    fixed = apply_burst_vote(frames, burst_vote(frames, 0.2))
    assert fixed[0][0]["species"] is None and fixed[0][1]["species"] == "coyote"


def test_empty_burst_vote_changes_nothing():
    assert apply_burst_vote([[], []], []) == [[], []]


def test_real_worker_failure_gives_unknown_not_a_crash(tmp_path):
    item = {"path": str(real("bobcat")), "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1, "confidence": 0.9}]}
    results, error = classify_boxes([item], tmp_path)
    assert error
    assert results[item["path"]] == [{"species": "unknown", "score": None}]


# --- the whole pipeline, run once on every edge case ---

@pytest.fixture(scope="module")
def detector():
    return Detector(CONFIG, ROOT)


@pytest.fixture(scope="module")
def run(detector, tmp_path_factory):
    """Process all edge cases plus real bursts in one call. Returns (pipeline, results, files)."""
    folder = tmp_path_factory.mktemp("edge")
    source = real("bobcat")
    base = Image.open(source).convert("RGB")
    files = {}

    def add(kind, name, make):
        files[kind] = folder / name
        make(files[kind])

    data = source.read_bytes()
    add("corrupted", "corrupted.jpg", lambda p: p.write_bytes(data[:len(data) // 3]))
    add("non_image", "notes.jpg", lambda p: p.write_text("this is not an image"))
    add("zero_byte", "empty.jpg", lambda p: p.write_bytes(b""))
    add("black", "black.jpg", lambda p: Image.new("RGB", (2048, 1494), (0, 0, 0)).save(p))
    add("white", "white.jpg", lambda p: Image.new("RGB", (2048, 1494), (255, 255, 255)).save(p))
    add("tiny", "tiny.jpg", lambda p: base.resize((32, 32)).save(p))
    add("huge", "huge.jpg", lambda p: base.resize((8000, 5836)).save(p))
    add("grayscale", "gray.jpg", lambda p: base.convert("L").save(p))
    add("rotated", "rotated.jpg", lambda p: base.rotate(90, expand=True).save(p))
    add("png", "real.png", lambda p: base.save(p))
    add("no_timestamp", "no_time.jpg", lambda p: base.save(p, quality=90))
    add("vehicle_only", "car.jpg", lambda p: shutil.copy(real("car"), p))
    add("original", "original.jpg", lambda p: shutil.copy(source, p))
    add("duplicate", "copy_of_original.jpg", lambda p: shutil.copy(source, p))
    hikers = Image.open(IMAGES / "58b6673d-23d2-11e8-a6a3-ec086b02610b.jpg").convert("RGB")
    add("person_only", "person.jpg", lambda p: hikers.crop(
        (int(hikers.width * 0.52), 0, hikers.width, hikers.height)).save(p, quality=95))

    top = max(detector.detect_file(source)["detections"], key=lambda d: d["confidence"])
    tile = crop_box(base, top).resize((300, 300))

    def sheet(path):
        canvas = Image.new("RGB", (1200, 900), (90, 90, 90))
        for i in range(12):
            canvas.paste(tile, ((i % 4) * 300, (i // 4) * 300))
        canvas.save(path)
    add("many", "many.jpg", sheet)

    pipeline = Pipeline(CONFIG, db_path=folder / "test.db", detector=detector)
    items = [{"path": str(p)} for p in files.values()]
    results = pipeline.process(items)
    return pipeline, {k: results[str(p.resolve())] for k, p in files.items()}


@pytest.mark.parametrize("kind", ["corrupted", "non_image", "zero_byte"])
def test_bad_files_are_stored_as_unreadable(run, kind):
    pipeline, results = run
    assert results[kind]["status"] == "unreadable" and results[kind]["error"]
    assert results[kind]["detections"] == [] and results[kind]["alert"] is None
    row = pipeline.conn.execute("SELECT status FROM images WHERE image_id = ?", (results[kind]["image_id"],)).fetchone()
    assert row["status"] == "unreadable"


@pytest.mark.parametrize("kind", ["black", "white"])
def test_blank_images_finish_with_no_alert(run, kind):
    _, results = run
    assert results[kind]["status"] == "ok" and results[kind]["is_empty"] is True
    assert results[kind]["alert"] is None


@pytest.mark.parametrize("kind", ["tiny", "huge", "grayscale", "rotated", "png", "many"])
def test_unusual_images_finish(run, kind):
    _, results = run
    assert results[kind]["status"] == "ok"


@pytest.mark.parametrize("kind", ["grayscale", "png"])
def test_format_changes_keep_the_alert(run, kind):
    _, results = run
    assert results[kind]["alert"] is not None
    assert results[kind]["alert"]["min_observed_count"] >= 1


def test_missing_timestamp_still_gives_an_alert(run):
    _, results = run
    assert results["no_timestamp"]["captured_at"] is None
    assert results["no_timestamp"]["is_night"] in (True, False)
    assert results["no_timestamp"]["alert"] is not None


def test_unknown_camera_has_no_village_but_still_alerts(run):
    _, results = run
    assert results["original"]["alert"]["village"] is None


def test_duplicate_is_flagged_and_not_stored_twice(run):
    pipeline, results = run
    assert results["duplicate"]["duplicate_of"] == results["original"]["image_id"]
    sha = pipeline.conn.execute("SELECT sha256 FROM images WHERE image_id = ?",
                                (results["original"]["image_id"],)).fetchone()[0]
    assert pipeline.conn.execute("SELECT COUNT(*) FROM images WHERE sha256 = ?", (sha,)).fetchone()[0] == 1


def test_real_person_raises_a_poacher_alert(run):
    _, results = run
    alert = results["person_only"]["alert"]
    assert alert["species"] == "person"
    assert alert["possible_poacher"] is True
    assert alert["level"] in ("HIGH", "CRITICAL")


def test_vehicle_only_image_raises_no_alert(run):
    _, results = run
    assert any(d["category"] == "vehicle" for d in results["vehicle_only"]["detections"])
    assert results["vehicle_only"]["alert"] is None


def test_result_has_every_documented_field(run):
    _, results = run
    assert {"schema_version", "image_id", "file_path", "status", "error", "camera_id", "captured_at", "is_ir",
            "is_night", "is_empty", "detections", "alert"} <= set(results["original"])
    detection = results["original"]["detections"][0]
    assert {"category", "confidence", "x", "y", "w", "h", "species", "species_confidence",
            "classifier"} == set(detection)
    assert {"alert_id", "burst_id", "species", "min_observed_count", "danger_score", "level", "rangers", "vehicle",
            "possible_poacher", "village", "message", "message_source", "is_guess"} == set(results["original"]["alert"])


def test_real_burst_count_is_max_not_sum_and_rerun_changes_nothing(detector, tmp_path):
    frames = [r for r in TRAIN if r["seq_id"] == next(
        x["seq_id"] for x in TRAIN if x["labels"] == "opossum" and x["seq_num_frames"] == "3")]
    items = [{"path": str(IMAGES / r["file_name"]), "image_id": r["image_id"], "seq_id": r["seq_id"],
              "camera_id": r["location"], "captured_at": r["date_captured"], "source": "cct20_sample"}
             for r in frames]
    pipeline = Pipeline(CONFIG, db_path=tmp_path / "burst.db", detector=detector)
    pipeline.process(items)
    threshold = CONFIG["detection"]["count_threshold"]
    per_frame = [pipeline.conn.execute("SELECT COUNT(*) FROM detections WHERE image_id = ? AND confidence >= ? "
                                       "AND category = 'animal'", (i["image_id"], threshold)).fetchone()[0]
                 for i in items]
    before = [tuple(r) for r in pipeline.conn.execute("SELECT burst_id, species, min_observed_count FROM bursts")]
    assert sum(r[2] for r in before) <= sum(per_frame)
    assert max(r[2] for r in before) <= max(per_frame)
    pipeline.process(items)
    after = [tuple(r) for r in pipeline.conn.execute("SELECT burst_id, species, min_observed_count FROM bursts")]
    assert before == after
    assert pipeline.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == len(
        [r for r in after if r[1] != "vehicle"])
    pipeline.conn.close()


# --- incoming folder ---

def test_incoming_reads_sidecars_and_survives_a_broken_one(detector, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    cfg = {**CONFIG, "paths": {**CONFIG["paths"], "incoming": str(incoming)}}
    shutil.copy(real("bobcat"), incoming / "good.jpg")
    (incoming / "good.json").write_text(json.dumps({"labels": ["bobcat"], "n_boxes": 1, "camera_id": "33"}))
    shutil.copy(real("coyote"), incoming / "broken.jpg")
    (incoming / "broken.json").write_text("{not json")
    (incoming / "skip.json").write_text("{}")
    pipeline = Pipeline(cfg, db_path=tmp_path / "in.db", detector=detector)
    items = {Path(i["path"]).name: i for i in incoming_items(cfg, pipeline.conn)}
    assert set(items) == {"good.jpg", "broken.jpg"}
    assert items["good.jpg"]["labels"] == ["bobcat"] and items["good.jpg"]["camera_id"] == "33"
    assert "labels" not in items["broken.jpg"]
    pipeline.process(list(items.values()))
    assert incoming_items(cfg, pipeline.conn) == [], "processed files must not run again"
    pipeline.conn.close()


# --- golden set and regression ---

GOLDEN = json.loads((ROOT / "tests" / "golden_expected.json").read_text())


def test_golden_set_has_the_required_mix():
    assert len(GOLDEN) == 30
    assert all((ROOT / CONFIG["paths"]["golden"] / g["file_name"]).exists() for g in GOLDEN)
    assert sum(g["is_empty"] for g in GOLDEN) >= 5
    assert sum(g["n_boxes"] >= 2 for g in GOLDEN) >= 3
    assert len({s for g in GOLDEN for s in g["labels"]}) >= 10
    assert not any(g["split"].endswith("_test") for g in GOLDEN), "test images must stay out"
    assert any(g["labels"] == ["car"] for g in GOLDEN)
    assert sum(g["expect_person"] for g in GOLDEN) == 1


def test_golden_set_contains_night_images():
    night = 0
    for g in GOLDEN:
        image = Image.open(ROOT / CONFIG["paths"]["golden"] / g["file_name"])
        night += is_night(is_ir_image(image, CONFIG["activity"]["ir_channel_tolerance"]),
                          is_dark_scene(image, CONFIG["activity"]))
    assert night >= 5


def test_regression_fails_on_a_drop_over_two_points():
    baseline = {"empty_accuracy": 0.9, "animal_found_rate": 0.9, "species_accuracy": 0.9,
                "person_found_rate": 1.0, "count_mae": 0.1}
    fine ={**baseline, "species_accuracy": 0.885, "count_mae": 0.115}
    assert regression.compare(fine, baseline) == []
    assert regression.compare({**baseline, "species_accuracy": 0.87}, baseline)
    assert regression.compare({**baseline, "count_mae": 0.15}, baseline)


def test_regression_ignores_a_metric_with_no_data():
    baseline = {"empty_accuracy": 0.9, "animal_found_rate": None, "species_accuracy": 0.9,
                "person_found_rate": None, "count_mae": None}
    assert regression.compare({**baseline, "animal_found_rate": 0.1, "count_mae": 5.0}, baseline) == []


def test_golden_metrics_have_not_dropped_against_the_baseline(detector):
    metrics, classifier_error = regression.run(detector)
    assert classifier_error is None
    baseline = json.loads((ROOT / "docs" / "reports" / "baseline.json").read_text())
    assert regression.compare(metrics, baseline) == [], f"REGRESSION: {metrics} vs {baseline}"
