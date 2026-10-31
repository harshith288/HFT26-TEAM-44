"""Phase 2: empty filter, species mapping, burst counting, and edge-case images.

Detector tests run the real MDv6 model on real CCT images. Edge-case files are made
from real CCT photos; only the all-black and all-white images are synthetic, because
that is what those cases are.
"""
import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.activity import corrected_hour, is_dark_scene, is_ir_image, is_night
from src.classify import CLASSES, map_speciesnet_label
from src.count import MIN_COUNT_LABEL, burst_animal_sum, burst_animal_total, burst_counts
from src.detect import Detector, crop_box, filter_detections, is_empty, load_image

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
IMAGES = ROOT / CONFIG["paths"]["images"]
ACTIVITY = CONFIG["activity"]
ROWS = list(csv.DictReader(open(ROOT / CONFIG["paths"]["manifest"])))
TRAIN = [r for r in ROWS if r["split"] == "train"]


def real_image(label):
    """A real train image with exactly one label and one box."""
    row = next(r for r in TRAIN if r["labels"] == label and r["n_boxes"] == "1")
    return IMAGES / row["file_name"]


def det(category="animal", species=None, confidence=0.9):
    return {"category": category, "species": species, "confidence": confidence,
            "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}


@pytest.fixture(scope="module")
def detector():
    return Detector(CONFIG, ROOT)


# --- feature 3: counting ---

def test_burst_count_is_max_of_frames_not_sum():
    frames = [[det(species="coyote")], [det(species="coyote"), det(species="coyote")], [det(species="coyote")]]
    assert burst_counts(frames) == {"coyote": 2}
    assert burst_animal_total(frames) == 2
    assert burst_animal_sum(frames) == 4


def test_each_species_gets_its_own_max():
    frames = [[det(species="bobcat"), det(species="bobcat")], [det(species="rabbit")]]
    assert burst_counts(frames) == {"bobcat": 2, "rabbit": 1}


def test_person_and_vehicle_are_counted_under_their_own_labels():
    frames = [[det("person"), det("vehicle"), det(species="dog")]]
    assert burst_counts(frames) == {"person": 1, "vehicle": 1, "dog": 1}


def test_unclassified_animal_is_unknown():
    assert burst_counts([[det(species=None)]]) == {"unknown": 1}


def test_empty_burst_counts_zero():
    assert burst_counts([[], []]) == {}
    assert burst_animal_total([[], []]) == 0
    assert burst_animal_total([]) == 0


def test_count_is_labeled_minimum_observed():
    assert MIN_COUNT_LABEL == "minimum observed count"


# --- feature 1: filter logic ---

def test_threshold_boundary_is_inclusive():
    assert filter_detections([det(confidence=0.05)], 0.05)
    assert not filter_detections([det(confidence=0.049)], 0.05)


def test_empty_rule():
    assert is_empty([], 0.05)
    assert is_empty([det(confidence=0.01)], 0.05)
    assert not is_empty([det("vehicle", confidence=0.5)], 0.05)


def test_crop_of_a_tiny_box_at_the_edge_is_valid():
    image = Image.open(real_image("bobcat"))
    tiny = {"x": 0.9999, "y": 0.9999, "w": 0.00001, "h": 0.00001}
    crop = crop_box(image, tiny)
    assert crop.width >= 1 and crop.height >= 1


# --- feature 2: label mapping, using real SpeciesNet label strings ---

@pytest.mark.parametrize("label,expected", [
    ("ba76d46e;mammalia;carnivora;felidae;lynx;rufus;bobcat", "bobcat"),
    ("9212982e;mammalia;carnivora;felidae;felis;catus;domestic cat", "cat"),
    ("e70effd7;mammalia;carnivora;mustelidae;taxidea;taxus;american badger", "badger"),
    ("b1352069;aves;;;;;bird", "bird"),
    ("94d8284e;aves;galliformes;phasianidae;meleagris;gallopavo;wild turkey", "bird"),
    ("3d80f1d6;mammalia;carnivora;canidae;canis;familiaris;domestic dog", "dog"),
    ("x;mammalia;carnivora;canidae;canis;latrans;coyote", "coyote"),
    ("x;mammalia;carnivora;canidae;urocyon;cinereoargenteus;gray fox", "fox"),
    ("x;mammalia;rodentia;sciuridae;otospermophilus;beecheyi;california ground squirrel", "squirrel"),
    ("x;mammalia;rodentia;muridae;rattus;rattus;black rat", "rodent"),
    ("x;mammalia;carnivora;felidae;puma;concolor;puma", "mountain_lion"),
])
def test_speciesnet_labels_map_to_cct(label, expected):
    assert map_speciesnet_label(label) == expected


@pytest.mark.parametrize("label", [
    None, "", "f1856211;;;;;;blank", "1f689929;;;;;;animal",
    "f2efdae9;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result",
    "3c5ca701;mammalia;carnivora;viverridae;viverra;zibetha;large indian civet",
    "990ae9dd;mammalia;primates;hominidae;homo;sapiens;human",
])
def test_labels_outside_cct_map_to_none(label):
    assert map_speciesnet_label(label) is None


def test_every_class_has_a_danger_weight_or_uses_unknown():
    assert set(CLASSES) - set(CONFIG["species_weights"]) == {"badger", "fox"}


# --- activity: the CCT 12 AM problem ---

def test_ir_image_at_hour_12_is_midnight():
    assert corrected_hour("2011-06-18 12:50:56", True) == 0
    assert is_night(True, False) is True


def test_color_image_at_hour_12_stays_noon():
    assert corrected_hour("2011-06-18 12:50:56", False) == 12
    assert is_night(False, False) is False


def test_missing_timestamp_gives_no_hour_but_night_still_works():
    assert corrected_hour(None, False) is None
    assert corrected_hour("", True) is None
    assert is_night(True, False) is True
    assert is_night(False, True) is True


def test_ir_detection_on_real_images():
    tolerance = ACTIVITY["ir_channel_tolerance"]
    night = next(r for r in TRAIN if r["date_captured"][11:13] == "02" and r["location"] != "120")
    noon = next(r for r in TRAIN if r["date_captured"][11:13] == "14")
    assert is_ir_image(Image.open(IMAGES / night["file_name"]), tolerance)
    assert not is_ir_image(Image.open(IMAGES / noon["file_name"]), tolerance)


def test_white_flash_camera_is_color_at_night_but_still_counts_as_night():
    """Camera 120 uses a white flash, so its night photos are not IR."""
    tolerance = ACTIVITY["ir_channel_tolerance"]
    row = next(r for r in ROWS if r["location"] == "120" and r["date_captured"][11:13] == "02")
    image = Image.open(IMAGES / row["file_name"])
    ir = is_ir_image(image, tolerance)
    dark = is_dark_scene(image, ACTIVITY)
    assert ir is False
    assert dark is True
    assert is_night(ir, dark) is True


# --- real detector on real images ---

def test_detector_finds_the_animal_in_a_real_image(detector):
    result = detector.detect_file(real_image("bobcat"))
    assert result["status"] == "ok"
    kept = filter_detections(result["detections"], CONFIG["detection"]["count_threshold"])
    assert [d["category"] for d in kept] == ["animal"]


def test_vehicle_only_image_is_detected_as_vehicle(detector):
    result = detector.detect_file(real_image("car"))
    kept = filter_detections(result["detections"], CONFIG["detection"]["count_threshold"])
    assert "vehicle" in {d["category"] for d in kept}


def test_person_only_image_is_detected_as_person(detector, tmp_path):
    """CCT has no person label, so this is a real hiker photo (labeled dog), confirmed
    by eye and cropped to one person, which removes the dog and the other hiker."""
    image = Image.open(IMAGES / "58b6673d-23d2-11e8-a6a3-ec086b02610b.jpg").convert("RGB")
    crop = image.crop((int(image.width * 0.52), 0, image.width, image.height))
    path = tmp_path / "person_only.jpg"
    crop.save(path, quality=95)
    kept = filter_detections(detector.detect_file(path)["detections"], CONFIG["detection"]["count_threshold"])
    assert "person" in {d["category"] for d in kept}
    assert "animal" not in {d["category"] for d in kept}


# --- edge cases: nothing may crash ---

def edge_file(kind, tmp_path):
    source = real_image("bobcat")
    if kind == "corrupted":
        data = source.read_bytes()
        path = tmp_path / "corrupted.jpg"
        path.write_bytes(data[:len(data) // 3])
    elif kind == "scrambled":
        path = tmp_path / "scrambled.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + source.read_bytes()[1000:3000])
    elif kind == "non_image":
        path = tmp_path / "notes.jpg"
        path.write_text("this is not an image")
    elif kind == "zero_byte":
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
    return path


@pytest.mark.parametrize("kind", ["corrupted", "scrambled", "non_image", "zero_byte"])
def test_bad_files_are_reported_not_raised(detector, tmp_path, kind):
    result = detector.detect_file(edge_file(kind, tmp_path))
    assert result["status"] == "unreadable"
    assert result["error"]
    assert result["detections"] == []


def transformed(kind, tmp_path):
    image = Image.open(real_image("bobcat")).convert("RGB")
    if kind == "black":
        image = Image.new("RGB", (2048, 1494), (0, 0, 0))
    elif kind == "white":
        image = Image.new("RGB", (2048, 1494), (255, 255, 255))
    elif kind == "tiny":
        image = image.resize((32, 32))
    elif kind == "huge":
        image = image.resize((8000, 5836))
    elif kind == "grayscale":
        image = image.convert("L")
    elif kind == "rotated":
        image = image.rotate(90, expand=True)
    suffix = ".png" if kind == "png" else ".jpg"
    path = tmp_path / f"{kind}{suffix}"
    image.save(path)
    return path


@pytest.mark.parametrize("kind", ["black", "white", "tiny", "huge", "grayscale", "rotated", "png"])
def test_unusual_images_do_not_crash(detector, tmp_path, kind):
    result = detector.detect_file(transformed(kind, tmp_path))
    assert result["status"] == "ok"
    assert isinstance(result["detections"], list)


@pytest.mark.parametrize("kind", ["grayscale", "png"])
def test_format_changes_do_not_lose_the_animal(detector, tmp_path, kind):
    result = detector.detect_file(transformed(kind, tmp_path))
    assert filter_detections(result["detections"], CONFIG["detection"]["count_threshold"])


def test_blank_images_are_filtered_as_empty(detector, tmp_path):
    for kind in ("black", "white"):
        result = detector.detect_file(transformed(kind, tmp_path))
        assert is_empty(result["detections"], CONFIG["detection"]["confidence_threshold"]), kind


def test_image_with_more_than_ten_animals_does_not_crash(detector, tmp_path):
    """No real CCT sample image has 10+ boxes, so a real animal crop is tiled 12 times."""
    source = Image.open(real_image("bobcat")).convert("RGB")
    result = detector.detect_file(real_image("bobcat"))
    top = max(result["detections"], key=lambda d: d["confidence"])
    crop = crop_box(source, top).resize((300, 300))
    sheet = Image.new("RGB", (1200, 900), (90, 90, 90))
    for i in range(12):
        sheet.paste(crop, ((i % 4) * 300, (i // 4) * 300))
    path = tmp_path / "many.jpg"
    sheet.save(path)
    assert detector.detect_file(path)["status"] == "ok"


def test_load_image_applies_exif_rotation(tmp_path):
    image = Image.open(real_image("bobcat")).convert("RGB")
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated: width and height swap when upright
    path = tmp_path / "exif.jpg"
    image.save(path, exif=exif)
    loaded, error = load_image(path)
    assert error is None and (loaded.width, loaded.height) == (image.height, image.width)


# --- evaluation integrity ---

TEST_IDS = {r["image_id"] for r in ROWS if r["split"] in ("cis_test", "trans_test")}
FINAL_REPORT = ROOT / "docs" / "reports" / "final_eval.json"


@pytest.mark.skipif(FINAL_REPORT.exists(), reason="Phase 5 has opened the test splits")
@pytest.mark.parametrize("name", ["detections.json", "ir.json", "speciesnet.json"])
def test_test_splits_are_still_unopened(name):
    cached = json.loads((ROOT / "data" / "cache" / name).read_text())
    assert not set(cached) & TEST_IDS


@pytest.mark.parametrize("name", ["phase2_detection.json", "phase2_species.json", "detection_speed.json",
                                  "speciesnet_speed.json", "final_eval.json", "upload_e2e.json",
                                  "llm_live_test.json"])
def test_reports_record_how_they_were_made(name):
    report = json.loads((ROOT / "docs" / "reports" / name).read_text())
    assert report["command"] and report["timestamp"] and report["hardware"]
    assert any(k.startswith("n_real_images") and v > 0 for k, v in report.items())


def test_discard_threshold_came_from_tuning_val():
    tuning = json.loads((ROOT / "docs" / "reports" / "discard_tuning.json").read_text())
    assert CONFIG["triage"]["discard_below"] == tuning["chosen_discard_below"]
    chosen = next(m for m in tuning["tuning_sweep"] if m["discard_below"] == tuning["chosen_discard_below"])
    assert chosen["pct_animal_bursts_lost"] < 1.0


def test_config_uses_the_val_selected_settings():
    detection = json.loads((ROOT / "docs" / "reports" / "phase2_detection.json").read_text())
    species = json.loads((ROOT / "docs" / "reports" / "phase2_species.json").read_text())
    assert CONFIG["detection"]["confidence_threshold"] == detection["chosen_filter_threshold"]
    assert CONFIG["detection"]["count_threshold"] == detection["chosen_count_threshold"]
    assert CONFIG["classification"]["method"] == species["best_by_val_macro_f1"]
