"""Upload identification: any animal, its real label, and CCT left untouched."""
import json
from pathlib import Path

import pytest
import yaml

from src import pipeline as pipeline_module
from src.classify import CLASSES, classify_global, global_decision, needs_fallback, parse_label, pick_species
from src.pipeline import Pipeline
from src.risk import species_weight
from src.species import group_for, load_species

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
SETTINGS = CONFIG["global_identification"]
# The real SpeciesNet output for the lion photo that the old rule called a bird.
LION = ["ddf59264;mammalia;carnivora;felidae;panthera;leo;lion", "x;mammalia;artiodactyla;bovidae;tragelaphus;oryx;common eland",
        "y;aves;;;;;bird"]
LION_SCORES = [0.99774, 0.00052, 0.00016]
BIOCLIP = {"species": "chital", "score": 0.9}


def test_the_lion_bug_is_fixed_and_the_old_rule_shows_why():
    assert pick_species(LION, LION_SCORES)[0] == "bird"  # the CCT rule, kept only for CCT evaluation
    assert global_decision(LION, LION_SCORES, None, SETTINGS) == ("lion", 0.99774, "SpeciesNet")


@pytest.mark.parametrize("label", ["a;mammalia;;;;;mammal", "a;mammalia;carnivora;;;;carnivorous mammal", "a;aves;;;;;bird",
                                   "a;mammalia;carnivora;felidae;;;cat family", "a;;;;;;blank", "a;;;;;;animal",
                                   "a;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result"])
def test_vague_labels_are_never_shown_as_the_answer(label):
    assert needs_fallback([label], [0.99], SETTINGS) or parse_label(label)[1][4] == "no cv result"


def test_vague_label_uses_bioclip_when_it_is_confident():
    assert global_decision(["a;mammalia;;;;;mammal"], [0.99], BIOCLIP, SETTINGS) == ("chital", 0.9, "BioCLIP")


def test_low_score_uses_bioclip():
    assert global_decision(LION, [0.5], BIOCLIP, SETTINGS) == ("chital", 0.9, "BioCLIP")


WEAK = {"species": "chital", "score": SETTINGS["bioclip_min_score"] - 0.01}


def test_below_both_thresholds_gives_a_labeled_guess_with_speciesnet_first():
    assert global_decision(LION, [0.5], WEAK, SETTINGS) == ("lion", 0.5, "SpeciesNet guess")


def test_vague_speciesnet_label_falls_to_a_bioclip_guess():
    assert global_decision(["a;mammalia;;;;;mammal"], [0.99], WEAK, SETTINGS) == ("chital", WEAK["score"], "BioCLIP guess")


def test_unknown_only_when_there_is_no_name_to_offer():
    assert global_decision([], [], None, SETTINGS) == ("unknown", None, None)
    assert global_decision(["a;mammalia;;;;;mammal"], [0.99], None, SETTINGS) == ("unknown", None, None)


def test_guesses_can_be_turned_off():
    off = {**SETTINGS, "show_best_guess": False}
    assert global_decision(LION, [0.5], WEAK, off) == ("unknown", None, None)


def test_a_guess_uses_the_unknown_weight_and_says_possibly():
    from src import llm
    from src.risk import assess
    sure, guess = assess("tiger", 1, False, "33", CONFIG), assess("tiger", 1, False, "33", CONFIG, guess=True)
    assert sure["danger_score"] == pytest.approx(0.9 * 1.5)
    assert guess["danger_score"] == pytest.approx(0.5 * 1.5) and guess["is_guess"] is True
    cow = assess("domestic cattle", 1, False, "33", CONFIG, guess=True)
    assert cow["danger_score"] == guess["danger_score"], "a guess must not lower the risk either"
    assert "possibly tiger" in llm.template_message({**guess, "camera_id": "33"})
    assert "possibly tiger" in llm.alert_prompt({**guess, "camera_id": "33"})


def test_thresholds_are_inclusive():
    assert global_decision(LION, [SETTINGS["speciesnet_min_score"]], None, SETTINGS)[2] == "SpeciesNet"
    assert global_decision(LION, [0], {"species": "chital", "score": SETTINGS["bioclip_min_score"]}, SETTINGS)[2] == "BioCLIP"


def test_human_label_becomes_person():
    assert parse_label("a;mammalia;primates;hominidae;homo;sapiens;human")[0] == "person"


def test_workers_failing_gives_unknown_not_a_crash(tmp_path):
    results, error = classify_global([{"path": "x.jpg", "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1, "confidence": 0.9}]}], tmp_path, CONFIG)
    assert error
    assert results["x.jpg"] == [{"species": "unknown", "score": None, "model": None}]


def test_uploads_go_global_and_cct_images_keep_the_cct_rule(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_module, "classify_boxes", lambda items, root: (calls.append(("cct", len(items))), ({i["path"]: [{"species": "bobcat", "score": 0.9}] for i in items}, None))[1])
    monkeypatch.setattr(pipeline_module, "classify_global", lambda items, root, cfg: (calls.append(("world", len(items))), ({i["path"]: [{"species": "lion", "score": 0.9, "model": "SpeciesNet"}] for i in items}, None))[1])
    runner = Pipeline.__new__(Pipeline)
    runner.cfg, runner.classifier_error = CONFIG, None
    box = lambda: {"category": "animal", "x": 0, "y": 0, "w": 0.5, "h": 0.5, "confidence": 0.9}  # noqa: E731
    upload, sample = box(), box()
    runner.add_species([({"source": "incoming", "file_path": "/a.jpg", "width": 100, "height": 100}, [upload]),
                        ({"source": "cct20_sample", "file_path": "/b.jpg", "width": 100, "height": 100}, [sample])])
    assert sorted(calls) == [("cct", 1), ("world", 1)]
    assert (upload["species"], upload["classifier"]) == ("lion", "SpeciesNet")
    assert (sample["species"], sample["classifier"]) == ("bobcat", CONFIG["classification"]["method"])


@pytest.mark.parametrize("name,weight", [
    ("lion", 0.9), ("tiger", 0.9), ("leopard", 0.9), ("asian elephant", 0.9), ("african elephant", 0.9), ("sloth bear", 0.8), ("polar bear", 0.8), ("black mamba", 0.6), ("western diamondback rattlesnake", 0.6),
    ("wild boar", 0.7), ("indian wolf", 0.7), ("striped hyena", 0.7), ("indian cobra", 0.6), ("russell's viper", 0.6), ("chital", 0.3),
    ("nilgai", 0.3), ("blackbuck", 0.3), ("rhesus macaque", 0.3), ("gray langur", 0.3), ("domestic cattle", 0.1), ("domestic dog", 0.1),
    ("domestic cat", 0.1), ("domestic horse", 0.1), ("dromedary camel", 0.1), ("domestic pig", 0.1), ("indian peafowl", 0.1),
    ("person", 1.0), ("unknown", 0.5), ("something new", 0.5)])
def test_danger_weights_follow_your_groups(name, weight):
    assert species_weight(name, CONFIG) == weight


def test_groups_match_taxonomy_not_name_text():
    assert group_for("leopard cat", CONFIG) != "big_cats"


def test_cct_weights_are_unchanged():
    for label in CLASSES + ["cat", "dog", "person", "vehicle", "unknown"]:
        expected = CONFIG["species_weights"].get(label, CONFIG["species_weights"]["unknown"])
        assert species_weight(label, CONFIG) == expected, label


def test_species_list_is_broad_and_has_indian_wildlife_and_livestock():
    species = load_species()
    assert len(species) >= 500
    counts = {}
    for s in species.values():
        counts[s["taxonomy"][0]] = counts.get(s["taxonomy"][0], 0) + 1
    assert all(counts[c] >= 40 for c in ("mammalia", "aves", "reptilia"))
    for name in ("bengal tiger", "sloth bear", "chital", "sambar", "nilgai", "gaur", "blackbuck", "indian cobra", "king cobra", "russell's viper",
                 "mugger crocodile", "gharial", "rhesus macaque", "gray langur", "indian peafowl", "striped hyena", "domestic cattle",
                 "zebu cattle", "domestic water buffalo", "domestic goat", "domestic sheep", "domestic chicken"):
        assert name in species, name


def test_uploads_use_no_country_and_no_geofence():
    worker = (ROOT / "src" / "scripts" / "speciesnet_worker.py").read_text(encoding="utf-8")
    assert 'instances.append({"filepath": item["path"]} if world else {"filepath": item["path"], "country": "USA"})' in worker
    assert 'components="classifier"' in worker
    assert "geofence" not in worker and "ensemble" not in worker.lower()
    source = (ROOT / "src" / "classify.py").read_text(encoding="utf-8")
    assert 'run_worker("speciesnet_worker.py", items, root, extra=("global",), timeout=timeout)' in source


def test_only_cct_images_can_reach_the_usa_path():
    source = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    assert 'record["source"] == "incoming" and self.cfg.get("upload_mode") == "global"' in source
    assert CONFIG["upload_mode"] == "global"


@pytest.mark.parametrize("area,names", [
    ("Africa", ["lion", "okapi", "ring-tailed lemur", "black mamba", "puff adder", "african forest elephant"]),
    ("Asia", ["tiger", "giant panda", "komodo dragon", "proboscis monkey", "saiga antelope", "king cobra"]),
    ("Europe", ["european hedgehog", "european adder", "european beaver", "white stork", "mute swan"]),
    ("North America", ["polar bear", "western diamondback rattlesnake", "american alligator", "gila monster", "muskox"]),
    ("South America", ["green anaconda", "alpaca", "vicuna", "galapagos tortoise", "geoffroy's spider monkey", "maned wolf"]),
    ("Oceania", ["platypus", "tasmanian devil", "southern cassowary", "coastal taipan", "tuatara", "north island brown kiwi"]),
    ("Antarctica and oceans", ["adelie penguin", "leopard seal", "weddell seal", "walrus", "sea otter", "orca", "dugong"]),
])
def test_species_list_covers_every_continent(area, names):
    species = load_species()
    assert [n for n in names if n not in species] == [], area


def test_thresholds_came_from_the_tuning_report():
    tuning = json.loads((ROOT / "docs" / "reports" / "global_tuning.json").read_text())
    assert SETTINGS["speciesnet_min_score"] == tuning["chosen"]["speciesnet_min_score"]
    assert SETTINGS["bioclip_min_score"] == tuning["chosen"]["bioclip_min_score"]
    assert "not the 20 test photos" in tuning["photos"]


def test_guess_method_came_from_tuning_photos():
    report = json.loads((ROOT / "docs" / "reports" / "global_guess.json").read_text())
    assert report["chosen"] == "speciesnet_first"
    assert report["scope"].startswith("tuning photos")


def test_real_lion_photo_through_the_real_upload_pipeline(tmp_path):
    photo = ROOT / "data" / "global_photos" / "test" / "01_lion.jpg"
    runner = Pipeline(CONFIG, db_path=tmp_path / "t.db")
    result = next(iter(runner.process([{"path": str(photo), "source": "incoming"}]).values()))
    runner.conn.close()
    top = max((d for d in result["detections"] if d["category"] == "animal"), key=lambda d: d["confidence"])
    assert top["species"] not in CLASSES
    assert top["classifier"] in ("SpeciesNet", "BioCLIP", "SpeciesNet guess", "BioCLIP guess", None)
