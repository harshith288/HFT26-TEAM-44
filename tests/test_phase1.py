"""Phase 1 contract: config, schema, gitignore, and the real sample manifest."""
import csv
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from src.db import DDL, SCHEMA_VERSION, init_db

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@pytest.fixture
def conn(tmp_path):
    connection = init_db(tmp_path / "test.db")
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def manifest():
    return list(csv.DictReader(open(ROOT / "data" / "sample" / "manifest.csv")))


def schema_md_tables():
    """Parse {table: [columns]} out of SCHEMA.md."""
    tables, current = {}, None
    for line in (ROOT / "docs" / "SCHEMA.md").read_text().splitlines():
        heading = re.match(r"### (\w+)", line)
        if heading:
            current = heading.group(1)
            tables[current] = []
        elif line.startswith("## ") and line != "## Tables":
            current = None
        elif current and line.startswith("| ") and not line.startswith("| column"):
            tables[current].append(line.split("|")[1].strip())
    return tables


# --- config ---

def test_species_weights_match_the_brief():
    expected = {"bobcat": 0.7, "coyote": 0.6, "mountain_lion": 0.9, "deer": 0.3, "raccoon": 0.2,
                "opossum": 0.1, "skunk": 0.2, "rabbit": 0.1, "squirrel": 0.1, "rodent": 0.1,
                "bird": 0.1, "dog": 0.4, "person": 1.0, "unknown": 0.5, "cat": 0.2}
    weights = CONFIG["species_weights"]
    assert {k: weights[k] for k in expected} == expected


def test_risk_thresholds_ascend():
    r = CONFIG["risk_levels"]
    assert 0 < r["medium_min"] < r["high_min"] < r["critical_min"]


def test_every_level_has_a_response_and_rangers_never_drop():
    rangers = [CONFIG["response"][level]["rangers"] for level in LEVELS]
    assert rangers == sorted(rangers)
    assert all(isinstance(CONFIG["response"][level]["vehicle"], bool) for level in LEVELS)


def test_night_factor_is_at_least_one():
    assert CONFIG["night_factor"] >= 1.0


def test_every_sampled_camera_has_exactly_one_village(manifest):
    mapped = [c for v in CONFIG["villages"].values() for c in v["cameras"]]
    assert len(mapped) == len(set(mapped)), "a camera is in two villages"
    assert {r["location"] for r in manifest} == set(mapped)


def test_config_holds_no_api_key():
    llm = CONFIG["llm"]
    assert "api_key" not in llm
    assert llm["api_key_env"] == "FEATHERLESS_API_KEY"
    assert llm["base_url"] == "https://api.featherless.ai/v1"


# --- schema ---

def test_schema_md_matches_real_database(conn):
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    real = {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] for t in names}
    assert schema_md_tables() == real


def test_schema_version_is_stored_and_documented(conn):
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert f"Schema version: {SCHEMA_VERSION}" in (ROOT / "docs" / "SCHEMA.md").read_text()


def test_init_db_can_run_twice(tmp_path):
    init_db(tmp_path / "x.db").close()
    init_db(tmp_path / "x.db").close()


def test_version_1_database_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    added = "processed_at TEXT,\n    top_confidence REAL,\n    triage TEXT CHECK (triage IN ('keep', 'review', 'discard'))"
    version_1 = DDL.replace(added, "processed_at TEXT")
    assert "top_confidence" not in version_1
    old = sqlite3.connect(path)
    old.executescript(version_1)
    old.execute("INSERT INTO images (image_id, file_path, source, status) VALUES ('x', 'a.jpg', 'incoming', 'ok')")
    old.commit()
    old.close()
    upgraded = init_db(path)
    assert {"top_confidence", "triage"} <= {c[1] for c in upgraded.execute("PRAGMA table_info(images)")}
    assert upgraded.execute("SELECT image_id FROM images").fetchone()[0] == "x", "existing rows must survive"
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert "is_guess" in {c[1] for c in upgraded.execute("PRAGMA table_info(alerts)")}
    upgraded.close()


def test_foreign_key_is_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO detections (image_id, category, confidence, x, y, w, h) "
                     "VALUES ('missing', 'animal', 0.9, 0, 0, 1, 1)")


def test_invalid_alert_level_is_rejected(conn):
    conn.execute("INSERT INTO bursts (burst_id, species, n_frames, min_observed_count) "
                 "VALUES ('b', 'coyote', 3, 1)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO alerts (burst_id, species, min_observed_count, danger_score, level, "
                     "rangers, vehicle, possible_poacher, created_at) "
                     "VALUES ('b', 'coyote', 1, 0.5, 'SEVERE', 0, 0, 0, 'now')")


def test_unreadable_image_can_be_stored_without_metadata(conn):
    conn.execute("INSERT INTO images (image_id, file_path, source, status, error) "
                 "VALUES ('bad', 'x.jpg', 'incoming', 'unreadable', 'corrupt')")
    assert conn.execute("SELECT seq_id, captured_at FROM images").fetchone()[1] is None


# --- repo safety ---

@pytest.mark.parametrize("pattern", [".env", "*.db", "data/raw/", "uploads/", "*.pt", "*.pth", ".venv/", "__pycache__/"])
def test_sensitive_paths_are_git_ignored(pattern):
    assert pattern in (ROOT / ".gitignore").read_text().splitlines(), f"{pattern} is not ignored"


def test_requirements_are_all_pinned():
    for name in ("requirements.txt", "requirements-torch.txt"):
        lines = [l for l in (ROOT / name).read_text().splitlines()
                 if l and not l.startswith(("#", "--"))]
        assert lines and all("==" in l for l in lines)


# --- real sample ---

def test_sample_is_about_1500_images_and_all_on_disk(manifest):
    assert 1450 <= len(manifest) <= 1550
    images = ROOT / "data" / "sample" / "images"
    assert all((images / r["file_name"]).stat().st_size > 0 for r in manifest)


def test_each_burst_is_whole_and_in_one_split(manifest):
    bursts = defaultdict(list)
    for r in manifest:
        bursts[r["seq_id"]].append(r)
    for rows in bursts.values():
        assert len({r["split"] for r in rows}) == 1
        assert len(rows) == int(rows[0]["seq_num_frames"])


def test_trans_cameras_are_unseen_in_train(manifest):
    cameras = defaultdict(set)
    for r in manifest:
        cameras[r["split"]].add(r["location"])
    assert not cameras["train"] & (cameras["trans_test"] | cameras["trans_val"])
    assert cameras["cis_test"] <= cameras["train"]
