"""SQLite layer. The tables here are the frozen schema documented in docs/SCHEMA.md."""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 5

DDL = """
CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('cct20_sample', 'incoming')),
    split TEXT CHECK (split IN ('train', 'cis_val', 'trans_val', 'cis_test', 'trans_test')),
    seq_id TEXT,
    frame_num INTEGER,
    camera_id TEXT,
    captured_at TEXT,
    width INTEGER,
    height INTEGER,
    sha256 TEXT,
    is_ir INTEGER,
    is_night INTEGER,
    is_empty INTEGER,
    gt_labels TEXT,
    gt_n_boxes INTEGER,
    status TEXT NOT NULL CHECK (status IN ('ok', 'unreadable')),
    error TEXT,
    processed_at TEXT,
    top_confidence REAL,
    triage TEXT CHECK (triage IN ('keep', 'review', 'discard'))
);
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL REFERENCES images (image_id),
    category TEXT NOT NULL CHECK (category IN ('animal', 'person', 'vehicle')),
    confidence REAL NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    w REAL NOT NULL,
    h REAL NOT NULL,
    species TEXT,
    species_confidence REAL,
    classifier TEXT
);
CREATE TABLE IF NOT EXISTS bursts (
    burst_id TEXT NOT NULL,
    species TEXT NOT NULL,
    camera_id TEXT,
    start_time TEXT,
    end_time TEXT,
    n_frames INTEGER NOT NULL,
    min_observed_count INTEGER NOT NULL,
    is_night INTEGER,
    PRIMARY KEY (burst_id, species)
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    burst_id TEXT NOT NULL,
    species TEXT NOT NULL,
    camera_id TEXT,
    village TEXT,
    min_observed_count INTEGER NOT NULL,
    danger_score REAL NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    rangers INTEGER NOT NULL,
    vehicle INTEGER NOT NULL,
    possible_poacher INTEGER NOT NULL,
    message TEXT,
    message_source TEXT CHECK (message_source IN ('llm', 'template')),
    created_at TEXT NOT NULL,
    is_guess INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (burst_id, species) REFERENCES bursts (burst_id, species)
);
CREATE TABLE IF NOT EXISTS alert_status (
    burst_id TEXT NOT NULL,
    species TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('new', 'dispatched', 'false_alarm')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (burst_id, species)
);
CREATE TABLE IF NOT EXISTS daily_summaries (
    summary_date TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    summary_source TEXT NOT NULL CHECK (summary_source IN ('llm', 'template')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS briefs (
    brief_date TEXT PRIMARY KEY,
    brief TEXT NOT NULL,
    species_notes TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('llm', 'template')),
    data_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_seq ON images (seq_id);
CREATE INDEX IF NOT EXISTS idx_detections_image ON detections (image_id);
"""


def connect(path):
    """Open the database with foreign keys on and rows readable by column name."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path):
    """Create all tables. Safe to call repeatedly."""
    conn = connect(path)
    conn.executescript(DDL)
    columns = [c[1] for c in conn.execute("PRAGMA table_info(images)")]
    if "top_confidence" not in columns:  # upgrade a version 1 database
        conn.execute("ALTER TABLE images ADD COLUMN top_confidence REAL")
        conn.execute("ALTER TABLE images ADD COLUMN triage TEXT CHECK (triage IN ('keep', 'review', 'discard'))")
    if "is_guess" not in [c[1] for c in conn.execute("PRAGMA table_info(alerts)")]:  # upgrade version 3
        conn.execute("ALTER TABLE alerts ADD COLUMN is_guess INTEGER NOT NULL DEFAULT 0")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def database_path():
    cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
    return ROOT / (os.environ.get("WILDWATCH_DB") or cfg["paths"]["database"])


def data_version(conn):
    row = conn.execute("SELECT value FROM app_state WHERE key = 'data_version'").fetchone()
    return row[0] if row else 0


def bump_data_version(conn):
    """Every page's cached data is keyed on this number, so adding 1 refreshes them all."""
    conn.execute("INSERT INTO app_state (key, value) VALUES ('data_version', 1) "
                 "ON CONFLICT (key) DO UPDATE SET value = value + 1")
    conn.commit()
    return data_version(conn)


def save_brief(conn, brief, notes, source, version):
    """One brief per day. A newer upload replaces today's."""
    stamp = datetime.now()
    conn.execute("INSERT OR REPLACE INTO briefs (brief_date, brief, species_notes, source, data_version, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (stamp.strftime("%Y-%m-%d"), brief, json.dumps(notes), source, version,
                                               stamp.strftime("%Y-%m-%d %H:%M:%S")))


def set_alert_status(conn, burst_id, species, status):
    conn.execute("INSERT OR REPLACE INTO alert_status (burst_id, species, status, updated_at) VALUES (?, ?, ?, ?)",
                 (burst_id, species, status, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    return bump_data_version(conn)


def read_frames(path):
    """Every table the pages use, as DataFrames. pandas loads here, not at import, because
    it crashes when it shares a process with torch and the pipeline imports this file."""
    import pandas as pd
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        def table(query):
            return pd.read_sql_query(query, conn)
        frames = {
            "images": table("SELECT *, COALESCE(captured_at, processed_at) AS seen_at FROM images"),
            "detections": table("SELECT d.*, i.seq_id, i.file_path, i.camera_id FROM detections d "
                                "JOIN images i ON d.image_id = i.image_id"),
            "bursts": table("""
                SELECT b.*, i.confidence, i.source, COALESCE(b.start_time, i.processed_at) AS seen_at
                FROM bursts b LEFT JOIN (SELECT seq_id, MAX(top_confidence) AS confidence, MAX(source) AS source,
                                                MAX(processed_at) AS processed_at
                                         FROM images GROUP BY seq_id) i ON b.burst_id = i.seq_id"""),
            "alerts": table("""
                SELECT a.*, b.start_time, b.is_night, COALESCE(s.status, 'new') AS status, i.confidence, i.source,
                       COALESCE(b.start_time, i.processed_at) AS seen_at,
                       (SELECT f.file_path FROM images f JOIN detections d ON d.image_id = f.image_id
                        WHERE f.seq_id = a.burst_id ORDER BY d.confidence DESC LIMIT 1) AS file_path
                FROM alerts a JOIN bursts b ON a.burst_id = b.burst_id AND a.species = b.species
                LEFT JOIN alert_status s ON a.burst_id = s.burst_id AND a.species = s.species
                LEFT JOIN (SELECT seq_id, MAX(top_confidence) AS confidence, MAX(source) AS source,
                                  MAX(processed_at) AS processed_at
                           FROM images GROUP BY seq_id) i ON a.burst_id = i.seq_id"""),
            "summaries": table("SELECT * FROM daily_summaries ORDER BY summary_date DESC"),
            "briefs": table("SELECT * FROM briefs ORDER BY brief_date DESC"),
        }
    finally:
        conn.close()
    return frames


def _read_filtered(version, path, filters):
    """version is unused here. It is in the arguments so a new data_version misses the cache."""
    from src.analytics import apply_filters
    frames = apply_filters(read_frames(path), filters)
    frames["data_version"] = version
    return frames


_cached_read = None


def load_data(filters=None):
    """The one reader for every page: DataFrames after the shared filters, or None with no
    database. Streamlit loads on the first call, so the pipeline never imports it."""
    global _cached_read
    path = database_path()
    if not path.exists():
        return None
    if _cached_read is None:
        import streamlit as st
        _cached_read = st.cache_data(max_entries=32, show_spinner=False)(_read_filtered)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        version = data_version(conn)
    finally:
        conn.close()
    return _cached_read(version, str(path), filters or {})
