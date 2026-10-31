"""Dashboard checks: the real app on the real pipeline database.

Run by tests/test_phase4.py in a separate process, because pyarrow crashes when it
shares a process with torch. Can also be run alone with pytest.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from src import analytics, brief
from src.brief import updated_label
from src.db import data_version, init_db, load_data, read_frames
from src.pipeline import process_upload

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())


def run_app(env_db=None):
    if env_db:
        os.environ["WILDWATCH_DB"] = str(env_db)
    else:
        os.environ.pop("WILDWATCH_DB", None)
    return AppTest.from_file(str(ROOT / "src" / "streamlit_app.py"), default_timeout=90).run()


def test_dashboard_loads_with_every_page_and_no_errors():
    app = run_app()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Live alerts", "Upload", "Needs review", "Species", "Activity", "Cameras", "Model accuracy"]
    assert [m.label for m in app.metric][:5] == ["Images analysed", "Auto-discarded", "Needs review", "Kept",
                                                 "Alerts at MEDIUM or above"]


def test_dashboard_numbers_match_the_database():
    import sqlite3
    conn = sqlite3.connect(ROOT / CONFIG["paths"]["database"])
    images = conn.execute("SELECT COUNT(*) FROM images WHERE status = 'ok'").fetchone()[0]
    tiers = dict(conn.execute("SELECT triage, COUNT(DISTINCT seq_id) FROM images WHERE status = 'ok' GROUP BY triage"))
    conn.close()
    app = run_app()
    values = {m.label: m.value for m in app.metric}
    assert values["Images analysed"] == f"{images:,}"
    assert values["Auto-discarded"] == f"{tiers.get('discard', 0):,}"
    assert values["Needs review"] == f"{tiers.get('review', 0):,}"
    assert values["Kept"] == f"{tiers.get('keep', 0):,}"


def test_dashboard_shows_the_missed_target_honestly():
    app = run_app()
    assert "accuracy target is not reached" in " ".join(w.value for w in app.warning)
    values = {m.label: m.value for m in app.metric}
    final = json.loads((ROOT / "docs" / "reports" / "final_eval.json").read_text())
    assert values["Test-set accuracy on Caltech species"] == f"{final['groups']['all']['species']['accuracy']:.1f}%"


def test_filtering_to_one_village_does_not_crash():
    app = run_app()
    app.multiselect[0].select("Larkmere").run()
    assert not app.exception


def test_filter_with_no_matching_alerts_does_not_crash():
    app = run_app()
    app.multiselect[0].select("Thornby Vale")
    app.multiselect[1].select("CRITICAL").run()
    assert not app.exception


def test_dashboard_without_a_database_explains_itself(tmp_path):
    app = run_app(tmp_path / "missing.db")
    os.environ.pop("WILDWATCH_DB", None)
    assert not app.exception
    assert "No results yet" in app.warning[0].value


def test_dashboard_on_an_empty_database_does_not_crash(tmp_path):
    init_db(tmp_path / "empty.db").close()
    app = run_app(tmp_path / "empty.db")
    os.environ.pop("WILDWATCH_DB", None)
    assert not app.exception


def test_torch_never_loads_in_the_dashboard():
    import sys
    run_app()
    assert not {"torch", "ultralytics", "speciesnet"} & set(sys.modules)


def test_review_tab_lists_real_bursts():
    import sqlite3
    conn = sqlite3.connect(ROOT / CONFIG["paths"]["database"])
    count = conn.execute("SELECT COUNT(DISTINCT seq_id) FROM images WHERE triage = 'review'").fetchone()[0]
    conn.close()
    values = {m.label: m.value for m in run_app().metric}
    assert values["Needs review"] == f"{count:,}"


def live_db(tmp_path):
    copy = tmp_path / "copy.db"
    shutil.copy(ROOT / CONFIG["paths"]["database"], copy)
    init_db(copy).close()
    return copy


class FakeLLM:
    """Labeled mock: returns fixed text instead of calling Featherless, and records every prompt."""
    def __init__(self, reply):
        self.reply, self.prompts = reply, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, model, max_tokens, messages):
        self.prompts.append(messages[0]["content"])
        if isinstance(self.reply, Exception):
            raise self.reply
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.reply))], usage=None)


def facts():
    return analytics.brief_facts(read_frames(ROOT / CONFIG["paths"]["database"]))


def test_filters_change_what_a_page_reads(tmp_path):
    db = live_db(tmp_path)
    frames = read_frames(db)
    one = analytics.apply_filters(frames, {"locations": ("Larkmere",)})
    assert set(one["bursts"]["camera_id"]) <= {"115", "120", "125"} and len(one["bursts"]) < len(frames["bursts"])
    species = frames["bursts"]["species"].mode()[0]
    only = analytics.apply_filters(frames, {"species": (species,)})
    assert set(only["bursts"]["species"]) == {species} and set(only["images"]["seq_id"]) <= set(only["bursts"]["burst_id"])
    sure = analytics.apply_filters(frames, {"min_confidence": 0.9})
    assert (sure["images"]["top_confidence"] >= 0.9).all() and (sure["alerts"]["confidence"] >= 0.9).all()
    dated = analytics.apply_filters(frames, {"dates": ("2013-01-01", "2013-12-31")})
    assert dated["bursts"]["seen_at"].str[:4].eq("2013").all()


def test_load_data_is_cached_until_data_version_changes(tmp_path, monkeypatch):
    db = live_db(tmp_path)
    monkeypatch.setenv("WILDWATCH_DB", str(db))
    first = load_data({})
    conn = init_db(db)
    conn.execute("UPDATE images SET camera_id = 'changed' WHERE image_id = (SELECT image_id FROM images LIMIT 1)")
    conn.commit()
    assert "changed" not in set(load_data({})["images"]["camera_id"])
    from src.db import bump_data_version
    assert bump_data_version(conn) == first["data_version"] + 1
    conn.close()
    assert "changed" in set(load_data({})["images"]["camera_id"])


def test_template_brief_has_real_numbers_and_the_missed_target():
    f = facts()
    text = brief.template_brief(f)
    assert 4 <= brief.sentence_count(text) <= 6
    assert f["species"][0]["species"].replace("_", " ") in text
    assert "88.3%" in text and "below the 90% target" in text
    assert "uploads not measured" in text


def test_llm_brief_is_used_when_every_number_is_ours():
    f = facts()
    top = f["species"][0]
    reply = (f"{top['species']} leads with {top['bursts']} bursts. It is mostly at {top['top_place']}. "
             f"{f['uncertain']['bursts_needing_review']} bursts need review.")
    client = FakeLLM(reply)
    text, notes, source = brief.write_brief(f, CONFIG, None, client)
    assert source == "llm" and len(notes) == len(f["species"])
    assert text == f"{reply} {brief.accuracy_line(f)}"
    assert "uploads not measured" in text


def test_text_that_is_not_english_is_not_used():
    f = facts()
    top = f["species"][0]
    client = FakeLLM(json.dumps({top["species"]: "\u6700\u6d3b\u8dc3\u591c\u95f4"}))
    assert brief.write_brief(f, CONFIG, None, client)[1][top["species"]]["source"] == "template"


def test_a_score_may_be_written_as_a_percentage():
    assert brief.uses_only_our_numbers("Confidence is 97%, or 0.97.", {"score": 0.9712})
    assert not brief.uses_only_our_numbers("Confidence is 96%.", {"score": 0.9712})


def test_llm_brief_with_an_invented_number_falls_back_to_the_template():
    f = facts()
    client = FakeLLM("Tigers rose by 98765 bursts. One. Two. Three.")
    text, _, source = brief.write_brief(f, CONFIG, None, client)
    assert source == "template" and "98765" not in text and text == brief.template_brief(f)


def test_api_failure_still_gives_a_full_brief_and_notes():
    f = facts()
    text, notes, source = brief.write_brief(f, CONFIG, None, FakeLLM(ConnectionError("down")))
    assert source == "template" and text
    assert set(notes) == {r["species"] for r in f["species"]}
    assert all(n["source"] == "template" and n["note"] for n in notes.values())


def test_no_key_still_gives_a_full_brief():
    text, notes, source = brief.write_brief(facts(), CONFIG)
    assert source == "template" and text and notes


def test_prompts_hold_numbers_only_never_images_or_paths():
    f = facts()
    client = FakeLLM("not json")
    brief.write_brief(f, CONFIG, None, client)
    sent = " ".join(client.prompts)
    assert len(client.prompts) == 2
    assert "data/" not in sent and ".jpg" not in sent and "base64" not in sent and "file_path" not in sent


def test_llm_brief_that_drops_the_top_species_falls_back_to_the_template():
    f = facts()
    second = f["species"][1]["species"]
    reply = f"{second} leads. It is common. Many need review. The data has limits."
    assert brief.write_brief(f, CONFIG, None, FakeLLM(reply))[2] == "template"


def test_species_notes_keep_good_ai_notes_and_replace_bad_ones():
    f = facts()
    first, second = f["species"][0], f["species"][1]
    client = FakeLLM(json.dumps({first["species"]: f"Seen in {first['bursts']} bursts.", second["species"]: "Seen in 99999 bursts."}))
    _, notes, _ = brief.write_brief(f, CONFIG, None, client)
    assert notes[first["species"]]["source"] == "llm"
    assert notes[second["species"]]["source"] == "template"


def test_explain_detection_template_and_llm():
    details = {"detected": "animal", "species": "bobcat", "species_confidence": 0.91, "detector_confidence": 0.95,
               "model": "speciesnet", "decision": "keep", "period": "night", "infrared": True}
    text, source = brief.explain_detection(details, CONFIG)
    assert source == "template" and "bobcat" in text and "0.95" in text and "high confidence" in text
    good, source = brief.explain_detection(details, CONFIG, None, FakeLLM("A bobcat, confidence 0.95."))
    assert source == "llm" and "bobcat" in good
    _, source = brief.explain_detection(details, CONFIG, None, FakeLLM("A bobcat, confidence 0.12."))
    assert source == "template"


def test_ask_data_answers_only_from_our_numbers():
    f = facts()
    answer, source = brief.ask_data("How many bursts?", f, CONFIG, None, FakeLLM(f"{f['bursts']} bursts."))
    assert source == "llm" and answer == f"{f['bursts']} bursts."
    answer, source = brief.ask_data("How many?", f, CONFIG, None, FakeLLM("Exactly 987654."))
    assert source == "template" and "987654" not in answer
    answer, source = brief.ask_data("Weather?", f, CONFIG, None, FakeLLM(brief.NO_ANSWER))
    assert answer == brief.NO_ANSWER
    answer, source = brief.ask_data("How many?", f, CONFIG, None, FakeLLM(ConnectionError("down")))
    assert source == "template" and answer


@pytest.mark.parametrize("seconds,label", [(5, "Updated just now"), (59, "Updated just now"), (600, "Updated at 09:50")])
def test_updated_label(seconds, label):
    now = datetime(2026, 10, 31, 10, 0, 0)
    from datetime import timedelta
    assert updated_label((now - timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S"), now) == label


def test_one_brief_per_day_is_replaced_not_added(tmp_path):
    db = live_db(tmp_path)
    for _ in range(2):
        brief.refresh(db, use_llm=False)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM briefs").fetchone()[0] == 1
    assert conn.execute("SELECT data_version FROM briefs").fetchone()[0] == conn.execute(
        "SELECT value FROM app_state WHERE key = 'data_version'").fetchone()[0]
    conn.close()


def test_process_upload_updates_everything_through_one_call(tmp_path, monkeypatch):
    """A real run: real photo, real models in their own process, on a copy of the database."""
    db = live_db(tmp_path)
    monkeypatch.setenv("WILDWATCH_DB", str(db))
    monkeypatch.setenv("FEATHERLESS_API_KEY", "")
    before = load_data({})
    photo = next((ROOT / CONFIG["paths"]["golden"]).rglob("*.jpg"))
    stages = []
    outcome = process_upload([("wildwatch_test_upload.jpg", photo.read_bytes() + b"\0")], "33", stages.append, use_llm=False)
    assert outcome["state"] == "done", outcome["error"]
    assert "brief" in stages
    after = load_data({})
    assert after["data_version"] == before["data_version"] + 1
    assert len(after["images"]) == len(before["images"]) + 1
    assert after["briefs"].iloc[0]["brief_date"] == datetime.now().strftime("%Y-%m-%d") and after["briefs"].iloc[0]["data_version"] == after["data_version"]
    readable = int((after["images"]["status"] == "ok").sum())
    assert f"{readable:,} images" in after["briefs"].iloc[0]["brief"]
    shutil.rmtree(ROOT / Path(outcome["results"][0]["file_path"]).parent)


def test_job_file_survives_being_read_while_it_is_replaced(tmp_path):
    """Regression: WinError 5 failed real uploads when the page read job.json during a write."""
    from src.pipeline import read_job, write_job
    write_job(tmp_path, "queued")
    code = f"from src.pipeline import write_job\nfor i in range(500): write_job({str(tmp_path)!r}, 'running', stage=str(i))"
    root = Path(__file__).resolve().parent.parent
    writer = subprocess.Popen([sys.executable, "-c", code], cwd=root, env={**os.environ, "PYTHONPATH": str(root)})
    while writer.poll() is None:
        assert read_job(tmp_path)["state"] in ("queued", "running")
    assert writer.returncode == 0
