"""Phase 4: LLM fallback and the dashboard.

MOCKS: FakeClient below is a mock of the LLM API, allowed for the LLM fallback tests
only. "API down" uses a real client pointed at a closed local port, not a mock.
Dashboard tests run the real app on the real pipeline database.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from openai import OpenAI

from src import llm
from src.db import init_db

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
ALERT = {"species": "coyote", "min_observed_count": 3, "level": "HIGH", "village": "Ambernook", "camera_id": "33",
         "rangers": 2, "vehicle": True}
ALERTS = [ALERT, {**ALERT, "species": "person", "level": "CRITICAL"}, {**ALERT, "village": None, "camera_id": None},
          {**ALERT, "species": "mountain_lion", "min_observed_count": 12}]


class FakeClient:
    """MOCK of the OpenAI client. Returns fixed text, or raises if given an exception."""

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, 0
        self.chat = self
        self.completions = self

    def create(self, **_):
        self.calls += 1
        if self.error:
            raise self.error
        message = type("M", (), {"content": self.reply})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


def dead_client():
    """A real client aimed at a closed port, so the failure is a real connection error."""
    return OpenAI(base_url="http://127.0.0.1:9/v1", api_key="test", timeout=1, max_retries=0)


def required_facts_present(text, alert):
    return (str(alert["species"]).replace("_", " ") in text and str(alert["min_observed_count"]) in text
            and alert["level"] in text and (alert["village"] or "unknown location") in text)


@pytest.fixture
def conn(tmp_path):
    connection = init_db(tmp_path / "t.db")
    yield connection
    connection.close()


@pytest.mark.parametrize("alert", ALERTS)
def test_template_always_has_required_facts(alert):
    assert required_facts_present(llm.template_message(alert), alert)


@pytest.mark.parametrize("alert", ALERTS)
def test_real_connection_failure_falls_back_to_template(alert, conn):
    text, source = llm.alert_message(alert, CONFIG, conn, dead_client())
    assert source == "template" and required_facts_present(text, alert)


def test_no_key_uses_template(monkeypatch, conn):
    monkeypatch.delenv(CONFIG["llm"]["api_key_env"], raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda *_: None)
    assert llm.make_client(CONFIG) is None
    assert llm.alert_message(ALERT, CONFIG, conn, None)[1] == "template"


@pytest.mark.parametrize("error", [TimeoutError("slow"), ValueError("bad"), KeyError("x"), RuntimeError("500")])
def test_any_api_error_falls_back(error, conn):
    assert llm.alert_message(ALERT, CONFIG, conn, FakeClient(error=error))[1] == "template"


@pytest.mark.parametrize("reply", ["", "   ", "All clear today.", "HIGH coyote alert.", "3 coyotes at Ambernook."])
def test_empty_or_incomplete_llm_output_is_rejected(reply, conn):
    text, source = llm.alert_message(ALERT, CONFIG, conn, FakeClient(reply))
    assert source == "template" and required_facts_present(text, ALERT)


def test_good_llm_output_is_used_and_cached(conn):
    reply = "HIGH alert at Ambernook: 3 coyote seen. Send 2 rangers with a vehicle."
    client = FakeClient(reply)
    assert llm.alert_message(ALERT, CONFIG, conn, client) == (reply, "llm")
    assert llm.alert_message(ALERT, CONFIG, conn, client) == (reply, "llm")
    assert client.calls == 1, "second call must come from the cache"
    assert llm.alert_message(ALERT, CONFIG, conn, None) == (reply, "llm"), "cache must work offline"


def test_low_alerts_never_call_the_api(conn):
    client = FakeClient("anything")
    assert llm.alert_message({**ALERT, "level": "LOW"}, CONFIG, conn, client)[1] == "template"
    assert client.calls == 0


def test_daily_summary_template_with_and_without_alerts(conn):
    text, source = llm.daily_summary(conn, CONFIG, "2011-01-01", dead_client())
    assert source == "template" and "no alerts" in text
    conn.execute("INSERT INTO bursts (burst_id, species, start_time, n_frames, min_observed_count) "
                 "VALUES ('b', 'coyote', '2011-01-01 02:00:00', 3, 2)")
    conn.execute("INSERT INTO alerts (burst_id, species, village, min_observed_count, danger_score, level, rangers, "
                 "vehicle, possible_poacher, created_at) VALUES ('b', 'coyote', 'Ambernook', 2, 1.8, 'HIGH', 2, 1, 0, 'now')")
    text, source = llm.daily_summary(conn, CONFIG, "2011-01-01", dead_client())
    assert source == "template"
    assert all(w in text for w in ("coyote", "HIGH", "Ambernook", "1 alert"))


def test_daily_summary_rejects_output_that_drops_the_numbers(conn):
    conn.execute("INSERT INTO bursts (burst_id, species, start_time, n_frames, min_observed_count) "
                 "VALUES ('b', 'coyote', '2011-01-01 02:00:00', 3, 2)")
    conn.execute("INSERT INTO alerts (burst_id, species, min_observed_count, danger_score, level, rangers, vehicle, "
                 "possible_poacher, created_at) VALUES ('b', 'coyote', 2, 1.8, 'HIGH', 2, 1, 0, 'now')")
    assert llm.daily_summary(conn, CONFIG, "2011-01-01", FakeClient("Nothing to report"))[1] == "template"


def test_repeated_failures_stop_further_api_calls(conn, monkeypatch):
    client = FakeClient(error=RuntimeError("down"))
    monkeypatch.setattr(llm, "make_client", lambda cfg: client)
    for burst in "abcde":
        conn.execute(f"INSERT INTO bursts (burst_id, species, n_frames, min_observed_count) VALUES ('{burst}', 'coyote', 1, 1)")
        conn.execute("INSERT INTO alerts (burst_id, species, camera_id, min_observed_count, danger_score, level, rangers, "
                     f"vehicle, possible_poacher, created_at) VALUES ('{burst}', 'coyote', '33', 1, 1.0, 'HIGH', 2, 1, 0, 'n')")
    assert llm.fill_alert_messages(conn, CONFIG) == 5
    assert client.calls == llm.MAX_API_FAILURES
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE message IS NULL").fetchone()[0] == 0


class PricedClient(FakeClient):
    """MOCK that also reports token usage, for the budget tests."""

    def create(self, **_):
        self.calls += 1
        message = type("M", (), {"content": "HIGH alert: 3 coyote at Ambernook."})
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 50})
        return type("R", (), {"choices": [type("C", (), {"message": message})], "usage": usage})


def test_usage_is_logged_with_the_api_token_counts(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "USAGE_PATH", tmp_path / "usage.json")
    llm.alert_message(ALERT, CONFIG, conn, PricedClient())
    log = llm.read_usage()
    assert log["total_input_tokens"] == 100 and log["total_output_tokens"] == 50
    assert log["estimated_usd"] == pytest.approx((100 * 0.17 + 50 * 0.2) / 1e6)
    assert len(log["calls"]) == 1 and "prompt" not in log["calls"][0]


def test_budget_stop_uses_the_template_and_makes_no_call(conn, tmp_path, monkeypatch):
    path = tmp_path / "usage.json"
    monkeypatch.setattr(llm, "USAGE_PATH", path)
    path.write_text(json.dumps({"calls": [], "total_input_tokens": 0, "total_output_tokens": 0, "estimated_usd": 15.0}))
    client = PricedClient()
    text, source = llm.alert_message(ALERT, CONFIG, conn, client)
    assert source == "template" and client.calls == 0
    assert required_facts_present(text, ALERT)


def test_cached_text_is_never_paid_for_twice_even_if_rejected(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "USAGE_PATH", tmp_path / "usage.json")
    client = FakeClient("Garbled reply")
    assert llm.alert_message(ALERT, CONFIG, conn, client)[1] == "template"
    assert llm.alert_message(ALERT, CONFIG, conn, client)[1] == "template"
    assert client.calls == 1


def test_person_alert_may_say_poacher():
    alert = {**ALERT, "species": "person", "level": "CRITICAL"}
    assert llm.has_required_facts("CRITICAL: 3 poachers at Ambernook", alert)


def test_rejected_text_does_not_turn_the_llm_off(conn, monkeypatch):
    """Text failing the facts check is not an API failure, so later alerts still try the LLM."""
    client = FakeClient("Nothing useful here.")
    monkeypatch.setattr(llm, "make_client", lambda cfg: client)
    conn.execute("INSERT INTO bursts (burst_id, species, n_frames, min_observed_count) VALUES ('a', 'coyote', 1, 1)")
    conn.execute("INSERT INTO bursts (burst_id, species, n_frames, min_observed_count) VALUES ('b', 'dog', 1, 1)")
    for burst, species in (("a", "coyote"), ("b", "dog")):
        conn.execute("INSERT INTO alerts (burst_id, species, camera_id, min_observed_count, danger_score, level, rangers, "
                     f"vehicle, possible_poacher, created_at) VALUES ('{burst}', '{species}', '33', 1, 1.0, 'HIGH', 2, 1, 0, 'n')")
    llm.fill_alert_messages(conn, CONFIG)
    assert client.calls == 2
    assert {r[0] for r in conn.execute("SELECT message_source FROM alerts")} == {"template"}


def test_no_key_is_in_tracked_config():
    assert "api_key:" not in (ROOT / "src" / "config.yaml").read_text()


# --- final results ---

def test_final_evaluation_cannot_be_run_twice():
    done = subprocess.run([sys.executable, str(ROOT / "src" / "scripts" / "final_eval.py")], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode != 0 and "already been evaluated" in done.stderr


def test_readme_numbers_match_the_final_report():
    final = json.loads((ROOT / "docs" / "reports" / "final_eval.json").read_text())
    s, t = final["groups"]["all"]["species"], final["groups"]["all"]["triage"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in (f"**{s['accuracy']:.1f}%**", f"**{s['macro_precision']:.1f}%**", f"{s['macro_f1']:.1f}%",
                 f"{t['animal_bursts_lost']} of {t['n_animal_bursts']} ({t['pct_animal_bursts_lost']:.2f}%)",
                 f"{t['auto_discarded']} ({t['pct_auto_discarded']:.1f}%)", f"{t['needs_review']} ({t['pct_needs_review']:.1f}%)",
                 f"{t['kept']} ({t['pct_kept']:.1f}%)", final["frozen_at_commit"]):
        assert text in readme, text
    assert "70%" not in readme


def test_usage_log_never_holds_prompts_or_keys():
    text = (ROOT / "docs" / "reports" / "llm_usage.json").read_text()
    assert "rc_" not in text and "prompt\"" not in text


# --- dashboard ---

def test_dashboard_checks_pass_in_their_own_process():
    """pyarrow crashes with an access violation when it shares a process with torch, and
    the real dashboard never loads torch. So tests/dashboard_checks.py runs separately,
    as the app does. Any failure there fails this test."""
    done = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests" / "dashboard_checks.py"), "-q",
                           "--tb=short"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-3000:]
