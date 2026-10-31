"""Live LLM check on real alerts: 5 alerts and 1 daily summary, no more.

Prints the raw model output, whether it passed the required-facts check, and the API's
token counts. Goes through the cache, so these calls are paid for only once. Writes
docs/reports/llm_live_test.json. Never prints the API key.

Usage: python src/scripts/test_llm.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import llm  # noqa: E402
from src.db import connect  # noqa: E402
from src.pipeline import load_config  # noqa: E402

cfg = load_config()
conn = connect(ROOT / cfg["paths"]["database"])
client = llm.make_client(cfg)
if client is None:
    sys.exit(f"NOT RUN: {cfg['llm']['api_key_env']} was not found in .env or the environment. Nothing was sent.")
print("client ready | model:", cfg["llm"]["model"])


def latest_tokens(before):
    calls = llm.read_usage()["calls"]
    return calls[-1] if len(calls) > before else None


rows = []
for level, species in (("CRITICAL", "person"), ("CRITICAL", "dog"), ("HIGH", "bobcat"), ("HIGH", "coyote"), ("MEDIUM", "deer")):
    rows.append(dict(conn.execute("SELECT * FROM alerts WHERE level = ? AND species = ? LIMIT 1", (level, species)).fetchone()))

results, failures = [], []
for alert in rows:
    before = len(llm.read_usage()["calls"])
    raw = llm.call_llm(llm.alert_prompt(alert), cfg, conn, client, failures)
    conn.commit()
    passed = bool(raw) and llm.has_required_facts(raw, alert)
    tokens = latest_tokens(before)
    print(f"\n--- {alert['level']} {alert['species']} at {alert['village']}, count {alert['min_observed_count']} | facts ok: {passed}")
    print("tokens:", tokens)
    print(raw)
    results.append({"alert": {k: alert[k] for k in ("species", "level", "village", "min_observed_count")},
                    "raw_output": raw, "passed_facts_check": passed, "tokens": tokens})

date = conn.execute("SELECT substr(b.start_time, 1, 10) AS d, COUNT(*) AS n FROM alerts a JOIN bursts b "
                    "ON a.burst_id = b.burst_id AND a.species = b.species GROUP BY d ORDER BY n DESC LIMIT 1").fetchone()["d"]
facts = llm.daily_facts(conn, date)
fallback = llm.template_summary(date, facts)
before = len(llm.read_usage()["calls"])
raw = llm.call_llm(llm.summary_prompt(fallback), cfg, conn, client, failures, "daily_summary")
conn.commit()
tokens = latest_tokens(before)
print(f"\n--- daily summary {date}, {len(facts)} alerts | keeps alert count: {bool(raw) and str(len(facts)) in raw}")
print("tokens:", tokens)
print(raw)
print("\ntemplate input was:", fallback)
usage = llm.read_usage()
print("\nAPI errors:", failures or "none")
print("total so far:", {k: usage[k] for k in ("total_input_tokens", "total_output_tokens", "estimated_usd")}, "calls:", len(usage["calls"]))
report = {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": len(rows),
          "hardware": "Featherless API", "model": cfg["llm"]["model"], "alerts": results, "api_errors": failures,
          "summary": {"date": date, "raw_output": raw, "tokens": tokens}}
(ROOT / "docs" / "reports" / "llm_live_test.json").write_text(json.dumps(report, indent=1))
print("LLM_TEST_DONE")
