"""Feature 8: alert messages and daily summaries.

Featherless writes the text through its OpenAI-compatible API. Any failure, or output
that leaves out a required fact, falls back to a template, so the LLM can never crash
the app. Outputs are cached in llm_cache so the demo works offline.
"""
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
USAGE_PATH = ROOT / "docs" / "reports" / "llm_usage.json"
LLM_LEVELS = ("MEDIUM", "HIGH", "CRITICAL")  # LOW alerts always use the template
MAX_API_FAILURES = 3


def species_name(species, guess=False):
    if species == "unknown":
        return "unknown animal"
    name = "person (possible poacher)" if species == "person" else str(species).replace("_", " ")
    return f"possibly {name} (low-confidence guess)" if guess else name


def location_of(alert):
    if alert.get("village"):
        return f"{alert['village']} (camera {alert.get('camera_id')})"
    return f"camera {alert['camera_id']}" if alert.get("camera_id") else "an unknown location"


def template_message(alert):
    """Always includes species, count, risk level and location."""
    vehicle = "with a vehicle" if alert["vehicle"] else "no vehicle"
    return (f"{alert['level']} alert: {species_name(alert['species'], alert.get('is_guess'))} at {location_of(alert)}. "
            f"Minimum observed count: {alert['min_observed_count']}. "
            f"Send {alert['rangers']} ranger(s), {vehicle}.")


def alert_prompt(alert):
    return ("Write a two-sentence alert for wildlife rangers using only these facts. "
            f"Species: {species_name(alert['species'], alert.get('is_guess'))}. Minimum observed count: {alert['min_observed_count']}. "
            f"Risk level: {alert['level']}. Location: {location_of(alert)}. Rangers to send: {alert['rangers']}. "
            f"Vehicle needed: {'yes' if alert['vehicle'] else 'no'}. Include the species, count, risk level "
            "and location exactly as given. Do not add any other facts.")


def has_required_facts(text, alert):
    lowered = text.lower()
    names = ("person", "poacher") if alert["species"] == "person" else (str(alert["species"]).replace("_", " ").lower(),)
    return (any(n in lowered for n in names) and str(alert["min_observed_count"]) in text
            and alert["level"].lower() in lowered and location_of(alert).split(" (")[0].lower() in lowered)


def make_client(cfg):
    """An OpenAI-compatible client, or None when there is no key."""
    load_dotenv(ROOT / ".env")
    key = os.environ.get(cfg["llm"]["api_key_env"])
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(base_url=cfg["llm"]["base_url"], api_key=key, timeout=cfg["llm"]["timeout_seconds"], max_retries=0)


def read_usage():
    if USAGE_PATH.exists():
        return json.loads(USAGE_PATH.read_text())
    return {"calls": [], "total_input_tokens": 0, "total_output_tokens": 0, "estimated_usd": 0.0}


def record_usage(reply, cfg, purpose, seconds):
    """Add one call to docs/reports/llm_usage.json. Tokens are the API's own numbers."""
    usage = getattr(reply, "usage", None)
    if usage is None:
        return None
    log = read_usage()
    price_in, price_out = cfg["llm"]["usd_per_million_input_tokens"], cfg["llm"]["usd_per_million_output_tokens"]
    cost = (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1e6
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "purpose": purpose, "input_tokens": usage.prompt_tokens,
             "output_tokens": usage.completion_tokens, "estimated_usd": round(cost, 7), "seconds": round(seconds, 2)}
    log["calls"].append(entry)
    log["total_input_tokens"] += usage.prompt_tokens
    log["total_output_tokens"] += usage.completion_tokens
    log["estimated_usd"] = round(log["estimated_usd"] + cost, 7)
    USAGE_PATH.write_text(json.dumps(log, indent=1))
    return entry


def over_budget(cfg):
    return read_usage()["estimated_usd"] >= cfg["llm"]["budget_stop_usd"]


def request_text(prompt, cfg, client, purpose="alert"):
    """One API call. Returns (text or None, error name or None). Refuses past the budget."""
    if over_budget(cfg):
        return None, "BudgetStop"
    start = time.perf_counter()
    try:
        reply = client.chat.completions.create(model=cfg["llm"]["model"], max_tokens=cfg["llm"]["max_tokens"],
                                               messages=[{"role": "user", "content": prompt}])
        text = (reply.choices[0].message.content or "").strip() or None
    except Exception as e:  # network, auth, rate limit or a bad reply must never stop the app
        return None, type(e).__name__
    record_usage(reply, cfg, purpose, time.perf_counter() - start)
    return text, None


def call_llm(prompt, cfg, conn, client, failures=None, purpose="alert"):
    """Returns text from the cache or the API, or None. Cached text is never regenerated,
    even if it failed the facts check. API errors are added to failures, so callers can
    tell a dead API from text that was merely rejected."""
    model = cfg["llm"]["model"]
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    if conn is not None:
        cached = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (key,)).fetchone()
        if cached:
            return cached[0]
    if client is None:
        return None
    text, error = request_text(prompt, cfg, client, purpose)
    if error:
        print(f"WARNING LLM failed, using template: {error}", flush=True)
        if failures is not None:
            failures.append(error)
        return None
    if text and conn is not None:
        conn.execute("INSERT OR REPLACE INTO llm_cache (cache_key, response, model, created_at) VALUES (?, ?, ?, ?)",
                     (key, text, model, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return text or None


def alert_message(alert, cfg, conn=None, client=None, failures=None):
    """Returns (text, 'llm' or 'template')."""
    if alert["level"] in LLM_LEVELS:
        text = call_llm(alert_prompt(alert), cfg, conn, client, failures)
        if text and has_required_facts(text, alert):
            return text, "llm"
    return template_message(alert), "template"


def fill_alert_messages(conn, cfg, use_llm=True):
    """Write a message on every alert without one. After MAX_API_FAILURES the rest use
    templates, so a dead API costs a few timeouts, not one per alert."""
    client = make_client(cfg) if use_llm else None
    rows = conn.execute("SELECT * FROM alerts WHERE message IS NULL").fetchall()
    failures = []
    for row in rows:
        alert = dict(row)
        text, source = alert_message(alert, cfg, conn, client, failures)
        if len(failures) >= MAX_API_FAILURES:  # rejected text does not count as a failure
            client = None
        conn.execute("UPDATE alerts SET message = ?, message_source = ? WHERE alert_id = ?",
                     (text, source, alert["alert_id"]))
        conn.commit()  # keep finished messages if a long run is interrupted
    return len(rows)


def daily_facts(conn, date):
    rows = conn.execute("SELECT a.species, a.level, a.village, a.min_observed_count FROM alerts a JOIN bursts b "
                        "ON a.burst_id = b.burst_id AND a.species = b.species "
                        "WHERE substr(b.start_time, 1, 10) = ?", (date,)).fetchall()
    return [dict(r) for r in rows]


def template_summary(date, facts):
    if not facts:
        return f"Daily summary for {date}: no alerts."
    levels = {}
    for f in facts:
        levels[f["level"]] = levels.get(f["level"], 0) + 1
    species = sorted({species_name(f["species"]) for f in facts})
    villages = sorted({f["village"] or "unmapped cameras" for f in facts})
    breakdown = ", ".join(f"{levels[k]} {k}" for k in ("CRITICAL", "HIGH", "MEDIUM", "LOW") if k in levels)
    return (f"Daily summary for {date}: {len(facts)} alert(s) ({breakdown}). "
            f"Species: {', '.join(species)}. Locations: {', '.join(villages)}.")


def summary_prompt(fallback):
    return f"Rewrite this as a short daily report for rangers. Use only these facts, and keep every number: {fallback}"


def daily_summary(conn, cfg, date, client=None, failures=None):
    """Returns (text, 'llm' or 'template')."""
    facts = daily_facts(conn, date)
    fallback = template_summary(date, facts)
    if not facts:
        return fallback, "template"
    text = call_llm(summary_prompt(fallback), cfg, conn, client, failures, "daily_summary")
    return (text, "llm") if text and str(len(facts)) in text else (fallback, "template")


def fill_daily_summaries(conn, cfg, use_llm=True):
    client = make_client(cfg) if use_llm else None
    dates = [r[0] for r in conn.execute("SELECT DISTINCT substr(start_time, 1, 10) FROM bursts "
                                        "WHERE start_time IS NOT NULL")]
    failures = []
    for date in dates:
        text, source = daily_summary(conn, cfg, date, client, failures)
        if len(failures) >= MAX_API_FAILURES:
            client = None
        conn.execute("INSERT OR REPLACE INTO daily_summaries (summary_date, summary, summary_source, created_at) "
                     "VALUES (?, ?, ?, ?)", (date, text, source, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    return len(dates)
