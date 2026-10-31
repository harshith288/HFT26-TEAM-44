"""Send one real alert to each candidate model and print the raw reply, to pick a model
that writes readable text. Each call is logged to docs/reports/llm_usage.json at that model's
public price. Used because the first configured model returned gibberish.

Usage: python src/scripts/probe_models.py
"""
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import llm  # noqa: E402
from src.db import connect  # noqa: E402
from src.pipeline import load_config  # noqa: E402

# (model, USD per million input tokens, USD per million output tokens), from the public list.
CANDIDATES = [("Qwen/Qwen2.5-7B-Instruct", 0.17, 0.2), ("mistralai/Mistral-Nemo-Instruct-2407", 0.2175, 0.2475),
              ("NousResearch/Hermes-3-Llama-3.1-8B", 0.2, 0.32)]

base = load_config()
conn = connect(ROOT / base["paths"]["database"])
client = llm.make_client(base)
if client is None:
    sys.exit("NOT RUN: no API key found.")
alert = dict(conn.execute("SELECT * FROM alerts WHERE level = 'CRITICAL' AND species = 'person' LIMIT 1").fetchone())
results = []
for model, price_in, price_out in CANDIDATES:
    cfg = copy.deepcopy(base)
    cfg["llm"].update(model=model, usd_per_million_input_tokens=price_in, usd_per_million_output_tokens=price_out,
                      max_tokens=120)
    before = len(llm.read_usage()["calls"])
    text, error = llm.request_text(llm.alert_prompt(alert), cfg, client, f"model_probe:{model}")
    calls = llm.read_usage()["calls"]
    tokens = calls[-1] if len(calls) > before else None
    passed = bool(text) and llm.has_required_facts(text, alert)
    print(f"\n--- {model} | facts ok: {passed} | error: {error}")
    print("tokens:", tokens)
    print(text)
    results.append({"model": model, "raw_output": text, "passed_facts_check": passed, "error": error, "tokens": tokens})
(ROOT / "docs" / "reports" / "llm_model_probe.json").write_text(json.dumps(
    {"command": " ".join(sys.argv), "timestamp": datetime.now().isoformat(), "n_real_images": 1,
     "hardware": "Featherless API", "results": results}, indent=1))
print("\ntotal spend so far:", llm.read_usage()["estimated_usd"])
print("PROBE_DONE")
