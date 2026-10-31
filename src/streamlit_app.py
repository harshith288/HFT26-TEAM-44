"""WildWatch dashboard. Works fully offline: it reads only SQLite and docs/reports/.

Run: powershell -ExecutionPolicy Bypass -File src/scripts/run.ps1 dashboard
"""
import base64
import io
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "src" / "config.yaml").read_text())
DB_PATH = Path(os.environ.get("WILDWATCH_DB", ROOT / CONFIG["paths"]["database"]))
REPORTS = ROOT / "docs" / "reports"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated reference palette, categorical slots 1 and 2
LEVEL_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
LEVEL_ICON = {"CRITICAL": "🛑", "HIGH": "⚠️", "MEDIUM": "🔶", "LOW": "ℹ️"}
TARGET = 90.0
NO_CAMERA = "Unknown camera"
JOB_TIMEOUT_SECONDS = 300

st.set_page_config(page_title="WildWatch", page_icon="🐾", layout="wide")


@st.cache_data(ttl=30)
def load_table(query):
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def report(name):
    """A report as a dict, or None if it has not been produced."""
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(max_entries=200)
def thumbnail_html(path, width=420):
    """An inline <img>. Streamlit's media route returned empty files for these images, and
    inline thumbnails also keep the page small. Returns None if the file cannot be read."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((width, width))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=80)
    except OSError:
        return None
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{data}" style="width:100%;border-radius:6px" alt="Camera photo">'


def start_upload_job(files, camera):
    """Save uploads and start the real pipeline in another process, so torch never loads here."""
    folder = ROOT / CONFIG["paths"]["incoming"] / "uploads" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    for file in files:
        (folder / Path(file.name).name).write_bytes(file.getvalue())
    (folder / "job.json").write_text(json.dumps({"state": "queued", "results": [], "error": None}))
    command = [sys.executable, "-m", "src.pipeline", "--job-dir", str(folder)]
    if camera != NO_CAMERA:
        command += ["--camera", camera]
    with open(folder / "job.log", "w") as log:
        subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    st.session_state.update(job=str(folder), job_started=time.time(), job_seen=False)


def show_upload_result(result):
    name = result["file"]
    if result["status"] == "unreadable":
        st.error(f"{name}: could not be read. {result['error']}")
        return
    if result.get("duplicate_of"):
        st.info(f"{name}: this photo was already analysed, so the earlier result is shown.")
    alert = result["alert"]
    if alert:
        st.success(f"{name}: {LEVEL_ICON[alert['level']]} {alert['level']} · {alert['species'].replace('_', ' ')} · "
                   f"minimum observed count {alert['min_observed_count']}")
        st.write(alert["message"])
    else:
        animals = sorted({d["species"] or d["category"] for d in result["detections"] if d["confidence"] >= 0.2})
        found = ", ".join(animals) if animals else "nothing confirmed"
        st.info(f"{name}: no alert. Decision: {result['triage']}. Found: {found}.")


@st.fragment(run_every="2s")
def upload_job_panel():
    """Poll the job's status file, then reload the whole page once when it finishes."""
    folder = st.session_state.get("job")
    if not folder:
        return
    job = json.loads((Path(folder) / "job.json").read_text())
    waited = time.time() - st.session_state["job_started"]
    if job["state"] in ("queued", "running"):
        if waited > JOB_TIMEOUT_SECONDS:
            st.error("The background job did not finish. See job.log in its upload folder.")
        else:
            st.info(f"Analysing in the background: {waited:.0f}s so far. This page updates by itself.")
        return
    if job["state"] == "failed":
        st.error(f"The background job failed: {job['error']}")
        return
    if not st.session_state["job_seen"]:
        st.session_state["job_seen"] = True
        st.cache_data.clear()
        st.rerun(scope="app")
    st.subheader("Upload results")
    st.caption(f"Analysis finished: {len(job['results'])} file(s).")
    for result in job["results"]:
        show_upload_result(result)


def tidy_chart(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), legend_title_text="",
                      bargap=0.35, hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


def show_alert(row):
    """Level is shown as an icon plus a word, never by color alone."""
    level = row["level"]
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            species = "person: possible poacher" if row["possible_poacher"] else row["species"].replace("_", " ")
            st.markdown(f"**{LEVEL_ICON[level]} {level} · {species}**")
            st.write(row["message"] or "No message yet.")
            place = row["village"] or "Unmapped camera"
            st.caption(f"{place} · camera {row['camera_id']} · minimum observed count {row['min_observed_count']} · "
                       f"danger {row['danger_score']:.2f} · rangers {row['rangers']} · "
                       f"vehicle {'yes' if row['vehicle'] else 'no'} · message from {row['message_source']}")
        with right:
            html = thumbnail_html(str(ROOT / str(row["file_path"])))
            if html:
                st.markdown(html, unsafe_allow_html=True)


if not DB_PATH.exists():
    st.title("WildWatch")
    st.warning("No results yet. Run `src/scripts/run.ps1 pipeline` to fill the database, then reload this page.")
    st.stop()

images = load_table("SELECT * FROM images")
bursts = load_table("SELECT * FROM bursts")
alerts = load_table("""
    SELECT a.*, b.start_time, b.is_night,
           (SELECT i.file_path FROM images i JOIN detections d ON d.image_id = i.image_id
            WHERE i.seq_id = a.burst_id ORDER BY d.confidence DESC LIMIT 1) AS file_path
    FROM alerts a JOIN bursts b ON a.burst_id = b.burst_id AND a.species = b.species""")
summaries = load_table("SELECT * FROM daily_summaries ORDER BY summary_date DESC")
camera_village = {c: name for name, v in CONFIG["villages"].items() for c in v["cameras"]}

title, refresh = st.columns([5, 1])
title.title("WildWatch")
title.caption("Camera-trap monitoring and village intrusion alerts, from Caltech Camera Traps images.")
if refresh.button("Refresh data", width="stretch"):
    st.cache_data.clear()
    st.rerun()

filter_a, filter_b = st.columns(2)
villages = filter_a.multiselect("Village", sorted(CONFIG["villages"]), placeholder="All villages")
levels = filter_b.multiselect("Alert level", LEVEL_ORDER, placeholder="All levels")
if villages:
    cameras = [c for c, v in camera_village.items() if v in villages]
    images = images[images["camera_id"].isin(cameras)]
    bursts = bursts[bursts["camera_id"].isin(cameras)]
    alerts = alerts[alerts["camera_id"].isin(cameras)]
if levels:
    alerts = alerts[alerts["level"].isin(levels)]

readable = images[images["status"] == "ok"]
review = readable[readable["triage"] == "review"]
serious = alerts[alerts["level"].isin(["MEDIUM", "HIGH", "CRITICAL"])]
tier_bursts = readable.groupby("triage")["seq_id"].nunique()
all_bursts = readable["seq_id"].nunique()


def tier_tile(column, label, tier, help_text):
    count = int(tier_bursts.get(tier, 0))
    share = 100 * count / all_bursts if all_bursts else 0.0
    column.metric(label, f"{count:,}", f"{share:.1f}% of bursts", delta_color="off", delta_arrow="off", help=help_text)


tiles = st.columns(5)
tiles[0].metric("Images analysed", f"{len(readable):,}", f"{all_bursts:,} bursts", delta_color="off", delta_arrow="off")
tier_tile(tiles[1], "Auto-discarded", "discard", f"Best detector score below {CONFIG['triage']['discard_below']}.")
tier_tile(tiles[2], "Needs review", "review", f"Best score from {CONFIG['triage']['discard_below']} to "
          f"{CONFIG['triage']['keep_min']}: a person decides.")
tier_tile(tiles[3], "Kept", "keep", f"Best score of {CONFIG['triage']['keep_min']} or higher.")
tiles[4].metric("Alerts at MEDIUM or above", f"{len(serious):,}")
st.caption("Triage works per burst. These shares describe this sample: train has no empty images and val empties "
           "were oversampled, so they are not natural CCT rates.")

tab_alerts, tab_upload, tab_review, tab_species, tab_activity, tab_cameras, tab_model = st.tabs(
    ["Live alerts", "Upload", "Needs review", "Species", "Activity", "Cameras", "Model accuracy"])

with tab_upload:
    st.subheader("Upload photos")
    st.caption("The full pipeline runs in a separate background process. Results and alerts appear here by themselves.")
    camera = st.selectbox("Camera", [NO_CAMERA] + sorted(camera_village, key=int))
    uploads = st.file_uploader("Camera photos", accept_multiple_files=True)
    if st.button("Analyse photos", disabled=not uploads):
        start_upload_job(uploads, camera)
    upload_job_panel()

with tab_review:
    st.subheader("Needs review")
    st.caption(f"Bursts whose best detector score is between {CONFIG['triage']['discard_below']} and "
               f"{CONFIG['triage']['keep_min']}. These are too doubtful to keep or to throw away, so a person decides.")
    if review.empty:
        st.info("Nothing is waiting for review.")
    else:
        best = review.sort_values("top_confidence", ascending=False).groupby("seq_id", as_index=False).first()
        best = best.sort_values("top_confidence", ascending=False)
        st.caption(f"Showing {min(12, len(best))} of {len(best)} bursts, most likely animals first.")
        columns = st.columns(3)
        for i, (_, row) in enumerate(best.head(12).iterrows()):
            with columns[i % 3], st.container(border=True):
                html = thumbnail_html(str(ROOT / str(row["file_path"])))
                if html:
                    st.markdown(html, unsafe_allow_html=True)
                place = camera_village.get(str(row["camera_id"]), "Unmapped")
                st.caption(f"{place} · camera {row['camera_id']} · best score {row['top_confidence']:.3f}")

with tab_alerts:
    poachers = alerts[alerts["possible_poacher"] == 1]
    if len(poachers):
        st.warning(f"{len(poachers)} possible poacher alert(s). A person detection is not proof: on this data "
                   "it also fires on hikers, drivers and parts of trucks. A ranger should check the photo.")
    order = alerts.assign(rank=alerts["level"].map(LEVEL_ORDER.index)).sort_values(["rank", "danger_score"],
                                                                                   ascending=[True, False])
    shown = order.head(20)
    st.caption(f"Showing {len(shown)} of {len(alerts)} alerts, most serious first.")
    if shown.empty:
        st.info("No alerts match these filters.")
    for _, row in shown.iterrows():
        show_alert(row)
    if len(summaries):
        st.subheader("Daily summary")
        date = st.selectbox("Date", summaries["summary_date"].tolist())
        chosen = summaries[summaries["summary_date"] == date].iloc[0]
        st.write(chosen["summary"])
        st.caption(f"Written by: {chosen['summary_source']}")

with tab_species:
    counts = bursts[~bursts["species"].isin(["vehicle"])].groupby("species").size().sort_values().reset_index(name="bursts")
    st.subheader("Species distribution")
    if counts.empty:
        st.info("No data for these filters.")
    else:
        fig = px.bar(counts, x="bursts", y="species", orientation="h", text="bursts",
                     color_discrete_sequence=[BLUE], labels={"bursts": "Bursts", "species": ""})
        st.plotly_chart(tidy_chart(fig, 30 * len(counts) + 80), width="stretch")
    st.subheader("Animals seen over time")
    st.caption("Monthly sum of minimum observed counts. Sample images only; camera 120's clock is unreliable.")
    dated = bursts.dropna(subset=["start_time"]).copy()
    dated = dated[~dated["species"].isin(["vehicle"])]
    if len(dated):
        dated["month"] = dated["start_time"].str[:7]
        monthly = dated.groupby("month")["min_observed_count"].sum().reset_index()
        fig = px.line(monthly, x="month", y="min_observed_count", markers=True, color_discrete_sequence=[BLUE],
                      labels={"month": "", "min_observed_count": "Animals (minimum observed)"})
        fig.update_traces(line_width=2, marker_size=8)
        st.plotly_chart(tidy_chart(fig), width="stretch")

with tab_activity:
    st.subheader("Day and night activity")
    st.warning("The sample was balanced across day and night on purpose, so these bars describe the sample, "
               "not natural behaviour. Night comes from the pixels (infrared or dark scene), not the clock.")
    animals = bursts[~bursts["species"].isin(["vehicle", "person", "unknown"])].copy()
    if animals.empty:
        st.info("No data for these filters.")
    else:
        animals["period"] = animals["is_night"].map({1: "Night", 0: "Day"})
        table = animals.groupby(["species", "period"]).size().reset_index(name="bursts")
        fig = px.bar(table, x="species", y="bursts", color="period", barmode="group",
                     color_discrete_map={"Day": BLUE, "Night": ORANGE}, labels={"species": "", "bursts": "Bursts"})
        st.plotly_chart(tidy_chart(fig), width="stretch")

with tab_cameras:
    st.subheader("Per-camera statistics")
    if readable.empty:
        st.info("No data for these filters.")
    else:
        stats = readable.groupby("camera_id").agg(images=("image_id", "size"), removed=("is_empty", "sum"),
                                                  night=("is_night", "sum")).reset_index()
        stats["bursts"] = stats["camera_id"].map(bursts.groupby("camera_id")["burst_id"].nunique()).fillna(0).astype(int)
        stats["alerts"] = stats["camera_id"].map(alerts.groupby("camera_id").size()).fillna(0).astype(int)
        stats["village"] = stats["camera_id"].map(camera_village).fillna("Unmapped")
        stats["zone factor"] = stats["village"].map(lambda v: CONFIG["villages"].get(v, {}).get(
            "zone_factor", CONFIG["default_zone_factor"]))
        stats["% removed"] = (100 * stats["removed"] / stats["images"]).round(1)
        columns = ["camera_id", "village", "zone factor", "images", "% removed", "night", "bursts", "alerts"]
        st.dataframe(stats.sort_values("alerts", ascending=False)[columns], hide_index=True, width="stretch")
        st.caption("Villages and zone factors are fictional, set in src/config.yaml.")

with tab_model:
    st.subheader("How good is the model?")
    improvement = report("improvement.json")
    final = report("final_eval.json")
    if final:
        result, triage = final["groups"]["all"]["species"], final["groups"]["all"]["triage"]
        confirmed = final["groups"]["all"]["species_confirmed_by_box"]
        st.subheader("Final test results")
        st.caption(f"Touched once, with everything frozen at commit {final['frozen_at_commit']}. "
                   f"{final['n_scored']} single-species images from {final['n_real_images']} test images.")
        a, b, c, d = st.columns(4)
        low, high = result["accuracy_95_range"]
        a.metric("Test-set accuracy on Caltech species", f"{result['accuracy']:.1f}%", f"95% range {low} to {high}", delta_color="off", delta_arrow="off")
        b.metric("Macro precision", f"{result['macro_precision']:.1f}%")
        c.metric("Animal bursts lost", f"{triage['animal_bursts_lost']} of {triage['n_animal_bursts']}",
                 f"{triage['pct_animal_bursts_lost']:.2f}%", delta_color="off", delta_arrow="off")
        d.metric("Accuracy, confirmed by a box", f"{confirmed['accuracy']:.1f}%")
        if result["accuracy"] < TARGET:
            st.warning(f"The {TARGET:.0f}% accuracy target is not reached on the full definition. All "
                       f"{final['missed_images']} missed images are in bursts with no ground-truth box, so they may be "
                       "label noise, though that is not proven.")
        st.subheader("Test results by group")
        rows = []
        for key, label in (("cis_test", "Seen cameras"), ("trans_test", "Unseen cameras"), ("day", "Day"),
                           ("ir_night", "IR-night"), ("color_night", "Color-night")):
            g = final["groups"][key]
            rows.append({"Group": label, "Images": g["species"]["n"], "Accuracy %": round(g["species"]["accuracy"], 1),
                         "95% range": str(g["species"]["accuracy_95_range"]),
                         "Macro precision %": round(g["species"]["macro_precision"], 1),
                         "Count MAE": round(g["count"]["mae"], 3), "Bursts counted": g["count"]["n_bursts"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Small groups have wide ranges, so day and night differences here may not be real.")
        st.subheader("Validation detail")
    else:
        st.info("Test splits: not measured yet. Every number here comes from validation images.")
    if improvement:
        fresh, tuning = improvement["fresh_val"], improvement["tuning_val"]
        accuracy = fresh["all"]["accuracy"]
        reached = accuracy >= TARGET and fresh["all"]["macro_precision"] >= TARGET
        a, b, c = st.columns(3)
        a.metric("Species accuracy, fresh images", f"{accuracy:.1f}%", f"{accuracy - TARGET:+.1f} vs {TARGET:.0f}% target")
        b.metric("Macro precision, fresh images", f"{fresh['all']['macro_precision']:.1f}%")
        c.metric("Accuracy, animal confirmed by a box", f"{fresh['bursts_with_a_ground_truth_box']['accuracy']:.1f}%")
        if not reached:
            st.warning(f"The {TARGET:.0f}% target is not reached on the full definition. "
                       f"{fresh['missed_with_a_ground_truth_box']} of {fresh['missed_images']} missed images have a "
                       "ground-truth box. The rest may be label noise, because CCT labels whole bursts.")
        st.caption(f"{fresh['n_scored']:,} fresh single-species images, rule: {improvement['rule']}.")
        rows = []
        for name, data in (("Fresh images", fresh), ("Tuning val", tuning)):
            for group, label in (("all", "All"), ("day", "Day"), ("ir_night", "IR-night"), ("color_night", "Color-night"),
                                 ("cis_val", "Seen cameras"), ("trans_val", "New cameras")):
                m = data[group]
                rows.append({"Set": name, "Group": label, "Images": m["n"], "Accuracy %": round(m.get("accuracy", 0), 1),
                             "Macro precision %": round(m.get("macro_precision", 0), 1)})
        st.subheader("Species accuracy by group")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Color-night holds very few images, so its numbers mean little. "
                   "New cameras is a single val camera, so it is not proof the model generalizes.")

    discard = report("discard_tuning.json")
    if discard and discard["fresh_val_after"]:
        st.subheader("Empty filter on fresh images")
        rows = [{"Rule": name, "Discarded": m["discarded_bursts"], "Needs review": m["review_bursts"], "Kept": m["kept_bursts"],
                 "Animal bursts lost": m["animal_bursts_lost"], "Animal bursts lost %": round(m["pct_animal_bursts_lost"], 2)}
                for name, m in (("Before: one threshold", discard["fresh_val_before"]),
                                ("After: discard and review", discard["fresh_val_after"]))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Discard threshold {discard['chosen_discard_below']}, chosen on tuning val only.")

    day_night = report("phase3_day_night.json")
    if day_night:
        st.subheader("Day vs night, including failures")
        rows = []
        for group, label in (("day", "Day"), ("ir_night", "IR-night"), ("color_night", "Color-night")):
            m = day_night["groups"][group]
            rows.append({"Group": label, "Images": m["filter"]["n_images"], "Empty precision": m["filter"]["precision_empty"],
                         "Empty recall": m["filter"]["recall_empty"], "Empty F1": m["filter"]["f1_empty"],
                         "Animal-labeled frames discarded %": m["filter"]["pct_animal_frames_discarded_all_labels"],
                         "Count MAE": m["count"]["mae_max_rule"], "Bursts counted": m["count"]["n_bursts"]})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        st.caption("Frame-level, on tuning val. The empty filter is clearly weaker at night.")

    species = report("phase2_species.json")
    if species:
        st.subheader("Models compared on val")
        rows = [{"Candidate": k, "Macro-F1 %": round(v["all"]["macro_f1"], 1), "Accuracy %": round(v["all"]["top1_accuracy"], 1)}
                for k, v in sorted(species["candidates"].items(), key=lambda kv: -kv[1]["all"]["macro_f1"])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        flagged = [k for k, v in species["overfitting"].items() if v["overfit_flag"]]
        if flagged:
            st.warning("Overfitting flagged on the linear probes: " + ", ".join(
                f"{k} ({species['overfitting'][k]['accuracy_gap_points']:.0f} point train/val gap)" for k in flagged))
    usage = report("llm_usage.json")
    if usage:
        st.subheader("LLM use")
        st.caption(f"{len(usage['calls'])} calls, {usage['total_input_tokens']:,} input and "
                   f"{usage['total_output_tokens']:,} output tokens, estimated ${usage['estimated_usd']:.4f}. "
                   f"Calls stop at ${CONFIG['llm']['budget_stop_usd']:.0f}.")
    matrix = REPORTS / ("confusion_matrix_test.png" if final else "confusion_matrix_val.png")
    if matrix.exists():
        st.subheader("Confusion matrix")
        st.markdown(thumbnail_html(str(matrix), 1400), unsafe_allow_html=True)
        st.caption("Burst vote on the test splits." if final else "Image-level rule on val, before burst voting.")
    st.subheader("Known limitations")
    st.markdown("- CCT holds US species, not Indian ones.\n- Counts are a minimum observed count.\n"
                "- CCT has no person label, so person detection has no measured accuracy.\n"
                "- Risk factors and thresholds are placeholders in src/config.yaml.")
