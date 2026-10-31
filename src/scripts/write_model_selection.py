"""Generate docs/reports/model_selection.md from the saved reports, so no number is typed by hand.

Usage: python src/scripts/write_model_selection.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load(name):
    return json.loads((ROOT / "docs" / "reports" / name).read_text())


species = load("phase2_species.json")
detection = load("phase2_detection.json")
improvement = load("improvement.json")
discard = load("discard_tuning.json")
final = load("final_eval.json")
fresh = improvement["fresh_val"]
test = final["groups"]["all"]
t, s, c, b = test["triage"], test["species"], test["count"], test["boxes"]
confirmed = test["species_confirmed_by_box"]
after = discard["fresh_val_after"]

lines = [
    "# Model selection and final results", "",
    f"- Command: `python {' '.join(sys.argv)}`",
    f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"- Hardware: {final['hardware']}",
    f"- Real images: {species['n_real_crops_train']} train crops, {detection['n_real_images']} tuning val images, "
    f"{fresh['n_real_images']} fresh val images, {final['n_real_images']} test images",
    f"- Everything was frozen at commit {final['frozen_at_commit']} before the test splits were opened, once.", "",
    "## Species model, chosen on val only", "",
    "All candidates classified the same MDv6 crops.", "",
    "| Candidate | Macro-F1 % | Accuracy % |", "|---|---|---|",
]
for name, r in sorted(species["candidates"].items(), key=lambda kv: -kv[1]["all"]["macro_f1"]):
    lines.append(f"| {name} | {r['all']['macro_f1']:.1f} | {r['all']['top1_accuracy']:.1f} |")
lines += ["", f"**Chosen: {species['best_by_val_macro_f1']}**, then a burst vote at confidence "
          f"{improvement['rule'].split('>= ')[-1]}.", ""]
lines += ["Overfitting check on the linear probes, which were not chosen:", "",
          "| Probe | Train accuracy % | Val accuracy % | Gap, points | Flag over 10 |", "|---|---|---|---|---|"]
for name, g in species["overfitting"].items():
    lines.append(f"| {name} | {g['train_accuracy']:.1f} | {g['val_accuracy']:.1f} | {g['accuracy_gap_points']:.1f} | "
                 f"{'yes' if g['overfit_flag'] else 'no'} |")
lines += ["", "## Thresholds, chosen on tuning val only", "",
          f"- Count threshold: {detection['chosen_count_threshold']}",
          f"- Discard below: {discard['chosen_discard_below']}; keep from {discard['keep_min']}",
          f"- Rule for discard: {discard['rule']}", "",
          "## Confirmation on fresh val", "",
          f"{fresh['n_scored']} single-species images at natural proportions: accuracy {fresh['all']['accuracy']:.1f}%, "
          f"macro precision {fresh['all']['macro_precision']:.1f}%. Discard rule: {after['animal_bursts_lost']} of "
          f"{after['n_animal_bursts']} animal bursts lost ({after['pct_animal_bursts_lost']:.2f}%), which missed the 1% target.", "",
          "## Final test results", "",
          "| Metric | Result |", "|---|---|",
          f"| Species accuracy | {s['accuracy']:.1f}% (95% range {s['accuracy_95_range'][0]} to {s['accuracy_95_range'][1]}), {s['n']} images |",
          f"| Macro precision | {s['macro_precision']:.1f}% |", f"| Macro-F1 | {s['macro_f1']:.1f}% |",
          f"| Accuracy, bursts with a ground-truth box | {confirmed['accuracy']:.1f}%, {confirmed['n']} images |",
          f"| Missed images with a ground-truth box | {final['missed_with_a_ground_truth_box']} of {final['missed_images']} |",
          f"| Animal bursts lost | {t['animal_bursts_lost']} of {t['n_animal_bursts']} ({t['pct_animal_bursts_lost']:.2f}%) |",
          f"| Confirmed animal bursts lost | {t['confirmed_animal_bursts_lost']} of {t['n_confirmed_animal_bursts']} |",
          f"| Auto-discarded / needs review / kept | {t['auto_discarded']} / {t['needs_review']} / {t['kept']} bursts |",
          f"| Empty bursts auto-discarded | {t['pct_empty_bursts_discarded']:.1f}% |",
          f"| Count MAE | {c['mae']:.3f}, {c['n_bursts']} bursts |",
          f"| Boxes at IoU 0.5 | precision {b['precision_iou50']:.3f}, recall {b['recall_iou50']:.3f} |",
          f"| Seen minus unseen camera accuracy | {final['cis_minus_trans_accuracy_points']:.1f} points |", "",
          "| Group | Images | Accuracy % | 95% range | Macro precision % | Count MAE |", "|---|---|---|---|---|---|"]
for key, label in (("cis_test", "Seen cameras"), ("trans_test", "Unseen cameras"), ("day", "Day"), ("ir_night", "IR-night"),
                   ("color_night", "Color-night")):
    g = final["groups"][key]
    lines.append(f"| {label} | {g['species']['n']} | {g['species']['accuracy']:.1f} | {g['species']['accuracy_95_range']} | "
                 f"{g['species']['macro_precision']:.1f} | {g['count']['mae']:.3f} |")
lines += ["", "Per-class results on test:", "", "| Class | Precision | Recall | Images |", "|---|---|---|---|"]
for name, m in final["per_class"].items():
    lines.append(f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | {m['n']} |")
lines += ["", "Confusion matrix: `docs/reports/confusion_matrix_test.png`.", "",
          "The 90% accuracy target was not reached. Macro precision passed it."]
(ROOT / "docs" / "reports" / "model_selection.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:9]))
print("MODEL_SELECTION_DONE")
