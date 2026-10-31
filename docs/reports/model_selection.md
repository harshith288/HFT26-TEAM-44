# Model selection and final results

- Command: `python scripts\write_model_selection.py`
- Generated: 2026-10-31T01:31:15
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU
- Real images: 505 train crops, 403 tuning val images, 1502 fresh val images, 502 test images
- Everything was frozen at commit 2bbcf7e before the test splits were opened, once.

## Species model, chosen on val only

All candidates classified the same MDv6 crops.

| Candidate | Macro-F1 % | Accuracy % |
|---|---|---|
| speciesnet_top_mapped | 80.1 | 73.7 |
| speciesnet_processed | 79.6 | 67.5 |
| probe_dinov2 | 43.5 | 44.2 |
| probe_bioclip | 35.6 | 35.0 |
| bioclip_zero_shot | 21.6 | 20.8 |

**Chosen: speciesnet_top_mapped**, then a burst vote at confidence 0.2.

Overfitting check on the linear probes, which were not chosen:

| Probe | Train accuracy % | Val accuracy % | Gap, points | Flag over 10 |
|---|---|---|---|---|
| bioclip | 100.0 | 35.0 | 65.0 | yes |
| dinov2 | 100.0 | 44.2 | 55.8 | yes |

## Thresholds, chosen on tuning val only

- Count threshold: 0.3
- Discard below: 0.005; keep from 0.2
- Rule for discard: highest threshold losing under 1% of animal bursts on tuning val

## Confirmation on fresh val

1173 single-species images at natural proportions: accuracy 89.1%, macro precision 98.5%. Discard rule: 7 of 423 animal bursts lost (1.65%), which missed the 1% target.

## Final test results

| Metric | Result |
|---|---|
| Species accuracy | 88.3% (95% range 81.8 to 94.0), 308 images |
| Macro precision | 95.4% |
| Macro-F1 | 91.6% |
| Accuracy, bursts with a ground-truth box | 96.7%, 272 images |
| Missed images with a ground-truth box | 0 of 21 |
| Animal bursts lost | 1 of 132 (0.76%) |
| Confirmed animal bursts lost | 0 of 120 |
| Auto-discarded / needs review / kept | 3 / 44 / 147 bursts |
| Empty bursts auto-discarded | 3.2% |
| Count MAE | 0.059, 85 bursts |
| Boxes at IoU 0.5 | precision 0.906, recall 0.913 |
| Seen minus unseen camera accuracy | 3.3 points |

| Group | Images | Accuracy % | 95% range | Macro precision % | Count MAE |
|---|---|---|---|---|---|
| Seen cameras | 150 | 90.0 | [80.7, 97.9] | 96.8 | 0.064 |
| Unseen cameras | 158 | 86.7 | [77.2, 95.0] | 98.7 | 0.053 |
| Day | 114 | 86.0 | [73.7, 96.4] | 85.5 | 0.100 |
| IR-night | 183 | 89.1 | [80.9, 96.6] | 97.9 | 0.045 |
| Color-night | 11 | 100.0 | [100.0, 100.0] | 100.0 | 0.000 |

Per-class results on test:

| Class | Precision | Recall | Images |
|---|---|---|---|
| badger | 1.00 | 0.75 | 8 |
| bird | 1.00 | 0.74 | 34 |
| bobcat | 0.97 | 0.82 | 34 |
| cat | 1.00 | 0.83 | 36 |
| coyote | 1.00 | 1.00 | 24 |
| deer | 0.67 | 1.00 | 12 |
| dog | 1.00 | 1.00 | 28 |
| fox | 1.00 | 1.00 | 3 |
| opossum | 0.84 | 0.90 | 29 |
| rabbit | 1.00 | 1.00 | 24 |
| raccoon | 0.89 | 1.00 | 24 |
| rodent | 1.00 | 0.68 | 22 |
| skunk | 1.00 | 1.00 | 14 |
| squirrel | 1.00 | 0.81 | 16 |

Confusion matrix: `reports/confusion_matrix_test.png`.

The 90% accuracy target was not reached. Macro precision passed it.
