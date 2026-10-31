# WildWatch

AI wildlife monitoring from camera traps, with village intrusion alerts. Built by Team Pentabytes.

**This is a hackathon prototype, not a field-ready system.** The villages are fictional and the risk factors are placeholders.

## Problem statement

Camera traps produce far more photos than rangers can check by hand, and many of those photos are empty. When a dangerous animal moves toward a village, that warning can sit unseen in a memory card or a folder until it is too late to act on.

WildWatch sorts the photos, names the species, counts the animals, scores the risk for the nearest village and writes an alert a ranger can act on. It also identifies animals in photos people upload, from any country.

## Features

1. **Burst triage:** MegaDetector v6 sorts each burst into auto-discarded, needs review or kept.
2. **Species identification:** SpeciesNet classifies each animal box, then the burst votes on one species.
3. **Worldwide upload identification:** SpeciesNet with no geofence, then BioCLIP on 2,227 species, then a clearly labeled best guess.
4. **Animal count:** the most animals in any single frame, labeled "minimum observed count".
5. **Activity patterns:** day and night per species. Night comes from the image, not the clock.
6. **Dashboard:** five designed pages (Home, Upload, Alerts, Analytics and 🦉 Ranger's Brief) served from the real database and reports.
7. **Intrusion alert:** danger = species weight × count × night factor × zone factor, with a recommended response. A person is always a possible poacher alert.
8. **LLM alerts:** Qwen2.5-7B on Featherless writes alerts and daily summaries, with a template fallback.
9. **🦉 Ranger's Brief:** an AI field report, updated with every upload, with per-species notes, "Explain a detection" and "Ask the data". It sends aggregated numbers only, never images. If the API fails, or a reply fails the checks, the page shows a template built from the numbers, so it is never blank.

## How one upload updates every page

1. The Upload page sends the photos to `src/server.py`, which saves them and queues a job.
2. A separate process runs detection, species and the alert rules, because torch crashes when it shares a process with pyarrow.
3. Everything is saved to SQLite.
4. The stats are recomputed and the Ranger's Brief is regenerated.
5. `data_version` rises by 1.

All pages read the same database through the API and refresh themselves, so Home, Alerts, Analytics and Ranger's Brief show the new data within about 10 seconds. The Ranger's Brief routes read through `load_data` in `src/db.py`, which is cached on `data_version`. The folder watcher takes the same path. `process_upload` in `src/pipeline.py` runs the same job for Python callers.

## Architecture

```text
 Upload page (src/frontend)            Folder watcher (data/incoming)
     |  POST /api/upload                        |
     v                                          |
 src/server.py saves the images, queues the job |
     |                                          |
     v                                          v
 Worker process, one upload at a time (torch stays out of the server)
     1. detection      MegaDetector v6
     2. species        SpeciesNet, then BioCLIP, then a labeled guess
     3. alert rules    count, night, danger score
     4. save           images, detections, bursts, alerts
                              |
                              v
 Brief process  python -m src.brief  (pandas stays away from torch)
     5. stats          src/analytics.py
     6. Ranger's Brief Featherless, aggregated numbers only
                       template if the reply fails the checks
     7. save           one brief per day
     8. data_version + 1
                              |
                              v
                 SQLite  (src/db.py, schema v5)
                              |
                 src/server.py API, read by the pages every few seconds
                              |
       +----------+-----------+----------+----------------+
       v          v           v          v                v
     Home      Upload      Alerts    Analytics     Ranger's Brief
```

More detail is in [docs/architecture.md](docs/architecture.md), and the database is described in [docs/SCHEMA.md](docs/SCHEMA.md).

## Tech stack and models

| Part | Used for | Licence |
|---|---|---|
| Python 3.11, PyTorch | Runtime | PSF, BSD-style and Apache-2.0 mix |
| FastAPI, Uvicorn | API and dashboard server | MIT, BSD-3-Clause |
| HTML, JavaScript | Dashboard pages | MIT, this project |
| Streamlit, Plotly | Backup dashboard | Apache-2.0, MIT |
| SQLite | Storage | Public domain |
| MegaDetector v6 extra (`MDV6-yolov9-e-1280`) | Animal, person and vehicle detection | **Unclear:** the docs list AGPL-3.0, the Zenodo record says CC-BY-4.0 |
| Ultralytics | Loads MegaDetector | AGPL-3.0 |
| SpeciesNet | Species classification | Apache-2.0 |
| BioCLIP through OpenCLIP | Fallback species identification | MIT, OpenCLIP is MIT |
| Qwen2.5-7B-Instruct on Featherless | Alerts and the Ranger's Brief, through the OpenAI SDK | Apache-2.0 |
| Caltech Camera Traps (CCT20) | Evaluation data | CDLA-Permissive |

WildWatch's own code is MIT. Ultralytics is a dependency under AGPL-3.0, and that may affect how a combined system can be distributed, so check this before any real use.

## Setup

Needs Windows, an NVIDIA GPU with at least 6 GB of memory, and Python on PATH with `uv` installed (`python -m pip install uv`). Model weights, images and the database are not in this repository. Setup downloads the models and the Caltech sample.

```bash
powershell -ExecutionPolicy Bypass -File src/scripts/setup.ps1
```

Add `-BrowserTests` to set up the browser tests. They use the installed Edge.

To turn on LLM text, copy `.env.example` to `.env` and fill in your key. Without it, WildWatch uses template text.

```bash
copy .env.example .env
```

## Run

Fill the database first:

```bash
powershell -ExecutionPolicy Bypass -File src/scripts/run.ps1 pipeline
```

Then open the dashboard at http://localhost:8000:

```bash
powershell -ExecutionPolicy Bypass -File src/scripts/run.ps1 dashboard
```

| `run.ps1` mode | What it does |
|---|---|
| `dashboard` | Starts the main dashboard and opens the browser. `-Port` picks another port and `-NoBrowser` keeps the browser closed |
| `streamlit` | Starts the backup Streamlit dashboard |
| `test` | Runs the full test suite |
| `pipeline` | Processes the train and val images |
| `once` | Processes new files in `data/incoming`, then stops |
| `watch` | Keeps checking `data/incoming` |

## Project layout

| Path | Contents |
|---|---|
| `src/server.py` | FastAPI server for the dashboard and API |
| `src/frontend/` | The five dashboard pages |
| `src/pipeline.py`, `src/db.py`, `src/analytics.py`, `src/brief.py` | Pipeline, database, shared stats and the Ranger's Brief |
| `src/streamlit_app.py` | Backup Streamlit dashboard |
| `src/scripts/` | Setup, run, evaluation and model worker scripts |
| `tests/` | Pytest suite, browser checks and golden set expectations |
| `docs/` | Architecture, schema, demo script, metrics and screenshots |

## Final results

These come from the test splits, opened once after everything was frozen at commit `2bbcf7e` of the development repository. Details are in [docs/reports/model_selection.md](docs/reports/model_selection.md).

| Metric | Result | Target |
|---|---|---|
| Species accuracy | **88.3%** (95% range 81.8 to 94.0), 308 images | 90%: not reached |
| Macro precision | **95.4%** | 90%: reached |
| Macro-F1 | 91.6% | |
| Accuracy on bursts with a ground-truth box | 96.7%, 272 images | |
| Animal bursts lost | **1 of 132 (0.76%)** | under 1%: reached |
| Confirmed animal bursts lost | 0 of 120 | |
| Count MAE, per burst | 0.059, 85 bursts | |
| Boxes at IoU 0.5 | precision 0.906, recall 0.913 | |

All 21 missed images are in bursts with no ground-truth box. CCT labels whole bursts, so these may be label noise, though that is not proven.

| Group | Images | Accuracy | 95% range | Macro precision | Count MAE |
|---|---|---|---|---|---|
| Seen cameras | 150 | 90.0% | 80.7 to 97.9 | 96.8% | 0.064 |
| Unseen cameras | 158 | 86.7% | 77.2 to 95.0 | 98.7% | 0.053 |
| Day | 114 | 86.0% | 73.7 to 96.4 | 85.5% | 0.100 |
| IR-night | 183 | 89.1% | 80.9 to 96.6 | 97.9% | 0.045 |
| Color-night | 11 | 100.0% | too few to judge | 100.0% | 0.000 |

The ranges overlap, so the gaps between groups may not be real.

### Three-tier triage

Each burst gets one decision from its highest detector score.

| Tier | Rule | Test splits, 194 bursts | Fresh val, 528 bursts | Whole sample, 584 bursts |
|---|---|---|---|---|
| Auto-discarded | Below 0.005 | 3 (1.5%) | 25 (4.7%) | 5 (0.9%) |
| Needs review | 0.005 to under 0.2 | 44 (22.7%) | 81 (15.3%) | 85 (14.6%) |
| Kept | 0.2 or higher | 147 (75.8%) | 422 (79.9%) | 494 (84.6%) |

Fresh val has natural class proportions. The test splits and whole sample do not: their empty bursts are oversampled, and train has none.

This setting favours never losing an animal, so it removes few empties by itself: only 3.2% of empty test bursts were auto-discarded, and the review queue carries most of the load. On fresh val, 1.65% of animal bursts were lost, so the under 1% target held on test but not there.

### Speed

Measured on an NVIDIA RTX 4050 Laptop GPU with 6 GB of memory.

| Step | Time |
|---|---|
| MDv6 detection | 238.4 ms per image |
| SpeciesNet classifier | 67.4 ms per crop |
| Full pipeline on the test images | 540 ms per image (502 images in 271.0 s) |
| Streamlit upload to result, 4 files | 54.7 s, including model start-up |

## Identifying any animal in uploads

Uploaded photos are not limited to Caltech species or to any country. Each animal box goes through these steps:

1. SpeciesNet runs with no country and no geofence, and its own label is used if it names a species with a score of at least 0.95.
2. Otherwise BioCLIP compares the crop with 2,227 mammals, birds, reptiles and amphibians from every continent and ocean, and its label is used if the score is at least 0.7.
3. Otherwise the app shows its best guess, marked "possibly" and "low confidence": SpeciesNet's specific label if it has one, or else BioCLIP's.
4. "Unknown animal" appears only if neither model offers a specific name.

The dashboard shows each photo's label, confidence and the model that named it. Only Caltech images use `country=USA` and the CCT mapping, so the final results above do not change, and they do not measure uploads.

The thresholds were tuned on 132 real non-CCT photos, allowing wrong labels on at most 10% of them. Above the thresholds that gave 58 right and 13 wrong. Of the 56 tuning photos below them, 23 best guesses were right (41.1%), so treat a guess as a hint. A guess uses the unknown danger weight of 0.5, so it can neither raise nor lower the risk.

Two checks on real photos are below. They are too small to be accuracy figures.

**Subjects chosen for an Indian village setting: 17 of 20 hits, 11 of them confident.**

| Photo | Hit | Label | Confidence | Model |
|---|---|---|---|---|
| lion | Yes | lion | 0.999 | SpeciesNet |
| tiger | Yes | tiger | 1.000 | SpeciesNet |
| leopard | Yes | leopard | 0.995 | SpeciesNet |
| elephant | Yes | african elephant | 0.962 | SpeciesNet |
| wild boar | Yes | wild boar | 0.966 | SpeciesNet |
| sloth bear | Yes | sloth bear | 0.884 | BioCLIP |
| monkey | Yes | possibly rhesus monkey | 0.326 | SpeciesNet guess |
| peacock | Yes | possibly indian peafowl | 0.598 | SpeciesNet guess |
| cobra | No | possibly russell's viper | 0.535 | BioCLIP guess |
| crocodile | No | possibly yellow-spotted monitor | 0.348 | SpeciesNet guess |
| deer | Yes | possibly chital | 0.873 | SpeciesNet guess |
| cow | Yes | possibly domestic cattle | 0.946 | SpeciesNet guess |
| dog | Yes | possibly domestic dog | 0.917 | SpeciesNet guess |
| cat | Yes | wild cat | 0.938 | BioCLIP |
| horse | No | hartebeest | 0.840 | BioCLIP |
| camel | Yes | dromedary camel | 0.993 | SpeciesNet |
| giraffe | Yes | giraffe | 0.967 | SpeciesNet |
| zebra | Yes | plains zebra | 0.968 | SpeciesNet |
| person | Yes | person | - | - |
| empty field | Yes | possibly daurian hedgehog | 0.123 | BioCLIP guess |

**Animals from every continent and ocean: 13 of 20 hits, 10 of them confident.**

| Photo | Hit | Label | Confidence | Model |
|---|---|---|---|---|
| hippopotamus | Yes | hippopotamus | 0.954 | SpeciesNet |
| ring-tailed lemur | Yes | ring-tailed lemur | 0.944 | BioCLIP |
| black rhinoceros | Yes | black rhinoceros | 0.981 | BioCLIP |
| giant panda | Yes | giant panda | 0.960 | SpeciesNet |
| komodo dragon | No | possibly common water monitor | 0.304 | BioCLIP guess |
| orangutan | Yes | bornean orangutan | 0.987 | SpeciesNet |
| european badger | Yes | american badger | 0.956 | SpeciesNet |
| eurasian lynx | Yes | canada lynx | 1.000 | SpeciesNet |
| red deer | No | chinese goral | 0.734 | BioCLIP |
| polar bear | No | possibly mountain goat | 0.200 | SpeciesNet guess |
| american bison | Yes | possibly american bison | 0.890 | SpeciesNet guess |
| moose | Yes | possibly moose | 0.902 | SpeciesNet guess |
| capybara | Yes | lesser capybara | 0.908 | BioCLIP |
| llama | No | possibly guanaco | 0.688 | BioCLIP guess |
| jaguar | No | possibly domestic dog | 0.582 | SpeciesNet guess |
| red kangaroo | Yes | possibly red kangaroo | 0.215 | SpeciesNet guess |
| koala | Yes | koala | 0.987 | BioCLIP |
| platypus | No | nothing | - | - |
| emperor penguin | Yes | emperor penguin | 0.950 | BioCLIP |
| walrus | No | possibly north american river otter | 0.174 | SpeciesNet guess |

Hits include near answers: the Asian elephant was named African, the domestic cat was named wild cat, and the Eurasian lynx was named Canada lynx. A weak box on an empty photo can also get a meaningless guess, like the hedgehog on the empty field. It raises no alert, because its box is below the count threshold.

Danger weights for confident uploads are in `danger_groups` in `src/config.yaml`. They are matched on taxonomy and are editable. Animals in no group, such as crocodile, giraffe and zebra, use the unknown weight of 0.5.

## How the model was chosen

Model selection used validation images only.

| Candidate | Val macro-F1 |
|---|---|
| SpeciesNet, top guess mapped to a CCT class (chosen) | 80.1% |
| SpeciesNet, processed output | 79.6% |
| Linear probe on DINOv2 | 43.5% |
| Linear probe on BioCLIP | 35.6% |
| BioCLIP zero-shot | 21.6% |

Both linear probes overfit badly, at 100% train accuracy against 35 to 44% on val.

## Known limitations

- **Species coverage:** the final results cover US species only. Uploads can name other animals, but only a small check supports that.
- **Minimum count:** the count is the most animals seen in one frame, so more may have been present.
- **Small test set:** about 250 images per test split, drawn from a 1,506-image sample, so ranges are wide. Almost every burst holds 1 to 3 animals, so large groups are untested.
- **Not all CCT cameras:** the unseen-camera result covers the 9 trans-test cameras in this sample.
- **Person detection has no measured accuracy:** CCT has no person label. It fired on real hikers and a driver, and also on parts of trucks, so "possible poacher" needs a human check.
- **Review load:** about 15 to 23% of bursts need a person to look at them.
- **Weak classes:** rodent recall was 0.68 and bird recall 0.74 on test. Deer precision was 0.67.
- **Timestamps:** CCT stores 12 AM as hour 12 and camera 120's clock is unreliable, so night comes from the pixels. A bright flash close to the camera could read as day.
- **Activity patterns:** the sample was balanced by day and night, so the charts describe the sample, not natural behaviour.
- **Risk scores:** weights, factors, thresholds and ranger numbers are placeholders with no field data behind them.
- **LLM text:** test-split alerts use templates to protect the credit. Alert text that spells a number as a word fails the facts check and also uses the template.
- **MegaDetector licensing:** the MDv6 documentation lists AGPL-3.0, while the Zenodo record says CC-BY-4.0. Check this before any real use.
- **SpeciesNet:** it must run in its own process, because it fails with `WinError 6714` alongside the other models. The cause is unknown.

## Testing

```bash
powershell -ExecutionPolicy Bypass -File src/scripts/run.ps1 test
```

The suite covers edge-case files, the risk rules, LLM fallback and budget stop, the Ranger's Brief checks, one real upload through `process_upload`, a 30-image golden regression check against `docs/reports/baseline.json`, the API against direct SQL, and the pages in a real browser. It needs the models, the sample images and a filled database, so run setup and the pipeline first. Some tests also need the extra evaluation photos, which setup does not download. The browser checks need `-BrowserTests` and Edge.

LLM tests use labeled mocks. Everything else runs on real CCT images, apart from the blank images those edge cases require. Tests run on a temporary copy of the database with LLM calls turned off.

## Team

Team Pentabytes:

- Akhil Murthy (lead)
- Shiva Ganesh
- Harshith
- Siddhartha
- Hruthik

## Licence

MIT. See [LICENSE](LICENSE).
