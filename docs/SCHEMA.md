# WildWatch data schema

**Schema version: 5.** Version 2 added `images.top_confidence` and `images.triage`.
Version 3 added the `alert_status` table. Version 4 added `alerts.is_guess`. `init_db` upgrades older databases.
Teammates build against this file.
Do not change it without bumping `SCHEMA_VERSION` in `src/db.py` and updating this
file. `tests/test_phase1.py` fails if this file and the real database disagree.

- Database: SQLite, path from `paths.database` in `src/config.yaml`.
- Times: ISO 8601 text, `YYYY-MM-DD HH:MM:SS`, camera local time, exactly as in CCT.
  CCT stores 12 AM as hour 12 and some camera clocks are wrong, so use
  `src.activity.corrected_hour` for hours and `is_night` for night, not the raw hour.
- Day/night group: `is_ir = 1` is IR-night; `is_ir = 0, is_night = 1` is color-night;
  `is_night = 0` is day.
- Booleans: integer 0 or 1.
- Unknown values: `NULL`. Never a made-up default.
- Boxes: normalized 0-1, `x, y` = top-left corner, `w, h` = size.
- `camera_id`: CCT `location` as text. Villages come from `src/config.yaml`.

## Tables

### images
One row per image file, including files that could not be read.

| column | type | notes |
|---|---|---|
| image_id | TEXT PK | CCT id, or SHA-256 prefix for incoming files |
| file_path | TEXT NOT NULL | path relative to the project root |
| source | TEXT NOT NULL | `cct20_sample` or `incoming` |
| split | TEXT | `train`, `cis_val`, `trans_val`, `cis_test`, `trans_test`; NULL for incoming |
| seq_id | TEXT | burst id; NULL if the image is unreadable |
| frame_num | INTEGER | frame position in the burst |
| camera_id | TEXT | NULL if unknown |
| captured_at | TEXT | NULL if the timestamp is missing |
| width | INTEGER | pixels |
| height | INTEGER | pixels |
| sha256 | TEXT | file hash, used to spot duplicates |
| is_ir | INTEGER | 1 if R, G, B are about equal |
| is_night | INTEGER | 1 if IR or a dark scene, from the pixels; never from the clock |
| is_empty | INTEGER | 1 if no detection passed the threshold |
| gt_labels | TEXT | JSON list of ground-truth labels; NULL if unlabeled |
| gt_n_boxes | INTEGER | ground-truth box count; NULL if the image has no boxes |
| status | TEXT NOT NULL | `ok` or `unreadable` |
| error | TEXT | why it was unreadable |
| processed_at | TEXT | when the pipeline finished this image |
| top_confidence | REAL | highest detector score in the image, any category, 0 if no box |
| triage | TEXT | burst decision: `keep`, `review` or `discard`; NULL if unreadable |

### detections
One row per detector box that passed the threshold.

| column | type | notes |
|---|---|---|
| detection_id | INTEGER PK | auto-assigned |
| image_id | TEXT NOT NULL | references `images` |
| category | TEXT NOT NULL | `animal`, `person` or `vehicle` |
| confidence | REAL NOT NULL | detector score, 0-1 |
| x | REAL NOT NULL | box left |
| y | REAL NOT NULL | box top |
| w | REAL NOT NULL | box width |
| h | REAL NOT NULL | box height |
| species | TEXT | CCT images: a CCT class. Uploads: the model's own common name, or `unknown`. NULL if not an animal |
| species_confidence | REAL | classifier score; NULL if not an animal |
| classifier | TEXT | the model that named it, such as `SpeciesNet` or `BioCLIP`. A name ending in `guess` is a low-confidence guess. NULL if there is no name |

### bursts
One row per species per burst.

| column | type | notes |
|---|---|---|
| burst_id | TEXT PK | same as `images.seq_id` |
| species | TEXT PK | species, `person` or `vehicle` |
| camera_id | TEXT | NULL if unknown |
| start_time | TEXT | earliest frame; NULL if no timestamps |
| end_time | TEXT | latest frame; NULL if no timestamps |
| n_frames | INTEGER NOT NULL | frames in the burst |
| min_observed_count | INTEGER NOT NULL | MAX in any single frame, never a sum |
| is_night | INTEGER | 1 if any frame is night |

### alerts
One row per burst and species that reached the alert step.

| column | type | notes |
|---|---|---|
| alert_id | INTEGER PK | auto-assigned |
| burst_id | TEXT NOT NULL | references `bursts` |
| species | TEXT NOT NULL | references `bursts` |
| camera_id | TEXT | NULL if unknown |
| village | TEXT | NULL if the camera has no village |
| min_observed_count | INTEGER NOT NULL | copied from `bursts` |
| danger_score | REAL NOT NULL | species_weight x count x night_factor x zone_factor |
| level | TEXT NOT NULL | `LOW`, `MEDIUM`, `HIGH` or `CRITICAL` |
| rangers | INTEGER NOT NULL | recommended rangers |
| vehicle | INTEGER NOT NULL | recommended vehicle, 0 or 1 |
| possible_poacher | INTEGER NOT NULL | 1 if a person was detected |
| message | TEXT | alert text |
| message_source | TEXT | `llm` or `template` |
| created_at | TEXT NOT NULL | when the alert was written |
| is_guess | INTEGER NOT NULL | 1 if the species is a low-confidence guess, scored with the unknown weight |

### alert_status
Ranger decisions. This is a separate table, keyed by burst and species, because
rebuilding a burst replaces its `alerts` rows and their `alert_id`. A row only exists
once someone acts, so an alert with no row has status `new`.

| column | type | notes |
|---|---|---|
| burst_id | TEXT PK | same as `alerts.burst_id` |
| species | TEXT PK | same as `alerts.species` |
| status | TEXT NOT NULL | `new`, `dispatched` or `false_alarm` |
| updated_at | TEXT NOT NULL | when the status last changed |

### daily_summaries

| column | type | notes |
|---|---|---|
| summary_date | TEXT PK | `YYYY-MM-DD` |
| summary | TEXT NOT NULL | report text |
| summary_source | TEXT NOT NULL | `llm` or `template` |
| created_at | TEXT NOT NULL | when it was written |

### llm_cache
Lets the demo run offline.

| column | type | notes |
|---|---|---|
| cache_key | TEXT PK | SHA-256 of model and prompt |
| response | TEXT NOT NULL | model output |
| model | TEXT NOT NULL | model name |
| created_at | TEXT NOT NULL | when it was cached |

### briefs
One Ranger's Brief per day. A newer upload replaces today's row.

| column | type | notes |
|---|---|---|
| brief_date | TEXT PK | `YYYY-MM-DD` |
| brief | TEXT NOT NULL | overall brief |
| species_notes | TEXT NOT NULL | JSON, `{species: {note, source}}` |
| source | TEXT NOT NULL | `llm` or `template` |
| data_version | INTEGER NOT NULL | `data_version` this brief belongs to |
| created_at | TEXT NOT NULL | when it was written |

### app_state
Holds `data_version`, which rises by 1 after each upload, pipeline run or alert decision. `load_data` caches on it.

| column | type | notes |
|---|---|---|
| key | TEXT PK | setting name |
| value | INTEGER NOT NULL | setting value |

## Pipeline JSON

`src/pipeline.py` returns this for each image. Fields have the same meaning as above.

```json
{
  "schema_version": 1,
  "image_id": "5a0c9d9f-23d2-11e8-a6a3-ec086b02610b",
  "file_path": "data/sample/images/5a0c9d9f-23d2-11e8-a6a3-ec086b02610b.jpg",
  "status": "ok",
  "error": null,
  "camera_id": "33",
  "captured_at": "2011-07-14 05:38:07",
  "is_ir": true,
  "is_night": true,
  "is_empty": false,
  "triage": "keep",
  "detections": [
    {"category": "animal", "confidence": 0.5, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4,
     "species": "bobcat", "species_confidence": 0.5, "classifier": "name"}
  ],
  "alert": {
    "alert_id": 1, "burst_id": "id", "species": "bobcat", "min_observed_count": 1, "danger_score": 0.5,
    "level": "MEDIUM", "rangers": 1, "vehicle": false, "possible_poacher": false,
    "village": "Ambernook", "message": "text", "message_source": "template", "is_guess": false
  }
}
```

The values above only show the format. They are not results. `alert` is the
highest-scoring alert for the image's burst, or `null` when no alert was raised. An
unreadable image has `status` `unreadable`, `error` set, and an empty `detections`
list. A file already in the database comes back as its earlier record, plus
`"duplicate_of": "<image_id>"`.
