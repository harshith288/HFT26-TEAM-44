# WildWatch architecture

## Data flow

One upload updates every page.

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

## Why the work runs in separate processes

- **Models:** torch crashes with an access violation when it shares a process with pyarrow, which pandas uses. So uploads run `python -m src.pipeline --job-dir` in a background process.
- **SpeciesNet** fails with `WinError 6714` when it shares a process with the other models, so it runs in `src/scripts/speciesnet_worker.py`. The cause is unknown.
- **BioCLIP** runs in `src/scripts/bioclip_worker.py` for the same reason.
- **The brief** uses pandas, so the worker starts `python -m src.brief` for it. If that fails, `data_version` still rises, so pages never go stale.

## One source of truth

SQLite is the single source of truth. The dashboard pages read it through the API in `src/server.py`. The Ranger's Brief routes use `load_data(filters)` in `src/db.py`, cached on `data_version`, and the stats in `src/analytics.py`. `src/db.py` loads Streamlit and pandas only when `load_data` is called, so the pipeline can import the file safely.

## Ranger's Brief checks

- The LLM gets aggregated numbers only, never images, paths or ids.
- It rewrites a report built from our numbers. Given raw JSON, it named the wrong top species and invented a date.
- A reply is rejected if it has a number that is not ours, is not English, has the wrong length, or leaves out the top species. The template is shown instead.
- The sentence on data limits and the 88.3% accuracy is added word for word, because the LLM invented a purpose for the data when it rewrote that sentence.

These checks catch wrong numbers and a missing top species, not every wrong claim, so treat the AI wording as a draft.

## Two identification paths

Caltech images use `country=USA`, the CCT class mapping and a burst vote. This path produced the final results.

Uploads use `upload_mode: global`, with no country and no geofence. They do not use the CCT mapping. The final results do not measure this path.

## Night detection

Night comes from the pixels (an IR image or a dark scene), not the clock. CCT stores 12 AM as hour 12, and camera 120 has an unreliable clock.

## Where the settings are

All thresholds, danger groups, villages and LLM settings are in `src/config.yaml`.
