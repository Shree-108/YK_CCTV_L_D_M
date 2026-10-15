# LDM monitoring and continuous improvement

Status 2026-10-12. Sections: **A** what the app already records · **B** what was implemented in this phase ·
**C** proposed metrics · **D** gaps · **E** improvement workflow · **F** retention and access (proposal) ·
**G** test report · **H** not implemented or not tested.

## A. Assessment: what the application already records (inspected, not assumed)

| Data | Where | Recorded? | Notes |
|---|---|---|---|
| Every detection (time, camera, source, frame/video time, class, confidence, box, status, rejection reason, track, model version) | `detections` table | ✅ | statuses rejected/weak/candidate/confirmed; UTC timestamps |
| Alerts (confirmed litter) with **operator review status** new / acknowledged / resolved / false_positive | `alerts` table, Alerts page | ✅ | the main source of human feedback; **0 of 15 reviewed today** |
| Confirmed objects per track | `confirmed_objects` | ✅ | |
| Video job timing: mean inference ms, processing fps, wall time, frames analysed, settings snapshot, errors | `jobs.result_json` / `error` | ✅ | |
| Image-upload inference time | API response only | ❌ | not stored (G1) |
| Scheduled capture status, duration, error, original + annotated image | `captures`, `storage/captures/` | ✅ | 0 captures so far (no camera connected) |
| Human verdicts true_positive / false_positive / **false_negative** (missed) / uncertain | `reviews` (Test Lab) | ✅ | 0 so far |
| Hard examples with review status | `hard_examples` (Hard Examples page) | ✅ | 0 so far; filled by evaluations and Test Lab, **not** by alert reviews (G3) |
| Evaluations on annotated datasets (P/R/mAP, per class/size, threshold sweep, FP on negatives, latency) | `storage/evaluations/*.json`, jobs | ✅ | 0 so far (no annotated test set) |
| System events (start/stop, errors, model activation, alert status changes) | `system_events` | ✅ | |
| Health (API, database, model, cameras, CPU/RAM) | `/api/health`, `/api/system/health` | ✅ | live only, not stored over time |
| Analytics (by camera/class/hour/day, confidence histogram, rejection reasons) | `/api/analytics`, Analytics page | ✅ | activity only; hour-of-day in UTC |
| Model registry (versions, metrics, active flag) | `models` table, `models/litter_yolo/*/model_meta.json` | ✅ | no approval field (G6) |
| Export-folder ingest ledger | `storage/ingest/ingest.db`, `ingest.log` | ✅ | Phase 9 |
| Retention / clean-up of stored footage and rows | none | ❌ | nothing is ever deleted automatically (G4) |

## B. Implemented in this phase

1. **`monitoring/ldm_monitor.py`**, a read-only monitoring report. It opens the database with SQLite `mode=ro`, so
   it cannot write. Output: `storage/monitoring/<timestamp>/` (git-ignored), containing `report.md`,
   `metrics.json`, `daily_activity.csv` and `camera_summary.csv`. It keeps these strictly separate:
   - **Operational** (facts): jobs and failures, inference latency, processing time, captures, restarts, ingest
     status.
   - **Activity**, labelled UNREVIEWED: detections and alerts by day, camera, class, status and confidence.
   - **Quality**, from human review only:
     - alert precision = resolved / (resolved + false_positive). It is shown only with ≥ `--min-reviewed`
       (default 20) decided alerts, and always with a 95 % Wilson interval, per camera, class and model.
     - missed litter, only from Test Lab `false_negative` verdicts and evaluations. A missing alert is never
       counted as "no litter".
     - hard-example review state and evaluation results.
   - **Register**: models with active flag and the sha256 of each `best.pt`, backup checksum verification, dataset
     versions with their verified-image counts, and the approved-change log check.
   - **Retention inventory**: files, MB and age per storage area against the proposed limits. Report only;
     nothing is deleted.
   - **Security**: logs scanned for real `.env` secret values and for URLs with embedded credentials; checks
     whether `.env` is tracked by git.
   - **Retraining rule:** "may be justified" only when reviewed evidence shows a need (precision CI below target,
     ≥ N missed-litter verdicts, or an evaluation below target) **and** enough reviewed data exists (≥ 30 reviewed
     false positives / approved hard examples, or ≥ 300 verified images). It never trains or activates anything.
2. **`governance/model_change_log.csv`** with `governance/README.md`: an append-only record of
   register / evaluate / activate / rollback / backup decisions, with **`approved_by` required for activations**.
   It is seeded only with verifiable facts. v3's original approval was never recorded, so the field is empty and
   the row is marked as a pre-existing baseline.
3. **`tests/test_monitoring.py`**, 6 tests (section G).

No application code, settings, model or data was changed in this phase.

## C. Proposed metrics (weekly report)

| Metric | Source | Status |
|---|---|---|
| Video job success rate, failures with reasons | jobs | **available now** |
| Inference latency (mean ms per job), processing fps, wall time | jobs | **available now** (video only) |
| Image-upload latency | — | needs G1 |
| Capture success rate / duration per camera | captures | available once cameras are connected |
| Ingest: recordings done / failed / pending, reasons | ingest ledger | **available now** |
| Detections and alerts per camera/class/day (activity, unreviewed) | detections, alerts | **available now** (label as unreviewed) |
| Review coverage = decided alerts / alerts | alerts | **available now** (0 % today) |
| **Alert precision** per camera / class / model, with 95 % CI | alerts reviewed by operators | **needs reviews** (≥ 20 decided) |
| **False positives** (confirmed by operators) per camera, class and distractor type | alerts + notes / hard examples | counts available after review; distractor type needs a review note or category (G5) |
| **Missed litter** | Test Lab false_negative verdicts; evaluation recall | **needs reviews / annotated test set** |
| Recall, mAP and the threshold sweep on a fixed test split, per model version | evaluations | **needs `cctv_ldm_v4` test split** |
| Confidence drift (histogram week by week) | detections | **available now**: a shift is a signal to review, not proof of a problem |
| Model, dataset and approval register consistency | models, files, change log | **available now** |
| Storage vs retention limits | file system | **available now** |

## D. Gaps

| ID | Gap | Impact | Proposal (needs approval) |
|---|---|---|---|
| G1 | Image-upload inference time not stored | no latency trend for image uploads | store `inference_ms` with the detections or in a jobs-like row |
| G2 | Alert full snapshots have boxes drawn; video uploads are deleted after processing | a reviewed false positive cannot become training data directly (only a small clean crop is kept) | opt-in setting "keep clean full frame for alerts" (adds storage, so falls under retention); or re-export the moment from KVMS using the alert time; scheduled captures already keep `original.jpg` |
| G3 | Marking an alert `false_positive` does not create a hard example | operator feedback does not reach the review queue | on `false_positive`, insert a `hard_examples` row (`source=alert:<id>`, `review_status=pending`) |
| G4 | No retention / clean-up | footage and rows grow without limit; privacy | approve section F, then add a clean-up job with a dry-run default |
| G5 | No reason/category for false positives (product, vehicle, shoe…) | FP-by-distractor-type cannot be measured | add an optional reason list to the "false positive" action, or use the alert `notes` consistently |
| G6 | Model registry has no approval field | approvals live only in the manual change log | keep the change log mandatory, or add `approved_by` to activation |
| G7 | Silent truncation of long videos (D-06) | partial analysis is invisible in monitoring | proposed fix in `handover/DEFECTS_AND_BLOCKERS.md` |
| G8 | Read endpoints have no login (D-08) | anyone who can reach the API sees footage-derived data | local-only today; SSO / reverse proxy before multi-PC use |
| G9 | Health is not stored over time | no uptime history | the report counts restarts from `system_events`; enough for now |
| G10 | Analytics hour-of-day is UTC | misleading for India (IST = UTC+5:30) | convert in the UI |

## E. Controlled improvement workflow

```
operators review alerts ──► weekly monitoring report ──► enough reviewed evidence? ──no──► keep v3, keep reviewing
   (resolved / false positive,      (python -m monitoring.ldm_monitor)          │yes
    Test Lab verdicts for misses)                                                ▼
                                  collect clean frames for the reviewed cases (KVMS re-export at the alert time /
                                  scheduled-capture originals) ─► extract_frames / import_images ─► annotate
                                  (docs/LDM_ANNOTATION_GUIDE.md) ─► validate ─► new dataset version
                                                                                 ▼
                     evaluate v3 AND the candidate on the SAME fixed test split (Model Lab) ─► acceptance criteria
                     (docs/LDM_FINETUNE_PLAN.md §4) ─► written approval ─► change-log row ─► register ─► activate
                                                                                 ▼
                                  monitor the next weeks; rollback = activate the previous version (+ log row)
```

1. **Daily or at each demo (operator):** in Alerts, set every alert to `resolved` (real litter) or
   `false_positive` (wrong). Write the reason in notes (`product`, `vehicle`, `tyre`, `shoe`, `shadow`…). Use
   `acknowledged` only for "seen, not yet judged".
2. **Missed litter (operator / QA):** record it in Test Lab as a `false_negative` verdict (image, box, class).
3. **Weekly (developer):** run `python -m monitoring.ldm_monitor --days 7`. Read section 3 (Quality) first.
   Activity numbers are not accuracy.
4. **Hard examples (developer):** review the Hard Examples page (approve / reject). Approved cases are
   re-collected as **clean** frames and added to `dataset/ldm_v4_workspace` with `source-type real_cctv`. Never use
   annotated snapshots.
5. **Dataset version (developer):** annotate, `ldm_review sync`, validate, then
   `build_ldm_dataset --name cctv_ldm_v<N>`. The test split stays fixed across versions, so results stay
   comparable. Add a `dataset` row to the change log.
6. **Retrain only when the report's retraining decision says it may be justified,** and with approval. Then
   evaluate, record an `evaluate` row, get approval, record an `activate` row (with `approved_by`), and activate in
   the Model Lab.
7. **Never:** retrain or activate automatically, treat unreviewed detections as labels, or change thresholds
   without an evaluation on the test split.

## F. Retention and access (proposal: not implemented, needs organisational approval)

| Data | Proposed retention | Reason |
|---|---|---|
| `storage/uploads` (temporary) | delete after processing (the app already does this) | |
| `storage/videos` (annotated videos) | 30 days | large; contains people |
| `storage/captures` | 30 days, except frames selected for the dataset | |
| `storage/alerts` snapshots | 90 days, or until the alert is resolved + 30 days | evidence for review |
| `storage/ingest` results | 90 days; **KVMS export originals stay in the export folder under the site's own policy** | the ingester never copies or deletes originals |
| `dataset/…/images` selected for training | for the life of the model, access-restricted | needed to reproduce a model |
| detections / alerts rows | 12 months, then aggregate | trends |
| monitoring reports | 12 months | |
| `.env` (admin token, any CCTV password) | rotate the admin token per pilot; view-only NVR account | |

Access: today the app is reachable from this PC only (127.0.0.1). Changes need the admin token; reads are open
(G8). Before any multi-user pilot: company SSO / authenticated reverse proxy, HTTPS, named accounts, and the admin
token held by named people only. CCTV footage must not leave approved machines; no cloud upload without written
approval. The monitor's report reads these sizes; nothing is deleted until a retention job is approved and
implemented.

## G. Test report (executed 2026-10-12)

| Test | Result |
|---|---|
| `tests/test_monitoring.py`: Wilson interval | ✅ |
| unreviewed alerts (30 × `new`) give **no** precision ("only 0 decided") | ✅ |
| 7 decided + 10 acknowledged: no precision (minimum 20); 24 decided (18 resolved / 6 FP): precision 0.75 with CI, per camera | ✅ |
| job latency, failed job with its error, Test Lab false_negative counted as missed litter | ✅ |
| retraining rule: 40 decided with precision 0.25 → "may be justified"; only unreviewed alerts → "not enough reviewed evidence" | ✅ |
| full report on a database: **sha256 of the database file unchanged**; a write through the monitor's connection is refused | ✅ |
| full test suite | ✅ 54 passed, 38 skipped (pre-existing skips: missing v1 model) |
| report on the live database (`storage/monitoring/20261012_232157/report.md`) | ✅ generated. 1 video job (mean 177 ms inference, 37.6 s), 1,470 detections / 15 alerts (unreviewed), **0 decided alerts, so no precision**, 0 verdicts, 0 evaluations; model sha and backup verified; change log: 3 entries, no problems; retention: nothing over the proposed limits; 10 logs scanned, no leaks. Verdict: **"NOT ENOUGH REVIEWED EVIDENCE TO JUDGE"** |

## H. Not implemented / not tested

- Fixes for G1–G10 (proposals above, need approval). No app code was changed.
- A UI dashboard page. The report is a file; the existing Analytics page shows activity only.
- Automatic retention or clean-up (proposal F only).
- Scheduling the weekly report (e.g. Windows Task Scheduler). Not set up, to avoid unattended jobs before approval.
- Monitoring of live cameras or captures (no camera connected), the ingest section with real KVMS exports, and
  evaluation-based metrics (no annotated test set).
