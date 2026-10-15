# Final UAT report: Yatrikart CCTV Litter Detection (LDM)

Date: 2026-10-12 · Location: `C:\SHREE\YATRIKART\PROJECTS\YK_CCTV_L_D_M_main\YK_CCTV_L_D_M` · Model under test:
`v3_store_cctv` (production) · Environment: Windows 11 laptop, CPU only.

## Decision: **READY FOR CONTROLLED DEMONSTRATION**

**Not** ready for a limited pilot. This is not production approval, which needs the organisation's authorisation.

The decision rests on this evidence:

- **The application works end to end on this PC.** These were all executed and passed:
  - the UI loads and talks to the API (including the upload path that was broken by CORS before Phase 8);
  - image and video inference produce boxes, classes, confidences, timestamps and annotated outputs;
  - invalid, corrupt and oversize inputs return clear errors;
  - missing model files are reported clearly;
  - configuration and the model survive a restart;
  - start, stop, re-run and recovery behave correctly;
  - backup and restore of model and data are verified;
  - no secrets are committed;
  - no notifications can be sent.

  See `TEST_CASES.md`: 61 cases with recorded results.
- **Why not a pilot**, any one of which blocks it:
  1. There is **no evaluated model improvement** and **no measured real-world accuracy**. The fine-tuned candidate
     does not exist, the dataset has 0 annotated images, and there is no unseen labelled test set (R2, R3, M15
     BLOCKED).
  2. A **false positive was seen** (a shoe sole flagged as plastic, D-07). The owner's original concerns (products,
     bottles, vehicles, missed small litter) cannot be confirmed or ruled out without unseen footage.
  3. **Recorded-footage retrieval from the NVR is unverified** (C01/C02 BLOCKED). Only manual KVMS export plus the
     automated export-folder pipeline works, and that pipeline is tested only on clips with made-up names, not a
     real KVMS export (C10).
  4. **No multi-user access:** local only, and read endpoints have no login (D-08, B-07). The manager and QA team
     can only use it at this PC.
  5. **Silent video truncation** above 1 hour (D-06) is open.

## Results at a glance

| Area | PASS | FAIL | BLOCKED | NOT TESTED |
|---|---|---|---|---|
| Readiness (R1–R7) | 5 | 0 | 2 | 0 |
| Model (M01–M16) | 12 (M06/M07 qualitative only) | 1 (M04 false positive) | 1 | 2 |
| Application (A01–A20) | 16 | 1 (A17b truncation not visible) | 0 | 1 |
| CCTV integration (C01–C12) | 7 | 0 | 3 | 2 |
| Security / operations (S01–S08) | 8 (S04 local demo only; S06 with gap D-06) | S04 for a multi-user pilot | 0 | 0 |

Regression: `pytest` 48 passed / 38 skipped (pre-existing skips: missing v1 model); TypeScript 0 errors; production
UI build OK.

## Fixed during this work (re-tested)

- **D-01:** "Backend offline" on :3010 (CORS).
- **D-02:** the start script hung when its output was captured.
- **D-03:** the build modified developer config files.
- **D-04:** an ingest dry run wrote to the ledger.
- **D-05:** the status badge didn't reflect the real health check, and the health check had no timeout.

Open items are in `DEFECTS_AND_BLOCKERS.md`.

## Known limitations

- **The model is v3 only.** Its accuracy on unseen cameras, night/IR, rain and clean scenes is unknown. Its
  training metrics are inflated (validation from the same scene).
- **CPU only:** about 0.13–0.27 s per frame, and memory is tight on this laptop.
- **Local only:** no login, so multi-PC use needs an IT-approved server or proxy.
- **CCTV:** no live or playback connection to the real NVR has ever been tested. Recordings must be exported
  manually from KVMS as MP4/AVI, at most 1 h and 500 MB each, and the KVMS file naming is unknown.
- **Uploads in the demo are stored as real alerts.** The live database now has 15 alerts. Alerts 7–15 were created
  together at 23:02 local time via an image upload in the UI (camera `upload`), not by the automated tests (those
  used isolated databases). **Please confirm they came from a manual demo upload.**
- **Hour-of-day analytics use UTC,** as stated on the Analytics page.

## Next steps for a controlled pilot

| Who | Action |
|---|---|
| **Manager** | authorise: (1) raw NVR exports from several days, times and cameras, including night, rain and clean scenes; (2) a GPU (Colab or a GPU PC) for training; (3) an IT-approved internal server or VM with login for multi-PC access; (4) the pilot scope and success criteria (`docs/LDM_FINETUNE_PLAN.md` §4) |
| **Developer** | confirm class definitions with the owner; annotate the 46 priority frames, then the new footage; restore `cctv_store_v2` if possible; build and validate `cctv_ldm_v4`; v3 baseline; approved fine-tune; offline comparison; then a controlled video test. Fix D-06 after approval; hard-negative data for D-07 |
| **QA** | run the demo with `deploy\README.md`; record false positives and misses with frame and time; re-run `TEST_CASES.md` cases after each change |
| **CCTV vendor / installer (CP PLUS)** | answer `docs/CCTV_RECORDING_RETRIEVAL.md` §4: ONVIF Profile G / SDK / playback-RTSP support for CP-UNR-104F1, a view-only account, LAN/VPN reachability, KVMS export formats and naming, NTP/time zone. Provide one sample KVMS export |
| **IT** | approve any network access (`-Lan` or a server) and a backup location |
