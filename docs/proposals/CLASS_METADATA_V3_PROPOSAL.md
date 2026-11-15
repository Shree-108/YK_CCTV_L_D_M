# Proposal: class metadata for `cup`, `newspaper` and `plastic_piece`

Date: 2026-11-13. Status: **PROPOSAL ONLY.** Nothing has been changed: `backend/core/class_metadata.json`, application
code, detection logic and all data are untouched.

The evidence comes from reading the code, plus read-only queries on **byte copies** of the production and demo databases. The live database file was never opened, and production was verified unchanged afterwards.

## 1. Current class metadata

`backend/core/class_metadata.json` describes 25 legacy classes. It is loaded once at backend start-up (`backend/services/state.py:32`).
For v3's 10 classes:

| v3 class | priority | bin | similarity group (dedup) | report group (evaluation) |
|---|---|---|---|---|
| plastic_bag | critical | general | soft_packaging | plastic_packaging |
| plastic_packaging | high | plastic | soft_packaging | plastic_packaging |
| food_wrapper | high | general | soft_packaging | wrapper |
| paper | medium | paper | paper_like | none |
| cardboard | low | paper | paper_like | none |
| plastic_bottle | medium | plastic | bottles_cans | none |
| **cup** | **missing → default "medium"** | **missing → default "general"** | **none** | none |
| **newspaper** | **missing → default "medium"** | **missing → default "general"** | **none** | none |
| **plastic_piece** | **missing → default "medium"** | **missing → default "general"** | **none** | none |
| other_litter | low | general | small_items | none |

The defaults come from `ClassMetadata.priority()` and `.bin()` (`backend/alerts/manager.py:41-45`). A class with no group is "similar" only to itself (`manager.py:47-53`). `GET /api/classes` reports the missing classes as `medium` / `general` with the medium colour. The UI therefore already shows them as "Medium", just implicitly.

## 2. Current severity rules

Defaults are in `backend/core/settings.py:88-92`; the database has no override (`settings` table empty in production and demo). Severity is computed **once, when an alert is created**, and stored in `alerts.severity` (`manager.py:175-182`).

| Rule | Condition |
|---|---|
| HIGH | priority `critical` **and** confidence ≥ 0.75, **or** ≥ 5 confirmed objects active on the camera at that moment (any class) |
| MEDIUM | priority `critical` or `high` |
| LOW | everything else |

What each priority produces:

| priority | severity (fewer than 5 active objects) | with ≥ 5 active objects |
|---|---|---|
| critical | HIGH if conf ≥ 0.75, else MEDIUM | HIGH |
| high | MEDIUM | HIGH |
| **medium** (and today's default) | **LOW** | HIGH |
| low | LOW | HIGH |

So `cup`, `newspaper` and `plastic_piece` alerts are currently **LOW**, or HIGH only in busy scenes. That is the same as `paper` and `plastic_bottle`.

## 3. Current deduplication behaviour

1. **One alert per track:** a tracked object that has already alerted is not alerted again (`one_alert_per_track`).
2. **Across tracks** (`find_duplicate`, `manager.py:101-124`): a new confirmed object becomes `occurrences + 1` on an existing alert, instead of a new alert, when all of these hold:
   - same camera **and** same job (uploads never merge with other uploads or with live);
   - the existing alert is from the last 900 s;
   - its track is not still visible;
   - **the class is the same or in the same similarity group** (`class_match = "group"`);
   - IoU ≥ 0.30 **or** centre distance ≤ 4 % of the frame diagonal.
3. Because the three classes have **no group**, an alert of one of them merges only with an earlier alert of the **same** class. A `newspaper` re-detected as `paper` on the same spot would produce a second alert.

## 4. Measured situation (copies of production and demo databases)

| | Production (23 alerts) | Demo (6 alerts) |
|---|---|---|
| `plastic_piece` alerts | 2 (#4 LOW conf 0.62; #13 HIGH conf 0.65 via the ≥ 5 active-objects rule) | 1 (LOW) |
| `cup` alerts | 0 (3 confirmed detections, never an alert) | 0 |
| `newspaper` alerts | 0 | 0 |
| Tracks with candidate/confirmed detections | 256 | 12 |
| Tracks where v3 gave **more than one class** | **1** (`food_wrapper` ↔ `paper`) | 0 |
| Tracks containing `plastic_piece` | 13, none mixed with another class | 3 |
| Nearest other alert to each existing `plastic_piece` alert | IoU 0.00, 7 % of the diagonal away: **would stay separate under any grouping** | same |

Class confusion within a track is rare in the data we have. The grouping decision therefore matters mostly for future footage, and that footage is not yet available.

## 5. Recommended values

```json
"cup":           { "priority": "medium", "bin": "general" },
"newspaper":     { "priority": "medium", "bin": "paper" },
"plastic_piece": { "priority": "medium", "bin": "plastic" }
```
Similarity groups (append to the existing lists):

- `cups_containers`: add `cup`
- `paper_like`: add `newspaper`
- `soft_packaging`: add `plastic_piece`

`report_groups`: **unchanged**. These are used only in evaluation reports, and there is no labelled v3 evaluation data yet.

### Rationale

| class | why this priority | why this bin | why this group |
|---|---|---|---|
| cup | Same as the legacy `plastic_cup` (medium). Keeps today's effective value, so severity is unchanged | Material unknown (paper, plastic or styrofoam tea/coffee cups), so `general` | Natural group (cups, containers, lids). No other v3 class is in it, so in practice it still merges only with `cup` |
| newspaper | Same as `paper` (medium). Keeps today's effective value | It is paper | Lets a `newspaper`/`paper` relabel of the same sheet on the same spot merge into one alert, like the `food_wrapper`/`paper` flip actually observed once |
| plastic_piece | Keeps today's effective value | It is plastic | v3's other plastic-litter classes (`food_wrapper`, `plastic_packaging`, `plastic_bag`) are the classes it is most plausibly confused with |

**The priority choice is a business decision.** With `medium`, nothing about severity changes; the entries only become explicit and documented. If the business wants any of these classes escalated, the alternatives are:

- `high` → **MEDIUM** severity;
- `critical` → **HIGH** at confidence ≥ 0.75.

## 6. Expected impact

| Area | With the recommended values (all `medium`) | If a priority is raised instead |
|---|---|---|
| Existing alerts (23 production, 6 demo) | **None.** Severity is stored at creation; dedup only looks at alerts from the last 900 s | **None** (not retroactive) |
| New alert severity | **Unchanged** (medium = today's default) | `high` → MEDIUM instead of LOW; `critical` → HIGH/MEDIUM |
| Dashboard "High-priority incidents", Analytics severity chart, Alerts page badges, notification threshold (`NOTIFY_MIN_SEVERITY` = HIGH) | Unchanged | New alerts of that class count as MEDIUM/HIGH; with `critical`, HIGH alerts would also be notified if a provider is configured |
| Analytics by class, alert/detection counts, CSV exports | Unchanged (class names are stored, not metadata) | Unchanged |
| Image page "Priority" badge, Video page priority counters | Unchanged ("Medium"; now from the metadata instead of the fallback) | Shown as High/Critical |
| `GET /api/classes` | `bin` becomes `paper` for newspaper and `plastic` for plastic_piece (display only) | + priority and colour |
| Dedup of **new** alerts | `newspaper` can merge with nearby `paper` or `cardboard`, and `plastic_piece` with nearby `food_wrapper`, `plastic_packaging` or `plastic_bag`, within 900 s in the same job. `cup` behaves as today | Same |
| Risk of the grouping | Two **different** objects of grouped classes lying within 4 % of the diagonal of each other (about 29 px at 640×368, about 88 px at 1080p) in the same job would give one alert with `occurrences = 2` instead of two alerts. This is the same rule that already applies to wrappers and bags | Same |
| Evaluation reports | Unchanged (`report_groups` untouched) | Unchanged |
| Detection, tracking, model | **Unchanged** (the metadata is not used by inference) | Unchanged |

**How to make the change (after approval):**

- edit `backend/core/class_metadata.json` only;
- restart the backend (the metadata is loaded at start-up);
- rollback = revert that file and restart.

**Tests:**

- `test_v3_classes_missing_from_class_metadata_get_default_handling` would then check an empty list and pass without testing anything. It should be replaced by a test asserting the explicit values and groups for all 10 v3 classes, plus a dedup test (`newspaper` merging into a `paper` alert).
- The full suite must stay at 250 passed / 7 skipped.

## 7. Decisions needed

1. Priorities: keep `medium` for all three (no severity change, recommended) or raise any of them?
2. Groups: approve `cup` → cups_containers, `newspaper` → paper_like, `plastic_piece` → soft_packaging?
   For `plastic_piece`, the alternative is `small_items` (merges with `other_litter`) or no group (today's behaviour).
3. Approve editing `class_metadata.json`, plus the test updates in §6.
