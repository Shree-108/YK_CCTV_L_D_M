# Model change control

`model_change_log.csv` is the approved-change record for the litter model: one row per registration, evaluation
decision, activation (deploy), rollback or backup. Append rows; never edit or delete old ones.

| column | content |
|---|---|
| date | YYYY-MM-DD |
| change | register · evaluate · activate · rollback · backup · baseline · dataset |
| model_version | as in `models/litter_yolo/<version>` |
| dataset_version | e.g. `cctv_ldm_v4` |
| evaluation_id | `eval_…` id from the Model Lab / `storage/evaluations/` |
| decision | what was decided and why (metrics, test split) |
| approved_by | **required for activate / rollback / deploy** (name and role) |
| notes | anything else |

`python -m monitoring.ldm_monitor` checks this file: each version must exist, every activation needs an approver, and
the active model needs an `activate` or `baseline` entry.
