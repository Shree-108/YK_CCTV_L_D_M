# S3 Storage Structure (reusable standard)

Taken from the uniform project (`YK_CCTV_U_D_M`, bucket `yk-cctv-prediction`,
code in `backend/core/s3.py` + `backend/extraction/s3_output.py`). This document
turns that layout into a standard any Yatrikart project can adopt.

---

## 1. What the uniform project does today

```
s3://yk-cctv-prediction/
└── MM-YYYY/                                 e.g. 11-2026/
    └── DD-MM-YYYY/                          e.g. 09-11-2026/
        ├── cart_<N>/
        │   └── input_data/                  raw videos, under the RECORDING day
        │       └── <safe original name>.mp4
        └── card_<N>/
            └── output_data/                 results, under the PROCESSING day (today, IST)
                ├── card<N>_YYYY_MM_DD.csv           one CSV per cart per day, append-only
                └── card<N>_YYYY_MM_DD_pic<K>.jpg    image for CSV row "pic<K>"
```

Real example:

```
s3://yk-cctv-prediction/11-2026/09-11-2026/cart_50/input_data/20261109_..._183009.mp4
s3://yk-cctv-prediction/11-2026/09-11-2026/card_50/output_data/card50_2026_11_09.csv
s3://yk-cctv-prediction/11-2026/09-11-2026/card_50/output_data/card50_2026_11_09_pic1.jpg
```

CSV header (uniform):
`cart_id, Store Name, Date, Time Stamp, img name, confidence score, remark`

> **Known quirk:** input uses `cart_<N>` but output uses `card_<N>`, so one cart has
> two sibling folders per day. Don't copy that into new projects; use one entity
> name everywhere (see section 2).

---

## 2. Standard structure for any project

```
s3://<bucket>/
└── [<project>/]                              only if the bucket is SHARED by several projects
    └── MM-YYYY/
        └── DD-MM-YYYY/
            └── <entity>_<id>/                cart_50, store_12, camera_3 ... same name in input & output
                ├── input_data/               raw inputs  → filed under the day they were RECORDED
                │   └── <safe original name>.<ext>
                └── output_data/              results     → filed under the day they were PROCESSED
                    ├── <entity><id>_YYYY_MM_DD.csv
                    └── <entity><id>_YYYY_MM_DD_pic<K>.<jpg|png>
```

| Part | Format | Example | Notes |
|---|---|---|---|
| `<bucket>` | from `.env` | `yk-litter-detection-data` | One bucket per project is preferred |
| `<project>/` | lowercase slug | `uniform/` | Add only when projects share a bucket |
| Month folder | `MM-YYYY` | `11-2026` | Groups days; makes month listing cheap |
| Day folder | `DD-MM-YYYY` | `25-11-2026` | Same day prefix for input and output |
| Entity folder | `<entity>_<id>` | `cart_50` | `[A-Za-z0-9_-]{1,64}`; bare `50` → `cart_50` |
| `input_data/` | fixed | | Recording date of the source file |
| `output_data/` | fixed | | Processing date, **Asia/Kolkata**, never the video date |
| Daily CSV | `<entity><id>_YYYY_MM_DD.csv` | `cart50_2026_11_25.csv` | Note the different date order to the folder |
| Row image | `<csv stem>_pic<K>.<ext>` | `cart50_2026_11_25_pic7.jpg` | `pic<K>` = the CSV row's `img name` |
| Name clash | `<csv stem>_pic<K>_<run_id>.<ext>` | | Used only if a different file already holds the name |

### File name sanitising (inputs)
Keep only the base name. Replace anything outside `A-Z a-z 0-9 . _ -` with `_`,
strip leading dots, limit the stem to 100 characters, lowercase the extension.
NVR names such as `20261104_..._192126.mp4` pass through unchanged.

---

## 3. Write rules (copy these, not just the paths)

1. **Never overwrite, never delete.**
   New objects use a conditional PUT with `IfNoneMatch="*"`. If the key is taken
   by the *same* file (same SHA-256), treat it as already uploaded. If it's taken by
   a *different* file, use the `_<run_id>` name.
2. **The daily CSV is append-only.**
   S3 has no append, so: GET (keep its ETag) → add rows below → PUT with
   `IfMatch=<etag>` (or `IfNoneMatch="*"` for the day's first write). On a 409 or 412
   conflict, re-read and retry (about 5 times). Keep one header and leave earlier rows
   byte-for-byte unchanged. Refuse to write if the existing header differs.
3. **Picture numbering continues.**
   Start the next `pic<K>` from 1 + the highest of: the row count, the highest
   `pic<K>` in the CSV, and the highest `pic<K>` image already in the folder.
4. **Integrity.**
   Send `ChecksumSHA256` with every PUT and store `Metadata={"sha256": ...}`.
   After upload, verify with `head_object(ChecksumMode="ENABLED")` that size and
   SHA-256 match.
5. **Idempotent retries.**
   Record in the DB which run's rows are already in the CSV (`csv_first_row`,
   `sha256`), so a retry never appends the same rows twice. Retries keep the day
   that was chosen on the first attempt.
6. **Credentials come only from `.env`**:
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`,
   `AWS_REGION`, and `<PREFIX>_S3_BUCKET`. Pass them to boto3 explicitly. Never
   log them or return them from an API.
7. **Upload in the background.**
   Use a queue worker with these states:
   `queued → generating_report → uploading → uploading_images → completed`
   (or `failed` + the failed stage). Resume unfinished uploads when the app starts.
   Provide a retry endpoint.

---

## 4. Drop-in path helpers (Python)

```python
import os, re
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "Asia/Kolkata")
ENTITY = "cart"                      # change per project: "store", "camera", ...
PROJECT = ""                         # e.g. "uniform" if the bucket is shared, else ""
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

def bucket() -> str:
    return os.environ["PROJECT_S3_BUCKET"]          # rename the variable per project

def today() -> date:
    return datetime.now(IST).date()                 # processing day for outputs

def entity_folder(value: str) -> str:
    v = value.strip()
    v = f"{ENTITY}_{v}" if v.isdigit() else v
    if not ID_RE.match(v):
        raise ValueError(f"bad {ENTITY} id: {value!r}")
    return v

def entity_number(folder: str) -> str:
    return re.sub(r"\D", "", folder) or folder

def day_prefix(d: date) -> str:
    root = f"{PROJECT}/" if PROJECT else ""
    return f"{root}{d:%m-%Y}/{d:%d-%m-%Y}/"

def input_prefix(recorded: date, ent: str) -> str:
    return f"{day_prefix(recorded)}{entity_folder(ent)}/input_data/"

def output_prefix(processed: date, ent: str) -> str:
    return f"{day_prefix(processed)}{entity_folder(ent)}/output_data/"

def daily_csv_key(processed: date, ent: str) -> str:
    n = entity_number(entity_folder(ent))
    return f"{output_prefix(processed, ent)}{ENTITY}{n}_{processed:%Y_%m_%d}.csv"

def image_key(csv_key: str, pic: int, ext: str = ".jpg", run_id: str | None = None) -> str:
    stem = csv_key.rsplit(".", 1)[0]
    return f"{stem}_pic{pic}{'_' + run_id if run_id else ''}{ext}"
```

The full safe-write implementation (`put_new`, `append_csv`, `verify`, `list_keys`)
is in `backend/core/s3.py` in either project. Copy that file and change only the
bucket environment variable and the `ENTITY` name.

---

## 5. Adopting it in an existing project — checklist

- [ ] Choose the bucket. Add `<project>/` only if the bucket is shared.
- [ ] Choose the entity name (`cart`, `store`, `camera`) and use it in **both** input and output paths.
- [ ] Add `AWS_*` + `<PREFIX>_S3_BUCKET` to `.env`, and placeholders to `.env.example`. Never commit keys.
- [ ] Copy `s3.py` (path helpers + `put_new` / `append_csv` / `verify`).
- [ ] Define the daily CSV header once. It must never change for an existing day's file.
- [ ] Add an on/off feature flag (e.g. `<PREFIX>_S3_UPLOAD=on`) and turn it off in tests.
- [ ] Upload in a background worker, record state per run, and add a retry endpoint.
- [ ] Test against a fake S3 (no network): layout, no-overwrite, append + conflict retry, idempotent retry.

---

## 6. How this project (litter, `YK_CCTV_L_D_M`) applies it

Bucket `yk-litter-detection-data` (ap-south-1). It is a dedicated bucket, so there is no `<project>/` level.
The entity is `cart` in both input and output paths, which fixes the uniform `card_` quirk.

```
s3://yk-litter-detection-data/11-2026/25-11-2026/cart_50/input_data/<video>.mp4
s3://yk-litter-detection-data/11-2026/25-11-2026/cart_50/output_data/cart50_2026_11_25.csv
s3://yk-litter-detection-data/11-2026/25-11-2026/cart_50/output_data/cart50_2026_11_25_pic1.jpg
```

Configuration: `LDM_S3_BUCKET`, `LDM_S3_UPLOAD` (in `backend/core/config.py`).
Code: `backend/core/s3.py`, `backend/services/s3_sync.py`.
