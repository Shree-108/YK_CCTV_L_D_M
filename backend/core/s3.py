"""S3 layout + safe writes for the litter-detection bucket (``LDM_S3_BUCKET``, e.g. ``yk-litter-detection-data``).

The bucket is dedicated to this project, so there is no ``<project>/`` level::

    s3://<bucket>/
    └── MM-YYYY/
        └── DD-MM-YYYY/
            └── cart_<N>/
                ├── input_data/            raw videos      - filed under the RECORDING day
                │   └── <safe original name>.mp4
                └── output_data/           results         - filed under the PROCESSING day (today, Asia/Kolkata)
                    ├── cart<N>_YYYY_MM_DD.csv              one CSV per cart per day, only ever appended to
                    └── cart<N>_YYYY_MM_DD_pic<K>.jpg       image of the CSV row whose ``pic`` is ``pic<K>``

Rules:
* the entity name is ``cart`` in input AND output paths (no cart/card mismatch);
* nothing is ever deleted or overwritten: new objects use ``If-None-Match: *``, the CSV is appended with ``If-Match``
  on the ETag that was read (``If-None-Match: *`` for the day's first write) and retried on a conflict;
* every upload carries its SHA-256 (S3 rejects a corrupted body) and is verified afterwards by size + SHA-256;
* credentials come only from the environment / ``.env``: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
  LDM_S3_BUCKET.
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import hashlib
import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

IST = dt.timezone(dt.timedelta(hours=5, minutes=30), "Asia/Kolkata")
INPUT_FOLDER, OUTPUT_FOLDER = "input_data", "output_data"
KIND = "cart"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PIC_RE = re.compile(r"_pic(\d+)\.jpg$")
APPEND_RETRIES = 5


class S3Error(RuntimeError):
    pass


class Exists(S3Error):
    """The key already exists (If-None-Match: * refused the write)."""


# ============================================================================================ layout
def today() -> dt.date:
    return dt.datetime.now(IST).date()


def entity_id(value: Any, kind: str = KIND) -> str:
    """``50`` / ``'50'`` / ``'cart_50'`` -> ``'cart_50'``."""
    v = str(value).strip()
    if v.isdigit():
        v = f"{kind}_{v}"
    if not ID_RE.fullmatch(v):
        raise S3Error(f"bad {kind} id {value!r}")
    return v


def number(eid: str) -> str:
    return re.sub(r"\D", "", eid) or eid


def safe_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "video").name).strip("._") or "video"
    return n[:200]


def day_prefix(day: dt.date) -> str:
    return f"{day:%m-%Y}/{day:%d-%m-%Y}/"


def input_prefix(day: dt.date, eid: str) -> str:
    return f"{day_prefix(day)}{eid}/{INPUT_FOLDER}/"


def output_prefix(day: dt.date, eid: str) -> str:
    return f"{day_prefix(day)}{eid}/{OUTPUT_FOLDER}/"


def input_key(day: dt.date, eid: str, original_name: str) -> str:
    return input_prefix(day, eid) + safe_name(original_name)


def file_stem(eid: str) -> str:
    """``cart_50`` -> ``cart50``; an id that is not ``cart_<N>`` (e.g. ``cart_unassigned``) is used as it is."""
    m = re.fullmatch(rf"{KIND}_(\d+)", eid)
    return f"{KIND}{m.group(1)}" if m else eid


def daily_csv_key(day: dt.date, eid: str) -> str:
    return f"{output_prefix(day, eid)}{file_stem(eid)}_{day:%Y_%m_%d}.csv"


def image_key(day: dt.date, eid: str, k: int) -> str:
    return daily_csv_key(day, eid)[:-4] + f"_pic{k}.jpg"


# ============================================================================================ client
@dataclass(frozen=True)
class S3Config:
    bucket: str
    region: str
    access_key: str
    secret_key: str

    @classmethod
    def from_env(cls) -> "S3Config":
        cfg = cls(os.getenv("LDM_S3_BUCKET", "").strip(), os.getenv("AWS_REGION", "").strip(),
                  os.getenv("AWS_ACCESS_KEY_ID", "").strip(), os.getenv("AWS_SECRET_ACCESS_KEY", "").strip())
        missing = [n for n, v in (("LDM_S3_BUCKET", cfg.bucket), ("AWS_REGION", cfg.region),
                                  ("AWS_ACCESS_KEY_ID", cfg.access_key),
                                  ("AWS_SECRET_ACCESS_KEY", cfg.secret_key)) if not v]
        if missing:
            raise S3Error(f"S3 not configured; set in .env: {', '.join(missing)}")
        return cfg


def client(cfg: Optional[S3Config] = None) -> Any:
    import boto3  # imported lazily: the app runs without boto3 while S3 upload is off
    cfg = cfg or S3Config.from_env()
    return boto3.client("s3", region_name=cfg.region, aws_access_key_id=cfg.access_key,
                        aws_secret_access_key=cfg.secret_key)


def _code(exc: Exception) -> str:
    r = getattr(exc, "response", None) or {}
    return str(r.get("Error", {}).get("Code") or r.get("ResponseMetadata", {}).get("HTTPStatusCode") or "")


def _is_precondition(exc: Exception) -> bool:
    return _code(exc) in ("PreconditionFailed", "412")


def _is_conflict(exc: Exception) -> bool:
    return _code(exc) in ("ConditionalRequestConflict", "409")


def _is_missing(exc: Exception) -> bool:
    return _code(exc) in ("NoSuchKey", "404", "NotFound")


def sha256_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode()


def sha256_file_b64(path: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return base64.b64encode(h.digest()).decode(), n


# ============================================================================================ safe writes
def verify(s3: Any, bucket: str, key: str, size: int, sha_b64: str) -> None:
    """Raise unless the stored object has exactly this size and SHA-256."""
    h = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    if int(h.get("ContentLength", -1)) != size:
        raise S3Error(f"verify {key}: size {h.get('ContentLength')} != {size}")
    stored = h.get("ChecksumSHA256")
    if stored != sha_b64:
        raise S3Error(f"verify {key}: SHA-256 {stored!r} != {sha_b64!r}")


def put_new(s3: Any, bucket: str, key: str, body: bytes | Path, content_type: str) -> dict[str, Any]:
    """Create ``key``; never overwrites (If-None-Match: *). Raises Exists when the key is taken. Verified."""
    if isinstance(body, Path):
        sha, size = sha256_file_b64(body)
        with open(body, "rb") as f:
            return _put(s3, bucket, key, f, size, sha, content_type)
    return _put(s3, bucket, key, body, len(body), sha256_b64(body), content_type)


def _put(s3: Any, bucket: str, key: str, body: Any, size: int, sha: str, content_type: str,
         **cond: str) -> dict[str, Any]:
    cond = cond or {"IfNoneMatch": "*"}
    try:
        r = s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type, ChecksumSHA256=sha, **cond)
    except Exception as exc:
        if _is_precondition(exc) and "IfNoneMatch" in cond:
            raise Exists(key) from exc
        raise
    verify(s3, bucket, key, size, sha)
    return {"key": key, "size": size, "sha256": sha, "etag": r.get("ETag", "")}


def put_new_or_same(s3: Any, bucket: str, key: str, path: Path, content_type: str) -> dict[str, Any]:
    """Upload a file once. If the key already holds the same bytes, that counts as done; different bytes raise."""
    try:
        return {**put_new(s3, bucket, key, path, content_type), "existed": False}
    except Exists:
        sha, size = sha256_file_b64(path)
        try:
            verify(s3, bucket, key, size, sha)
        except S3Error as exc:
            raise S3Error(f"{key} already exists with different content; left untouched") from exc
        return {"key": key, "size": size, "sha256": sha, "existed": True}


def read_text(s3: Any, bucket: str, key: str) -> tuple[Optional[str], Optional[str]]:
    """(text, etag) of an object, (None, None) when it does not exist."""
    try:
        r = s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _is_missing(exc):
            return None, None
        raise
    return r["Body"].read().decode("utf-8"), r["ETag"]


def csv_rows(text: Optional[str]) -> list[dict[str, str]]:
    if not text or not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text, newline="")))


def append_csv(s3: Any, bucket: str, key: str, columns: Sequence[str], rows_fn: Any) -> dict[str, Any]:
    """Append rows to a CSV object without ever changing existing rows.

    ``rows_fn(existing_rows) -> new_rows`` is called on every attempt (so numbering / de-duplication see the latest
    file). The write is conditional on the object being unchanged since it was read; a conflict re-reads and retries.
    """
    for attempt in range(APPEND_RETRIES):
        text, etag = read_text(s3, bucket, key)
        if text:
            header = next(csv.reader(io.StringIO(text, newline="")), [])
            if header != list(columns):
                raise S3Error(f"{key} has a different header ({header}); it is left untouched")
        existing = csv_rows(text)
        new = rows_fn(existing)
        if not new:
            return {"key": key, "appended": 0, "existing": len(existing)}
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="raise", lineterminator="\r\n")
        if text is None or not text.strip():
            w.writeheader()
            prefix = ""
        else:
            prefix = text if text.endswith("\n") else text + "\r\n"
        w.writerows(new)
        body = (prefix + buf.getvalue()).encode("utf-8")
        cond = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            _put(s3, bucket, key, body, len(body), sha256_b64(body), "text/csv; charset=utf-8", **cond)
            return {"key": key, "appended": len(new), "existing": len(existing)}
        except Exists:
            pass                                   # someone created it between our read and write
        except Exception as exc:
            if not (_is_precondition(exc) or _is_conflict(exc)):
                raise
        time.sleep(0.2 * (attempt + 1))
    raise S3Error(f"{key}: gave up after {APPEND_RETRIES} conflicting writers")


def list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: Optional[str] = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return keys
        token = r.get("NextContinuationToken")


def max_pic(keys: Sequence[str], rows: Sequence[dict[str, str]]) -> int:
    """Highest K used by a ``pic<K>`` image in the folder or the CSV's ``pic`` column."""
    ks = [int(m.group(1)) for k in keys if (m := PIC_RE.search(k))]
    ks += [int(r["pic"][3:]) for r in rows if re.fullmatch(r"pic\d+", r.get("pic") or "")]
    return max(ks, default=0)
