"""Database access for the four-screenshot analysis (tables sampled_media / sampled_jobs / sampled_frames)."""
from __future__ import annotations

import json
from typing import Any, Optional

from backend.database.db import Database

_JSON = ("roi_json", "proposal_json", "counts_json", "result_json")


def _decode(r: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if r is None:
        return None
    for k in _JSON:
        if k in r:
            r[k.removesuffix("_json")] = json.loads(r.pop(k) or "null")
    for k in ("auto_accept", "create_alerts"):
        if k in r:
            r[k] = bool(r[k])
    return r


def _encode(fields: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in fields.items():
        if k in ("roi", "proposal", "counts", "result"):
            out[f"{k}_json"] = None if v is None else json.dumps(v)
        else:
            out[k] = v
    return out


class SampledRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ media
    def insert_media(self, m: dict[str, Any]) -> None:
        cols = ["id", "created_at", "filename", "path", "sha256", "size_bytes", "width", "height", "fps", "frames",
                "duration_sec", "suggested_recording_start"]
        self.db._write(f"INSERT INTO sampled_media({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                       [m.get(c) for c in cols])

    def media(self, media_id: str) -> Optional[dict[str, Any]]:
        return self.db._one("SELECT * FROM sampled_media WHERE id=?", (media_id,))

    def media_by_sha(self, sha: str, exclude: str = "") -> list[dict[str, Any]]:
        return self.db._query("SELECT * FROM sampled_media WHERE sha256=? AND id<>? ORDER BY created_at", (sha, exclude))

    def count_media(self) -> int:
        return int(self.db._one("SELECT COUNT(*) AS n FROM sampled_media")["n"])  # type: ignore[index]

    # ------------------------------------------------------------------ jobs
    def insert_job(self, j: dict[str, Any]) -> None:
        j = _encode(j)
        cols = list(j)
        self.db._write(f"INSERT INTO sampled_jobs({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                       [j[c] for c in cols])

    def update_job(self, job_id: str, **fields: Any) -> None:
        f = _encode(fields)
        if f:
            self.db._write(f"UPDATE sampled_jobs SET {', '.join(f'{k}=?' for k in f)} WHERE id=?", [*f.values(), job_id])

    def job(self, job_id: str) -> Optional[dict[str, Any]]:
        return _decode(self.db._one("SELECT * FROM sampled_jobs WHERE id=?", (job_id,)))

    def jobs(self, where: str = "", params: tuple = (), limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        sql = ("SELECT j.*, m.filename AS video_name, m.duration_sec, m.width, m.height, m.fps, m.frames, m.size_bytes "
               f"FROM sampled_jobs j JOIN sampled_media m ON m.id=j.media_id {where} "
               "ORDER BY j.created_at DESC LIMIT ? OFFSET ?")
        return [_decode(r) for r in self.db._query(sql, (*params, limit, offset))]  # type: ignore[misc]

    def count_jobs(self, where: str = "", params: tuple = ()) -> int:
        return int(self.db._one(f"SELECT COUNT(*) AS n FROM sampled_jobs j JOIN sampled_media m ON m.id=j.media_id {where}",
                                params)["n"])  # type: ignore[index]

    def active_job_for_media(self, media_id: str) -> Optional[dict[str, Any]]:
        return _decode(self.db._one("SELECT * FROM sampled_jobs WHERE media_id=? AND state IN "
                                    "('queued','running','awaiting_roi') ORDER BY created_at DESC LIMIT 1", (media_id,)))

    def active_auto_job_for_sha(self, sha: str) -> Optional[dict[str, Any]]:
        """A queued/running automatic job of any upload with this file content (duplicate drop protection)."""
        return _decode(self.db._one("SELECT j.* FROM sampled_jobs j JOIN sampled_media m ON m.id=j.media_id "
                                    "WHERE m.sha256=? AND j.mode='auto' AND j.state IN ('queued','running') "
                                    "ORDER BY j.created_at DESC LIMIT 1", (sha,)))

    def delete_media(self, media_id: str) -> None:
        """Remove a media row that no job references (a duplicate upload that was not used)."""
        self.db._write("DELETE FROM sampled_media WHERE id=? AND NOT EXISTS "
                       "(SELECT 1 FROM sampled_jobs WHERE media_id=?)", (media_id, media_id))

    def jobs_in_states(self, states: tuple[str, ...]) -> list[dict[str, Any]]:
        q = ",".join("?" * len(states))
        return [_decode(r) for r in self.db._query(f"SELECT * FROM sampled_jobs WHERE state IN ({q})", states)]  # type: ignore[misc]

    # ------------------------------------------------------------------ frames
    def insert_frames(self, rows: list[dict[str, Any]]) -> None:
        cols = ["job_id", "idx", "frame_idx", "video_time_sec", "status"]
        with self.db._wlock, self.db.connect() as c:
            c.executemany(f"INSERT INTO sampled_frames({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          [tuple(r[k] for k in cols) for r in rows])

    def update_frame(self, job_id: str, idx: int, **fields: Any) -> None:
        f = _encode(fields)
        self.db._write(f"UPDATE sampled_frames SET {', '.join(f'{k}=?' for k in f)} WHERE job_id=? AND idx=?",
                       [*f.values(), job_id, idx])

    def frames(self, job_id: str) -> list[dict[str, Any]]:
        return [_decode(r) for r in self.db._query("SELECT * FROM sampled_frames WHERE job_id=? ORDER BY idx", (job_id,))]  # type: ignore[misc]

    def frame(self, job_id: str, idx: int) -> Optional[dict[str, Any]]:
        return _decode(self.db._one("SELECT * FROM sampled_frames WHERE job_id=? AND idx=?", (job_id, idx)))
