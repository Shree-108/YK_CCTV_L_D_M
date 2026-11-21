"""Call the ai_garbage_detection exporter after a finished run without any risk to that run.

The exporter is imported and called inside the try, so not even an import error of backend/services/garbage_export.py
can escape into the job runners, whose outer error handlers would otherwise turn a finished job into a failed one.
Only the standard library is imported at module level. A failure is logged and recorded as
``<out_dir>/ai_garbage_export.json`` with ``ok: false``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

STATUS_FILE = "ai_garbage_export.json"


def safe_auto_export(state: Any, *, kind: str, source_id: str, out_dir: Optional[Path]) -> Optional[dict[str, Any]]:
    """garbage_export.auto_export(); never raises. Returns its result, or the failure record."""
    try:
        from backend.services import garbage_export
        return garbage_export.auto_export(state, kind=kind, source_id=source_id, out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - an export problem must never change the finished job
        log.exception("ai_garbage_detection export of %s %s raised; the job itself is unaffected", kind, source_id)
        res = {"ok": False, "schema": "ai_garbage_detection", "source_id": source_id,
               "error": f"{type(exc).__name__}: {exc}"}
        if out_dir is not None:
            try:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / STATUS_FILE).write_text(json.dumps(res, indent=2), encoding="utf-8")
            except Exception:  # noqa: BLE001
                log.exception("could not write %s for %s", STATUS_FILE, source_id)
        return res
