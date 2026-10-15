"""Read-only smoke test of the EXISTING LDM integration with the production model, in an isolated temp storage."""
import io, json, sys, tempfile, time
from pathlib import Path
import cv2, numpy as np
from fastapi.testclient import TestClient
from backend.core.config import AppConfig, PROJECT_ROOT
from backend.main import create_app

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

tmp = Path(tempfile.mkdtemp(prefix="ldm_smoke_"))
cfg = AppConfig(); cfg.storage_dir = tmp / "storage"; cfg.db_path = tmp / "storage" / "smoke.db"
cfg.admin_token = "smoke-admin-token"; cfg.start_cameras = False; cfg.ensure_dirs()
print("models dir (read-only):", cfg.yolo_models_dir, "| temp storage:", cfg.storage_dir)
frame = cv2.imread(str(next((PROJECT_ROOT / "dataset/ldm_v4_workspace/images/train").glob("store_entrance_poc3min__*.jpg"))))
ok, jpg = cv2.imencode(".jpg", frame)
with TestClient(create_app(cfg)) as c:
    h = c.get("/api/health").json()
    check("health ok", h["status"] == "ok", h["status"])
    check("model initialised = production v3", h["model"] and h["model"]["version"] == "v3_store_cctv" and not h["model_error"], str(h["model"] and h["model"]["version"]))
    check("no notification provider configured (no real alerts can be sent)", not any(n["configured"] for n in h["notifications"]), str(h["notifications"]))
    names = c.get("/api/classes").json()
    check("classes endpoint", "plastic_bag" in json.dumps(names))
    t0 = time.perf_counter()
    r = c.post("/api/detect/image", files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")})
    dt_ms = (time.perf_counter() - t0) * 1000
    j = r.json() if r.status_code == 200 else {}
    dets = j.get("detections", [])
    check("image upload + inference", r.status_code == 200, f"HTTP {r.status_code}, {len(dets)} detections, {dt_ms:.0f} ms round trip")
    if dets:
        d = dets[0]; check("detection has class, confidence, bbox", {"class", "confidence", "bbox"} <= set(d) or {"cls_name","confidence","bbox"} <= set(d), str(sorted(d))[:160])
    check("annotated image returned", any(k for k in j if "annot" in k.lower()), str([k for k in j])[:200])
    r = c.post("/api/detect/image", files={"file": ("broken.jpg", b"not an image at all", "image/jpeg")})
    check("corrupt image rejected gracefully", r.status_code == 400, f"HTTP {r.status_code} {r.text[:90]}")
    r = c.post("/api/detect/image", files={"file": ("notes.txt", b"hello", "text/plain")})
    check("unsupported file type rejected", 400 <= r.status_code < 500, f"HTTP {r.status_code} {r.text[:90]}")
    # short real clip: first 10 s of the store video (read-only source)
    src = cv2.VideoCapture(r"C:\SHREE\Screen Captures\cctv_litter_poc_3min_h264.mp4"); fps = src.get(5)
    clip = tmp / "clip10s.mp4"; vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 368))
    for _ in range(int(fps * 10)):
        okf, f = src.read()
        if not okf: break
        vw.write(f)
    vw.release(); src.release()
    t0 = time.perf_counter()
    r = c.post("/api/detect/video", files={"file": ("clip10s.mp4", clip.read_bytes(), "video/mp4")})
    check("video upload accepted (202 + job id)", r.status_code == 202 and "job_id" in r.json(), f"HTTP {r.status_code}")
    job = r.json().get("job_id"); st = {}
    while job and time.perf_counter() - t0 < 600:
        st = c.get(f"/api/jobs/{job}").json()
        if st.get("status") in ("done", "failed", "error"): break
        time.sleep(1)
    check("video job finished", st.get("status") == "done", f"status {st.get('status')} in {time.perf_counter()-t0:.0f} s; {str(st.get('error') or '')[:100]}")
    dj = c.get(f"/api/jobs/{job}/files/detections")
    check("video detections file", dj.status_code == 200, f"HTTP {dj.status_code}")
    if dj.status_code == 200:
        body = dj.json(); items = body if isinstance(body, list) else body.get("detections") or body.get("frames") or []
        txt = json.dumps(body)[:20000]
        check("video detections carry timestamps", "video_time" in txt or "time_sec" in txt or "timestamp" in txt, f"{len(items)} records")
    vv = c.get(f"/api/jobs/{job}/files/video")
    check("annotated video served", vv.status_code == 200 and len(vv.content) > 1000, f"HTTP {vv.status_code}, {len(vv.content)//1024} KB")
    r = c.post("/api/models/activate", json={"version": "does_not_exist"}, headers={"Authorization": "Bearer smoke-admin-token"})
    check("activating unknown model refused, v3 stays active", r.status_code == 404 and c.get("/api/health").json()["model"]["version"] == "v3_store_cctv", f"HTTP {r.status_code}")
    r = c.post("/api/models/activate", json={"version": "v3_store_cctv"})
    check("model activation requires admin", r.status_code in (401, 403), f"HTTP {r.status_code}")
# missing weights: empty models dir
tmp2 = Path(tempfile.mkdtemp(prefix="ldm_smoke_nomodel_"))
cfg2 = AppConfig(); cfg2.storage_dir = tmp2 / "storage"; cfg2.db_path = tmp2 / "storage" / "s.db"; cfg2.models_dir = tmp2 / "models"
cfg2.admin_token = "x"; cfg2.start_cameras = False; cfg2.ensure_dirs()
with TestClient(create_app(cfg2)) as c:
    h = c.get("/api/health").json()
    check("missing weights: API still starts and reports it", h["model"] is None or h["model_error"], f"model={h['model']} error={str(h['model_error'])[:80]}")
    r = c.post("/api/detect/image", files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")})
    check("missing weights: detection refused gracefully", r.status_code in (409, 503, 500) and r.status_code != 200, f"HTTP {r.status_code} {r.text[:90]}")
print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} passed")
