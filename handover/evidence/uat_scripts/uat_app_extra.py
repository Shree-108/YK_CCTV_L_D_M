import json, tempfile, time
from pathlib import Path
import cv2
from fastapi.testclient import TestClient
from backend.core.config import AppConfig
from backend.main import create_app
R = []
def check(i, name, ok, detail=""):
    R.append((i, name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + i + " " + name + (f"  [{detail}]" if detail else ""))
tmp = Path(tempfile.mkdtemp(prefix="ldm_uat_"))
def cfg(max_mb=500):
    c = AppConfig(); c.storage_dir = tmp / "storage"; c.db_path = tmp / "storage" / "uat.db"
    c.admin_token = "uat-admin-token"; c.start_cameras = False; c.max_video_mb = max_mb; c.ensure_dirs(); return c
src = cv2.VideoCapture(r"C:\SHREE\Screen Captures\cctv_litter_poc_3min_h264.mp4"); fps = src.get(5)
clip = tmp / "clip10s.mp4"; vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 368))
for _ in range(int(fps * 10)):
    ok, f = src.read(); vw.write(f)
vw.release()
ADMIN = {"Authorization": "Bearer uat-admin-token"}
# A16 oversize upload
with TestClient(create_app(cfg(max_mb=1))) as c:
    r = c.post("/api/detect/video", files={"file": ("big.mp4", clip.read_bytes(), "video/mp4")})
    check("A16", "video above the upload limit is refused with a clear message", r.status_code == 413, f"HTTP {r.status_code} {r.text[:110]}")
# A17 long video vs max_video_seconds, A18 settings persist across restart
with TestClient(create_app(cfg())) as c:
    r = c.put("/api/settings", json={"video": {"max_video_seconds": 4}}, headers=ADMIN)
    check("A18a", "admin can change a setting (isolated copy)", r.status_code == 200, f"HTTP {r.status_code}")
    job = c.post("/api/detect/video", files={"file": ("clip10s.mp4", clip.read_bytes(), "video/mp4")}).json()["job_id"]
    for _ in range(120):
        st = c.get(f"/api/jobs/{job}").json()
        if st["status"] in ("done", "failed"): break
        time.sleep(1)
    stats = c.get(f"/api/jobs/{job}/files/stats").json()
    analysed_s = stats.get("frames_analysed", 0) * stats.get("sample_every_n_frames", 1) / stats["video"]["fps"]
    flag = [k for k in json.dumps(stats).split('"') if "trunc" in k.lower() or "partial" in k.lower()]
    check("A17", "10 s video with a 4 s limit: job finishes safely", st["status"] == "done", f"analysed ~{analysed_s:.1f} s of {stats['video']['duration_sec']} s")
    check("A17b", "truncation is visible in the job result / stats", bool(flag), f"keys mentioning truncation: {flag or 'none'}")
with TestClient(create_app(cfg())) as c:
    s = c.get("/api/settings").json()
    v = (s.get("settings") or s).get("video", {})
    h = c.get("/api/health").json()
    check("A18", "after restart: changed setting and active model are kept", v.get("max_video_seconds") == 4 and h["model"]["version"] == "v3_store_cctv",
          f"max_video_seconds={v.get('max_video_seconds')}, model={h['model']['version']}")
print(f"{sum(x[2] for x in R)}/{len(R)} passed")
import shutil; shutil.rmtree(tmp, ignore_errors=True)
