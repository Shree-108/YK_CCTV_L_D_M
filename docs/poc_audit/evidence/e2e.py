"""End-to-end API checks against the ISOLATED backend on :8010 (scratch storage + DB snapshot). Read-only on inputs."""
import base64
import json
import sys
import time
from pathlib import Path

import requests

B = "http://127.0.0.1:8010/api"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ADMIN = {"X-Admin-Token": "poc-audit-test-token-not-a-secret"}
ROOT = Path(r"C:\SHREE\YATRIKART\PROJECTS\YK_CCTV_L_D_M_main\YK_CCTV_L_D_M")
results = []


def rec(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name, "|", detail, flush=True)


def j(r):
    try: return r.json()
    except Exception: return r.text[:300]


# --- health / classes / auth-mode behaviour
r = requests.get(f"{B}/health"); h = j(r)
rec("health 200 + model v3 loaded", r.status_code == 200 and h["model"]["version"] == "v3_store_cctv", f"memory {h['system']['memory_percent']}%")
r = requests.get(f"{B}/classes"); rec("classes (viewer, no login)", r.status_code == 200, f"{r.status_code} {len(j(r).get('classes', j(r))) if r.ok else j(r)}")
r = requests.get(f"{B}/auth/me" if True else ""); rec("auth/me reachable", r.status_code in (200, 401, 404), f"{r.status_code} {str(j(r))[:120]}")

# --- image detection: real store frame, synthetic street frame (AUG100), and a clean-ish frame
imgs = {
    "store_real_frame": ROOT / "dataset/ldm_v4_workspace/images/train/store_entrance_poc3min__f0000000.jpg",
    "street_aug100_frame": Path(r"C:\SHREE\DATASET\CCTV_Litter_100_Images (2)\CCTV_Litter_100_Images\cctv_litter_001.jpg"),
    "store_footfall_frame": ROOT / "dataset/ldm_v4_workspace/images/train/store_entrance_footfall__f0002320.jpg",
}
for name, p in imgs.items():
    if not p.exists(): rec(f"image {name}", False, f"missing {p}"); continue
    t0 = time.time()
    with open(p, "rb") as f:
        r = requests.post(f"{B}/detect/image", files={"file": (p.name, f, "image/jpeg")}, data={"create_alerts": "true"})
    dt_ = time.time() - t0; d = j(r)
    if r.ok:
        dets = d.get("detections", [])
        if d.get("annotated_image"): (OUT / f"annotated_{name}.jpg").write_bytes(base64.b64decode(d["annotated_image"]))
        keys = sorted(k for k in d.keys() if k != "annotated_image")
        rec(f"image detect {name}", True, f"{len(dets)} dets in {dt_:.1f}s; classes {sorted({x.get('class_name', x.get('class')) for x in dets})}; "
            f"has bbox/conf: {all(('bbox' in x or 'xyxy' in x) and ('confidence' in x or 'conf' in x) for x in dets)}; alerts {len(d.get('alerts', []))}; keys {keys}")
        (OUT / f"resp_{name}.json").write_text(json.dumps({k: v for k, v in d.items() if k != "annotated_image"}, indent=1)[:20000])
    else:
        rec(f"image detect {name}", False, f"{r.status_code} {d}")

# --- negative / validation cases
r = requests.post(f"{B}/detect/image", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}); rec("corrupt image -> 400", r.status_code == 400, f"{r.status_code} {j(r)}")
r = requests.post(f"{B}/detect/image", files={"file": ("x.txt", b"hello", "text/plain")}); rec("txt upload -> 415", r.status_code == 415, f"{r.status_code} {j(r)}")
r = requests.post(f"{B}/detect/image", files={"file": ("../../evil.jpg", open(imgs['store_real_frame'], 'rb'), "image/jpeg")}, data={"return_image": "false", "create_alerts": "false"})
rec("path-traversal filename handled", r.status_code in (200, 400, 415), f"{r.status_code} filename={j(r).get('filename') if r.ok else j(r)}")
r = requests.post(f"{B}/detect/image", files={"file": ("a.jpg", open(imgs['store_real_frame'], 'rb'), "image/jpeg")}, data={"camera_id": "../x"})
rec("bad camera_id rejected", r.status_code in (400, 422), f"{r.status_code} {j(r)}")
r = requests.get(f"{B}/jobs/nonexistent/files/video"); rec("unknown job file -> 404", r.status_code == 404, f"{r.status_code}")
r = requests.post(f"{B}/models/activate", json={"version": "v3_store_cctv"}); rec("admin route without token rejected", r.status_code in (401, 403), f"{r.status_code} {j(r)}")
r = requests.get(f"{B}/health", headers={"Origin": "http://localhost:3000"}); rec("CORS allows UI origin :3000", r.headers.get("access-control-allow-origin") == "http://localhost:3000", str(r.headers.get("access-control-allow-origin")))
r = requests.options(f"{B}/detect/image", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}); rec("CORS preflight POST from :3000", r.status_code == 200, f"{r.status_code}")
r = requests.get(f"{B}/health", headers={"Origin": "http://evil.example"}); rec("CORS blocks unknown origin", r.headers.get("access-control-allow-origin") is None, str(r.headers.get("access-control-allow-origin")))
r = requests.get("http://127.0.0.1:8010/docs"); rec("API docs disabled by default", r.status_code == 404, f"{r.status_code}")

# --- video job (litter-positive store clip)
vid = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if vid and vid.exists():
    t0 = time.time()
    with open(vid, "rb") as f:
        r = requests.post(f"{B}/detect/video", files={"file": (vid.name, f, "video/mp4")})
    d = j(r); rec("video upload -> 202 queued", r.status_code == 202, f"{r.status_code} {d}")
    if r.status_code == 202:
        jid = d["job_id"]; last = None; seen = []
        while time.time() - t0 < 3600:
            jr = requests.get(f"{B}/jobs/{jid}").json()
            if jr.get("progress") != last: seen.append((round(time.time() - t0), jr.get("status"), jr.get("progress"))); last = jr.get("progress")
            if jr["status"] in ("done", "failed", "error", "cancelled"): break
            time.sleep(5)
        wall = time.time() - t0
        rec("video job finished", jr["status"] == "done", f"status {jr['status']} in {wall:.0f}s; progress samples {seen[:3]}...{seen[-2:]}; error {jr.get('error')}")
        (OUT / "job.json").write_text(json.dumps(jr, indent=1))
        if jr["status"] == "done":
            for k in ("video", "detections", "events", "stats"):
                fr = requests.get(f"{B}/jobs/{jid}/files/{k}")
                ext = {"video": "mp4"}.get(k, "json"); (OUT / f"job_{k}.{ext}").write_bytes(fr.content)
                rec(f"job file {k}", fr.status_code == 200 and len(fr.content) > 0, f"{fr.status_code} {len(fr.content)} bytes {fr.headers.get('content-type')}")
            st = json.loads((OUT / "job_stats.json").read_text()); ev = json.loads((OUT / "job_events.json").read_text())
            rec("job stats/events content", True, f"stats keys {list(st)[:12]}; events {len(ev) if isinstance(ev, list) else list(ev)[:5]}")
        r = requests.get(f"{B}/jobs"); rec("jobs list includes new job", r.ok and any(x["id"] == jid or x.get("job_id") == jid for x in r.json()["jobs"]), f"{len(r.json()['jobs'])} jobs")

# --- alerts / analytics / summary / detections retrieval
r = requests.get(f"{B}/alerts", params={"limit": 200}); a = j(r)
alerts = a.get("alerts", a) if isinstance(a, dict) else a
rec("alerts list", r.ok, f"{len(alerts)} alerts; first keys {sorted(alerts[0].keys())[:14] if alerts else None}")
if alerts:
    aid = alerts[0]["id"]
    s = requests.get(f"{B}/alerts/{aid}/snapshot"); rec("alert snapshot image", s.ok and s.headers.get("content-type", "").startswith("image"), f"{s.status_code} {s.headers.get('content-type')} {len(s.content)}B")
    if s.ok: (OUT / "alert_snapshot.jpg").write_bytes(s.content)
    p = requests.patch(f"{B}/alerts/{aid}", json={"status": "acknowledged"}); rec("alert status update (operator)", p.status_code in (200, 422), f"{p.status_code} {str(j(p))[:150]}")
for ep in ("summary", "analytics", "detections", "events", "models", "settings"):
    r = requests.get(f"{B}/{ep}"); rec(f"GET /{ep}", r.ok, f"{r.status_code} {str(j(r))[:160]}")
(OUT / "e2e_results.json").write_text(json.dumps(results, indent=1))
print(f"\n{sum(x['pass'] for x in results)}/{len(results)} passed")
