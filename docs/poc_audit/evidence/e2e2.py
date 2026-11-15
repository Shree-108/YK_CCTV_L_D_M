"""Second batch against the isolated backend: CSV exports + 2 more videos (crowded footfall, indoor 'clean')."""
import json, sys, time
from pathlib import Path
import requests

B = "http://127.0.0.1:8010/api"; OUT = Path(sys.argv[1])
for kind in ("alerts", "detections"):
    r = requests.get(f"{B}/export/{kind}.csv")
    (OUT / f"export_{kind}.csv").write_bytes(r.content)
    lines = r.text.splitlines()
    print(f"CSV {kind}: {r.status_code} {r.headers.get('content-type')} rows={len(lines)-1} header={lines[0][:160] if lines else None}")

for label, vid in (("footfall", sys.argv[2]), ("indoor_clean", sys.argv[3])):
    p = Path(vid)
    if not p.exists(): print(label, "missing", p); continue
    a0 = requests.get(f"{B}/analytics").json()["alerts_total"]; t0 = time.time()
    with open(p, "rb") as f:
        r = requests.post(f"{B}/detect/video", files={"file": (p.name, f, "video/mp4")}, data={"camera_id": f"test_{label}"})
    if r.status_code != 202: print(label, "upload", r.status_code, r.text[:200]); continue
    jid = r.json()["job_id"]
    while True:
        jr = requests.get(f"{B}/jobs/{jid}").json()
        if jr["status"] in ("done", "failed"): break
        time.sleep(5)
    st = requests.get(f"{B}/jobs/{jid}/files/stats").json() if jr["status"] == "done" else {}
    ev = requests.get(f"{B}/jobs/{jid}/files/events").json() if jr["status"] == "done" else []
    a1 = requests.get(f"{B}/analytics").json()["alerts_total"]
    (OUT / f"job_{label}_stats.json").write_text(json.dumps(st, indent=1)); (OUT / f"job_{label}_events.json").write_text(json.dumps(ev, indent=1))
    print(f"VIDEO {label}: {jr['status']} in {time.time()-t0:.0f}s; video={st.get('video')}; frames_analysed={st.get('frames_analysed')}; "
          f"confirmed_objects={st.get('unique_confirmed_objects')}; by_class={st.get('confirmed_by_class')}; new_alerts={a1-a0}; "
          f"detections_by_status={st.get('detections_by_status')}; error={jr.get('error')}")
