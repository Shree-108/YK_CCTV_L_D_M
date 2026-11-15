"""Browser walkthrough of the DEMO instance (UI :3110, API :8100) with the installed Microsoft Edge (headless).
Inputs are only read. Records console errors, failed requests, screenshots and download behaviour."""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

UI = "http://localhost:3110"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
IMG = sys.argv[2]; VID = sys.argv[3]
log = {"console_errors": [], "page_errors": [], "failed_requests": [], "http_errors": [], "steps": []}


def step(name, ok, detail=""):
    log["steps"].append({"step": name, "pass": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name, "|", detail, flush=True)


def try_download(page, locator, label):
    """Click and classify: real download (with file name/size), navigation away, popup, or nothing."""
    start_url = page.url; got = {}
    page.once("popup", lambda p: got.setdefault("popup", p.url))
    try:
        with page.expect_download(timeout=20000) as d:
            locator.click()
        dl = d.value; dest = OUT / f"dl_{label}_{dl.suggested_filename}"; dl.save_as(dest)
        got.update(kind="download", filename=dl.suggested_filename, bytes=dest.stat().st_size)
    except Exception as e:
        got.update(kind="no-download", err=type(e).__name__)
    time.sleep(1)
    got["navigated_away"] = page.url != start_url
    got["url_after"] = page.url
    if got["navigated_away"]:
        page.go_back(); page.wait_for_load_state("networkidle")
    return got


with sync_playwright() as pw:
    br = pw.chromium.launch(channel="msedge", headless=True)
    ctx = br.new_context(accept_downloads=True, viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.on("console", lambda m: m.type == "error" and log["console_errors"].append({"url": page.url, "text": m.text[:300]}))
    page.on("pageerror", lambda e: log["page_errors"].append({"url": page.url, "text": str(e)[:300]}))
    page.on("requestfailed", lambda r: log["failed_requests"].append({"url": r.url[:200], "err": r.failure}))
    page.on("response", lambda r: r.status >= 400 and log["http_errors"].append({"url": r.url[:200], "status": r.status}))

    # 1. dashboard + connection
    page.goto(UI + "/", wait_until="networkidle"); time.sleep(3)
    body = page.inner_text("body")
    step("dashboard loads", "Yatrikart" in page.content(), f"title={page.title()}")
    step("UI connected to backend", "Backend offline" not in body, "offline banner absent" if "Backend offline" not in body else "OFFLINE")
    page.screenshot(path=OUT / "01_dashboard_before.png", full_page=True)

    # 2. image detection
    page.goto(UI + "/detect/image", wait_until="networkidle")
    page.locator("input[type=file]").first.set_input_files(IMG)
    with page.expect_response(lambda r: "/api/detect/image" in r.url, timeout=120000) as resp:
        page.get_by_role("button", name="Analyze Image").click()
    r = resp.value; data = r.json() if r.ok else {}
    time.sleep(2)
    rects = page.locator("svg rect").count()
    step("image upload + detection", r.ok, f"HTTP {r.status}; detections {data.get('count')}; svg rects drawn {rects}")
    txt = page.inner_text("body")
    step("confidence shown in UI", "%" in txt, "percent values present")
    page.screenshot(path=OUT / "02_image_result.png", full_page=True)

    # 3. video detection (full mode)
    page.goto(UI + "/detect/video?mode=full", wait_until="networkidle")
    page.locator("input[type=file]").first.set_input_files(VID)
    time.sleep(2)
    t0 = time.time()
    with page.expect_response(lambda r: "/api/detect/video" in r.url and r.request.method == "POST", timeout=180000) as resp:
        page.get_by_role("button", name="Analyze Video").first.click()
    r = resp.value; job = r.json() if r.ok else {}
    step("video upload accepted", r.status == 202, f"HTTP {r.status} job {job.get('job_id')}")
    page.screenshot(path=OUT / "03_video_progress.png", full_page=True)
    try:
        page.get_by_text("Download processed video").first.wait_for(timeout=900000)
        step("video processed, results shown", True, f"{time.time() - t0:.0f}s from click")
    except Exception as e:
        step("video processed, results shown", False, str(e)[:200])
    time.sleep(3)
    vinfo = page.evaluate("""async () => { const v = document.querySelector('video'); if (!v) return null;
        v.muted = true; try { await v.play() } catch (e) {}; await new Promise(r => setTimeout(r, 3000));
        return {src: v.currentSrc, readyState: v.readyState, duration: v.duration, currentTime: v.currentTime, error: v.error && v.error.code} }""")
    step("annotated video plays in browser", bool(vinfo) and vinfo["currentTime"] > 0.5 and not vinfo["error"], json.dumps(vinfo))
    page.screenshot(path=OUT / "04_video_result.png", full_page=True)

    # 4. download behaviour on the video page
    d = try_download(page, page.get_by_role("button", name="Download processed video").first, "video_page_mp4")
    step("video page: 'Download processed video'", d.get("kind") == "download" and not d["navigated_away"], json.dumps(d))
    det = page.locator("summary", has_text="Technical details")
    if det.count():
        det.first.click(); time.sleep(1)
        d = try_download(page, page.get_by_role("button", name="stats.json").first, "video_page_stats")
        step("video page: 'stats.json'", d.get("kind") == "download" and not d["navigated_away"], json.dumps(d))

    # 5. alerts, history, analytics
    for path, name in (("/alerts", "05_alerts"), ("/history", "06_history"), ("/analytics", "07_analytics")):
        page.goto(UI + path, wait_until="networkidle"); time.sleep(3)
        page.screenshot(path=OUT / f"{name}.png", full_page=True)
        step(f"page {path} renders", "Something went wrong" not in page.inner_text("body"), page.title())

    # 6. reports: CSV buttons + job file links
    page.goto(UI + "/reports", wait_until="networkidle"); time.sleep(3)
    for lab in ("Alerts CSV", "Detections CSV"):
        d = try_download(page, page.get_by_role("button", name=lab), lab.replace(" ", "_"))
        rows = None
        if d.get("kind") == "download":
            f = next(OUT.glob(f"dl_{lab.replace(' ', '_')}_*")); rows = len(f.read_text(encoding="utf-8", errors="replace").splitlines()) - 1
        step(f"reports: '{lab}'", d.get("kind") == "download" and not d["navigated_away"], json.dumps({**d, "csv_rows": rows}))
    links = page.get_by_role("link", name="stats", exact=True)
    if links.count():
        d = try_download(page, links.first, "reports_stats_link")
        step("reports: job 'stats' link", d.get("kind") == "download" and not d["navigated_away"], json.dumps(d))
    links = page.get_by_role("link", name="video", exact=True)
    if links.count():
        d = try_download(page, links.first, "reports_video_link")
        step("reports: job 'video' link", d.get("kind") == "download" and not d["navigated_away"], json.dumps(d))
    page.screenshot(path=OUT / "08_reports.png", full_page=True)

    page.goto(UI + "/", wait_until="networkidle"); time.sleep(4)
    page.screenshot(path=OUT / "09_dashboard_after.png", full_page=True)
    step("dashboard after processing", True, page.inner_text("main")[:300].replace("\n", " | ") if page.locator("main").count() else "")
    br.close()

(OUT / "walkthrough.json").write_text(json.dumps(log, indent=1))
print(f"\n{sum(s['pass'] for s in log['steps'])}/{len(log['steps'])} steps passed; console errors {len(log['console_errors'])}; "
      f"page errors {len(log['page_errors'])}; failed requests {len(log['failed_requests'])}; HTTP>=400 {len(log['http_errors'])}")
