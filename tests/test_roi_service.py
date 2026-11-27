"""Monitored region (automatic pavement ROI) - backend/services/roi_service.py. Owner brief 2026-11-24.

Synthetic frames + a fake segmenter (ground = pavement-coloured pixels), so no model weights are needed, except in
the last test (skipped when the SegFormer weights are not installed).
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.settings import DetectionSettings
from backend.detection.types import Detection, InferenceInfo
from backend.services import roi_service as rs
from backend.services.pipeline import StreamPipeline, detect_image

AUTH = {"Authorization": "Bearer test-admin-token"}
W, H = 640, 360
PAVE = (150, 150, 150)
SHOP = (40, 80, 160)


def settings(**roi) -> DetectionSettings:
    base = {"exclude_people": False, "vote_frames": 5}
    return DetectionSettings.model_validate({"roi": {**base, **roi}})


class FakeSeg:
    """P(ground) = 1 on pavement-coloured pixels (any grey level), 0 elsewhere; works at a lower 'model' size."""
    name = "fake-seg"

    def __init__(self, width: int = 320) -> None:
        self.width, self.calls = width, 0

    def ground_prob(self, bgr: np.ndarray) -> np.ndarray:
        self.calls += 1
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_NEAREST).astype(int)
        grey = (np.abs(small[..., 0] - small[..., 1]) < 12) & (np.abs(small[..., 1] - small[..., 2]) < 12)
        return (grey & (small[..., 0] > 60)).astype(np.float32)


class FakeEngine:
    """Records what the detector is given; returns the configured boxes (in the coordinates it was given)."""
    version = "fake"
    device = "cpu"

    def __init__(self, boxes=()) -> None:
        self.boxes, self.seen = list(boxes), []

    def _dets(self):
        return [Detection(list(b[:4]), 0, "plastic_bag", b[4] if len(b) > 4 else 0.9) for b in self.boxes]

    def detect(self, image, s=None, force_tiling=None):
        self.seen.append(image.copy())
        return self._dets(), InferenceInfo()

    def detect_many(self, images, s=None, force_tiling=None):
        return [self.detect(i, s, force_tiling) for i in images]


@pytest.fixture(autouse=True)
def fake_seg():
    seg = FakeSeg()
    rs.set_segmenter(seg)
    yield seg
    rs.set_segmenter(None)


def noisy(img: np.ndarray, sigma: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def street(grey=PAVE) -> np.ndarray:
    """Top 35 %: shop interior (coloured shelves), bottom: flat pavement."""
    img = np.full((H, W, 3), grey, np.uint8)
    img[: int(H * 0.35)] = SHOP
    img[20:100, 50:200] = (20, 200, 40)          # brightly coloured stock inside the shop
    return noisy(img)


def ctx_for(img, s=None, cam="CAM1", tmp=None) -> rs.RoiContext:
    svc = rs.RoiService(tmp / "camera_roi.json") if tmp else rs.RoiService(rs.Path("unused.json"))
    return svc.resolve(cam, [img], s or settings(), save=False)


# ============================================================================== gate
@pytest.mark.parametrize("grey", [(245, 245, 245), (175, 175, 175), (70, 70, 70)], ids=["white", "light", "dark"])
def test_flat_region_is_clean_with_gate_message_and_skips_detector(grey, tmp_path):
    img = street(grey)
    ctx = ctx_for(img, tmp=tmp_path)
    assert ctx.active and ctx.source == "auto"
    eng = FakeEngine([(10, 10, 60, 60, 0.95)])
    res = detect_image(eng, img, settings(), roi=ctx)
    assert eng.seen == []                                     # detector skipped on a uniform region
    assert res["roi"]["status"] == "clean" and res["roi"]["message"] == rs.UNIFORM_MSG and res["count"] == 0
    assert rs.UNIFORM_MSG in rs.roi_remarks(res["roi"]) and "roi=auto" in rs.roi_remarks(res["roi"])


def test_small_red_patch_is_a_possible_candidate_not_litter_without_detector(tmp_path):
    img = street()
    img[260:275, 300:318] = (30, 30, 220)                     # small red object on the pavement
    ctx = ctx_for(img, tmp=tmp_path)
    eng = FakeEngine([])                                      # detector finds nothing
    res = detect_image(eng, img, settings(), roi=ctx)
    cands = res["roi"]["candidates"]
    assert len(cands) == 1 and cands[0]["kind"] == "chroma" and cands[0]["state"] == "possible"
    x1, y1, x2, y2 = cands[0]["bbox"]
    assert x1 <= 302 and x2 >= 316 and y1 <= 262 and y2 >= 273
    assert res["count"] == 0 and res["roi"]["status"] == "clean"             # a candidate alone is never litter
    assert res["roi"]["message"] == rs.NO_LITTER_MSG
    assert not any("anomal" in p for p in rs.roi_remarks(res["roi"]))     # phase 2: never in the remark
    assert len(eng.seen) == 1                                 # not uniform -> detector ran


def test_red_patch_confirmed_when_detector_confirms(tmp_path):
    img = street()
    img[260:275, 300:318] = (30, 30, 220)
    ctx = ctx_for(img, tmp=tmp_path)
    mask = ctx.mask(img.shape)
    ys, xs = np.nonzero(mask)
    x0, y0 = max(0, xs.min() - int(0.01 * W)), max(0, ys.min() - int(0.01 * W))   # crop offset (1 % pad)
    eng = FakeEngine([(300 - x0, 258 - y0, 318 - x0, 276 - y0, 0.9)])            # crop coordinates
    res = detect_image(eng, img, settings(), roi=ctx)
    assert res["count"] == 1 and res["roi"]["status"] == "litter"
    c = res["roi"]["candidates"][0]
    assert c["state"] == "confirmed" and c["confirmed_by"] == "detector"
    assert res["detections"][0]["bbox"][0] == pytest.approx(300, abs=1)          # mapped back to the frame


def test_shadow_only_change_is_ignored(tmp_path):
    img = np.full((H, W, 3), PAVE, np.uint8)
    img[: int(H * 0.35)] = SHOP
    pts = np.array([[0, H], [0, 200], [380, 140], [W, 330], [W, H]], np.int32)
    shadow = np.zeros((H, W), np.uint8)
    cv2.fillPoly(shadow, [pts], 1)
    img[(shadow > 0) & (np.arange(H)[:, None] > H * 0.35)] = (85, 85, 85)       # same hue, darker
    img = noisy(img)
    r = settings().roi
    ctx = ctx_for(img, tmp=tmp_path)
    gate = rs.analyse_region(img, ctx.mask(img.shape), r)
    assert gate.candidates == []                              # no colour or light/dark candidate from a shadow


def test_people_and_vehicles_drop_candidates(tmp_path):
    img = street()
    img[260:275, 300:318] = (30, 30, 220)
    ctx = ctx_for(img, tmp=tmp_path)
    r = settings().roi
    assert rs.analyse_region(img, ctx.mask(img.shape), r, people=[[290, 200, 330, 290]]).candidates == []
    assert len(rs.analyse_region(img, ctx.mask(img.shape), r, people=[[0, 0, 50, 50]]).candidates) == 1


# ============================================================================== inference inside the ROI
def test_shop_interior_never_reaches_the_detector(tmp_path):
    img = street()
    ctx = ctx_for(img, tmp=tmp_path)
    eng = FakeEngine([])
    rs.roi_detect(eng, img, settings(gate_enabled=False), ctx)
    crop = eng.seen[0]
    assert crop.shape[0] < H * 0.75                           # cropped to the region's bounding box
    shop_like = np.abs(crop.astype(int) - np.array(SHOP)).sum(2) < 40
    stock = np.abs(crop.astype(int) - np.array((20, 200, 40))).sum(2) < 40
    assert not shop_like.any() and not stock.any()            # interior pixels replaced by the median pavement


def test_keep_rule_bottom_centre_and_half_overlap():
    mask = np.zeros((100, 100), bool)
    mask[50:] = True
    d = lambda b: Detection(list(b), 0, "x", 0.9)             # noqa: E731
    inside = d((10, 60, 30, 80))
    straddle_ok = d((10, 40, 30, 70))                         # 2/3 inside, foot inside
    foot_out = d((10, 20, 30, 49))
    mostly_out = d((10, 0, 30, 55))                           # foot inside, only ~10 % inside
    kept, dropped = rs.keep_in_roi([inside, straddle_ok, foot_out, mostly_out], mask, 0.5)
    assert kept == [inside, straddle_ok] and dropped == 2


def test_roi_crop_is_the_detector_input_and_leaves_the_frame_alone():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (H, W, 3), np.uint8)
    mask = np.zeros((H, W), bool)
    mask[200:350, 100:600] = True
    mask[300:350, 100:150] = False                            # a hole: filled, not cropped away
    before = frame.copy()
    crop, x0, y0 = rs.roi_crop(frame, mask)
    pad = int(0.01 * max(H, W))                               # the inline geometry roi_detect used before the helper
    ref = frame[200 - pad:min(H, 350 + pad), 100 - pad:600 + pad].copy()
    ref[~mask[200 - pad:min(H, 350 + pad), 100 - pad:600 + pad]] = np.median(frame[mask], axis=0).astype(np.uint8)
    assert (x0, y0) == (100 - pad, 200 - pad) and np.array_equal(crop, ref)
    assert np.array_equal(frame, before)                      # the caller's frame is never modified
    eng = FakeEngine([])
    ctx = rs.RoiContext("C", {"polygons": rs.mask_to_polygons(mask, settings().roi)}, "manual")
    _, info, _ = rs.roi_detect(eng, frame, settings(gate_enabled=False), ctx)
    c2, a, b = rs.roi_crop(frame, ctx.mask(frame.shape))
    assert info.extra["roi_crop"] == [a, b, a + c2.shape[1], b + c2.shape[0]]
    assert len(eng.seen) == 1 and np.array_equal(eng.seen[0], c2)   # the detector got exactly roi_crop's pixels


def test_failure_uses_on_failure_with_warning_and_remark(tmp_path, fake_seg):
    img = np.full((H, W, 3), SHOP, np.uint8)                  # no ground at all
    svc = rs.RoiService(tmp_path / "camera_roi.json")
    ctx = svc.resolve("CAM1", [img], settings())
    assert not ctx.active and "not determined" in ctx.warning
    eng = FakeEngine([(10, 10, 50, 50, 0.9)])
    dets, _, block = rs.roi_detect(eng, img, settings(), ctx)
    assert len(dets) == 1 and block["warning"] and "roi_not_determined" in rs.roi_remarks(block)
    assert any(t.startswith("WARNING") for t in rs.banner_lines(block))
    ctx_skip = svc.resolve("CAM1", [img], settings(on_failure="skip"))
    dets, _, block = rs.roi_detect(FakeEngine([(10, 10, 50, 50)]), img, settings(on_failure="skip"), ctx_skip)
    assert dets == [] and block["status"] == "roi_not_determined"


def test_missing_model_is_a_clear_failure_not_a_crash(tmp_path):
    rs.set_segmenter(None)
    s = settings(model="does-not-exist")
    ctx = rs.RoiService(tmp_path / "r.json").resolve("CAM1", [street()], s)
    assert not ctx.active and "segmentation model missing" in ctx.warning and "fetch" in ctx.warning


def test_roi_off_is_full_frame_and_untagged(tmp_path):
    img = street()
    assert rs.RoiService(tmp_path / "r.json", enabled=False).resolve("C", [img], settings()) is None
    assert rs.RoiService(tmp_path / "r.json").resolve("C", [img], settings(enabled=False)) is None
    eng = FakeEngine([(1, 1, 30, 30)])
    res = detect_image(eng, img, settings(), roi=None)
    assert "roi" not in res and eng.seen[0].shape == img.shape          # no roi key at all when ROI is off


# ============================================================================== polygons
def _warped_scene(kind: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((H, W, 3), SHOP, np.uint8)
    if kind == "rotated":
        rect = cv2.boxPoints(((W / 2, H / 2), (420, 180), 27)).astype(np.int32)
    else:                                                       # perspective: a pavement trapezoid towards a horizon
        src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        dst = np.float32([[260, 90], [380, 90], [620, 350], [20, 350]])
        M = cv2.getPerspectiveTransform(src, dst)
        rect = cv2.perspectiveTransform(src[None], M)[0].astype(np.int32)
    cv2.fillPoly(img, [rect], PAVE)
    truth = np.all(img == PAVE, axis=2)
    return noisy(img), truth


@pytest.mark.parametrize("kind", ["rotated", "perspective"])
def test_rotated_and_perspective_frames_give_a_valid_polygon(kind):
    img, truth = _warped_scene(kind)
    res = rs.build_roi([img], settings().roi)
    rec = res.record
    assert rec and len(rec["polygons"]) == 1
    pts = rec["polygons"][0]["points"]
    assert 3 <= len(pts) <= 12 and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pts)
    m = rs.polygons_mask(rec["polygons"], (H, W))
    iou = (m & truth).sum() / (m | truth).sum()
    assert iou > 0.9 and rec["confidence"] > 0.8
    assert rec["needs_review"] and rec["approved"] is False            # phase 2: needs review until approved


def test_several_regions_and_holes_are_kept():
    img = np.full((H, W, 3), SHOP, np.uint8)
    img[200:, :250] = PAVE
    img[200:, 350:] = PAVE
    img[260:330, 430:560] = SHOP                               # a kiosk standing on the right-hand pavement
    rec = rs.build_roi([noisy(img)], settings(close_frac=0.0).roi).record
    assert len(rec["polygons"]) == 2
    assert sum(len(p["holes"]) for p in rec["polygons"]) == 1


def test_majority_vote_removes_passers_by():
    frames = []
    for i in range(5):
        f = street()
        x = 60 + i * 110
        f[150:330, x:x + 40] = (200, 60, 20)                   # a person walking through
        frames.append(f)
    rec = rs.build_roi(frames, settings().roi).record
    assert rec["frames_used"] == 5
    assert all(not p["holes"] for p in rec["polygons"])        # no person-shaped hole
    assert rec["area_frac"] > 0.6


def test_tiny_or_huge_region_needs_review():
    img = np.full((H, W, 3), SHOP, np.uint8)
    img[340:, 600:] = PAVE                                     # ~0.3 % of the frame
    rec = rs.build_roi([img], settings(min_component_frac=0.001).roi).record
    assert rec["needs_review"] and rec["confidence"] <= 0.2
    rec = rs.build_roi([np.full((H, W, 3), PAVE, np.uint8)], settings().roi).record
    assert rec["needs_review"] and rec["area_frac"] > 0.95
    ctx = rs.RoiContext("C", rec, "auto")
    assert ctx.remark_tags() == ["roi=auto", "needs_review"]


def test_floor_is_never_a_ground_class():
    assert "floor" not in DetectionSettings().roi.ground_classes
    assert set(DetectionSettings().roi.ground_classes) == {"road", "sidewalk", "path", "earth"}


# ============================================================================== cache, precedence, camera moved
def test_cache_signature_and_camera_moved(tmp_path, fake_seg):
    svc = rs.RoiService(tmp_path / "camera_roi.json", tmp_path / "refs")
    img = street()
    s = settings()
    a = svc.resolve("CAM1", [img], s)
    assert a.source == "auto" and fake_seg.calls == 1
    doc = json.loads((tmp_path / "camera_roi.json").read_text())
    assert set(doc["CAM1"]) >= {"polygons", "source", "confidence", "needs_review", "frame_signature"}
    assert (tmp_path / "refs" / "CAM1.jpg").is_file()
    brighter = cv2.convertScaleAbs(img, alpha=1.15, beta=10)    # lighting change: same camera
    assert svc.resolve("CAM1", [brighter], s).source == "auto" and fake_seg.calls == 1
    moved = np.roll(street(), 150, axis=0)                      # the camera now points elsewhere
    b = svc.resolve("CAM1", [moved], s)
    assert fake_seg.calls == 2 and b.notes and "camera moved" in b.notes[0]
    assert b.record["polygons"] != a.record["polygons"]


def test_precedence_request_manual_auto_global(tmp_path, fake_seg):
    svc = rs.RoiService(tmp_path / "camera_roi.json")
    img = street()
    s = settings()
    manual = [{"points": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]], "holes": []}]
    svc.store.put("CAM1", {"polygons": manual, "source": "manual", "confidence": 1.0, "needs_review": False,
                           "frame_signature": rs.frame_signature(img)})
    req = [[[0, 0.8], [1, 0.8], [1, 1]]]
    assert svc.resolve("CAM1", [img], s, req).source == "request"
    m = svc.resolve("CAM1", [img], s)
    assert m.source == "manual" and m.polygons == manual and fake_seg.calls == 0      # manual beats auto
    assert svc.resolve("CAM2", [img], s).source == "auto"                             # no record -> auto
    glob = [{"points": [[0, 0.9], [1, 0.9], [1, 1], [0, 1]], "holes": []}]
    svc.store.put(rs.GLOBAL_KEY, {"polygons": glob, "source": "manual", "needs_review": False})
    nothing = np.full((H, W, 3), SHOP, np.uint8)
    g = svc.resolve("CAM3", [nothing], s)
    assert g.source == "global" and g.polygons == glob and g.remark_tags() == ["roi=global"]
    # a moved camera keeps its manual ROI but flags it for review
    moved = svc.resolve("CAM1", [np.roll(img, 150, axis=0)], s)
    assert moved.source == "manual" and moved.needs_review and svc.store.get("CAM1")["review_reason"] == "camera moved"


def test_uploads_are_never_cached(tmp_path, fake_seg):
    svc = rs.RoiService(tmp_path / "camera_roi.json")
    svc.resolve("upload", [street()], settings())
    svc.resolve("upload", [street()], settings())
    assert fake_seg.calls == 2 and not (tmp_path / "camera_roi.json").exists()


def test_legacy_store_is_imported(tmp_path):
    legacy = tmp_path / "storage_camera_roi.json"
    legacy.write_text(json.dumps({"version": 1, "global": None, "cameras": {"CAMX": {
        "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]], "holes": [], "source": "auto_edited",
        "needs_review": False}}}))
    store = rs.RoiStore(tmp_path / "camera_roi.json", legacy)
    rec = store.get("CAMX")
    assert rec["source"] == "manual" and rec["polygons"][0]["points"][0] == [0, 0.5]


def test_video_frames_spread_over_the_first_seconds(make_video):
    frames = [np.full((64, 64, 3), i * 10, np.uint8) for i in range(20)]
    path = make_video(frames, repeat=5, fps=10.0)              # 10 s
    got = rs.video_frames(path, 6, 5.0)
    assert len(got) == 6
    means = [int(f.mean()) for f in got]
    assert means == sorted(means) and means[-1] <= 110         # only the first 5 s


# ============================================================================== video: static candidates, drawing
def test_static_candidate_becomes_confirmed_after_n_seconds(tmp_path):
    img = street()
    img[260:275, 300:318] = (30, 30, 220)
    ctx = ctx_for(img, tmp=tmp_path)
    s = settings(static_confirm_sec=4.0)
    pipe = StreamPipeline(FakeEngine([]), s, None, None, "CAM1", "video", roi=ctx)
    states = [pipe.process(img, i, video_time=float(i)).roi["candidates"][0]["state"] for i in range(6)]
    assert states[:4] == ["possible"] * 4 and states[4] == "confirmed"
    last = pipe.process(img, 6, video_time=6.0)
    assert last.summary["confirmed_litter"] == 0                # reported, never a litter row
    assert not any("anomal" in p for p in rs.roi_remarks(last.roi))      # phase 2: not in the remark


def test_live_roi_waits_then_builds(tmp_path, fake_seg):
    svc = rs.RoiService(tmp_path / "camera_roi.json")
    s = settings(vote_frames=3, vote_seconds=3.0)
    live = rs.LiveRoi(svc, "CAMLIVE", lambda: s)
    img = street()
    assert live.update(img, now=0.0) is None and live.pending
    live.update(img, now=1.1)
    live.update(img, now=2.2)
    live._thread.join(10)
    assert live.current is not None and live.current.active and not live.pending
    live2 = rs.LiveRoi(svc, "CAMLIVE", lambda: s)              # restart: cached record, ready at once
    assert live2.update(img, now=0.0).active and fake_seg.calls == 3


def test_overlay_colour_label_and_toggle(tmp_path):
    img = street()
    ctx = ctx_for(img, tmp=tmp_path)
    res = detect_image(FakeEngine([]), img, settings(), roi=ctx)
    out = res["annotated"]
    ring = np.all(np.abs(out.astype(int) - np.array(rs.ROI_BGR)) <= 2, axis=2)
    assert ring.sum() > 200                                    # outline + label in light blue (RGB 135,206,250)
    inside = cv2.erode(ctx.mask(img.shape).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~ring
    diff = out[inside].astype(float).mean(0) - img[inside].astype(float).mean(0)
    assert np.allclose(diff, 0.20 * (np.array(rs.ROI_BGR) - img[inside].astype(float).mean(0)), atol=3)
    off = detect_image(FakeEngine([]), img, settings(draw_overlay=False), roi=ctx)["annotated"]
    assert np.all(np.abs(off.astype(int) - np.array(rs.ROI_BGR)) <= 2, axis=2).sum() < ring.sum() / 4
    assert rs.NO_LITTER_MSG in " ".join(rs.banner_lines(res["roi"]))


# ============================================================================== API (automatic execution)
@pytest.fixture
def client(cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("LDM_ROI", "on")
    cfg.roi_mode = "on"
    cfg.roi_store_path = tmp_path / "config" / "camera_roi.json"
    from backend.main import create_app
    with TestClient(create_app(cfg)) as c:
        yield c


def _cam(client, make_video, cam_id="CAMR"):
    path = make_video([street()] * 3, repeat=4, fps=5.0, name=f"{cam_id}.mp4")
    r = client.post("/api/cameras", headers=AUTH, json={"id": cam_id, "name": "roi", "source": str(path),
                                                        "enabled": False})
    assert r.status_code == 201, r.text


def test_api_auto_roi_routes_use_the_new_store(client, make_video, fake_seg):
    _cam(client, make_video)
    assert client.get("/api/cameras/CAMR/auto-roi").json()["resolved_from"] == "none"
    r = client.post("/api/cameras/CAMR/auto-roi", json={"frames": 3}, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["computed"] and body["roi"]["source"] == "auto" and body["roi"]["polygons"]
    assert body["roi"]["polygon"] == body["roi"]["polygons"][0]["points"]        # Cameras editor compatibility
    for kind in ("frame.jpg", "overlay.jpg"):
        assert client.get(f"/api/cameras/CAMR/auto-roi/{kind}").status_code == 200
    poly = [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]
    e = client.put("/api/cameras/CAMR/auto-roi", headers=AUTH, json={"polygon": poly}).json()
    assert e["roi"]["source"] == "manual" and e["roi"]["polygon"] == poly
    assert client.put("/api/cameras/CAMR/auto-roi", headers=AUTH,
                      json={"polygon": [[0, 0], [5, 0], [1, 1]]}).status_code == 422
    assert e["roi"]["needs_review"] and e["roi"]["approved"] is False          # edited: needs approval
    a = client.post("/api/cameras/CAMR/auto-roi/approve", headers=AUTH).json()
    assert a["roi"]["approved"] is True and a["roi"]["needs_review"] is False
    rr = client.post("/api/cameras/CAMR/auto-roi", json={"frames": 3}, headers=AUTH).json()   # Re-run replaces it
    assert rr["computed"] and rr["roi"]["source"] == "auto" and rr["roi"]["approved"] is False
    assert client.delete("/api/cameras/CAMR/auto-roi", headers=AUTH).json() == {"cleared": True}
    listing = client.get("/api/auto-roi").json()
    assert listing["cameras"] == {} and listing["precedence"][:4] == ["request", "manual", "auto", "global"]


def test_api_image_upload_runs_the_roi_automatically(client, fake_seg):
    img = street()
    img[260:275, 300:318] = (30, 30, 220)
    r = client.post("/api/detect/image", headers=AUTH, data={"save_outputs": "false", "create_alerts": "false"},
                    files={"file": ("shop.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpeg")})
    if r.status_code == 503:
        pytest.skip("no YOLO model registered in this checkout")
    assert r.status_code == 200, r.text
    roi = r.json()["roi"]
    assert roi["applied"] and roi["source"] == "auto" and fake_seg.calls == 1
    assert "roi=auto" in roi["remark_tags"]
    out = cv2.imdecode(np.frombuffer(base64.b64decode(r.json()["annotated_image"]), np.uint8), 1)
    assert out.shape == img.shape
    req = json.dumps([[[0, 0.6], [1, 0.6], [1, 1], [0, 1]]])
    r2 = client.post("/api/detect/image", headers=AUTH, data={"save_outputs": "false", "roi_polygons": req},
                     files={"file": ("shop.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpeg")})
    assert r2.json()["roi"]["source"] == "request" and fake_seg.calls == 1
    bad = client.post("/api/detect/image", headers=AUTH, data={"roi_polygons": "[[0,0],[9,9]]"},
                      files={"file": ("shop.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpeg")})
    assert bad.status_code == 422


def test_settings_expose_roi_and_validate(client):
    s = client.get("/api/settings").json()["settings"]
    assert s["roi"]["enabled"] is True and s["roi"]["draw_overlay"] is True
    assert client.put("/api/settings", headers=AUTH, json={"roi": {"min_area_frac": 0.99}}).status_code == 422
    assert client.put("/api/settings", headers=AUTH, json={"roi": {"chroma_min": 14}}).json()["settings"]["roi"][
        "chroma_min"] == 14


@pytest.mark.skipif(not (rs.model_dir(DetectionSettings().roi.model) / "model.safetensors").is_file(),
                    reason="SegFormer weights not installed (python -m backend.services.roi_service fetch)")
def test_real_segmenter_loads_outdoor_classes_only():
    seg = rs.HFSegmenter(DetectionSettings().roi.model, DetectionSettings().roi.ground_classes)
    names = {seg.model.config.id2label[i] for i in seg.ground_idx}
    assert names == {"road", "sidewalk", "path", "earth"}
    with pytest.raises(ValueError):
        rs.HFSegmenter(DetectionSettings().roi.model, ["road", "floor"])


# ============================================================================== phase 2 (pilot prep)
class _UnknownSeg:
    """Ground everywhere below row 200, but a person stands on the pavement (rows 250-330, cols 300-360) in EVERY
    frame but one: those pixels are unknown, not non-ground."""
    name = "fake-unknown"

    def __init__(self, n_person_frames):
        self.i, self.n = 0, n_person_frames

    def ground_and_unknown(self, bgr):
        g = np.zeros(bgr.shape[:2], np.float32)
        g[200:] = 0.9
        u = np.zeros_like(g)
        if self.i < self.n:
            g[250:330, 300:360] = 0.0
            u[250:330, 300:360] = 0.95
        self.i += 1
        return g, u

    def ground_prob(self, bgr):
        return self.ground_and_unknown(bgr)[0]


def test_person_pixels_are_unknown_not_non_ground():
    frames = [street() for _ in range(10)]
    r = settings(fill_hole_frac=0.0).roi                       # no hole filling: the vote alone must keep it
    rec = rs.build_roi(frames, r, segmenter=_UnknownSeg(9)).record
    m = rs.polygons_mask(rec["polygons"], (H, W))
    assert m[290, 330]                                          # the person's footprint is ground (1 known vote)
    assert sum(len(p["holes"]) for p in rec["polygons"]) == 0
    assert rec["unknown_px_share"] > 0


def test_vote_without_unknown_masks_is_unchanged():
    a = np.zeros((4, 4), np.float32)
    b = np.ones((4, 4), np.float32)
    m, share = rs.vote([a, b, b], 0.5, 0.5)
    assert m.all() and np.allclose(share, 2 / 3)


def test_holes_below_two_percent_are_filled():
    assert settings().roi.fill_hole_frac == 0.02
    img = np.zeros((H, W), bool)
    img[200:] = True
    img[300:320, 300:330] = False                              # 600 px hole = 0.2 % of the frame
    assert rs.clean_mask(img, settings().roi)[310, 310]


def test_video_frames_spread_over_the_whole_video(make_video):
    frames = [np.full((64, 96, 3), 4 * i, np.uint8) for i in range(60)]          # frame i has grey level 4i
    path = make_video(frames, fps=10.0)                                            # 6 s
    whole = rs.video_frames(path, 5, None)
    first = rs.video_frames(path, 5, 0.5)
    assert len(whole) == 5 and float(whole[-1].mean()) > 200                     # reaches the end of the video
    assert float(first[-1].mean()) < 40                                          # only the first 0.5 s
    from backend.core.settings import RoiSettings
    assert RoiSettings().vote_frames == 40 and RoiSettings().video_whole


def test_annotated_outline_is_thick_and_scaled():
    assert rs.roi_line_px(640) == 4 and rs.roi_line_px(2560) == 10
    img = np.full((1440, 2560, 3), 120, np.uint8)
    block = {"applied": True, "polygons": [{"points": [[0.1, 0.5], [0.9, 0.5], [0.9, 0.95], [0.1, 0.95]],
                                            "holes": []}], "needs_review": False}
    rs.draw_roi(img, block)
    col = img[:, 1280]
    blue = np.all(np.abs(col.astype(int) - np.array(rs.ROI_BGR)) <= 2, axis=1)
    run = max(len(s) for s in "".join("1" if b else "0" for b in blue).split("0"))
    assert run >= 9                                            # 0.4 % of 2560 = 10 px outline


def test_unapproved_record_tags_needs_review_until_approved(tmp_path):
    rec = {"polygons": [{"points": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]], "holes": []}], "source": "auto",
           "needs_review": False, "approved": False}
    ctx = rs.RoiContext("CAM", rec, "auto")
    assert "needs_review" in ctx.remark_tags() and ctx.summary()["approved"] is False
    ctx2 = rs.RoiContext("CAM", {**rec, "approved": True}, "auto")
    assert "needs_review" not in ctx2.remark_tags()
    legacy = rs.RoiContext("CAM", {k: v for k, v in rec.items() if k != "approved"}, "auto")
    assert "needs_review" not in legacy.remark_tags()           # records from before phase 2 keep their flag
