"""Is box transfer from one reference frame to every variant geometrically valid?
For each scene: choose the reference with the widest view, fit ref->variant homography (ORB+RANSAC, full res),
decompose it, and measure LOCAL alignment after warping (patch NCC on gradients) to catch non-planar / lens /
content changes that a single homography cannot express. Read-only."""
import sys, json
from pathlib import Path
import numpy as np, cv2

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
orb = cv2.ORB_create(6000, scaleFactor=1.2, nlevels=10)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def feats(n):
    g = cv2.imread(str(SRC / n), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    k, d = orb.detectAndCompute(clahe, None)
    return g, k, d

def homog(a, b):  # maps a -> b
    (_, ka, da), (_, kb, db) = a, b
    m = bf.knnMatch(da, db, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 12: return None, 0, None
    s = np.float32([ka[g.queryIdx].pt for g in good]); d = np.float32([kb[g.trainIdx].pt for g in good])
    H, mask = cv2.findHomography(s, d, cv2.USAC_MAGSAC, 3.0, maxIters=10000, confidence=0.999)
    if H is None: return None, 0, None
    inl = mask.ravel().astype(bool)
    pr = cv2.perspectiveTransform(s[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(((pr - d[inl]) ** 2).sum(1).mean()))
    return H, int(inl.sum()), rms

def decompose(H):
    A = H[:2, :2] / H[2, 2]
    det = float(np.linalg.det(A))
    sx = float(np.hypot(A[0, 0], A[1, 0])); rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    sy = det / sx
    shear = float((A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / det)
    persp = float(np.hypot(H[2, 0], H[2, 1]) / H[2, 2] * 1000)
    return dict(scale=round(float(np.sqrt(abs(det))), 3), aniso=round(sx / sy, 3), rot_deg=round(rot, 2),
                shear=round(shear, 4), persp_e3=round(persp, 4), flip=det < 0)

def local_ncc(ref_g, var_g, H):  # warp variant into reference frame
    h, w = ref_g.shape
    warped = cv2.warpPerspective(var_g, np.linalg.inv(H), (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full_like(var_g, 255), np.linalg.inv(H), (w, h)) > 250
    gr = lambda im: cv2.magnitude(cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 0, 1))
    A, B = gr(ref_g), gr(warped)
    vals, P = [], 48
    for y in range(0, h - P, P):
        for x in range(0, w - P, P):
            if not valid[y:y+P, x:x+P].all(): continue
            a = A[y:y+P, x:x+P].ravel(); b = B[y:y+P, x:x+P].ravel()
            if a.std() < 4: continue  # textureless patch: uninformative
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    v = np.array(vals)
    return dict(patches=len(v), ncc_median=round(float(np.median(v)), 3), ncc_p05=round(float(np.percentile(v, 5)), 3),
                frac_below_0_5=round(float((v < 0.5).mean()), 3), coverage=round(float(valid.mean()), 3)), warped

report = {}
for scene, ids in (("A_shopfront", range(1, 101, 2)), ("B_roadside_bin", range(2, 101, 2))):
    names = [f"cctv_litter_{i:03d}.jpg" for i in ids]
    F = {n: feats(n) for n in names}
    # reference = frame whose view contains the most others (others map inside it with scale<=1)
    best, best_score = None, -1
    for r in names:
        score = 0
        for v in names[::5]:
            if v == r: continue
            H, inl, _ = homog(F[v], F[r])  # v -> r
            if H is None: continue
            h, w = F[v][0].shape
            c = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
            inside = ((c[:, 0] >= -8) & (c[:, 0] <= w + 8) & (c[:, 1] >= -8) & (c[:, 1] <= h + 8)).mean()
            score += inside
        if score > best_score: best, best_score = r, score
    rows = []
    for v in names:
        if v == best: continue
        H, inl, rms = homog(F[best], F[v])  # ref -> variant (direction boxes travel)
        if H is None:
            rows.append(dict(file=v, ok=False)); continue
        loc, warped = local_ncc(F[best][0], F[v][0], H)
        rows.append(dict(file=v, inliers=inl, rms_px=round(rms, 2), **decompose(H), **loc))
        if v in (names[1], names[10], names[25]):
            vis = np.dstack([F[best][0], warped, F[best][0]])  # magenta/green = misalignment
            cv2.imwrite(str(OUT / f"align_{scene}_{v}"), vis)
    arr = lambda k: [r[k] for r in rows if k in r]
    report[scene] = dict(reference=best, n_variants=len(rows), failed=[r["file"] for r in rows if r.get("ok") is False],
        scale=[min(arr("scale")), max(arr("scale"))], rot_deg=[min(arr("rot_deg")), max(arr("rot_deg"))],
        aniso=[min(arr("aniso")), max(arr("aniso"))], shear=[min(arr("shear")), max(arr("shear"))],
        persp_e3_max=max(arr("persp_e3")), flips=sum(arr("flip")), rms_px=[min(arr("rms_px")), max(arr("rms_px"))],
        ncc_median_min=min(arr("ncc_median")), ncc_p05_min=min(arr("ncc_p05")),
        worst=sorted(rows, key=lambda r: r.get("ncc_p05", -1))[:5], coverage_min=min(arr("coverage")))
    (OUT / f"geometry_{scene}.json").write_text(json.dumps(rows, indent=1))
    print(scene, json.dumps({k: v for k, v in report[scene].items() if k != "worst"}))
    for w in report[scene]["worst"]: print("   worst", w)
(OUT / "geometry_summary.json").write_text(json.dumps(report, indent=1))
