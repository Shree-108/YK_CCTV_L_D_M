"""Read-only audit of the 100-image litter dataset: inventory, quality, hashing, ORB grouping."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

def dct_matrix(n):
    m = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            m[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    return m
D32 = dct_matrix(32)

def phash(g):
    a = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(float)
    d = D32 @ a @ D32.T
    low = d[:8, :8].flatten()
    return low > np.median(low[1:])

def dhash(g):
    a = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(float)
    return (a[:, 1:] > a[:, :-1]).flatten()

files = sorted(SRC.glob("*.jpg"))
rows, ph, dh, grays, kps = [], [], [], [], []
orb = cv2.ORB_create(2000)
for f in files:
    raw = f.read_bytes()
    im = Image.open(f)
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    noise = float(np.std(g.astype(float) - cv2.GaussianBlur(g, (3, 3), 0).astype(float)))
    q = None
    try:
        qt = im.quantization  # JPEG quant tables -> rough quality estimate
        q = float(np.mean(qt[0]))
    except Exception:
        pass
    rows.append(dict(file=f.name, w=w, h=h, mode=im.mode, fmt=im.format, kb=round(len(raw)/1024, 1),
                     md5=hashlib.md5(raw).hexdigest(), sharp=round(lap, 1), bright=round(float(g.mean()), 1),
                     contrast=round(float(g.std()), 1), noise=round(noise, 2), jpeg_q_mean=q))
    ph.append(phash(g)); dh.append(dhash(g))
    small = cv2.resize(g, (int(w * 800 / max(w, h)), int(h * 800 / max(w, h)))) if max(w, h) > 800 else g
    k, des = orb.detectAndCompute(small, None)
    kps.append((k, des))

n = len(files)
P = np.array([[int((ph[i] != ph[j]).sum()) for j in range(n)] for i in range(n)])
Dm = np.array([[int((dh[i] != dh[j]).sum()) for j in range(n)] for i in range(n)])

# ORB + RANSAC homography inliers -> geometric relationship (crop/zoom of same scene)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
def inliers(i, j):
    (ki, di), (kj, dj) = kps[i], kps[j]
    if di is None or dj is None or len(ki) < 10 or len(kj) < 10:
        return 0
    m = bf.knnMatch(di, dj, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return len(good) // 4
    src = np.float32([ki[a.queryIdx].pt for a in good]); dst = np.float32([kj[a.trainIdx].pt for a in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0
I = np.zeros((n, n), int)
for i in range(n):
    for j in range(i + 1, n):
        I[i, j] = I[j, i] = inliers(i, j)

# Union-find grouping on strong geometric match
par = list(range(n))
def find(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
TH = 25
for i in range(n):
    for j in range(i + 1, n):
        if I[i, j] >= TH:
            par[find(i)] = find(j)
groups = {}
for i in range(n):
    groups.setdefault(find(i), []).append(files[i].name)

md5s = {}
for r in rows: md5s.setdefault(r["md5"], []).append(r["file"])
exact = [v for v in md5s.values() if len(v) > 1]
near_ph = [(files[i].name, files[j].name, int(P[i, j])) for i in range(n) for j in range(i+1, n) if P[i, j] <= 6]
near_dh = [(files[i].name, files[j].name, int(Dm[i, j])) for i in range(n) for j in range(i+1, n) if Dm[i, j] <= 6]

iu = np.triu_indices(n, 1)
summary = dict(
    n=n, dims=sorted({(r["w"], r["h"]) for r in rows}), modes=sorted({r["mode"] for r in rows}),
    exact_dup_sets=exact, near_dup_phash_le6=len(near_ph), near_dup_dhash_le6=len(near_dh),
    phash_dist=dict(min=int(P[iu].min()), median=float(np.median(P[iu])), max=int(P[iu].max())),
    orb_inliers=dict(min=int(I[iu].min()), median=float(np.median(I[iu])), max=int(I[iu].max()),
                     pct_pairs_ge_TH=round(float((I[iu] >= TH).mean()) * 100, 1)),
    orb_groups={f"G{k+1}": v for k, v in enumerate(sorted(groups.values(), key=len, reverse=True))},
    quality=dict(
        sharp=dict(min=min(r["sharp"] for r in rows), median=float(np.median([r["sharp"] for r in rows])), max=max(r["sharp"] for r in rows)),
        bright=dict(min=min(r["bright"] for r in rows), max=max(r["bright"] for r in rows)),
        noise=dict(min=min(r["noise"] for r in rows), max=max(r["noise"] for r in rows)),
        kb=dict(min=min(r["kb"] for r in rows), max=max(r["kb"] for r in rows))),
)
(OUT / "per_image.json").write_text(json.dumps(rows, indent=1))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
np.save(OUT / "orb_inliers.npy", I); np.save(OUT / "phash_dist.npy", P)
(OUT / "near_phash.json").write_text(json.dumps(near_ph))
print(json.dumps({k: v for k, v in summary.items() if k != "orb_groups"}, indent=1))
for k, v in summary["orb_groups"].items():
    print(k, len(v), v[:6], "..." if len(v) > 6 else "")
