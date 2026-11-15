"""Read-only inference benchmark: model x imgsz x (full / tiled) on a few frames from each scene group."""
import sys, time, json
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); weights = sys.argv[3:]
frames = [f"cctv_litter_{i:03d}.jpg" for i in (1, 21, 41, 61, 81, 2, 22, 42, 62, 82)]
imgs = {n: cv2.imread(str(SRC / n)) for n in frames}

def tiles(im, t=640, ov=0.25):
    h, w = im.shape[:2]; step = int(t * (1 - ov))
    xs = list(range(0, max(w - t, 0) + 1, step)) or [0]; ys = list(range(0, max(h - t, 0) + 1, step)) or [0]
    if xs[-1] + t < w: xs.append(w - t)
    if ys[-1] + t < h: ys.append(h - t)
    return [(x, y, im[y:y+t, x:x+t]) for y in ys for x in xs]

res = []
for wpath in weights:
    m = YOLO(wpath); names = m.names
    m.predict(imgs[frames[0]], imgsz=640, verbose=False, device="cpu")  # warmup
    for mode, sz in (("full", 640), ("full", 1280), ("tiled640", 640)):
        for conf in (0.25, 0.10):
            tot, ms, cls_count, sizes = 0, [], {}, []
            for n, im in imgs.items():
                t0 = time.perf_counter()
                boxes = []
                if mode == "full":
                    r = m.predict(im, imgsz=sz, conf=conf, verbose=False, device="cpu")[0]
                    boxes = [(b.xyxy[0].tolist(), int(b.cls), float(b.conf)) for b in r.boxes]
                else:
                    for x, y, tile in tiles(im):
                        r = m.predict(tile, imgsz=sz, conf=conf, verbose=False, device="cpu")[0]
                        boxes += [([b.xyxy[0][0]+x, b.xyxy[0][1]+y, b.xyxy[0][2]+x, b.xyxy[0][3]+y], int(b.cls), float(b.conf)) for b in r.boxes]
                ms.append((time.perf_counter() - t0) * 1000)
                tot += len(boxes)
                for bb, c, _ in boxes:
                    cls_count[names[c]] = cls_count.get(names[c], 0) + 1
                    sizes.append(float(min(bb[2]-bb[0], bb[3]-bb[1])))
                if conf == 0.25 and mode != "tiled640" or (conf == 0.25 and mode == "tiled640"):
                    vis = im.copy()
                    for bb, c, cf in boxes:
                        p = [int(v) for v in bb]; cv2.rectangle(vis, p[:2], p[2:], (0, 0, 255), 2)
                        cv2.putText(vis, f"{names[c]} {cf:.2f}", (p[0], p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    if n in ("cctv_litter_001.jpg", "cctv_litter_002.jpg"):
                        cv2.imwrite(str(OUT / f"pred_{Path(wpath).parent.name}_{mode}{sz}_{n}"), vis)
            res.append(dict(model=Path(wpath).parent.name or wpath, mode=mode, imgsz=sz, conf=conf, dets_per_img=round(tot/len(imgs), 1),
                            ms_per_img=round(float(np.median(ms))), classes=cls_count,
                            min_side_px_median=round(float(np.median(sizes)), 1) if sizes else None))
            print(json.dumps(res[-1]))
(OUT / "bench.json").write_text(json.dumps(res, indent=1))
