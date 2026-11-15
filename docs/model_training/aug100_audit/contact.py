import sys
from pathlib import Path
import cv2, numpy as np
SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2])
def sheet(names, out, cols=4, w=480):
    tiles = []
    for n in names:
        im = cv2.imread(str(SRC / n)); h = int(im.shape[0] * w / im.shape[1])
        im = cv2.resize(im, (w, h)); cv2.putText(im, n[-7:-4], (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows))
odd = [f"cctv_litter_{i:03d}.jpg" for i in range(1, 101, 2)]
even = [f"cctv_litter_{i:03d}.jpg" for i in range(2, 101, 2)]
sheet(odd[:12], OUT / "sheet_G1.jpg"); sheet(even[:12], OUT / "sheet_G2.jpg")
for n in ["cctv_litter_001.jpg", "cctv_litter_002.jpg"]:
    cv2.imwrite(str(OUT / ("full_" + n)), cv2.imread(str(SRC / n)))
