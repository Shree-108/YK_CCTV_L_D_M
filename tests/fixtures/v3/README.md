# Test fixture for the v3 tests

| file | what |
|---|---|
| `aug100_A_083.jpg` | `cctv_litter_083.jpg` from the AUG100 source folder `CCTV_Litter_100_Images` (byte-identical, SHA-256 `e95497aa…7a1d`). **AI-generated / synthetic** CCTV-style street frame (scene A), so it shows no real people. |
| `aug100_A_083.txt` | 19 litter boxes (YOLO format, single class `0 = litter`), from `dataset/exp_aug100/labels/train/cctv_litter_083.txt`. Each proposal was reviewed visually on 2026-11-13 (`experiments/exp_aug100/reference_review.py`); provisional ROI = pavement + kerb + gutter. |

**What the tests may and may not conclude from it**

- This scene was very probably in v3's training data (`dataset/cctv_store_v2` lists "street scenes A+C"). The tests that use it are **regression and smoke tests**: they check that the pipeline still finds what it found before. They are **not** an accuracy measurement and say nothing about unseen CCTV footage.
- The labels are single-class. They are used only for box-level matching (IoU), never for per-class metrics.
- Measured on 2026-11-13 with v3 at default settings: 19 confirmed detections, 17/19 labelled boxes matched (IoU ≥ 0.3), largest box 0.4 % of the frame.
- The test floors (recall ≥ 0.70, no box > 10 % of the frame) are set well below these values on purpose.
