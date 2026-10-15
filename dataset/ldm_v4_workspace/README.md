# ldm_v4_workspace: annotation workspace (NOT a training dataset)

Every candidate frame for fine-tuning `v3_store_cctv` sits in `images/train/`. That folder name exists only so the
Ground Truth page can open it. The real train/val/test split is made later by `training/build_ldm_dataset.py`.

- `manifest.csv`: one row per image with provenance (source file, session, frame, timestamp) and review state.
- `labels/train/`: YOLO labels saved from the Ground Truth page. A missing file means the image is not reviewed yet.
- `reports/`: gap report and work lists (`python -m training.ldm_review report`).

Images and labels are git-ignored. Back them up together with `manifest.csv`.
Full instructions: `docs/LDM_DATASET_GUIDE.md`.

## Sources added on 2026-10-12

| session | source | type | frames | notes |
|---|---|---|---|---|
| store_entrance_poc3min | `C:\SHREE\Screen Captures\cctv_litter_poc_3min_h264.mp4` (640x368, 180 s) | real_cctv | 28 of 90 sampled (2 s), 62 near-duplicates skipped | store entrance by day, litter on the pavement, product fridges in frame. Probably the '3min' clip v3 was trained on, so it is kept in train only |
| store_entrance_footfall | `C:\SHREE\Screen Captures\footfall_3min_selected (1).mp4` (640x368, 187 s) | real_cctv | 88 of 90 sampled (2 s) | same camera, crowded, car + tyre + motorbikes. **Burned-in green person boxes**: 69 frames flagged overlay_suspect and excluded |
| street_aug100 | `C:\SHREE\DATASET\CCTV_Litter_100_Images (2)\CCTV_Litter_100_Images` (1280x720) | synthetic | 99 of 100 (1 near-duplicate) | the source README says these are augmented variations of 2 street scenes, not independent captures |

Commands used:

```bash
python -m training.extract_frames "C:/SHREE/Screen Captures/cctv_litter_poc_3min_h264.mp4" --every 2 --session store_entrance_poc3min --camera store_entrance --conditions day --used-in-previous-training likely
python -m training.extract_frames "C:/SHREE/Screen Captures/footfall_3min_selected (1).mp4" --every 2 --session store_entrance_footfall --camera store_entrance --conditions "day;crowded" --used-in-previous-training unknown
python -m training.import_images "C:/SHREE/DATASET/CCTV_Litter_100_Images (2)/CCTV_Litter_100_Images" --source-type synthetic --session street_aug100 --used-in-previous-training likely
```

Not used, and why:
- `storage/videos/*/annotated.mp4` and `storage/alerts/*`: the app's own output with detection boxes drawn in. The tools refuse them.
- `Uniform Detection*.mp4`: these belong to the separate Uniform project.
- `C:\Users\chinm\Videos\Screen Recordings\*.mp4`: screen recordings, not camera exports. Not inspected.
