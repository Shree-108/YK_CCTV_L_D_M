# CCTV litter dataset: collection, annotation and build guide

This guide covers preparing a fine-tuning dataset for the deployed litter model `models/litter_yolo/v3_store_cctv`
(YOLO11s, 10 classes). It stops before training. Every command is run from the project folder.

```
sources (videos / images, never modified)
   │  extract_frames.py · import_images.py        copy + de-duplicate + record provenance
   ▼
dataset/ldm_v4_workspace/   images/train/*  labels/train/*  manifest.csv  reports/
   │  Ground Truth page (draw boxes) + manifest.csv (category, conditions, status)
   │  ldm_review.py sync · mark-empty · report     review bookkeeping + gap report
   ▼
build_ldm_dataset.py  ──►  dataset/cctv_ldm_v4/ images|labels/{train,val,test}  data.yaml  README.md
   ▼
validate_ldm_dataset.py    must report 0 errors before any training
```

The workspace is not a training dataset. Its `data.yaml` has no `val` split on purpose, so Ultralytics refuses to
train on it by accident.

## 1–4. Classes, litter rules, box drawing and annotation tools

These are all in **`docs/LDM_ANNOTATION_GUIDE.md`**: the class table and the definitions still to be confirmed,
what is and isn't litter, how to draw boxes, how to handle no-litter and ambiguous images, the Ground Truth page,
the CVAT / Label Studio round trip (`training/annotation_io.py`), and how to write labels by hand.

The old 13-class names map the same way the project's own `training/build_store_dataset.py` maps them:
chips_packet and cigarette_packet go to `food_wrapper`, tissue goes to `paper`, plastic_cup goes to `cup`, and
metal_can and food_container go to `other_litter`.

## 5. Filling in manifest.csv

Open `dataset/ldm_v4_workspace/manifest.csv` in Excel. **Close it before running any command**, because Excel locks
the file. The tools fill in most columns themselves. A person fills in:

| column | values |
|---|---|
| `category` | `litter` (any litter in frame) · `hard_negative` (no litter, but distracting objects) · `clean` (no litter, plain scene) |
| `subcategory` | `;`-separated. Hard negatives: product_display, shop_inventory, car, motorcycle, tyre, entrance, shadow, reflection, pavement_marking, person_carrying. Litter: snack_packet, wrapper, paper, newspaper, cup, bottle, plastic_piece, cardboard, small_distant, occluded. Clean: empty_pavement, shop_surroundings |
| `conditions` | `;`-separated: day, dusk, night, ir, rain_wet, glare, harsh_shadow, low_light, crowded, blur, compression |
| `review_status` | set by `sync`. Set `ambiguous` or `rejected` (unusable frame) by hand |
| `overlay_suspect` | `yes` = the frame has burned-in analytics boxes, so it is excluded. Change it to `no` only after looking at the frame |
| `split_lock` | leave empty. Use `test` for a whole session you want kept as unseen test footage |

A frame can contain litter **and** distractors, such as the store camera with wrappers on the pavement and bottles
in the fridge. Its category is `litter`. List the distractors in `subcategory` too, and box only the litter.

## 6. Commands

```bash
# add material (sources are only read)
python -m training.extract_frames <video or folder> --every 2 --session <cam>_<yyyymmdd> --camera <cam> --conditions day
python -m training.import_images <folder> --source-type real_cctv            # or synthetic / other_real
python -m training.import_images storage/captures --source-type real_cctv     # scheduled Live CCTV screenshots

# annotate with an external tool (optional)
python -m training.annotation_io export --batch real_priority1
python -m training.annotation_io import <cvat_export.zip>   # empty files ignored unless --empty-means-reviewed

# after an annotation sitting
python -m training.ldm_review sync          # records what you saved in the Ground Truth page
python -m training.validate_ldm_dataset dataset/ldm_v4_workspace   # reports/files_to_fix.csv
python -m training.ldm_review mark-empty    # empty labels for rows you set to reviewed_empty in manifest.csv
python -m training.ldm_review report        # gap_report.md + annotation_queue.csv + needs_annotation.txt

# when the gap report looks healthy
python -m training.build_ldm_dataset --name cctv_ldm_v4 [--include-dataset dataset/cctv_store_v2]
python -m training.validate_ldm_dataset dataset/cctv_ldm_v4
```

Rules the build enforces:
- One recording session stays in one split, and near-identical frames are tied together.
- Synthetic images and footage v3 was already trained on are kept in train only.
- An existing dataset version is never overwritten.

## 7. Getting more real CCTV images

The Live CCTV page can capture screenshots on a schedule. Each capture saves the clean `original.jpg` next to the
annotated copy, and `import_images storage/captures` picks up only the originals, with camera and time filled in.
Beyond that:

- **Record from the NVR at the original resolution.** Export the raw recording, not a screen recording and not
  footage with analytics boxes drawn on it.
- **Cover time and weather.** Capture morning, noon, evening, night/IR, rain or wet ground, harsh sun and shadow,
  on weekdays and weekends.
- **Prefer many short sessions to one long one.** For example, 15 minutes at a time on different days and cameras.
  Splits are made per session, so 10+ sessions are needed for honest val/test sets.
- **Collect hard negatives on purpose.** Point existing cameras at product displays, crates of stock, parked bikes
  and cars, and clean pavements in sun and shade.
- **Keep one complete session (a camera or a day) aside as `split_lock = test`** before looking at model results.

Synthetic images, if used, must be imported with `--source-type synthetic`. They stay in train only and never
replace real footage.

## 8. Privacy

CCTV frames show people and number plates. Keep the dataset on authorised machines, don't upload it to public
services, and share it only as needed for training.
