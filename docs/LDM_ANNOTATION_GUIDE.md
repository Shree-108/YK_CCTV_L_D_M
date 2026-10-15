# Litter annotation guide (beginner)

You are teaching the litter model `v3_store_cctv` what litter looks like on your cameras. You do this by drawing a
box around every piece of genuine litter in each image. Images with no litter matter just as much: they teach the
model what to ignore.

Work in `dataset/ldm_v4_workspace` and follow the order in `reports/annotation_queue.csv`, real CCTV first.
Collection and dataset building are covered in `docs/LDM_DATASET_GUIDE.md`.

---

## 1. The classes (never change the numbers or the order)

They come from the deployed model (`models/litter_yolo/v3_store_cctv/class_names.json`).

| id | class | box this |
|---|---|---|
| 0 | plastic_bag | carry bags, polythene bags (flat or crumpled) |
| 1 | plastic_packaging | clear/printed plastic film, blister packs, shrink wrap, non-food pouches |
| 2 | food_wrapper | chips/snack packets, biscuit and chocolate wrappers, cigarette packets |
| 3 | paper | paper scraps, tissues, receipts, napkins |
| 4 | cardboard | cardboard boxes, flattened cartons, card pieces |
| 5 | plastic_bottle | a discarded bottle lying on the ground |
| 6 | cup | disposable paper or plastic cups |
| 7 | newspaper | newspaper sheets or pages |
| 8 | plastic_piece | small broken plastic fragments, caps, straws |
| 9 | other_litter | clear litter that fits none of the above: cans, food containers, foil |

> ⚠️ **Confirm these definitions before the first annotation sitting.** The ids and names are fixed by the model,
> but the project never wrote down what each class covers. The "box this" column is a proposal, and it must match
> how the original `cctv_store_v2` labels used the classes. The overlap between plastic_piece, plastic_packaging
> and food_wrapper matters most. If two annotators use a class differently, the model learns noise.

## 2. Is it litter?

**Litter = something that has been thrown away.** It is lying on the ground, kerb, steps or a ledge, and nobody is
holding, selling or storing it.

**Never box an object just because of what it is.** A bottle or wrapper is litter only when it has been discarded.

| You see | Box it? |
|---|---|
| Wrapper, cup, bottle or paper lying on the pavement | ✅ Yes |
| Bottles and juices in the fridge or on a display | ❌ No: shop products |
| Stock on shelves, in crates, or stacked at the door | ❌ No: shop inventory |
| Something in a hand, a bag, or being carried | ❌ No |
| Cars, motorcycles, tyres, bicycles, people | ❌ No |
| Shadows, reflections, paint marks, stains, leaves, drain covers | ❌ No |
| Rubbish inside a dustbin | ❌ No |
| Loose rubbish next to a bin, or a scattered pile | ✅ Box each item you can tell apart |
| A blur that might be paper or might be a stone | ⚠️ Don't guess: flag it as **ambiguous** (section 5) |

## 3. Drawing good boxes

```
   too loose              good (tight)            too small
 ┌──────────────┐          ┌─────┐               ┌───┐
 │              │          │▓▓▓▓▓│               │▓▓▓│▓▓
 │   ▓▓▓▓▓      │          │▓▓▓▓▓│               └───┘▓▓
 │   ▓▓▓▓▓      │          └─────┘                 ▓▓▓▓▓
 └──────────────┘
```

1. **Tight:** the box edges touch the outside of the object.
2. **Every item:** box all litter in the frame, including small, far-away and half-hidden pieces. Missing a real
   piece teaches the model it is *not* litter.
3. **One box per item:** two wrappers touching get two boxes.
4. **Partly hidden or cut off by the image edge:** box only the visible part, up to the edge.
5. **Too small to recognise** without zooming in? Skip it and write it in `notes`.

## 4. Images with no litter

These are just as important. Look at the whole image. If there is no litter:

- **Ground Truth page:** click **Save labels** with no boxes. This writes an empty label file.
- **CVAT / Label Studio:** leave the image without boxes, and finish every image in the job before exporting
  (see 6b).

Then set `category` in `manifest.csv`:
- `hard_negative`: no litter, but things that could fool the model, such as fridges of bottles, stock, cars, tyres,
  motorbikes, shadows or reflections. Add them to `subcategory`, for example `product_display;car;tyre`.
- `clean`: an ordinary scene with nothing confusing.

**An image nobody has looked at must never get an empty label.** "No label yet" and "checked: no litter" mean
different things, and the tools keep them apart.

## 5. When you are not sure

Don't guess. In `manifest.csv` set `review_status` = `ambiguous` and write the reason in `notes`, for example
"white object near drain, paper or plastic bag?". Ambiguous images stay out of the dataset until someone decides.
Set `rejected` for frames that are unusable (corrupted, totally dark, camera moving).

Frames with **drawn boxes burned into the video** (`overlay_suspect = yes`) are skipped automatically. If you open
one and it has no drawn boxes, change it to `no`.

## 6. Tools

### 6a. Recommended: the app's Ground Truth page (already installed)

1. Start the app, open **Settings → Security** and sign in as admin with the `ADMIN_API_TOKEN` from `.env`.
2. Open **Ground Truth** and choose dataset `ldm_v4_workspace`.
3. Pick an image (follow `annotation_queue.csv`).
4. Choose **Class for new boxes**, click **Add box**, then click two opposite corners of the object. Repeat for
   every item.
5. Click **Save labels**, with no boxes if there is no litter. The previous version is kept as `.txt.bak`.
6. After each sitting, run:

```bash
python -m training.ldm_review sync
```

```bash
python -m training.validate_ldm_dataset dataset/ldm_v4_workspace
```

### 6b. CVAT (good for teams or many images)

Use a **self-hosted CVAT** (Docker) or the app's own page. CCTV shows people and number plates, so don't upload it
to a public cloud service without permission.

1. Export a batch of images to annotate:

```bash
python -m training.annotation_io export --batch real_priority1
```

2. In CVAT create a task. Under **Labels**, click **Raw** and paste the contents of
   `annotation_batches/real_priority1/cvat_labels.json`. This creates the 10 classes in the **correct order**.
3. Upload `annotation_batches/real_priority1/images/*`. Draw **rectangles** only.
4. When every image is checked, set the job to **completed**, then **Export task dataset → YOLO 1.1**.
5. Import the labels:

```bash
python -m training.annotation_io import <export.zip>
```

CVAT writes an empty file for every image, *even unopened ones*, so empty files are ignored by default. If every
image in the task really was checked, add `--empty-means-reviewed`. Then run `ldm_review sync`.

Label Studio ("YOLO" export, `classes.txt`) and LabelImg (YOLO mode, with `obj.names` as the class file) work the
same way. The import refuses any export whose class list is in a different order.

### 6c. No tool at all: write the label file by hand

A YOLO label is a text file named like the image (`frame.jpg` → `labels/train/frame.txt`), with one line per box:

```
class_id  x_center  y_center  width  height        (all divided by the image size, so they are 0..1)
```

Get the corners in pixels: Paint shows the mouse position at the bottom left. Then calculate:

```
x_center = (x1 + x2) / 2 / image_width        width  = (x2 - x1) / image_width
y_center = (y1 + y2) / 2 / image_height       height = (y2 - y1) / image_height
```

Example: a food wrapper from (100, 287) to (136, 305) in a 640×368 frame gives

```
2 0.184375 0.804348 0.056250 0.048913
```

An image reviewed as having no litter gets an **empty** file of the same name. Then run `ldm_review sync` and the
validator, which catches any typing mistakes.

## 7. Before you finish a sitting: checklist

- [ ] Every litter item has a tight box with the right class
- [ ] Nothing in a fridge, on a shelf, in a hand, or on a vehicle is boxed
- [ ] No-litter images were saved with zero boxes **and** given a category (`hard_negative` / `clean`)
- [ ] Unsure images are `ambiguous` with a note
- [ ] `python -m training.ldm_review sync` run, and `validate_ldm_dataset` shows **0 errors**
- [ ] Each sitting, a second person re-checks a random 10 % of images (write "QA ok" or the problem in `notes`)
