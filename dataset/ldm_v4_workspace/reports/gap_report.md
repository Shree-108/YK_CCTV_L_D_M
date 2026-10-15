# Dataset gap report

Workspace: `C:\SHREE\YATRIKART\PROJECTS\YK_CCTV_L_D_M_main\YK_CCTV_L_D_M\dataset\ldm_v4_workspace`

- Images listed: **215**
- Usable now (reviewed, no overlay): **0** (real CCTV: 0)
- Still need annotation / review: **215** (see `needs_annotation.txt`)
- Ambiguous, waiting for a decision: **0** (see `ambiguous.txt`)
- Reviewed empty but no category yet: **0** (see `needs_category.txt`)
- Flagged for burned-in boxes: **70** (see `overlay_suspect.txt`)

## Review status (all images)

| status | images |
|---|---|
| unreviewed | 215 |

## Source type (all images)

| source | images |
|---|---|
| real_cctv | 116 |
| synthetic | 99 |

## Usable images by category

_none_

## Boxes per class (usable images)

| class | boxes |
|---|---|
| plastic_bag | 0 |
| plastic_packaging | 0 |
| food_wrapper | 0 |
| paper | 0 |
| cardboard | 0 |
| plastic_bottle | 0 |
| cup | 0 |
| newspaper | 0 |
| plastic_piece | 0 |
| other_litter | 0 |

Small boxes (< 32x32 px): **0**

## Sessions (usable images)

_none_

## Conditions tagged (usable images)

_none_

## Gaps against the first fine-tune targets

| needed | have | target | missing |
|---|---|---|---|
| genuine litter images (real) | 0 | 300 | 300 |
| hard-negative images (real) | 0 | 200 | 200 |
| clean-scene images (real) | 0 | 100 | 100 |
| boxes of class 'plastic_bag' | 0 | 50 | 50 |
| boxes of class 'plastic_packaging' | 0 | 50 | 50 |
| boxes of class 'food_wrapper' | 0 | 50 | 50 |
| boxes of class 'paper' | 0 | 50 | 50 |
| boxes of class 'cardboard' | 0 | 50 | 50 |
| boxes of class 'plastic_bottle' | 0 | 50 | 50 |
| boxes of class 'cup' | 0 | 50 | 50 |
| boxes of class 'newspaper' | 0 | 50 | 50 |
| boxes of class 'plastic_piece' | 0 | 50 | 50 |
| boxes of class 'other_litter' | 0 | 50 | 50 |
| small / distant litter boxes (< 32x32 px) | 0 | 150 | 150 |
| hard negative: product_display | 0 | 20 | 20 |
| hard negative: shop_inventory | 0 | 20 | 20 |
| hard negative: car | 0 | 20 | 20 |
| hard negative: motorcycle | 0 | 20 | 20 |
| hard negative: tyre | 0 | 20 | 20 |
| hard negative: entrance | 0 | 20 | 20 |
| hard negative: shadow | 0 | 20 | 20 |
| hard negative: reflection | 0 | 20 | 20 |
| hard negative: pavement_marking | 0 | 20 | 20 |
| hard negative: person_carrying | 0 | 20 | 20 |
| condition: day | 0 | 30 | 30 |
| condition: dusk | 0 | 30 | 30 |
| condition: night / IR | 0 | 30 | 30 |
| condition: rain / wet ground | 0 | 30 | 30 |
| condition: glare / harsh shadow | 0 | 30 | 30 |
| independent recording sessions | 0 | 8 | 8 |

