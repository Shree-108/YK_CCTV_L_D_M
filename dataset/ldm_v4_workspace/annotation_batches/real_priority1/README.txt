Annotation batch 'real_priority1' - 46 images, exported 2026-10-12 22:06.
CVAT: create a task, Labels -> Raw -> paste cvat_labels.json (keeps the class ORDER), upload images/*.
Annotate following docs/LDM_ANNOTATION_GUIDE.md. Mark the job 'completed' only when every image was checked.
Export: 'YOLO 1.1', then: python -m training.annotation_io import <export.zip>
