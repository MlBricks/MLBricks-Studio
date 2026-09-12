# Step 10V — COCO128 Cloud Data Pipeline

- Adds `COCO128 Cloud` as a reusable Data Builder source.
- Fetches the official Ultralytics COCO128 ZIP only when the pipeline runs.
- Download/extraction use a temporary session directory; no COCO images or labels are bundled in the repo, wheel, or PyPI package and no persistent dataset cache is created by this source.
- Materializes image bytes plus YOLO annotations into the normal Studio detection contract: `image`, `boxes` (pixel xywh), `class_ids`, `image_id`.
- Preserves the 80 COCO class names through `Sequence(ClassLabel)` metadata.
- Adds a Vision Gallery `COCO128 Cloud` data preset wired as an editable pipeline: cloud source → detection processing → train/validation/test split → prepared dataset.
- Exposes dataset class count/names in prepared-dataset metadata.
- Adds frontend and backend detector class-count compatibility guards so an 80-class COCO128 pipeline cannot silently train a 3-class detection head.
- Persists class names/num classes into supervised checkpoints and final model metadata for later visual runtime labels.
