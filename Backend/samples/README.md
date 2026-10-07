# Sample images

Samples are public example analyses (spec D-07): everyone sees them in the history (`scope=samples`), they cannot be
deleted through the API, and their image links need no signature. Each one is produced by running a photo through the
real pipeline, so the stored result is exactly what the model said at seeding time.

```
samples/
  tree_classification/      <- one folder per model key
    manifest.json
    banana_tree_01.jpg
    banana_tree_02.jpg
    non_banana_tree_01.jpg
    non_banana_tree_02.jpg
  leaf_segmentation/
    manifest.json
    banana_leaf_closeup.jpg
    banana_leaf_crop_upper.jpg
    banana_leaf_crop_midrib.jpg
    banana_leaf_crop_lower.jpg
  leaf_disease/
    manifest.json
    (same four files as leaf_segmentation, reused as development placeholders)
```

## Status per model

`tree_classification` uses real whole-tree photos (from the Roboflow banana-tree-detection
test split, via `experiment/01_tree_classification.ipynb`) analysed by the real trained
checkpoint (`tree_cls_real_v1`) — see each entry's `description` for its real confidence score.

`leaf_segmentation` is also promoted to its real trained checkpoint (`leaf_seg_real_v1`, via
`experiment/02_leaf_segmentation.ipynb`). Its four images are still the project's own single leaf
photo and three crops of it (not a curated farm sample set) — replace them with real field photos
when available — but the measurements shown are genuine model output, not placeholder numbers.

`leaf_disease` is now also promoted to its real trained checkpoint (`leaf_disease_real_v1`, via
`experiment/03_leaf_disease_segmentation.ipynb`). It reuses the same leaf photo/crops as
`leaf_segmentation` (not a curated farm sample set) — replace them with real field photos when
available — but the diagnoses shown are genuine model output, not random dummy-weight numbers.

## manifest.json

A JSON list, one object per photo:

```json
[
  {
    "file": "healthy_leaf_01.jpg",
    "title": "Healthy leaf, Chitwan farm",
    "description": "Photographed in the morning, no visible damage.",
    "latitude": 27.5291,
    "longitude": 84.3542
  }
]
```

| Field | Required | Rules |
|-------|----------|-------|
| `file` | yes | Path of the image **inside this folder** (sub-folders are fine). No `..`, no absolute paths. The file must exist. |
| `title` | yes | 1-200 characters. Shown in the history table. |
| `description` | no | Up to 1000 characters. |
| `latitude`, `longitude` | no | Give both or neither. Degrees, -90..90 and -180..180. |

Any other key is an error (a typo such as `lat` is reported, not ignored). The whole manifest is checked before anything is
written, and every problem is listed at once.

## Adding your real photos

1. Copy the photos into the model's folder (JPEG, PNG or WEBP; keep them small, about 1000 px on the long side is plenty,
   because they are committed to git). Phone photos work, EXIF rotation is applied and EXIF is not stored.
2. Add one entry per photo to that folder's `manifest.json`, and delete the placeholder entries and images.
3. Seed (from `Backend/`, venv active):

   ```powershell
   python -m scripts.seed_samples --model tree_classification
   python -m scripts.seed_samples --model leaf_segmentation
   ```

   The script prints one line per file: `SEEDED`, `SKIPPED` (already there) or `FAILED` (with the reason). It exits with
   code 1 if any file failed and 2 if nothing could be attempted (bad manifest, model unavailable, database not ready).
4. Running it again is safe: an image that is already a sample of that model is skipped. (Identity is the hash of the stored,
   normalised JPEG, so re-saving the same photo is still recognised.)

## Refreshing after the weights change

Samples keep the result of whatever model made them. After installing the real weights (and updating `*_MODEL_VERSION` and
`*_IS_PLACEHOLDER` in `.env`), re-analyse them:

```powershell
python -m scripts.seed_samples --model tree_classification --force
python -m scripts.seed_samples --model leaf_segmentation --force
```

`--force` analyses each image again, then soft-deletes the old row. If the new analysis fails, the old sample stays
visible. The old image files remain on disk (soft delete keeps files).

No SQL Server yet? `python -m scripts.dev_server_sqlite --seed` seeds both sets into a throwaway SQLite database (see the
README).
