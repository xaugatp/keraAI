# KeraAI — Experiment Notebooks

End-to-end, runnable training notebooks for all three KeraAI models, built from the real
Roboflow exports in `../Data` (git-ignored — see root `.gitignore`). These are **experiments**:
they produce real metrics on real data, but their weights are not wired into `Backend/` and no
ADR has been written for Model 3 yet (see its notebook's Section 0).

## Setup

```powershell
cd experiment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab      # or open the .ipynb files in VS Code / your usual notebook UI
```

Python 3.11, CPU-only torch (this laptop has an Intel Arc GPU, not NVIDIA — no CUDA). The same
code trains faster on a CUDA machine with no changes; device selection is automatic
(`common.pick_device()`).

Each notebook extracts the zip it needs from `../Data` into `experiment/data_cache/` on first run
(cached after that — delete `data_cache/` to force re-extraction). Nothing under `data_cache/`,
`outputs/`, `runs/`, or `checkpoints/` is committed (see `.gitignore`).

Every notebook also respects a `SMOKE_TEST=1` environment variable, which cuts epochs/batches down
to a 2-epoch correctness check — used to validate all three notebooks run end-to-end before
committing to a full training run. Example: `$env:SMOKE_TEST=1; jupyter nbconvert --execute ...`

## Notebooks

| Notebook | Model | Architecture | Data source |
|---|---|---|---|
| `01_tree_classification.ipynb` | Model 1 — tree classification | YOLOv8n-cls (per ADR 0013) | `Data/Banana-Tree-Detection.v1i.yolov8.zip` |
| `02_leaf_segmentation.ipynb` | Model 2 — leaf segmentation | `smp.Unet(resnet34)`, 2-channel sigmoid (per ADR 0012) | `Data/Leaf Segmentation.v1i.yolov8.zip` |
| `03_leaf_disease_segmentation.ipynb` | Model 3 — leaf disease ID | `smp.Unet(resnet34)`, 3-channel sigmoid (**architecture decision documented in-notebook, no ADR yet**) | `Data/Leaf Segmentation.v1i.yolov8.zip` |

Each is end-to-end: data loading → EDA → leakage/stratification-aware split → preprocessing →
architecture → loss design → a short optimizer comparison (the ask was explicitly "deciding best
optimizers" — each notebook runs a small AdamW/SGD/[RMSProp] bake-off rather than assuming one) →
full training with LR scheduling → evaluation (IoU/Dice/precision/recall, confusion matrices,
threshold tuning) → inference-speed profiling → qualitative failure analysis → known limitations.

`common.py` holds the logic shared by all three (seeding, device pick, zip extraction, YOLO-polygon
mask rasterization, group-aware splitting, Dice/BCE loss) so it's implemented and reviewed once,
not copy-pasted three times. The `build_nb*.py` scripts are the notebooks' source of truth — each
one programmatically assembles its `.ipynb` via `nbformat`; regenerate with
`python build_nb1_tree_classification.py` (etc.) after editing a cell's content there, rather than
hand-editing notebook JSON.

## Data audit — what's actually in `../Data` (verified, not assumed)

| Path | Contents | Used here? |
|---|---|---|
| `Data/Banana-Tree-Detection.v1i.yolov8.zip` | 708 images, `Banana`/`Non Banana`, full-frame boxes (i.e. really a classification export) | ✅ Model 1 |
| `Data/Banana-Tree-Detection.v1i.yolov8-obb.zip` | Same images, oriented boxes that are always the full frame — no extra signal | ❌ not used |
| `Data/Leaf Segmentation.v1i.yolov8.zip` | 161 images, polygon labels: `Leaf_boundary` + one of `{healthy_leaf, black_sigatoka_disease, cordana, yellow_sigatoka}` per image | ✅ Models 2 & 3 |
| `Data/Leaf Segmentation.v1i.coco-segmentation.zip` | Same data, COCO JSON export | ❌ not used (used the YOLO-polygon export instead, see `common.polygon_row_to_mask` — avoids a `pycocotools` dependency with known Windows build friction, for an identical result) |
| `Data/Old/banana-tree-detection.*` | 113 images, 1 class, real (non-full-frame) boxes — matches the reference GitHub repo's Stage 1 exactly | ❌ superseded, not used |
| `Data/Old/banana-leaf-segmentation.*` | 50 images, 4 raw categories — matches the reference repo's Stage 2 exactly | ❌ superseded, not used |

Every category above was verified by opening actual label files and sample images, not inferred
from filenames alone — see each notebook's Section 2 for the checks (class distributions,
co-occurrence, a visual nesting check, and — for Model 1 — a filename-based leakage check that
turned up **real cross-split duplication in Roboflow's own pre-made split**, fixed by re-splitting
grouped by capture session).

## Reference code

`_ref_repo/` is a local, untracked clone of `https://github.com/xaugatp/PRT691_AIPratices`
(git-ignored) — the prior PRT691 coursework notebooks for Models 1 & 2 on the old, smaller
datasets. Notebook 02 here reuses its architecture and loss design nearly unchanged (just on new,
larger data, with one methodology fix: the affected-area binarization threshold is now tuned on
validation, not test — see Notebook 02 Section 9.2, and ADR 0012's own critique of the original).
Notebook 01 deliberately **diverges** from the reference repo's detector in favor of the
classifier ADR 0013 mandates. Notebook 03 has no reference counterpart; see its Section 0 for the
from-scratch architecture reasoning.

## Known cross-cutting limitations

- **CPU-only training** on this machine — every notebook's epoch/batch budget is sized for that;
  swapping to a CUDA machine needs no code changes.
- **Small datasets** (108–708 images per task) — expect sensitivity to the specific lighting/
  background conditions captured here; each notebook's "Known limitations" section is explicit
  about where this bites hardest (e.g. Model 3's `yellow_sigatoka` class, n=7).
- **None of these weights are wired into `Backend/`** — `Backend/weights/*_dummy_v0.pt` remains
  the active placeholder there (ADR 0010) until the owner reviews these results and decides to
  promote a checkpoint.
