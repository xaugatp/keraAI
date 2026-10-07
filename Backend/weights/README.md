# Model weights

Weights are **never committed** (`*.pt` and everything in this folder except this
README is git-ignored). Copy the files here by hand.

| File | Model | Format | Must satisfy |
|------|-------|--------|--------------|
| `tree_cls_v1.pt` | Model 1 — `tree_classification` | Original Ultralytics `.pt` (a zip archive) | `model.task == "classify"`; class names include `TREE_POSITIVE_CLASS` (`Banana`, matching the real checkpoint's class names) and every key of `TREE_DISPLAY_NAMES` |
| `leaf_seg_v1.pt` | Model 2 — `leaf_segmentation` | Bare PyTorch `state_dict` (`torch.save(model.state_dict(), ...)`) | U-Net, encoder = `LEAF_SEG_ENCODER` (`resnet34`), 3 input channels, 2 output channels (leaf, affected) |
| `leaf_disease_v1.pt` | Model 3 — `leaf_disease` | Bare PyTorch `state_dict` (`torch.save(model.state_dict(), ...)`) | U-Net, encoder = `LEAF_DISEASE_ENCODER` (`resnet34`), 3 input channels, 3 output channels (leaf, black_sigatoka, yellow_sigatoka) |

Paths are configurable (`TREE_WEIGHTS_PATH`, `LEAF_SEG_WEIGHTS_PATH`, `LEAF_DISEASE_WEIGHTS_PATH`); a relative
path is resolved against `Backend/`, not the working directory.

## Do not unzip a `.pt`

A `.pt` file is itself a zip archive. Use it as one single file. `Backend/yolov8n/`
is an *unzipped* copy of Ultralytics' stock COCO detector — it is **not** the banana
tree model (`task=detect`, 80 classes) and must not be used or re-zipped.

## Dummy weights (plumbing tests only)

Until the real checkpoints exist:

```powershell
.venv\Scripts\activate
python -m scripts.make_dummy_weights          # refuses to overwrite existing files; --force to overwrite
```

This writes **randomly initialised** stand-ins in the exact formats above. Their
predictions are meaningless. Put these lines in `Backend/.env` so database rows
from the dummy era are identifiable and the UI can badge them:

```dotenv
TREE_MODEL_VERSION=tree_cls_dummy_v0
TREE_IS_PLACEHOLDER=true
LEAF_SEG_MODEL_VERSION=leaf_seg_dummy_v0
LEAF_SEG_IS_PLACEHOLDER=true
LEAF_DISEASE_MODEL_VERSION=leaf_disease_dummy_v0
LEAF_DISEASE_IS_PLACEHOLDER=true
```

When the real files arrive: replace them, restore the real `*_MODEL_VERSION`
(e.g. `tree_cls_v1`), and set both `*_IS_PLACEHOLDER` values back to `false`.

## Verify

```powershell
python -m scripts.check_env
python -m scripts.predict_cli --model tree_classification path\to\image.jpg
```
