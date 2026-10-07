# KeraAI

Computer vision for banana farming. A grower photographs a banana plant or leaf, and KeraAI
identifies the tree, maps leaf damage, and flags disease — all from a single photo.

## What it does

| Model | Task | Architecture | Status |
|---|---|---|---|
| Tree Classification | Banana tree vs. not | YOLOv8-cls | Trained & live |
| Leaf Segmentation | Healthy vs. damaged leaf tissue | U-Net (ResNet-34) | Trained & live |
| Leaf Disease ID | Black Sigatoka / Yellow Sigatoka detection | U-Net (ResNet-34) | Trained & live |

## Stack

- **Frontend** — React 19, Vite, Tailwind CSS
- **Backend** — FastAPI, SQLAlchemy 2.0, SQL Server
- **Models** — PyTorch / Ultralytics, trained on real field data

## Repository layout

```
Frontend/     React web app — see Frontend/README.md
Backend/      FastAPI service — see Backend/README.md
experiment/   Training notebooks for all three models
```

## Getting started

Each half of the app is self-contained and documents its own setup:

- **Backend setup** → [`Backend/README.md`](./Backend/README.md)
- **Frontend setup** → [`Frontend/README.md`](./Frontend/README.md)
