# 0003 — One uvicorn worker, per-model lock around predict (P-03)

**Status:** accepted (spec P-03) · **Date:** 2026-10-05

## Context
One copy of each model should live in memory (this laptop has no CUDA GPU; the U-Net weights are ~98 MB), and
Ultralytics/torch modules are not guaranteed thread-safe while the service is called from a threadpool.

## Decision
Run a single worker. `Predictor.predict()` holds a per-model `threading.Lock` for the whole
preprocess → infer → postprocess call.

## Consequences
+ Safe and simple.
− Requests for the same model are serialised (CPU latency ≈0.3–0.5 s for Model 2). Fine for a single-owner
  laptop deployment. Scaling later means a separate inference worker/queue (out of v1 scope); the interfaces
  already isolate it.
