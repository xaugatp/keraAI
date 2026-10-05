# 0011 — Ultralytics forced offline; the registry degrades per model; reasons are redacted

**Status:** accepted · **Date:** 2026-10-05

## Context
`import ultralytics` resolves public DNS names at import to decide it is "online", which enables usage analytics and
asset downloads. The app must start even when weights are missing, and `/health/ready` and `GET /models` are
client-visible (spec §11).

## Decision
- `TreeClassifier.load()` forces `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=False`, `YOLO_VERBOSE=False` (forced, not
  `setdefault`). A subprocess test proves zero socket attempts (its control run makes 2).
- `smp.Unet(encoder_weights=None)` avoids the ImageNet download; checkpoints load with `torch.load(weights_only=True)`.
- `ModelRegistry` loads each enabled model. Any failure marks only that model `unavailable` with a reason (absolute paths
  reduced to file names for clients; full detail in logs and `check_env`); predict then returns `503 MODEL_UNAVAILABLE`.
- Warm-up ordering: the registry marks a model ready, then warms it up, and reverts to unavailable if the warm-up fails
  (`warmup()` goes through `predict()`, which needs the ready flag).

## Consequences
+ No outbound network from inference, no pickle-execution risk, degraded mode instead of a crash.
− Ultralytics still writes `%APPDATA%\Ultralytics\settings.json` on first import.
− The ready-then-warm-up ordering is a small wart; a later refactor can let `warmup()` bypass the ready gate.
