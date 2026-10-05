from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(preference: str) -> str:
    """`auto` -> `cuda:0` when CUDA is usable, otherwise `cpu`; explicit values pass through.

    torch is imported here (not at module top) so the rest of `app.ml` stays
    importable in the fast test run and in degraded mode.
    """
    pref = preference.strip().lower()
    if pref != "auto":
        return pref
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("inference_device_resolved", extra={"device": device})
    return device
