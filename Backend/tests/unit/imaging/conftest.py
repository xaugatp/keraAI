from __future__ import annotations

import pytest
from PIL import Image, ImageFile


@pytest.fixture(autouse=True)
def _restore_pillow_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """decode() sets Pillow's process-wide limits; keep tests from leaking them.

    A low MAX_IMAGE_PIXELS left behind by a decompression-bomb test would make
    any later Image.open in the suite (even in unrelated tests) fail.
    """
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", ImageFile.LOAD_TRUNCATED_IMAGES)
