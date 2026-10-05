"""Behaviour of the `Predictor` template method and `ModelInfo` (spec §8)."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

import app.ml.base as base
from app.core.errors import AppError, InferenceError, InvalidImageError, ModelUnavailableError
from app.ml.base import ModelInfo, ModelLoadError, PredictionOutput, Predictor, Timings
from tests.fakes import FakePredictor, make_tree_details


class Stub(Predictor):
    """Minimal concrete predictor whose stages are plain callables, for precise control."""

    key = "stub"

    def __init__(self, *, infer=None, preprocess=None, postprocess=None, **kwargs) -> None:
        super().__init__(display_name="Stub", version="v1", task="classify", **kwargs)
        self._infer = infer or (lambda x: x)
        self._pre = preprocess or (lambda image: image)
        self._post = postprocess or (
            lambda raw: PredictionOutput(
                label="a",
                display_label="A",
                confidence=0.9,
                is_uncertain=False,
                details=make_tree_details(),
            )
        )
        self.calls: list[str] = []

    def load(self) -> None:
        pass

    def preprocess(self, image):
        self.calls.append("preprocess")
        return self._pre(image)

    def infer(self, x):
        self.calls.append("infer")
        return self._infer(x)

    def postprocess(self, raw):
        self.calls.append("postprocess")
        return self._post(raw)


def image() -> Image.Image:
    return Image.new("RGB", (64, 64), (10, 20, 30))


def raiser(exc: BaseException):
    """A stage callable that always raises `exc`."""

    def stage(*_args):
        raise exc

    return stage


class TestReadiness:
    def test_a_fresh_predictor_is_not_ready(self):
        predictor = Stub()
        assert predictor.is_ready is False
        assert predictor.info.status == "unavailable"
        assert predictor.info.reason == "not loaded yet"

    def test_predict_refuses_when_not_ready(self):
        predictor = Stub()
        with pytest.raises(ModelUnavailableError) as raised:
            predictor.predict(image())
        assert raised.value.status_code == 503
        assert raised.value.code == "MODEL_UNAVAILABLE"
        assert predictor.calls == []  # no stage ran

    def test_mark_ready_then_unavailable_round_trip(self):
        predictor = Stub()
        predictor.mark_ready()
        assert predictor.is_ready and predictor.info.status == "ready"
        assert predictor.info.reason is None
        predictor.mark_unavailable("weights vanished")
        assert not predictor.is_ready
        assert predictor.info.status == "unavailable"
        assert predictor.info.reason == "weights vanished"
        with pytest.raises(ModelUnavailableError):
            predictor.predict(image())

    def test_info_reports_identity_classes_and_placeholder_flag(self):
        predictor = Stub(is_placeholder=True)
        predictor.classes = ["a", "b"]
        predictor.mark_ready()
        assert predictor.info == ModelInfo(
            key="stub",
            display_name="Stub",
            version="v1",
            task="classify",
            classes=["a", "b"],
            status="ready",
            reason=None,
            is_placeholder=True,
        )

    def test_info_classes_is_a_copy(self):
        predictor = Stub()
        predictor.classes = ["a"]
        predictor.info.classes.append("mutated")
        assert predictor.classes == ["a"]

    def test_placeholder_defaults_to_false(self):
        assert Stub().info.is_placeholder is False


class TestPipeline:
    def test_stages_run_in_order_and_feed_each_other(self):
        seen: dict[str, object] = {}

        def infer(x):
            seen["infer_in"] = x
            return ("raw", x)

        def postprocess(raw):
            seen["post_in"] = raw
            return PredictionOutput("a", "A", 0.9, False, make_tree_details())

        predictor = Stub(
            preprocess=lambda img: ("pre", img.size), infer=infer, postprocess=postprocess
        )
        predictor.mark_ready()
        output = predictor.predict(image())
        assert predictor.calls == ["preprocess", "infer", "postprocess"]
        assert seen["infer_in"] == ("pre", (64, 64))
        assert seen["post_in"] == ("raw", ("pre", (64, 64)))
        assert output.label == "a"

    def test_timings_are_filled_per_stage(self, monkeypatch):
        ticks = iter([0.000, 0.010, 0.050, 0.052])  # t0, t1, t2, t3
        monkeypatch.setattr(base, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
        predictor = Stub()
        predictor.mark_ready()
        output = predictor.predict(image())
        assert output.timings == Timings(preprocess_ms=10, inference_ms=40, postprocess_ms=2)
        assert output.timings.model_total_ms == 52

    def test_real_clock_produces_non_negative_timings(self):
        predictor = Stub(infer=lambda x: time.sleep(0.02) or x)
        predictor.mark_ready()
        timings = predictor.predict(image()).timings
        assert timings.inference_ms >= 15
        assert min(timings.preprocess_ms, timings.postprocess_ms) >= 0

    def test_default_timings_are_zero(self):
        assert Timings().model_total_ms == 0


class TestErrorHandling:
    def test_unexpected_exception_becomes_inference_error(self, caplog):
        boom = RuntimeError("cuda exploded")
        predictor = Stub(infer=raiser(boom))
        predictor.mark_ready()
        with (
            caplog.at_level(logging.ERROR, logger="app.ml.base"),
            pytest.raises(InferenceError) as raised,
        ):
            predictor.predict(image())
        assert raised.value.__cause__ is boom
        assert raised.value.status_code == 500
        assert raised.value.code == "INFERENCE_FAILED"
        assert "RuntimeError: cuda exploded" in raised.value.detail
        assert any("inference_failed" in record.getMessage() for record in caplog.records)

    @pytest.mark.parametrize("stage", ["preprocess", "infer", "postprocess"])
    def test_failure_in_any_stage_is_wrapped(self, stage):
        predictor = Stub(**{stage: raiser(ValueError(f"bad {stage}"))})
        predictor.mark_ready()
        with pytest.raises(InferenceError, match=f"ValueError: bad {stage}"):
            predictor.predict(image())

    def test_app_errors_pass_through_unwrapped(self):
        original = InvalidImageError("nope")
        predictor = Stub(infer=raiser(original))
        predictor.mark_ready()
        with pytest.raises(InvalidImageError) as raised:
            predictor.predict(image())
        assert raised.value is original

    def test_inference_error_raised_by_a_stage_is_not_double_wrapped(self):
        original = InferenceError("model said no")
        predictor = Stub(postprocess=raiser(original))
        predictor.mark_ready()
        with pytest.raises(InferenceError) as raised:
            predictor.predict(image())
        assert raised.value is original
        assert isinstance(raised.value, AppError)

    def test_lock_is_released_after_a_failure(self):
        predictor = FakePredictor(fail=True)
        with pytest.raises(InferenceError):
            predictor.predict(image())
        predictor.fail = False
        assert predictor.predict(image()).label == "banana_tree"  # would deadlock otherwise

    def test_model_load_error_is_a_plain_exception_not_an_app_error(self):
        # It must never reach the HTTP layer: only the registry (startup) sees it.
        assert not issubclass(ModelLoadError, AppError)


class TestSerialisation:
    def test_predict_calls_are_serialised_by_the_lock(self):
        active = 0
        peak = 0
        guard = threading.Lock()

        def slow_infer(x):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with guard:
                active -= 1
            return x

        predictor = Stub(infer=slow_infer)
        predictor.mark_ready()
        results: list[PredictionOutput] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                results.append(predictor.predict(image()))
            except BaseException as exc:  # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        assert len(results) == 8
        assert peak == 1  # never two inferences at once


class TestWarmup:
    def test_warmup_runs_one_dummy_inference_through_predict(self):
        predictor = Stub()
        predictor.mark_ready()
        predictor.warmup()
        assert predictor.calls == ["preprocess", "infer", "postprocess"]

    def test_warmup_image_is_a_small_rgb_picture(self):
        warm = Stub().warmup_image()
        assert warm.mode == "RGB" and warm.size == (256, 256)

    def test_warmup_requires_the_ready_flag(self):
        # Documented consequence of the template method: the registry marks the
        # model ready just before warming it up.
        with pytest.raises(ModelUnavailableError):
            Stub().warmup()


class TestFakePredictorItself:
    def test_returns_fresh_output_each_call(self):
        fake = FakePredictor()
        first, second = fake.predict(image()), fake.predict(image())
        assert first is not second
        assert first.details is not second.details
        assert first.details == second.details

    def test_deterministic_defaults_match_the_spec_example(self):
        output = FakePredictor().predict(image())
        assert (output.label, output.display_label, output.confidence) == (
            "banana_tree",
            "Banana tree",
            0.947,
        )
        assert output.is_uncertain is False
        assert output.details.model_dump()["probabilities"][1]["probability"] == 0.053

    def test_not_ready_fake_refuses(self):
        with pytest.raises(ModelUnavailableError):
            FakePredictor(ready=False).predict(image())

    def test_result_image_is_copied(self):
        overlay = Image.new("RGB", (8, 8), (255, 0, 0))
        output = FakePredictor(result_image=overlay).predict(image())
        assert output.result_image is not overlay
        assert output.result_image is not None and output.result_image.size == (8, 8)

    def test_custom_error_and_key(self):
        fake = FakePredictor("leaf_segmentation", fail=True, error=InvalidImageError("x"))
        assert fake.key == "leaf_segmentation"
        with pytest.raises(InvalidImageError):
            fake.predict(image())

    def test_records_what_it_saw(self):
        fake = FakePredictor()
        picture = image()
        fake.predict(picture)
        assert fake.predict_images == [picture]
