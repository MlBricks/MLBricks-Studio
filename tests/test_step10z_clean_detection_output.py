from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mlb_studio import model_runtime as runtime
from mlb_studio.graph import primitive_catalog


class _Detector(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 1, 8, 1, 1), device=x.device)


def _compiled():
    model = _Detector()
    return SimpleNamespace(
        raw_model=model,
        model=model,
        device=torch.device("cpu"),
        precision="fp32",
    )


def test_runtime_uses_clean_detection_preview_defaults(monkeypatch):
    captured = {}

    def fake_decode(*args, **kwargs):
        captured.update(kwargs)
        # Two almost identical boxes with different classes. Class-aware NMS
        # would keep both; the display-only agnostic pass should keep one.
        return [torch.tensor([
            [0.10, 0.10, 0.80, 0.80, 0.90, 0.0],
            [0.11, 0.11, 0.79, 0.79, 0.82, 56.0],
        ])]

    monkeypatch.setattr(runtime, "decode_detection_predictions", fake_decode)
    image = torch.zeros((1, 1, 16, 16))
    out = runtime.run_universal_inference(
        _compiled(),
        image,
        input_kind="image",
        output_type="detection_pyramid_head",
        task="analyze",
        metadata={},
    )

    assert captured["score_threshold"] == pytest.approx(0.40)
    assert captured["iou_threshold"] == pytest.approx(0.45)
    assert captured["max_detections"] == 20
    assert out["metadata"]["display_class_agnostic_nms"] is True
    assert out["metadata"]["display_nms_iou_threshold"] == pytest.approx(0.60)
    assert out["metadata"]["detections"] == 1
    assert len(out["data"][0]) == 1
    assert out["data"][0][0]["score"] == pytest.approx(0.90)


def test_runtime_detection_cleanup_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "decode_detection_predictions",
        lambda *args, **kwargs: [torch.tensor([
            [0.10, 0.10, 0.80, 0.80, 0.90, 0.0],
            [0.11, 0.11, 0.79, 0.79, 0.82, 56.0],
        ])],
    )
    image = torch.zeros((1, 1, 16, 16))
    out = runtime.run_universal_inference(
        _compiled(),
        image,
        input_kind="image",
        output_type="detection_pyramid_head",
        task="analyze",
        metadata={"display_class_agnostic_nms": False},
    )
    assert out["metadata"]["detections"] == 2
    assert len(out["data"][0]) == 2


def test_detection_nms_component_uses_preview_friendly_defaults():
    catalog = {item["type"]: item for item in primitive_catalog()}
    api = {item["key"]: item["value"] for item in catalog["detection_nms"]["api"]}
    assert api["score_threshold"] == pytest.approx(0.40)
    assert api["iou_threshold"] == pytest.approx(0.45)
    assert api["max_detections"] == 20


def test_low_level_decoder_defaults_remain_backward_compatible():
    # Training metrics and direct decoder callers keep their original defaults;
    # only the user-facing runtime preview becomes stricter.
    import inspect
    sig = inspect.signature(runtime.decode_detection_predictions)
    assert sig.parameters["score_threshold"].default == pytest.approx(0.25)
    assert sig.parameters["iou_threshold"].default == pytest.approx(0.5)
    assert sig.parameters["max_detections"].default == 100
