from types import SimpleNamespace
from pathlib import Path
import torch
import pytest

from mlb_studio import model_runtime as runtime


class _Detector(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 1, 8, 1, 1), device=x.device)


def test_detection_runtime_returns_visual_envelope(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "decode_detection_predictions",
        lambda *args, **kwargs: [torch.tensor([[0.10, 0.20, 0.75, 0.85, 0.91, 2.0]])],
    )
    compiled = SimpleNamespace(
        raw_model=_Detector(),
        model=_Detector(),
        device=torch.device("cpu"),
        precision="fp32",
    )
    image = torch.linspace(0, 1, 16 * 16).reshape(1, 1, 16, 16)
    out = runtime.run_universal_inference(
        compiled,
        image,
        input_kind="image",
        output_type="detection_pyramid_head",
        task="analyze",
        metadata={},
    )
    assert out["kind"] == "detection"
    assert out["mime"] == "application/x-mlbricks-detection"
    assert out["metadata"]["coordinate_space"] == "normalized_xyxy"
    assert out["metadata"]["detections"] == 1
    assert out["metadata"]["input_image"].startswith("data:image/png;base64,")
    assert out["data"][0][0]["class_id"] == 2
    assert out["data"][0][0]["box_xyxy"] == pytest.approx([0.1, 0.2, 0.75, 0.85])


def test_frontend_renders_detection_boxes_over_image():
    root = Path(__file__).resolve().parents[1]
    react = (root / "frontend" / "src" / "react-runtime.js").read_text(encoding="utf-8")
    legacy = (root / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    css = (root / "frontend" / "src" / "builder.css").read_text(encoding="utf-8")
    assert "env.kind==='detection'" in react
    assert 'env.kind==="detection"' in legacy
    assert "mlb-detection-stage" in react
    assert "mlb-detection-box" in legacy
    assert ".mlb-detection-box" in css
    assert "Raw detection data" in react
