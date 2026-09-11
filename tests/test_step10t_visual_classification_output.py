from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mlb_studio import model_runtime as runtime


class _Classifier(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32, device=x.device)


def test_image_classification_runtime_returns_semantic_visual_envelope():
    model = _Classifier()
    compiled = SimpleNamespace(
        raw_model=model,
        model=model,
        device=torch.device("cpu"),
        precision="fp32",
    )
    image = torch.linspace(0, 1, 16 * 16).reshape(1, 1, 16, 16)
    out = runtime.run_universal_inference(
        compiled,
        image,
        input_kind="image",
        output_type="classifier",
        task="analyze",
        metadata={},
    )
    assert out["kind"] == "classification"
    assert out["mime"] == "application/x-mlbricks-classification"
    assert out["data"]["predicted_class"] == 1
    assert out["data"]["confidence"] == pytest.approx(max(out["data"]["probabilities"]))
    assert sum(out["data"]["probabilities"]) == pytest.approx(1.0)
    assert out["data"]["logits"] == pytest.approx([1.0, 3.0, 2.0])
    assert out["metadata"]["input_image"].startswith("data:image/png;base64,")


def test_binary_classifier_uses_sigmoid_probability():
    class Binary(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0]], dtype=torch.float32, device=x.device)
    model = Binary()
    compiled = SimpleNamespace(raw_model=model, model=model, device=torch.device("cpu"), precision="fp32")
    out = runtime.run_universal_inference(
        compiled,
        torch.tensor([[0.0, 1.0]]),
        input_kind="tabular",
        output_type="classifier",
        task="classify",
        metadata={},
    )
    assert out["data"]["predicted_class"] == 1
    assert len(out["data"]["probabilities"]) == 2
    assert sum(out["data"]["probabilities"]) == pytest.approx(1.0)


def test_frontend_renders_classification_as_prediction_not_raw_logits():
    root = Path(__file__).resolve().parents[1]
    react = (root / "frontend/src/react-runtime.js").read_text(encoding="utf-8")
    legacy = (root / "frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    css = (root / "frontend/src/builder.css").read_text(encoding="utf-8")
    assert "env.kind==='classification'" in react
    assert 'env.kind==="classification"' in legacy
    assert "mlb-classification-image" in react
    assert "mlb-classification-prob-row" in legacy
    assert "Raw logits" in react
    assert ".mlb-classification-card" in css
