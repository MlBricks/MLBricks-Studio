from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mlb_studio import model_runtime as runtime


class _HalfAutoencoder(torch.nn.Module):
    def forward(self, x):
        return x * 0.5


def test_autoencoder_runtime_returns_visual_reconstruction_envelope():
    model = _HalfAutoencoder()
    compiled = SimpleNamespace(
        raw_model=model,
        model=model,
        device=torch.device("cpu"),
        precision="fp32",
    )
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    out = runtime.run_universal_inference(
        compiled,
        image,
        input_kind="image",
        output_type="tensor_output",
        task="analyze",
        metadata={},
    )
    assert out["kind"] == "reconstruction"
    assert out["mime"] == "application/x-mlbricks-reconstruction"
    assert out["data"]["reconstructed_image"].startswith("data:image/png;base64,")
    assert out["metadata"]["input_image"].startswith("data:image/png;base64,")
    assert out["data"]["mse"] == pytest.approx(((image * 0.5 - image) ** 2).mean().item())
    assert out["data"]["mae"] == pytest.approx((image * 0.5 - image).abs().mean().item())
    assert out["data"]["psnr_db"] is not None


def test_non_image_tensor_output_remains_tensor():
    class Features(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 8), device=x.device)

    model = Features()
    compiled = SimpleNamespace(raw_model=model, model=model, device=torch.device("cpu"), precision="fp32")
    out = runtime.run_universal_inference(
        compiled,
        torch.zeros((1, 3, 16, 16)),
        input_kind="image",
        output_type="tensor_output",
        task="analyze",
        metadata={},
    )
    assert out["kind"] == "tensor"


def test_frontend_renders_autoencoder_side_by_side_images():
    root = Path(__file__).resolve().parents[1]
    react = (root / "frontend/src/react-runtime.js").read_text(encoding="utf-8")
    legacy = (root / "frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    css = (root / "frontend/src/builder.css").read_text(encoding="utf-8")
    static_js = (root / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    assert "env.kind==='reconstruction'" in react
    assert 'env.kind==="reconstruction"' in legacy
    assert "MODEL INPUT" in react
    assert "RECONSTRUCTION" in legacy
    assert "mlb-reconstruction-images" in react
    assert ".mlb-reconstruction-card" in css
    assert "mlb-reconstruction-images" in static_js
