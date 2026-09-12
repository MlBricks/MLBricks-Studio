from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from mlb_studio import model_runtime as runtime
from mlb_studio.universal_io import load_single_input, normalize_input_config


class _Detector(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 1, 8, 1, 1), device=x.device)


def test_image_input_keeps_pre_resize_display_preview(tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (320, 180), (40, 80, 120)).save(path)
    env = normalize_input_config({
        "input_kind": "image",
        "input_mode": "single",
        "input_source_type": "path_or_url",
        "input_source": str(path),
    })
    tensor, meta = load_single_input(env, image_size=16, image_channels=1)
    assert tuple(tensor.shape) == (1, 1, 16, 16)
    assert (meta["source_width"], meta["source_height"]) == (320, 180)
    assert (meta["display_width"], meta["display_height"]) == (320, 180)
    assert meta["display_image"].startswith("data:image/jpeg;base64,")


def test_detection_runtime_prefers_pre_resize_preview(monkeypatch):
    monkeypatch.setattr(runtime, "decode_detection_predictions", lambda *args, **kwargs: [torch.tensor([[0.1, 0.2, 0.8, 0.9, 0.95, 1.0]])])
    model = _Detector()
    compiled = SimpleNamespace(raw_model=model, model=model, device=torch.device("cpu"), precision="fp32")
    preview = "data:image/jpeg;base64,AAA"
    out = runtime.run_universal_inference(
        compiled,
        torch.zeros((1, 1, 16, 16)),
        input_kind="image",
        output_type="detection_pyramid_head",
        task="analyze",
        metadata={"display_image": preview, "display_width": 640, "display_height": 360, "display_format": "jpeg"},
    )
    assert out["metadata"]["input_image"] == preview
    assert out["metadata"]["image_width"] == 640
    assert out["metadata"]["image_height"] == 360
    assert out["metadata"]["preview_source"] == "pre_resize"


def test_detection_frontend_expands_stage_and_preserves_aspect_ratio():
    root = Path(__file__).resolve().parents[1]
    react = (root / "frontend/src/react-runtime.js").read_text(encoding="utf-8")
    legacy = (root / "frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    css = (root / "frontend/src/builder.css").read_text(encoding="utf-8")
    assert "imageWidth=Number(env.meta&&env.meta.image_width)" in react
    assert "style:stageStyle" in react
    assert "stage.style.aspectRatio" in legacy
    assert "width:min(640px,100%)" in css
    assert ".mlb-detection-image{display:block;width:100%;height:100%" in css
