from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mlb_studio import model_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


class _ScalarJEPA(torch.nn.Module):
    def forward(self, x):
        return torch.tensor(0.0005829947622260079, dtype=torch.float32, device=x.device)


def test_jepa_image_runtime_returns_semantic_visual_envelope():
    model = _ScalarJEPA()
    compiled = SimpleNamespace(raw_model=model, model=model, device=torch.device("cpu"), precision="fp32")
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    out = runtime.run_universal_inference(
        compiled,
        image,
        input_kind="image",
        output_type="tensor_output",
        task="analyze",
        metadata={"training_mode": "jepa", "training_task": "jepa", "model_name": "Image JEPA"},
    )
    assert out["kind"] == "jepa"
    assert out["mime"] == "application/x-mlbricks-jepa"
    assert out["data"]["latent_prediction_loss"] == pytest.approx(0.0005829947622260079)
    assert out["metadata"]["lower_is_better"] is True
    assert out["metadata"]["input_image"].startswith("data:image/png;base64,")


def test_coco_classifier_filters_unlabeled_sentinel_before_cuda():
    split = {
        "image": [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 2.0], [2.0, 2.0]]],
        ],
        "label": [0, -1, 16],
    }
    info = {"input_type": "image_input", "task": "classification", "output_classes": 80}
    x, y, _ = runtime._supervised_xy(split, info)
    assert tuple(x.shape) == (2, 1, 2, 2)
    assert y.tolist() == [0, 16]


def test_classifier_rejects_out_of_range_labels_before_cross_entropy():
    split = {
        "image": [[[[0.0, 0.0], [0.0, 0.0]]]],
        "label": [80],
    }
    info = {"input_type": "image_input", "task": "classification", "output_classes": 80}
    with pytest.raises(ValueError, match="Valid labels are 0..79"):
        runtime._supervised_xy(split, info)


def test_detection_runtime_preserves_coco_class_names():
    class Detector(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 1, 85, 1, 1), device=x.device)

    model = Detector()
    compiled = SimpleNamespace(raw_model=model, model=model, device=torch.device("cpu"), precision="fp32")
    names = [f"c{i}" for i in range(80)]
    out = runtime.run_universal_inference(
        compiled,
        torch.zeros((1, 3, 8, 8)),
        input_kind="image",
        output_type="detection_pyramid_head",
        task="analyze",
        metadata={"class_names": names, "score_threshold": 1.1},
    )
    assert out["kind"] == "detection"
    assert out["metadata"]["class_names"] == names


def test_builder_passes_model_semantics_into_universal_runtime():
    source = (ROOT / "src/mlb_studio/builder.py").read_text(encoding="utf-8")
    assert 'class_names = entry.get("class_names")' in source
    assert 'runtime_meta["training_mode"]' in source
    assert 'runtime_meta["training_task"]' in source
    assert 'source_entry["class_names"]' in source


def test_frontends_render_jepa_semantically():
    react = (ROOT / "frontend/src/react-runtime.js").read_text(encoding="utf-8")
    legacy = (ROOT / "frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    static = (ROOT / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    for js in (react, legacy, static):
        assert "LATENT PREDICTION LOSS" in js
        assert "Lower is better" in js
    assert "env.kind==='jepa'" in react
    assert 'env.kind==="jepa"' in legacy
