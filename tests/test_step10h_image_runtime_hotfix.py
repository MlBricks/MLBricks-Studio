from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mlb_studio.builder import Builder
from mlb_studio import universal_io
from mlb_studio.universal_io import load_single_input, normalize_input_config


def test_google_image_result_url_unwraps_embedded_imgurl():
    direct = "https://cdn.example.org/picture.jpg?x=1"
    wrapped = (
        "https://www.google.com/imgres?q=demo&imgurl="
        "https%3A%2F%2Fcdn.example.org%2Fpicture.jpg%3Fx%3D1&imgrefurl=https%3A%2F%2Fexample.org"
    )
    assert universal_io._embedded_media_url(wrapped) == direct


def test_image_url_that_returns_html_has_actionable_error(monkeypatch):
    class FakeResponse:
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"<html><body>search page</body></html>"

    monkeypatch.setattr(universal_io.urllib.request, "urlopen", lambda request, timeout=15: FakeResponse())
    with pytest.raises(ValueError, match="returned text/html instead of an image"):
        universal_io._read_bytes("https://example.org/search?q=cat", expected_media="image")


def test_runtime_image_loader_honors_grayscale_model_contract(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (20, 10), (30, 60, 90)).save(path)
    env = normalize_input_config(
        {
            "input_kind": "image",
            "input_mode": "single",
            "task_type": "classify",
            "input_source_type": "path_or_url",
            "input_source": str(path),
            "prompt": "stale prompt should disappear",
        }
    )
    assert env.prompt == ""
    value, meta = load_single_input(env, image_size=16, image_channels=1)
    assert tuple(value.shape) == (1, 1, 16, 16)
    assert {k: meta[k] for k in ("width", "height", "channels")} == {"width": 16, "height": 16, "channels": 1}
    assert meta["display_image"].startswith("data:image/jpeg;base64,")


def test_old_gallery_cnn_3x224_metadata_is_repaired_from_graph_shape():
    entry = {
        "name": "CNN",
        "architecture": {
            "nodes": [
                {"id": "x", "type": "image_input", "params": {"channels": 3, "image_size": 224}},
                {"id": "c1", "type": "conv2d", "params": {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1}},
                {"id": "a1", "type": "relu", "params": {}},
                {"id": "p1", "type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2, "padding": 0}},
                {"id": "c2", "type": "conv2d", "params": {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
                {"id": "a2", "type": "relu", "params": {}},
                {"id": "p2", "type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2, "padding": 0}},
                {"id": "flat", "type": "flatten", "params": {"start_dim": 1, "end_dim": -1}},
                {"id": "head", "type": "classifier", "params": {"dim": 256, "classes": 3}},
            ]
        },
    }
    assert Builder._runtime_input_image_spec(entry) == (16, 1)


def test_frontend_cnn_contract_and_prompt_gating_are_published():
    root = Path(__file__).parents[1]
    legacy = (root / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    react = (root / "frontend" / "src" / "react-runtime.js").read_text(encoding="utf-8")
    static = (root / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")

    expected = 'const x=add("image_input","Image Input · COCO128",{channels:3,image_size:128,input_mode:"single"});'
    assert expected in legacy
    assert expected in static
    assert "function runtimeTaskUsesPrompt(kind,task)" in legacy
    assert "function runtimeTaskUsesPrompt(kind,task)" in react
    assert "runtimeTaskUsesPrompt(kind,config.task_type)" in legacy
    assert "prompt:runtimeTaskUsesPrompt(kind,task)?" in react
