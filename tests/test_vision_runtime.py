from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mlb_studio.model_runtime import ModelCompileError, TensorGraph


class _FakeVisionClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = dict(kwargs)
        self.in_channels = int(kwargs.get("in_channels", 3))
        self.num_classes = int(kwargs.get("num_classes", 10))
        self.head = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        pooled = images.mean(dim=(-2, -1))
        return self.head(pooled)


def _graph(component_type, *, engine="Serpentine", num_classes=7):
    nodes = [
        {"id": "input", "type": "image_input", "name": "Image Input", "params": {}},
        {
            "id": "vision",
            "type": component_type,
            "name": "VESA" if component_type == "vesa" else "VisualBOLT",
            "params": {
                "image_size": 16,
                "patch_size": 4,
                "in_channels": 3,
                "num_classes": num_classes,
                "dim": 24,
                "depth": 2,
                "heads": 4,
                "latent_dim": 8,
                "engine": engine,
                "position": "auto",
                "scan": "cross",
                "backend": "pytorch",
            },
        },
    ]
    edges = [{
        "source": "input",
        "target": "vision",
        "source_port": "main_out",
        "target_port": "main_in",
        "kind": "main",
    }]
    return TensorGraph(
        nodes=nodes,
        edges=edges,
        custom_components={},
        runtime={"device": "cpu", "backend": "pytorch", "precision": "fp32", "model_dim": 24},
    )


@pytest.mark.parametrize("component_type", ["vesa", "visualbolt"])
def test_image_input_to_vision_classifier_executes_and_backpropagates(monkeypatch, component_type):
    from mlb_studio import api_graph_runtime

    original = api_graph_runtime.IMPORT_POOL.resolve_component

    def resolve(key, *args, **kwargs):
        if key == component_type:
            return _FakeVisionClassifier
        return original(key, *args, **kwargs)

    monkeypatch.setattr(api_graph_runtime.IMPORT_POOL, "resolve_component", resolve)

    graph = _graph(component_type, engine="Serpentine", num_classes=7)
    images = torch.randn(3, 3, 16, 16, requires_grad=True)
    logits = graph(images)

    assert logits.shape == (3, 7)
    loss = logits.square().mean()
    loss.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


@pytest.mark.parametrize("component_type", ["vesa", "visualbolt"])
@pytest.mark.parametrize("engine", ["ViT", "VisionTransformer", "CNN"])
def test_supported_vision_classifier_engines_compile(monkeypatch, component_type, engine):
    from mlb_studio import api_graph_runtime

    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key, *args, **kwargs: _FakeVisionClassifier,
    )

    graph = _graph(component_type, engine=engine, num_classes=5)
    result = graph(torch.randn(2, 3, 16, 16))
    assert result.shape == (2, 5)


@pytest.mark.parametrize("component_type", ["vesa", "visualbolt"])
@pytest.mark.parametrize("engine", ["Diffusion", "AR"])
def test_non_classifier_vision_modes_fail_with_clear_contract_error(monkeypatch, component_type, engine):
    from mlb_studio import api_graph_runtime

    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key, *args, **kwargs: _FakeVisionClassifier,
    )

    with pytest.raises(ModelCompileError, match="not an image-classification TensorGraph mode"):
        _graph(component_type, engine=engine)


def test_image_input_is_exact_identity_source():
    graph = TensorGraph(
        nodes=[{"id": "input", "type": "image_input", "name": "Image Input", "params": {}}],
        edges=[],
        custom_components={},
        runtime={"device": "cpu", "backend": "pytorch", "precision": "fp32"},
    )
    images = torch.randn(2, 3, 8, 8)
    assert graph(images) is images
