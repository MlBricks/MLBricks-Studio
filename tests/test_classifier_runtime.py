from __future__ import annotations

import pytest
import torch

from mlb_studio.model_runtime import TensorGraph


def _runtime(dim=8):
    return {
        "device": "cpu",
        "backend": "pytorch",
        "precision": "fp32",
        "model_dim": dim,
        "heads": 2,
    }


def _graph(params=None, runtime_dim=8):
    node = {
        "id": "classifier-1",
        "type": "classifier",
        "name": "Classifier Head",
        "params": dict(params or {}),
    }
    return TensorGraph(
        nodes=[node],
        edges=[],
        custom_components={},
        runtime=_runtime(runtime_dim),
    )


def test_classifier_sequence_input_mean_pools_and_backpropagates():
    torch.manual_seed(7)
    graph = _graph({"dim": 8, "classes": 3})

    x = torch.randn(4, 5, 8, requires_grad=True)
    logits = graph(x)

    assert logits.shape == (4, 3)
    assert logits.dtype == x.dtype
    assert logits.device == x.device

    loss = logits.square().mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()

    trainable = [p for p in graph.parameters() if p.requires_grad]
    assert trainable
    assert all(p.grad is not None for p in trainable)


def test_classifier_accepts_already_pooled_2d_features_and_runtime_dim_default():
    graph = _graph({"classes": 5}, runtime_dim=8)
    x = torch.randn(2, 8)

    logits = graph(x)

    assert logits.shape == (2, 5)


def test_classifier_rejects_wrong_rank_or_feature_width():
    graph = _graph({"dim": 8, "classes": 2})

    with pytest.raises(ValueError, match=r"\[B,D\].*\[B,T,D\]"):
        graph(torch.randn(8))

    with pytest.raises(ValueError, match="feature width 8"):
        graph(torch.randn(2, 4, 7))
