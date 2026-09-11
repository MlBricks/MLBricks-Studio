from __future__ import annotations

import pytest
import torch

from mlb_studio.graph import primitive_catalog
from mlb_studio.model_runtime import TensorGraph


def _runtime(dim=8):
    return {
        "device": "cpu",
        "backend": "pytorch",
        "precision": "fp32",
        "model_dim": dim,
        "heads": 2,
    }


def _edge(source, target, *, target_port="main_in", source_port="main_out", kind="main"):
    return {
        "id": f"{source}-{target}-{target_port}",
        "source": source,
        "target": target,
        "source_port": source_port,
        "target_port": target_port,
        "kind": kind,
    }


def test_foundation_categories_are_public_in_component_catalog():
    catalog = primitive_catalog()
    by_category = {}
    for item in catalog:
        by_category.setdefault(item.get("category"), set()).add(item.get("type"))

    assert {
        "feature_input", "linear_regression", "logistic_regression", "polynomial_features",
        "knn_classifier", "decision_tree_classifier", "kmeans", "pca",
    } <= by_category["ML Core"]
    assert {
        "linear", "relu", "sigmoid", "conv2d", "rnn", "lstm", "gru", "self_attention",
        "embedding", "dropout", "layernorm", "rmsnorm", "ffn",
    } <= by_category["Deep Learning Core"]
    assert {
        "learnable_parameter", "constant", "matmul", "tensor_add", "tensor_multiply",
        "reduce_mean", "tensor_exp", "transpose", "reshape", "flatten", "unsqueeze", "squeeze",
    } <= by_category["Math & Tensor Ops"]


def test_linear_model_can_be_built_from_math_primitives_and_backpropagate():
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Features", "params": {}},
        {"id": "w", "type": "learnable_parameter", "name": "W", "params": {"shape": "3,2", "init": "ones"}},
        {"id": "matmul", "type": "matmul", "name": "MatMul", "params": {}},
        {"id": "b", "type": "learnable_parameter", "name": "b", "params": {"shape": "2", "init": "zeros"}},
        {"id": "add", "type": "tensor_add", "name": "Add", "params": {}},
        {"id": "out", "type": "tensor_output", "name": "Tensor Output", "params": {}},
    ]
    edges = [
        _edge("x", "matmul", target_port="named_in:a", kind="named"),
        _edge("w", "matmul", target_port="named_in:b", kind="named"),
        _edge("matmul", "add", target_port="named_in:a", kind="named"),
        _edge("b", "add", target_port="named_in:b", kind="named"),
        _edge("add", "out"),
    ]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime())
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]], requires_grad=True)
    y = graph(x)

    assert y.shape == (2, 2)
    assert torch.allclose(y, torch.tensor([[6.0, 6.0], [3.0, 3.0]]))
    y.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in graph.parameters() if p.requires_grad)


def test_linear_and_logistic_regression_convenience_blocks_execute():
    linear = TensorGraph(
        nodes=[{"id": "lr", "type": "linear_regression", "name": "Linear Regression", "params": {"in_features": 4, "out_features": 2}}],
        edges=[], custom_components={}, runtime=_runtime(4),
    )
    logits = TensorGraph(
        nodes=[{"id": "logr", "type": "logistic_regression", "name": "Logistic Regression", "params": {"in_features": 4, "out_features": 1, "output": "logits"}}],
        edges=[], custom_components={}, runtime=_runtime(4),
    )
    probs = TensorGraph(
        nodes=[{"id": "logr", "type": "logistic_regression", "name": "Logistic Regression", "params": {"in_features": 4, "out_features": 1, "output": "probability"}}],
        edges=[], custom_components={}, runtime=_runtime(4),
    )
    x = torch.randn(5, 4)
    assert linear(x).shape == (5, 2)
    assert logits(x).shape == (5, 1)
    p = probs(x)
    assert p.shape == (5, 1)
    assert torch.all((p >= 0) & (p <= 1))


def test_cnn_core_components_compose_into_an_editable_graph():
    nodes = [
        {"id": "input", "type": "feature_input", "name": "Image Tensor", "params": {}},
        {"id": "conv", "type": "conv2d", "name": "Conv2D", "params": {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}},
        {"id": "relu", "type": "relu", "name": "ReLU", "params": {}},
        {"id": "pool", "type": "maxpool2d", "name": "MaxPool2D", "params": {"kernel_size": 2, "stride": 2}},
        {"id": "flat", "type": "flatten", "name": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
    ]
    edges = [_edge("input", "conv"), _edge("conv", "relu"), _edge("relu", "pool"), _edge("pool", "flat")]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime())
    out = graph(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 64)


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru"])
def test_recurrent_core_components_support_sequence_and_last_output(kind):
    seq_graph = TensorGraph(
        nodes=[{"id": "rec", "type": kind, "name": kind.upper(), "params": {"input_size": 5, "hidden_size": 7, "output": "sequence"}}],
        edges=[], custom_components={}, runtime=_runtime(5),
    )
    last_graph = TensorGraph(
        nodes=[{"id": "rec", "type": kind, "name": kind.upper(), "params": {"input_size": 5, "hidden_size": 7, "output": "last"}}],
        edges=[], custom_components={}, runtime=_runtime(5),
    )
    x = torch.randn(2, 4, 5)
    assert seq_graph(x).shape == (2, 4, 7)
    assert last_graph(x).shape == (2, 7)


def test_self_attention_core_preserves_sequence_shape_and_backpropagates():
    graph = TensorGraph(
        nodes=[{"id": "attn", "type": "self_attention", "name": "Self Attention", "params": {"dim": 8, "heads": 2, "causal": True}}],
        edges=[], custom_components={}, runtime=_runtime(8),
    )
    x = torch.randn(2, 5, 8, requires_grad=True)
    y = graph(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None
