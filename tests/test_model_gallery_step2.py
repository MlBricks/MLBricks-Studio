from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mlb_studio.model_runtime import TensorGraph

ROOT = Path(__file__).resolve().parents[1]


def _runtime(dim=32):
    return {"device": "cpu", "backend": "pytorch", "precision": "fp32", "model_dim": dim, "heads": 1}


def _edge(source, target, *, kind="main", source_port="main_out", target_port="main_in"):
    return {
        "id": f"{source}-{target}-{kind}-{target_port}",
        "source": source,
        "target": target,
        "kind": kind,
        "source_port": source_port,
        "target_port": target_port,
    }


def _named(source, target, target_key):
    return _edge(source, target, kind="named", source_port="named_out:main", target_port=f"named_in:{target_key}")


def test_step2_model_gallery_has_categories_data_links_and_language_reorg():
    js = (ROOT / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    built = (ROOT / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")

    for text in (js, built):
        assert 'const mlbricksCoreCategories=["All Core","Machine Learning","Deep Learning","Signal Processing"]' in text
        assert 'const mlbricksModelCategories=["All Models","Language","Vision","Audio","JEPA","Multimodal","State & Memory"]' in text
        assert 'mlb-model-gallery-filter-select' in text
        assert 'mlb-gallery-flat-nav-tools' in text
        assert 'btn("Open Data","mlb-gallery-action")' in text
        assert '[["core","Core"],["models","Models"],["mine","My Models"],["components","Components"],["data","Data"],["drafts","Drafts"]]' in text
        assert 'name:"50M SLM",category:"Language"' in text
        assert 'name:"200M SLM · SOUP",category:"Language"' in text


def test_step2b_gallery_contains_ready_educational_and_classical_fit_models():
    js = (ROOT / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    for name in [
        "Linear Regression", "Logistic Regression", "Single Neuron", "ANN", "CNN", "RNN", "LSTM", "GRU", "Autoencoder"
    ]:
        assert f'name:"{name}"' in js
    for name in ["KNN", "Decision Tree", "K-Means", "PCA"]:
        assert f'name:"{name}"' in js
    assert 'template:"knn"' in js
    assert 'template:"decision_tree"' in js
    assert 'template:"kmeans"' in js
    assert 'template:"pca"' in js
    for ident in ('model_knn','model_decision_tree','model_kmeans','model_pca'):
        line = next(line for line in js.splitlines() if f'id:"{ident}"' in line)
        assert 'planned:true' not in line
        assert 'template:' in line
    assert 'loadEducationalModelPreset' in js


def test_first_principles_linear_regression_graph_executes():
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 4}},
        {"id": "w", "type": "learnable_parameter", "name": "Weight W", "params": {"shape": "4,1", "init": "ones"}},
        {"id": "mm", "type": "matmul", "name": "X × W", "params": {}},
        {"id": "b", "type": "learnable_parameter", "name": "Bias b", "params": {"shape": "1", "init": "zeros"}},
        {"id": "add", "type": "tensor_add", "name": "+ Bias", "params": {}},
        {"id": "out", "type": "tensor_output", "name": "Regression Output", "params": {}},
    ]
    edges = [_named("x", "mm", "a"), _named("w", "mm", "b"), _named("mm", "add", "a"), _named("b", "add", "b"), _edge("add", "out")]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime(4))
    y = graph(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert y.shape == (1, 1)
    assert torch.allclose(y, torch.tensor([[10.0]]))


def test_first_principles_logistic_regression_outputs_probability():
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 4}},
        {"id": "w", "type": "learnable_parameter", "name": "Weight W", "params": {"shape": "4,1", "init": "zeros"}},
        {"id": "mm", "type": "matmul", "name": "X × W", "params": {}},
        {"id": "b", "type": "learnable_parameter", "name": "Bias b", "params": {"shape": "1", "init": "zeros"}},
        {"id": "add", "type": "tensor_add", "name": "+ Bias", "params": {}},
        {"id": "sig", "type": "sigmoid", "name": "Sigmoid Probability", "params": {}},
        {"id": "out", "type": "tensor_output", "name": "Probability Output", "params": {}},
    ]
    edges = [_named("x", "mm", "a"), _named("w", "mm", "b"), _named("mm", "add", "a"), _named("b", "add", "b"), _edge("add", "sig"), _edge("sig", "out")]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime(4))
    y = graph(torch.randn(6, 4))
    assert y.shape == (6, 1)
    assert torch.allclose(y, torch.full((6, 1), 0.5))


def test_ann_gallery_shape():
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {}},
        {"id": "l1", "type": "linear", "name": "Dense 8 → 32", "params": {"in_features": 8, "out_features": 32}},
        {"id": "a1", "type": "relu", "name": "ReLU 1", "params": {}},
        {"id": "l2", "type": "linear", "name": "Dense 32 → 16", "params": {"in_features": 32, "out_features": 16}},
        {"id": "a2", "type": "relu", "name": "ReLU 2", "params": {}},
        {"id": "head", "type": "classifier", "name": "3-Class Head", "params": {"dim": 16, "classes": 3}},
    ]
    edges = [_edge("x", "l1"), _edge("l1", "a1"), _edge("a1", "l2"), _edge("l2", "a2"), _edge("a2", "head")]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime(8))
    assert graph(torch.randn(5, 8)).shape == (5, 3)


def test_cnn_gallery_shape():
    nodes = [
        {"id": "x", "type": "image_input", "name": "Image Input", "params": {}},
        {"id": "c1", "type": "conv2d", "name": "Conv2D 1 → 8", "params": {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "padding": 1}},
        {"id": "a1", "type": "relu", "name": "ReLU 1", "params": {}},
        {"id": "p1", "type": "maxpool2d", "name": "MaxPool", "params": {"kernel_size": 2, "stride": 2}},
        {"id": "c2", "type": "conv2d", "name": "Conv2D 8 → 16", "params": {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "padding": 1}},
        {"id": "a2", "type": "relu", "name": "ReLU 2", "params": {}},
        {"id": "p2", "type": "maxpool2d", "name": "MaxPool 2", "params": {"kernel_size": 2, "stride": 2}},
        {"id": "flat", "type": "flatten", "name": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
        {"id": "head", "type": "classifier", "name": "3-Class Head", "params": {"dim": 256, "classes": 3}},
    ]
    ids=[n["id"] for n in nodes]
    edges=[_edge(ids[i],ids[i+1]) for i in range(len(ids)-1)]
    graph=TensorGraph(nodes=nodes,edges=edges,custom_components={},runtime=_runtime())
    assert graph(torch.randn(4,1,16,16)).shape==(4,3)


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru"])
def test_recurrent_gallery_models_shape(kind):
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Sequence Input", "params": {}},
        {"id": "u", "type": "unsqueeze", "name": "Add Feature Dimension", "params": {"dim": -1}},
        {"id": "rec", "type": kind, "name": kind.upper(), "params": {"input_size": 1, "hidden_size": 16, "output": "last"}},
        {"id": "head", "type": "classifier", "name": "Binary Classifier", "params": {"dim": 16, "classes": 2}},
    ]
    edges=[_edge("x","u"),_edge("u","rec"),_edge("rec","head")]
    graph=TensorGraph(nodes=nodes,edges=edges,custom_components={},runtime=_runtime())
    assert graph(torch.randn(3,32)).shape==(3,2)


def test_autoencoder_gallery_restores_image_shape():
    nodes = [
        {"id": "x", "type": "image_input", "name": "Image Input", "params": {}},
        {"id": "flat", "type": "flatten", "name": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
        {"id": "e1", "type": "linear", "name": "Encoder", "params": {"in_features": 256, "out_features": 64}},
        {"id": "a1", "type": "relu", "name": "ReLU", "params": {}},
        {"id": "lat", "type": "linear", "name": "Latent", "params": {"in_features": 64, "out_features": 16}},
        {"id": "a2", "type": "relu", "name": "Latent ReLU", "params": {}},
        {"id": "d1", "type": "linear", "name": "Decoder", "params": {"in_features": 16, "out_features": 64}},
        {"id": "a3", "type": "relu", "name": "Decoder ReLU", "params": {}},
        {"id": "d2", "type": "linear", "name": "Reconstruction", "params": {"in_features": 64, "out_features": 256}},
        {"id": "shape", "type": "reshape", "name": "Restore Image", "params": {"shape": "0,1,16,16"}},
        {"id": "out", "type": "tensor_output", "name": "Reconstructed Image", "params": {}},
    ]
    ids=[n["id"] for n in nodes]
    edges=[_edge(ids[i],ids[i+1]) for i in range(len(ids)-1)]
    graph=TensorGraph(nodes=nodes,edges=edges,custom_components={},runtime=_runtime())
    assert graph(torch.randn(2,1,16,16)).shape==(2,1,16,16)
