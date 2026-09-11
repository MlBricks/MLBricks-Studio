from __future__ import annotations

from pathlib import Path

import torch

from mlb_studio.model_runtime import CompiledModel, TensorGraph, run_universal_inference
from mlb_studio.universal_io import load_single_input, normalize_input_config

ROOT = Path(__file__).resolve().parents[1]


def _runtime(dim=8):
    return {"device": "cpu", "backend": "pytorch", "precision": "fp32", "model_dim": dim, "heads": 1}


def test_classical_model_contract_forces_old_signal_config_to_tabular():
    env = normalize_input_config(
        {
            "input_kind": "signal",  # stale pre-hotfix browser draft
            "input_mode": "static",
            "task_type": "analyze",
            "input_data": "1,2,3,4,5,6,7,8",
            "prompt": "Once upon a time",
        },
        {
            "requirements": {
                "modality": "signal",
                "training_mode": "classical_fit",
                "feature_dim": 8,
            }
        },
    )
    assert env.kind == "tabular"
    value, meta = load_single_input(env)
    assert value.shape == (1, 8)
    assert meta == {"rows": 1, "features": 8}


def test_tabular_inline_json_batch_stays_2d():
    env = normalize_input_config(
        {
            "input_kind": "tabular",
            "input_mode": "batch",
            "input_data": "[[1,2,3],[4,5,6]]",
        }
    )
    value, meta = load_single_input(env)
    assert value.shape == (2, 3)
    assert meta == {"rows": 2, "features": 3}


def test_fitted_knn_accepts_tabular_row_without_signal_dimension():
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 8}},
        {"id": "fit", "type": "knn_classifier", "name": "KNN", "params": {"neighbors": 1}},
        {"id": "out", "type": "tensor_output", "name": "Predicted Class", "params": {}},
    ]
    edges = [
        {"id": "a", "source": "x", "target": "fit", "kind": "main", "source_port": "main_out", "target_port": "main_in"},
        {"id": "b", "source": "fit", "target": "out", "kind": "main", "source_port": "main_out", "target_port": "main_in"},
    ]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime())
    graph.mods["fit"].fit(
        torch.tensor([[0.0] * 8, [1.0] * 8]),
        torch.tensor([0, 1]),
    )
    compiled = CompiledModel(
        model=graph,
        raw_model=graph,
        training_model=None,
        device=torch.device("cpu"),
        precision="fp32",
        vocab_size=0,
        parameter_count=0,
        compile_used=False,
        compile_error=None,
    )
    env = normalize_input_config({"input_kind": "tabular", "input_data": "1,1,1,1,1,1,1,1"})
    value, _ = load_single_input(env)
    output = run_universal_inference(
        compiled,
        value,
        input_kind="tabular",
        output_type="tensor_output",
        task="classify",
    )
    assert output["data"] in (1, [1])



def _fit_graph(kind, feature_dim, params, x, y=None):
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": feature_dim}},
        {"id": "fit", "type": kind, "name": kind, "params": params},
        {"id": "out", "type": "tensor_output", "name": "Output", "params": {}},
    ]
    edges = [
        {"id": "a", "source": "x", "target": "fit", "kind": "main", "source_port": "main_out", "target_port": "main_in"},
        {"id": "b", "source": "fit", "target": "out", "kind": "main", "source_port": "main_out", "target_port": "main_in"},
    ]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime(feature_dim))
    graph.mods["fit"].fit(x, y) if y is not None else graph.mods["fit"].fit(x)
    return graph


def test_all_classical_gallery_algorithms_accept_feature_rows():
    # Decision Tree: 8-feature classification row.
    x8 = torch.tensor([[0.0] * 8, [1.0] * 8, [0.1] * 8, [0.9] * 8])
    y = torch.tensor([0, 1, 0, 1])
    tree = _fit_graph("decision_tree_classifier", 8, {"max_depth": 3}, x8, y)
    tree_out = tree(torch.ones(1, 8))
    assert tree_out.shape in (torch.Size([]), torch.Size([1]))

    # K-Means: 2-feature clustering row.
    clusters = torch.tensor([[0.0, 0.0], [0.1, 0.1], [3.0, 3.0], [3.1, 3.1], [-3.0, 3.0], [-3.1, 3.1]])
    kmeans = _fit_graph("kmeans", 2, {"clusters": 3, "max_iter": 20, "seed": 1}, clusters)
    km_out = kmeans(torch.tensor([[3.0, 3.0]]))
    assert km_out.shape in (torch.Size([]), torch.Size([1]))

    # PCA: 16-feature row transforms to configured 2 principal components.
    torch.manual_seed(7)
    pca_x = torch.randn(24, 16)
    pca = _fit_graph("pca", 16, {"components": 2, "center": True, "whiten": False}, pca_x)
    pca_out = pca(torch.randn(1, 16))
    assert pca_out.shape == (1, 2)

def test_frontend_classical_ml_runtime_is_numeric_not_prompt_based():
    legacy = (ROOT / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    react = (ROOT / "frontend" / "src" / "react-runtime.js").read_text(encoding="utf-8")
    built = (ROOT / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    for text in (legacy, built):
        assert '{value:"tabular",label:"Tabular / Numeric Features"}' in text
        assert 'runtimeField("Feature Values","textarea",config.input_data' in text
        assert 'return "tabular"' in text
        assert 'start:"Predict Class"' in text
        assert 'start:"Assign Cluster"' in text
        assert 'start:"Transform Features"' in text
    assert "FEATURE VALUES" in react
    assert "title:x.isText?'Generated Output':'Model Output'" in react
