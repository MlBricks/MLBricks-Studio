from __future__ import annotations

import torch

from mlb_studio.graph import primitive_catalog
from mlb_studio.model_runtime import TensorGraph


def _runtime(dim=8):
    return {"device": "cpu", "backend": "pytorch", "precision": "fp32", "model_dim": dim, "heads": 1}


def _edge(a, b):
    return {"id": f"{a}-{b}", "source": a, "target": b, "kind": "main", "source_port": "main_out", "target_port": "main_in"}


def _graph(kind, params, dim):
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": dim}},
        {"id": "fit", "type": kind, "name": kind, "params": params},
        {"id": "out", "type": "tensor_output", "name": "Output", "params": {}},
    ]
    return TensorGraph(nodes=nodes, edges=[_edge("x", "fit"), _edge("fit", "out")], custom_components={}, runtime=_runtime(dim))


def test_classical_components_are_public_ml_core_components():
    by_type = {item["type"]: item for item in primitive_catalog()}
    for kind in ("knn_classifier", "decision_tree_classifier", "kmeans", "pca"):
        assert kind in by_type
        assert by_type[kind]["category"] == "ML Core"
        assert by_type[kind]["builder_utility"] is True


def test_knn_fit_predict_and_state_dict_roundtrip():
    graph = _graph("knn_classifier", {"neighbors": 3, "weights": "uniform", "p": 2}, 2)
    model = graph.mods["fit"]
    x = torch.tensor([[-2.0, -2.0], [-1.8, -2.2], [2.0, 2.0], [2.2, 1.8]])
    y = torch.tensor([0, 0, 1, 1])
    model.fit(x, y)
    pred = graph(torch.tensor([[-2.1, -1.9], [2.1, 2.1]]))
    assert pred.tolist() == [0, 1]

    clone = _graph("knn_classifier", {"neighbors": 3, "weights": "uniform", "p": 2}, 2)
    clone.load_state_dict(graph.state_dict(), strict=True)
    assert clone(torch.tensor([[-2.1, -1.9], [2.1, 2.1]])).tolist() == [0, 1]


def test_decision_tree_fit_predict_and_state_dict_roundtrip():
    graph = _graph("decision_tree_classifier", {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}, 2)
    model = graph.mods["fit"]
    x = torch.tensor([[-2.0, 0.0], [-1.0, 0.2], [1.0, 0.0], [2.0, -0.2]])
    y = torch.tensor([0, 0, 1, 1])
    model.fit(x, y)
    assert graph(torch.tensor([[-1.5, 0.0], [1.5, 0.0]])).tolist() == [0, 1]

    clone = _graph("decision_tree_classifier", {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}, 2)
    clone.load_state_dict(graph.state_dict(), strict=True)
    assert clone(torch.tensor([[-1.5, 0.0], [1.5, 0.0]])).tolist() == [0, 1]


def test_kmeans_fits_three_clusters_and_persists_centroids():
    graph = _graph("kmeans", {"clusters": 3, "max_iter": 50, "tolerance": 1e-5, "seed": 7}, 2)
    model = graph.mods["fit"]
    x = torch.tensor([
        [-3.0, -3.0], [-2.8, -3.2],
        [3.0, -3.0], [3.2, -2.8],
        [0.0, 3.0], [0.2, 3.2],
    ])
    model.fit(x)
    labels = graph(x)
    assert len(torch.unique(labels)) == 3
    assert model.centroids_.shape == (3, 2)
    assert torch.isfinite(model.inertia_)

    clone = _graph("kmeans", {"clusters": 3, "max_iter": 50, "tolerance": 1e-5, "seed": 7}, 2)
    clone.load_state_dict(graph.state_dict(), strict=True)
    assert torch.equal(clone(x), labels)


def test_pca_reduces_dimensions_and_state_dict_roundtrip():
    graph = _graph("pca", {"components": 2, "center": True, "whiten": False}, 4)
    model = graph.mods["fit"]
    base = torch.linspace(-2, 2, 20)
    x = torch.stack([base, 2 * base, -base + 0.1, 0.5 * base], dim=1)
    model.fit(x)
    transformed = graph(x)
    assert transformed.shape == (20, 2)
    assert model.components_.shape == (2, 4)
    assert 0.99 <= float(model.explained_variance_ratio_.sum()) <= 1.00001

    clone = _graph("pca", {"components": 2, "center": True, "whiten": False}, 4)
    clone.load_state_dict(graph.state_dict(), strict=True)
    assert torch.allclose(clone(x), transformed)


def _split(payload):
    class Split:
        def __init__(self, data):
            self.data = data
            self.column_names = list(data)
        def __getitem__(self, key):
            return self.data[key]
        def __len__(self):
            return len(next(iter(self.data.values()))) if self.data else 0
    return Split(payload)


def test_train_builder_model_dispatches_knn_fit_and_reports_metrics(monkeypatch, tmp_path):
    import threading
    import mlb_studio.model_runtime as runtime

    graph = _graph("knn_classifier", {"neighbors": 3, "weights": "uniform", "p": 2}, 2)
    compiled = runtime.CompiledModel(
        model=graph, raw_model=graph, training_model=None, device=torch.device("cpu"),
        precision="fp32", vocab_size=0, parameter_count=0, compile_used=False, compile_error=None,
    )
    monkeypatch.setattr(runtime, "compile_builder_model", lambda *args, **kwargs: (compiled, None))

    saved = {}
    def fake_save(model, path, metadata=None):
        saved["path"] = str(path)
        saved["metadata"] = metadata
        saved["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(runtime.IMPORT_POOL, "resolve_api", lambda key: fake_save if key == "lifecycle.save" else None)

    architecture = {
        "nodes": [
            {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 2}},
            {"id": "fit", "type": "knn_classifier", "name": "KNN", "params": {"neighbors": 3, "weights": "uniform", "p": 2}},
            {"id": "out", "type": "tensor_output", "name": "Output", "params": {}},
        ],
        "edges": [_edge("x", "fit"), _edge("fit", "out")],
    }
    entry = {"name": "KNN Demo", "architecture": architecture}
    train = _split({"feature_1": [-2.0, -1.8, 2.0, 2.2], "feature_2": [-2.0, -2.2, 2.0, 1.8], "label": [0, 0, 1, 1]})
    val = _split({"feature_1": [-2.1, 2.1], "feature_2": [-1.9, 2.1], "label": [0, 1]})
    events = []
    result = runtime.train_builder_model(
        state={"project": {}, "custom_components": {}}, model_entry=entry,
        dataset={"train": train, "validation": val}, dataset_meta={"name": "demo"},
        config={"output_dir": str(tmp_path), "validation_split": "validation"},
        progress=events.append, stop_event=threading.Event(),
    )
    update = result["model_update"]
    assert update["training_mode"] == "classical_fit"
    assert update["fit_algorithm"] == "knn_classifier"
    assert update["fit_metrics"]["train_accuracy"] == 1.0
    assert update["fit_metrics"]["validation_accuracy"] == 1.0
    assert saved["metadata"]["training_mode"] == "classical_fit"
    assert "mods.fit.train_x" in saved["state"]
    assert events[-1]["status"] == "done"


def test_train_builder_model_dispatches_pca_fit(monkeypatch, tmp_path):
    import threading
    import mlb_studio.model_runtime as runtime

    graph = _graph("pca", {"components": 2, "center": True, "whiten": False}, 4)
    compiled = runtime.CompiledModel(
        model=graph, raw_model=graph, training_model=None, device=torch.device("cpu"),
        precision="fp32", vocab_size=0, parameter_count=0, compile_used=False, compile_error=None,
    )
    monkeypatch.setattr(runtime, "compile_builder_model", lambda *args, **kwargs: (compiled, None))
    monkeypatch.setattr(runtime.IMPORT_POOL, "resolve_api", lambda key: (lambda *args, **kwargs: None) if key == "lifecycle.save" else None)

    values = list(torch.linspace(-2, 2, 20).tolist())
    train = _split({
        "feature_1": values,
        "feature_2": [2*v for v in values],
        "feature_3": [-v + 0.1 for v in values],
        "feature_4": [0.5*v for v in values],
        "label": [0] * 20,
    })
    architecture = {
        "nodes": [
            {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 4}},
            {"id": "fit", "type": "pca", "name": "PCA", "params": {"components": 2, "center": True, "whiten": False}},
            {"id": "out", "type": "tensor_output", "name": "Output", "params": {}},
        ],
        "edges": [_edge("x", "fit"), _edge("fit", "out")],
    }
    result = runtime.train_builder_model(
        state={"project": {}, "custom_components": {}}, model_entry={"name": "PCA Demo", "architecture": architecture},
        dataset={"train": train}, dataset_meta={"name": "demo"},
        config={"output_dir": str(tmp_path)}, progress=lambda event: None, stop_event=threading.Event(),
    )
    assert result["model_update"]["fit_algorithm"] == "pca"
    assert result["model_update"]["fit_metrics"]["explained_variance_total"] > 0.99
    assert result["model_update"]["feature_columns"] == ["feature_1", "feature_2", "feature_3", "feature_4"]
