from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest
import torch

import mlb_studio.model_runtime as runtime


class Split:
    def __init__(self, data):
        self.data = data
        self.column_names = list(data)
    def __getitem__(self, key):
        return self.data[key]
    def __len__(self):
        return len(next(iter(self.data.values()))) if self.data else 0


def edge(a, b):
    return {"id": f"{a}-{b}", "source": a, "target": b, "kind": "main", "source_port": "main_out", "target_port": "main_in"}


def entry(name, nodes, edges, *, modality, task):
    return {
        "name": name,
        "architecture": {"nodes": nodes, "edges": edges},
        "requirements": {
            "modality": modality,
            "training_mode": "supervised",
            "training_task": task,
            "requires_tokenizer": False,
        },
    }


def config(tmp_path, **overrides):
    cfg = {
        "seed": 7,
        "batch_size": 8,
        "gradient_accumulation": 1,
        "budget_type": "steps",
        "max_steps": 3,
        "epochs": 1,
        "optimizer": "sgd",
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "warmup_steps": 0,
        "validation_split": "validation",
        "validate_every": 1,
        "validation_steps": 1,
        "checkpoint_every": 0,
        "device": "cpu",
        "backend": "pytorch",
        "execution_mode": "eager",
        "precision": "fp32",
        "output_dir": str(tmp_path),
    }
    cfg.update(overrides)
    return cfg


def patch_save(monkeypatch):
    original = runtime.IMPORT_POOL.resolve_api
    saved = {}

    def fake_save(model, path, metadata=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.pt").write_bytes(b"test")
        saved["path"] = path
        saved["metadata"] = metadata or {}
        saved["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def resolve(key):
        if key == "lifecycle.save":
            return fake_save
        return original(key)

    monkeypatch.setattr(runtime.IMPORT_POOL, "resolve_api", resolve)
    return saved


def test_compile_supervised_graph_never_loads_tokenizer(monkeypatch):
    monkeypatch.setattr(runtime, "_tokenizer_for", lambda *a, **k: (_ for _ in ()).throw(AssertionError("tokenizer should not load")))
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 1}},
        {"id": "n", "type": "linear", "name": "Neuron", "params": {"in_features": 1, "out_features": 1}},
        {"id": "o", "type": "tensor_output", "name": "Output", "params": {}},
    ]
    model_entry = entry("Single Neuron", nodes, [edge("x", "n"), edge("n", "o")], modality="signal", task="regression")
    compiled, tokenizer = runtime.compile_builder_model(
        {"project": {}, "custom_components": {}}, model_entry, {},
        {"device": "cpu", "backend": "pytorch", "precision": "fp32", "execution_mode": "eager"},
        for_training=True,
    )
    assert tokenizer is None
    assert compiled.training_model is None
    assert compiled.raw_model(torch.randn(4, 1)).shape == (4, 1)


def test_single_neuron_supervised_regression_trains_and_saves(monkeypatch, tmp_path):
    saved = patch_save(monkeypatch)
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 1}},
        {"id": "n", "type": "linear", "name": "Single Dense Neuron", "params": {"in_features": 1, "out_features": 1}},
        {"id": "o", "type": "tensor_output", "name": "Neuron Output", "params": {}},
    ]
    e = entry("Single Neuron", nodes, [edge("x", "n"), edge("n", "o")], modality="signal", task="regression")
    xs = [i / 10 for i in range(-20, 20)]
    ys = [1.5 * x + 0.3 for x in xs]
    ds = {"train": Split({"feature_1": xs, "target": ys}), "validation": Split({"feature_1": xs[:8], "target": ys[:8]})}
    events = []
    result = runtime.train_builder_model(
        state={"project": {"task": "Regression"}, "custom_components": {}}, model_entry=e,
        dataset=ds, dataset_meta={"name": "Neuron Regression Demo"}, config=config(tmp_path, max_steps=8),
        progress=events.append, stop_event=threading.Event(),
    )
    update = result["model_update"]
    assert result["tokenizer"] is None
    assert update["training_mode"] == "supervised"
    assert update["training_task"] == "regression"
    assert update["samples_seen"] > 0
    assert math.isfinite(update["last_loss"])
    assert saved["metadata"]["training_mode"] == "supervised"
    assert any(ev.get("samples_per_sec") for ev in events if ev.get("phase") == "train")


def test_ann_supervised_classification_reports_accuracy(monkeypatch, tmp_path):
    patch_save(monkeypatch)
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Feature Input", "params": {"feature_dim": 2}},
        {"id": "l1", "type": "linear", "name": "Dense", "params": {"in_features": 2, "out_features": 8}},
        {"id": "a", "type": "relu", "name": "ReLU", "params": {}},
        {"id": "h", "type": "classifier", "name": "2-Class Head", "params": {"dim": 8, "classes": 2}},
    ]
    e = entry("ANN", nodes, [edge("x", "l1"), edge("l1", "a"), edge("a", "h")], modality="signal", task="classification")
    f1 = [-2.0, -1.8, -1.5, -1.2, 1.2, 1.5, 1.8, 2.0] * 4
    f2 = [-1.0, -1.2, -0.8, -1.1, 1.1, 0.8, 1.2, 1.0] * 4
    labels = [0, 0, 0, 0, 1, 1, 1, 1] * 4
    ds = {"train": Split({"feature_1": f1, "feature_2": f2, "label": labels}), "validation": Split({"feature_1": f1[:8], "feature_2": f2[:8], "label": labels[:8]})}
    result = runtime.train_builder_model(
        state={"project": {"task": "Tabular classification"}, "custom_components": {}}, model_entry=e,
        dataset=ds, dataset_meta={"name": "Tabular Classification Demo"}, config=config(tmp_path, max_steps=12, learning_rate=0.05),
        progress=lambda event: None, stop_event=threading.Event(),
    )
    metrics = result["model_update"]["supervised_metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["validation_accuracy"] <= 1.0


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru"])
def test_sequence_models_use_generic_supervised_training(monkeypatch, tmp_path, kind):
    patch_save(monkeypatch)
    nodes = [
        {"id": "x", "type": "feature_input", "name": "Sequence Input", "params": {"feature_dim": 8}},
        {"id": "u", "type": "unsqueeze", "name": "Add Feature Dimension", "params": {"dim": -1}},
        {"id": "r", "type": kind, "name": kind.upper(), "params": {"input_size": 1, "hidden_size": 4, "num_layers": 1, "output": "last"}},
        {"id": "h", "type": "classifier", "name": "Binary Classifier", "params": {"dim": 4, "classes": 2}},
    ]
    e = entry(kind.upper(), nodes, [edge("x", "u"), edge("u", "r"), edge("r", "h")], modality="signal", task="classification")
    seq0 = [[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4] for _ in range(8)]
    seq1 = [[1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4] for _ in range(8)]
    seqs = seq0 + seq1
    labels = [0] * 8 + [1] * 8
    ds = {"train": Split({"sequence": seqs, "label": labels}), "validation": Split({"sequence": seqs[:8] + seqs[-8:], "label": labels[:8] + labels[-8:]})}
    result = runtime.train_builder_model(
        state={"project": {"task": "Sequence classification"}, "custom_components": {}}, model_entry=e,
        dataset=ds, dataset_meta={"name": "Sequence Classification Demo"}, config=config(tmp_path / kind, max_steps=2, batch_size=4),
        progress=lambda event: None, stop_event=threading.Event(),
    )
    assert result["model_update"]["training_mode"] == "supervised"
    assert result["model_update"]["training_task"] == "classification"


def test_cnn_and_autoencoder_image_tensor_contracts():
    image_split = Split({"image": [[[0.0] * 16 for _ in range(16)] for _ in range(3)], "label": [0, 1, 2]})
    cnn_info = {"input_type": "image_input", "task": "classification"}
    x, y, fields = runtime._supervised_xy(image_split, cnn_info)
    assert x.shape == (3, 1, 16, 16)
    assert y.shape == (3,)
    assert fields == ["image"]

    recon = [[[float((i + j) % 2) for j in range(16)] for i in range(16)] for _ in range(3)]
    ae_split = Split({"image": recon, "target_image": recon})
    ae_info = {"input_type": "image_input", "task": "reconstruction"}
    x2, y2, _ = runtime._supervised_xy(ae_split, ae_info)
    assert x2.shape == (3, 1, 16, 16)
    assert y2.shape == (3, 1, 16, 16)


def test_step3_ui_marks_non_text_gradient_graphs_as_supervised():
    root = Path(__file__).resolve().parents[1]
    js = (root / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    react = (root / "frontend" / "src" / "react-runtime.js").read_text(encoding="utf-8")
    assert 'training_mode:fitAlgorithm?"classical_fit":(isAudioGeneration?"audio_generation":(modality==="text"?"gradient":"supervised"))' in js
    assert 'const budgetOptions=supervised?["steps","samples","epochs"]:["steps","tokens","epochs"]' in js
    assert 'runtimeField("Sample Budget"' in js
    assert "SupervisedTrainingMetrics" in react
    assert "Samples/s" in react
    assert "Val Accuracy" in react


def test_cnn_gallery_can_train_end_to_end(monkeypatch, tmp_path):
    patch_save(monkeypatch)
    nodes = [
        {"id":"x","type":"image_input","name":"Image Input","params":{}},
        {"id":"c","type":"conv2d","name":"Conv","params":{"in_channels":1,"out_channels":2,"kernel_size":3,"padding":1}},
        {"id":"a","type":"relu","name":"ReLU","params":{}},
        {"id":"p","type":"maxpool2d","name":"Pool","params":{"kernel_size":2,"stride":2}},
        {"id":"f","type":"flatten","name":"Flatten","params":{"start_dim":1,"end_dim":-1}},
        {"id":"h","type":"classifier","name":"3-Class Head","params":{"dim":128,"classes":3}},
    ]
    e=entry("CNN",nodes,[edge("x","c"),edge("c","a"),edge("a","p"),edge("p","f"),edge("f","h")],modality="image",task="classification")
    images=[]; labels=[]
    for label in range(3):
        for _ in range(4):
            img=[[0.0]*16 for _ in range(16)]
            if label==0:
                for y in range(16): img[y][7]=1.0
            elif label==1:
                for x in range(16): img[7][x]=1.0
            else:
                for i in range(16): img[i][i]=1.0
            images.append(img); labels.append(label)
    ds={"train":Split({"image":images,"label":labels}),"validation":Split({"image":images,"label":labels})}
    result=runtime.train_builder_model(
        state={"project":{"task":"Image classification"},"custom_components":{}},model_entry=e,dataset=ds,dataset_meta={"name":"Image Classification Demo"},
        config=config(tmp_path,max_steps=1,batch_size=6),progress=lambda event:None,stop_event=threading.Event())
    assert result["model_update"]["training_task"]=="classification"
    assert math.isfinite(result["model_update"]["last_loss"])


def test_autoencoder_gallery_can_train_end_to_end(monkeypatch, tmp_path):
    patch_save(monkeypatch)
    nodes=[
        {"id":"x","type":"image_input","name":"Image Input","params":{}},
        {"id":"f","type":"flatten","name":"Flatten","params":{"start_dim":1,"end_dim":-1}},
        {"id":"e","type":"linear","name":"Encoder","params":{"in_features":256,"out_features":8}},
        {"id":"a","type":"relu","name":"ReLU","params":{}},
        {"id":"d","type":"linear","name":"Decoder","params":{"in_features":8,"out_features":256}},
        {"id":"r","type":"reshape","name":"Restore","params":{"shape":"0,1,16,16"}},
        {"id":"o","type":"tensor_output","name":"Reconstruction","params":{}},
    ]
    e=entry("Autoencoder",nodes,[edge("x","f"),edge("f","e"),edge("e","a"),edge("a","d"),edge("d","r"),edge("r","o")],modality="image",task="reconstruction")
    images=[[[float((i+j+k)%2) for j in range(16)] for i in range(16)] for k in range(8)]
    ds={"train":Split({"image":images,"target_image":images}),"validation":Split({"image":images,"target_image":images})}
    result=runtime.train_builder_model(
        state={"project":{"task":"Image reconstruction"},"custom_components":{}},model_entry=e,dataset=ds,dataset_meta={"name":"Image Reconstruction Demo"},
        config=config(tmp_path,max_steps=1,batch_size=4),progress=lambda event:None,stop_event=threading.Event())
    assert result["model_update"]["training_task"]=="reconstruction"
    assert math.isfinite(result["model_update"]["last_loss"])
