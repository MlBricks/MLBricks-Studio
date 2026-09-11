from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest
import torch

import mlb_studio.model_runtime as runtime
from mlb_studio.graph import primitive_catalog
from mlb_studio.runner import EXECUTABLE_TYPES
from mlb_studio import data as data_api


class Split:
    def __init__(self, data):
        self.data = data
        self.column_names = list(data)
    def __getitem__(self, key):
        if isinstance(key, int):
            return {name: values[key] for name, values in self.data.items()}
        return self.data[key]
    def __len__(self):
        return len(next(iter(self.data.values()))) if self.data else 0
    def map(self, fn):
        rows = [fn(dict(self[i])) for i in range(len(self))]
        keys = list(rows[0]) if rows else list(self.data)
        return Split({key: [row.get(key) for row in rows] for key in keys})


class FakeDatasetFactory:
    @staticmethod
    def from_dict(data):
        return Split(data)


class FakeDatasets:
    Dataset = FakeDatasetFactory


def edge(a, b):
    return {"id": f"{a}-{b}", "source": a, "target": b, "kind": "main", "source_port": "main_out", "target_port": "main_in"}


def entry(name, nodes, edges, *, task):
    return {
        "name": name,
        "task": task,
        "architecture": {"nodes": nodes, "edges": edges},
        "requirements": {"modality": "signal", "training_mode": "supervised", "training_task": task, "requires_tokenizer": False},
    }


def config(tmp_path, **overrides):
    cfg = {
        "seed": 7, "batch_size": 4, "gradient_accumulation": 1,
        "budget_type": "steps", "max_steps": 1, "epochs": 1,
        "optimizer": "sgd", "learning_rate": 0.02, "weight_decay": 0.0,
        "beta1": 0.9, "beta2": 0.95, "warmup_steps": 0,
        "validation_split": "validation", "validate_every": 1, "validation_steps": 1,
        "checkpoint_every": 0, "device": "cpu", "backend": "pytorch",
        "execution_mode": "eager", "precision": "fp32", "output_dir": str(tmp_path),
    }
    cfg.update(overrides)
    return cfg


def patch_save(monkeypatch):
    original = runtime.IMPORT_POOL.resolve_api
    def fake_save(model, path, metadata=None):
        path = Path(path); path.mkdir(parents=True, exist_ok=True); (path / "model.pt").write_bytes(b"test")
    def resolve(key):
        if key == "lifecycle.save": return fake_save
        return original(key)
    monkeypatch.setattr(runtime.IMPORT_POOL, "resolve_api", resolve)


def test_signal_public_components_and_data_runtime_are_registered():
    by_type = {item["type"]: item for item in primitive_catalog()}
    assert "signal_process" in by_type
    assert "signal_fft" in by_type
    assert by_type["signal_process"]["category"] == "Signal"
    assert by_type["signal_fft"]["category"] == "Signal"
    assert "signal_process" in EXECUTABLE_TYPES


def test_signal_schema_mapper_stacks_multichannel_and_maps_targets():
    ds = Split({
        "a": [[1.0, 2.0], [3.0, 4.0]],
        "b": [[5.0, 6.0], [7.0, 8.0]],
        "future": [[0.1, 0.2], [0.3, 0.4]],
        "label": [0, 1],
    })
    out = data_api.process_signal_dataset(ds, signal_columns="a,b", target_column="future", normalize=False)
    assert out.column_names.count("signal") == 1
    assert out.column_names.count("target") == 1
    assert out[0]["signal"] == [[1.0, 2.0], [5.0, 6.0]]
    assert out[1]["target"] == pytest.approx([0.3, 0.4])
    assert out[1]["label"] == 1


def test_signal_demo_contracts_cover_forecast_spectral_denoise_fusion_rf_and_long(monkeypatch):
    monkeypatch.setattr(data_api, "_datasets", lambda: FakeDatasets)
    forecast = data_api.generate_demo_dataset("timeseries_forecast", samples=4, sequence_length=32)
    assert len(forecast[0]["context"]) == 32 and len(forecast[0]["target"]) == 8
    spectral = data_api.generate_demo_dataset("spectral_signal", samples=4, sequence_length=32)
    assert len(spectral[0]["spectrum"]) == 16 and spectral[0]["target"] == spectral[0]["spectrum"]
    denoise = data_api.generate_demo_dataset("signal_denoise", samples=4, sequence_length=32)
    assert len(denoise[0]["noisy_signal"]) == len(denoise[0]["clean_signal"]) == 64
    fusion = data_api.generate_demo_dataset("sensor_fusion", samples=4, sequence_length=32)
    assert all(len(fusion[0][k]) == 32 for k in ("sensor_a", "sensor_b", "sensor_c"))
    rf = data_api.generate_demo_dataset("rf_iq", samples=4, sequence_length=32)
    assert len(rf[0]["i"]) == len(rf[0]["q"]) == 64
    long_ds = data_api.generate_demo_dataset("long_signal", samples=4, sequence_length=32)
    assert len(long_ds[0]["signal"]) == 256 and len(long_ds[0]["target"]) == 16


def test_fft_magnitude_component_compiles_and_is_differentiable():
    nodes = [
        {"id":"x","type":"signal_input","name":"Signal","params":{}},
        {"id":"f","type":"signal_fft","name":"FFT","params":{"bins":8,"log_scale":True,"remove_dc":False,"flatten_channels":True}},
        {"id":"o","type":"tensor_output","name":"Out","params":{}},
    ]
    e = entry("Spectral", nodes, [edge("x","f"), edge("f","o")], task="regression")
    compiled, _ = runtime.compile_builder_model({"project":{},"custom_components":{}}, e, {}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"}, for_training=True)
    x = torch.randn(3, 64, requires_grad=True)
    y = compiled.raw_model(x)
    assert y.shape == (3, 8)
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_time_series_predictor_trains_through_generic_supervised_runtime(monkeypatch, tmp_path):
    patch_save(monkeypatch)
    nodes = [
        {"id":"x","type":"signal_input","name":"Signal","params":{}},
        {"id":"u","type":"unsqueeze","name":"Channel","params":{"dim":1}},
        {"id":"c","type":"conv1d","name":"Conv","params":{"in_channels":1,"out_channels":8,"kernel_size":3,"padding":1}},
        {"id":"a","type":"gelu","name":"GELU","params":{}},
        {"id":"p","type":"adaptive_avgpool1d","name":"Pool","params":{"output_size":1}},
        {"id":"f","type":"flatten","name":"Flatten","params":{"start_dim":1,"end_dim":-1}},
        {"id":"h","type":"linear","name":"Forecast","params":{"in_features":8,"out_features":4}},
        {"id":"o","type":"tensor_output","name":"Out","params":{}},
    ]
    e = entry("Time-Series Predictor", nodes, [edge(nodes[i]["id"], nodes[i+1]["id"]) for i in range(len(nodes)-1)], task="Forecasting")
    xs = [[math.sin((j+i)*0.2) for j in range(16)] for i in range(16)]
    ys = [[math.sin((16+j+i)*0.2) for j in range(4)] for i in range(16)]
    ds = {"train": Split({"signal": xs, "target": ys}), "validation": Split({"signal": xs[:8], "target": ys[:8]})}
    result = runtime.train_builder_model(state={"project":{"task":"Forecasting"},"custom_components":{}}, model_entry=e, dataset=ds, dataset_meta={"name":"Forecast"}, config=config(tmp_path), progress=lambda event: None, stop_event=threading.Event())
    assert result["model_update"]["training_task"] == "regression"
    assert math.isfinite(result["model_update"]["last_loss"])


def test_multichannel_signal_classifier_supports_sensor_fusion_and_rf_shapes():
    for channels in (2, 3):
        nodes = [
            {"id":"x","type":"signal_input","name":"Signal","params":{}},
            {"id":"c","type":"conv1d","name":"Conv","params":{"in_channels":channels,"out_channels":8,"kernel_size":3,"padding":1}},
            {"id":"p","type":"adaptive_avgpool1d","name":"Pool","params":{"output_size":1}},
            {"id":"f","type":"flatten","name":"Flatten","params":{"start_dim":1,"end_dim":-1}},
            {"id":"h","type":"classifier","name":"Head","params":{"dim":8,"classes":3}},
        ]
        e = entry("Signal Classifier", nodes, [edge(nodes[i]["id"], nodes[i+1]["id"]) for i in range(len(nodes)-1)], task="classification")
        compiled, _ = runtime.compile_builder_model({"project":{},"custom_components":{}}, e, {}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"}, for_training=True)
        out = compiled.raw_model(torch.randn(4, channels, 32))
        assert out.shape == (4, 3)


def test_step6_gallery_contains_all_signal_models_and_visible_signal_data_mapping():
    root = Path(__file__).resolve().parents[1]
    js = (root / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    for name in [
        "Time-Series Predictor", "Signal Classifier", "Anomaly Detector", "Signal Denoiser",
        "Sensor Fusion", "Spectral Model", "RF/IQ Model", "SOUP Signal",
    ]:
        assert f'name:"{name}"' in js
    assert 'const sig=makeNode(cat(catalog,"signal_process"))' in js
    assert 'sensor_a,sensor_b,sensor_c' in js
    assert 'columns:"i,q"' in js
    assert 'add("signal_fft","FFT Magnitude · 16 Bins"' in js
    assert 'add("soup","SOUP Signal State"' in js
