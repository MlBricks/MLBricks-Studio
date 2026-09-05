from __future__ import annotations

from pathlib import Path

from mlb_studio import Builder


def _builder_js() -> str:
    return (Path(__file__).resolve().parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")


def test_training_ui_uses_device_neutral_token_rate_and_cleanup_button():
    js = _builder_js()
    assert 'statusMetric("Tok/s"' in js
    assert 'GPU Tok/s' not in js
    assert 'btn("Clean GPU VRAM","mlb-vram-clean-btn")' in js
    assert 'requestMaintenanceCommand("runtime_clear_memory"' in js


def test_repository_inspectors_expose_delete_actions():
    js = _builder_js()
    assert 'btn("Delete Dataset","mlb-danger-btn")' in js
    assert 'btn("Delete Model","mlb-danger-btn")' in js
    assert 'requestMaintenanceCommand("delete_dataset"' in js
    assert 'requestMaintenanceCommand("delete_model"' in js


def test_delete_prepared_dataset_clears_registry_memory_and_model_references(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    builder = Builder()
    dataset_id = "dataset_test"
    builder.state["prepared_datasets"] = [{"id": dataset_id, "name": "Prepared Test"}]
    builder.prepared_datasets[dataset_id] = object()
    builder.state["components"] = {
        "model": {
            "nodes": [
                {
                    "type": "text_input",
                    "params": {
                        "input_mode": "prepared_dataset",
                        "dataset_id": dataset_id,
                        "dataset_split": "train",
                    },
                }
            ]
        }
    }
    builder.state["model_outputs"] = [
        {"id": "model_a", "selected_dataset_id": dataset_id, "dataset": "Prepared Test"}
    ]
    builder.state["project"] = {"dataset": "Prepared Test"}

    result = builder.delete_prepared_dataset(dataset_id)

    assert result["dataset_id"] == dataset_id
    assert builder.state["prepared_datasets"] == []
    assert dataset_id not in builder.prepared_datasets
    params = builder.state["components"]["model"]["nodes"][0]["params"]
    assert params["input_mode"] == "prompt"
    assert params["dataset_id"] == ""
    assert builder.state["model_outputs"][0]["selected_dataset_id"] is None
    assert builder.state["project"]["dataset"] is None


def test_delete_model_stops_server_and_releases_cached_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    builder = Builder()
    model_id = "model_test"
    builder.state["model_outputs"] = [{"id": model_id, "name": "Test Model"}]
    builder.trained_models[model_id] = {"compiled": object(), "tokenizer": object()}

    class Server:
        stopped = False

        def stop(self):
            self.stopped = True

    server = Server()
    builder._model_servers[model_id] = server

    result = builder.delete_model_output(model_id)

    assert result["model_id"] == model_id
    assert result["stopped_server"] is True
    assert server.stopped is True
    assert model_id not in builder.trained_models
    assert model_id not in builder._model_servers
    assert builder.state["model_outputs"] == []
