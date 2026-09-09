from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mlb_studio.builder import Builder


class Dummy:
    def __init__(self, value=""):
        self.value = value


def test_run_error_does_not_keep_original_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    try:
        marker = object()
        raise AssertionError(f"boom {id(marker)}")
    except AssertionError as exc:
        assert exc.__traceback__ is not None
        remembered = builder._remember_run_error(exc)
        assert remembered is builder.last_run_error
        assert remembered is not exc
        assert isinstance(remembered, RuntimeError)
        assert exc.__traceback__ is None
        assert "AssertionError: boom" in str(remembered)


def test_delete_refuses_active_training_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    builder.state["model_outputs"] = [{"id": "model-1", "name": "Model 1"}]
    builder._active_bridge_action = "train"
    builder._active_bridge_model_id = "model-1"

    with pytest.raises(RuntimeError, match="Stop the active training/generation"):
        builder.delete_model_output("model-1")

    assert builder.state["model_outputs"][0]["id"] == "model-1"


def test_atomic_train_request_keeps_state_and_command_together(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    state = builder.to_dict()
    state["project"]["name"] = "Atomic Retrain"
    widgets = {"state": Dummy("{}"), "command": Dummy("{}"), "progress": Dummy("")}
    builder._bridge_widgets = widgets
    observed = []

    def fake_start():
        observed.append((json.loads(widgets["state"].value), json.loads(widgets["command"].value)))

    builder._start_bridge_run = fake_start
    ok = builder._dispatch_bridge_request_envelope(json.dumps({
        "request_id": "train:test",
        "state": state,
        "command": {"action": "train", "model_id": "model-1"},
    }))

    assert ok is True
    assert observed[0][0]["project"]["name"] == "Atomic Retrain"
    assert observed[0][1] == {"action": "train", "model_id": "model-1"}


def test_compiled_retrain_retries_once_after_assertion(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    entry = {
        "id": "model-1",
        "name": "Model 1",
        "selected_dataset_id": "data-1",
        "training_config": {"execution_mode": "compiled"},
    }
    builder.state["model_outputs"] = [entry]
    builder.state["prepared_datasets"] = [{"id": "data-1", "name": "Data 1"}]
    builder.prepared_datasets["data-1"] = {"train": [1, 2, 3]}

    import mlb_studio.model_runtime as runtime

    calls = []
    fake_compiled = SimpleNamespace(compile_used=True)

    def fake_train_builder_model(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AssertionError("stale compile cache")
        return {
            "model_update": {"training_status": "trained", "weights_ready": True},
            "compiled": fake_compiled,
            "tokenizer": object(),
            "last_sample": None,
        }

    monkeypatch.setattr(runtime, "train_builder_model", fake_train_builder_model)
    monkeypatch.setattr(builder, "_cleanup_failed_runtime", lambda **kwargs: None)
    monkeypatch.setattr(builder, "_reset_torch_compiler", lambda torch: True)
    events = []

    result = builder.train_model("model-1", progress_callback=events.append)

    assert result["training_status"] == "trained"
    assert len(calls) == 2
    assert any(event.get("phase") == "compile_retry" for event in events)
    assert builder.trained_models["model-1"]["compiled"] is fake_compiled
