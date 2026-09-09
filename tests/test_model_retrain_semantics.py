from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlb_studio.builder import Builder


def _ready_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    builder = Builder()
    output_root = tmp_path / "models"
    existing = output_root / "My-Model"
    (existing / "last").mkdir(parents=True)
    (existing / "last" / "model.pt").write_bytes(b"healthy-old-weights")
    (existing / "keep-old.txt").write_text("old", encoding="utf-8")
    builder.state["prepared_datasets"] = [{"id": "d1", "name": "Prepared"}]
    builder.prepared_datasets["d1"] = {"train": [{"input_ids": [1, 2, 3]}]}
    builder.state["model_outputs"] = [{
        "id": "m1", "name": "My Model", "selected_dataset_id": "d1",
        "training_config": {"output_dir": str(output_root), "execution_mode": "eager"},
    }]
    info = {
        "present": True, "resumable": True, "path": str(existing / "last"),
        "kind": "trained_model", "resume_mode": "weights", "metadata": {}, "reason": None,
    }
    monkeypatch.setattr(builder, "_existing_model_training_artifact", lambda *a, **k: dict(info))
    return builder, output_root, existing


def test_retrain_loads_existing_weights_and_swaps_only_after_success(tmp_path, monkeypatch):
    builder, output_root, existing = _ready_builder(tmp_path, monkeypatch)
    import mlb_studio.model_runtime as runtime

    calls = []
    compiled = SimpleNamespace(compile_used=False)

    def fake_train_builder_model(**kwargs):
        calls.append(kwargs)
        staged = Path(kwargs["config"]["output_dir"]) / "My-Model"
        (staged / "last" / "tokenizer").mkdir(parents=True)
        (staged / "last" / "model.pt").write_bytes(b"new-weights")
        return {
            "model_update": {
                "training_status": "trained", "weights_ready": True,
                "path": str(staged / "last"), "checkpoint_path": str(staged / "last"),
                "tokenizer_path": str(staged / "last" / "tokenizer"),
                "execution_mode_used": "eager",
            },
            "compiled": compiled, "tokenizer": object(), "last_sample": None,
        }

    monkeypatch.setattr(runtime, "train_builder_model", fake_train_builder_model)
    result = builder.train_model("m1", resume_existing=True)

    assert calls[0]["resume_from"] == str(existing / "last")
    assert calls[0]["resume_mode"] == "weights"
    assert ".mlb-retrain-" in calls[0]["config"]["output_dir"]
    assert (existing / "last" / "model.pt").read_bytes() == b"new-weights"
    assert not (existing / "keep-old.txt").exists()
    assert result["path"] == str(existing / "last")
    assert result["checkpoint_path"] == str(existing / "last")
    assert result["retrained_from"] == str(existing / "last")


def test_failed_retrain_leaves_previous_trained_model_untouched(tmp_path, monkeypatch):
    builder, _, existing = _ready_builder(tmp_path, monkeypatch)
    import mlb_studio.model_runtime as runtime

    def fail_train(**kwargs):
        staged = Path(kwargs["config"]["output_dir"]) / "My-Model"
        staged.mkdir(parents=True)
        (staged / "partial.tmp").write_text("partial", encoding="utf-8")
        raise RuntimeError("training failed")

    monkeypatch.setattr(runtime, "train_builder_model", fail_train)

    with pytest.raises(RuntimeError, match="training failed"):
        builder.train_model("m1", resume_existing=True)

    assert (existing / "last" / "model.pt").read_bytes() == b"healthy-old-weights"
    assert (existing / "keep-old.txt").read_text(encoding="utf-8") == "old"
    assert not list(existing.parent.glob(".mlb-retrain-*"))


def test_old_override_flag_is_reinterpreted_as_retrain_for_healthy_model(tmp_path, monkeypatch):
    builder, _, existing = _ready_builder(tmp_path, monkeypatch)
    import mlb_studio.model_runtime as runtime

    seen = {}
    compiled = SimpleNamespace(compile_used=False)

    def fake_train_builder_model(**kwargs):
        seen.update(kwargs)
        staged = Path(kwargs["config"]["output_dir"]) / "My-Model"
        (staged / "last").mkdir(parents=True)
        (staged / "last" / "model.pt").write_bytes(b"new")
        return {
            "model_update": {"training_status": "trained", "weights_ready": True, "path": str(staged / "last"), "checkpoint_path": str(staged / "last"), "execution_mode_used": "eager"},
            "compiled": compiled, "tokenizer": object(), "last_sample": None,
        }

    monkeypatch.setattr(runtime, "train_builder_model", fake_train_builder_model)
    builder.train_model("m1", overwrite_existing=True)

    assert seen["resume_from"] == str(existing / "last")
    assert seen["resume_mode"] == "weights"
