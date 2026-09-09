from __future__ import annotations

from pathlib import Path

import pytest

from mlb_studio.builder import ArtifactConflictError, Builder


def _builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    return Builder()


def test_dataset_collision_requires_explicit_override(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch)
    node = builder._prepared_output_node()
    target = Path(builder.local_environment["paths"]["data"]) / "existing-data"
    target.mkdir(parents=True)
    node["params"].update({
        "dataset_name": "Existing Data",
        "save_to_disk": "true",
        "path": str(target),
    })
    builder.state["prepared_datasets"] = [{
        "id": "dataset_keep",
        "name": "Existing Data",
        "path": str(target),
    }]

    with pytest.raises(ArtifactConflictError) as exc:
        builder._preflight_prepared_dataset_output()
    assert exc.value.kind == "dataset"
    assert str(target) in exc.value.paths

    allowed = builder._preflight_prepared_dataset_output(overwrite_existing=True)
    assert allowed["replace_metadata"]["id"] == "dataset_keep"
    assert str(target) in allowed["conflicting_paths"]


def test_directory_override_is_rollback_safe(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch)
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    staged = builder._stage_local_directory_override(target)
    assert staged is not None
    assert not target.exists()
    Path(staged["target"]).mkdir()
    (Path(staged["target"]) / "new.txt").write_text("new", encoding="utf-8")

    builder._rollback_local_directory_overrides([staged])
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()


def test_model_training_directory_requires_override_before_compile(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch)
    output_root = tmp_path / "models"
    existing = output_root / "My-Model"
    existing.mkdir(parents=True)
    (existing / "model.pt").write_bytes(b"old")

    builder.state["prepared_datasets"] = [{
        "id": "d1", "name": "Prepared", "path": None,
        "splits": {"train": {"rows": 1, "columns": ["input_ids"]}},
    }]
    builder.prepared_datasets["d1"] = {"train": [{"input_ids": [1, 2, 3]}]}
    builder.state["model_outputs"] = [{
        "id": "m1",
        "name": "My Model",
        "selected_dataset_id": "d1",
        "training_config": {"output_dir": str(output_root), "execution_mode": "eager"},
    }]

    with pytest.raises(ArtifactConflictError) as exc:
        builder.train_model("m1")
    assert exc.value.kind == "model"
    assert Path(exc.value.paths[0]) == existing


def test_frontend_contains_override_confirmation_flow():
    source = (Path(__file__).parents[1] / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    assert "overwrite_required" in source
    assert "Override it?" in source
    assert "overwrite_existing" in source
    assert "requestRunWithOverwrite(true)" in source
    assert "startTrainingFromRuntime(entry,true)" in source
