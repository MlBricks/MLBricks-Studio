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


def test_corrupted_model_directory_requires_fresh_start_before_compile(tmp_path, monkeypatch):
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
    assert exc.value.action == "fresh_start"
    assert Path(exc.value.paths[0]) == existing


def test_healthy_trained_model_requires_retrain_not_override(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch)
    output_root = tmp_path / "models"
    existing = output_root / "My-Model"
    existing.mkdir(parents=True)
    builder.state["prepared_datasets"] = [{"id": "d1", "name": "Prepared"}]
    builder.prepared_datasets["d1"] = {"train": [{"input_ids": [1, 2, 3]}]}
    builder.state["model_outputs"] = [{
        "id": "m1", "name": "My Model", "selected_dataset_id": "d1",
        "training_config": {"output_dir": str(output_root), "execution_mode": "eager"},
    }]
    monkeypatch.setattr(builder, "_existing_model_training_artifact", lambda *args, **kwargs: {
        "present": True, "resumable": True, "path": str(existing / "last"),
        "kind": "trained_model", "resume_mode": "weights", "metadata": {}, "reason": None,
    })

    with pytest.raises(ArtifactConflictError) as exc:
        builder.train_model("m1")

    assert exc.value.kind == "model"
    assert exc.value.action == "retrain"
    assert "Retrain" in str(exc.value)


def test_frontend_contains_override_confirmation_flow():
    source = (Path(__file__).parents[1] / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    assert "overwrite_required" in source
    assert "Override it?" in source
    assert "overwrite_existing" in source
    assert "requestRunWithOverwrite(true)" in source
    assert 'startTrainingFromRuntime(entry,"resume")' in source
    assert 'startTrainingFromRuntime(entry,"fresh")' in source
    assert "resume_existing" in source
    assert "start_fresh" in source
    assert "Retrain it?" in source
    assert "Start fresh?" in source
    # A conflict response terminates the first request before confirmation.
    # Store overwrite_required in execution first so trainingIsRunning() does
    # not suppress the confirmed replacement request.
    assert "execution=cp(next);" in source
    assert source.index("execution=cp(next);") < source.index("handleOverwriteRequired(next);")


def test_run_data_pipeline_public_method_keeps_override_keyword():
    import inspect

    signature = inspect.signature(Builder.run_data_pipeline)
    assert "overwrite_existing" in signature.parameters
    assert signature.parameters["overwrite_existing"].default is False
    assert signature.parameters["overwrite_existing"].kind is inspect.Parameter.KEYWORD_ONLY


def test_builder_has_only_one_run_data_pipeline_definition():
    import ast

    path = Path(__file__).parents[1] / "src" / "mlb_studio" / "builder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    builder_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Builder"
    )
    definitions = [
        node for node in builder_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_data_pipeline"
    ]
    assert len(definitions) == 1
