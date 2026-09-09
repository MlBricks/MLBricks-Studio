from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlb_studio.builder import ArtifactConflictError, Builder
from mlb_studio.local_runtime import MANAGED_DATA_INDEX, MANAGED_MODEL_INDEX


def _write_dataset(path: Path, name: str):
    path.mkdir(parents=True)
    (path / "dataset_dict.json").write_text("{}", encoding="utf-8")
    (path / "mlbricks_dataset.json").write_text(
        json.dumps({
            "id": "dataset_disk",
            "name": name,
            "storage": "disk",
            "path": str(path),
            "splits": {"train": {"rows": 12, "columns": ["input_ids"]}},
        }),
        encoding="utf-8",
    )


def _write_model(path: Path, name: str):
    last = path / "last"
    last.mkdir(parents=True)
    # Deliberately invalid tensor bytes. The startup index must never load model.pt.
    (last / "model.pt").write_bytes(b"not-a-real-torch-checkpoint")
    (last / "metadata.json").write_text(
        json.dumps({
            "format": "mlbricks.model",
            "metadata": {
                "kind": "trained_model",
                "builder_package": {"model_entry": {"name": name}},
            },
        }),
        encoding="utf-8",
    )


def _workspace(tmp_path: Path):
    root = tmp_path / "mlbricks_workspace"
    return root, root / "models", root / "data"


def test_builder_creates_separate_shallow_model_and_data_indexes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    root, models, data = _workspace(tmp_path)
    _write_model(models / "indexed-model", "Indexed Model")
    _write_dataset(data / "indexed-data", "Indexed Data")

    builder = Builder()

    assert Path(builder.local_environment["paths"]["models"]) == models
    assert Path(builder.local_environment["paths"]["data"]) == data
    assert models != data
    assert (models / MANAGED_MODEL_INDEX).is_file()
    assert (data / MANAGED_DATA_INDEX).is_file()

    model_index = json.loads((models / MANAGED_MODEL_INDEX).read_text(encoding="utf-8"))
    data_index = json.loads((data / MANAGED_DATA_INDEX).read_text(encoding="utf-8"))
    assert [x["name"] for x in model_index["entries"]] == ["Indexed Model"]
    assert [x["name"] for x in data_index["entries"]] == ["Indexed Data"]


def test_indexed_dataset_is_registered_metadata_only_and_loaded_lazily(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    _, _, data = _workspace(tmp_path)
    path = data / "ready-data"
    _write_dataset(path, "Ready Data")

    builder = Builder()
    meta = next(x for x in builder.state["prepared_datasets"] if x["name"] == "Ready Data")

    # No Dataset/DatasetDict body is loaded during startup indexing.
    assert meta["indexed_only"] is True
    assert meta["id"] not in builder.prepared_datasets
    assert Path(meta["path"]) == path


def test_model_retrain_prompt_happens_before_dataset_or_torch_runtime_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    _, models, _ = _workspace(tmp_path)
    _write_model(models / "Fast-Model", "Fast Model")

    builder = Builder()
    builder.state["prepared_datasets"] = [{"id": "d1", "name": "Huge Dataset", "path": "/never/load"}]
    builder.state["model_outputs"] = [{
        "id": "m1",
        "name": "Fast Model",
        "selected_dataset_id": "d1",
        "training_config": {"output_dir": "mlbricks_workspace/models", "execution_mode": "compiled"},
    }]

    def forbidden(*args, **kwargs):
        raise AssertionError("dataset must not load before Retrain confirmation")

    monkeypatch.setattr(builder, "get_prepared_dataset", forbidden)

    with pytest.raises(ArtifactConflictError) as exc:
        builder.train_model("m1")

    assert exc.value.action == "retrain"
    assert "Retrain" in str(exc.value)


def test_data_name_collision_refreshes_only_managed_data_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    builder = Builder()
    data = Path(builder.local_environment["paths"]["data"])

    # Simulate a dataset appearing after Builder startup. A shallow child-signature
    # check refreshes the dedicated data index; no broad filesystem scan is needed.
    _write_dataset(data / "late-data", "Late Data")
    node = builder._prepared_output_node()
    node["params"]["dataset_name"] = "late data"

    with pytest.raises(ArtifactConflictError) as exc:
        builder._preflight_prepared_dataset_output()

    assert exc.value.kind == "dataset"
    assert "already exists" in str(exc.value)


def test_default_local_repository_scan_uses_indexes_not_environment_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    root, models, data = _workspace(tmp_path)
    _write_model(models / "repo-model", "Repo Model")
    _write_dataset(data / "repo-data", "Repo Data")
    # An unrelated loadable-looking file outside Studio's managed roots must not
    # appear in the default repository scan.
    (tmp_path / "outside.pt").write_bytes(b"outside")

    builder = Builder()
    scan = builder.scan_local_runtime_files()
    paths = {str(Path(x["path"])) for x in scan["entries"]}

    assert scan["indexed"] is True
    assert str(tmp_path / "outside.pt") not in paths
    assert any("repo-model" in x for x in paths)
    assert str(data / "repo-data") in paths


def test_indexed_dataset_repairs_stale_design_entry_without_loading_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    _, _, data = _workspace(tmp_path)
    path = data / "prepared-dataset"
    _write_dataset(path, "Prepared Dataset")

    builder = Builder()
    # Simulate browser/draft state arriving after a kernel restart. The design
    # remembers the logical dataset id/name but not the managed disk location.
    builder.state["prepared_datasets"] = [{
        "id": "design_dataset_id",
        "name": "Prepared Dataset",
        "splits": {"train": {"rows": 12, "columns": ["input_ids"]}},
    }]
    builder.prepared_datasets.clear()

    builder._ensure_managed_artifact_index_current()
    builder._hydrate_indexed_prepared_datasets()

    assert len(builder.state["prepared_datasets"]) == 1
    meta = builder.state["prepared_datasets"][0]
    # Preserve the design id so existing model references do not break.
    assert meta["id"] == "design_dataset_id"
    assert Path(meta["path"]) == path
    assert meta["indexed_only"] is True
    assert meta["storage"] == "disk"
    assert "design_dataset_id" not in builder.prepared_datasets


def test_get_prepared_dataset_reconciles_index_then_loads_only_on_demand(tmp_path, monkeypatch):
    import sys
    import types

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    _, _, data = _workspace(tmp_path)
    path = data / "prepared-dataset"
    _write_dataset(path, "Prepared Dataset")

    builder = Builder()
    builder.state["prepared_datasets"] = [{"id": "design_dataset_id", "name": "Prepared Dataset"}]
    builder.prepared_datasets.clear()

    loaded = {"train": object(), "validation": object()}
    calls = []

    def fake_load_from_disk(value):
        calls.append(str(value))
        return loaded

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_from_disk=fake_load_from_disk))

    result = builder.get_prepared_dataset("design_dataset_id")

    assert result is loaded
    assert calls == [str(path)]
    meta = builder.state["prepared_datasets"][0]
    assert meta["id"] == "design_dataset_id"
    assert Path(meta["path"]) == path
    assert meta["indexed_only"] is False
    assert meta["storage"] == "disk+memory"
