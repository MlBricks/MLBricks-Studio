from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlb_studio.builder import Builder
from mlb_studio.data import prepared_dataset_output
from mlb_studio.local_runtime import detect_local_kind, scan_model_candidates


def _builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path / ".studio"))
    return Builder()


def test_prepared_dataset_duplicate_name_is_rejected_before_fetch(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch)
    node = builder._prepared_output_node()
    node["params"]["dataset_name"] = "My Dataset"
    builder.state["prepared_datasets"] = [{"id": "d1", "name": "my dataset"}]

    with pytest.raises(FileExistsError, match="already exists.*different Dataset Name"):
        builder._preflight_prepared_dataset_output()


def test_managed_dataset_default_path_becomes_name_specific(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch)
    node = builder._prepared_output_node()
    node["params"].update({
        "dataset_name": "Tiny Stories V2",
        "save_to_disk": "true",
        "path": str(Path(builder.local_environment["paths"]["data"]) / "prepared_dataset"),
    })

    target = builder._preflight_prepared_dataset_output()
    expected = Path(builder.local_environment["paths"]["data"]) / "tiny-stories-v2"
    assert Path(target["path"]) == expected
    assert Path(node["params"]["path"]) == expected


def test_prepared_dataset_existing_storage_path_is_never_overwritten(tmp_path):
    class FakeDataset:
        def save_to_disk(self, path):
            raise AssertionError("save_to_disk must not be called for an existing target")

    target = tmp_path / "dataset-a"
    target.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepared_dataset_output(FakeDataset(), save_to_disk=True, path=str(target))


def test_disk_metadata_marker_blocks_same_dataset_name_after_restart(tmp_path, monkeypatch):
    builder = _builder(tmp_path, monkeypatch)
    data_root = Path(builder.local_environment["paths"]["data"])
    existing = data_root / "old-copy"
    existing.mkdir(parents=True)
    (existing / "mlbricks_dataset.json").write_text(
        json.dumps({"name": "Persistent Dataset"}), encoding="utf-8"
    )
    node = builder._prepared_output_node()
    node["params"]["dataset_name"] = "persistent dataset"

    with pytest.raises(FileExistsError, match="local Studio data directory"):
        builder._preflight_prepared_dataset_output()


def _write_model_artifact(path: Path, *, kind: str):
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"x")
    (path / "metadata.json").write_text(
        json.dumps({"format": "mlbricks.model", "metadata": {"kind": kind}}),
        encoding="utf-8",
    )


def test_training_checkpoints_are_not_scanned_as_separate_models(tmp_path):
    root = tmp_path / "mlbricks_workspace" / "models" / "50M-SLM"
    _write_model_artifact(root / "checkpoints" / "step_000500", kind="training_checkpoint")
    _write_model_artifact(root / "checkpoints" / "step_001000", kind="training_checkpoint")
    _write_model_artifact(root / "last", kind="trained_model")

    assert detect_local_kind(root / "checkpoints" / "step_000500")["kind"] == "training_checkpoint"
    scan = scan_model_candidates(root, max_depth=8)
    paths = {Path(item["path"]) for item in scan["entries"]}
    assert root / "last" in paths
    assert root / "checkpoints" / "step_000500" not in paths
    assert root / "checkpoints" / "step_001000" not in paths
    assert len(paths) == 1
