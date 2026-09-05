from __future__ import annotations

import copy
from pathlib import Path

from mlb_studio.builder import Builder


def _builder_js() -> str:
    return (Path(__file__).resolve().parents[1] / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")


def test_prepare_draft_loads_without_replacing_active_builder_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    saved = builder.to_dict()
    saved["project"]["name"] = "Prepared Draft"
    builder.persistence.save_draft("draft-load-gate", "Prepared Draft", saved, workspace="model")

    builder.state["project"]["name"] = "Still Active"
    events = []
    builder._execute_persistence_command(
        {
            "action": "persistence_prepare_draft",
            "persistence": {"draft_id": "draft-load-gate", "updated_at": 123.0},
        },
        progress_callback=events.append,
    )

    event = events[-1]
    assert event["phase"] == "prepare_draft"
    assert event["prepared_draft"]["draft_id"] == "draft-load-gate"
    assert event["prepared_draft"]["state"]["project"]["name"] == "Prepared Draft"
    assert builder.state["project"]["name"] == "Still Active"
    assert "state_replace" not in event


def test_prepare_repository_component_returns_memory_patch_without_installing_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    before = copy.deepcopy(builder.state.get("custom_components") or {})
    builder.persistence.save_repository_item(
        kind="component",
        name="Load First Component",
        item_id="local_component_load_first",
        payload={
            "root_definition_id": "custom_saved",
            "definitions": {
                "custom_saved": {
                    "id": "custom_saved",
                    "local_id": "component_saved",
                    "name": "Load First Component",
                    "nodes": [],
                    "edges": [],
                }
            },
            "component_cache": {},
        },
    )

    events = []
    builder._execute_persistence_command(
        {
            "action": "persistence_prepare_item",
            "persistence": {"item_id": "local_component_load_first"},
        },
        progress_callback=events.append,
    )

    event = events[-1]
    assert event["phase"] == "prepare_item"
    prepared = event["prepared_item"]
    assert prepared["item_id"] == "local_component_load_first"
    assert prepared["kind"] == "component"
    root_id = prepared["component_restore"]["root_definition_id"]
    assert root_id in prepared["component_restore"]["custom_components"]
    assert (builder.state.get("custom_components") or {}) == before
    assert "state_replace" not in event


def test_gallery_local_storage_requires_load_before_open_or_recover():
    text = _builder_js()
    assert 'requestPersistenceCommand("persistence_prepare_draft"' in text
    assert 'requestPersistenceCommand("persistence_prepare_item"' in text
    assert 'recover.disabled=!prepared;' in text
    assert 'open.disabled=!prepared;' in text
    assert 'prepared?"Loaded ✓":loading?"Loading…":"Load"' in text
    assert 'function recoverPreparedDraft(item)' in text
    assert 'function openPreparedRepositoryItem(item)' in text
