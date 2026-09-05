from __future__ import annotations

import json
import sqlite3

from mlb_studio.persistence import StudioPersistence, mask_secret, sanitize_design


def test_sanitize_design_removes_secrets_and_heavy_state():
    payload = {
        "project": {"name": "Demo", "estimated_parameters": 123},
        "token": "hf_super_secret",
        "nested": {
            "connection_string": "AccountKey=secret",
            "state_dict": {"weight": [1, 2, 3]},
            "checkpoint_path": "/tmp/model.pt",
        },
        "nodes": [{"params": {"dim": 384}}],
    }
    clean = sanitize_design(payload)
    assert "token" not in clean
    assert "connection_string" not in clean["nested"]
    assert "state_dict" not in clean["nested"]
    assert clean["nested"]["checkpoint_path"] == "/tmp/model.pt"
    assert clean["nodes"][0]["params"]["dim"] == 384


def test_draft_and_repository_survive_new_instance(tmp_path):
    first = StudioPersistence(tmp_path)
    state = {
        "project": {"name": "ESA Design", "local_id": "p1"},
        "active_workspace": "model",
        "components": {"root": {"id": "root", "nodes": [], "edges": []}},
        "root_component_id": "root",
        "token": "must-not-persist",
    }
    first.save_draft("p1", "ESA Design", state)
    saved = first.save_repository_item(kind="model", name="ESA 200M", payload={"state": state})

    second = StudioPersistence(tmp_path)
    draft = second.load_draft("p1")
    item = second.load_repository_item(saved["id"])
    assert draft["project"]["name"] == "ESA Design"
    assert "token" not in draft
    assert item["payload"]["state"]["project"]["name"] == "ESA Design"
    assert item["metadata"] == {}


def test_credential_database_contains_masks_not_plain_secret(tmp_path, monkeypatch):
    store = StudioPersistence(tmp_path)
    monkeypatch.setattr(store, "_keyring_backend", lambda: None)
    result = store.save_credentials("huggingface", "Default", {"token": "hf_abcdefghijklmnop"})
    assert result["persistent"] is False
    assert result["masks"]["token"].startswith("hf_")
    assert "abcdefghijklmnop" not in result["masks"]["token"]
    assert store.get_credentials("huggingface", "Default")["token"] == "hf_abcdefghijklmnop"

    raw = store.db_path.read_bytes()
    assert b"hf_abcdefghijklmnop" not in raw
    assert store.list_credentials()[0]["name"] == "Default"


def test_mask_secret_keeps_only_small_identifier_edges():
    masked = mask_secret("github_pat_1234567890abcdef")
    assert "1234567890" not in masked
    assert masked.endswith("cdef")


def test_draft_summary_reports_active_graph_progress(tmp_path):
    store = StudioPersistence(tmp_path)
    state = {
        "project": {"name": "Progress", "local_id": "progress-1"},
        "active_workspace": "model",
        "workspaces": {"model": {"root_component_id": "root"}},
        "components": {
            "root": {
                "id": "root",
                "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                "edges": [{"id": "e1"}, {"id": "e2"}],
            }
        },
        "root_component_id": "root",
    }
    store.save_draft("progress-1", "Progress", state, workspace="model")
    draft = store.list_drafts()[0]
    assert draft["node_count"] == 3
    assert draft["edge_count"] == 2


def test_component_draft_uses_unique_component_id_and_is_removed_on_save(tmp_path):
    from mlb_studio.builder import Builder

    builder = Builder()
    builder.persistence = StudioPersistence(tmp_path)

    project_id = "project_demo"
    component_local_id = "component_123456"
    definition_id = "custom_demo"
    view_id = "view_custom_demo"

    builder.state.setdefault("project", {})["local_id"] = project_id
    builder.state["project"]["name"] = "Demo Model"
    builder.state.setdefault("custom_components", {})[definition_id] = {
        "id": definition_id,
        "local_id": component_local_id,
        "name": "My Component",
        "implementation": "graph",
        "nodes": [{"id": "n1", "type": "linear"}],
        "edges": [],
    }
    builder.state.setdefault("components", {})[view_id] = {
        "id": view_id,
        "name": "My Component",
        "kind": "custom_edit",
        "definition_id": definition_id,
        "nodes": [{"id": "n1", "type": "linear"}],
        "edges": [],
    }
    builder.state["view_component_id"] = view_id

    # Normal project recovery state remains independent.
    builder._execute_persistence_command({
        "action": "persistence_save_draft",
        "persistence": {
            "draft_id": project_id,
            "draft_name": "Demo Model",
            "draft_kind": "project",
            "workspace": "model",
        },
    })

    component_draft_id = f"draft_{component_local_id}"
    builder._execute_persistence_command({
        "action": "persistence_save_draft",
        "persistence": {
            "draft_id": component_draft_id,
            "draft_name": "My Component",
            "draft_kind": "component",
            "workspace": "component",
            "component_local_id": component_local_id,
            "definition_id": definition_id,
        },
    })

    before = {item["id"] for item in builder.persistence.list_drafts()}
    assert project_id in before
    assert component_draft_id in before

    result = builder._execute_persistence_command({
        "action": "persistence_save_item",
        "persistence": {
            "kind": "component",
            "name": "My Component",
            "definition_id": definition_id,
            "component_local_id": component_local_id,
            "item_id": f"local_{component_local_id}",
            "source_draft_id": component_draft_id,
        },
    })

    after = {item["id"] for item in builder.persistence.list_drafts()}
    assert project_id in after
    assert component_draft_id not in after
    assert result["id"] == f"local_{component_local_id}"
    assert result["component_local_id"] == component_local_id
    assert result["cleared_draft_id"] == component_draft_id

    saved = builder.persistence.load_repository_item(result["id"])
    assert saved["metadata"]["component_local_id"] == component_local_id
    root_id = saved["payload"]["root_definition_id"]
    assert saved["payload"]["definitions"][root_id]["local_id"] == component_local_id


def test_component_draft_summary_counts_outer_component_editor(tmp_path):
    store = StudioPersistence(tmp_path)
    state = {
        "project": {"name": "Demo", "local_id": "project_demo"},
        "active_workspace": "model",
        "custom_components": {
            "outer": {"id": "outer", "local_id": "component_outer", "name": "Outer"},
            "inner": {"id": "inner", "local_id": "component_inner", "name": "Inner"},
        },
        "components": {
            "outer_view": {
                "id": "outer_view", "kind": "custom_edit", "definition_id": "outer",
                "nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"id": "e"}],
            },
            "inner_view": {
                "id": "inner_view", "kind": "custom_edit", "definition_id": "inner",
                "nodes": [{"id": "x"}], "edges": [],
                "parent_edit_return": {"view_id": "outer_view"},
            },
        },
        "view_component_id": "inner_view",
    }
    store.save_draft("draft_component_outer", "Outer", state, workspace="component")
    draft = store.list_drafts()[0]
    assert draft["workspace"] == "component"
    assert draft["node_count"] == 2
    assert draft["edge_count"] == 1
