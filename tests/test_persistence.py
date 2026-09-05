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
