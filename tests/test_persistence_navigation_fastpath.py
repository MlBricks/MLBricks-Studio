from __future__ import annotations

import json
import time

from mlb_studio.builder import Builder


class Dummy:
    def __init__(self, value=""):
        self.value = value


def test_load_draft_fast_path_does_not_parse_current_bridge_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    builder.persistence.save_draft("draft-fast", "Fast", builder.to_dict(), workspace="model")
    progress = Dummy("")
    builder._bridge_widgets = {
        "state": Dummy("{this is deliberately invalid and must be skipped"),
        "command": Dummy(json.dumps({"action": "persistence_load_draft", "persistence": {"draft_id": "draft-fast"}})),
        "progress": progress,
    }

    builder._start_bridge_run()
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            payload = json.loads(progress.value or "{}")
        except Exception:
            payload = {}
        if payload.get("status") in {"done", "error"}:
            break
        time.sleep(0.02)

    payload = json.loads(progress.value)
    assert payload["status"] == "done"
    assert payload["phase"] == "load_draft"
    assert "Could not read Builder state" not in payload.get("message", "")


def test_component_repository_open_returns_small_patch_not_whole_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    builder.persistence.save_repository_item(
        kind="component",
        name="Fast Component",
        item_id="local_component_fast",
        payload={
            "root_definition_id": "custom_old",
            "definitions": {
                "custom_old": {
                    "id": "custom_old",
                    "name": "Fast Component",
                    "nodes": [],
                    "edges": [],
                }
            },
            "component_cache": {},
        },
    )
    events = []
    builder._execute_persistence_command(
        {"action": "persistence_load_item", "persistence": {"item_id": "local_component_fast"}},
        progress_callback=events.append,
    )
    event = events[-1]
    assert event["status"] == "done"
    assert event["phase"] == "load_item"
    assert "component_restore" in event
    assert "state_replace" not in event
    assert event["persistence_result"]["root_definition_id"] in event["component_restore"]["custom_components"]
    assert "persistence_summary" not in event
