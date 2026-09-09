from __future__ import annotations
import json, time

from mlb_studio.builder import Builder

class Dummy:
    def __init__(self, value=""):
        self.value = value


def test_busy_bridge_queues_recover_instead_of_dropping_click(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    state = json.dumps(builder.to_dict())
    widgets = {
        "state": Dummy(state),
        "command": Dummy(json.dumps({"action": "persistence_save_draft"})),
        "progress": Dummy(""),
    }
    builder._bridge_widgets = widgets
    calls = []

    def fake_execute(command, progress_callback=None):
        calls.append(command["action"])
        if command["action"] == "persistence_save_draft":
            time.sleep(0.12)
        if progress_callback:
            progress_callback({"status":"done", "runtime_kind":"persistence", "phase":"load_draft" if "load_draft" in command["action"] else "save_draft", "overall":100})

    builder._execute_persistence_command = fake_execute
    builder._start_bridge_run()
    widgets["command"].value = json.dumps({"action":"persistence_load_draft", "persistence":{"draft_id":"x"}})
    builder._start_bridge_run()

    deadline = time.time() + 2.0
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.03)

    assert calls == ["persistence_save_draft", "persistence_load_draft"]


def test_generate_bypasses_active_background_bridge_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()
    builder._active_bridge_action = "persistence_save_draft"

    class AliveThread:
        @staticmethod
        def is_alive():
            return True

    builder._run_thread = AliveThread()
    widgets = {
        "state": Dummy(json.dumps(builder.to_dict())),
        "command": Dummy(json.dumps({"action":"generate", "model_id":"model-hot", "generation_config":{"prompt":"hello"}})),
        "progress": Dummy(""),
    }
    builder._bridge_widgets = widgets
    calls = []
    builder._start_hot_bridge_generation = lambda command: calls.append(command) or True

    builder._start_bridge_run()

    assert len(calls) == 1
    assert calls[0]["action"] == "generate"
    assert calls[0]["generation_config"]["prompt"] == "hello"
    assert widgets["command"].value == "{}"
    assert builder._bridge_pending_requests == []
