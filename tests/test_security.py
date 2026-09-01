from __future__ import annotations

import copy
import zipfile
from pathlib import Path

import pytest
import torch

from mlb_studio import Builder
from mlb_studio.graph import new_project
from mlb_studio.model_runtime import ModelCompileError, _APIOperation
from mlb_studio.security import UnsafeCheckpointError, safe_extract_zip, safe_torch_load


def test_safe_checkpoint_loads_tensor_only_payload(tmp_path):
    path = tmp_path / "safe.pt"
    torch.save({"model_state": {"weight": torch.arange(4)}, "step": 3}, path)
    payload = safe_torch_load(path)
    assert payload["step"] == 3
    assert torch.equal(payload["model_state"]["weight"], torch.arange(4))


def test_malicious_pickle_checkpoint_is_blocked_without_execution(tmp_path):
    marker = tmp_path / "executed.txt"

    class Evil:
        def __reduce__(self):
            import os
            return (os.system, (f'echo compromised > "{marker}"',))

    path = tmp_path / "evil.pt"
    torch.save({"model_state": {}, "evil": Evil()}, path)

    with pytest.raises(UnsafeCheckpointError):
        safe_torch_load(path)
    assert not marker.exists()


def test_zip_traversal_is_blocked(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "blocked")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="Unsafe ZIP path"):
        safe_extract_zip(archive, out)
    assert not (tmp_path / "escape.txt").exists()


def _project_with_user_code():
    state = new_project("Untrusted")
    state["custom_components"] = {
        "custom_1": {
            "id": "custom_1",
            "name": "Custom API",
            "implementation": "api",
            "api_binding": {
                "call_type": "user_function",
                "user_function_name": "run",
                "user_code": "def run(x):\n    return x",
            },
        }
    }
    return state


def test_external_project_is_untrusted_and_trust_is_session_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = Builder(project=copy.deepcopy(_project_with_user_code()))
    info = b.project_trust_info()
    assert info["trusted"] is False
    assert info["requires_trust"] is True
    assert info["executable_features"]
    with pytest.raises(PermissionError, match="untrusted"):
        b.ensure_external_import("os.path.join")

    assert b.trust_project()["trusted"] is True
    # Trust is not serialized into the project state.
    assert "trusted" not in b.to_dict()
    assert "project_trust" not in b.to_dict()


def test_untrusted_api_source_is_not_executed(tmp_path):
    marker = tmp_path / "api_executed.txt"
    source = f'''from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\ndef run(x):\n    return x\n'''
    binding = {
        "call_type": "user_function",
        "user_function_name": "run",
        "user_code": source,
        "parameters": [],
    }
    with pytest.raises(ModelCompileError, match="untrusted"):
        _APIOperation(binding=binding, params={}, runtime={"allow_user_code": False}, label="Danger")
    assert not marker.exists()
