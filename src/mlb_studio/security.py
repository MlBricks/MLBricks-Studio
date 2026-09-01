from __future__ import annotations

import copy
import warnings
import stat
import zipfile
from pathlib import Path
from typing import Any


class UnsafeCheckpointError(RuntimeError):
    """Raised when a legacy pickle checkpoint cannot be loaded safely."""


def safe_torch_load(path: str | Path, *, map_location: str = "cpu", allow_unsafe_pickle: bool = False) -> Any:
    """Load a PyTorch checkpoint without arbitrary pickle execution by default.

    PyTorch's ``weights_only=True`` restricted unpickler accepts tensors and the
    simple containers used by current MLB Studio legacy checkpoints. Older
    checkpoints containing arbitrary Python objects are rejected unless the
    caller makes the explicit ``allow_unsafe_pickle=True`` trust decision.
    """
    import torch

    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        # Extremely old PyTorch versions do not implement weights_only. Never
        # silently fall back to unrestricted pickle in that situation.
        if not allow_unsafe_pickle:
            raise UnsafeCheckpointError(
                "This PyTorch build does not support safe checkpoint loading. "
                "Upgrade PyTorch, or explicitly opt in to unsafe legacy pickle loading "
                "only for a checkpoint you created or fully trust."
            ) from exc
    except Exception as exc:
        if not allow_unsafe_pickle:
            raise UnsafeCheckpointError(
                f"Legacy checkpoint {path.name!r} was blocked because it requires "
                "Python pickle deserialization. Only load it with unsafe legacy mode "
                "after verifying that you trust its source."
            ) from exc

    warnings.warn(
        "Unsafe legacy checkpoint loading is enabled. torch.load(weights_only=False) "
        "can execute code embedded in a malicious checkpoint.",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.load(path, map_location=map_location, weights_only=False)


def project_executable_features(state: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return executable/import-capable features embedded in a Builder project."""
    state = state or {}
    found: list[dict[str, str]] = []

    def inspect_binding(binding: dict[str, Any], label: str) -> None:
        call_type = str(binding.get("call_type") or binding.get("target_kind") or "").lower()
        if str(binding.get("user_code") or "").strip():
            found.append({"kind": "python_source", "label": label, "detail": "User Function source"})
        if str(binding.get("user_class_code") or "").strip():
            found.append({"kind": "python_source", "label": label, "detail": "User Class source"})
        import_path = str(
            binding.get("import_path")
            or binding.get("target")
            or binding.get("module_path")
            or binding.get("module")
            or ""
        ).strip()
        if import_path and call_type not in {"user_function", "user_class"}:
            found.append({"kind": "python_import", "label": label, "detail": import_path})

    custom = copy.deepcopy(state.get("custom_components") or {})
    for definition_id, definition in custom.items():
        label = str(definition.get("name") or definition_id)
        binding = definition.get("api_binding") or {}
        if binding:
            inspect_binding(binding, label)
        for node in definition.get("nodes") or []:
            if str(node.get("type") or "") == "api_step":
                inspect_binding(node.get("api_binding") or {}, str(node.get("name") or label))

    # A model snapshot can carry a custom component even when the live project
    # custom-component table no longer contains it.
    for entry in state.get("model_outputs") or []:
        for definition_id, definition in (entry.get("custom_components_snapshot") or {}).items():
            label = str(definition.get("name") or definition_id)
            binding = definition.get("api_binding") or {}
            if binding:
                inspect_binding(binding, label)
            for node in definition.get("nodes") or []:
                if str(node.get("type") or "") == "api_step":
                    inspect_binding(node.get("api_binding") or {}, str(node.get("name") or label))

    unique = []
    seen = set()
    for item in found:
        key = (item["kind"], item["label"], item["detail"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def safe_extract_zip(archive: str | Path, destination: str | Path) -> Path:
    """Extract a ZIP while rejecting traversal paths and symlink entries."""
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            name = str(info.filename or "")
            if not name:
                continue
            target = (destination / name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe ZIP path was blocked: {name!r}") from exc
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise RuntimeError(f"ZIP symlink entry was blocked: {name!r}")
        zf.extractall(destination)
    return destination


__all__ = ["UnsafeCheckpointError", "safe_torch_load", "project_executable_features", "safe_extract_zip"]
