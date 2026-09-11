from __future__ import annotations
import html
import base64
import gzip
import ast
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import uuid
import threading
import time
import platform
import os
import re
import copy
import shutil
import subprocess
import sys
import tempfile
import zipfile
import traceback
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from collections.abc import Mapping

from .graph import (
    new_project, primitive_catalog, tinystories_30m_project,
    slm_200m_project, stateaware_esa_200m_project, soup_200m_project, soup_30m_1l_project,
)
from .runtime import get_mlbricks_info
from .security import project_executable_features, safe_extract_zip, safe_torch_load
from .persistence import StudioPersistence, sanitize_design
from .version import __version__, FORMAT_VERSION
from .api_registry import discover_mlbricks_api, refresh_component_api
from .import_pool import IMPORT_POOL
from .runner import execute_data_pipeline, validate_data_pipeline, PipelineValidationError, PipelineStopped

_STATIC = Path(__file__).parent / "static"

# Notebook pages only need to parse/execute the large frontend bundle once per
# live Python session.  Plain _repr_html_ remains standalone for compatibility.
_FRONTEND_ASSETS_EMITTED = False
_FRONTEND_ASSETS_LOCK = threading.Lock()
_FRONTEND_BUNDLE_CACHE = None
_WINDOWS_CUDA_PROBE_CACHE = None
_MACOS_MPS_PROBE_CACHE = None


class ArtifactConflictError(FileExistsError):
    """A local Studio artifact exists and needs an explicit user decision.

    Datasets use ``action="override"``. Existing model weights are different:
    a healthy artifact should be *retrained from its parameters*, never silently
    replaced. Only an incomplete/corrupted model asks for ``fresh_start``.
    """

    def __init__(self, *, kind, name, paths=None, message=None, action="override", reason=None):
        self.kind = str(kind or "artifact")
        self.name = str(name or self.kind.title())
        self.paths = [str(x) for x in (paths or []) if x]
        self.action = str(action or "override")
        self.reason = str(reason or "").strip() or None
        if message is None:
            where = f" at {self.paths[0]}" if self.paths else ""
            if self.action == "retrain":
                message = (
                    f'{self.kind.title()} "{self.name}" already has trained parameters{where}. '
                    'Confirm Retrain to continue from the existing weights.'
                )
            elif self.action == "fresh_start":
                message = (
                    f'{self.kind.title()} "{self.name}" cannot be resumed safely{where}. '
                    'Confirm Start Fresh to replace the unusable local artifact.'
                )
            else:
                message = (
                    f'{self.kind.title()} "{self.name}" already exists{where}. '
                    'Choose another name/path, or confirm Override to replace the existing local artifact.'
                )
        super().__init__(message)

    def payload(self):
        return {
            "kind": self.kind,
            "name": self.name,
            "paths": list(self.paths),
            "message": str(self),
            "action": self.action,
            "reason": self.reason,
        }


def _probe_windows_torch_cuda():
    """Return CUDA devices visible to this Python environment on Windows.

    Windows normally has neither ``/dev/nvidia*`` device files nor CUDA
    visibility environment variables. Probe in a child interpreter so the
    selector is accurate without importing torch (and initializing CUDA) in
    Studio's long-lived UI process.
    """
    global _WINDOWS_CUDA_PROBE_CACHE
    if platform.system().lower() != "windows":
        return None
    if _WINDOWS_CUDA_PROBE_CACHE is not None:
        return copy.deepcopy(_WINDOWS_CUDA_PROBE_CACHE)

    code = (
        "import json, torch; "
        "available=bool(torch.cuda.is_available()); "
        "devices=[]; "
        "[(lambda p,i: devices.append({"
        "'index':i,'name':torch.cuda.get_device_name(i),"
        "'compute_capability':'.'.join(map(str,torch.cuda.get_device_capability(i))),"
        "'total_memory':int(p.total_memory)"
        "}))(torch.cuda.get_device_properties(i),i) "
        "for i in range(torch.cuda.device_count())] if available else None; "
        "print(json.dumps({"
        "'probed':True,'available':available,'devices':devices,"
        "'cuda_version':torch.version.cuda,'torch_version':torch.__version__"
        "}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "CUDA probe failed").strip())
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        result = json.loads(lines[-1])
        if not isinstance(result, dict):
            raise ValueError("CUDA probe returned an invalid payload")
    except Exception as exc:
        result = {"probed": False, "available": False, "devices": [], "error": str(exc)}

    _WINDOWS_CUDA_PROBE_CACHE = result
    return copy.deepcopy(result)


def _probe_macos_torch_mps():
    """Return whether this Python environment can use Apple Metal via MPS."""
    global _MACOS_MPS_PROBE_CACHE
    if platform.system().lower() != "darwin":
        return None
    if _MACOS_MPS_PROBE_CACHE is not None:
        return copy.deepcopy(_MACOS_MPS_PROBE_CACHE)

    code = (
        "import json, torch; "
        "mps=getattr(torch.backends,'mps',None); "
        "built=bool(mps and mps.is_built()); "
        "available=bool(mps and mps.is_available()); "
        "print(json.dumps({"
        "'probed':True,'built':built,'available':available,"
        "'torch_version':torch.__version__"
        "}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "MPS probe failed").strip())
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        result = json.loads(lines[-1])
        if not isinstance(result, dict):
            raise ValueError("MPS probe returned an invalid payload")
    except Exception as exc:
        result = {"probed": False, "built": False, "available": False, "error": str(exc)}

    _MACOS_MPS_PROBE_CACHE = result
    return copy.deepcopy(result)

def _compressed_frontend_bundle():
    """Return gzip+base64 frontend assets, cached for this Python process."""
    global _FRONTEND_BUNDLE_CACHE
    if _FRONTEND_BUNDLE_CACHE is None:
        css = (_STATIC / "builder.css").read_bytes()
        js = (_STATIC / "builder.js").read_bytes()
        _FRONTEND_BUNDLE_CACHE = (
            base64.b64encode(gzip.compress(css, compresslevel=9)).decode("ascii"),
            base64.b64encode(gzip.compress(js, compresslevel=9)).decode("ascii"),
        )
    return _FRONTEND_BUNDLE_CACHE


class _LocalBridgeValue:
    """Tiny value holder used by Builder.app() in place of ipywidgets."""

    def __init__(self, value=""):
        self.value = value


class Builder:
    """
    Kaggle/Jupyter-safe Builder.

    This class intentionally does not inherit from AnyWidget or ipywidgets.
    Notebook rendering uses the standard `_repr_html_` protocol.
    """

    def __init__(self, project=None, preset=None):
        if project is not None:
            self.state = project
        elif preset in {"tinystories", "tinystories-30m", "tinystories-50m", "50m", "50m-slm", "slm-50m", "demo"}:
            self.state = tinystories_30m_project()
        elif preset in {"esa-200m", "200m", "200m-slm", "slm-200m"}:
            self.state = slm_200m_project()
        elif preset in {"stateaware-esa-200m", "stateaware_esa_200m", "200m-stateaware", "stateaware-200m"}:
            self.state = stateaware_esa_200m_project()
        elif preset in {"soup-200m", "soup-200m-3l", "soup_200m"}:
            self.state = soup_200m_project()
        elif preset in {"soup-30m", "soup-30m-1l", "soup_30m_1l", "soup-50m", "soup-50m-2l", "soup_50m_2l"}:
            self.state = soup_30m_1l_project()
        else:
            self.state = new_project()
        # Trust is deliberately session-local and is never serialized. New/preset
        # projects are authored in this Python session; externally supplied states
        # must be explicitly trusted before imports or embedded Python can execute.
        self._project_trusted = project is None
        self._project_trust_origin = "local-session" if self._project_trusted else "external-constructor"
        self.catalog = primitive_catalog()
        self.import_pool = IMPORT_POOL
        # Source schema is available immediately; MLBricks modules themselves
        # are resolved lazily through the shared import pool as components are used.
        self.mlbricks_api = discover_mlbricks_api()
        for item in self.catalog:
            real = self.mlbricks_api.get(item.get("type"))
            if real:
                # Keep one canonical frontend copy of runtime API metadata.
                # The catalog only needs the parameter schema used to construct nodes.
                item["api"] = real.get("parameters", item.get("api", []))
                if real.get("description"):
                    item["description"] = real["description"]

        # UI naming policy: keep the adaptive precision component branded simply
        # as "ElasticBit". Older saved projects may still contain the historical
        # "ElasticBit 4-32" / "ElasticBit 4–32" display label, so normalize
        # those labels on load as well as the live catalog.
        for item in self.catalog:
            if item.get("type") == "elasticbit_runtime":
                item["name"] = "ElasticBit"
        for component in (self.state.get("components") or {}).values():
            for node in component.get("nodes") or []:
                if node.get("type") != "elasticbit_runtime":
                    continue
                for key in ("name", "display_name"):
                    value = str(node.get(key) or "").strip()
                    if value in {"ElasticBit 4-32", "ElasticBit 4–32", "ElasticBit 4—32"}:
                        node[key] = "ElasticBit"
        self._instance_id = f"mlb_{uuid.uuid4().hex}"
        self._run_thread = None
        # Generation normally uses the main runtime worker.  If a notebook-only
        # background autosave/import is occupying that worker, a tiny dedicated
        # hot-path thread is allowed to use the already-resident model instead of
        # making an interactive response wait behind disk/UI housekeeping.
        self._hot_generation_thread = None
        self._hot_generation_model_id = None
        self._stop_event = threading.Event()
        self._bridge_widgets = None
        # Sequenced local-app progress queue: preserve token events instead of
        # overwriting them in the single textarea bridge before the browser polls.
        self._progress_events = deque(maxlen=4096)
        self._progress_event_seq = 0
        self._progress_event_lock = threading.Lock()
        self._app_server = None
        self._app_thread = None
        self._app_url = None
        # The notebook bridge has one Run button shared by imports, autosaves,
        # persistence navigation and runtime actions. Keep quick requests queued
        # instead of dropping a click when another bridge worker is finishing.
        self._bridge_pending_requests = []
        self._bridge_queue_lock = threading.Lock()
        self._bridge_drain_scheduled = False
        self._active_bridge_action = None
        self._active_bridge_model_id = None
        self.last_data_result = None
        self.last_run_error = None
        self.trained_models = {}
        self._unsafe_legacy_checkpoints = set()
        self._model_servers = {}
        # Actual Dataset/DatasetDict objects stay in Python memory. The serializable
        # metadata lives in state["prepared_datasets"] and is saved with the design.
        self.prepared_datasets = {}
        self.state.setdefault("prepared_datasets", [])
        self.state.setdefault("component_cache", {})
        self.state.setdefault("project", {})
        self.state["project"].setdefault("local_id", f"project_{uuid.uuid4().hex}")
        # Local persistence stores design/configuration state only. Heavy model
        # tensors, optimizer state, datasets and credentials are intentionally
        # excluded by StudioPersistence.
        self.persistence = StudioPersistence()
        self.runtime_capabilities = self._detect_runtime_capabilities()
        from .local_runtime import (
            detect_local_environment, ensure_mlbricks_workspace,
            refresh_managed_artifact_indexes,
        )
        self.local_environment = detect_local_environment()
        self.local_environment["paths"] = ensure_mlbricks_workspace(self.local_environment)
        # Studio owns two dedicated artifact roots. Build one shallow metadata
        # index at startup so later Build/Fetch/Train clicks can detect existing
        # artifacts immediately without recursively searching the notebook disk
        # or loading model weights/dataset rows into RAM.
        self.local_artifact_index = refresh_managed_artifact_indexes(
            self.local_environment.get("paths") or {}
        )
        actual_root = Path(self.local_environment["paths"]["root"]).parent
        expected_root = Path(self.local_environment.get("workspace_root") or self.local_environment.get("default_root") or actual_root)
        if actual_root != expected_root:
            self.local_environment["workspace_root"] = str(actual_root)
            self.local_environment["default_root"] = str(actual_root)
            roots = [str(actual_root), *(self.local_environment.get("roots") or [])]
            self.local_environment["roots"] = list(dict.fromkeys(roots))
        self._apply_local_workspace_defaults()
        self._hydrate_indexed_prepared_datasets()

    def _refresh_managed_artifact_index(self):
        from .local_runtime import refresh_managed_artifact_indexes
        self.local_artifact_index = refresh_managed_artifact_indexes(
            self.local_environment.get("paths") or {}
        )
        return self.local_artifact_index

    def _ensure_managed_artifact_index_current(self):
        from .local_runtime import load_managed_artifact_indexes
        self.local_artifact_index = load_managed_artifact_indexes(
            self.local_environment.get("paths") or {}
        )
        return self.local_artifact_index

    def _indexed_data_records(self):
        return copy.deepcopy(((self.local_artifact_index or {}).get("data") or {}).get("entries") or [])

    def _indexed_model_records(self):
        return copy.deepcopy(((self.local_artifact_index or {}).get("models") or {}).get("entries") or [])

    def _hydrate_indexed_prepared_datasets(self):
        """Reconcile managed dataset metadata without loading dataset rows.

        Saved Studio designs can contain an older metadata-only dataset entry with
        the same name but no disk path.  The managed data index is authoritative
        for *where* Studio-saved datasets live, so enrich the existing registry
        entry instead of skipping the indexed copy.  Preserving the existing id is
        important because model Text Input nodes reference that id.

        ``get_prepared_dataset`` remains the only place that calls
        ``datasets.load_from_disk`` and therefore startup/state synchronization
        never loads dataset rows into RAM.
        """
        registry = self.state.setdefault("prepared_datasets", [])

        def resolved_path(value):
            if not value:
                return ""
            try:
                return str(Path(value).expanduser().resolve())
            except Exception:
                return str(value)

        by_id = {str(item.get("id")): item for item in registry if item.get("id")}
        by_name = {
            self._artifact_name_key(item.get("name")): item
            for item in registry
            if self._artifact_name_key(item.get("name"))
        }
        by_path = {
            resolved_path(item.get("path")): item
            for item in registry
            if item.get("path")
        }

        for record in self._indexed_data_records():
            if record.get("status") != "ready":
                continue
            path = record.get("path")
            if not path:
                continue
            resolved = resolved_path(path)
            marker = copy.deepcopy(record.get("metadata") or {})
            record_id = str(marker.get("id") or record.get("id") or "")
            record_name = str(marker.get("name") or record.get("name") or Path(path).name)
            name_key = self._artifact_name_key(record_name)

            existing = (
                by_path.get(resolved)
                or (by_id.get(record_id) if record_id else None)
                or (by_name.get(name_key) if name_key else None)
            )

            if existing is None:
                existing = {
                    "id": record_id or f"dataset_{uuid.uuid4().hex[:12]}",
                    "name": record_name,
                    "created_at": marker.get("created_at"),
                    "output_node_id": marker.get("output_node_id"),
                    "pipeline": copy.deepcopy(marker.get("pipeline") or {}),
                }
                registry.append(existing)

            # Keep the browser/design id when present so model references remain
            # valid, but always repair the managed storage location from the index.
            existing_id = str(existing.get("id") or record_id or f"dataset_{uuid.uuid4().hex[:12]}")
            existing["id"] = existing_id
            if not existing.get("name"):
                existing["name"] = record_name
            existing["path"] = resolved
            in_memory = existing_id in self.prepared_datasets
            existing["storage"] = "disk+memory" if in_memory else "disk"
            existing["indexed_only"] = not in_memory

            for key in (
                "created_at", "output_node_id", "splits", "rows", "columns",
                "tokenizer_name", "hub_repo_id", "hub_url", "hub_revision", "pipeline",
            ):
                indexed_value = marker.get(key, record.get(key))
                if indexed_value is not None and (key not in existing or existing.get(key) in (None, "", {}, [])):
                    existing[key] = copy.deepcopy(indexed_value)

            by_id[existing_id] = existing
            if record_id:
                by_id.setdefault(record_id, existing)
            by_path[resolved] = existing
            if name_key:
                by_name[name_key] = existing

        return registry

    def _apply_local_workspace_defaults(self):
        """Replace legacy Kaggle-only defaults with this session's workspace paths."""
        paths = self.local_environment.get("paths") or {}
        workspace_root = self.local_environment.get("workspace_root") or self.local_environment.get("default_root") or "."
        data_default = str(Path(paths.get("data") or (Path(workspace_root) / "mlbricks_workspace" / "data")) / "prepared_dataset")

        for item in self.catalog:
            for field in item.get("api") or []:
                if item.get("type") == "local_dataset" and field.get("key") == "path":
                    current = str(field.get("value") or "")
                    if current == "." or current.startswith("/kaggle/"):
                        field["value"] = str(workspace_root)
                if item.get("type") == "prepared_dataset" and field.get("key") == "path":
                    current = str(field.get("value") or "")
                    if current in {"mlbricks_workspace/data/prepared_dataset", "mlbricks/data/prepared_dataset"} or current.startswith("/kaggle/"):
                        field["value"] = data_default

        for component in (self.state.get("components") or {}).values():
            for node in component.get("nodes") or []:
                params = node.setdefault("params", {})
                local_path = str(params.get("path") or "")
                if node.get("type") == "local_dataset" and (local_path == "." or local_path.startswith("/kaggle/")):
                    params["path"] = str(workspace_root)
                prepared_path = str(params.get("path") or "")
                if node.get("type") == "prepared_dataset" and (prepared_path in {"mlbricks_workspace/data/prepared_dataset", "mlbricks/data/prepared_dataset"} or prepared_path.startswith("/kaggle/")):
                    params["path"] = data_default

    def _detect_runtime_capabilities(self):
        """Detect runtime choices without importing PyTorch during Studio startup.

        Importing torch can initialize CUDA and take many seconds on hosted notebook
        runtimes.  Studio only needs lightweight device hints for the initial UI;
        the real runtime validates the selected device when training/generation starts.
        """
        devices = [
            {
                "id": "auto",
                "label": "Auto (recommended)",
                "kind": "auto",
                "available": True,
            },
            {
                "id": "cpu",
                "label": f"CPU — {platform.processor() or platform.machine() or 'System CPU'}",
                "kind": "cpu",
                "available": True,
            },
        ]

        # Read installed package metadata without importing torch.
        try:
            torch_version = importlib.metadata.version("torch")
        except Exception:
            torch_version = None

        # Hosted Linux notebooks use cheap environment/device-file hints. Local
        # Windows sessions have neither of those in the common case, so query
        # the same Python environment in an isolated child process. This keeps
        # torch out of Studio's UI process while making Windows GPU discovery
        # authoritative instead of relying on a generic hardware hint.
        cuda_version = None
        cuda_hint = False
        cuda_probe = _probe_windows_torch_cuda()
        if cuda_probe and cuda_probe.get("probed"):
            cuda_version = cuda_probe.get("cuda_version")
            for item in cuda_probe.get("devices") or []:
                index = int(item.get("index", len(devices) - 2))
                name = str(item.get("name") or f"CUDA device {index}")
                devices.append({
                    "id": f"cuda:{index}",
                    "label": f"GPU {index} — {name}",
                    "kind": "cuda",
                    "index": index,
                    "name": name,
                    "available": True,
                    "compute_capability": item.get("compute_capability"),
                    "total_memory": item.get("total_memory"),
                })

        visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
        nvidia_visible = str(os.environ.get("NVIDIA_VISIBLE_DEVICES", "")).strip()
        if not (cuda_probe and cuda_probe.get("probed")):
            if visible and visible not in {"-1", "none", "None"}:
                cuda_hint = True
            elif nvidia_visible and nvidia_visible.lower() not in {"none", "void"}:
                cuda_hint = True
            elif Path("/dev/nvidia0").exists():
                cuda_hint = True
        if cuda_hint and not any(item.get("kind") == "cuda" for item in devices):
            devices.append({
                "id": "cuda:0",
                "label": "GPU — CUDA device",
                "kind": "cuda",
                "index": 0,
                "name": "CUDA device",
                "available": True,
                "provisional": True,
            })

        mps_probe = _probe_macos_torch_mps()
        if mps_probe and mps_probe.get("available"):
            devices.append({
                "id": "mps",
                "label": "GPU — Apple Metal (MPS)",
                "kind": "mps",
                "name": "Apple Metal Performance Shaders",
                "available": True,
            })

        hf_info = {
            "package_available": importlib.util.find_spec("huggingface_hub") is not None,
            "token_found": bool(
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            ),
        }

        return {
            "devices": devices,
            "backends": ["auto", "native", "pytorch"],
            "execution_modes": ["eager", "compiled"],
            "compile_modes": ["default", "reduce-overhead", "max-autotune"],
            "precisions": ["auto", "fp32", "fp16", "bf16"],
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "cuda_probe_error": (cuda_probe or {}).get("error"),
            "mps_available": bool((mps_probe or {}).get("available")),
            "mps_probe_error": (mps_probe or {}).get("error"),
            "huggingface": hf_info,
        }

    def to_dict(self):
        return json.loads(json.dumps(self.state))

    def project_trust_info(self):
        features = project_executable_features(self.state)
        return {
            "trusted": bool(self._project_trusted),
            "origin": self._project_trust_origin,
            "executable_features": features,
            "requires_trust": bool(features),
        }

    def trust_project(self, trusted=True):
        """Explicitly allow or block executable project features for this session.

        The decision is intentionally not saved into the project file, so a project
        downloaded or reopened later must be reviewed/trusted again.
        """
        self._project_trusted = bool(trusted)
        self._project_trust_origin = "explicit-user-trust" if self._project_trusted else "explicitly-untrusted"
        return self.project_trust_info()

    def untrust_project(self):
        return self.trust_project(False)

    def _mark_external_project_untrusted(self, origin):
        self._project_trusted = False
        self._project_trust_origin = str(origin or "external-project")
        self._unsafe_legacy_checkpoints.clear()

    def _runtime_with_project_trust(self, config, *, checkpoint_path=None):
        runtime = dict(config or {})
        runtime["allow_user_code"] = bool(self._project_trusted)
        if checkpoint_path:
            try:
                resolved = str(Path(checkpoint_path).expanduser().resolve())
            except Exception:
                resolved = str(checkpoint_path)
            runtime["allow_unsafe_legacy_checkpoint"] = resolved in self._unsafe_legacy_checkpoints
        return runtime

    def save(self, path):
        path = Path(path)
        if path.suffix != ".mlbricks":
            path = path.with_suffix(".mlbricks")
        path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        return path

    def load(self, path):
        self.state = json.loads(Path(path).read_text(encoding="utf-8"))
        self._mark_external_project_untrusted("local-project-file")
        return self

    def component_api(self, component_type=None, *, ensure_import=False):
        if component_type is None:
            return self.mlbricks_api
        component_type = str(component_type)
        if ensure_import:
            self.ensure_component_import(component_type)
        return self.mlbricks_api.get(component_type)

    def ensure_component_import(self, component_type):
        """Resolve one MLBricks component through the lazy import pool.

        The canonical submodule is attempted first (for example
        ``mlbricks.components.Embedding``), with the compact top-level export
        kept only as a compatibility fallback.  Successful imports are cached.
        """
        component_type = str(component_type or "").strip()
        if not component_type:
            raise ValueError("component_type is required")
        if not self.import_pool.is_known_component(component_type):
            return {
                "component_type": component_type,
                "ok": True,
                "builder_only": True,
                "imported_now": False,
                "cached": False,
                "message": "Builder utility component; no MLBricks import required.",
            }
        status = self.import_pool.ensure_component(component_type)
        if status.get("ok"):
            refreshed = refresh_component_api(component_type)
            if refreshed is not None:
                self.mlbricks_api[component_type] = refreshed
                status["api"] = copy.deepcopy(refreshed)
            status["message"] = (
                f"{component_type} ready from {status.get('resolved_from')}."
                if status.get("resolved_from")
                else f"{component_type} import ready."
            )
        else:
            status["message"] = status.get("error") or f"Could not import {component_type}."
        return status

    def ensure_external_import(self, import_path, *, label=None):
        """Resolve a user-bound custom component API through the shared import pool."""
        import_path = str(import_path or "").strip()
        if not import_path:
            raise ValueError("import_path is required")
        if not self._project_trusted:
            raise PermissionError(
                "This project is untrusted. External Python imports are blocked until "
                "you review the project and call builder.trust_project()."
            )
        result = self.import_pool.ensure_external(import_path, label=label)
        result["message"] = (
            f"{label or import_path} ready from {import_path}."
            if result.get("ok")
            else result.get("error") or f"Could not import {import_path}."
        )
        return result

    @staticmethod
    def _user_source_dependencies(source):
        """Return top-level imported package names and whether they are available.

        This only inspects imports.  It never installs dependencies and does not
        import third-party packages as part of validation.
        """
        try:
            tree = ast.parse(str(source or ""), mode="exec")
        except SyntaxError:
            return []
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = str(alias.name or "").split(".", 1)[0]
                    if root:
                        names.add(root)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = str(node.module).split(".", 1)[0]
                if root:
                    names.add(root)
        result = []
        for name in sorted(names):
            try:
                available = importlib.util.find_spec(name) is not None
            except Exception:
                available = False
            result.append({"name": name, "available": bool(available)})
        return result

    def validate_user_function(self, source, function_name, *, label=None):
        """Validate cached User Function source without executing its body."""
        source = str(source or "")
        function_name = str(function_name or "").strip()
        if not source.strip():
            return {"ok": False, "error": "Python source is empty.", "message": "Python source is empty."}
        if not function_name:
            return {"ok": False, "error": "Function name is required.", "message": "Function name is required."}
        try:
            tree = ast.parse(source, filename=f"<MLB Studio:{label or function_name}>", mode="exec")
        except SyntaxError as exc:
            msg = f"Syntax error on line {exc.lineno}: {exc.msg}"
            return {"ok": False, "error": msg, "message": msg}
        functions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if function_name not in functions:
            msg = f"No top-level function named {function_name!r} was found."
            return {"ok": False, "error": msg, "message": msg, "functions": sorted(functions)}
        function_node = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
        )
        positional_nodes = [*function_node.args.posonlyargs, *function_node.args.args]
        positional_default_start = len(positional_nodes) - len(function_node.args.defaults)
        signature_parameters = []
        for index, arg in enumerate(positional_nodes):
            signature_parameters.append({
                "name": arg.arg,
                "kind": "positional_only" if index < len(function_node.args.posonlyargs) else "positional_or_keyword",
                "required": index < positional_default_start,
            })
        if function_node.args.vararg is not None:
            signature_parameters.append({
                "name": function_node.args.vararg.arg,
                "kind": "var_positional",
                "required": False,
            })
        for arg, default in zip(function_node.args.kwonlyargs, function_node.args.kw_defaults):
            signature_parameters.append({
                "name": arg.arg,
                "kind": "keyword_only",
                "required": default is None,
            })
        if function_node.args.kwarg is not None:
            signature_parameters.append({
                "name": function_node.args.kwarg.arg,
                "kind": "var_keyword",
                "required": False,
            })
        dependencies = self._user_source_dependencies(source)
        missing = [item["name"] for item in dependencies if not item["available"]]
        message = f"User Function {function_name} is syntactically valid."
        if missing:
            message += " Missing dependencies must be installed explicitly: " + ", ".join(missing) + "."
        else:
            message += " Dependencies are available and the function is ready to build."
        return {
            "ok": not bool(missing),
            "function_name": function_name,
            "functions": sorted(functions),
            "signature": {
                "name": function_name,
                "parameters": signature_parameters,
            },
            "dependencies": dependencies,
            "missing_dependencies": missing,
            "message": message,
        }

    def validate_user_class(self, source, class_name, *, label=None):
        """Validate cached User Class source and its external dependencies."""
        source = str(source or "")
        class_name = str(class_name or "").strip()
        if not source.strip():
            return {"ok": False, "error": "Python source is empty.", "message": "Python source is empty."}
        if not class_name:
            return {"ok": False, "error": "Class name is required.", "message": "Class name is required."}
        try:
            tree = ast.parse(source, filename=f"<MLB Studio:{label or class_name}>", mode="exec")
        except SyntaxError as exc:
            msg = f"Syntax error on line {exc.lineno}: {exc.msg}"
            return {"ok": False, "error": msg, "message": msg}
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if class_name not in classes:
            msg = f"No top-level class named {class_name!r} was found."
            return {"ok": False, "error": msg, "message": msg, "classes": sorted(classes)}
        dependencies = self._user_source_dependencies(source)
        missing = [item["name"] for item in dependencies if not item["available"]]
        message = f"User Class {class_name} is syntactically valid."
        if missing:
            message += " Missing dependencies must be installed explicitly: " + ", ".join(missing) + "."
        else:
            message += " Dependencies are available and the class is ready to build."
        return {
            "ok": not bool(missing),
            "class_name": class_name,
            "classes": sorted(classes),
            "dependencies": dependencies,
            "missing_dependencies": missing,
            "message": message,
        }

    def validate_component_imports(self, *, eager=True):
        """Validate that every MLBricks-backed catalog component has an import route.

        With ``eager=True`` (default) this actually resolves every registered
        component once and returns the canonical/fallback route used.  Builder
        utilities such as Dropout and Text Input are reported as not requiring
        an MLBricks import.
        """
        report = []
        for item in self.catalog:
            component_type = str(item.get("type") or "")
            if item.get("builder_utility"):
                report.append({
                    "component_type": component_type,
                    "name": item.get("name"),
                    "ok": True,
                    "builder_only": True,
                    "resolved_from": None,
                    "error": None,
                })
                continue
            if component_type == "stateaware_esa_stack":
                deps = ["esa", "rmsnorm", "saffn", "rescontroller"]
                statuses = [self.import_pool.ensure_component(dep) for dep in deps] if eager else []
                errors = [status.get("error") for status in statuses if not status.get("ok")]
                report.append({
                    "component_type": component_type,
                    "name": item.get("name"),
                    "ok": not errors,
                    "compound": True,
                    "dependencies": deps,
                    "resolved_from": [status.get("resolved_from") for status in statuses if status.get("resolved_from")],
                    "error": "; ".join(errors) if errors else None,
                })
                continue
            if not self.import_pool.is_known_component(component_type):
                report.append({
                    "component_type": component_type,
                    "name": item.get("name"),
                    "ok": False,
                    "resolved_from": None,
                    "error": "No import-pool route registered.",
                })
                continue
            status = self.import_pool.ensure_component(component_type) if eager else self.import_pool.import_info(component_type)
            report.append({
                "component_type": component_type,
                "name": item.get("name"),
                "ok": bool(status.get("ok", status.get("known", False))),
                "resolved_from": status.get("resolved_from") or status.get("canonical_path"),
                "error": status.get("error"),
            })
        return {
            "ok": all(item.get("ok") for item in report),
            "components": report,
            "failures": [item for item in report if not item.get("ok")],
            "import_pool": self.import_pool.status(),
        }

    def _prepared_output_node(self):
        workspaces = self.state.get("workspaces") or {}
        data_ws = workspaces.get("data") or {}
        component = (self.state.get("components") or {}).get(data_ws.get("root_component_id"), {})
        for node in component.get("nodes") or []:
            if node.get("type") == "prepared_dataset":
                return node
        return None

    @staticmethod
    def _split_summary(value):
        target = value
        # DataLoader-like objects expose their underlying dataset.
        if not hasattr(target, "column_names") and hasattr(target, "dataset"):
            target = target.dataset
        try:
            rows = len(target)
        except Exception:
            rows = None
        columns = list(getattr(target, "column_names", []) or [])
        return {"rows": rows, "columns": columns}

    def _summarize_prepared_result(self, result):
        # DatasetDict is mapping-like and also exposes column_names. Detecting
        # it as Mapping prevents the old "Train = 3" split-count bug.
        if isinstance(result, Mapping):
            splits = {
                str(name): self._split_summary(split)
                for name, split in result.items()
            }
        else:
            splits = {"train": self._split_summary(result)}

        total_rows = 0
        known_total = True
        for info in splits.values():
            rows = info.get("rows")
            if rows is None:
                known_total = False
            else:
                total_rows += int(rows)

        default_split = "train" if "train" in splits else next(iter(splits), None)
        default_columns = list((splits.get(default_split) or {}).get("columns") or []) if default_split else []
        has_input_ids = "input_ids" in default_columns
        return {
            "splits": splits,
            "total_rows": total_rows if known_total else None,
            "default_split": default_split,
            "capabilities": {
                "token_stream": has_input_ids,
                "runtime_context_repack": has_input_ids,
            },
        }

    def _data_pipeline_snapshot(self):
        """Snapshot source, processing, split and tokenizer settings."""
        workspaces = self.state.get("workspaces") or {}
        data_ws = workspaces.get("data") or {}
        component = (self.state.get("components") or {}).get(
            data_ws.get("root_component_id"), {}
        )
        snapshot = {
            "steps": [], "source": None, "text_processing": None,
            "split": None, "tokenizer": None, "image_processing": None,
            "audio_processing": None, "signal_processing": None, "batch": None, "output": None,
        }
        source_types = {"demo_dataset", "manual_dataset", "hf_dataset", "kaggle_dataset", "url_dataset", "local_dataset"}
        for node in component.get("nodes") or []:
            params = json.loads(json.dumps(node.get("params") or {}))
            snapshot["steps"].append({"id":node.get("id"),"type":node.get("type"),"name":node.get("name"),"params":params})
            value = {"type":node.get("type"),"name":node.get("name"),**params}
            t=node.get("type")
            if t in source_types: snapshot["source"] = value
            elif t=="text_process": snapshot["text_processing"] = value
            elif t=="train_test_split": snapshot["split"] = value
            elif t=="tokenize_text": snapshot["tokenizer"] = value
            elif t=="image_process": snapshot["image_processing"] = value
            elif t=="audio_process": snapshot["audio_processing"] = value
            elif t=="signal_process": snapshot["signal_processing"] = value
            elif t=="batch_data": snapshot["batch"] = value
            elif t=="prepared_dataset": snapshot["output"] = value
        return snapshot

    @staticmethod
    def _artifact_name_key(value):
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    @staticmethod
    def _artifact_slug(value):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("._-")
        return (slug or "prepared-dataset").lower()

    @staticmethod
    def _model_artifact_slug(value):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "model")).strip("-.")
        return slug or "model"

    def _managed_models_root(self):
        return Path((self.local_environment.get("paths") or {}).get("models") or "mlbricks_workspace/models").expanduser()

    def _resolved_model_output_root(self, config):
        managed = self._managed_models_root()
        raw = str((config or {}).get("output_dir") or "").strip()
        legacy = {"", "mlbricks_workspace/models", "mlbricks/models"}
        if raw.replace("\\", "/") in legacy:
            return managed
        path = Path(raw).expanduser()
        # Legacy hosted-runtime defaults should follow the current Studio
        # workspace instead of becoming a second model repository.
        if raw.startswith("/kaggle/") and str(managed).startswith("/kaggle/"):
            return managed
        return path

    def _fast_existing_model_training_artifact(self, output_path, entry):
        """Metadata-only model collision check used before importing torch/data.

        The managed model index is created during Builder startup from only the
        dedicated models directory. It never opens model.pt. For imported or
        custom output paths we inspect only the exact path referenced by the
        model entry; Studio never recursively searches unrelated directories.
        """
        self._ensure_managed_artifact_index_current()
        output_path = Path(output_path).expanduser()
        try:
            output_resolved = str(output_path.resolve())
        except Exception:
            output_resolved = str(output_path)
        wanted_name = self._artifact_name_key((entry or {}).get("name"))

        for record in self._indexed_model_records():
            try:
                record_path = str(Path(record.get("path") or "").expanduser().resolve())
            except Exception:
                record_path = str(record.get("path") or "")
            if record_path != output_resolved and self._artifact_name_key(record.get("name")) != wanted_name:
                continue
            resumable = bool(record.get("resumable"))
            return {
                "present": True,
                "resumable": resumable,
                "path": str(record.get("artifact_path") or "") or None,
                "kind": record.get("artifact_kind"),
                "resume_mode": record.get("resume_mode"),
                "metadata": copy.deepcopy(record.get("metadata") or {}),
                "reason": record.get("reason") or (None if resumable else "the indexed model artifact is incomplete"),
                "indexed": True,
            }

        # If this exact directory was created after Builder startup, recognize it
        # without scanning any parent tree. This is O(1) for the requested model.
        if output_path.exists() or output_path.is_symlink():
            last = output_path / "last"
            if (last / "model.pt").is_file() and (last / "metadata.json").is_file():
                payload = None
                try:
                    payload = json.loads((last / "metadata.json").read_text(encoding="utf-8"))
                except Exception:
                    payload = None
                meta = (payload or {}).get("metadata") or {}
                if (payload or {}).get("format") in {None, "mlbricks.model"}:
                    return {
                        "present": True, "resumable": True, "path": str(last),
                        "kind": str(meta.get("kind") or "trained_model"), "resume_mode": "weights",
                        "metadata": copy.deepcopy(meta), "reason": None, "indexed": False,
                    }
            checkpoint_root = output_path / "checkpoints"
            if checkpoint_root.is_dir():
                try:
                    for candidate in sorted(
                        (x for x in checkpoint_root.iterdir() if x.is_dir()),
                        key=lambda x: x.name, reverse=True,
                    ):
                        if (candidate / "model.pt").is_file() and (candidate / "metadata.json").is_file():
                            return {
                                "present": True, "resumable": True, "path": str(candidate),
                                "kind": "training_checkpoint", "resume_mode": "checkpoint",
                                "metadata": {}, "reason": None, "indexed": False,
                            }
                except OSError:
                    pass
            return {
                "present": True, "resumable": False, "path": None, "kind": None,
                "resume_mode": None, "metadata": {},
                "reason": "no complete trained model marker was found", "indexed": False,
            }

        # Exact external/imported model references are allowed without scanning
        # their surrounding directory.
        for key in ("checkpoint_path", "path"):
            value = (entry or {}).get(key)
            if not value:
                continue
            candidate = Path(str(value)).expanduser()
            if not candidate.exists() and not candidate.is_symlink():
                continue
            if candidate.is_dir() and (candidate / "model.pt").is_file() and (candidate / "metadata.json").is_file():
                return {
                    "present": True, "resumable": True, "path": str(candidate),
                    "kind": "trained_model", "resume_mode": "weights",
                    "metadata": {}, "reason": None, "indexed": False,
                }
            if candidate.is_file():
                # Legacy checkpoint validity is confirmed only after Retrain is
                # approved; existence alone is enough for the immediate prompt.
                return {
                    "present": True, "resumable": True, "path": str(candidate),
                    "kind": "legacy_checkpoint", "resume_mode": "checkpoint",
                    "metadata": {}, "reason": None, "indexed": False,
                }

        return {
            "present": False, "resumable": False, "path": None, "kind": None,
            "resume_mode": None, "metadata": {}, "reason": None, "indexed": False,
        }

    def _disk_prepared_dataset_records(self):
        """Return cached metadata for datasets in Studio's dedicated data root.

        The directory was shallow-indexed during Builder startup. This method
        intentionally does not recurse through the filesystem and never loads
        Arrow/Parquet dataset bodies into RAM.
        """
        self._ensure_managed_artifact_index_current()
        records = []
        for item in self._indexed_data_records():
            marker = copy.deepcopy(item.get("metadata") or {})
            records.append({
                "id": item.get("id"),
                "name": str(marker.get("name") or item.get("name") or Path(str(item.get("path") or "dataset")).name),
                "path": str(item.get("path") or ""),
                "status": item.get("status"),
                "metadata": marker,
            })
        return records

    def _disk_prepared_dataset_names(self):
        return {
            self._artifact_name_key(item.get("name"))
            for item in self._disk_prepared_dataset_records()
            if self._artifact_name_key(item.get("name"))
        }

    @staticmethod
    def _remove_local_artifact_path(path):
        target = Path(path)
        if not target.exists() and not target.is_symlink():
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    def _stage_local_directory_override(self, path):
        """Move an existing artifact aside so replacement is rollback-safe."""
        target = Path(path).expanduser()
        if not target.exists() and not target.is_symlink():
            return None
        backup = target.with_name(target.name + f".mlb-override-backup-{uuid.uuid4().hex[:10]}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(backup))
        return {"target": str(target), "backup": str(backup)}

    def _defer_local_artifact_cleanup(self, paths):
        """Delete obsolete local artifacts without blocking the runtime UI.

        Model/dataset overrides can leave a large previous artifact behind after
        the new canonical directory has already been switched into place. On
        notebook filesystems, recursively deleting checkpoints or Arrow shards
        can take seconds or minutes. Finalization must not remain at 99% while
        that best-effort housekeeping runs, so cleanup happens in a daemon thread.
        """
        targets = [Path(value) for value in (paths or []) if value]
        if not targets:
            return None

        def cleanup():
            for target in targets:
                try:
                    if target.exists() or target.is_symlink():
                        self._remove_local_artifact_path(target)
                except Exception:
                    # The canonical artifact has already been committed. A stale
                    # .mlb-* backup is recoverable and must never turn a successful
                    # train/fetch into a failed or permanently-running UI state.
                    pass

        worker = threading.Thread(
            target=cleanup,
            name=f"mlbricks-artifact-cleanup-{uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        worker.start()
        return worker

    def _commit_local_directory_overrides(self, staged, *, defer_cleanup=False):
        backups = [
            Path(item["backup"])
            for item in (staged or [])
            if item and item.get("backup")
        ]
        if defer_cleanup:
            return self._defer_local_artifact_cleanup(backups)
        for backup in backups:
            if backup.exists() or backup.is_symlink():
                self._remove_local_artifact_path(backup)
        return None

    def _rollback_local_directory_overrides(self, staged):
        for item in reversed(staged or []):
            target = Path(item["target"])
            backup = Path(item["backup"])
            try:
                if target.exists() or target.is_symlink():
                    self._remove_local_artifact_path(target)
                if backup.exists() or backup.is_symlink():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(backup), str(target))
            except Exception:
                # Never hide the original training/fetch failure with rollback
                # cleanup. A leftover *.mlb-override-backup-* remains recoverable.
                pass

    def _preflight_prepared_dataset_output(self, *, overwrite_existing=False):
        """Validate dataset identity and return collision/replacement metadata.

        A duplicate is never silently replaced. Without explicit confirmation an
        ``ArtifactConflictError`` is raised. With confirmation, the caller can
        stage the old directory and replace it transactionally.
        """
        node = self._prepared_output_node() or {}
        params = node.setdefault("params", {})
        requested_name = str(params.get("dataset_name") or "Prepared Dataset").strip() or "Prepared Dataset"
        name_key = self._artifact_name_key(requested_name)

        registry_matches = [
            item for item in self.state.setdefault("prepared_datasets", [])
            if self._artifact_name_key(item.get("name")) == name_key
        ]
        disk_matches = [
            item for item in self._disk_prepared_dataset_records()
            if self._artifact_name_key(item.get("name")) == name_key
        ]

        save_to_disk = str(params.get("save_to_disk", "false")).lower() == "true"
        target = None
        if save_to_disk:
            data_root = Path((self.local_environment.get("paths") or {}).get("data") or "mlbricks_workspace/data").expanduser()
            configured = Path(str(params.get("path") or (data_root / "prepared_dataset"))).expanduser()
            legacy_defaults = {
                Path("mlbricks_workspace/data/prepared_dataset"),
                Path("mlbricks/data/prepared_dataset"),
            }
            try:
                configured_resolved = configured.resolve()
                placeholder_resolved = (data_root / "prepared_dataset").resolve()
                use_managed_name = configured_resolved == placeholder_resolved
            except Exception:
                use_managed_name = False
            if configured in legacy_defaults:
                use_managed_name = True
            target = (data_root / self._artifact_slug(requested_name) if use_managed_name else configured).expanduser()
            params["path"] = str(target)

        conflicting_paths = []
        for item in registry_matches:
            if item.get("path"):
                conflicting_paths.append(str(Path(item["path"]).expanduser()))
        for item in disk_matches:
            if item.get("path"):
                conflicting_paths.append(str(Path(item["path"]).expanduser()))
        if target is not None and (target.exists() or target.is_symlink()):
            conflicting_paths.append(str(target))

        # Also catch a target path owned by another registered dataset name.
        if target is not None:
            try:
                target_resolved = str(target.resolve())
            except Exception:
                target_resolved = str(target)
            for item in self.state.get("prepared_datasets") or []:
                value = item.get("path")
                if not value:
                    continue
                try:
                    existing_resolved = str(Path(value).expanduser().resolve())
                except Exception:
                    existing_resolved = str(value)
                if existing_resolved == target_resolved and item not in registry_matches:
                    registry_matches.append(item)
                    conflicting_paths.append(str(Path(value).expanduser()))

        # Preserve order but deduplicate equivalent relative/absolute paths.
        normalized_paths = []
        seen_paths = set()
        for value in conflicting_paths:
            try:
                key = str(Path(value).expanduser().resolve())
            except Exception:
                key = str(Path(value).expanduser())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            normalized_paths.append(key)
        conflicting_paths = normalized_paths
        has_conflict = bool(registry_matches or disk_matches or conflicting_paths)
        if has_conflict and not overwrite_existing:
            if disk_matches:
                collision_text = (
                    f'Prepared dataset "{requested_name}" already exists in the local Studio data directory'
                    + (f' at {conflicting_paths[0]}' if conflicting_paths else '')
                    + '. Choose a different Dataset Name or Save Path, or confirm Override to replace it.'
                )
            else:
                collision_text = (
                    f'Prepared dataset "{requested_name}" already exists'
                    + (f' at {conflicting_paths[0]}' if conflicting_paths else '')
                    + '. Choose a different Dataset Name or Save Path, or confirm Override to replace it.'
                )
            raise ArtifactConflictError(
                kind="dataset",
                name=requested_name,
                paths=conflicting_paths,
                message=collision_text,
            )

        replacement = registry_matches[0] if registry_matches else None
        return {
            "name": requested_name,
            "path": str(target) if target is not None else None,
            "replace_metadata": replacement,
            "registry_conflicts": registry_matches,
            "disk_conflicts": disk_matches,
            "conflicting_paths": conflicting_paths,
        }

    def _register_prepared_dataset(self, result, *, overwrite_existing=False, preflight=None):
        node = self._prepared_output_node() or {}
        params = node.get("params") or {}
        requested_name = str(params.get("dataset_name") or "Prepared Dataset").strip() or "Prepared Dataset"
        name_key = self._artifact_name_key(requested_name)
        existing = [
            item for item in self.state.setdefault("prepared_datasets", [])
            if self._artifact_name_key(item.get("name")) == name_key
        ]
        if existing and not overwrite_existing:
            raise ArtifactConflictError(kind="dataset", name=requested_name, paths=[x.get("path") for x in existing if x.get("path")])

        replacement = (preflight or {}).get("replace_metadata") or (existing[0] if existing else None)
        dataset_id = str((replacement or {}).get("id") or f"dataset_{uuid.uuid4().hex[:12]}")
        summary = self._summarize_prepared_result(result)
        save_to_disk = str(params.get("save_to_disk", "false")).lower() == "true"
        path = str(params.get("path") or "") if save_to_disk else None

        metadata = {
            "id": dataset_id,
            "name": requested_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_node_id": node.get("id"),
            "storage": "disk+memory" if save_to_disk else "memory",
            "path": path,
            "pipeline": self._data_pipeline_snapshot(),
            **summary,
        }

        if save_to_disk and path:
            try:
                marker = Path(path) / "mlbricks_dataset.json"
                marker.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(f"Dataset was prepared but Studio could not write its metadata marker: {exc}") from exc

        # Replace conflicting registry entries only after the new dataset has
        # completed successfully. Keeping the same id preserves model selections.
        conflicts = (preflight or {}).get("registry_conflicts") or existing
        conflict_ids = {str(x.get("id")) for x in conflicts if x.get("id") and str(x.get("id")) != dataset_id}
        next_items = []
        replaced = False
        for item in self.state.setdefault("prepared_datasets", []):
            iid = str(item.get("id") or "")
            if iid == dataset_id:
                if not replaced:
                    next_items.append(metadata)
                    replaced = True
                continue
            if iid in conflict_ids:
                self.prepared_datasets.pop(iid, None)
                continue
            next_items.append(item)
        if not replaced:
            next_items.append(metadata)
        self.state["prepared_datasets"] = next_items
        self.prepared_datasets[dataset_id] = result
        self.state.setdefault("project", {})["dataset"] = requested_name
        if save_to_disk and path:
            # Keep the shallow data index current so subsequent Fetch clicks can
            # answer existence checks instantly without another directory scan.
            self._refresh_managed_artifact_index()
        return metadata

    def run_data_pipeline(self, progress_callback=None, *, overwrite_existing=False):
        """Execute Data Processing with explicit, rollback-safe overwrite support."""
        self._stop_event.clear()
        self.last_run_error = None
        last_progress = {}
        staged = []

        def relay(payload):
            enriched = dict(payload or {})
            enriched.setdefault("runtime_kind", "data")
            last_progress.clear()
            last_progress.update(enriched)
            if progress_callback:
                progress_callback(enriched)

        try:
            preflight = self._preflight_prepared_dataset_output(overwrite_existing=overwrite_existing)
            if overwrite_existing:
                # Deepest paths first prevents a rare parent/child collision from
                # invalidating the second path while backups are staged.
                candidates = [Path(value).expanduser() for value in (preflight.get("conflicting_paths") or [])]
                candidates.sort(key=lambda value: len(value.parts), reverse=True)
                for path in candidates:
                    if path.exists() or path.is_symlink():
                        staged_item = self._stage_local_directory_override(path)
                        if staged_item:
                            staged.append(staged_item)

            self.last_data_result = execute_data_pipeline(
                self.state,
                progress_callback=relay,
                stop_event=self._stop_event,
                credential_resolver=lambda provider, name: self.persistence.get_credentials(provider, name),
            )
            metadata = self._register_prepared_dataset(
                self.last_data_result,
                overwrite_existing=overwrite_existing,
                preflight=preflight,
            )

            final_payload = dict(last_progress or {})
            final_payload.update({
                "status": "done",
                "runtime_kind": "data",
                "overall": 100,
                "message": f'Data ready: {metadata["name"]}',
                "prepared_dataset": metadata,
                "available_datasets": self.available_datasets(),
            })
            if progress_callback:
                progress_callback(final_payload)

            # The replacement is already live. Removing a previous multi-shard
            # dataset is housekeeping and must not hold Data Fetch at 99/100%.
            if staged:
                self._commit_local_directory_overrides(staged, defer_cleanup=True)
                staged = []
            return self.last_data_result
        except Exception as exc:
            if staged:
                self._rollback_local_directory_overrides(staged)
            self._remember_run_error(exc)
            raise

    def available_datasets(self):
        """Return serializable metadata for every prepared dataset in this project."""
        return json.loads(json.dumps(self.state.get("prepared_datasets") or []))

    def get_prepared_dataset(self, dataset_id_or_name, split=None):
        """Return a prepared Dataset/DatasetDict by registry id or display name.

        Managed datasets are indexed metadata-first.  If browser state contains a
        stale design entry after a kernel restart, reconcile it against Studio's
        dedicated data index before deciding that the rows are unavailable.
        """
        wanted = str(dataset_id_or_name)

        def find_metadata():
            for item in self.state.get("prepared_datasets") or []:
                if item.get("id") == wanted or str(item.get("name", "")).lower() == wanted.lower():
                    return item
            return None

        metadata = find_metadata()
        if metadata is None or not metadata.get("path"):
            # Cheap path: this loads the small JSON index and only shallow-rescans
            # Studio's managed data directory if its child signature changed.
            self._ensure_managed_artifact_index_current()
            self._hydrate_indexed_prepared_datasets()
            metadata = find_metadata()

        if metadata is None:
            raise KeyError(f"Prepared dataset not found: {dataset_id_or_name!r}")

        dataset_id = metadata["id"]
        result = self.prepared_datasets.get(dataset_id)

        path_value = metadata.get("path")
        if result is None and path_value:
            path_obj = Path(path_value).expanduser()
            if not path_obj.exists():
                # The design may point to an old managed path. Refresh the small
                # index once and let it repair the path by id/name before failing.
                self._ensure_managed_artifact_index_current()
                self._hydrate_indexed_prepared_datasets()
                metadata = find_metadata() or metadata
                path_value = metadata.get("path")

            if path_value:
                try:
                    from datasets import load_from_disk
                    result = load_from_disk(path_value)
                    self.prepared_datasets[dataset_id] = result
                    metadata["indexed_only"] = False
                    metadata["storage"] = "disk+memory"
                except Exception as exc:
                    raise RuntimeError(
                        f'{metadata["name"]!r} is indexed on disk but could not be loaded '
                        f'from {path_value!r}: {exc}'
                    ) from exc

        if result is None:
            raise RuntimeError(
                f'{metadata["name"]!r} is a session-only dataset and its rows are not '
                "available in this Python session. Re-run its Data Processing pipeline "
                "and enable Save To Disk if you want it to survive a kernel restart."
            )

        if split:
            try:
                return result[split]
            except Exception as exc:
                available = list((metadata.get("splits") or {}).keys())
                raise KeyError(
                    f"Split {split!r} is unavailable. Available splits: {available}"
                ) from exc
        return result

    def validate_data_pipeline(self):
        """Return (ordered_nodes, errors) for the current Data Processing graph."""
        return validate_data_pipeline(self.state)


    def _model_output(self, model_id):
        for item in self.state.get("model_outputs") or []:
            if item.get("id") == model_id:
                return item
        raise KeyError(f"Built model not found: {model_id!r}")

    def _dataset_meta(self, dataset_id):
        for item in self.state.get("prepared_datasets") or []:
            if item.get("id") == dataset_id:
                return item
        raise KeyError(f"Prepared dataset metadata not found: {dataset_id!r}")

    def _remember_run_error(self, exc):
        """Remember an error without retaining its traceback/GPU object graph.

        Keeping the original exception object in ``last_run_error`` retains every
        traceback frame. A failed training/compile frame can own the model,
        optimizer, warm-up batches, and CUDA tensors, which prevents them from
        being reclaimed and makes the next retrain/delete fail unpredictably.
        Store only a detached diagnostic exception instead.
        """
        try:
            tb = getattr(exc, "__traceback__", None)
            if tb is not None:
                try:
                    traceback.clear_frames(tb)
                except Exception:
                    pass
            try:
                exc.__traceback__ = None
            except Exception:
                pass
        finally:
            self.last_run_error = RuntimeError(f"{type(exc).__name__}: {exc}")
        return self.last_run_error

    @staticmethod
    def _reset_torch_compiler(torch):
        """Best-effort reset of torch.compile/Dynamo state between model lives."""
        reset_done = False
        try:
            compiler = getattr(torch, "compiler", None)
            reset = getattr(compiler, "reset", None)
            if callable(reset):
                reset()
                reset_done = True
        except Exception:
            pass
        if not reset_done:
            try:
                dynamo = getattr(torch, "_dynamo", None)
                reset = getattr(dynamo, "reset", None)
                if callable(reset):
                    reset()
                    reset_done = True
            except Exception:
                pass
        return reset_done

    def _cleanup_failed_runtime(self, *, reset_compiler=False):
        """Drop traceback-held references and release allocator/compiler caches."""
        import gc
        gc.collect()
        try:
            import torch
            if reset_compiler:
                self._reset_torch_compiler(torch)
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                    if callable(ipc_collect):
                        ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
        gc.collect()

    def _release_cached_runtime_models(self, *, preserve_server_models=True, reset_compiler=False):
        """Release cached compiled model objects and return memory cleanup details.

        Saved model artifacts/checkpoints are left untouched. Live API servers keep
        their model references unless the caller explicitly stops/deletes them.
        """
        import gc

        torch = None
        cuda_available = False
        before_allocated = before_reserved = None
        try:
            import torch as _torch
            torch = _torch
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                before_allocated = float(torch.cuda.memory_allocated()) / (1024 ** 3)
                before_reserved = float(torch.cuda.memory_reserved()) / (1024 ** 3)
        except Exception:
            torch = None

        preserved = set(self._model_servers) if preserve_server_models else set()
        removed = []
        for cached_id in list(self.trained_models):
            if cached_id in preserved:
                continue
            self.trained_models.pop(cached_id, None)
            removed.append(cached_id)

        gc.collect()

        compiler_reset = False
        if torch is not None and reset_compiler:
            compiler_reset = self._reset_torch_compiler(torch)

        after_allocated = after_reserved = None
        if torch is not None and cuda_available:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
            except Exception:
                pass
            gc.collect()
            try:
                after_allocated = float(torch.cuda.memory_allocated()) / (1024 ** 3)
                after_reserved = float(torch.cuda.memory_reserved()) / (1024 ** 3)
            except Exception:
                pass

        return {
            "cuda_available": cuda_available,
            "released_model_caches": len(removed),
            "released_model_ids": removed,
            "preserved_server_models": sorted(preserved),
            "before_allocated_gb": before_allocated,
            "before_reserved_gb": before_reserved,
            "after_allocated_gb": after_allocated,
            "after_reserved_gb": after_reserved,
            "compiler_cache_reset": compiler_reset,
        }

    def clear_runtime_memory(self):
        """Release inactive model runtimes and empty CUDA allocator/compiler caches."""
        return self._release_cached_runtime_models(
            preserve_server_models=True, reset_compiler=True
        )

    def delete_prepared_dataset(self, dataset_id):
        """Remove a prepared dataset from Studio without deleting external files."""
        import gc

        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            raise ValueError("Dataset id is required.")
        metadata = None
        for item in self.state.get("prepared_datasets") or []:
            if str(item.get("id")) == dataset_id:
                metadata = item
                break
        self.state["prepared_datasets"] = [
            item for item in (self.state.get("prepared_datasets") or [])
            if str(item.get("id")) != dataset_id
        ]
        removed_object = self.prepared_datasets.pop(dataset_id, None) is not None

        for component in (self.state.get("components") or {}).values():
            for node in component.get("nodes") or []:
                params = node.get("params") or {}
                if node.get("type") == "text_input" and str(params.get("dataset_id") or "") == dataset_id:
                    params["input_mode"] = "prompt"
                    params["dataset_id"] = ""
                    params["dataset_split"] = "train"
        for entry in self.state.get("model_outputs") or []:
            if str(entry.get("selected_dataset_id") or "") == dataset_id:
                entry["selected_dataset_id"] = None
                entry["dataset"] = None

        project = self.state.setdefault("project", {})
        if metadata and project.get("dataset") == metadata.get("name"):
            project["dataset"] = None
        gc.collect()
        return {
            "dataset_id": dataset_id,
            "name": (metadata or {}).get("name") or dataset_id,
            "removed_from_memory": removed_object,
        }

    def delete_model_output(self, model_id):
        """Remove a model registry entry and its live/cache references, not disk files.

        Deletion is deliberately refused while the same model is actively training
        or generating. Removing the registry/cache underneath a live worker used to
        produce follow-on KeyError/AssertionError failures and could leave compiled
        CUDA graphs alive until the next run.
        """
        import gc

        model_id = str(model_id or "").strip()
        if not model_id:
            raise ValueError("Model id is required.")
        active_same_model = (
            self._active_bridge_action in {"train", "generate"}
            and str(self._active_bridge_model_id or "") == model_id
        )
        hot_thread = self._hot_generation_thread
        hot_same_model = bool(
            hot_thread is not None and hot_thread.is_alive()
            and str(self._hot_generation_model_id or "") == model_id
        )
        if active_same_model or hot_same_model:
            raise RuntimeError(
                "Stop the active training/generation run before deleting this model."
            )

        metadata = None
        for item in self.state.get("model_outputs") or []:
            if str(item.get("id")) == model_id:
                metadata = item
                break

        server = self._model_servers.pop(model_id, None)
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass

        cached_runtime = self.trained_models.pop(model_id, None)
        had_cache = cached_runtime is not None
        compiled_cache = bool(
            cached_runtime
            and getattr((cached_runtime or {}).get("compiled"), "compile_used", False)
        )
        if cached_runtime is not None:
            try:
                cached_runtime.clear()
            except Exception:
                pass
            del cached_runtime

        self.state["model_outputs"] = [
            item for item in (self.state.get("model_outputs") or [])
            if str(item.get("id")) != model_id
        ]
        # A prior failed run must not pin its traceback/model after deletion.
        self.last_run_error = None
        gc.collect()
        try:
            import torch
            if compiled_cache:
                self._reset_torch_compiler(torch)
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return {
            "model_id": model_id,
            "name": (metadata or {}).get("name") or model_id,
            "stopped_server": server is not None,
            "released_runtime_cache": had_cache,
            "compiler_cache_reset": compiled_cache,
        }

    def _existing_model_training_artifact(self, output_path, entry):
        """Inspect existing model storage and find the best reusable weights.

        ``last`` is preferred because it is the completed trained model. If a
        run was interrupted before ``last`` was written, the newest valid
        training checkpoint is resumable as a recovery point. Merely having a
        directory is not enough: a malformed/partial artifact is reported as
        present-but-unusable so the UI can ask for an explicit fresh start.
        """
        output_path = Path(output_path).expanduser()
        candidates = []

        def add_candidate(value):
            if not value:
                return
            try:
                path = Path(value).expanduser()
            except Exception:
                return
            key = str(path)
            if all(str(item) != key for item in candidates):
                candidates.append(path)

        # Completed model first. An entry may point to an imported/external
        # artifact, so include its saved path even when output_dir changed.
        add_candidate(output_path / "last")
        add_candidate(entry.get("checkpoint_path"))
        add_candidate(entry.get("path"))
        add_candidate(output_path)
        add_candidate(output_path / "last.pt")

        checkpoint_root = output_path / "checkpoints"
        if checkpoint_root.is_dir():
            try:
                for path in sorted(checkpoint_root.iterdir(), key=lambda x: x.name, reverse=True):
                    add_candidate(path)
            except OSError:
                pass

        present = bool(output_path.exists() or output_path.is_symlink())
        errors = []
        inspect_api = None
        for candidate in candidates:
            if not candidate.exists() and not candidate.is_symlink():
                continue
            present = True
            if candidate.is_dir():
                model_file = candidate / "model.pt"
                if not model_file.is_file():
                    continue
                try:
                    if model_file.stat().st_size <= 0:
                        raise RuntimeError("model.pt is empty")
                    if inspect_api is None:
                        inspect_api = IMPORT_POOL.resolve_api("lifecycle.inspect")
                    info = inspect_api(candidate)
                    if not isinstance(info, dict) or info.get("format") != "mlbricks.model":
                        raise RuntimeError(f"unsupported artifact format {getattr(info, 'get', lambda *_: None)('format')!r}")
                    metadata = copy.deepcopy(info.get("metadata") or {})
                    kind = str(metadata.get("kind") or "trained_model").lower()
                    if kind == "training_checkpoint":
                        resume_mode = "checkpoint"
                    else:
                        resume_mode = "weights"
                    return {
                        "present": True, "resumable": True, "path": str(candidate),
                        "kind": kind, "resume_mode": resume_mode, "metadata": metadata,
                        "reason": None,
                    }
                except Exception as exc:
                    errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                    continue

            if candidate.is_file():
                try:
                    payload = safe_torch_load(candidate, map_location="cpu", allow_unsafe_pickle=False)
                    if not isinstance(payload, dict) or "model_state" not in payload:
                        raise RuntimeError("model_state was not found")
                    return {
                        "present": True, "resumable": True, "path": str(candidate),
                        "kind": "legacy_checkpoint", "resume_mode": "checkpoint",
                        "metadata": copy.deepcopy(payload.get("metadata") or {}), "reason": None,
                    }
                except Exception as exc:
                    errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

        reason = "; ".join(errors[-3:]) if errors else "no readable trained weights were found"
        return {
            "present": present, "resumable": False, "path": None, "kind": None,
            "resume_mode": None, "metadata": {}, "reason": reason,
        }

    def train_model(
        self, model_id, *, progress_callback=None, overwrite_existing=False,
        resume_existing=False, start_fresh=False,
    ):
        """Train a model without discarding healthy learned parameters.

        Existing *valid* model weights are a retraining source, not an overwrite
        target. Studio asks the user to Retrain and initializes the new run from
        those weights. A fresh model is allowed only when the existing artifact
        is missing/incomplete/corrupted (or an older client explicitly confirms
        replacement), and that destructive path remains rollback-safe.

        Collision detection is deliberately metadata-only and happens before
        importing PyTorch or loading the selected dataset. This keeps the
        Retrain/Start Fresh prompt immediate even for large on-disk datasets.
        """
        entry = self._model_output(model_id)
        config = self._runtime_with_project_trust(entry.get("training_config") or {})
        if model_id in self._model_servers:
            raise RuntimeError("Stop this model's API server before starting training.")

        output_root = self._resolved_model_output_root(config)
        # Canonicalize legacy/default storage into Studio's dedicated model root.
        config["output_dir"] = str(output_root)
        safe_name = self._model_artifact_slug(entry.get("name", "model"))
        output_path = output_root / safe_name

        fast_existing = self._fast_existing_model_training_artifact(output_path, entry)
        try:
            managed_root = self._managed_models_root().resolve()
            resolved_output = output_path.resolve()
            in_managed_root = resolved_output == managed_root or managed_root in resolved_output.parents
        except Exception:
            in_managed_root = False

        # For Studio-managed models the startup index is authoritative for the
        # first prompt. For an explicitly custom/external output path, inspect
        # only that exact artifact (still no recursive filesystem search).
        existing = fast_existing
        if fast_existing.get("present") and not in_managed_root and not fast_existing.get("indexed"):
            existing = self._existing_model_training_artifact(output_path, entry)

        # Backward compatibility for a frontend that still sends
        # overwrite_existing=True. Never destroy healthy weights: reinterpret
        # old "Override" approval as Retrain when a reusable model exists.
        if overwrite_existing and not resume_existing and not start_fresh:
            if existing.get("resumable"):
                resume_existing = True
            else:
                start_fresh = True

        if not resume_existing and not start_fresh:
            if existing.get("resumable"):
                resume_path_hint = str(existing.get("path") or output_path / "last")
                raise ArtifactConflictError(
                    kind="model", action="retrain",
                    name=str(entry.get("name") or "model"), paths=[resume_path_hint],
                    message=(
                        f'Model "{entry.get("name") or "model"}" already has trained parameters at {resume_path_hint}. '
                        'Retrain from these existing weights instead of overriding them? '
                        'The current artifact will remain untouched until retraining finishes successfully.'
                    ),
                )
            if existing.get("present"):
                reason = str(existing.get("reason") or "the existing artifact is incomplete")
                raise ArtifactConflictError(
                    kind="model", action="fresh_start", reason=reason,
                    name=str(entry.get("name") or "model"), paths=[str(output_path)],
                    message=(
                        f'Model "{entry.get("name") or "model"}" has existing local files, but its trained parameters '
                        f'cannot be resumed safely ({reason}). Start fresh? The old directory will be backed up and '
                        'restored automatically if fresh training fails.'
                    ),
                )

        # Only after the user has chosen New / Retrain / Start Fresh do we touch
        # the heavy runtime. If Retrain was selected, validate the exact indexed
        # artifact before loading the dataset so a corrupt model cannot trigger
        # an unnecessary dataset load.
        validated_existing = existing
        if resume_existing:
            validated_existing = self._existing_model_training_artifact(output_path, entry)
            if not validated_existing.get("resumable"):
                reason = str(validated_existing.get("reason") or "the trained artifact is no longer readable")
                raise ArtifactConflictError(
                    kind="model", action="fresh_start", reason=reason,
                    name=str(entry.get("name") or "model"), paths=[str(output_path)],
                    message=(
                        f'The existing model can no longer be loaded for retraining ({reason}). '
                        'Confirm Start Fresh to rebuild it from untrained parameters.'
                    ),
                )

        dataset_id = entry.get("selected_dataset_id")
        if not dataset_id:
            raise RuntimeError("Select a prepared training dataset first.")
        meta = self._dataset_meta(dataset_id)
        dataset = self.get_prepared_dataset(dataset_id)

        if progress_callback:
            progress_callback({
                "status":"running","runtime_kind":"train","phase":"runtime_import",
                "overall":0,"model_id":model_id,
                "message":"Loading the PyTorch training runtime…",
            })
        from .model_runtime import train_builder_model, ExistingModelArtifactError

        resume_path = None
        resume_mode = None
        staged_output = None
        retrain_parent = None
        retrain_output = None
        replacement_backup = None

        if resume_existing:
            # ``validated_existing`` was checked immediately after confirmation,
            # before the dataset was loaded. Reuse it here instead of repeating
            # artifact inspection after expensive data/runtime preparation.
            existing = validated_existing
            resume_path = str(existing["path"])
            resume_mode = str(existing.get("resume_mode") or "weights")
            # Retraining writes to a private sibling staging directory. The old
            # trained model therefore remains fully usable until the new run is
            # complete. On success the directories are swapped transactionally.
            retrain_parent = output_root / f".mlb-retrain-{safe_name}-{uuid.uuid4().hex[:10]}"
            retrain_output = retrain_parent / safe_name
        elif start_fresh:
            # Fresh start is only the recovery path for unusable/corrupted local
            # model storage. Preserve it as a rollback backup until success.
            if output_path.exists() or output_path.is_symlink():
                staged_output = self._stage_local_directory_override(output_path)

        # Retraining is a new runtime lifetime. Drop every inactive resident
        # model and stale traceback/compiler state before rebuilding the graph.
        self.last_run_error = None
        cached_runtimes = [
            value for key, value in self.trained_models.items()
            if key not in self._model_servers
        ]
        reset_compiler = (
            str(config.get("execution_mode") or "eager").lower() == "compiled"
            or any(
                bool(getattr((item or {}).get("compiled"), "compile_used", False))
                for item in cached_runtimes
            )
        )
        if cached_runtimes:
            self._release_cached_runtime_models(
                preserve_server_models=True,
                reset_compiler=reset_compiler,
            )
        elif reset_compiler:
            try:
                import torch
                self._reset_torch_compiler(torch)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self._stop_event.clear()

        def visible_path(value):
            if not value or retrain_output is None:
                return value
            try:
                raw_path = Path(str(value))
                rel = raw_path.relative_to(retrain_output)
                return str(output_path / rel)
            except Exception:
                return value

        def emit(payload):
            if progress_callback:
                enriched = dict(payload or {})
                enriched.setdefault("model_id", model_id)
                if retrain_output is not None and enriched.get("checkpoint_path"):
                    enriched["checkpoint_path"] = visible_path(enriched.get("checkpoint_path"))
                progress_callback(enriched)

        run_base_config = dict(config)
        if retrain_parent is not None:
            run_base_config["output_dir"] = str(retrain_parent)
            run_base_config["_studio_logical_output_dir"] = str(output_root)

        active_config = dict(run_base_config)
        attempt = 0

        def clean_partial_run_output():
            target = retrain_output if retrain_output is not None else output_path
            if target.exists() or target.is_symlink():
                self._remove_local_artifact_path(target)

        def run_training_once(run_config=None):
            nonlocal active_config, attempt
            active_config = dict(run_config or active_config)
            # A compiler retry must never reuse files from a failed attempt.
            if attempt > 0:
                clean_partial_run_output()
            attempt += 1
            return train_builder_model(
                state=self.state, model_entry=entry, dataset=dataset, dataset_meta=meta,
                config=active_config, progress=emit, stop_event=self._stop_event,
                resume_from=resume_path, resume_mode=resume_mode,
            )

        def cleanup_failed_storage():
            nonlocal staged_output
            if retrain_parent is not None and (retrain_parent.exists() or retrain_parent.is_symlink()):
                self._remove_local_artifact_path(retrain_parent)
            if staged_output:
                self._rollback_local_directory_overrides([staged_output])
                staged_output = None

        compile_recovery_warning = None
        try:
            result = run_training_once(run_base_config)
        except ExistingModelArtifactError as resume_exc:
            self._remember_run_error(resume_exc)
            self._cleanup_failed_runtime(reset_compiler=reset_compiler)
            cleanup_failed_storage()
            raise ArtifactConflictError(
                kind="model", action="fresh_start", reason=str(resume_exc),
                name=str(entry.get("name") or "model"), paths=[str(getattr(resume_exc, "path", resume_path) or output_path)],
                message=(
                    f'The saved parameters for "{entry.get("name") or "model"}" could not be loaded safely. '
                    f'{resume_exc} Start fresh from untrained parameters? The existing artifact will be backed up first.'
                ),
            ) from resume_exc
        except AssertionError as first_exc:
            if str(config.get("execution_mode") or "eager").lower() != "compiled":
                cleanup_failed_storage()
                raise
            first_detail = str(first_exc).strip() or type(first_exc).__name__
            self._remember_run_error(first_exc)
            self._cleanup_failed_runtime(reset_compiler=True)
            retry_config = dict(run_base_config)
            requested_compile_mode = str(config.get("compile_mode") or "default")
            retry_config["compile_mode"] = "default"
            emit({
                "status":"running","runtime_kind":"train","phase":"compile_retry",
                "overall":0,"step":0,
                "message":(
                    "Compiled warm-up hit a PyTorch assertion. Resetting compiler state and "
                    + ("retrying with compile mode default…" if requested_compile_mode != "default" else "rebuilding the compiled graph once…")
                ),
                "compile_recovery_reason": first_detail,
                "requested_compile_mode": requested_compile_mode,
                "retry_compile_mode": "default",
            })
            self.last_run_error = None
            try:
                result = run_training_once(retry_config)
                if requested_compile_mode != "default":
                    compile_recovery_warning = (
                        f"Requested compile mode {requested_compile_mode!r} failed during warm-up; "
                        "training recovered with compile mode 'default'."
                    )
            except ExistingModelArtifactError as resume_exc:
                self._remember_run_error(resume_exc)
                self._cleanup_failed_runtime(reset_compiler=True)
                cleanup_failed_storage()
                raise ArtifactConflictError(
                    kind="model", action="fresh_start", reason=str(resume_exc),
                    name=str(entry.get("name") or "model"), paths=[str(getattr(resume_exc, "path", resume_path) or output_path)],
                    message=(
                        f'The saved parameters for "{entry.get("name") or "model"}" could not be loaded safely. '
                        'Confirm Start Fresh to rebuild from untrained parameters.'
                    ),
                ) from resume_exc
            except AssertionError as second_exc:
                second_detail = str(second_exc).strip() or type(second_exc).__name__
                self._remember_run_error(second_exc)
                self._cleanup_failed_runtime(reset_compiler=True)
                eager_config = dict(run_base_config)
                eager_config["execution_mode"] = "eager"
                emit({
                    "status":"running","runtime_kind":"train","phase":"compile_fallback",
                    "overall":0,"step":0,
                    "message":"Compiled warm-up failed again. Rebuilding safely in eager mode so training can continue…",
                    "compile_recovery_reason": second_detail,
                    "requested_execution_mode": "compiled",
                    "fallback_execution_mode": "eager",
                })
                self.last_run_error = None
                try:
                    result = run_training_once(eager_config)
                    compile_recovery_warning = (
                        "PyTorch compiled warm-up failed twice in this kernel; "
                        "training automatically continued in eager mode. Restart the kernel "
                        "before retrying compiled execution if compiled speed is required."
                    )
                except ExistingModelArtifactError as resume_exc:
                    self._remember_run_error(resume_exc)
                    self._cleanup_failed_runtime(reset_compiler=True)
                    cleanup_failed_storage()
                    raise ArtifactConflictError(
                        kind="model", action="fresh_start", reason=str(resume_exc),
                        name=str(entry.get("name") or "model"), paths=[str(getattr(resume_exc, "path", resume_path) or output_path)],
                        message=(
                            f'The saved parameters for "{entry.get("name") or "model"}" could not be loaded safely. '
                            'Confirm Start Fresh to rebuild from untrained parameters.'
                        ),
                    ) from resume_exc
                except Exception as eager_exc:
                    self._remember_run_error(eager_exc)
                    self._cleanup_failed_runtime(reset_compiler=True)
                    cleanup_failed_storage()
                    raise
            except Exception as retry_exc:
                self._remember_run_error(retry_exc)
                self._cleanup_failed_runtime(reset_compiler=True)
                cleanup_failed_storage()
                raise
        except Exception as exc:
            self._remember_run_error(exc)
            self._cleanup_failed_runtime(reset_compiler=reset_compiler)
            cleanup_failed_storage()
            raise

        update = result["model_update"]
        if compile_recovery_warning:
            existing_warning = str(update.get("compile_warning") or "").strip()
            update["compile_warning"] = (
                f"{existing_warning} {compile_recovery_warning}".strip()
                if existing_warning else compile_recovery_warning
            )
            update["compile_recovered"] = True
            update["requested_execution_mode"] = str(config.get("execution_mode") or "eager")
            update["requested_compile_mode"] = str(config.get("compile_mode") or "default")
            update["execution_mode_used"] = str(active_config.get("execution_mode") or update.get("execution_mode_used") or "eager")
            update["compile_mode_used"] = (
                str(active_config.get("compile_mode") or "default")
                if update["execution_mode_used"] == "compiled" else None
            )

        # Commit a successful retrain atomically. Until this point the old model
        # directory has not been modified at all.
        if retrain_output is not None:
            emit({
                "status":"running","runtime_kind":"train","phase":"retrain_commit","overall":99,
                "message":"Retraining finished · committing the new model artifact atomically…",
            })
            try:
                if not retrain_output.exists():
                    raise RuntimeError("Retraining completed without producing a model artifact.")
                if output_path.exists() or output_path.is_symlink():
                    replacement_backup = self._stage_local_directory_override(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Both paths are siblings under the same model root, so this is a
                # fast directory rename in normal Studio storage. Do not delete
                # the old model here: a checkpoint-heavy backup can take minutes
                # to remove and used to leave the Training UI stuck at 99%.
                shutil.move(str(retrain_output), str(output_path))
                emit({
                    "status":"running","runtime_kind":"train","phase":"retrain_committed","overall":99,
                    "message":"Retrained model committed · updating Studio runtime and index…",
                })
            except Exception:
                if replacement_backup:
                    self._rollback_local_directory_overrides([replacement_backup])
                if retrain_parent.exists() or retrain_parent.is_symlink():
                    self._remove_local_artifact_path(retrain_parent)
                raise

            for key in ("path", "checkpoint_path", "tokenizer_path"):
                if update.get(key):
                    update[key] = visible_path(update[key])
            update["retrained_from"] = str(resume_path) if resume_path and resume_mode != "checkpoint" else update.get("retrained_from")
            update["resumed_checkpoint"] = str(resume_path) if resume_path and resume_mode == "checkpoint" else update.get("resumed_checkpoint")

        # Keep rollback backups until the canonical artifact, Studio state, index,
        # and resident runtime are all finalized. Cleanup is scheduled only after
        # the terminal 100% event has been delivered.
        pending_overrides = []
        if replacement_backup:
            pending_overrides.append(replacement_backup)
        if staged_output:
            pending_overrides.append(staged_output)

        entry_before_finalize = copy.deepcopy(entry)
        try:
            entry.update(update)
            # A successful new train/retrain changed the canonical model artifact.
            # Refresh only Studio's shallow managed indexes; no weights are loaded.
            self._refresh_managed_artifact_index()
            resident_runtime_config = dict(active_config)
            resident_runtime_config.pop("_studio_logical_output_dir", None)
            resident_runtime_config["output_dir"] = str(config.get("output_dir") or "mlbricks_workspace/models")
            self.trained_models[model_id] = {
                "compiled": result["compiled"], "tokenizer": result["tokenizer"],
                "runtime": resident_runtime_config,
            }
            mode_text = "Retraining complete" if resume_path else "Training complete"
            payload = {
                "status":"done","runtime_kind":"train","phase":"done","overall":100,
                "message":f'{mode_text}: {entry.get("name", "model")}',
                "model_id":model_id,"model_update":update,
                "sample_text":result.get("last_sample"),
            }
            if progress_callback:
                progress_callback(payload)
        except Exception:
            # Because old artifacts are retained until this point we can still
            # restore the previous canonical model if final metadata/index work
            # fails after the directory cutover.
            if pending_overrides:
                self._rollback_local_directory_overrides(pending_overrides)
            entry.clear()
            entry.update(entry_before_finalize)
            self.trained_models.pop(model_id, None)
            raise

        if pending_overrides:
            self._commit_local_directory_overrides(pending_overrides, defer_cleanup=True)
            staged_output = None
        if retrain_parent is not None and (retrain_parent.exists() or retrain_parent.is_symlink()):
            # Usually this is now an empty staging parent and can disappear
            # instantly. If not, never block successful finalization on cleanup.
            try:
                retrain_parent.rmdir()
            except OSError:
                self._defer_local_artifact_cleanup([retrain_parent])
        return update

    def _resident_generation_runtime(self, model_id, config, *, emit=None):
        """Return an inference view of a model that is already resident in RAM/VRAM.

        The Python object produced by training is the authoritative hot runtime.
        Generation must not serialize/rebuild/reload that model just because the
        Generation panel says ``Auto``.  ``Auto`` therefore means *keep the
        resident device/precision/backend*.  A checkpoint load is only required
        when no live Python model exists, or when the user explicitly asks for a
        different construction-time backend.
        """
        cached = self.trained_models.get(model_id)
        if not cached:
            return None
        resident = cached.get("compiled")
        tokenizer = cached.get("tokenizer")
        modality = str(config.get("input_kind") or "text").lower()
        needs_tokenizer = modality in {"", "unknown", "text"}
        if resident is None or (needs_tokenizer and tokenizer is None) or getattr(resident, "raw_model", None) is None:
            return None

        from .model_runtime import CompiledModel, resolve_device, resolve_precision
        import torch

        previous_runtime = cached.get("runtime") or {}
        requested_backend = str(config.get("backend") or "auto").strip().lower()
        resident_backend = str(previous_runtime.get("backend") or "auto").strip().lower()
        # Backend changes can alter component construction. Never pretend an
        # existing graph changed backend; explicit changes take the cold path.
        if requested_backend not in {"", "auto"} and requested_backend != resident_backend:
            return None

        requested_device = str(config.get("device") or "auto").strip().lower()
        desired_device = resident.device if requested_device in {"", "auto"} else resolve_device(requested_device)
        raw = resident.raw_model
        if str(desired_device) != str(resident.device):
            # Moving the already-trained object is dramatically cheaper and safer
            # than loading a second 50M+ graph from disk. Keep FP32 master weights
            # intact; generation precision is handled by autocast below.
            raw.to(device=desired_device)
            resident.device = desired_device
            resident.model = raw
            cached.pop("generation_compiled", None)

        requested_precision = str(config.get("precision") or "auto").strip().lower()
        if requested_precision in {"", "auto"}:
            desired_precision = str(resident.precision)
        else:
            desired_precision, _ = resolve_precision(requested_precision, desired_device)

        execution = str(config.get("execution_mode") or "eager").strip().lower()
        inference_model = raw
        compile_used = False
        compile_error = None
        if execution == "compiled":
            if not hasattr(torch, "compile"):
                raise RuntimeError("Compiled execution was selected, but torch.compile is unavailable in this PyTorch build.")
            mode = str(config.get("compile_mode") or "default")
            generation_compiled = cached.setdefault("generation_compiled", {})
            inference_model = generation_compiled.get(mode)
            if inference_model is None:
                if emit:
                    emit({
                        "status":"running","runtime_kind":"generate","phase":"compile",
                        "overall":0,"runtime_source":"resident",
                        "message":f"Compiling resident model in RAM/VRAM ({mode})…",
                    })
                inference_model = torch.compile(raw, mode=mode, dynamic=None, fullgraph=False)
                generation_compiled[mode] = inference_model
            compile_used = True

        compiled = CompiledModel(
            inference_model, raw, resident.training_model, desired_device,
            desired_precision, resident.vocab_size, resident.parameter_count,
            compile_used, compile_error,
        )
        cached["compiled"] = compiled
        cached["runtime"] = dict(config)
        cached["resident"] = True
        if emit:
            emit({
                "status":"running","runtime_kind":"generate","phase":"resident_reuse",
                "overall":0,"runtime_source":"resident",
                "message":f"Using resident model already in {'VRAM' if desired_device.type == 'cuda' else 'RAM'} on {desired_device} · no checkpoint reload",
            })
        return compiled, tokenizer

    @staticmethod
    def _runtime_input_image_size(entry):
        for node in ((entry or {}).get("architecture") or {}).get("nodes") or []:
            if str(node.get("type") or "") in {"image_input", "video_input", "stream_input"}:
                try:
                    value = int((node.get("params") or {}).get("image_size") or 0)
                except (TypeError, ValueError):
                    value = 0
                if value > 0:
                    return value
        return None

    def _run_universal_input_runtime(self, compiled, entry, envelope, emit):
        """Run finite or live non-text input through the current model runtime."""
        from .model_runtime import TrainingStopped, run_universal_inference
        from .universal_io import iter_input_stream, load_single_input

        output_type = str((entry.get("requirements") or {}).get("output_type") or "unknown")
        image_size = self._runtime_input_image_size(entry)
        stream_like = bool(
            envelope.continuous
            or envelope.kind == "video"
            or envelope.mode in {"sequence", "batch"}
        )
        processed = 0
        last_output = None
        last_live_emit = 0.0
        live_emit_interval = 1.0 / max(float(envelope.fps or 5.0), 0.25)

        emit({
            "status":"running","runtime_kind":"generate","phase":"input_ready","overall":0,
            "message":f"{envelope.kind.title()} input ready · {envelope.mode} · {envelope.task}",
            "input_envelope":envelope.public_dict(),"processed_items":0,
        })

        if stream_like:
            samples = iter_input_stream(
                envelope, image_size=image_size, stop_event=self._stop_event
            )
        else:
            value, meta = load_single_input(envelope, image_size=image_size)
            samples = iter(((value, meta),))

        for value, input_meta in samples:
            if self._stop_event.is_set():
                raise TrainingStopped("Universal input runtime stopped.")
            started = time.perf_counter()
            output = run_universal_inference(
                compiled, value, input_kind=envelope.kind,
                output_type=output_type, task=envelope.task,
                prompt=envelope.prompt, metadata=input_meta,
            )
            processed += 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            output_meta = dict(output.get("metadata") or {})
            output_meta["input"] = dict(input_meta or {})
            output_meta["processed_items"] = processed
            output["metadata"] = output_meta
            last_output = output
            now = time.monotonic()
            should_emit = (not envelope.continuous) or processed == 1 or (now - last_live_emit) >= live_emit_interval
            if should_emit:
                last_live_emit = now
                emit({
                    "status":"running","runtime_kind":"generate",
                    "phase":"monitor" if envelope.continuous else "process",
                    "overall":0 if stream_like else 100,
                    "message":(
                        f"Monitoring {envelope.kind} · processed {processed} item(s)"
                        if envelope.continuous else
                        f"Processed {processed} {envelope.kind} item(s)"
                    ),
                    "processed_items":processed,"items_per_sec":1.0/elapsed,
                    "input_envelope":envelope.public_dict(),
                    "generated_output":output,
                    "generated_output_kind":output.get("kind"),
                    "generated_output_mime":output.get("mime"),
                    "generated_output_meta":output_meta,
                })

        if self._stop_event.is_set():
            raise TrainingStopped("Universal input runtime stopped.")
        if last_output is None:
            raise RuntimeError("The input source ended before Studio received a sample.")

        entry["last_generated_output"] = last_output
        entry["last_generated_output_kind"] = last_output.get("kind")
        entry["last_generated_output_mime"] = last_output.get("mime")
        entry["last_generated_output_meta"] = dict(last_output.get("metadata") or {})
        entry["generated_at"] = datetime.now(timezone.utc).isoformat()
        emit({
            "status":"done","runtime_kind":"generate","phase":"done","overall":100,
            "message":f"{envelope.task.title()} complete · processed {processed} item(s).",
            "processed_items":processed,"input_envelope":envelope.public_dict(),
            "generated_output":last_output,
            "generated_output_kind":last_output.get("kind"),
            "generated_output_mime":last_output.get("mime"),
            "generated_output_meta":last_output.get("metadata") or {},
            "model_update":{
                "last_generated_output":last_output,
                "last_generated_output_kind":last_output.get("kind"),
                "last_generated_output_mime":last_output.get("mime"),
                "last_generated_output_meta":last_output.get("metadata") or {},
                "generated_at":entry["generated_at"],
            },
        })
        return last_output

    def generate_model(self, model_id, *, progress_callback=None):
        """Run the universal Studio input runtime; text keeps the token-generation fast path."""
        if progress_callback:
            progress_callback({
                "status":"running","runtime_kind":"generate","phase":"runtime_import",
                "overall":0,"model_id":model_id,
                "message":"Preparing model runtime…",
            })
        from .model_runtime import load_trained_for_generation, generate_text
        from .universal_io import normalize_input_config
        entry = self._model_output(model_id)
        if not entry.get("weights_ready"):
            raise RuntimeError("This model has no trained/loaded weights yet.")
        dataset_id = entry.get("selected_dataset_id")
        meta = self._dataset_meta(dataset_id) if dataset_id else copy.deepcopy(entry.get("hub_dataset_meta") or {})
        config = self._runtime_with_project_trust(entry.get("generation_config") or {}, checkpoint_path=entry.get("checkpoint_path") or entry.get("path"))
        input_envelope = normalize_input_config(config, entry)
        self._stop_event.clear()

        def emit(payload):
            if progress_callback:
                enriched = dict(payload or {})
                enriched.setdefault("model_id", model_id)
                progress_callback(enriched)

        resident = self._resident_generation_runtime(model_id, config, emit=emit)
        if resident is not None:
            compiled, tokenizer = resident
        else:
            compiled, tokenizer = load_trained_for_generation(
                state=self.state, model_entry=entry, dataset_meta=meta, config=config,
                checkpoint_path=entry.get("checkpoint_path") or entry.get("path"), progress=emit,
            )
            self.trained_models[model_id] = {
                "compiled": compiled, "tokenizer": tokenizer,
                "runtime": dict(config), "resident": True,
            }
            emit({
                "status":"running","runtime_kind":"generate","phase":"resident_ready",
                "overall":0,"runtime_source":"loaded",
                "message":f"Model loaded once and kept resident on {compiled.device} for following responses",
            })

        requirements = dict(entry.get("requirements") or {})
        output_type = str(requirements.get("output_type") or "unknown").strip().lower()
        training_mode = str(requirements.get("training_mode") or "").strip().lower()
        if (
            input_envelope.kind != "text"
            or output_type == "audio_output"
            or training_mode == "audio_generation"
        ):
            return self._run_universal_input_runtime(
                compiled, entry, input_envelope, emit
            )

        from .model_runtime import runtime_int, runtime_float
        context = runtime_int(
            entry.get("context_length") or self.state.get("project",{}).get("context_length"),
            512, "Model Context", minimum=2,
        )
        text, count = generate_text(
            compiled.model, tokenizer, config.get("prompt") or "Once upon a time",
            max_new_tokens=runtime_int(config.get("max_new_tokens"),128,"New Token Count",minimum=1),
            context=context,
            device=compiled.device, precision=compiled.precision,
            temperature=runtime_float(config.get("temperature"),0.8,"Temperature",minimum=0.00001),
            top_k=runtime_int(config.get("top_k"),50,"Top K",minimum=0),
            top_p=runtime_float(config.get("top_p"),0.95,"Top P",minimum=0.0,maximum=1.0),
            seed=runtime_int(config.get("seed"),42,"Seed"),
            progress=emit, stop_event=self._stop_event,
            # Local app has a lossless event queue, so emit each token. Notebook
            # mode remains coalesced to avoid ipywidgets transport backlog.
            stream_every_token=self._app_server is not None,
        )
        entry["last_generation"] = text
        entry["last_generated_output"] = {"kind":"text","mime":"text/plain","data":text}
        entry["last_generated_output_kind"] = "text"
        entry["last_generated_output_mime"] = "text/plain"
        entry["last_generated_output_meta"] = {"tokens": count}
        entry["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload={
            "status":"done","runtime_kind":"generate","phase":"done","overall":100,
            "message":f"Generated {count} tokens.","model_id":model_id,
            "generated_tokens":count,"generated_text":text,
            "input_envelope":input_envelope.public_dict(),
            "generated_output":entry["last_generated_output"],"generated_output_kind":"text",
            "generated_output_mime":"text/plain","generated_output_meta":{"tokens": count},
            "model_update":{"last_generation":text,"generated_at":entry["generated_at"]},
        }
        if progress_callback: emit(payload)
        return text

    def _generation_runtime_for_entry(self, entry, serve_config):
        runtime=dict(entry.get("generation_config") or {})
        for key in ("device","backend","execution_mode","compile_mode","precision"):
            value=(serve_config or {}).get(key)
            if value not in (None,""): runtime[key]=value
        return self._runtime_with_project_trust(runtime, checkpoint_path=entry.get("checkpoint_path") or entry.get("path"))

    def start_model_server(self, model_id, *, config=None, api_key=None, ngrok_token=None, progress_callback=None):
        from .model_runtime import load_trained_for_generation
        from .serve import ModelHTTPRuntime
        entry=self._model_output(model_id)
        if not entry.get("weights_ready"): raise RuntimeError("Train or load model weights before serving.")
        config=dict(config or entry.get("serve_config") or {})
        runtime=self._generation_runtime_for_entry(entry,config)
        dataset_id=entry.get("selected_dataset_id")
        meta=self._dataset_meta(dataset_id) if dataset_id else copy.deepcopy(entry.get("hub_dataset_meta") or {})
        old=self._model_servers.pop(model_id,None)
        if old is not None: old.stop()

        entry["serve_status"] = "starting"
        entry["serve_urls"] = {}
        entry["serve_tunnel_error"] = None

        def emit(message,overall):
            if progress_callback: progress_callback({
                "status":"running","runtime_kind":"serve","phase":"starting","overall":overall,
                "message":message,"model_id":model_id,
                "model_update":{"serve_status":"starting","serve_urls":{}},
            })
        emit("Loading trained model for API server…",10)

        compiled=tokenizer=None
        resident=self._resident_generation_runtime(model_id,runtime,emit=None)
        if resident is not None:
            compiled,tokenizer=resident
        if compiled is None or tokenizer is None:
            compiled,tokenizer=load_trained_for_generation(
                state=self.state,model_entry=entry,dataset_meta=meta,config=runtime,
                checkpoint_path=entry.get("checkpoint_path") or entry.get("path"),progress=None)
            self.trained_models[model_id]={"compiled":compiled,"tokenizer":tokenizer,"runtime":dict(runtime),"resident":True}

        emit("Starting HTTP inference server…",55)
        server=ModelHTTPRuntime(
            model_id=model_id,model_name=entry.get("name") or "MLBricks Model",
            compiled=compiled,tokenizer=tokenizer,
            context=entry.get("context_length") or (self.state.get("project") or {}).get("context_length") or 512,
            generation_defaults=entry.get("generation_config") or {},
            host=config.get("host") or "127.0.0.1",port=config.get("port") if config.get("port") is not None else 8000,
            cors_origin=config.get("cors_origin") or "same-origin",api_key_required=bool(config.get("require_api_key",True)),
            api_key=api_key or None,
            max_request_bytes=config.get("max_request_bytes",1_048_576),
            max_prompt_chars=config.get("max_prompt_chars",32_768),
            max_new_tokens=config.get("max_server_new_tokens",2048),
            request_timeout_seconds=config.get("request_timeout_seconds",120),
            max_concurrent_requests=config.get("max_concurrent_requests",2),
            rate_limit_per_minute=config.get("rate_limit_per_minute",60),
            debug_errors=bool(config.get("debug_errors",False)))
        info=server.start()

        # Register the live HTTP server immediately. If a public tunnel later
        # fails, Stop/Restart must still be able to clean up this server.
        self._model_servers[model_id]=server

        tunnel=str(config.get("public_tunnel") or "off").lower()
        tunnel_error=None
        if tunnel=="ngrok":
            emit("Opening public HTTPS tunnel…",80)
            try:
                server.start_ngrok(auth_token=ngrok_token or None)
            except Exception as exc:
                tunnel_error=f"{type(exc).__name__}: {exc}"
            info=server.info()

        safe_config={"host":server.host,"port":info.port,"cors_origin":server.cors_origin,
                     "require_api_key":server.api_key_required,"public_tunnel":tunnel,
                     "max_request_bytes":server.max_request_bytes,"max_prompt_chars":server.max_prompt_chars,
                     "max_server_new_tokens":server.max_new_tokens,"request_timeout_seconds":server.request_timeout_seconds,
                     "max_concurrent_requests":server.max_concurrent_requests,"rate_limit_per_minute":server.rate_limit_per_minute,
                     "device":runtime.get("device","auto"),"backend":runtime.get("backend","pytorch"),
                     "execution_mode":runtime.get("execution_mode","eager"),
                     "compile_mode":runtime.get("compile_mode","reduce-overhead"),
                     "precision":runtime.get("precision","fp16")}
        entry["serve_config"]=safe_config
        entry["serve_status"]="running"
        entry["serve_urls"]={"local_url":info.local_url,"lan_url":info.lan_url,"public_url":info.public_url}
        entry["serve_tunnel_error"]=tunnel_error

        result=info.to_dict(include_secret=True)
        result["public_tunnel_error"]=tunnel_error
        result["running"]=True

        message = f'Model API running on port {info.port}.'
        if getattr(info, "used_port_fallback", False):
            message = (
                f'Port {getattr(info, "requested_port", config.get("port", 8000))} was busy; '
                f'Builder automatically started the API on port {info.port}.'
            )
        if tunnel_error:
            message += " Local API is running, but the public ngrok tunnel failed."

        if progress_callback: progress_callback({
            "status":"done","runtime_kind":"serve","phase":"running","overall":100,
            "message":message,"model_id":model_id,"serve_info":result,
            "model_update":{
                "serve_config":safe_config,
                "serve_status":"running",
                "serve_urls":entry["serve_urls"],
                "serve_tunnel_error":tunnel_error,
            }
        })
        return result

    def stop_model_server(self, model_id, *, progress_callback=None):
        entry=self._model_output(model_id); server=self._model_servers.pop(model_id,None)
        if server is not None: server.stop()
        entry["serve_status"]="stopped"; entry["serve_urls"]={}; entry["serve_tunnel_error"]=None
        payload={"status":"stopped","runtime_kind":"serve","phase":"stopped","overall":100,
                 "message":"Model API server stopped.","model_id":model_id,
                 "serve_info":{"model_id":model_id,"model_name":entry.get("name"),"running":False},
                 "model_update":{"serve_status":"stopped","serve_urls":{}}}
        if progress_callback: progress_callback(payload)
        return payload["serve_info"]

    def model_server_status(self, model_id, *, progress_callback=None):
        entry=self._model_output(model_id); server=self._model_servers.get(model_id)
        if server is None:
            info={"model_id":model_id,"model_name":entry.get("name"),"running":False}; message="Model API server is not running."; status="stopped"
        else:
            info=server.info().to_dict(include_secret=False); info["running"]=True
            info["public_tunnel_error"]=entry.get("serve_tunnel_error")
            message=f'Model API running on port {info["port"]}.'
            if info.get("public_tunnel_error"):
                message += " Public tunnel is unavailable."
            status="done"
        if progress_callback: progress_callback({"status":status,"runtime_kind":"serve","phase":"status","overall":100,
                                                 "message":message,"model_id":model_id,"serve_info":info})
        return info

    def _execute_serve_command(self, command, progress_callback=None):
        action=str(command.get("action") or ""); model_id=command.get("model_id"); serve=command.get("serve") or {}
        credentials=serve.get("credentials") or {}
        if action=="serve_start": return self.start_model_server(model_id,config=serve.get("config") or {},
            api_key=credentials.get("api_key"),ngrok_token=credentials.get("ngrok_token"),progress_callback=progress_callback)
        if action=="serve_stop": return self.stop_model_server(model_id,progress_callback=progress_callback)
        if action=="serve_status": return self.model_server_status(model_id,progress_callback=progress_callback)
        raise ValueError(f"Unknown serve command: {action!r}")

    def hub_status(self, token=None):
        from .hub import auth_status
        return auth_status(token=token)

    def _hub_model_package(self, entry):
        model_ws = (self.state.get("workspaces") or {}).get("model") or {}
        root_id = model_ws.get("root_component_id") or self.state.get("root_component_id")
        component = copy.deepcopy((self.state.get("components") or {}).get(root_id) or {})
        dataset_meta = None
        dataset_id = entry.get("selected_dataset_id")
        if dataset_id:
            try:
                dataset_meta = copy.deepcopy(self._dataset_meta(dataset_id))
            except Exception:
                dataset_meta = None
        return {
            "project": copy.deepcopy(self.state.get("project") or {}),
            "model_component": component,
            "custom_components": copy.deepcopy(self.state.get("custom_components") or {}),
            "component_cache": copy.deepcopy(self.state.get("component_cache") or {}),
            "model_entry": copy.deepcopy(entry),
            "dataset_meta": dataset_meta,
        }

    def push_dataset_to_hub(self, dataset_id, repo_id, *, private=True, token=None):
        from .hub import push_dataset
        dataset = self.get_prepared_dataset(dataset_id)
        meta = self._dataset_meta(dataset_id)
        result = push_dataset(
            dataset,
            repo_id=repo_id,
            metadata=meta,
            private=private,
            token=token,
        )
        meta["hub_repo_id"] = result["repo_id"]
        meta["hub_url"] = result["url"]
        meta["hub_revision"] = "main"
        return result

    def load_dataset_from_hub(self, repo_id, *, revision=None, token=None):
        from .hub import load_dataset
        dataset, saved_meta, result = load_dataset(repo_id, revision=revision, token=token)

        summary = self._summarize_prepared_result(dataset)
        dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
        name = (
            (saved_meta or {}).get("name")
            or str(repo_id).rstrip("/").split("/")[-1]
            or "Hub Dataset"
        )
        pipeline = copy.deepcopy((saved_meta or {}).get("pipeline") or {
            "source": {
                "type": "hf_dataset",
                "name": "Hugging Face Dataset",
                "dataset_id": repo_id,
                "split": "train",
            },
            "text_processing": None,
            "split": None,
            "tokenizer": None,
            "image_processing": None,
            "audio_processing": None,
            "signal_processing": None,
            "batch": None,
            "output": None,
            "steps": [],
        })
        metadata = {
            "id": dataset_id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_node_id": None,
            "storage": "hub+memory",
            "path": None,
            "pipeline": pipeline,
            "hub_repo_id": result["repo_id"],
            "hub_url": result["url"],
            "hub_revision": result["revision"],
            **summary,
        }
        self.state.setdefault("prepared_datasets", []).append(metadata)
        self.prepared_datasets[dataset_id] = dataset
        return metadata

    def push_model_to_hub(self, model_id, repo_id, *, private=True, token=None):
        from .hub import push_model
        entry = self._model_output(model_id)
        cached = self.trained_models.get(model_id) or {}
        tokenizer = cached.get("tokenizer")
        result = push_model(
            repo_id=repo_id,
            package=self._hub_model_package(entry),
            checkpoint_path=entry.get("checkpoint_path") or entry.get("path"),
            tokenizer=tokenizer,
            private=private,
            token=token,
        )
        entry["hub_repo_id"] = result["repo_id"]
        entry["hub_url"] = result["url"]
        entry["hub_revision"] = "main"
        return result

    def load_model_from_hub(self, repo_id, *, revision=None, token=None):
        from .hub import load_model
        package, folder, result = load_model(repo_id, revision=revision, token=token)

        model_ws = (self.state.get("workspaces") or {}).get("model") or {}
        root_id = model_ws.get("root_component_id") or self.state.get("root_component_id")
        if not root_id:
            raise RuntimeError("Model Builder workspace is unavailable.")

        component = copy.deepcopy(package.get("model_component") or {})
        if not component.get("nodes"):
            raise RuntimeError("The Hub model package does not contain a Builder model graph.")
        component["id"] = root_id
        self.state.setdefault("components", {})[root_id] = component
        self.state["view_component_id"] = root_id
        imported_custom = copy.deepcopy(package.get("custom_components") or {})
        self.state.setdefault("custom_components", {}).update(imported_custom)
        if project_executable_features({"custom_components": imported_custom}):
            self._mark_external_project_untrusted("huggingface-model")
        self.state.setdefault("component_cache", {}).update(
            copy.deepcopy(package.get("component_cache") or {})
        )

        loaded_project = copy.deepcopy(package.get("project") or {})
        current_project = self.state.setdefault("project", {})
        for key in (
            "name", "context_length", "batch_size", "model_settings",
            "estimated_parameters"
        ):
            if key in loaded_project:
                current_project[key] = loaded_project[key]

        source_entry = copy.deepcopy(package.get("model_entry") or {})
        new_id = f"model_{uuid.uuid4().hex[:12]}"
        source_entry["id"] = new_id
        source_entry["architecture"] = copy.deepcopy(component)
        source_entry["hub_repo_id"] = result["repo_id"]
        source_entry["hub_url"] = result["url"]
        source_entry["hub_revision"] = result["revision"]
        source_entry["selected_dataset_id"] = None

        hub_meta = copy.deepcopy(package.get("dataset_meta") or {})
        tokenizer_dir = package.get("tokenizer_dir")
        if tokenizer_dir and (folder / tokenizer_dir).exists():
            hub_meta.setdefault("pipeline", {}).setdefault("tokenizer", {})[
                "tokenizer_name"
            ] = str(folder / tokenizer_dir)
        source_entry["hub_dataset_meta"] = hub_meta

        artifact_dir = package.get("model_artifact_dir")
        artifact = folder / artifact_dir if artifact_dir else None
        checkpoint_file = package.get("checkpoint_file")
        checkpoint = folder / checkpoint_file if checkpoint_file else None
        if artifact is not None and (artifact / "model.pt").exists():
            source_entry["path"] = str(artifact)
            source_entry["checkpoint_path"] = str(artifact)
            source_entry["weights_ready"] = True
            source_entry["training_status"] = "trained"
            source_entry["status"] = "trained"
            source_entry["format"] = "MLBricks model artifact"
            source_entry["artifact_format"] = "mlbricks.model"
        elif checkpoint is not None and checkpoint.exists():
            # Backward compatibility with Builder Hub repositories that stored
            # the pre-lifecycle weights/optimizer .pt checkpoint.
            source_entry["path"] = str(checkpoint)
            source_entry["checkpoint_path"] = str(checkpoint)
            source_entry["weights_ready"] = True
            source_entry["training_status"] = "trained"
            source_entry["status"] = "trained"
            source_entry["format"] = source_entry.get("format") or "PyTorch checkpoint"
        else:
            source_entry["weights_ready"] = False
            source_entry["training_status"] = source_entry.get("training_status") or "untrained"
            source_entry["status"] = "built"

        self.state.setdefault("model_outputs", []).append(source_entry)
        return source_entry

    def push_project_to_hub(self, repo_id, *, private=True, token=None):
        from .hub import push_project
        return push_project(repo_id=repo_id, state=self.state, private=private, token=token)

    def load_project_from_hub(self, repo_id, *, revision=None, token=None):
        from .hub import load_project
        loaded, result = load_project(repo_id, revision=revision, token=token)
        if not isinstance(loaded, dict) or not loaded.get("components"):
            raise RuntimeError("Downloaded Builder project is invalid.")
        self.state = loaded
        self.state.setdefault("prepared_datasets", [])
        self.state.setdefault("model_outputs", [])
        self.state.setdefault("project_files", [])
        self.prepared_datasets = {}
        self.trained_models = {}
        self._mark_external_project_untrusted("huggingface-hub")
        return result

    @staticmethod
    def _safe_cloud_name(value):
        text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "mlbricks")).strip("-")
        return text or "mlbricks"

    def _clean_state_for_export(self):
        clean = copy.deepcopy(self.state)
        clean.pop("_runtime_command", None)
        clean.pop("_session_secrets", None)
        return clean

    def _create_cloud_bundle(self, content_type, artifact_id, destination):
        content_type = str(content_type or "project").lower()
        destination = Path(destination)
        with tempfile.TemporaryDirectory(prefix="mlbricks_bundle_") as td:
            root = Path(td) / "bundle"
            root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "mlbricks-cloud-bundle-v1",
                "builder_version": "1.0.0b2",
                "content_type": content_type,
            }

            if content_type == "project":
                manifest["name"] = (self.state.get("project") or {}).get("name") or "MLBricks Project"
                (root / "project.json").write_text(
                    json.dumps(self._clean_state_for_export(), indent=2),
                    encoding="utf-8",
                )
            elif content_type == "dataset":
                meta = copy.deepcopy(self._dataset_meta(artifact_id))
                dataset = self.get_prepared_dataset(artifact_id)
                manifest["name"] = meta.get("name") or "Prepared Dataset"
                (root / "dataset_meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                data_dir = root / "dataset"
                if not hasattr(dataset, "save_to_disk"):
                    raise RuntimeError(
                        "This prepared object cannot be bundled. Push before converting to a DataLoader."
                    )
                dataset.save_to_disk(str(data_dir))
            elif content_type == "model":
                entry = self._model_output(artifact_id)
                package = self._hub_model_package(entry)
                manifest["name"] = entry.get("name") or "MLBricks Model"
                (root / "model_package.json").write_text(
                    json.dumps(package, indent=2), encoding="utf-8"
                )
                checkpoint = entry.get("checkpoint_path") or entry.get("path")
                if checkpoint and Path(checkpoint).exists():
                    source_path = Path(checkpoint)
                    if source_path.is_dir() and (source_path / "model.pt").exists():
                        artifact = root / "model_artifact"
                        shutil.copytree(source_path, artifact, dirs_exist_ok=True)
                        manifest["model_artifact_dir"] = "model_artifact"
                    elif source_path.is_file():
                        # Legacy Builder checkpoint support. New training runs use
                        # the directory-based mlbricks.save() artifact above.
                        weights = root / "weights"
                        weights.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, weights / "last.pt")
                        manifest["checkpoint_file"] = "weights/last.pt"
                cached = self.trained_models.get(artifact_id) or {}
                tokenizer = cached.get("tokenizer")
                if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                    tok = root / "tokenizer"
                    tok.mkdir(parents=True, exist_ok=True)
                    try:
                        tokenizer.save_pretrained(str(tok))
                        manifest["tokenizer_dir"] = "tokenizer"
                    except Exception:
                        pass
            else:
                raise ValueError(f"Unsupported cloud content type: {content_type!r}")

            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            archive = shutil.make_archive(str(destination.with_suffix("")), "zip", root)
            shutil.move(archive, destination)
        return manifest

    def _restore_cloud_bundle(self, archive_path):
        archive_path = Path(archive_path)
        with tempfile.TemporaryDirectory(prefix="mlbricks_restore_") as td:
            root = Path(td)
            safe_extract_zip(archive_path, root)
            manifest_path = root / "manifest.json"
            if not manifest_path.exists():
                raise RuntimeError("Downloaded file is not an MLBricks cloud bundle.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != "mlbricks-cloud-bundle-v1":
                raise RuntimeError("Unsupported MLBricks cloud bundle format.")

            content_type = manifest.get("content_type")
            if content_type == "project":
                loaded = json.loads((root / "project.json").read_text(encoding="utf-8"))
                if not loaded.get("components"):
                    raise RuntimeError("Cloud project does not contain a valid Builder state.")
                self.state = loaded
                self.state.setdefault("prepared_datasets", [])
                self.state.setdefault("model_outputs", [])
                self.state.setdefault("project_files", [])
                self.prepared_datasets = {}
                self.trained_models = {}
                self._mark_external_project_untrusted("cloud-bundle")
                return {"content_type": "project", "name": manifest.get("name")}

            if content_type == "dataset":
                try:
                    from datasets import load_from_disk
                except ImportError as exc:
                    raise RuntimeError("Loading a cloud dataset bundle needs `datasets`.") from exc
                dataset = load_from_disk(str(root / "dataset"))
                saved_meta = json.loads((root / "dataset_meta.json").read_text(encoding="utf-8"))
                summary = self._summarize_prepared_result(dataset)
                dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
                meta = copy.deepcopy(saved_meta)
                meta.update({
                    "id": dataset_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "storage": "cloud+memory",
                    "path": None,
                    **summary,
                })
                self.state.setdefault("prepared_datasets", []).append(meta)
                self.prepared_datasets[dataset_id] = dataset
                return {"content_type": "dataset", "dataset": meta, "name": meta.get("name")}

            if content_type == "model":
                package = json.loads((root / "model_package.json").read_text(encoding="utf-8"))
                model_ws = (self.state.get("workspaces") or {}).get("model") or {}
                root_id = model_ws.get("root_component_id") or self.state.get("root_component_id")
                component = copy.deepcopy(package.get("model_component") or {})
                if not component.get("nodes"):
                    raise RuntimeError("Cloud model bundle does not contain a model graph.")
                component["id"] = root_id
                self.state.setdefault("components", {})[root_id] = component
                self.state["view_component_id"] = root_id
                imported_custom = copy.deepcopy(package.get("custom_components") or {})
                self.state.setdefault("custom_components", {}).update(imported_custom)
                if project_executable_features({"custom_components": imported_custom}):
                    self._mark_external_project_untrusted("cloud-model")
                self.state.setdefault("component_cache", {}).update(
                    copy.deepcopy(package.get("component_cache") or {})
                )
                source = copy.deepcopy(package.get("model_entry") or {})
                source["id"] = f"model_{uuid.uuid4().hex[:12]}"
                source["architecture"] = copy.deepcopy(component)
                source["selected_dataset_id"] = None
                artifact_dir = manifest.get("model_artifact_dir")
                artifact_source = root / artifact_dir if artifact_dir else None
                checkpoint_file = manifest.get("checkpoint_file")
                if artifact_source is not None and (artifact_source / "model.pt").exists():
                    # Persist the complete MLBricks artifact outside the temporary
                    # extraction folder so mlbricks.load()/inspect() keep working.
                    cache_dir = Path.home() / ".cache" / "mlb_studio" / "cloud_models" / source["id"]
                    artifact = cache_dir / "model"
                    shutil.rmtree(artifact, ignore_errors=True)
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(artifact_source, artifact)
                    source["path"] = str(artifact)
                    source["checkpoint_path"] = str(artifact)
                    source["weights_ready"] = True
                    source["training_status"] = "trained"
                    source["status"] = "trained"
                    source["format"] = "MLBricks model artifact"
                    source["artifact_format"] = "mlbricks.model"
                elif checkpoint_file and (root / checkpoint_file).exists():
                    # Persist legacy .pt checkpoints for backward compatibility.
                    cache_dir = Path.home() / ".cache" / "mlb_studio" / "cloud_models" / source["id"]
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = cache_dir / "last.pt"
                    shutil.copy2(root / checkpoint_file, checkpoint)
                    source["path"] = str(checkpoint)
                    source["checkpoint_path"] = str(checkpoint)
                    source["weights_ready"] = True
                    source["training_status"] = "trained"
                    source["status"] = "trained"
                    source["format"] = source.get("format") or "PyTorch checkpoint"
                else:
                    source["weights_ready"] = False
                    source["status"] = "built"
                self.state.setdefault("model_outputs", []).append(source)
                return {"content_type": "model", "model": source, "name": source.get("name")}

            raise RuntimeError(f"Unknown cloud bundle type: {content_type!r}")

    def scan_local_runtime_files(self, roots=None):
        """List local artifacts without broad filesystem scans by default.

        With no explicit roots, Local Repository is backed by the two managed
        Studio indexes. A user-supplied root still opts into the older recursive
        local-environment scan for importing external artifacts.
        """
        if roots:
            from .local_runtime import scan_local_files
            return scan_local_files(roots=roots)

        self._ensure_managed_artifact_index_current()
        entries = []
        model_root = str(self._managed_models_root())
        data_root = str(Path((self.local_environment.get("paths") or {}).get("data") or "mlbricks_workspace/data"))
        for record in self._indexed_model_records():
            artifact_path = str(record.get("artifact_path") or record.get("path") or "")
            if not artifact_path:
                continue
            entries.append({
                "path": artifact_path,
                "name": str(record.get("name") or Path(artifact_path).name),
                "relative": str(Path(artifact_path).name),
                "root": model_root,
                "kind": "model_artifact" if record.get("resumable") else "folder",
                "label": "MLBricks Model" if record.get("resumable") else "Incomplete Model",
                "size": None, "size_label": "—", "is_dir": True,
                "indexed": True,
            })
        for record in self._indexed_data_records():
            path = str(record.get("path") or "")
            if not path:
                continue
            entries.append({
                "path": path,
                "name": str(record.get("name") or Path(path).name),
                "relative": str(Path(path).name),
                "root": data_root,
                "kind": "dataset_dir",
                "label": "Prepared Dataset" if record.get("status") == "ready" else "Incomplete Dataset",
                "size": None, "size_label": "—", "is_dir": True,
                "indexed": True,
            })
        return {
            "roots": [model_root, data_root],
            "entries": entries,
            "truncated": False,
            "indexed": True,
        }

    def _register_local_dataset(self, dataset, path, *, tokenizer_name="gpt2", saved_meta=None):
        path = str(Path(path).resolve())
        summary = self._summarize_prepared_result(dataset)
        dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
        name = (saved_meta or {}).get("name") or Path(path).name or "Local Dataset"
        pipeline = copy.deepcopy((saved_meta or {}).get("pipeline") or {})
        if not pipeline:
            columns = set()
            for split in (summary.get("splits") or {}).values():
                columns.update(split.get("columns") or [])
            tokenized = "input_ids" in columns
            pipeline = {
                "source": {"type": "local_dataset", "name": "Kaggle / Local Dataset", "path": path},
                "text_processing": None,
                "split": None,
                "tokenizer": ({
                    "type": "tokenize_text", "name": "Loaded Tokenizer",
                    "tokenizer_name": tokenizer_name or "gpt2",
                    "context_length": (self.state.get("project") or {}).get("context_length") or 512,
                    "text_column": "text", "truncation": True,
                } if tokenized else None),
                "image_processing": None, "audio_processing": None, "signal_processing": None, "batch": None,
                "output": {"type": "prepared_dataset", "name": "Loaded from Kaggle / Local"},
                "steps": [],
            }
        metadata = copy.deepcopy(saved_meta or {})
        metadata.update({
            "id": dataset_id, "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_node_id": None, "storage": "local+memory", "path": path,
            "pipeline": pipeline, "local_source": True, **summary,
        })
        self.state.setdefault("prepared_datasets", []).append(metadata)
        self.prepared_datasets[dataset_id] = dataset
        return metadata

    def load_local_dataset_path(self, path, *, tokenizer_name="gpt2", text_column="text"):
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Local dataset path was not found: {path_obj}")
        saved_meta = None
        if path_obj.is_dir():
            try:
                from datasets import load_from_disk
            except ImportError as exc:
                raise RuntimeError("Loading a saved Kaggle dataset needs `datasets`. Install it with: pip install datasets") from exc
            try:
                dataset = load_from_disk(str(path_obj))
            except Exception as exc:
                raise RuntimeError(f"{path_obj} is not a valid Hugging Face save_to_disk dataset: {exc}") from exc
            for meta_name in ("mlbricks_dataset.json", "dataset_meta.json"):
                meta_path = path_obj / meta_name
                if meta_path.exists():
                    try: saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception: pass
                    break
        else:
            from .data import load_local_dataset
            dataset = load_local_dataset(path_obj, text_column=text_column or "", max_rows=None)
        return self._register_local_dataset(dataset, path_obj, tokenizer_name=tokenizer_name, saved_meta=saved_meta)

    @staticmethod
    def _normalized_model_name(value):
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    @staticmethod
    def _architecture_shape_signature(component):
        nodes = component.get("nodes") or []
        id_to_index = {node.get("id"): i for i, node in enumerate(nodes)}
        node_signature = []
        for node in nodes:
            node_signature.append((
                str(node.get("type") or ""),
                Builder._normalized_model_name(node.get("name")),
            ))
        edges = []
        for edge in component.get("edges") or []:
            source = id_to_index.get(edge.get("source"))
            target = id_to_index.get(edge.get("target"))
            if source is None or target is None:
                continue
            edges.append((
                source,
                target,
                str(edge.get("kind") or "main"),
            ))
        return tuple(node_signature), tuple(sorted(edges))

    def _current_model_component(self):
        model_ws = (self.state.get("workspaces") or {}).get("model") or {}
        root_id = model_ws.get("root_component_id") or self.state.get("root_component_id")
        return copy.deepcopy((self.state.get("components") or {}).get(root_id) or {})

    def _recover_legacy_custom_components(self, architecture, embedded_custom):
        """
        Old checkpoints saved the root graph but not custom-component definitions.
        If the currently open Builder graph has the same top-level node/edge shape,
        remap the stale checkpoint definition IDs to the current matching custom
        nodes by position/name. This is safe only after exact graph-shape matching.
        """
        architecture = copy.deepcopy(architecture or {})
        embedded_custom = copy.deepcopy(embedded_custom or {})
        current = self._current_model_component()
        current_custom = copy.deepcopy(self.state.get("custom_components") or {})

        referenced = {
            node.get("definition_id")
            for node in architecture.get("nodes") or []
            if node.get("type") == "custom" and node.get("definition_id")
        }
        available = set(embedded_custom) | set(current_custom)
        missing = sorted(x for x in referenced if x not in available)
        if not missing:
            return architecture, embedded_custom, {
                "recovered": False,
                "remapped": {},
                "source": None,
            }

        if not current.get("nodes"):
            return architecture, embedded_custom, {
                "recovered": False,
                "remapped": {},
                "source": None,
                "missing": missing,
            }

        if self._architecture_shape_signature(architecture) != self._architecture_shape_signature(current):
            return architecture, embedded_custom, {
                "recovered": False,
                "remapped": {},
                "source": None,
                "missing": missing,
            }

        checkpoint_nodes = architecture.get("nodes") or []
        current_nodes = current.get("nodes") or []
        remap = {}

        for old_node, current_node in zip(checkpoint_nodes, current_nodes):
            if old_node.get("type") != "custom":
                continue
            old_id = old_node.get("definition_id")
            if not old_id or old_id not in missing:
                continue
            new_id = current_node.get("definition_id")
            if not new_id or new_id not in current_custom:
                return architecture, embedded_custom, {
                    "recovered": False,
                    "remapped": {},
                    "source": None,
                    "missing": missing,
                }
            prior = remap.get(old_id)
            if prior is not None and prior != new_id:
                return architecture, embedded_custom, {
                    "recovered": False,
                    "remapped": {},
                    "source": None,
                    "missing": missing,
                }
            remap[old_id] = new_id

        if set(remap) != set(missing):
            return architecture, embedded_custom, {
                "recovered": False,
                "remapped": remap,
                "source": None,
                "missing": missing,
            }

        for node in checkpoint_nodes:
            if node.get("type") == "custom" and node.get("definition_id") in remap:
                node["definition_id"] = remap[node["definition_id"]]

        recovered_custom = copy.deepcopy(embedded_custom)
        for new_id in set(remap.values()):
            recovered_custom[new_id] = copy.deepcopy(current_custom[new_id])

        return architecture, recovered_custom, {
            "recovered": True,
            "remapped": remap,
            "source": "current_model",
            "missing": [],
        }

    def _restore_model_checkpoint(self, path, *, allow_unsafe_legacy_checkpoint=False):
        """Restore either a unified MLBricks artifact or a legacy Builder checkpoint."""
        import torch

        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Model checkpoint/artifact was not found: {path_obj}")

        package = {}
        source_entry = {}
        artifact_info = None
        payload = None
        is_artifact = path_obj.is_dir() and (path_obj / "model.pt").exists()

        if is_artifact:
            try:
                mlbricks_inspect = IMPORT_POOL.resolve_api("lifecycle.inspect")
                artifact_info = mlbricks_inspect(path_obj)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not inspect MLBricks model artifact {path_obj}: {exc}"
                ) from exc
            if artifact_info.get("format") != "mlbricks.model":
                raise RuntimeError(
                    f"Unsupported model artifact format: {artifact_info.get('format')!r}."
                )
            metadata = copy.deepcopy(artifact_info.get("metadata") or {})
            if str(metadata.get("kind") or "").lower() == "training_checkpoint":
                raise RuntimeError(
                    "This path is an intermediate training checkpoint, not a separate Studio model. "
                    "Load the model's 'last' artifact instead. Training checkpoints are hidden from the model repository."
                )
            package = copy.deepcopy(metadata.get("builder_package") or {})
            source_entry = copy.deepcopy(package.get("model_entry") or {})
        else:
            payload = safe_torch_load(path_obj, map_location="cpu", allow_unsafe_pickle=allow_unsafe_legacy_checkpoint)
            if allow_unsafe_legacy_checkpoint:
                self._unsafe_legacy_checkpoints.add(str(path_obj))
            if not isinstance(payload, dict) or "model_state" not in payload:
                raise RuntimeError(
                    "The selected file is neither an MLBricks model artifact nor a "
                    "legacy MLB Studio checkpoint (model_state was not found)."
                )
            package = copy.deepcopy(payload.get("builder_package") or {})
            source_entry = copy.deepcopy(package.get("model_entry") or payload.get("model_entry") or {})

        candidate_name = str(source_entry.get("name") or path_obj.parent.name or path_obj.name).strip()
        candidate_key = self._normalized_model_name(candidate_name)
        candidate_path = str(path_obj)
        for existing in self.state.get("model_outputs") or []:
            existing_key = self._normalized_model_name(existing.get("name"))
            existing_paths = set()
            for key in ("path", "checkpoint_path", "local_path"):
                value = existing.get(key)
                if value:
                    try:
                        existing_paths.add(str(Path(value).expanduser().resolve()))
                    except Exception:
                        existing_paths.add(str(value))
            if candidate_path in existing_paths:
                # Scanning a directory again should be idempotent. Do not create
                # another registry item for the exact same artifact.
                return existing
            if candidate_key and existing_key == candidate_key:
                raise FileExistsError(
                    f'Model "{candidate_name}" already exists in Studio. '
                    'Delete/rename the existing model or use a different model name; duplicate model entries are not created.'
                )

        architecture = copy.deepcopy(package.get("model_component") or source_entry.get("architecture") or {})
        if not architecture.get("nodes"):
            kind = "artifact" if is_artifact else "checkpoint"
            raise RuntimeError(
                f"The {kind} contains weights but no Builder model architecture. "
                "Load the matching Builder project first, or train/save once with "
                "a current MLB Studio release."
            )

        custom_components = copy.deepcopy(package.get("custom_components") or {})
        architecture, custom_components, legacy_recovery = self._recover_legacy_custom_components(
            architecture, custom_components,
        )
        referenced = {
            n.get("definition_id")
            for n in architecture.get("nodes") or []
            if n.get("type") == "custom" and n.get("definition_id")
        }
        available = set(custom_components) | set(self.state.get("custom_components") or {})
        missing = sorted(x for x in referenced if x not in available)
        if missing:
            raise RuntimeError(
                "The saved model references custom components missing from this artifact: "
                + ", ".join(missing)
                + ". Open/build the matching model project and scan again."
            )

        model_ws = (self.state.get("workspaces") or {}).get("model") or {}
        root_id = model_ws.get("root_component_id") or self.state.get("root_component_id")
        if not root_id:
            raise RuntimeError("Model Builder workspace is unavailable.")
        architecture["id"] = root_id
        self.state.setdefault("components", {})[root_id] = architecture
        self.state["view_component_id"] = root_id
        if project_executable_features({"custom_components": custom_components}):
            self._mark_external_project_untrusted("loaded-model-artifact")
        self.state.setdefault("custom_components", {}).update(custom_components)
        self.state.setdefault("component_cache", {}).update(
            copy.deepcopy(package.get("component_cache") or {})
        )

        project = copy.deepcopy(package.get("project") or {})
        current_project = self.state.setdefault("project", {})
        for key in ("name", "context_length", "batch_size", "model_settings", "estimated_parameters"):
            if key in project:
                current_project[key] = project[key]

        new_id = f"model_{uuid.uuid4().hex[:12]}"
        source_entry.update({
            "id": new_id,
            "architecture": copy.deepcopy(architecture),
            "custom_components_snapshot": copy.deepcopy(custom_components),
            "path": str(path_obj),
            "checkpoint_path": str(path_obj),
            "weights_ready": True,
            "training_status": "trained",
            "status": "trained",
            "format": "MLBricks model artifact" if is_artifact else "PyTorch checkpoint",
            "artifact_format": "mlbricks.model" if is_artifact else None,
            "selected_dataset_id": None,
            "local_source": True,
            "legacy_recovered": bool(legacy_recovery.get("recovered")),
            "legacy_definition_remap": copy.deepcopy(legacy_recovery.get("remapped") or {}),
        })

        if is_artifact:
            metadata = copy.deepcopy((artifact_info or {}).get("metadata") or {})
            source_entry["trained_steps"] = metadata.get("step", source_entry.get("trained_steps"))
            source_entry["tokens_seen"] = metadata.get("tokens_seen", source_entry.get("tokens_seen"))
            source_entry["effective_vocab_size"] = metadata.get("vocab_size", source_entry.get("effective_vocab_size"))
            source_entry["parameter_count"] = (artifact_info or {}).get("parameters", source_entry.get("parameter_count"))
        else:
            source_entry["trained_steps"] = payload.get("step", source_entry.get("trained_steps"))
            source_entry["tokens_seen"] = payload.get("tokens_seen", source_entry.get("tokens_seen"))
            source_entry["effective_vocab_size"] = payload.get("vocab_size", source_entry.get("effective_vocab_size"))

        dataset_meta = copy.deepcopy(package.get("dataset_meta") or {})
        tokenizer_dir = path_obj / "tokenizer" if is_artifact else None
        if tokenizer_dir is not None and tokenizer_dir.exists():
            dataset_meta.setdefault("pipeline", {}).setdefault("tokenizer", {})["tokenizer_name"] = str(tokenizer_dir)
            source_entry["tokenizer_path"] = str(tokenizer_dir)
        if dataset_meta:
            source_entry["hub_dataset_meta"] = dataset_meta

        self.state.setdefault("model_outputs", []).append(source_entry)
        return source_entry

    def _load_local_project_file(self, path):
        path_obj = Path(path).expanduser().resolve()
        raw = path_obj.read_bytes()
        if path_obj.name.lower().endswith(".mlbricks.bin"):
            magic = b"MLBRICKS-BIN-1\\n"
            if not raw.startswith(magic): raise RuntimeError("Invalid MLBricks BIN file.")
            raw = raw[len(magic):]
        payload = json.loads(raw.decode("utf-8"))
        loaded = payload.get("state") if payload.get("format") == "mlb-studio-design" else payload
        if not isinstance(loaded, dict) or not loaded.get("components"):
            raise RuntimeError("The selected file is not a valid MLB Studio project.")
        self.state = loaded
        self.state.setdefault("prepared_datasets", [])
        self.state.setdefault("model_outputs", [])
        self.state.setdefault("project_files", [])
        self.prepared_datasets = {}
        self.trained_models = {}
        self._mark_external_project_untrusted("local-project-file")
        return {"content_type": "project", "name": (self.state.get("project") or {}).get("name") or path_obj.name, "path": str(path_obj)}

    def load_local_runtime_path(self, path, *, content_type="auto", tokenizer_name="gpt2", text_column="text", allow_unsafe_legacy_checkpoint=False):
        from .local_runtime import detect_local_kind
        path_obj = Path(path).expanduser().resolve()
        info = detect_local_kind(path_obj)
        kind = info.get("kind")
        requested = str(content_type or "auto").lower()
        if requested == "dataset" or (requested == "auto" and kind in {"dataset_dir", "data_file"}):
            meta = self.load_local_dataset_path(path_obj, tokenizer_name=tokenizer_name, text_column=text_column)
            return {"content_type": "dataset", "dataset": meta, "name": meta["name"]}
        if requested == "model" or (requested == "auto" and kind in {"model_artifact", "model_checkpoint"}):
            model = self._restore_model_checkpoint(path_obj, allow_unsafe_legacy_checkpoint=allow_unsafe_legacy_checkpoint)
            return {"content_type": "model", "model": model, "name": model.get("name") or path_obj.name}
        if requested == "project" or (requested == "auto" and kind in {"project_json", "project_bin"}):
            return self._load_local_project_file(path_obj)
        if kind == "bundle":
            restored = self._restore_cloud_bundle(path_obj); restored["path"] = str(path_obj); return restored
        raise RuntimeError(f"Could not determine how to load {path_obj}. Detected: {info.get('label')}. Choose Dataset, Model, or Project explicitly if needed.")

    def _existing_local_dataset_paths(self):
        paths = set()
        for entry in self.state.get("prepared_datasets") or []:
            for key in ("path", "local_path"):
                value = entry.get(key)
                if not value:
                    continue
                try:
                    paths.add(str(Path(value).expanduser().resolve()))
                except Exception:
                    paths.add(str(value))
        return paths

    def _require_local_data_runtime(self):
        try:
            import datasets  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Local environment data import requires the `datasets` package. "
                "In Kaggle run: %pip install -q datasets pyarrow pandas, then restart the kernel."
            ) from exc

    def import_data_from_local_path(
        self,
        base_path,
        *,
        max_depth=12,
        max_entries=1000,
        progress_callback=None,
    ):
        from .local_runtime import scan_data_candidates

        self._require_local_data_runtime()
        scan = scan_data_candidates(
            base_path,
            max_entries=int(max_entries or 1000),
            max_depth=int(max_depth or 12),
        )
        candidates = scan.get("entries") or []
        existing = self._existing_local_dataset_paths()
        imported, skipped, errors = [], [], []
        total = max(len(candidates), 1)

        for index, item in enumerate(candidates, start=1):
            path = str(Path(item["path"]).expanduser().resolve())

            if progress_callback:
                progress_callback({
                    "status": "running",
                    "runtime_kind": "local",
                    "phase": "import_data",
                    "overall": min(90, int((index - 1) / total * 90)),
                    "message": f'Scanning data {index}/{len(candidates)} · {Path(path).name}',
                    "current_path": path,
                })

            if path in existing:
                skipped.append({"path": path, "reason": "Already imported"})
                continue

            try:
                kind = item.get("kind")
                if kind in {"dataset_dir", "data_file"}:
                    # Empty text_column means raw CSV/JSON/Parquet files are accepted
                    # even when their text column has another name. Users can process
                    # or select the relevant column later in Data Processing.
                    result = self.load_local_runtime_path(
                        path,
                        content_type="dataset",
                        tokenizer_name="gpt2",
                        text_column="",
                    )
                elif kind == "bundle":
                    result = self.load_local_runtime_path(path, content_type="auto")
                    if result.get("content_type") != "dataset":
                        skipped.append({
                            "path": path,
                            "reason": f'Bundle contains {result.get("content_type")}, not dataset',
                        })
                        continue
                else:
                    skipped.append({"path": path, "reason": "Not a dataset artifact"})
                    continue

                meta = result.get("dataset") or {}
                meta["local_path"] = path
                meta["source_root"] = scan.get("root")
                meta["repository_source"] = "Local Environment"
                meta["local_source"] = True

                # Keep the registered state entry synchronized with the metadata
                # object returned by load_local_runtime_path.
                for state_meta in self.state.get("prepared_datasets") or []:
                    if state_meta.get("id") == meta.get("id"):
                        state_meta.update(copy.deepcopy(meta))
                        break

                imported.append(copy.deepcopy(meta))
                existing.add(path)
            except Exception as exc:
                errors.append({
                    "path": path,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return {
            "root": scan.get("root"),
            "found": len(candidates),
            "imported": imported,
            "imported_count": len(imported),
            "skipped": skipped,
            "skipped_count": len(skipped),
            "errors": errors,
            "error_count": len(errors),
            "truncated": bool(scan.get("truncated")),
        }

    def _existing_local_model_paths(self):
        paths = set()
        for entry in self.state.get("model_outputs") or []:
            for key in ("checkpoint_path", "path", "local_path"):
                value = entry.get(key)
                if not value:
                    continue
                try:
                    paths.add(str(Path(value).expanduser().resolve()))
                except Exception:
                    paths.add(str(value))
        return paths

    def import_models_from_local_path(
        self,
        base_path,
        *,
        max_depth=12,
        max_entries=1000,
        progress_callback=None,
    ):
        from .local_runtime import scan_model_candidates

        scan = scan_model_candidates(
            base_path,
            max_entries=int(max_entries or 1000),
            max_depth=int(max_depth or 12),
        )
        candidates = scan.get("entries") or []
        existing = self._existing_local_model_paths()
        imported, skipped, errors = [], [], []
        total = max(len(candidates), 1)

        for index, item in enumerate(candidates, start=1):
            path = str(Path(item["path"]).expanduser().resolve())

            if progress_callback:
                progress_callback({
                    "status": "running",
                    "runtime_kind": "local",
                    "phase": "import_models",
                    "overall": min(90, int((index - 1) / total * 90)),
                    "message": f'Scanning {index}/{len(candidates)} · {Path(path).name}',
                    "current_path": path,
                })

            if path in existing:
                skipped.append({"path": path, "reason": "Already imported"})
                continue

            try:
                if item.get("kind") in {"model_artifact", "model_checkpoint"}:
                    result = self.load_local_runtime_path(path, content_type="model")
                else:
                    result = self.load_local_runtime_path(path, content_type="auto")
                    if result.get("content_type") != "model":
                        skipped.append({
                            "path": path,
                            "reason": f'Bundle contains {result.get("content_type")}, not model',
                        })
                        continue

                model = result.get("model") or {}
                model["local_path"] = path
                model["source_root"] = scan.get("root")
                model["repository_source"] = "Local Environment"
                imported.append(copy.deepcopy(model))
                existing.add(path)
            except Exception as exc:
                errors.append({
                    "path": path,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return {
            "root": scan.get("root"),
            "found": len(candidates),
            "imported": imported,
            "imported_count": len(imported),
            "skipped": skipped,
            "skipped_count": len(skipped),
            "errors": errors,
            "error_count": len(errors),
            "truncated": bool(scan.get("truncated")),
        }

    @staticmethod
    def _merge_local_import_results(results, *, environment=None):
        merged = {
            "root": None,
            "roots": [],
            "found": 0,
            "imported": [],
            "imported_count": 0,
            "skipped": [],
            "skipped_count": 0,
            "errors": [],
            "error_count": 0,
            "truncated": False,
            "environment": copy.deepcopy(environment or {}),
        }
        for result in results:
            if not result:
                continue
            root = result.get("root")
            if root and root not in merged["roots"]:
                merged["roots"].append(root)
            merged["found"] += int(result.get("found") or 0)
            merged["imported"].extend(copy.deepcopy(result.get("imported") or []))
            merged["skipped"].extend(copy.deepcopy(result.get("skipped") or []))
            merged["errors"].extend(copy.deepcopy(result.get("errors") or []))
            merged["truncated"] = merged["truncated"] or bool(result.get("truncated"))
        merged["imported_count"] = len(merged["imported"])
        merged["skipped_count"] = len(merged["skipped"])
        merged["error_count"] = len(merged["errors"])
        env_name = (environment or {}).get("name") or "Local Environment"
        merged["root"] = env_name + (" · " + ", ".join(merged["roots"]) if merged["roots"] else "")
        return merged

    def _execute_local_command(self, command, progress_callback=None):
        local = command.get("local") or {}; action = str(command.get("action") or "")
        def emit(payload):
            if progress_callback: progress_callback(payload)

        if action == "local_scan":
            scan = self.scan_local_runtime_files(local.get("roots"))
            emit({"status":"done","runtime_kind":"local","phase":"scan","overall":100,"message":f"Found {len(scan.get('entries') or [])} loadable local items.","local_scan":scan})
            return scan

        if action == "local_import_models":
            environment = copy.deepcopy(self.local_environment)
            environment_scan = bool(local.get("environment_scan"))
            roots = list(local.get("roots") or environment.get("roots") or []) if environment_scan else []
            if not roots:
                roots = [local.get("path")]
            roots = [root for root in roots if root]
            emit({"status":"running","runtime_kind":"local","phase":"import_models","overall":2,"message":f'Scanning {environment.get("name") or "local environment"} for models…'})
            parts=[]
            for root in roots:
                parts.append(self.import_models_from_local_path(
                    root,
                    max_depth=local.get("max_depth") or 12,
                    max_entries=local.get("max_entries") or 1000,
                    progress_callback=emit,
                ))
            result = self._merge_local_import_results(parts, environment=environment) if environment_scan else parts[0]
            message = f'Imported {result["imported_count"]} model{"s" if result["imported_count"] != 1 else ""} from {result["root"]}.'
            if result["skipped_count"]:
                message += f' {result["skipped_count"]} duplicate/non-model item(s) skipped.'
            if result["error_count"]:
                message += f' {result["error_count"]} incompatible/older checkpoint(s) reported.'
            emit({
                "status":"done","runtime_kind":"local","phase":"import_models","overall":100,
                "message":message,"local_import":result,"local_import_type":"model","state_replace":self.to_dict()
            })
            return result

        if action == "local_import_data":
            environment = copy.deepcopy(self.local_environment)
            environment_scan = bool(local.get("environment_scan"))
            roots = list(local.get("roots") or environment.get("roots") or []) if environment_scan else []
            if not roots:
                roots = [local.get("path")]
            roots = [root for root in roots if root]
            emit({"status":"running","runtime_kind":"local","phase":"import_data","overall":2,"message":f'Scanning {environment.get("name") or "local environment"} for datasets…'})
            parts=[]
            for root in roots:
                parts.append(self.import_data_from_local_path(
                    root,
                    max_depth=local.get("max_depth") or 12,
                    max_entries=local.get("max_entries") or 1000,
                    progress_callback=emit,
                ))
            result = self._merge_local_import_results(parts, environment=environment) if environment_scan else parts[0]
            message = f'Imported {result["imported_count"]} dataset{"s" if result["imported_count"] != 1 else ""} from {result["root"]}.'
            if result["skipped_count"]:
                message += f' {result["skipped_count"]} duplicate/non-data item(s) skipped.'
            if result["error_count"]:
                message += f' {result["error_count"]} incompatible data item(s) reported.'
            emit({
                "status":"done","runtime_kind":"local","phase":"import_data","overall":100,
                "message":message,"local_import":result,"local_import_type":"data","state_replace":self.to_dict()
            })
            return result

        if action != "local_load":
            raise ValueError(f"Unknown local runtime command: {action!r}")

        emit({"status":"running","runtime_kind":"local","phase":"load","overall":10,"message":"Loading from Kaggle / local filesystem…"})
        result = self.load_local_runtime_path(local.get("path"), content_type=local.get("content_type") or "auto", tokenizer_name=local.get("tokenizer_name") or "gpt2", text_column=local.get("text_column") or "text")
        emit({"status":"done","runtime_kind":"local","phase":"load","overall":100,"message":f"Loaded {result.get('name') or 'local content'}.","local_result":result,"state_replace":self.to_dict()})
        return result

    def _persistence_component_snapshot(self, definition_id):
        definition_id = str(definition_id or "").strip()
        definitions = self.state.get("custom_components") or {}
        if not definition_id or definition_id not in definitions:
            raise ValueError("A saved Module/API Component definition is required.")

        selected = {}
        pending = [definition_id]
        while pending:
            current_id = pending.pop()
            if current_id in selected or current_id not in definitions:
                continue
            definition = copy.deepcopy(definitions[current_id])
            selected[current_id] = definition
            for node in definition.get("nodes") or []:
                child = str(node.get("definition_id") or "").strip()
                if child and child not in selected:
                    pending.append(child)
        return {
            "root_definition_id": definition_id,
            "definitions": selected,
            "component_cache": copy.deepcopy(self.state.get("component_cache") or {}),
        }

    def _persistence_repository_payload(self, kind, config):
        kind = str(kind or "project").strip().lower()
        if kind == "component":
            return self._persistence_component_snapshot(config.get("definition_id"))
        # Models, data pipelines and complete projects are lightweight graph/config
        # snapshots. sanitize_design strips credentials and heavyweight runtime state.
        return {"state": sanitize_design(self.state), "focus_kind": kind}

    def _prepare_persistence_component_restore(self, payload):
        """Build a small browser-side patch for a saved component.

        Opening a Local Repository component does not replace the current model, so
        returning the entire Builder state is unnecessary and expensive in notebook
        bridges. Remap the saved component tree and send only its definitions/cache.
        """
        definitions = copy.deepcopy(payload.get("definitions") or {})
        root_old = str(payload.get("root_definition_id") or "")
        if root_old not in definitions:
            raise RuntimeError("Local component item is missing its root definition.")
        remap = {old_id: f"custom_{uuid.uuid4().hex}" for old_id in definitions}
        restored = {}
        for old_id, definition in definitions.items():
            new_id = remap[old_id]
            definition["id"] = new_id
            definition["gallery_entry_id"] = None
            if old_id == root_old:
                definition["palette_hidden"] = False
                definition["palette_installed"] = True
            else:
                definition["palette_hidden"] = True
            for node in definition.get("nodes") or []:
                child = str(node.get("definition_id") or "")
                if child in remap:
                    node["definition_id"] = remap[child]
            restored[new_id] = definition
        cache = {
            cache_id: copy.deepcopy(item)
            for cache_id, item in (payload.get("component_cache") or {}).items()
            if cache_id and item
        }
        return {
            "root_definition_id": remap[root_old],
            "custom_components": restored,
            "component_cache": cache,
        }

    def _restore_persistence_component(self, payload):
        restore = self._prepare_persistence_component_restore(payload)
        self.state.setdefault("custom_components", {}).update(restore["custom_components"])
        self.state.setdefault("component_cache", {}).update(restore["component_cache"])
        return restore["root_definition_id"]

    def _execute_persistence_command(self, command, progress_callback=None):
        config = command.get("persistence") or {}
        action = str(command.get("action") or "")

        def emit(payload):
            if progress_callback:
                progress_callback(payload)

        if action == "persistence_save_draft":
            project = self.state.setdefault("project", {})
            draft_kind = str(config.get("draft_kind") or "project").strip().lower() or "project"
            requested_id = str(config.get("draft_id") or "").strip()
            if draft_kind == "component":
                component_local_id = str(config.get("component_local_id") or "").strip()
                draft_id = requested_id or (f"draft_{component_local_id}" if component_local_id else f"draft_component_{uuid.uuid4().hex}")
                draft_name = str(config.get("draft_name") or "Untitled Component").strip() or "Untitled Component"
                workspace = "component"
                # Never replace the owning project's stable ID with a component
                # draft ID. Component drafts are independent recovery records.
                project.setdefault("local_id", f"project_{uuid.uuid4().hex}")
            else:
                draft_id = requested_id or str(project.get("local_id") or f"project_{uuid.uuid4().hex}")
                project["local_id"] = str(project.get("local_id") or draft_id)
                draft_name = str(config.get("draft_name") or project.get("name") or "Untitled Model").strip() or "Untitled Model"
                workspace = str(config.get("workspace") or self.state.get("active_workspace") or "model")
            result = self.persistence.save_draft(
                draft_id,
                draft_name,
                self.state,
                workspace=workspace,
            )
            result["draft_kind"] = draft_kind
            if config.get("sync_hash"):
                result["sync_hash"] = str(config.get("sync_hash"))
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "save_draft",
                "overall": 100, "message": "Draft autosaved locally.",
                "persistence_result": result,
                "persistence_summary": self.persistence.summary(),
            })
            return result

        if action == "persistence_prepare_draft":
            draft_id = str(config.get("draft_id") or "").strip()
            incoming = self.persistence.load_draft(draft_id)
            if not isinstance(incoming, dict) or not incoming.get("components"):
                raise FileNotFoundError("Local draft was not found or is invalid.")
            # Load is intentionally non-destructive. The browser keeps this
            # prepared snapshot in memory and only applies it when the user
            # explicitly presses Recover. This separates the slow DB/notebook
            # transfer from the navigation action and makes Recover instant.
            prepared = copy.deepcopy(incoming)
            prepared.setdefault("project", {}).setdefault(
                "local_id", draft_id or f"project_{uuid.uuid4().hex}"
            )
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "prepare_draft",
                "overall": 100, "message": f'Draft {(prepared.get("project") or {}).get("name") or "design"} loaded and ready to recover.',
                "prepared_draft": {
                    "draft_id": draft_id,
                    "updated_at": config.get("updated_at"),
                    "state": prepared,
                },
            })
            return prepared

        if action == "persistence_load_draft":
            incoming = self.persistence.load_draft(config.get("draft_id"))
            if not isinstance(incoming, dict) or not incoming.get("components"):
                raise FileNotFoundError("Local draft was not found or is invalid.")
            self.state = incoming
            self.state.setdefault("project", {}).setdefault("local_id", str(config.get("draft_id") or f"project_{uuid.uuid4().hex}"))
            self._apply_local_workspace_defaults()
            self._ensure_managed_artifact_index_current()
            self._hydrate_indexed_prepared_datasets()
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "load_draft",
                "overall": 100, "message": f'Draft {(self.state.get("project") or {}).get("name") or "design"} recovered.',
                # Loading does not change repository metadata, so avoid rebuilding
                # the Workshop summary (which parses every saved draft payload).
                "state_replace": self.to_dict(),
            })
            return self.state

        if action == "persistence_delete_draft":
            removed = self.persistence.delete_draft(config.get("draft_id"))
            summary = self.persistence.summary()
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "delete_draft",
                "overall": 100, "message": "Draft removed." if removed else "Draft was already absent.",
                "persistence_summary": summary,
            })
            return removed

        if action == "persistence_list":
            summary = self.persistence.summary()
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "list",
                "overall": 100, "message": "Local repository refreshed.",
                "persistence_summary": summary,
            })
            return summary

        if action == "persistence_save_item":
            kind = str(config.get("kind") or "project").lower()
            name = str(config.get("name") or (self.state.get("project") or {}).get("name") or "Untitled").strip()

            component_local_id = ""
            if kind == "component":
                definition_id = str(config.get("definition_id") or "").strip()
                definition = (self.state.get("custom_components") or {}).get(definition_id)
                if not isinstance(definition, dict):
                    raise ValueError("A saved Module/API Component definition is required.")
                component_local_id = str(config.get("component_local_id") or definition.get("local_id") or "").strip()
                if not component_local_id:
                    component_local_id = f"component_{uuid.uuid4().hex}"
                definition["local_id"] = component_local_id
                config["component_local_id"] = component_local_id

            payload = self._persistence_repository_payload(kind, config)
            metadata = {
                "design_only": True,
                "contains_model_parameters": False,
                "source_project_id": (self.state.get("project") or {}).get("local_id"),
            }
            item_id = config.get("item_id") or None
            if kind == "component":
                metadata["component_local_id"] = component_local_id
                # A logical component owns one Local Repository record. Saving it
                # again updates that record instead of creating duplicates.
                item_id = item_id or f"local_{component_local_id}"

            result = self.persistence.save_repository_item(
                kind=kind,
                name=name,
                payload=payload,
                item_id=item_id,
                metadata=metadata,
            )

            if kind == "component":
                # Component editor drafts exist only while the component is
                # unsaved. Commit + draft removal happen in the same Python
                # persistence command so the Workshop cannot briefly retain a
                # saved component as a draft because of bridge timing.
                source_draft_id = str(config.get("source_draft_id") or f"draft_{component_local_id}").strip()
                if source_draft_id:
                    self.persistence.delete_draft(source_draft_id)
                    result["cleared_draft_id"] = source_draft_id
                result["component_local_id"] = component_local_id

            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "save_item",
                "overall": 100, "message": f'{name} saved to Local Repository (design only).',
                "persistence_result": result,
                "persistence_summary": self.persistence.summary(),
            })
            return result

        if action == "persistence_prepare_item":
            item_id = str(config.get("item_id") or "").strip()
            item = self.persistence.load_repository_item(item_id)
            if not item:
                raise FileNotFoundError("Local Repository item was not found.")
            payload = item.get("payload") or {}
            prepared = {
                "item_id": item_id,
                "updated_at": item.get("updated_at"),
                "kind": item.get("kind"),
                "name": item.get("name"),
            }
            if item.get("kind") == "component":
                prepared["component_restore"] = self._prepare_persistence_component_restore(payload)
            else:
                incoming = payload.get("state")
                if not isinstance(incoming, dict) or not incoming.get("components"):
                    raise RuntimeError("Saved Local Repository item does not contain a valid Builder design.")
                loaded_state = copy.deepcopy(incoming)
                loaded_state.setdefault("project", {}).setdefault("local_id", f"project_{uuid.uuid4().hex}")
                prepared["state"] = loaded_state
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "prepare_item",
                "overall": 100, "message": f'{item.get("name") or "Design"} loaded and ready to open.',
                "prepared_item": prepared,
            })
            return prepared

        if action == "persistence_load_item":
            item = self.persistence.load_repository_item(config.get("item_id"))
            if not item:
                raise FileNotFoundError("Local Repository item was not found.")
            payload = item.get("payload") or {}
            state_replace = None
            component_restore = None
            if item.get("kind") == "component":
                component_restore = self._prepare_persistence_component_restore(payload)
                root_definition_id = component_restore["root_definition_id"]
                result = {"item": {k: item[k] for k in ("id", "kind", "name", "updated_at")}, "root_definition_id": root_definition_id}
            else:
                incoming = payload.get("state")
                if not isinstance(incoming, dict) or not incoming.get("components"):
                    raise RuntimeError("Saved Local Repository item does not contain a valid Builder design.")
                self.state = incoming
                self.state.setdefault("project", {}).setdefault("local_id", f"project_{uuid.uuid4().hex}")
                self._apply_local_workspace_defaults()
                self._ensure_managed_artifact_index_current()
                self._hydrate_indexed_prepared_datasets()
                result = {"item": {k: item[k] for k in ("id", "kind", "name", "updated_at")}}
                state_replace = self.to_dict()
            event = {
                "status": "done", "runtime_kind": "persistence", "phase": "load_item",
                "overall": 100, "message": f'{item.get("name") or "Design"} loaded from Local Repository.',
                "persistence_result": result,
            }
            # Opening does not mutate the Local Repository listing. Avoid the
            # expensive full summary rebuild on this latency-sensitive path.
            if state_replace is not None:
                event["state_replace"] = state_replace
            if component_restore is not None:
                event["component_restore"] = component_restore
            emit(event)
            return result

        if action == "persistence_delete_item":
            removed = self.persistence.delete_repository_item(config.get("item_id"))
            summary = self.persistence.summary()
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "delete_item",
                "overall": 100, "message": "Local Repository item removed." if removed else "Local Repository item was already absent.",
                "persistence_summary": summary,
            })
            return removed

        if action == "persistence_save_credentials":
            provider = str(config.get("provider") or "").lower()
            name = str(config.get("name") or "Default")
            result = self.persistence.save_credentials(provider, name, config.get("credentials") or {})
            message = (
                f'{provider} credential "{name}" saved in the operating-system credential store.'
                if result.get("persistent")
                else f'{provider} credential "{name}" is available for this session; only its masked reference was saved locally because no OS keyring is available.'
            )
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "save_credentials",
                "overall": 100, "message": message,
                "credential_result": result,
                "persistence_summary": self.persistence.summary(),
            })
            return result

        if action == "persistence_delete_credentials":
            provider = str(config.get("provider") or "").lower()
            name = str(config.get("name") or "Default")
            removed = self.persistence.delete_credentials(provider, name)
            emit({
                "status": "done", "runtime_kind": "persistence", "phase": "delete_credentials",
                "overall": 100, "message": "Saved credential removed." if removed else "Saved credential was already absent.",
                "persistence_summary": self.persistence.summary(),
            })
            return removed

        raise ValueError(f"Unknown persistence command: {action!r}")

    def _resolve_cloud_credentials(self, provider, cloud):
        direct = dict(cloud.get("credentials") or {})
        profile = str(cloud.get("credential_name") or "").strip()
        if profile:
            saved = self.persistence.get_credentials(provider, profile)
            # Explicit non-empty values override a saved field without ever being
            # copied into the persistent Builder state.
            saved.update({k: v for k, v in direct.items() if v not in {None, ""}})
            direct = saved
        return direct

    def _cloud_provider_status(self, provider, cloud):
        provider = str(provider or "").lower()
        credentials = self._resolve_cloud_credentials(provider, cloud)
        if provider == "huggingface":
            return self.hub_status(token=credentials.get("token"))
        from . import cloud as cloud_backend
        if provider == "github":
            return cloud_backend.github_status(token=credentials.get("token") or "")
        raise ValueError(f"Unknown cloud provider: {provider!r}")

    def _push_generic_cloud(self, provider, cloud, progress_callback=None):
        from . import cloud as cloud_backend
        provider = str(provider).lower()
        credentials = self._resolve_cloud_credentials(provider, cloud)
        content_type = cloud.get("content_type") or "project"
        artifact_id = cloud.get("artifact_id")
        name = self._safe_cloud_name(
            cloud.get("name")
            or (
                (self.state.get("project") or {}).get("name")
                if content_type == "project"
                else artifact_id
            )
        )
        with tempfile.TemporaryDirectory(prefix="mlbricks_cloud_push_") as td:
            archive = Path(td) / f"{name}.mlbricks.zip"
            if progress_callback:
                progress_callback("packaging", 18, "Packaging local content for upload…", None, None)
            self._create_cloud_bundle(content_type, artifact_id, archive)
            total = archive.stat().st_size if archive.exists() else None
            if progress_callback:
                progress_callback("upload", 28, "Uploading bundle to remote storage…", 0, total)

            def transfer(done, total_bytes):
                if self._stop_event.is_set():
                    raise PipelineStopped()
                if total_bytes:
                    fraction = max(0.0, min(1.0, float(done or 0) / float(total_bytes)))
                    pct = 28 + int(round(fraction * 62))
                else:
                    pct = 55
                if progress_callback:
                    progress_callback("upload", pct, "Uploading bundle to remote storage…", done, total_bytes)

            if self._stop_event.is_set():
                raise PipelineStopped()
            if provider == "github":
                return cloud_backend.github_upload(
                    archive,
                    repo=cloud.get("repo"),
                    path_in_repo=cloud.get("object_path") or f"mlbricks/{archive.name}",
                    branch=cloud.get("branch") or "main",
                    token=credentials.get("token") or "",
                    commit_message=f"Push {content_type} from MLB Studio",
                    progress_callback=transfer,
                )
        raise ValueError(f"Unsupported generic cloud provider: {provider!r}")

    def _load_generic_cloud(self, provider, cloud, progress_callback=None):
        from . import cloud as cloud_backend
        provider = str(provider).lower()
        credentials = self._resolve_cloud_credentials(provider, cloud)
        with tempfile.TemporaryDirectory(prefix="mlbricks_cloud_load_") as td:
            archive = Path(td) / "download.mlbricks.zip"
            if progress_callback:
                progress_callback("download", 24, "Downloading remote bundle…", 0, None)

            def transfer(done, total_bytes):
                if self._stop_event.is_set():
                    raise PipelineStopped()
                if total_bytes:
                    fraction = max(0.0, min(1.0, float(done or 0) / float(total_bytes)))
                    pct = 24 + int(round(fraction * 56))
                else:
                    pct = 52
                if progress_callback:
                    progress_callback("download", pct, "Downloading remote bundle…", done, total_bytes)

            if self._stop_event.is_set():
                raise PipelineStopped()
            if provider != "github":
                raise ValueError(f"Unsupported generic cloud provider: {provider!r}")
            result = cloud_backend.github_download(
                archive,
                repo=cloud.get("repo"),
                path_in_repo=cloud.get("object_path"),
                branch=cloud.get("branch") or "main",
                token=credentials.get("token") or None,
                progress_callback=transfer,
            )
            if self._stop_event.is_set():
                raise PipelineStopped()
            if progress_callback:
                size = archive.stat().st_size if archive.exists() else None
                progress_callback("restore", 86, "Restoring downloaded content into Studio…", size, size)
            restored = self._restore_cloud_bundle(archive)
            result["restored"] = restored
            return result

    @staticmethod
    def _cloud_target(provider, cloud):
        provider = str(provider or "").lower()
        if provider == "huggingface":
            return {"repository": cloud.get("repo"), "revision": cloud.get("revision") or "main"}
        if provider == "github":
            return {"repository": cloud.get("repo"), "branch": cloud.get("branch") or "main", "path": cloud.get("object_path")}
        return {}

    def _execute_cloud_command(self, command, progress_callback=None):
        cloud = command.get("cloud") or {}
        action = str(command.get("action") or "")
        provider = str(cloud.get("provider") or "huggingface").lower()
        if provider not in {"huggingface", "github"}:
            raise ValueError(
                f"Unsupported cloud provider: {provider!r}. "
                "MLBricks Studio supports Hugging Face and GitHub."
            )
        content_type = str(cloud.get("content_type") or "project")
        target = self._cloud_target(provider, cloud)

        def emit(status, phase, overall, message, **extra):
            payload = {
                "status": status,
                "runtime_kind": "cloud",
                "phase": phase,
                "overall": int(max(0, min(100, overall))),
                "message": message,
                "cloud_action": action,
                "cloud_provider": provider,
                "cloud_content_type": content_type,
                "cloud_target": target,
            }
            payload.update(extra)
            if progress_callback:
                progress_callback(payload)

        if action == "cloud_status":
            emit("running", "connection", 5, f"Checking {provider} connection…")
            status = self._cloud_provider_status(provider, cloud)
            emit(
                "done", "status", 100,
                status.get("message", "Connection checked."),
                cloud_status={"provider": provider, **status},
            )
            return status

        emit("running", "connection", 4, f"Connecting to {provider}…")
        connection = self._cloud_provider_status(provider, cloud)
        emit(
            "running", "connection", 10,
            connection.get("message", f"{provider} connection ready."),
            cloud_status={"provider": provider, **connection},
        )
        if self._stop_event.is_set():
            raise PipelineStopped()

        if provider == "huggingface":
            credentials = self._resolve_cloud_credentials(provider, cloud)
            token = credentials.get("token")
            requested_repo_id = str(cloud.get("repo") or "").strip()
            from .hub import resolve_repo_id
            repo_id = resolve_repo_id(requested_repo_id, token=token)
            revision = str(cloud.get("revision") or "main").strip() or "main"
            private = bool(cloud.get("private", True))
            artifact_id = cloud.get("artifact_id")
            target = {"repository": repo_id, "revision": revision}
            if repo_id and repo_id != requested_repo_id:
                emit(
                    "running", "namespace", 16,
                    f"Resolved Hugging Face namespace to {repo_id.split('/', 1)[0]}.",
                )
            if action == "cloud_push":
                emit("running", "upload", 22, f"Uploading {content_type} to {repo_id}…")
                if content_type == "dataset":
                    result = self.push_dataset_to_hub(artifact_id, repo_id, private=private, token=token)
                elif content_type == "model":
                    result = self.push_model_to_hub(artifact_id, repo_id, private=private, token=token)
                else:
                    result = self.push_project_to_hub(repo_id, private=private, token=token)
                emit("running", "finalizing", 94, "Finalizing Hugging Face repository metadata…")
                message = f'{content_type.title()} pushed to {repo_id}.'
            elif action == "cloud_load":
                emit("running", "download", 22, f"Downloading {content_type} from {repo_id}@{revision}…")
                if content_type == "dataset":
                    restored = self.load_dataset_from_hub(repo_id, revision=revision, token=token)
                    result = {"restored": {"content_type": "dataset", "dataset": restored}}
                elif content_type == "model":
                    restored = self.load_model_from_hub(repo_id, revision=revision, token=token)
                    result = {"restored": {"content_type": "model", "model": restored}}
                else:
                    result = self.load_project_from_hub(repo_id, revision=revision, token=token)
                emit("running", "restore", 92, "Registering downloaded content in Studio…")
                message = f'{content_type.title()} loaded from {repo_id}.'
            else:
                raise ValueError(f"Unknown cloud action: {action!r}")
        else:
            def stage(phase, pct, msg, done=None, total=None):
                emit("running", phase, pct, msg, bytes_done=done, bytes_total=total)
            if action == "cloud_push":
                result = self._push_generic_cloud(provider, cloud, progress_callback=stage)
                emit("running", "finalizing", 94, "Finalizing remote upload…")
                message = f'{content_type.title()} pushed to {provider}.'
            elif action == "cloud_load":
                result = self._load_generic_cloud(provider, cloud, progress_callback=stage)
                emit("running", "finalizing", 94, "Finalizing restored content…")
                message = f'Content loaded from {provider}.'
            else:
                raise ValueError(f"Unknown cloud action: {action!r}")

        emit(
            "done", action, 100, message,
            cloud_result=result,
            state_replace=self.to_dict(),
        )
        return result

    def _execute_hub_command(self, command, progress_callback=None):
        hub = command.get("hub") or {}
        action = str(command.get("action") or "")
        repo_id = str(hub.get("repo_id") or "").strip()
        revision = str(hub.get("revision") or "main").strip() or "main"
        private = bool(hub.get("private", True))

        def emit(payload):
            if progress_callback:
                progress_callback(payload)

        if action == "hub_status":
            status = self.hub_status()
            emit({
                "status": "done", "runtime_kind": "hub", "phase": "status",
                "overall": 100, "message": status.get("message", "Hub status checked."),
                "hub_status": status,
            })
            return status

        emit({
            "status": "running", "runtime_kind": "hub", "phase": action,
            "overall": 5, "message": "Connecting to Hugging Face Hub…",
        })

        if action == "hub_push_dataset":
            result = self.push_dataset_to_hub(
                hub.get("artifact_id"), repo_id, private=private
            )
            message = f'Dataset pushed to {result["repo_id"]}.'
        elif action == "hub_load_dataset":
            meta = self.load_dataset_from_hub(repo_id, revision=revision)
            result = {"dataset": meta, "repo_id": repo_id, "url": meta.get("hub_url")}
            message = f'Dataset loaded: {meta["name"]}.'
        elif action == "hub_push_model":
            result = self.push_model_to_hub(
                hub.get("artifact_id"), repo_id, private=private
            )
            message = f'Model pushed to {result["repo_id"]}.'
        elif action == "hub_load_model":
            model = self.load_model_from_hub(repo_id, revision=revision)
            result = {"model": model, "repo_id": repo_id, "url": model.get("hub_url")}
            message = f'Model loaded: {model.get("name", repo_id)}.'
        elif action == "hub_push_project":
            result = self.push_project_to_hub(repo_id, private=private)
            message = f'Project pushed to {result["repo_id"]}.'
        elif action == "hub_load_project":
            result = self.load_project_from_hub(repo_id, revision=revision)
            message = f'Project loaded from {result["repo_id"]}.'
        else:
            raise ValueError(f"Unknown Hugging Face command: {action!r}")

        emit({
            "status": "done",
            "runtime_kind": "hub",
            "phase": action,
            "overall": 100,
            "message": message,
            "hub_result": result,
            "state_replace": self.to_dict(),
        })
        return result

    def stop(self):
        """Request that the current pipeline stop after the active step."""
        self._stop_event.set()

    def _publish_bridge_progress(self, payload):
        widgets = self._bridge_widgets or {}
        progress = widgets.get("progress")
        enriched = dict(payload or {})
        enriched["ts"] = time.time()
        with self._progress_event_lock:
            self._progress_event_seq += 1
            enriched["event_seq"] = self._progress_event_seq
            try:
                raw = json.dumps(enriched, default=str)
                safe_event = json.loads(raw)
            except Exception:
                raw = json.dumps({
                    "status":"error", "runtime_kind":"bridge", "overall":0,
                    "message":"Studio could not serialize a runtime progress event.",
                    "ts":time.time(), "event_seq":self._progress_event_seq,
                })
                safe_event = json.loads(raw)
            self._progress_events.append(safe_event)
        if progress is not None:
            try:
                progress.value = raw
            except Exception:
                pass

    def _progress_events_after(self, after=0, limit=512):
        try:
            after = max(0, int(after or 0))
        except Exception:
            after = 0
        limit = max(1, min(1024, int(limit or 512)))
        with self._progress_event_lock:
            events = [
                event for event in self._progress_events
                if int(event.get("event_seq", 0)) > after
            ][:limit]
            latest = self._progress_event_seq
            oldest = int(self._progress_events[0].get("event_seq", latest)) if self._progress_events else latest
        return {
            "events": events,
            "last_seq": int(events[-1].get("event_seq", after)) if events else after,
            "latest_seq": latest,
            "dropped": bool(after and after < max(0, oldest - 1)),
        }

    @staticmethod
    def _bridge_action_from_raw(command_raw):
        try:
            command = json.loads(command_raw or "{}")
        except Exception:
            command = {}
        if not isinstance(command, dict):
            command = {}
        return str(command.get("action") or "data").lower(), command

    def _queue_bridge_widget_request(self, state_raw, command_raw):
        action, command = self._bridge_action_from_raw(command_raw)
        navigation = {"persistence_prepare_draft", "persistence_prepare_item", "persistence_load_draft", "persistence_load_item"}
        background = {"persistence_save_draft", "ensure_component_import"}
        interactive_runtime = {"data", "train", "generate", "serve_start", "serve_stop"}
        request = {
            "state_raw": state_raw or "{}",
            "command_raw": command_raw or "{}",
            "action": action,
            "component_type": str(command.get("component_type") or ""),
        }
        with self._bridge_queue_lock:
            queued_actions = {item.get("action") for item in self._bridge_pending_requests}
            # Never let an old autosave/import run after a requested Recover/Open;
            # its captured state may predate the recovered design.
            if action in background and (self._active_bridge_action in navigation or queued_actions & navigation):
                return False
            if action in navigation:
                self._bridge_pending_requests = [
                    item for item in self._bridge_pending_requests
                    if item.get("action") not in background | navigation
                ]
                self._bridge_pending_requests.insert(0, request)
                return True
            if action == "persistence_save_draft":
                self._bridge_pending_requests = [
                    item for item in self._bridge_pending_requests
                    if item.get("action") != "persistence_save_draft"
                ]
                self._bridge_pending_requests.append(request)
                return True
            if action in interactive_runtime:
                # User-triggered Train/Generate/Serve work must never sit behind
                # stale autosave/import chores. The currently running worker is
                # allowed to finish, then this request is next. Coalesce duplicate
                # runtime clicks for the same model/action while it is queued.
                self._bridge_pending_requests = [
                    item for item in self._bridge_pending_requests
                    if item.get("action") not in background
                    and not (item.get("action") == action and item.get("model_id") == command.get("model_id"))
                ]
                request["model_id"] = command.get("model_id")
                self._bridge_pending_requests.insert(0, request)
                return True
            if action == "ensure_component_import":
                ctype = request["component_type"]
                if any(
                    item.get("action") == action and item.get("component_type") == ctype
                    for item in self._bridge_pending_requests
                ):
                    return False
            self._bridge_pending_requests.append(request)
            return True

    def _start_hot_bridge_generation(self, command):
        """Run resident generation without waiting for notebook housekeeping.

        Full Window and notebook mode share one hidden Run button.  Autosave or
        lazy component-import work can legitimately still be finishing when the
        user presses Generate.  Those jobs do not touch the resident nn.Module,
        so generation gets a dedicated fast lane rather than spending seconds (or
        minutes) queued behind them.  Training/model deletion/memory cleanup are
        intentionally *not* bypassed by the caller.
        """
        model_id = command.get("model_id")
        thread = self._hot_generation_thread
        if thread is not None and thread.is_alive():
            self._publish_bridge_progress({
                "status":"error","runtime_kind":"generate","phase":"busy","overall":0,
                "model_id":model_id,
                "message":"A generation response is already running for the resident model.",
            })
            return False

        if isinstance(command.get("generation_config"), dict):
            generation_entry = self._model_output(model_id)
            generation_entry["generation_config"] = copy.deepcopy(command["generation_config"])

        self._stop_event.clear()
        self._hot_generation_model_id = model_id

        def hot_worker():
            try:
                self.generate_model(model_id, progress_callback=self._publish_bridge_progress)
            except Exception as exc:
                if type(exc).__name__ in {"TrainingStopped", "PipelineStopped"}:
                    self._publish_bridge_progress({
                        "status":"stopped","runtime_kind":"generate","phase":"generate","overall":0,
                        "model_id":model_id,"message":"Runtime stopped.",
                    })
                else:
                    self._remember_run_error(exc)
                    self._publish_bridge_progress({
                        "status":"error","runtime_kind":"generate","phase":"generate","overall":0,
                        "model_id":model_id,"message":f"{type(exc).__name__}: {exc}",
                    })
            finally:
                self._hot_generation_model_id = None

        self._hot_generation_thread = threading.Thread(
            target=hot_worker,
            name=f"mlb-studio-generate-{self._instance_id}",
            daemon=True,
        )
        self._hot_generation_thread.start()
        return True

    def _schedule_bridge_queue_drain(self, delay=0.04):
        with self._bridge_queue_lock:
            if self._bridge_drain_scheduled or not self._bridge_pending_requests:
                return
            self._bridge_drain_scheduled = True

        def drain_later():
            try:
                self._drain_bridge_queue()
            finally:
                pass

        timer = threading.Timer(delay, drain_later)
        timer.daemon = True
        timer.start()

    def _drain_bridge_queue(self):
        with self._bridge_queue_lock:
            self._bridge_drain_scheduled = False
            if self._run_thread is not None and self._run_thread.is_alive():
                needs_retry = bool(self._bridge_pending_requests)
                request = None
            else:
                needs_retry = False
                request = self._bridge_pending_requests.pop(0) if self._bridge_pending_requests else None
        if needs_retry:
            self._schedule_bridge_queue_drain(0.05)
            return
        if not request:
            return
        widgets = self._bridge_widgets or {}
        state_widget = widgets.get("state")
        command_widget = widgets.get("command")
        if state_widget is None:
            return
        try:
            state_widget.value = request.get("state_raw") or "{}"
            if command_widget is not None:
                command_widget.value = request.get("command_raw") or "{}"
        except Exception as exc:
            self._publish_bridge_progress({
                "status": "error", "runtime_kind": "bridge", "overall": 0,
                "message": f"Could not restore queued Studio action: {exc}",
            })
            self._schedule_bridge_queue_drain()
            return
        self._start_bridge_run()

    def _start_bridge_run(self):
        widgets = self._bridge_widgets or {}
        state_widget = widgets.get("state")
        command_widget = widgets.get("command")

        if self._run_thread is not None and self._run_thread.is_alive():
            # Capture the exact command/state now. The previous implementation
            # discarded the click while a background import/autosave was active.
            state_raw = getattr(state_widget, "value", "{}") if state_widget is not None else "{}"
            command_raw = getattr(command_widget, "value", "{}") if command_widget is not None else "{}"
            preview_action, preview_command = self._bridge_action_from_raw(command_raw)
            if command_widget is not None:
                try:
                    command_widget.value = "{}"
                except Exception:
                    pass

            # Generation is latency-sensitive and state-independent.  Do not let
            # notebook housekeeping become part of model response latency.  The
            # resident model can safely run while these background jobs finish.
            generation_safe_background = {
                "persistence_save_draft", "ensure_component_import",
                "ensure_external_import", "validate_user_function",
                "validate_user_class",
            }
            if preview_action == "generate" and self._active_bridge_action in generation_safe_background:
                self._start_hot_bridge_generation(preview_command)
                return

            self._queue_bridge_widget_request(state_raw, command_raw)
            self._schedule_bridge_queue_drain()
            return

        command = {}

        # v0.7.6: runtime actions use a dedicated tiny command widget instead of
        # being embedded in the entire project JSON. This avoids Kaggle iframe
        # races for Local/Data/Model/Cloud actions and keeps secrets out of state.
        if command_widget is not None:
            try:
                parsed = json.loads(command_widget.value or "{}")
                if isinstance(parsed, dict):
                    command = parsed
                command_widget.value = "{}"
            except Exception:
                command = {}

        action = str(command.get("action") or "data").lower()
        # These operations are fully DB/credential backed and do not consume the
        # current Builder graph. Skipping the often-large state textarea removes a
        # full JSON parse/copy from every Workshop Recover/Open operation.
        state_independent_actions = {
            "persistence_prepare_draft", "persistence_prepare_item",
            "persistence_load_draft", "persistence_load_item",
            "persistence_list", "persistence_delete_draft", "persistence_delete_item",
            "persistence_save_credentials", "persistence_delete_credentials",
            "runtime_clear_memory",
            # Generation carries its tiny prompt/sampling config in command_raw.
            # Do not deserialize/replace the entire Studio project for every chat
            # response; that would sever the Python-hot-path UX from the resident
            # model object and adds needless JSON traffic.
            "generate",
        }
        if state_widget is not None and action not in state_independent_actions:
            try:
                incoming = json.loads(state_widget.value)
                if isinstance(incoming, dict) and incoming.get("components"):
                    legacy_command = incoming.pop("_runtime_command", None) or {}
                    incoming.pop("_session_secrets", None)
                    if not command:
                        command = legacy_command
                    self.state = incoming
                    self._ensure_managed_artifact_index_current()
                    self._hydrate_indexed_prepared_datasets()
            except Exception as exc:
                self._publish_bridge_progress({
                    "status": "error",
                    "message": f"Could not read Builder state: {exc}",
                    "overall": 0,
                    "nodes": {},
                })
                return

        self._stop_event.clear()
        self.last_run_error = None

        model_id = command.get("model_id")
        if action == "generate" and isinstance(command.get("generation_config"), dict):
            # Fast lane: synchronize only the values generation actually needs.
            # The resident nn.Module/tokenizer stay attached to this Builder
            # instance in self.trained_models and are never serialized to JS.
            generation_entry = self._model_output(model_id)
            generation_entry["generation_config"] = copy.deepcopy(command["generation_config"])
        self._active_bridge_action = action
        self._active_bridge_model_id = model_id

        def worker():
            try:
                if action == "ensure_component_import":
                    component_type = str(command.get("component_type") or "").strip()
                    result = self.ensure_component_import(component_type)
                    self._publish_bridge_progress({
                        "status": "done" if result.get("ok") else "error",
                        "runtime_kind": "import",
                        "phase": "component",
                        "overall": 100,
                        "message": result.get("message") or "Component import checked.",
                        "component_import": result,
                        "component_api": result.get("api"),
                        "component_type": component_type,
                    })
                elif action == "ensure_external_import":
                    import_path = str(command.get("import_path") or "").strip()
                    label = str(command.get("label") or import_path).strip()
                    result = self.ensure_external_import(import_path, label=label)
                    self._publish_bridge_progress({
                        "status": "done" if result.get("ok") else "error",
                        "runtime_kind": "external_import",
                        "phase": "custom_component",
                        "overall": 100,
                        "message": result.get("message") or "Custom API import checked.",
                        "external_import": result,
                        "definition_id": command.get("definition_id"),
                    })
                elif action == "validate_user_function":
                    source = str(command.get("source") or "")
                    function_name = str(command.get("function_name") or "").strip()
                    label = str(command.get("label") or function_name).strip()
                    result = self.validate_user_function(source, function_name, label=label)
                    self._publish_bridge_progress({
                        "status": "done" if result.get("ok") else "error",
                        "runtime_kind": "user_function_validation",
                        "phase": "custom_component",
                        "overall": 100,
                        "message": result.get("message") or "User Function checked.",
                        "user_function_validation": result,
                        "definition_id": command.get("definition_id"),
                    })
                elif action == "validate_user_class":
                    source = str(command.get("source") or "")
                    class_name = str(command.get("class_name") or "").strip()
                    label = str(command.get("label") or class_name).strip()
                    result = self.validate_user_class(source, class_name, label=label)
                    self._publish_bridge_progress({
                        "status": "done" if result.get("ok") else "error",
                        "runtime_kind": "user_class_validation",
                        "phase": "custom_component",
                        "overall": 100,
                        "message": result.get("message") or "User Class checked.",
                        "user_class_validation": result,
                        "definition_id": command.get("definition_id"),
                    })
                elif action == "delete_dataset":
                    result = self.delete_prepared_dataset(command.get("dataset_id"))
                    self._publish_bridge_progress({
                        "status":"done","runtime_kind":"maintenance","phase":"delete_dataset","overall":100,
                        "message":f'Deleted {result.get("name") or "dataset"} from the Studio Data Repository.',
                        "maintenance_result":result,
                    })
                elif action == "delete_model":
                    result = self.delete_model_output(command.get("model_id"))
                    self._publish_bridge_progress({
                        "status":"done","runtime_kind":"maintenance","phase":"delete_model","overall":100,
                        "message":f'Deleted {result.get("name") or "model"} from the Studio Model Repository.',
                        "maintenance_result":result,
                    })
                elif action == "runtime_clear_memory":
                    result = self.clear_runtime_memory()
                    if result.get("cuda_available"):
                        before = result.get("before_reserved_gb")
                        after = result.get("after_reserved_gb")
                        detail = "" if before is None or after is None else f" Reserved VRAM {before:.2f} → {after:.2f} GB."
                        message = f'GPU VRAM cache cleaned. Released {result.get("released_model_caches", 0)} cached model runtime(s).{detail}'
                    else:
                        message = f'Runtime memory cleaned. No CUDA GPU is active; released {result.get("released_model_caches", 0)} cached model runtime(s).'
                    self._publish_bridge_progress({
                        "status":"done","runtime_kind":"maintenance","phase":"runtime_clear_memory","overall":100,
                        "message":message,"maintenance_result":result,
                    })
                elif action == "train":
                    self.train_model(
                        model_id,
                        progress_callback=self._publish_bridge_progress,
                        overwrite_existing=bool(command.get("overwrite_existing")),
                        resume_existing=bool(command.get("resume_existing")),
                        start_fresh=bool(command.get("start_fresh")),
                    )
                elif action == "generate":
                    self.generate_model(model_id, progress_callback=self._publish_bridge_progress)
                elif action.startswith("hub_"):
                    self._execute_hub_command(
                        command,
                        progress_callback=self._publish_bridge_progress,
                    )
                elif action.startswith("persistence_"):
                    self._execute_persistence_command(
                        command,
                        progress_callback=self._publish_bridge_progress,
                    )
                elif action.startswith("cloud_"):
                    self._execute_cloud_command(
                        command,
                        progress_callback=self._publish_bridge_progress,
                    )
                elif action.startswith("local_"):
                    self._execute_local_command(
                        command,
                        progress_callback=self._publish_bridge_progress,
                    )
                elif action.startswith("serve_"):
                    self._execute_serve_command(
                        command,
                        progress_callback=self._publish_bridge_progress,
                    )
                else:
                    self.last_data_result = self.run_data_pipeline(
                        progress_callback=self._publish_bridge_progress,
                        overwrite_existing=bool(command.get("overwrite_existing")),
                    )
            except ArtifactConflictError as exc:
                runtime_kind = "train" if action == "train" else "data" if action == "data" else action
                self._publish_bridge_progress({
                    "status": "overwrite_required" if exc.action == "override" else "artifact_action_required",
                    "runtime_kind": runtime_kind,
                    "phase": "overwrite_check" if exc.action == "override" else "artifact_action_check",
                    "overall": 0,
                    "model_id": model_id,
                    "message": str(exc),
                    "overwrite_request": exc.payload(),
                })
            except PipelineStopped:
                stopped_kind = "cloud" if str(action).startswith("cloud_") else action
                stopped_message = "Cloud transfer cancelled." if stopped_kind == "cloud" else f"{action.title()} stopped."
                self._publish_bridge_progress({
                    "status":"stopped","runtime_kind":stopped_kind,"phase":action,"overall":0,
                    "message":stopped_message
                })
            except Exception as exc:
                # Keep model_runtime lazy at Studio startup. TrainingStopped is
                # recognized by type name here rather than importing torch-backed
                # runtime classes before the user starts a model operation.
                if type(exc).__name__ == "TrainingStopped":
                    self._publish_bridge_progress({
                        "status":"stopped","runtime_kind":action,"overall":0,
                        "message":f"{action.title()} stopped."
                    })
                    return
                self._remember_run_error(exc)
                runtime_kind = "serve" if str(action).startswith("serve_") else action
                if str(action).startswith("cloud_"):
                    runtime_kind = "cloud"
                elif str(action).startswith("hub_"):
                    runtime_kind = "hub"
                elif str(action).startswith("local_"):
                    runtime_kind = "local"
                elif str(action).startswith("persistence_"):
                    runtime_kind = "persistence"
                elif action in {"delete_dataset", "delete_model", "runtime_clear_memory"}:
                    runtime_kind = "maintenance"
                error_payload = {
                    "status":"error","runtime_kind":runtime_kind,"phase":action,"overall":0,
                    "message":f"{type(exc).__name__}: {exc}"
                }
                if str(action).startswith("cloud_"):
                    cloud_cfg = command.get("cloud") or {}
                    cloud_provider = str(cloud_cfg.get("provider") or "huggingface").lower()
                    error_payload.update({
                        "cloud_action": action,
                        "cloud_provider": cloud_provider,
                        "cloud_content_type": str(cloud_cfg.get("content_type") or "project"),
                        "cloud_target": self._cloud_target(cloud_provider, cloud_cfg),
                    })
                if runtime_kind == "serve":
                    error_payload.update({
                        "phase": action,
                        "model_id": command.get("model_id"),
                        "model_update": {"serve_status":"error"},
                    })
                self._publish_bridge_progress(error_payload)
            finally:
                self._active_bridge_action = None
                self._active_bridge_model_id = None
                self._schedule_bridge_queue_drain()

        self._run_thread = threading.Thread(
            target=worker,
            name=f"mlb-studio-run-{self._instance_id}",
            daemon=True,
        )
        self._run_thread.start()

    def _dispatch_bridge_request_envelope(self, raw):
        """Atomically dispatch one browser request carrying state + command.

        Standard ipywidgets synchronize each hidden widget independently. On
        hosted notebooks (especially Kaggle), updating the state textarea and
        then clicking a separate hidden Button can race: Python may receive the
        click before the new project state. Data Fetch, Training, and destructive
        repository maintenance are state-sensitive, so the browser sends both
        pieces through one observed textarea. The observer runs in Python and
        starts/queues the exact snapshot without a second browser comm.
        """
        try:
            envelope = json.loads(raw or "{}")
        except Exception as exc:
            self._publish_bridge_progress({
                "status": "error", "runtime_kind": "data", "phase": "dispatch",
                "overall": 0, "nodes": {},
                "message": f"Could not read atomic Studio request: {exc}",
            })
            return False
        if not isinstance(envelope, dict):
            return False

        state_payload = envelope.get("state")
        command = envelope.get("command") or {"action": "data"}
        if not isinstance(command, dict):
            command = {"action": "data"}
        action = str(command.get("action") or "data").lower()
        atomic_actions = {"data", "train", "delete_model", "delete_dataset"}
        if action not in atomic_actions:
            return False
        if not isinstance(state_payload, dict) or not state_payload.get("components"):
            runtime_kind = "maintenance" if action.startswith("delete_") else action
            self._publish_bridge_progress({
                "status": "error", "runtime_kind": runtime_kind, "phase": "dispatch",
                "overall": 0, "nodes": {},
                "message": "Studio request did not include a valid project snapshot.",
            })
            return False

        widgets = self._bridge_widgets or {}
        state_widget = widgets.get("state")
        command_widget = widgets.get("command")
        if state_widget is None or command_widget is None:
            return False

        # We are already in Python here, so these assignments are synchronous.
        # _start_bridge_run() therefore observes exactly this state/command pair.
        state_widget.value = json.dumps(state_payload)
        command_widget.value = json.dumps(command)
        if action == "data":
            self._publish_bridge_progress({
                "status": "running", "runtime_kind": "data", "phase": "dispatch",
                "overall": 0, "nodes": {},
                "message": "Data request accepted by Python…",
                "request_id": envelope.get("request_id"),
            })
        self._start_bridge_run()
        return True

    def _setup_widget_bridge(self):
        """Create a bridge using only standard ipywidgets (no custom frontend module)."""
        try:
            import ipywidgets as widgets
        except Exception:
            return None

        suffix = self._instance_id.replace("-", "_")
        hidden = widgets.Layout(
            width="3px",
            height="3px",
            min_width="3px",
            min_height="3px",
            visibility="hidden",
            overflow="hidden",
        )
        state_widget = widgets.Textarea(value=json.dumps(self.state), layout=hidden)
        command_widget = widgets.Textarea(value="{}", layout=hidden)
        request_widget = widgets.Textarea(value="", layout=hidden)
        run_widget = widgets.Button(description="", layout=hidden)
        stop_widget = widgets.Button(description="", layout=hidden)
        progress_widget = widgets.Textarea(
            value=json.dumps({"status": "idle", "message": "Ready", "overall": 0, "nodes": {}}),
            layout=hidden,
        )

        classes = {
            "state": f"mlb-state-bridge-{suffix}",
            "command": f"mlb-command-bridge-{suffix}",
            "request": f"mlb-request-bridge-{suffix}",
            "run": f"mlb-run-bridge-{suffix}",
            "stop": f"mlb-stop-bridge-{suffix}",
            "progress": f"mlb-progress-bridge-{suffix}",
        }
        state_widget.add_class(classes["state"])
        command_widget.add_class(classes["command"])
        request_widget.add_class(classes["request"])
        run_widget.add_class(classes["run"])
        stop_widget.add_class(classes["stop"])
        progress_widget.add_class(classes["progress"])

        def on_atomic_request(change):
            raw = str((change or {}).get("new") or "")
            if not raw:
                return
            # Clear first so retrying an identical request still emits a change.
            try:
                request_widget.value = ""
            except Exception:
                pass
            try:
                self._dispatch_bridge_request_envelope(raw)
            except Exception as exc:
                self._remember_run_error(exc)
                self._publish_bridge_progress({
                    "status": "error", "runtime_kind": "data", "phase": "dispatch",
                    "overall": 0, "nodes": {},
                    "message": f"{type(exc).__name__}: {exc}",
                })

        request_widget.observe(on_atomic_request, names="value")
        run_widget.on_click(lambda _: self._start_bridge_run())
        stop_widget.on_click(lambda _: self.stop())

        self._bridge_widgets = {
            "state": state_widget,
            "command": command_widget,
            "request": request_widget,
            "run": run_widget,
            "stop": stop_widget,
            "progress": progress_widget,
            "classes": classes,
        }
        return self._bridge_widgets

    def diagnostics(self):
        info = get_mlbricks_info()
        available = [k for k, v in self.mlbricks_api.items() if v.get("available")]
        unavailable = {k: v.get("error") for k, v in self.mlbricks_api.items() if not v.get("available")}
        return {
            "builder_version": __version__,
            "frontend_version": __version__,
            "mlbricks": info,
            "import_pool": self.import_pool.status(),
            "api_components_available": available,
            "api_components_unavailable": unavailable,
        }

    def mlbricks_info(self):
        return get_mlbricks_info()

    def _html(self, bridge=None, *, include_assets=True, allow_full_window=True):
        css_gzip_b64, js_gzip_b64 = _compressed_frontend_bundle() if include_assets else ("", "")
        trust = self.project_trust_info()

        # Parameter schemas already live in catalog[item]["api"].  Do not send
        # the same lists/descriptions a second time inside mlbricks_api.
        frontend_api = {
            component_type: {
                key: value
                for key, value in info.items()
                if key not in {"parameters", "description"}
            }
            for component_type, info in self.mlbricks_api.items()
        }
        payload = json.dumps({
            "state": self.state,
            "catalog": self.catalog,
            "mlbricks_api": frontend_api,
            "bridge": bridge,
            "runtime_capabilities": self.runtime_capabilities,
            "local_environment": self.local_environment,
            "instance_id": self._instance_id,
            "allow_full_window": bool(allow_full_window),
            "project_trust": trust,
            "local_persistence": self.persistence.summary(),
        }, separators=(",", ":")).replace("</", "<\\/")
        # The component catalog grows as Studio gains more educational and research
        # primitives. Compress the per-instance payload in notebook HTML just like
        # the frontend assets so adding components does not make every Builder cell
        # exceed notebook/browser output envelopes. Full-window HTML can still use
        # the raw object because it reuses the already-loaded frontend runtime.
        if include_assets:
            payload_gzip_b64 = base64.b64encode(
                gzip.compress(payload.encode("utf-8"), compresslevel=9)
            ).decode("ascii")
            payload_loader = (
                "window.__MLB_STUDIO_DECODE_GZIP__("
                + json.dumps(payload_gzip_b64)
                + ").then(JSON.parse)"
            )
        else:
            payload_loader = "Promise.resolve(" + payload + ")"
        warning = ""
        if trust.get("requires_trust") and not trust.get("trusted"):
            count = len(trust.get("executable_features") or [])
            warning = (
                '<div class="mlb-security-warning" style="font:600 12px/1.5 system-ui,sans-serif;'
                'padding:10px 14px;margin:0 0 8px;border:1px solid #8a6d2f;border-radius:8px;'
                'background:#2b2212;color:#f4d58a">'
                f'Untrusted project: {count} executable Python/import binding(s) are blocked. '
                'Review them, then run <code>builder.trust_project()</code> in Python to enable execution for this session.'
                '</div>'
            )
        shared_style_id = "mlb-studio-shared-style"
        assets = ""
        if include_assets:
            assets = f"""
<script>
window.__MLB_STUDIO_ASSETS_READY__ = (async function() {{
  async function decodeGzipBase64(value) {{
    const raw = atob(value);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    if (typeof DecompressionStream !== "function") {{
      throw new Error("This browser does not support DecompressionStream required by MLB Studio.");
    }}
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return await new Response(stream).text();
  }}
  window.__MLB_STUDIO_DECODE_GZIP__ = decodeGzipBase64;
  const [cssText, jsText] = await Promise.all([
    decodeGzipBase64({json.dumps(css_gzip_b64)}),
    decodeGzipBase64({json.dumps(js_gzip_b64)})
  ]);
  let style = document.getElementById({json.dumps(shared_style_id)});
  if (!style) {{
    style = document.createElement("style");
    style.id = {json.dumps(shared_style_id)};
    document.head.appendChild(style);
  }}
  style.textContent = cssText;
  window.__MLB_STUDIO_CSS_ELEMENT__ = style;
  try {{ delete window.MLBricksBuilder; }} catch (e) {{ window.MLBricksBuilder = undefined; }}
  const runtimeScript = document.createElement("script");
  runtimeScript.textContent = jsText;
  document.head.appendChild(runtimeScript);
  runtimeScript.remove();
  window.__MLB_STUDIO_FRONTEND_VERSION__ = {json.dumps(__version__)};
}})();
</script>
"""
        allow_full_window_marker = "true" if allow_full_window else "false"
        return f"""
{assets}
{warning}
<!-- compatibility marker: \"allow_full_window\":{allow_full_window_marker} -->
<div id="{html.escape(self._instance_id)}" class="mlb-root" data-mlb-studio-version="{html.escape(__version__)}"></div>
<script>
(function() {{
  const root = document.getElementById({json.dumps(self._instance_id)});
  root.innerHTML = '<div class="mlb-startup-shell"><div class="mlb-startup-mark">MLBRICKS STUDIO</div><div class="mlb-startup-text">Loading workspace…</div></div>';
  Promise.resolve(window.__MLB_STUDIO_ASSETS_READY__).then(function() {{
    if (!window.MLBricksBuilder || typeof window.MLBricksBuilder.mount !== "function") {{
      root.innerHTML = '<div class="mlb-startup-shell"><div class="mlb-startup-mark">MLBRICKS STUDIO</div><div class="mlb-startup-text">Frontend assets are not loaded. Re-run this Builder cell.</div></div>';
      return null;
    }}
    return {payload_loader};
  }}).then(function(mountPayload) {{
    if (mountPayload === null) return;
    window.MLBricksBuilder.mount(root, mountPayload);
  }}).catch(function(error) {{
    root.innerHTML = '<div class="mlb-startup-shell"><div class="mlb-startup-mark">MLBRICKS STUDIO</div><div class="mlb-startup-text">Frontend load failed: '+String(error && error.message || error)+'</div></div>';
    console.error("MLB Studio frontend load failed", error);
  }});
}})();
</script>
"""

    def web(self):
        """Launch MLB Studio inside Jupyter, Colab, Kaggle, or another IPython notebook.

        This is the explicit notebook/web entry point. It uses the same standard
        ipywidgets bridge as normal notebook display so Build, Data, Cloud,
        training, serving, and persistence actions continue to execute in Python.
        """
        try:
            self._ipython_display_()
        except ImportError as exc:
            raise RuntimeError(
                "Builder.web() requires an IPython/Jupyter environment. "
                "Use Builder.app() when running MLB Studio directly on a local system."
            ) from exc
        return None

    def _local_app_bridge_classes(self):
        suffix = self._instance_id.replace("-", "_")
        return {
            "state": f"mlb-local-state-bridge-{suffix}",
            "command": f"mlb-local-command-bridge-{suffix}",
            "run": f"mlb-local-run-bridge-{suffix}",
            "stop": f"mlb-local-stop-bridge-{suffix}",
            "progress": f"mlb-local-progress-bridge-{suffix}",
        }

    def _setup_local_app_bridge(self):
        bridge = self._local_app_bridge_classes()
        self._bridge_widgets = {
            "state": _LocalBridgeValue(json.dumps(self.state)),
            "command": _LocalBridgeValue("{}"),
            "progress": _LocalBridgeValue(json.dumps({
                "status": "idle", "message": "Ready", "overall": 0, "nodes": {}
            })),
        }
        return bridge

    def _local_app_html(self, bridge):
        """Return the full standalone localhost page used by Builder.app()."""
        fragment = self._html(bridge=bridge, include_assets=True, allow_full_window=False)
        bridge_json = json.dumps(bridge)
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MLB Studio</title>
  <link rel="icon" href="/favicon.ico?v=2" sizes="any">
  <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png?v=2">
  <link rel="icon" type="image/png" sizes="64x64" href="/favicon.png?v=2">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg?v=2">
  <link rel="shortcut icon" type="image/x-icon" href="/favicon.ico?v=2">
  <style>
    html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#0b1118}}
    body{{padding:0}}
    body.mlb-local-app .mlb-root{{width:100vw!important;height:100vh!important;min-height:100vh!important;max-height:100vh!important;min-width:0!important;margin-bottom:0!important;border-radius:0!important;border:0!important;box-shadow:none!important}}
    .mlb-local-bridge{{position:fixed!important;left:-10000px!important;top:-10000px!important;width:1px!important;height:1px!important;opacity:0!important;pointer-events:none!important}}
  </style>
</head>
<body class="mlb-local-app">
  <div class="mlb-local-bridge" aria-hidden="true">
    <textarea class="{bridge['state']}"></textarea>
    <textarea class="{bridge['command']}"></textarea>
    <button type="button" class="{bridge['run']}"></button>
    <button type="button" class="{bridge['stop']}"></button>
    <textarea class="{bridge['progress']}"></textarea>
  </div>
  {fragment}
  <script>
  (function() {{
    const bridge = {bridge_json};
    const state = document.querySelector('.'+bridge.state);
    const command = document.querySelector('.'+bridge.command);
    const run = document.querySelector('.'+bridge.run);
    const stop = document.querySelector('.'+bridge.stop);
    const progress = document.querySelector('.'+bridge.progress);
    let progressBusy = false;
    let progressSeq = 0;
    const progressQueue = [];
    let progressDrainScheduled = false;

    function deliverProgressEvent(event) {{
      progress.value = JSON.stringify(event || {{}});
      progress.dispatchEvent(new Event('input', {{bubbles:true}}));
    }}

    function drainProgressQueue() {{
      progressDrainScheduled = false;
      let budget = 3;
      while (budget-- > 0 && progressQueue.length) deliverProgressEvent(progressQueue.shift());
      if (progressQueue.length) {{
        progressDrainScheduled = true;
        requestAnimationFrame(drainProgressQueue);
      }}
    }}

    function enqueueProgressEvents(events) {{
      if (!Array.isArray(events) || !events.length) return;
      progressQueue.push(...events);
      if (!progressDrainScheduled) {{
        progressDrainScheduled = true;
        requestAnimationFrame(drainProgressQueue);
      }}
    }}

    async function postJson(url, payload) {{
      const response = await fetch(url, {{
        method: 'POST',
        headers: {{'Content-Type':'application/json'}},
        body: JSON.stringify(payload || {{}}),
        cache: 'no-store'
      }});
      if (!response.ok) throw new Error(await response.text() || ('HTTP '+response.status));
      return response;
    }}

    run.addEventListener('click', function() {{
      postJson('/api/run', {{
        state_raw: state.value || '{{}}',
        command_raw: command.value || '{{}}'
      }}).then(function() {{ command.value='{{}}'; }}).catch(function(error) {{
        progress.value = JSON.stringify({{status:'error',runtime_kind:'bridge',overall:0,message:'Local app bridge error: '+String(error)}});
      }});
    }});

    stop.addEventListener('click', function() {{
      postJson('/api/stop', {{}}).catch(function(error) {{
        progress.value = JSON.stringify({{status:'error',runtime_kind:'bridge',overall:0,message:'Local app stop error: '+String(error)}});
      }});
    }});

    async function pollProgress() {{
      if (progressBusy) return;
      progressBusy = true;
      try {{
        const response = await fetch('/api/progress-events?after='+progressSeq+'&ts='+Date.now(), {{cache:'no-store'}});
        if (response.ok) {{
          const payload = await response.json();
          const events = Array.isArray(payload.events) ? payload.events : [];
          if (events.length) {{
            progressSeq = Number(payload.last_seq || events[events.length-1].event_seq || progressSeq);
            enqueueProgressEvents(events);
          }}
        }}
      }} catch (_) {{
        try {{
          const response = await fetch('/api/progress?ts='+Date.now(), {{cache:'no-store'}});
          if (response.ok) {{
            const raw = await response.text();
            if (raw) {{
              const event = JSON.parse(raw);
              const seq = Number(event.event_seq || progressSeq);
              if (!seq || seq > progressSeq) {{
                progressSeq = seq || progressSeq;
                enqueueProgressEvents([event]);
              }}
            }}
          }}
        }} catch (_) {{}}
      }}
      finally {{ progressBusy = false; }}
    }}
    pollProgress();
    setInterval(pollProgress, 50);
  }})();
  </script>
</body>
</html>"""

    def app(self, host="127.0.0.1", port=0, *, open_browser=True, block=True):
        """Run MLB Studio as a standalone local application in the system browser.

        Unlike :meth:`web`, this mode does not require Jupyter/IPython and does
        not show the ``Full Window`` action because the Studio itself already
        occupies the local browser window. The server binds to localhost by
        default and uses a small local bridge for Python runtime actions.
        """
        import http.server
        import socketserver
        import webbrowser
        from urllib.parse import urlparse, parse_qs

        if self._app_server is not None:
            if self._app_url and open_browser:
                webbrowser.open(self._app_url, new=1)
            if block:
                try:
                    while self._app_server is not None:
                        time.sleep(0.25)
                except KeyboardInterrupt:
                    self.stop_app()
            return self._app_url

        bridge = self._setup_local_app_bridge()
        builder = self
        favicon_svg_path = _STATIC / "favicon.svg"
        favicon_png_path = _STATIC / "favicon.png"
        favicon_32_path = _STATIC / "favicon-32.png"
        favicon_ico_path = _STATIC / "favicon.ico"
        page_html = self._local_app_html(bridge).encode("utf-8")

        class LocalStudioHandler(http.server.BaseHTTPRequestHandler):
            server_version = "MLBStudioLocal/1.0"

            def log_message(self, format, *args):
                return

            def _send(self, status, body=b"", content_type="text/plain; charset=utf-8"):
                if isinstance(body, str):
                    body = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self):
                try:
                    size = int(self.headers.get("Content-Length") or "0")
                except Exception:
                    size = 0
                if size < 0 or size > 64 * 1024 * 1024:
                    raise ValueError("Request body is too large.")
                raw = self.rfile.read(size) if size else b"{}"
                value = json.loads(raw.decode("utf-8") or "{}")
                if not isinstance(value, dict):
                    raise ValueError("Request JSON must be an object.")
                return value

            def do_GET(self):
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    self._send(200, page_html, "text/html; charset=utf-8")
                    return
                if path == "/favicon.ico":
                    if favicon_ico_path.exists():
                        self._send(200, favicon_ico_path.read_bytes(), "image/x-icon")
                    else:
                        self._send(404, "Not found")
                    return
                if path == "/favicon-32.png":
                    if favicon_32_path.exists():
                        self._send(200, favicon_32_path.read_bytes(), "image/png")
                    else:
                        self._send(404, "Not found")
                    return
                if path == "/favicon.png":
                    if favicon_png_path.exists():
                        self._send(200, favicon_png_path.read_bytes(), "image/png")
                    else:
                        self._send(404, "Not found")
                    return
                if path == "/favicon.svg":
                    if favicon_svg_path.exists():
                        self._send(200, favicon_svg_path.read_bytes(), "image/svg+xml")
                    else:
                        self._send(404, "Not found")
                    return
                if path == "/api/progress":
                    progress = (builder._bridge_widgets or {}).get("progress")
                    self._send(200, getattr(progress, "value", "{}"), "application/json; charset=utf-8")
                    return
                if path == "/api/progress-events":
                    query = parse_qs(urlparse(self.path).query)
                    try:
                        after = int((query.get("after") or [0])[0])
                    except Exception:
                        after = 0
                    batch = builder._progress_events_after(after)
                    self._send(200, json.dumps(batch), "application/json; charset=utf-8")
                    return
                self._send(404, "Not found")

            def do_POST(self):
                path = urlparse(self.path).path
                try:
                    payload = self._read_json()
                    if path == "/api/run":
                        widgets = builder._bridge_widgets or {}
                        state_widget = widgets.get("state")
                        command_widget = widgets.get("command")
                        if state_widget is None or command_widget is None:
                            raise RuntimeError("Local Studio bridge is not initialized.")
                        state_widget.value = str(payload.get("state_raw") or "{}")
                        command_widget.value = str(payload.get("command_raw") or "{}")
                        builder._start_bridge_run()
                        self._send(202, json.dumps({"ok": True}), "application/json; charset=utf-8")
                        return
                    if path == "/api/stop":
                        builder.stop()
                        self._send(200, json.dumps({"ok": True}), "application/json; charset=utf-8")
                        return
                    self._send(404, "Not found")
                except Exception as exc:
                    self._send(400, json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), "application/json; charset=utf-8")

        class LocalThreadingServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        server = LocalThreadingServer((host, int(port)), LocalStudioHandler)
        actual_host, actual_port = server.server_address[:2]
        public_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
        self._app_url = f"http://{public_host}:{actual_port}/"
        self._app_server = server

        if open_browser:
            threading.Timer(0.15, lambda: webbrowser.open(self._app_url, new=1)).start()

        if block:
            print(f"MLB Studio running at {self._app_url}")
            print("Press Ctrl+C to stop the local app.")
            try:
                server.serve_forever(poll_interval=0.25)
            except KeyboardInterrupt:
                pass
            finally:
                self.stop_app()
        else:
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.25},
                daemon=True,
                name=f"mlb-studio-app-{self._instance_id}",
            )
            self._app_thread = thread
            thread.start()
        return self._app_url

    def stop_app(self):
        """Stop a local server previously started by :meth:`app`."""
        server = self._app_server
        self._app_server = None
        self._app_url = None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        self._app_thread = None
        return None


    def _repr_html_(self):
        # Plain-HTML fallback. Editing works; Python execution uses
        # run_data_pipeline() when a standard-widget bridge is unavailable.
        return self._html(bridge=None)

    def _ipython_display_(self):
        """Display the Builder plus a standard-ipywidgets Python execution bridge."""
        from IPython.display import HTML, display

        bridge_widgets = self._setup_widget_bridge()
        bridge_payload = None

        if bridge_widgets:
            # The widgets are intentionally visually hidden. They provide standard
            # Jupyter comms so the custom HTML Run/Stop controls can talk to Python
            # without requiring AnyWidget or a custom JavaScript extension.
            box = None
            try:
                import ipywidgets as widgets
                box = widgets.HBox([
                    bridge_widgets["state"],
                    bridge_widgets["command"],
                    bridge_widgets["request"],
                    bridge_widgets["run"],
                    bridge_widgets["stop"],
                    bridge_widgets["progress"],
                ], layout=widgets.Layout(
                    width="3px",
                    height="3px",
                    min_height="3px",
                    max_height="3px",
                    overflow="hidden",
                    visibility="hidden",
                    margin="0",
                    padding="0",
                ))
                display(box)
                bridge_payload = dict(bridge_widgets["classes"])
            except Exception:
                bridge_payload = None

        global _FRONTEND_ASSETS_EMITTED
        with _FRONTEND_ASSETS_LOCK:
            include_assets = not _FRONTEND_ASSETS_EMITTED
            if include_assets:
                _FRONTEND_ASSETS_EMITTED = True
        display(HTML(self._html(bridge=bridge_payload, include_assets=include_assets)))


BuilderWidget = Builder
