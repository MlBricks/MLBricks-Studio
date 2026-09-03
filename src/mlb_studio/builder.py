from __future__ import annotations
import html
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
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Mapping

from .graph import (
    new_project, primitive_catalog, tinystories_30m_project,
    stateaware_esa_200m_project, soup_200m_project, soup_30m_1l_project,
)
from .runtime import get_mlbricks_info
from .security import project_executable_features, safe_extract_zip, safe_torch_load
from .version import __version__, FORMAT_VERSION
from .api_registry import discover_mlbricks_api, refresh_component_api
from .import_pool import IMPORT_POOL
from .runner import execute_data_pipeline, validate_data_pipeline, PipelineValidationError, PipelineStopped

_STATIC = Path(__file__).parent / "static"

class Builder:
    """
    Kaggle/Jupyter-safe Builder.

    This class intentionally does not inherit from AnyWidget or ipywidgets.
    Notebook rendering uses the standard `_repr_html_` protocol.
    """

    def __init__(self, project=None, preset=None):
        if project is not None:
            self.state = project
        elif preset in {"tinystories", "tinystories-30m", "demo"}:
            self.state = tinystories_30m_project()
        elif preset in {"esa-200m", "stateaware-esa-200m", "stateaware_esa_200m"}:
            self.state = stateaware_esa_200m_project()
        elif preset in {"soup-200m", "soup_200m"}:
            self.state = soup_200m_project()
        elif preset in {"soup-30m", "soup-30m-1l", "soup_30m_1l"}:
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
                item["real_api"] = real
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
        self._stop_event = threading.Event()
        self._bridge_widgets = None
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
        self.runtime_capabilities = self._detect_runtime_capabilities()
        from .local_runtime import detect_local_environment, ensure_mlbricks_workspace
        self.local_environment = detect_local_environment()
        self.local_environment["paths"] = ensure_mlbricks_workspace(self.local_environment)
        actual_root = Path(self.local_environment["paths"]["root"]).parent
        expected_root = Path(self.local_environment.get("workspace_root") or self.local_environment.get("default_root") or actual_root)
        if actual_root != expected_root:
            self.local_environment["workspace_root"] = str(actual_root)
            self.local_environment["default_root"] = str(actual_root)
            roots = [str(actual_root), *(self.local_environment.get("roots") or [])]
            self.local_environment["roots"] = list(dict.fromkeys(roots))
        self._apply_local_workspace_defaults()

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

        # Do not run nvidia-smi during Builder construction. Even a bounded
        # subprocess can stall hosted notebooks while drivers wake up. Use only
        # cheap environment/device-file hints for the initial selector; the real
        # runtime performs authoritative CUDA discovery when execution starts.
        cuda_version = None
        cuda_hint = False
        visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
        nvidia_visible = str(os.environ.get("NVIDIA_VISIBLE_DEVICES", "")).strip()
        if visible and visible not in {"-1", "none", "None"}:
            cuda_hint = True
        elif nvidia_visible and nvidia_visible.lower() not in {"none", "void"}:
            cuda_hint = True
        elif Path("/dev/nvidia0").exists():
            cuda_hint = True
        if cuda_hint:
            devices.append({
                "id": "cuda:0",
                "label": "GPU — CUDA device",
                "kind": "cuda",
                "index": 0,
                "name": "CUDA device",
                "available": True,
                "provisional": True,
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
            "audio_processing": None, "batch": None, "output": None,
        }
        source_types = {"manual_dataset", "hf_dataset", "kaggle_dataset", "url_dataset", "local_dataset"}
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
            elif t=="batch_data": snapshot["batch"] = value
            elif t=="prepared_dataset": snapshot["output"] = value
        return snapshot

    def _register_prepared_dataset(self, result):
        node = self._prepared_output_node() or {}
        params = node.get("params") or {}
        requested_name = str(params.get("dataset_name") or "Prepared Dataset").strip() or "Prepared Dataset"

        existing_meta = None
        for item in self.state.setdefault("prepared_datasets", []):
            if str(item.get("name", "")).strip().lower() == requested_name.lower():
                existing_meta = item
                break

        dataset_id = (
            existing_meta.get("id")
            if existing_meta
            else f"dataset_{uuid.uuid4().hex[:12]}"
        )

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

        registry = self.state.setdefault("prepared_datasets", [])
        if existing_meta:
            index = registry.index(existing_meta)
            registry[index] = metadata
        else:
            registry.append(metadata)

        self.prepared_datasets[dataset_id] = result
        self.state.setdefault("project", {})["dataset"] = requested_name
        return metadata

    def available_datasets(self):
        """Return serializable metadata for every prepared dataset in this project."""
        return json.loads(json.dumps(self.state.get("prepared_datasets") or []))

    def get_prepared_dataset(self, dataset_id_or_name, split=None):
        """Return a prepared Dataset/DatasetDict by registry id or display name."""
        wanted = str(dataset_id_or_name)
        metadata = None
        for item in self.state.get("prepared_datasets") or []:
            if item.get("id") == wanted or str(item.get("name", "")).lower() == wanted.lower():
                metadata = item
                break
        if metadata is None:
            raise KeyError(f"Prepared dataset not found: {dataset_id_or_name!r}")

        dataset_id = metadata["id"]
        result = self.prepared_datasets.get(dataset_id)

        if result is None and metadata.get("path"):
            try:
                from datasets import load_from_disk
                result = load_from_disk(metadata["path"])
                self.prepared_datasets[dataset_id] = result
            except Exception as exc:
                raise RuntimeError(
                    f'{metadata["name"]!r} is not in memory and could not be loaded '
                    f'from {metadata.get("path")!r}: {exc}'
                ) from exc

        if result is None:
            raise RuntimeError(
                f'{metadata["name"]!r} is listed in the design but its actual data is '
                "not in this Python session. Re-run its Data Processing pipeline, or "
                "enable Save To Disk before saving the design."
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

    def run_data_pipeline(self, progress_callback=None):
        """Execute Data Processing, register the result, and publish split metadata."""
        self._stop_event.clear()
        self.last_run_error = None
        last_progress = {}

        def relay(payload):
            enriched = dict(payload or {})
            enriched.setdefault("runtime_kind", "data")
            last_progress.clear()
            last_progress.update(enriched)
            if progress_callback:
                progress_callback(enriched)

        try:
            self.last_data_result = execute_data_pipeline(
                self.state,
                progress_callback=relay,
                stop_event=self._stop_event,
            )
            metadata = self._register_prepared_dataset(self.last_data_result)

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

            return self.last_data_result
        except Exception as exc:
            self.last_run_error = exc
            raise

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

    def train_model(self, model_id, *, progress_callback=None):
        """Compile and really train a supported Builder language model."""
        from .model_runtime import train_builder_model
        entry = self._model_output(model_id)
        dataset_id = entry.get("selected_dataset_id")
        if not dataset_id:
            raise RuntimeError("Select a prepared training dataset first.")
        meta = self._dataset_meta(dataset_id)
        dataset = self.get_prepared_dataset(dataset_id)
        config = self._runtime_with_project_trust(entry.get("training_config") or {})
        self._stop_event.clear()
        def emit(payload):
            if progress_callback:
                enriched = dict(payload or {})
                enriched.setdefault("model_id", model_id)
                progress_callback(enriched)

        result = train_builder_model(
            state=self.state, model_entry=entry, dataset=dataset, dataset_meta=meta,
            config=config, progress=emit, stop_event=self._stop_event,
        )
        update = result["model_update"]
        entry.update(update)
        self.trained_models[model_id] = {
            "compiled": result["compiled"], "tokenizer": result["tokenizer"],
            "runtime": dict(config),
        }
        payload = {
            "status":"done","runtime_kind":"train","phase":"done","overall":100,
            "message":f'Training complete: {entry.get("name", "model")}',
            "model_id":model_id,"model_update":update,
            "sample_text":result.get("last_sample"),
        }
        if progress_callback: progress_callback(payload)
        return update

    def generate_model(self, model_id, *, progress_callback=None):
        """Generate tokens from a trained Builder language model."""
        from .model_runtime import load_trained_for_generation, generate_text
        entry = self._model_output(model_id)
        if not entry.get("weights_ready"):
            raise RuntimeError("This model has no trained/loaded weights yet.")
        dataset_id = entry.get("selected_dataset_id")
        meta = self._dataset_meta(dataset_id) if dataset_id else copy.deepcopy(entry.get("hub_dataset_meta") or {})
        config = self._runtime_with_project_trust(entry.get("generation_config") or {}, checkpoint_path=entry.get("checkpoint_path") or entry.get("path"))
        self._stop_event.clear()

        def emit(payload):
            if progress_callback:
                enriched = dict(payload or {})
                enriched.setdefault("model_id", model_id)
                progress_callback(enriched)
        cached = self.trained_models.get(model_id)
        if cached is not None:
            compiled, tokenizer = cached["compiled"], cached["tokenizer"]
            previous_runtime = cached.get("runtime") or {}
            runtime_keys = ("device", "backend", "execution_mode", "compile_mode", "precision")
            runtime_changed = any(str(previous_runtime.get(k, "auto")) != str(config.get(k, "auto")) for k in runtime_keys)
            if runtime_changed:
                compiled, tokenizer = load_trained_for_generation(
                    state=self.state, model_entry=entry, dataset_meta=meta, config=config,
                    checkpoint_path=entry.get("checkpoint_path"), progress=emit,
                )
                self.trained_models[model_id] = {"compiled":compiled,"tokenizer":tokenizer,"runtime":dict(config)}
        else:
            compiled, tokenizer = load_trained_for_generation(
                state=self.state, model_entry=entry, dataset_meta=meta, config=config,
                checkpoint_path=entry.get("checkpoint_path"), progress=emit,
            )
            self.trained_models[model_id] = {"compiled":compiled,"tokenizer":tokenizer,"runtime":dict(config)}
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
        )
        entry["last_generation"] = text
        entry["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload={
            "status":"done","runtime_kind":"generate","phase":"done","overall":100,
            "message":f"Generated {count} tokens.","model_id":model_id,
            "generated_tokens":count,"generated_text":text,
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
        cached=self.trained_models.get(model_id)
        if cached:
            previous=cached.get("runtime") or {}
            keys=("device","backend","execution_mode","compile_mode","precision")
            if all(str(previous.get(k,"auto"))==str(runtime.get(k,"auto")) for k in keys):
                compiled,tokenizer=cached.get("compiled"),cached.get("tokenizer")
        if compiled is None or tokenizer is None:
            compiled,tokenizer=load_trained_for_generation(
                state=self.state,model_entry=entry,dataset_meta=meta,config=runtime,
                checkpoint_path=entry.get("checkpoint_path") or entry.get("path"),progress=None)
            self.trained_models[model_id]={"compiled":compiled,"tokenizer":tokenizer,"runtime":dict(runtime)}

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
                "builder_version": "1.0.0",
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
        from .local_runtime import scan_local_files
        return scan_local_files(roots=roots)

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
                "image_processing": None, "audio_processing": None, "batch": None,
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

    def _cloud_provider_status(self, provider, cloud):
        provider = str(provider or "").lower()
        credentials = cloud.get("credentials") or {}
        if provider == "huggingface":
            return self.hub_status(token=credentials.get("token"))
        from . import cloud as cloud_backend
        if provider == "github":
            return cloud_backend.github_status(token=credentials.get("token") or "")
        if provider == "aws":
            return cloud_backend.s3_status(credentials)
        if provider == "gcp":
            return cloud_backend.gcs_status(credentials)
        if provider == "azure":
            return cloud_backend.azure_status(credentials)
        raise ValueError(f"Unknown cloud provider: {provider!r}")

    def _push_generic_cloud(self, provider, cloud):
        from . import cloud as cloud_backend
        provider = str(provider).lower()
        credentials = cloud.get("credentials") or {}
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
            self._create_cloud_bundle(content_type, artifact_id, archive)

            if provider == "github":
                return cloud_backend.github_upload(
                    archive,
                    repo=cloud.get("repo"),
                    path_in_repo=cloud.get("object_path") or f"mlbricks/{archive.name}",
                    branch=cloud.get("branch") or "main",
                    token=credentials.get("token") or "",
                    commit_message=f"Push {content_type} from MLB Studio",
                )
            if provider == "aws":
                return cloud_backend.s3_upload(
                    archive,
                    bucket=cloud.get("bucket"),
                    object_key=cloud.get("object_path") or archive.name,
                    credentials=credentials,
                )
            if provider == "gcp":
                return cloud_backend.gcs_upload(
                    archive,
                    bucket=cloud.get("bucket"),
                    object_name=cloud.get("object_path") or archive.name,
                    credentials=credentials,
                )
            if provider == "azure":
                return cloud_backend.azure_upload(
                    archive,
                    container=cloud.get("container"),
                    blob_name=cloud.get("object_path") or archive.name,
                    credentials=credentials,
                )
        raise ValueError(f"Unsupported generic cloud provider: {provider!r}")

    def _load_generic_cloud(self, provider, cloud):
        from . import cloud as cloud_backend
        provider = str(provider).lower()
        credentials = cloud.get("credentials") or {}
        with tempfile.TemporaryDirectory(prefix="mlbricks_cloud_load_") as td:
            archive = Path(td) / "download.mlbricks.zip"
            if provider == "github":
                result = cloud_backend.github_download(
                    archive,
                    repo=cloud.get("repo"),
                    path_in_repo=cloud.get("object_path"),
                    branch=cloud.get("branch") or "main",
                    token=credentials.get("token") or None,
                )
            elif provider == "aws":
                result = cloud_backend.s3_download(
                    archive,
                    bucket=cloud.get("bucket"),
                    object_key=cloud.get("object_path"),
                    credentials=credentials,
                )
            elif provider == "gcp":
                result = cloud_backend.gcs_download(
                    archive,
                    bucket=cloud.get("bucket"),
                    object_name=cloud.get("object_path"),
                    credentials=credentials,
                )
            elif provider == "azure":
                result = cloud_backend.azure_download(
                    archive,
                    container=cloud.get("container"),
                    blob_name=cloud.get("object_path"),
                    credentials=credentials,
                )
            else:
                raise ValueError(f"Unsupported generic cloud provider: {provider!r}")
            restored = self._restore_cloud_bundle(archive)
            result["restored"] = restored
            return result

    def _execute_cloud_command(self, command, progress_callback=None):
        cloud = command.get("cloud") or {}
        action = str(command.get("action") or "")
        provider = str(cloud.get("provider") or "huggingface").lower()

        def emit(payload):
            if progress_callback:
                progress_callback(payload)

        if action == "cloud_status":
            status = self._cloud_provider_status(provider, cloud)
            emit({
                "status": "done", "runtime_kind": "cloud", "phase": "status",
                "overall": 100, "message": status.get("message", "Connection checked."),
                "cloud_status": {"provider": provider, **status},
            })
            return status

        emit({
            "status": "running", "runtime_kind": "cloud", "phase": action,
            "overall": 5, "message": f"Connecting to {provider}…",
        })

        if provider == "huggingface":
            credentials = cloud.get("credentials") or {}
            token = credentials.get("token")
            repo_id = str(cloud.get("repo") or "").strip()
            revision = str(cloud.get("revision") or "main").strip() or "main"
            private = bool(cloud.get("private", True))
            content_type = cloud.get("content_type") or "project"
            artifact_id = cloud.get("artifact_id")
            if action == "cloud_push":
                if content_type == "dataset":
                    result = self.push_dataset_to_hub(artifact_id, repo_id, private=private, token=token)
                elif content_type == "model":
                    result = self.push_model_to_hub(artifact_id, repo_id, private=private, token=token)
                else:
                    result = self.push_project_to_hub(repo_id, private=private, token=token)
                message = f'{content_type.title()} pushed to {repo_id}.'
            elif action == "cloud_load":
                if content_type == "dataset":
                    restored = self.load_dataset_from_hub(repo_id, revision=revision, token=token)
                    result = {"restored": {"content_type": "dataset", "dataset": restored}}
                elif content_type == "model":
                    restored = self.load_model_from_hub(repo_id, revision=revision, token=token)
                    result = {"restored": {"content_type": "model", "model": restored}}
                else:
                    result = self.load_project_from_hub(repo_id, revision=revision, token=token)
                message = f'{content_type.title()} loaded from {repo_id}.'
            else:
                raise ValueError(f"Unknown cloud action: {action!r}")
        else:
            if action == "cloud_push":
                result = self._push_generic_cloud(provider, cloud)
                message = f'{(cloud.get("content_type") or "content").title()} pushed to {provider}.'
            elif action == "cloud_load":
                result = self._load_generic_cloud(provider, cloud)
                message = f'Content loaded from {provider}.'
            else:
                raise ValueError(f"Unknown cloud action: {action!r}")

        emit({
            "status": "done", "runtime_kind": "cloud", "phase": action,
            "overall": 100, "message": message,
            "cloud_result": result,
            "state_replace": self.to_dict(),
        })
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
        if progress is None:
            return
        enriched = dict(payload)
        enriched["ts"] = time.time()
        try:
            progress.value = json.dumps(enriched)
        except Exception:
            pass

    def _start_bridge_run(self):
        if self._run_thread is not None and self._run_thread.is_alive():
            self._publish_bridge_progress({
                "status": "running",
                "message": "A pipeline run is already active.",
                "overall": 0,
                "nodes": {},
            })
            return

        widgets = self._bridge_widgets or {}
        state_widget = widgets.get("state")
        command_widget = widgets.get("command")
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

        if state_widget is not None:
            try:
                incoming = json.loads(state_widget.value)
                if isinstance(incoming, dict) and incoming.get("components"):
                    legacy_command = incoming.pop("_runtime_command", None) or {}
                    incoming.pop("_session_secrets", None)
                    if not command:
                        command = legacy_command
                    self.state = incoming
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

        action = str(command.get("action") or "data").lower()
        model_id = command.get("model_id")

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
                elif action == "train":
                    self.train_model(model_id, progress_callback=self._publish_bridge_progress)
                elif action == "generate":
                    self.generate_model(model_id, progress_callback=self._publish_bridge_progress)
                elif action.startswith("hub_"):
                    self._execute_hub_command(
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
                    )
            except PipelineStopped:
                self._publish_bridge_progress({
                    "status":"stopped","runtime_kind":action,"overall":0,
                    "message":f"{action.title()} stopped."
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
                self.last_run_error = exc
                runtime_kind = "serve" if str(action).startswith("serve_") else action
                error_payload = {
                    "status":"error","runtime_kind":runtime_kind,"overall":0,
                    "message":f"{type(exc).__name__}: {exc}"
                }
                if runtime_kind == "serve":
                    error_payload.update({
                        "phase": action,
                        "model_id": command.get("model_id"),
                        "model_update": {"serve_status":"error"},
                    })
                self._publish_bridge_progress(error_payload)

        self._run_thread = threading.Thread(
            target=worker,
            name=f"mlb-studio-run-{self._instance_id}",
            daemon=True,
        )
        self._run_thread.start()

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
        run_widget = widgets.Button(description="", layout=hidden)
        stop_widget = widgets.Button(description="", layout=hidden)
        progress_widget = widgets.Textarea(
            value=json.dumps({"status": "idle", "message": "Ready", "overall": 0, "nodes": {}}),
            layout=hidden,
        )

        classes = {
            "state": f"mlb-state-bridge-{suffix}",
            "command": f"mlb-command-bridge-{suffix}",
            "run": f"mlb-run-bridge-{suffix}",
            "stop": f"mlb-stop-bridge-{suffix}",
            "progress": f"mlb-progress-bridge-{suffix}",
        }
        state_widget.add_class(classes["state"])
        command_widget.add_class(classes["command"])
        run_widget.add_class(classes["run"])
        stop_widget.add_class(classes["stop"])
        progress_widget.add_class(classes["progress"])

        run_widget.on_click(lambda _: self._start_bridge_run())
        stop_widget.on_click(lambda _: self.stop())

        self._bridge_widgets = {
            "state": state_widget,
            "command": command_widget,
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

    def _html(self, bridge=None):
        css = (_STATIC / "builder.css").read_text(encoding="utf-8")
        js = (_STATIC / "builder.js").read_text(encoding="utf-8")
        trust = self.project_trust_info()
        payload = json.dumps({
            "state": self.state,
            "catalog": self.catalog,
            "mlbricks_api": self.mlbricks_api,
            "bridge": bridge,
            "runtime_capabilities": self.runtime_capabilities,
            "local_environment": self.local_environment,
            "instance_id": self._instance_id,
            "project_trust": trust,
        }).replace("</", "<\\/")
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
        style_id = f"{self._instance_id}_assets"
        return f"""
<style id="{html.escape(style_id)}">{css}</style>
{warning}
<div id="{html.escape(self._instance_id)}" class="mlb-root" data-mlb-studio-version="{html.escape(__version__)}"></div>
<script>
window.__MLB_STUDIO_CSS_ELEMENT__ = document.getElementById({json.dumps(style_id)});
try {{ delete window.MLBricksBuilder; }} catch (e) {{ window.MLBricksBuilder = undefined; }}
{js}
window.MLBricksBuilder.mount(
  document.getElementById({json.dumps(self._instance_id)}),
  {payload}
);
</script>
"""

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

        display(HTML(self._html(bridge=bridge_payload)))


BuilderWidget = Builder
