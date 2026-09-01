from __future__ import annotations

"""Lazy MLBricks import registry used by AI Builder.

The Builder UI can expose many MLBricks components without importing the whole
MLBricks package up-front.  A component is resolved only when it is actually
used (or explicitly preloaded by the UI).  Canonical submodule imports are
preferred; compact top-level re-exports are compatibility fallbacks only.
"""

from dataclasses import dataclass
import importlib
import threading
from typing import Any, Iterable


@dataclass(frozen=True)
class ImportCandidate:
    module: str
    symbol: str

    @property
    def path(self) -> str:
        return f"{self.module}.{self.symbol}"


# Canonical module first, compact top-level compatibility export second.
COMPONENT_IMPORTS: dict[str, tuple[ImportCandidate, ...]] = {
    "embedding": (
        ImportCandidate("mlbricks.components", "Embedding"),
        ImportCandidate("mlbricks", "Embedding"),
    ),
    "esa": (
        ImportCandidate("mlbricks.esa", "ESA"),
        ImportCandidate("mlbricks", "ESA"),
    ),
    "soup": (
        ImportCandidate("mlbricks.soup", "SOUP"),
        ImportCandidate("mlbricks", "SOUP"),
    ),
    "bolt": (
        ImportCandidate("mlbricks.bolt", "Bolt"),
        ImportCandidate("mlbricks", "Bolt"),
    ),
    "vesa": (
        ImportCandidate("mlbricks.vesa", "Vesa"),
        ImportCandidate("mlbricks", "Vesa"),
    ),
    "visualbolt": (
        ImportCandidate("mlbricks.visionbolt", "VisionBolt"),
        ImportCandidate("mlbricks", "VisionBolt"),
    ),
    "ffn": (
        ImportCandidate("mlbricks.components", "FFN"),
        ImportCandidate("mlbricks", "FFN"),
    ),
    "saffn": (
        ImportCandidate("mlbricks.ffnbrick", "StateAwareFFN"),
        ImportCandidate("mlbricks", "StateAwareFFN"),
    ),
    "micro_ffn": (
        ImportCandidate("mlbricks.ffnbrick", "MicroVirtualFFN"),
        ImportCandidate("mlbricks", "MicroVirtualFFN"),
    ),
    "virtual_saffn": (
        ImportCandidate("mlbricks.ffnbrick", "VirtualStateAwareFFN"),
        ImportCandidate("mlbricks", "VirtualStateAwareFFN"),
    ),
    "rmsnorm": (
        ImportCandidate("mlbricks.components", "RMSNorm"),
        ImportCandidate("mlbricks", "RMSNorm"),
    ),
    "layernorm": (
        ImportCandidate("mlbricks.components", "LayerNorm"),
        ImportCandidate("mlbricks", "LayerNorm"),
    ),
    "residual": (
        ImportCandidate("mlbricks.components", "Residual"),
        ImportCandidate("mlbricks", "Residual"),
    ),
    "rescontroller": (
        ImportCandidate("mlbricks.residualbrick", "ResController"),
        ImportCandidate("mlbricks", "ResController"),
    ),
    "lm_head": (
        ImportCandidate("mlbricks.components", "LMHead"),
        ImportCandidate("mlbricks", "LMHead"),
    ),
    "linear": (
        ImportCandidate("mlbricks.components", "Linear"),
        ImportCandidate("mlbricks", "Linear"),
    ),
    "elasticbit_runtime": (
        ImportCandidate("mlbricks.elasticbit", "ElasticBit"),
        ImportCandidate("mlbricks", "ElasticBit"),
    ),
    "rope": (
        ImportCandidate("mlbricks.position", "RoPE"),
        ImportCandidate("mlbricks", "RoPE"),
    ),
    "learned_position": (
        ImportCandidate("mlbricks.position", "LearnedPosition"),
        ImportCandidate("mlbricks", "LearnedPosition"),
    ),
    "sinusoidal_position": (
        ImportCandidate("mlbricks.position", "SinusoidalPosition"),
        ImportCandidate("mlbricks", "SinusoidalPosition"),
    ),
}


# Non-component MLBricks APIs used by Builder runtime/lifecycle code.
API_IMPORTS: dict[str, tuple[ImportCandidate, ...]] = {
    "config.vesa": (
        ImportCandidate("mlbricks.vesa", "VesaConfig"),
        ImportCandidate("mlbricks", "VesaConfig"),
    ),
    "config.visualbolt": (
        ImportCandidate("mlbricks.visionbolt", "VisionBoltConfig"),
        ImportCandidate("mlbricks", "VisionBoltConfig"),
    ),
    "optim.Adam": (
        ImportCandidate("mlbricks.optim", "Adam"),
        ImportCandidate("mlbricks", "Adam"),
    ),
    "optim.AdamW": (
        ImportCandidate("mlbricks.optim", "AdamW"),
        ImportCandidate("mlbricks", "AdamW"),
    ),
    "lifecycle.save": (
        ImportCandidate("mlbricks.lifecycle", "save"),
        ImportCandidate("mlbricks", "save"),
    ),
    "lifecycle.load": (
        ImportCandidate("mlbricks.lifecycle", "load"),
        ImportCandidate("mlbricks", "load"),
    ),
    "lifecycle.inspect": (
        ImportCandidate("mlbricks.lifecycle", "inspect"),
        ImportCandidate("mlbricks", "inspect"),
    ),
    "lifecycle.predict": (
        ImportCandidate("mlbricks.lifecycle", "predict"),
        ImportCandidate("mlbricks", "predict"),
    ),
    "lifecycle.generate": (
        ImportCandidate("mlbricks.lifecycle", "generate"),
        ImportCandidate("mlbricks", "generate"),
    ),
    "lifecycle.compile": (
        ImportCandidate("mlbricks.lifecycle", "compile"),
        ImportCandidate("mlbricks", "compile"),
    ),
    "lifecycle.quantize": (
        ImportCandidate("mlbricks.lifecycle", "quantize"),
        ImportCandidate("mlbricks", "quantize"),
    ),
}


CONFIG_KEYS = {
    "vesa": "config.vesa",
    "visualbolt": "config.visualbolt",
}


class ComponentImportError(ImportError):
    """Raised when no registered import route can resolve a Builder component."""


def _resolve_dotted_object(path: str) -> Any:
    """Resolve an arbitrary user-supplied dotted Python object path.

    The longest importable module prefix is imported first, then remaining
    attributes are traversed. This supports paths such as ``torch.nn.Linear``,
    ``mamba_ssm.Mamba`` and ``torch.nn.functional.gelu`` without requiring a
    hard-coded Builder registry entry.
    """
    path = str(path or "").strip().replace(":", ".")
    if not path or "." not in path:
        raise ComponentImportError(
            "Custom API import path must be a dotted path such as 'torch.nn.Linear'."
        )
    parts = [part for part in path.split(".") if part]
    failures: list[str] = []
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
            return obj
        except Exception as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    raise ComponentImportError(
        f"Could not resolve custom API {path!r}. Tried: {'; '.join(failures)}"
    )


class MLBricksImportPool:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._resolved: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _component_key(component_type: str) -> str:
        return f"component.{str(component_type).strip()}"

    def is_known_component(self, component_type: str) -> bool:
        return str(component_type) in COMPONENT_IMPORTS

    def canonical_candidate(self, component_type: str) -> ImportCandidate | None:
        candidates = COMPONENT_IMPORTS.get(str(component_type)) or ()
        return candidates[0] if candidates else None

    def canonical_config_candidate(self, component_type: str) -> ImportCandidate | None:
        key = CONFIG_KEYS.get(str(component_type))
        candidates = API_IMPORTS.get(key or "") or ()
        return candidates[0] if candidates else None

    def _resolve_candidates(
        self,
        key: str,
        candidates: Iterable[ImportCandidate],
        *,
        required: bool = True,
    ) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]

            failures: list[str] = []
            for candidate in tuple(candidates):
                try:
                    module = importlib.import_module(candidate.module)
                    obj = getattr(module, candidate.symbol)
                    self._cache[key] = obj
                    self._resolved[key] = candidate.path
                    self._errors.pop(key, None)
                    return obj
                except Exception as exc:
                    failures.append(
                        f"{candidate.path}: {type(exc).__name__}: {exc}"
                    )

            message = "; ".join(failures) if failures else "no import routes registered"
            self._errors[key] = message
            if required:
                raise ComponentImportError(
                    f"Could not import MLBricks API {key!r}. Tried: {message}"
                )
            return None

    def resolve_component(self, component_type: str, *, required: bool = True) -> Any:
        component_type = str(component_type)
        candidates = COMPONENT_IMPORTS.get(component_type)
        if not candidates:
            if required:
                raise ComponentImportError(
                    f"No MLBricks import route is registered for component {component_type!r}."
                )
            return None
        return self._resolve_candidates(
            self._component_key(component_type), candidates, required=required
        )

    def resolve_api(self, key: str, *, required: bool = True) -> Any:
        key = str(key)
        candidates = API_IMPORTS.get(key)
        if not candidates:
            if required:
                raise ComponentImportError(f"No MLBricks import route is registered for API {key!r}.")
            return None
        return self._resolve_candidates(f"api.{key}", candidates, required=required)

    def resolve_config(self, component_type: str, *, required: bool = True) -> Any:
        key = CONFIG_KEYS.get(str(component_type))
        if not key:
            if required:
                raise ComponentImportError(
                    f"Component {component_type!r} has no registered config API."
                )
            return None
        return self.resolve_api(key, required=required)

    def ensure_component(self, component_type: str) -> dict[str, Any]:
        """Import one component if it is not already cached and return status."""
        component_type = str(component_type)
        key = self._component_key(component_type)
        was_cached = key in self._cache
        try:
            self.resolve_component(component_type)
            return {
                "component_type": component_type,
                "ok": True,
                "cached": was_cached,
                "imported_now": not was_cached,
                "resolved_from": self._resolved.get(key),
                "error": None,
            }
        except Exception as exc:
            return {
                "component_type": component_type,
                "ok": False,
                "cached": False,
                "imported_now": False,
                "resolved_from": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def resolve_external(self, import_path: str, *, required: bool = True) -> Any:
        """Resolve and cache an arbitrary API object used by a custom component."""
        import_path = str(import_path or "").strip().replace(":", ".")
        key = f"external.{import_path}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            try:
                obj = _resolve_dotted_object(import_path)
                self._cache[key] = obj
                self._resolved[key] = import_path
                self._errors.pop(key, None)
                return obj
            except Exception as exc:
                self._errors[key] = f"{type(exc).__name__}: {exc}"
                if required:
                    raise
                return None

    def ensure_external(self, import_path: str, *, label: str | None = None) -> dict[str, Any]:
        import_path = str(import_path or "").strip().replace(":", ".")
        key = f"external.{import_path}"
        was_cached = key in self._cache
        try:
            self.resolve_external(import_path)
            return {
                "component_type": label or import_path,
                "import_path": import_path,
                "ok": True,
                "cached": was_cached,
                "imported_now": not was_cached,
                "resolved_from": self._resolved.get(key),
                "error": None,
            }
        except Exception as exc:
            return {
                "component_type": label or import_path,
                "import_path": import_path,
                "ok": False,
                "cached": False,
                "imported_now": False,
                "resolved_from": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def ensure_graph(
        self,
        nodes: Iterable[dict[str, Any]],
        custom_components: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Preload imports needed by a graph, including user API components."""
        custom_components = custom_components or {}
        needed: set[str] = set()
        external: dict[str, tuple[str, str]] = {}
        visited_defs: set[str] = set()

        def visit(items: Iterable[dict[str, Any]]) -> None:
            for node in items or ():
                component_type = str(node.get("type") or "")
                if component_type in COMPONENT_IMPORTS:
                    needed.add(component_type)
                if component_type == "custom":
                    definition_id = str(node.get("definition_id") or "")
                    if definition_id and definition_id not in visited_defs:
                        visited_defs.add(definition_id)
                        definition = custom_components.get(definition_id) or {}
                        if str(definition.get("implementation") or "graph") == "api":
                            definition_nodes = definition.get("nodes") or []
                            steps = [n for n in definition_nodes if str(n.get("type") or "") == "api_step"]
                            if steps:
                                for step in steps:
                                    binding = step.get("api_binding") or {}
                                    path = str(binding.get("import_path") or "").strip()
                                    if not path:
                                        module = str(binding.get("module_path") or "").strip().strip(".")
                                        symbol = str(binding.get("symbol") or "").strip().strip(".")
                                        path = ".".join(part for part in (module, symbol) if part)
                                    if path:
                                        key = f"{definition_id}:{step.get('id') or len(external)}"
                                        label = f"{definition.get('name') or 'API Component'} / {step.get('name') or path}"
                                        external[key] = (path, label)
                                # API Components may also contain supported built-in
                                # MLBricks nodes. Visit those nodes so their canonical
                                # imports are preflighted by the same import pool.
                                visit([n for n in definition_nodes if str(n.get("type") or "") != "api_step"])
                            else:
                                binding = definition.get("api_binding") or {}
                                path = str(binding.get("import_path") or "").strip()
                                if path:
                                    external[definition_id] = (path, str(definition.get("name") or path))
                        else:
                            visit(definition.get("nodes") or [])
                if component_type == "stateaware_esa_stack":
                    needed.update({"esa", "rmsnorm", "saffn", "rescontroller"})

        visit(nodes)
        result = {name: self.ensure_component(name) for name in sorted(needed)}
        for definition_id, (path, label) in external.items():
            result[f"custom:{definition_id}"] = self.ensure_external(path, label=label)
        return result

    def import_info(self, component_type: str) -> dict[str, Any]:
        component_type = str(component_type)
        candidate = self.canonical_candidate(component_type)
        config = self.canonical_config_candidate(component_type)
        key = self._component_key(component_type)
        return {
            "component_type": component_type,
            "known": candidate is not None,
            "canonical_module": candidate.module if candidate else None,
            "canonical_symbol": candidate.symbol if candidate else None,
            "canonical_path": candidate.path if candidate else None,
            "config_module": config.module if config else None,
            "config_symbol": config.symbol if config else None,
            "config_path": config.path if config else None,
            "loaded": key in self._cache,
            "resolved_from": self._resolved.get(key),
            "error": self._errors.get(key),
        }

    def status(self) -> dict[str, Any]:
        return {
            "loaded": dict(self._resolved),
            "errors": dict(self._errors),
            "known_components": sorted(COMPONENT_IMPORTS),
            "known_apis": sorted(API_IMPORTS),
        }

    def clear(self) -> None:
        """Clear Builder's cache. Imported Python modules remain in sys.modules."""
        with self._lock:
            self._cache.clear()
            self._resolved.clear()
            self._errors.clear()


IMPORT_POOL = MLBricksImportPool()


def resolve_component(component_type: str) -> Any:
    return IMPORT_POOL.resolve_component(component_type)


def resolve_api(key: str) -> Any:
    return IMPORT_POOL.resolve_api(key)
