from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .import_pool import (
    COMPONENT_IMPORTS,
    CONFIG_KEYS,
    IMPORT_POOL,
)

_SCHEMA_PATH = Path(__file__).with_name("mlbricks_api_schema.json")


# These components intentionally keep the richer source-derived Builder schema
# instead of replacing it with a plain constructor signature at runtime.
# SOUP accepts scalar-or-per-layer sequences/config mappings, while the primary
# ElasticBit 4-32 UI represents ElasticBit.RuntimeMatrix.from_auto.
SOURCE_DEFINED_FIELDS = {"soup", "elasticbit_runtime", "lm_head"}

CHOICES = {
    "backend": ["auto", "native", "pytorch"],
    "precision": ["fp32", "fp16", "bf16"],
    "activation": ["gelu", "gelu_tanh", "relu", "silu", "swish", "tanh"],
    "device": ["auto", "cpu", "cuda", "None"],
    "position": ["none", "auto", "learned", "sinusoidal", "rope"],
    "engine": ["Serpentine", "ViT", "CNN", "Diffusion", "AR"],
    "scan": ["cross", "raster", "serpentine"],
    "ffn": ["standard", "ffnbrick", "virtual_ffnbrick", "micro_ffnbrick"],
    "residual": ["standard", "rescontroller"],
    "norm": ["rmsnorm", "layernorm"],
}


def _fallback_schema():
    payload = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return deepcopy(payload["components"])


def _safe_default(value: Any):
    if value is inspect._empty:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _field_type(name, annotation, default):
    if name in CHOICES:
        return "select"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (int, float)):
        return "number"
    text = "" if annotation is inspect._empty else str(annotation).lower()
    if "bool" in text:
        return "bool"
    if "int" in text or "float" in text:
        return "number"
    return "text"


def _fields(obj):
    sig = inspect.signature(obj)
    out = []
    for name, p in sig.parameters.items():
        if name in {"self", "args", "kwargs"}:
            continue
        default = _safe_default(p.default)
        item = {
            "key": name,
            "label": name.replace("_", " ").title(),
            "type": _field_type(name, p.annotation, default),
            "required": p.default is inspect._empty,
            "value": default,
            "annotation": "" if p.annotation is inspect._empty else str(p.annotation),
        }
        if name in CHOICES:
            item["options"] = CHOICES[name]
        out.append(item)
    return sig, out


def _canonical_metadata(component_type: str, fallback: dict[str, Any]) -> dict[str, Any]:
    info = IMPORT_POOL.import_info(component_type)
    canonical_path = info.get("canonical_path")
    public_name = info.get("canonical_symbol") or fallback.get("public_name")
    payload = {
        **fallback,
        "available": True,
        "runtime_available": None,
        "source": "MLBricks source schema + lazy import pool",
        "public_name": public_name,
        "import_path": canonical_path or fallback.get("import_path"),
        "import_module": info.get("canonical_module"),
        "import_symbol": info.get("canonical_symbol"),
        "import_candidates": [candidate.path for candidate in COMPONENT_IMPORTS.get(component_type, ())],
        "loaded": bool(info.get("loaded")),
        "resolved_from": info.get("resolved_from"),
        "runtime_error": info.get("error"),
    }
    if info.get("config_path"):
        config = deepcopy(payload.get("config_api") or {})
        config.update({
            "public_name": info.get("config_symbol"),
            "import_path": info.get("config_path"),
            "import_module": info.get("config_module"),
            "import_symbol": info.get("config_symbol"),
        })
        payload["config_api"] = config
    return payload


def _inspect_one(component_type: str, fallback: dict[str, Any]) -> dict[str, Any]:
    result = _canonical_metadata(component_type, fallback)
    try:
        obj = IMPORT_POOL.resolve_component(component_type)
        sig, inspected_fields = _fields(obj)
        source_defined = component_type in SOURCE_DEFINED_FIELDS
        fields = deepcopy(fallback.get("parameters", [])) if source_defined else inspected_fields

        config_info = result.get("config_api")
        if component_type in CONFIG_KEYS:
            cfg_obj = IMPORT_POOL.resolve_config(component_type)
            cfg_sig, cfg_fields = _fields(cfg_obj)
            if len(cfg_fields) > 1:
                fields = cfg_fields
                config_info = {
                    **(config_info or {}),
                    "public_name": cfg_obj.__name__,
                    "signature": f"{cfg_obj.__name__}{cfg_sig}",
                    "parameters": cfg_fields,
                }

        doc = inspect.getdoc(obj) or ""
        runtime_available = True
        runtime_error = None
        if component_type == "elasticbit_runtime":
            checker = getattr(obj, "native_runtime_available", None)
            if callable(checker):
                try:
                    runtime_available = bool(checker())
                except Exception as exc:
                    runtime_available = False
                    runtime_error = f"{type(exc).__name__}: {exc}"
            if not runtime_available and runtime_error is None:
                runtime_error = "ElasticBit native 4-32 CUDA runtime is not built in this environment."

        info = IMPORT_POOL.import_info(component_type)
        return {
            **result,
            "available": True,
            "runtime_available": runtime_available,
            "source": "runtime inspection + MLBricks source schema" if source_defined else "runtime inspection",
            "public_name": fallback.get("public_name", obj.__name__) if source_defined else obj.__name__,
            "signature": fallback.get("signature", f"{obj.__name__}{sig}") if source_defined else f"{obj.__name__}{sig}",
            "description": fallback.get("description", "") if source_defined else (doc.splitlines()[0] if doc else fallback.get("description", "")),
            "parameters": fields if fields else fallback.get("parameters", []),
            "config_api": config_info,
            "runtime_error": runtime_error,
            "loaded": True,
            "resolved_from": info.get("resolved_from"),
        }
    except Exception as exc:
        info = IMPORT_POOL.import_info(component_type)
        return {
            **result,
            "available": True,
            "runtime_available": False,
            "loaded": False,
            "source": "MLBricks source schema + lazy import pool",
            "runtime_error": f"{type(exc).__name__}: {exc}",
            "resolved_from": info.get("resolved_from"),
        }


def discover_mlbricks_api(
    component_types: Iterable[str] | None = None,
    *,
    eager: bool = False,
):
    """Return Builder API metadata without requiring eager MLBricks imports.

    By default, the supplied source-derived schema is used and canonical import
    routes are attached to each component.  If ``eager=True`` (or a specific
    ``component_types`` iterable is supplied), only those requested components
    are imported and inspected.  This is the import-pool behavior used by the
    Builder UI: adding a component can warm just that component's API instead of
    importing the whole MLBricks package.
    """
    result = _fallback_schema()
    requested = set(str(x) for x in (component_types or ()))
    inspect_all = bool(eager and component_types is None)

    for component_type in COMPONENT_IMPORTS:
        fallback = result.get(component_type, {})
        if inspect_all or component_type in requested:
            result[component_type] = _inspect_one(component_type, fallback)
        else:
            result[component_type] = _canonical_metadata(component_type, fallback)

    return result


def refresh_component_api(component_type: str) -> dict[str, Any] | None:
    component_type = str(component_type)
    if component_type not in COMPONENT_IMPORTS:
        return None
    fallback = _fallback_schema().get(component_type, {})
    return _inspect_one(component_type, fallback)
