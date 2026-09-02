from __future__ import annotations

"""Declarative runtime contracts for MLBricks components.

Studio is an orchestrator: contracts describe how a visual node maps to the
original MLBricks constructor and forward API.  The graph executor itself does
not need per-component branches.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .import_pool import IMPORT_POOL

_SCHEMA_PATH = Path(__file__).with_name("mlbricks_api_schema.json")
_SCHEMA_CACHE: dict[str, Any] | None = None


def _schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        try:
            _SCHEMA_CACHE = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        except Exception:
            _SCHEMA_CACHE = {}
    return _SCHEMA_CACHE or {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce(value: Any, annotation: str | None) -> Any:
    kind = str(annotation or "").lower()
    if value is None:
        return None
    if "bool" in kind:
        return _bool(value)
    if "int" in kind:
        return int(float(value))
    if "float" in kind:
        return float(value)
    return value


@dataclass(frozen=True)
class APIComponentContract:
    """Mapping between one Studio node and an original MLBricks API.

    ``input_ports`` maps Studio input-port names to forward argument names.
    ``output_ports`` maps Studio output-port names to result positions/keys.
    For the common one-input/one-output case no custom code is required.
    """

    component_type: str
    import_key: str
    input_ports: Mapping[str, str]
    output_ports: Mapping[str, Any]
    parameter_aliases: Mapping[str, tuple[str, ...]] = None
    runtime_sources: Mapping[str, str] = None

    def _parameter_value(self, key: str, spec: dict[str, Any], params: dict[str, Any], runtime: dict[str, Any]):
        aliases = (self.parameter_aliases or {}).get(key, ())
        candidates = (key, *aliases)
        for candidate in candidates:
            if candidate in params and params[candidate] not in (None, ""):
                return params[candidate], True
        runtime_key = (self.runtime_sources or {}).get(key)
        if runtime_key and runtime.get(runtime_key) not in (None, ""):
            return runtime[runtime_key], True
        if spec.get("value") is not None:
            return spec.get("value"), True
        if spec.get("default") is not None:
            return spec.get("default"), True
        return None, False

    def constructor_kwargs(self, node: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
        params = deepcopy(node.get("params") or {})
        component_schema = (_schema().get("components") or _schema()).get(self.component_type) or {}
        kwargs: dict[str, Any] = {}
        for spec in component_schema.get("parameters") or []:
            key = str(spec.get("key") or spec.get("name") or "").strip()
            if not key:
                continue
            value, present = self._parameter_value(key, spec, params, runtime)
            if not present:
                if spec.get("required"):
                    raise ValueError(f"{node.get('name', self.component_type)} requires {key}.")
                continue
            # UI/runtime values override constructor defaults, but 'auto' backend
            # is intentionally preserved because it is part of the original API.
            kwargs[key] = _coerce(value, spec.get("annotation") or spec.get("type"))
        return kwargs

    def instantiate(self, node: dict[str, Any], runtime: dict[str, Any]):
        cls = IMPORT_POOL.resolve_component(self.import_key)
        return cls(**self.constructor_kwargs(node, runtime))

    def execute(self, module, inputs: Mapping[str, Any]):
        kwargs = {
            api_arg: inputs[port]
            for port, api_arg in self.input_ports.items()
            if port in inputs
        }
        result = module(**kwargs)
        if len(self.output_ports) == 1 and "main" in self.output_ports:
            selector = self.output_ports["main"]
            if selector is None:
                return {"main": result}
        outputs: dict[str, Any] = {}
        for port, selector in self.output_ports.items():
            if selector is None:
                outputs[port] = result
            elif isinstance(selector, int):
                outputs[port] = result[selector]
            else:
                outputs[port] = result[selector] if isinstance(result, dict) else getattr(result, str(selector))
        return outputs


class APIComponentRegistry:
    def __init__(self):
        self._contracts: dict[str, APIComponentContract] = {}

    def register(self, contract: APIComponentContract) -> APIComponentContract:
        self._contracts[contract.component_type] = contract
        return contract

    def get(self, component_type: str | None) -> APIComponentContract | None:
        return self._contracts.get(str(component_type or ""))

    def __contains__(self, component_type: str) -> bool:
        return component_type in self._contracts


API_COMPONENTS = APIComponentRegistry()

# First proof of the universal path.  No BOLT-specific branch is required in
# TensorGraph: this declaration maps the visual node to Bolt's original API.
API_COMPONENTS.register(APIComponentContract(
    component_type="bolt",
    import_key="bolt",
    input_ports={"main": "x"},
    output_ports={"main": None},
    parameter_aliases={
        "d_model": ("dim", "hidden_size"),
        "num_heads": ("heads", "head"),
        "backend": ("kernel",),
    },
    runtime_sources={
        "d_model": "model_dim",
        "num_heads": "heads",
        "backend": "backend",
    },
))

# Multi-input proof: the existing Studio Main/Skip lanes map directly to the
# original ResController API.  Main carries the update tensor; Skip carries
# the residual stream.  TensorGraph does not need a ResController branch.
API_COMPONENTS.register(APIComponentContract(
    component_type="rescontroller",
    import_key="rescontroller",
    input_ports={"main": "update", "skip": "residual"},
    output_ports={"main": None},
    runtime_sources={
        "backend": "backend",
    },
))
