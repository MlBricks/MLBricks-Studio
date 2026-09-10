from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import ast
import math
import copy
import random
import re
import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .import_pool import IMPORT_POOL
from .api_graph_runtime import API_COMPONENTS
from .security import safe_torch_load
from .version import __version__


# Tokenizers are immutable during Studio inference. Reusing the already parsed
# tokenizer avoids repeated Hugging Face startup work on every cold generation.
_TOKENIZER_CACHE: dict[str, Any] = {}


class ModelCompileError(RuntimeError):
    pass


class TrainingStopped(RuntimeError):
    pass


class ExistingModelArtifactError(RuntimeError):
    """Existing model weights could not be loaded safely for retraining."""

    def __init__(self, path, message):
        self.path = str(path)
        super().__init__(message)


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _none(v: Any):
    if v is None: return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null"}: return None
    return v


def _literal_or_text(value: Any):
    """Parse JSON/Python literals from Builder text fields without eval()."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass
    return text


def _scalar_or_sequence(value: Any, *, cast: Callable[[Any], Any], label: str):
    parsed = _literal_or_text(value)
    if isinstance(parsed, str) and "," in parsed:
        parsed = [part.strip() for part in parsed.split(",") if part.strip()]
    if isinstance(parsed, (list, tuple)):
        try:
            return [cast(item) for item in parsed]
        except Exception as exc:
            raise ValueError(f"{label} contains an invalid value: {parsed!r}") from exc
    try:
        return cast(parsed)
    except Exception as exc:
        raise ValueError(f"{label} is invalid: {value!r}") from exc


def _config_value(value: Any, *, label: str):
    parsed = _literal_or_text(value)
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, (list, tuple)) and all(isinstance(item, dict) for item in parsed):
        return list(parsed)
    raise ValueError(f"{label} must be a JSON object or a list of JSON objects.")


def _missing_number(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def runtime_int(
    value: Any,
    default: int | None,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Convert runtime/UI values without leaking int(None) to the user."""
    if _missing_number(value):
        if default is None:
            raise ValueError(f"{label} is required.")
        value = default
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a number; received {value!r}.") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}; received {number}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}; received {number}.")
    return number


def runtime_float(
    value: Any,
    default: float | None,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Float counterpart to runtime_int with field-specific errors."""
    if _missing_number(value):
        if default is None:
            raise ValueError(f"{label} is required.")
        value = default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a number; received {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; received {value!r}.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}; received {number}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}; received {number}.")
    return number


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "model")).strip("-.")
    return value or "model"


def resolve_device(requested: str | None) -> torch.device:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available(): return torch.device("cuda:0")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available(): return torch.device("mps")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available(): return torch.device("xpu:0")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{value} was selected but CUDA is unavailable.")
    return device


def resolve_precision(name: str | None, device: torch.device) -> tuple[str, torch.dtype | None]:
    value = str(name or "auto").strip().lower()
    if value == "auto":
        value = "fp16" if device.type == "cuda" else "fp32"
    # Float16 CPU kernels are frequently unsupported or dramatically slower
    # than float32. Saved GPU-oriented projects may still request fp16 after
    # Auto falls back to CPU, so make that fallback safe and usable.
    if device.type == "cpu" and value == "fp16":
        value = "fp32"
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if value not in mapping:
        raise ValueError(f"Unsupported precision: {name!r}")
    return value, mapping[value]


def _topological(nodes: list[dict], edges: list[dict]) -> list[dict]:
    by_id={n["id"]:n for n in nodes}
    incoming={n["id"]:0 for n in nodes}
    outgoing={n["id"]:[] for n in nodes}
    for e in edges:
        a,b=e.get("source"),e.get("target")
        if a in by_id and b in by_id:
            outgoing[a].append(b); incoming[b]+=1
    q=[n["id"] for n in nodes if incoming[n["id"]]==0]
    order=[]
    while q:
        nid=q.pop(0); order.append(by_id[nid])
        for nxt in outgoing[nid]:
            incoming[nxt]-=1
            if incoming[nxt]==0:q.append(nxt)
    if len(order)!=len(nodes): raise ModelCompileError("Graph contains a cycle.")
    return order


class _Identity(nn.Module):
    def forward(self,x): return x


class _ClassifierHead(nn.Module):
    """Trainable Studio classification head.

    Two-dimensional input ``[B,D]`` is projected directly. Sequence input
    ``[B,T,D]`` is mean-pooled across T before projection so the node produces
    one class-score vector per sample.
    """

    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.dim = int(dim)
        self.classes = int(classes)
        if self.dim < 1:
            raise ValueError("Classifier hidden dim must be >= 1.")
        if self.classes < 1:
            raise ValueError("Classifier classes must be >= 1.")
        self.proj = nn.Linear(self.dim, self.classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.mean(dim=-2)
        elif x.ndim != 2:
            raise ValueError("Classifier Head expects [B,D] or [B,T,D] input.")

        if x.size(-1) != self.dim:
            raise ValueError(
                f"Classifier Head expected feature width {self.dim}, "
                f"received {x.size(-1)}."
            )
        return self.proj(x)


class _PreviousValueBuffer(nn.Module):
    """Builder utility for previous physical-depth tensors.

    ``hold`` keeps the connected tensor available as the previous-depth value
    for a later node. ``zero_init`` creates the correctly shaped zero vector
    used by the first recurrent physical depth. No trainable parameters or
    hidden mutable Python state are introduced.
    """
    def __init__(self, *, mode: str = "hold", width: int = 0):
        super().__init__()
        mode = str(mode or "hold").strip().lower()
        if mode not in {"hold", "zero_init"}:
            raise ValueError("Previous Value Buffer mode must be 'hold' or 'zero_init'.")
        self.mode = mode
        self.width = int(width or 0)
        if self.width < 0:
            raise ValueError("Previous Value Buffer width must be 0 (Auto) or a positive integer.")

    def forward(self, x):
        if self.mode == "hold":
            return x
        width = self.width or int(x.shape[-1])
        return x.new_zeros(*x.shape[:-1], width)


class _StateAwareESAStack(nn.Module):
    """Exact StateAware ESA depth stack used by the supplied 200M notebook."""
    def __init__(self, *, dim, state_dim, layers, heads, block, batch, depth_dim,
                 compass, backend, precision, update_ratio_start=0.20,
                 update_ratio_end=0.14, stream_ratio=1.08):
        super().__init__()
        # Resolve only the APIs this compound component needs. The shared
        # import pool prefers canonical submodules and caches them for reuse.
        ESA = IMPORT_POOL.resolve_component("esa")
        RMSNorm = IMPORT_POOL.resolve_component("rmsnorm")
        StateAwareFFN = IMPORT_POOL.resolve_component("saffn")
        ResController = IMPORT_POOL.resolve_component("rescontroller")
        self.dim=int(dim); self.state_dim=int(state_dim); self.layer_count=int(layers)
        if self.layer_count < 1: raise ValueError("StateAware ESA layers must be >= 1.")
        self.layers=nn.ModuleList(); self.write_gates=nn.ParameterList()
        for index in range(self.layer_count):
            depth=index/max(self.layer_count-1,1)
            ratio=float(update_ratio_start)+depth*(float(update_ratio_end)-float(update_ratio_start))
            self.layers.append(nn.ModuleDict({
                "norm": RMSNorm(self.dim),
                "esa": ESA(embd=self.dim, head=int(heads), batch=int(batch), block=int(block),
                           backend=backend, precision=precision, compass=int(compass), auto_compile=False),
                "ffn": StateAwareFFN(d_model=self.dim, state_dim=self.state_dim,
                                     depth_embedding_dim=int(depth_dim), layer_index=index,
                                     total_layers=self.layer_count, backend="pytorch"),
                "residual": ResController(update_ratio=ratio, stream_ratio=float(stream_ratio),
                                          update_softness=8.0, stream_softness=8.0, backend="pytorch"),
            }))
            self.write_gates.append(nn.Parameter(torch.tensor(-1.0)))
    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self,x):
        state=x.new_zeros(*x.shape[:-1],self.state_dim); previous_esa=torch.zeros_like(x)
        for layer,write_gate in zip(self.layers,self.write_gates):
            z=layer["norm"](x); esa_update=layer["esa"](z)
            ffn_update,state=layer["ffn"](z,esa_update,previous_esa,state)
            update=torch.sigmoid(write_gate)*(esa_update.float()+ffn_update.float())
            x=layer["residual"](x,update); previous_esa=esa_update
        return x


def _custom_coerce(value: Any, type_name: str, *, label: str) -> Any:
    kind = str(type_name or "str").lower()
    if kind in {"int", "integer", "number-int"}:
        try:
            return int(value)
        except Exception as exc:
            raise ModelCompileError(f"{label} must be an integer, got {value!r}.") from exc
    if kind in {"float", "number", "number-float"}:
        try:
            return float(value)
        except Exception as exc:
            raise ModelCompileError(f"{label} must be a number, got {value!r}.") from exc
    if kind in {"bool", "boolean"}:
        return _bool(value)
    if kind in {"json", "dict", "list", "tuple"}:
        parsed = _literal_or_text(value)
        if kind == "dict" and not isinstance(parsed, dict):
            raise ModelCompileError(f"{label} must be a JSON object.")
        if kind in {"list", "tuple"} and not isinstance(parsed, (list, tuple)):
            raise ModelCompileError(f"{label} must be a JSON list.")
        return tuple(parsed) if kind == "tuple" else parsed
    if kind in {"none", "null"}:
        return None
    return str(value) if value is not None else ""


def _bound_parameter_value(spec: dict[str, Any], params: dict[str, Any], runtime: dict[str, Any], x=None, skip=None, extra=None):
    name = str(spec.get("name") or spec.get("key") or "argument")
    source = str(spec.get("source") or "user").lower()
    if source in {"input", "main"}:
        return x
    if source == "skip":
        return skip
    if source == "extra":
        return extra
    if source == "device":
        return str(runtime.get("device") or "auto")
    if source in {"dtype", "precision"}:
        precision = str(runtime.get("precision") or "fp16")
        return {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32}.get(precision.lower(), precision)
    if source == "model_dim":
        raw = runtime.get("model_dim", params.get(name, spec.get("default")))
    elif source == "heads":
        raw = runtime.get("heads", params.get(name, spec.get("default")))
    elif source == "context":
        raw = runtime.get("context_length", params.get(name, spec.get("default")))
    elif source == "batch":
        raw = runtime.get("batch_size", params.get(name, spec.get("default")))
    else:
        raw = params.get(name, spec.get("default"))
    if (raw is None or (isinstance(raw, str) and raw == "")) and not spec.get("required"):
        return None
    return _custom_coerce(raw, str(spec.get("type") or "str"), label=name)


def _api_binding_import_path(binding: dict[str, Any]) -> str:
    path = str(binding.get("import_path") or "").strip()
    if path:
        return path
    module = str(binding.get("module_path") or "").strip().strip(".")
    symbol = str(binding.get("symbol") or "").strip().strip(".")
    return ".".join(part for part in (module, symbol) if part)


class _LayerBlock(nn.Module):
    """Explicit ESA layer with separate Signal and Residual lanes.

    Studio historically flattened nested layers into one main tensor lane.  This
    block keeps the residual stream explicit at the graph boundary so layer-to-
    layer wiring cannot lose or reinterpret the skip path.  Signal Out and
    Residual Out intentionally carry the same post-residual tensor for the next
    canonical Pre-LN block; they remain separate graph ports for architecture
    visibility and future state-aware variants.
    """

    def __init__(
        self, *, dim: int, heads: int, ffn_dim: int, batch: int, block: int,
        backend: str, precision: str, compass: int = 16, activation: str = "gelu",
        dropout: float = 0.0, norm_eps: float = 1e-5, device: str = "auto",
    ):
        super().__init__()
        LayerNorm = IMPORT_POOL.resolve_component("layernorm")
        ESA = IMPORT_POOL.resolve_component("esa")
        FFN = IMPORT_POOL.resolve_component("ffn")

        self.dim = int(dim)
        self.norm1 = LayerNorm(self.dim, eps=float(norm_eps), elementwise_affine=True, bias=True)
        self.esa = ESA(
            embd=self.dim, head=int(heads), batch=int(batch), block=int(block),
            backend=backend, precision=precision, compass=int(compass),
            dropout=float(dropout), gate_min=0.8, gate_max=0.995, eps=1e-5,
            device=device, auto_compile=False, auto_move_input=True, strict_checks=False,
        )
        self.norm2 = LayerNorm(self.dim, eps=float(norm_eps), elementwise_affine=True, bias=True)
        self.ffn = FFN(
            self.dim, int(ffn_dim), activation=activation, dropout=float(dropout),
            bias=True, gated=False,
        )
        self.update_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    @staticmethod
    def _result(value):
        # Named ports plus universal-lane aliases keep both explicit wiring and
        # older Main/Skip graph consumers compatible.
        return {
            "main": value,
            "skip": value,
            "signal": value,
            "residual": value,
        }

    def _finish(self, signal, residual, esa_update):
        base = signal if residual is None else residual
        residual_mid = base + self.update_dropout(esa_update)
        ffn_update = self.ffn(self.norm2(residual_mid))
        residual_out = residual_mid + self.update_dropout(ffn_update)
        return self._result(residual_out)

    def forward(self, signal, residual=None):
        if signal is None:
            raise ModelCompileError("Layer Block requires Signal In.")
        esa_update = self.esa(self.norm1(signal))
        return self._finish(signal, residual, esa_update)

    @torch.no_grad()
    def prefill(self, signal, residual=None):
        prefill = getattr(self.esa, "prefill", None)
        if not callable(prefill):
            raise ModelCompileError("Layer Block ESA does not expose prefill().")
        esa_update, state = prefill(self.norm1(signal))
        return self._finish(signal, residual, esa_update), state

    @torch.no_grad()
    def decode_step(self, signal, state, residual=None):
        decode = getattr(self.esa, "decode_step", None)
        if not callable(decode):
            raise ModelCompileError("Layer Block ESA does not expose decode_step().")
        esa_update, state = decode(self.norm1(signal), state)
        return self._finish(signal, residual, esa_update), state


class _APILaneOutputs:
    """Internal three-lane result produced by an API/User Function node."""
    __slots__ = ("main", "skip", "extra")

    def __init__(self, main=None, skip=None, extra=None):
        self.main = main
        self.skip = skip
        self.extra = extra


class _APINamedOutputs:
    """Arbitrary named outputs produced by a User Defined Function node."""
    __slots__ = ("values", "main")

    def __init__(self, values):
        self.values = dict(values or {})
        self.main = next(iter(self.values.values()), None)


class _APIMixedOutputs:
    """Standard Main/Skip/Extra outputs plus user-created named terminals.

    Custom API terminals are additive: the six universal Studio sockets remain
    available while extra named outputs can be wired independently.
    """
    __slots__ = ("standard", "values")

    def __init__(self, standard, values):
        self.standard = standard
        self.values = dict(values or {})


def _lane_output(value, lane):
    if isinstance(value, dict):
        if lane in value:
            return value[lane]
        if lane == "main" and "main" in value:
            return value["main"]
        raise ModelCompileError(f"API output lane {lane!r} is connected but has no mapped return value.")
    if isinstance(value, _APIMixedOutputs):
        return _lane_output(value.standard, lane)
    if isinstance(value, _APILaneOutputs):
        selected = getattr(value, lane, None)
        if selected is None:
            raise ModelCompileError(f"API output lane {lane!r} is connected but has no mapped return value.")
        return selected
    if isinstance(value, _APINamedOutputs):
        if lane in value.values:
            return value.values[lane]
        if lane == "main" and value.main is not None:
            return value.main
        raise ModelCompileError(f"Named User Function output has no {lane!r} value.")
    return value


def _named_output(value, key):
    key = str(key or "").strip()
    if isinstance(value, dict):
        lookup = "main" if key in {"", "output"} else key
        if lookup in value:
            return value[lookup]
        raise ModelCompileError(f"API contract output has no named output port {key!r}.")
    if isinstance(value, _APIMixedOutputs):
        if key in value.values:
            return value.values[key]
        if key in {"", "main", "skip", "extra", "output"}:
            return _lane_output(value.standard, "main" if key in {"", "output"} else key)
        raise ModelCompileError(f"Custom API has no named output terminal {key!r}.")
    if isinstance(value, _APINamedOutputs):
        if key in value.values:
            return value.values[key]
        raise ModelCompileError(f"User Function has no named output port {key!r}.")
    if isinstance(value, _APILaneOutputs) and key in {"main", "skip", "extra"}:
        return _lane_output(value, key)
    if key in {"", "main", "skip", "extra", "output"}:
        return _lane_output(value, "main" if key in {"", "output"} else key)
    raise ModelCompileError(f"Output {key!r} is not available from the connected source node.")


class _APIOperation(nn.Module):
    """One Python/PyTorch operation inside a user-authored API Component graph.

    V1.0 object-aware API nodes distinguish imports from object instances.  A
    node can create/register an object once, call normal/static/class methods,
    or reuse an object created by another node.  The registry is scoped to one
    TensorGraph instance so state is preserved across later API nodes without
    re-constructing the object.
    """

    _CALL_TYPES = {"function", "user_function", "user_class", "static_method", "class_method", "instance_method", "constructor"}

    def __init__(self, *, binding, params, runtime, label, object_registry=None):
        super().__init__()
        self.binding = deepcopy(binding or {})
        self.params = deepcopy(params or {})
        self.runtime = deepcopy(runtime or {})
        self.label = str(label or "API Function")
        self.object_registry = object_registry if object_registry is not None else {}
        self.parameters = list(self.binding.get("parameters") or [])

        legacy_kind = str(self.binding.get("target_kind") or "module").lower()
        call_type = str(self.binding.get("call_type") or "").strip().lower()
        if not call_type:
            call_type = "function" if legacy_kind == "function" else "instance_method"
        if call_type not in self._CALL_TYPES:
            raise ModelCompileError(f"API function {self.label!r} has unsupported call type {call_type!r}.")
        self.call_type = call_type
        self.object_mode = "existing" if str(self.binding.get("object_mode") or "new").lower() == "existing" else "new"
        self.object_id = str(self.binding.get("object_id") or f"object::{self.label}").strip()
        self.object_name = str(self.binding.get("object_name") or self.label).strip()
        self.object_ref = str(self.binding.get("object_ref") or "").strip()
        self.result_object_id = str(self.binding.get("result_object_id") or f"result::{self.label}").strip()
        self.result_object_name = str(self.binding.get("result_object_name") or f"{self.label} result").strip()
        self.auto_main_input = bool(self.binding.get("auto_main_input", True))
        self.register_result_object = bool(self.binding.get("register_result_object"))
        self.result_output_mode = "passthrough" if str(self.binding.get("result_output_mode") or "result").lower() == "passthrough" else "result"
        self.multi_output = bool(self.binding.get("multi_output"))
        self.output_map = deepcopy(self.binding.get("output_map") or {})
        _port_mode = str(self.binding.get("port_mode") or "standard").lower()
        self.port_mode = _port_mode if _port_mode in {"standard", "extended", "named"} else "standard"
        self.input_ports = deepcopy(self.binding.get("input_ports") or [])
        self.output_ports = deepcopy(self.binding.get("output_ports") or [])

        self.api_target = None
        self.created_object = None

        if not _bool(self.runtime.get("allow_user_code", True)):
            raise ModelCompileError(
                f"Executable API component {self.label!r} is blocked because this project is untrusted. "
                "Review the project's Python source/import bindings, then call builder.trust_project() "
                "in the current session before training, generation, or serving."
            )

        # User-authored source is embedded in the component/project cache.  It is
        # compiled in the user's active Python environment.  Third-party imports
        # are deliberately not installed by Studio; missing libraries produce an
        # explicit install-the-dependency error instead.
        if self.call_type in {"user_function", "user_class"}:
            is_class = self.call_type == "user_class"
            source_key = "user_class_code" if is_class else "user_code"
            name_key = "user_class_name" if is_class else "user_function_name"
            kind_label = "User Class" if is_class else "User Function"
            source = str(self.binding.get(source_key) or "").strip()
            entry_name = str(self.binding.get(name_key) or "").strip()
            if not source:
                raise ModelCompileError(f"{kind_label} {self.label!r} has no Python source code.")
            if not entry_name:
                raise ModelCompileError(f"{kind_label} {self.label!r} has no entry name configured.")
            namespace = {"torch": torch, "nn": nn}
            try:
                exec(compile(source, f"<MLB Studio:{self.label}>", "exec"), namespace, namespace)
            except ModuleNotFoundError as exc:
                missing = getattr(exc, "name", None) or str(exc)
                raise ModelCompileError(
                    f"{kind_label} {self.label!r} needs dependency {missing!r}. "
                    f"Install it explicitly in the active Python environment before building this component."
                ) from exc
            except Exception as exc:
                raise ModelCompileError(f"Could not compile {kind_label} {self.label!r}: {exc}") from exc
            target = namespace.get(entry_name)
            if not callable(target):
                raise ModelCompileError(f"{kind_label} {self.label!r} did not define callable {entry_name!r}.")
            self.api_target = target
            if is_class:
                init_specs = [spec for spec in self.parameters if str(spec.get("stage") or "init").lower() == "init"]
                init_args, init_kwargs = self._build_arguments(init_specs, x=None)
                try:
                    instance = target(*init_args, **init_kwargs)
                except Exception as exc:
                    raise ModelCompileError(
                        f"Could not construct user object {self.object_name!r} from class {entry_name!r}: {exc}"
                    ) from exc
                self.created_object = instance
                self.object_registry[self.object_id] = instance
            return

        # Reusing an existing object does not need another import or constructor.
        if self.call_type == "instance_method" and self.object_mode == "existing":
            if not self.object_ref:
                raise ModelCompileError(f"API instance method {self.label!r} has no existing object selected.")
            return

        import_path = _api_binding_import_path(self.binding)
        if not import_path:
            raise ModelCompileError(f"API function {self.label!r} has no import/module + function/class configured.")
        target = IMPORT_POOL.resolve_external(import_path)
        self.api_target = target

        if self.call_type in {"constructor", "instance_method"} and self.object_mode == "new":
            init_specs = [spec for spec in self.parameters if str(spec.get("stage") or "init").lower() == "init"]
            init_args, init_kwargs = self._build_arguments(init_specs, x=None)
            if not callable(target):
                raise ModelCompileError(f"API target {import_path} is not constructible/callable.")
            try:
                instance = target(*init_args, **init_kwargs)
            except Exception as exc:
                raise ModelCompileError(
                    f"Could not construct API object {self.object_name!r} for {self.label!r} from {import_path}: {exc}"
                ) from exc
            # If this is an nn.Module, assigning it here registers its parameters
            # exactly once with PyTorch.  Reuser nodes keep only the plain shared
            # registry reference and therefore do not duplicate module ownership.
            self.created_object = instance
            self.object_registry[self.object_id] = instance
        elif self.call_type == "function":
            if not callable(target):
                raise ModelCompileError(f"API target {import_path} is not callable.")
        elif self.call_type in {"static_method", "class_method"}:
            method = str(self.binding.get("call_method") or "").strip()
            if not method:
                raise ModelCompileError(f"API {self.call_type.replace('_', ' ')} {self.label!r} needs a method name.")
            try:
                candidate = getattr(target, method)
            except Exception as exc:
                raise ModelCompileError(f"API target {import_path} has no method {method!r}.") from exc
            if not callable(candidate):
                raise ModelCompileError(f"API target {import_path}.{method} is not callable.")

    def _build_arguments(self, specs, x, skip=None, extra=None):
        positional = []
        keywords = {}
        for spec in specs:
            name = str(spec.get("name") or spec.get("key") or "").strip()
            if not name and not spec.get("positional"):
                continue
            value = _bound_parameter_value(spec, self.params, self.runtime, x=x, skip=skip, extra=extra)
            if value is None and not spec.get("required"):
                continue
            if bool(spec.get("positional")):
                positional.append(value)
            else:
                keywords[name] = value
        return positional, keywords

    @staticmethod
    def _select_output(value, selector):
        text = str(selector or "auto").strip()
        if text in {"", "auto"}:
            if torch.is_tensor(value):
                return value
            if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                return value[0]
            return value
        if isinstance(value, (list, tuple)):
            try:
                return value[int(text)]
            except Exception:
                pass
        if isinstance(value, dict) and text in value:
            return value[text]
        if hasattr(value, text):
            return getattr(value, text)
        raise ModelCompileError(f"API output selector {text!r} could not be applied for {self.label!r}.")

    def _resolve_call_target(self):
        method = str(self.binding.get("call_method") or "").strip()
        if self.call_type in {"function", "user_function"}:
            return self.api_target
        if self.call_type in {"static_method", "class_method"}:
            return getattr(self.api_target, method)
        if self.call_type == "instance_method":
            if self.object_mode == "existing":
                if self.object_ref not in self.object_registry:
                    raise ModelCompileError(
                        f"API node {self.label!r} references object {self.object_ref!r}, but it is not available yet. "
                        "Create/register that object in an upstream API node first."
                    )
                instance = self.object_registry[self.object_ref]
            else:
                instance = self.created_object
            if instance is None:
                raise ModelCompileError(f"API node {self.label!r} has no object instance available.")
            if method and method not in {"__call__", "forward"}:
                try:
                    return getattr(instance, method)
                except Exception as exc:
                    raise ModelCompileError(
                        f"Object {self.object_name!r} used by {self.label!r} has no method {method!r}."
                    ) from exc
            return instance
        raise ModelCompileError(f"API node {self.label!r} is a constructor and has no call target.")

    def forward(self, x, skip=None, extra=None, named_inputs=None):
        # A constructor is an object-lifecycle node, not a tensor transform.  It
        # creates/registers the object once in __init__ and transparently passes
        # the Main lane through at execution time.
        if self.call_type in {"constructor", "user_class"}:
            return x

        call_specs = [spec for spec in self.parameters if str(spec.get("stage") or "call").lower() == "call"]
        if call_specs:
            args, kwargs = self._build_arguments(call_specs, x=x, skip=skip, extra=extra)
            tensor_bound = any(
                str(spec.get("source") or "user").lower() in {"input", "main", "skip", "extra"}
                for spec in call_specs
            )
            if not tensor_bound and self.auto_main_input:
                args = [x, *args]
        else:
            args, kwargs = ([x], {}) if self.auto_main_input else ([], {})

        # User-created terminals are additive in ``extended`` mode: Main/Skip/
        # Extra keep their universal behavior and these values are appended to
        # the call using the configured function-parameter mapping.  ``named``
        # remains supported for legacy designs whose custom ports replaced the
        # universal layout.
        if self.input_ports and self.port_mode in {"extended", "named"}:
            named_inputs = dict(named_inputs or {})
            for port in self.input_ports:
                port_id = str(port.get("id") or "").strip()
                port_name = str(port.get("name") or port_id).strip()
                parameter = str(port.get("parameter") or port_name).strip()
                if port_id not in named_inputs:
                    if port.get("required", True):
                        raise ModelCompileError(
                            f"API function {self.label!r} input terminal {port_name!r} is not connected."
                        )
                    continue
                value = named_inputs[port_id]
                if bool(port.get("positional")):
                    args.append(value)
                else:
                    kwargs[parameter] = value

        target = self._resolve_call_target()
        try:
            out = target(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"API function {self.label!r} failed: {exc}") from exc

        custom_outputs = {}
        if self.output_ports and self.port_mode in {"extended", "named"}:
            for port in self.output_ports:
                port_id = str(port.get("id") or port.get("name") or "output").strip()
                selector = port.get("selector", "auto")
                custom_outputs[port_id] = self._select_output(out, selector)

        if self.register_result_object:
            self.object_registry[self.result_object_id] = out
            if self.result_output_mode == "passthrough":
                standard_result = x
                return _APIMixedOutputs(standard_result, custom_outputs) if custom_outputs else standard_result

        # Legacy named-only layouts keep their historical behavior.
        if self.port_mode == "named":
            if not custom_outputs:
                custom_outputs = {"output": self._select_output(out, "auto")}
            return _APINamedOutputs(custom_outputs)

        if self.multi_output:
            mapping = self.output_map or {}
            def selected(lane, default=""):
                selector = str(mapping.get(lane, default) or "").strip()
                if not selector:
                    return None
                return self._select_output(out, selector)
            main_value = selected("main", "0")
            if main_value is None:
                raise ModelCompileError(f"User/API function {self.label!r} needs a Main output mapping.")
            standard_result = _APILaneOutputs(
                main=main_value,
                skip=selected("skip"),
                extra=selected("extra"),
            )
        else:
            standard_result = self._select_output(out, self.binding.get("output_selector"))

        if custom_outputs:
            return _APIMixedOutputs(standard_result, custom_outputs)
        return standard_result


class _APIBoundComponent(nn.Module):
    """Reusable API Component represented as an explicit mixed execution DAG.

    API Components may contain ``api_step`` nodes, supported built-in Builder
    components, and reusable graph Modules. Nested API Components are kept out
    of the authoring UI, while the shared custom-definition stack still guards
    against circular Module dependencies. Legacy single-binding API Components
    remain supported.
    """

    def __init__(self, *, definition, params, runtime, custom_components=None, _custom_stack=()):
        super().__init__()
        self.definition = deepcopy(definition)
        self.params = deepcopy(params or {})
        self.runtime = deepcopy(runtime or {})
        self.custom_components = custom_components or {}
        self._custom_stack = tuple(_custom_stack or ())
        nodes = [deepcopy(n) for n in (self.definition.get("nodes") or [])]
        self.legacy = None
        self.graph = None
        if not nodes:
            binding = self.definition.get("api_binding") or {}
            self.legacy = _APIOperation(
                binding=binding,
                params=self.params,
                runtime=self.runtime,
                label=self.definition.get("name") or "API Component",
                object_registry={},
            )
            return

        # Apply exposed API parameter values to their owning api_step before the
        # generic TensorGraph builds modules. Built-in component parameters are
        # already serialized directly on their nodes.
        for node in nodes:
            if str(node.get("type") or "") != "api_step":
                continue
            binding = deepcopy(node.get("api_binding") or {})
            node_params = node.setdefault("params", {})
            for spec in binding.get("parameters") or []:
                name = str(spec.get("name") or spec.get("key") or "").strip()
                if not name:
                    continue
                expose_key = str(spec.get("expose_key") or f"{node.get('id')}::{name}")
                if expose_key in self.params:
                    node_params[name] = self.params[expose_key]
                elif name not in node_params and spec.get("default") is not None:
                    node_params[name] = spec.get("default")

        # TensorGraph is defined below this class and is available by the time a
        # compiled model instantiates an API component.
        self.graph = TensorGraph(
            nodes=nodes,
            edges=deepcopy(self.definition.get("edges") or []),
            custom_components=self.custom_components,
            runtime=self.runtime,
            _custom_stack=self._custom_stack,
        )

    def forward(self, x, skip=None, extra=None):
        if self.legacy is not None:
            return self.legacy(x, skip=skip, extra=extra)
        return self.graph(x, graph_skip=skip, graph_extra=extra)


class TensorGraph(nn.Module):
    """Small tensor DAG compiler for the model components Builder can execute today."""
    def __init__(self, *, nodes, edges, custom_components, runtime, vocab_override=None, _custom_stack=()):
        super().__init__()
        self.nodes=deepcopy(nodes)
        self.edges=deepcopy(edges)
        self.custom_components=custom_components
        self.runtime=runtime
        self.vocab_override=vocab_override
        self._custom_stack=tuple(_custom_stack or ())
        self.order=_topological(self.nodes,self.edges)
        self.by_id={n["id"]:n for n in self.nodes}
        # Shared only by API operation nodes in this graph instance.  Directly
        # created objects are registered during module construction; results can
        # be registered during forward for later upstream-connected API nodes.
        self.api_object_registry={}
        self.in_main={n["id"]:[] for n in self.nodes}
        self.in_skip={n["id"]:[] for n in self.nodes}
        self.in_extra={n["id"]:[] for n in self.nodes}
        self.in_main_edges={n["id"]:[] for n in self.nodes}
        self.in_skip_edges={n["id"]:[] for n in self.nodes}
        self.in_extra_edges={n["id"]:[] for n in self.nodes}
        self.in_named={n["id"]:[] for n in self.nodes}
        self.outgoing={n["id"]:[] for n in self.nodes}
        for e in self.edges:
            a,b=e.get("source"),e.get("target")
            if a not in self.by_id or b not in self.by_id: continue
            kind=str(e.get("kind") or "main").lower()
            target_port=str(e.get("target_port") or "").strip().lower()

            # Destination routing is determined by the *target* port, not by
            # whether the source happens to be a named API output. A stateful
            # component may expose ``named_out:main`` and feed that tensor into
            # an ordinary downstream ``main_in`` (for example SAFFN -> ESA).
            if target_port.startswith("named_in:"):
                self.in_named[b].append(deepcopy(e))
            elif target_port in {"main", "main_in"}:
                self.in_main[b].append(a);self.in_main_edges[b].append(deepcopy(e))
            elif target_port in {"skip", "skip_in"}:
                self.in_skip[b].append(a);self.in_skip_edges[b].append(deepcopy(e))
            elif target_port in {"extra", "extra_in"}:
                self.in_extra[b].append(a);self.in_extra_edges[b].append(deepcopy(e))
            elif kind == "named":
                # Backward compatibility for older serialized named-input
                # edges that omitted an explicit named_in:* target port.
                self.in_named[b].append(deepcopy(e))
            elif kind in {"residual","skip"}:
                self.in_skip[b].append(a);self.in_skip_edges[b].append(deepcopy(e))
            elif kind in {"main",""}:
                self.in_main[b].append(a);self.in_main_edges[b].append(deepcopy(e))
            elif kind in {"aux","extra"}:
                self.in_extra[b].append(a);self.in_extra_edges[b].append(deepcopy(e))
            else:
                self.in_main[b].append(a);self.in_main_edges[b].append(deepcopy(e))
            self.outgoing[a].append(b)
        self.mods=nn.ModuleDict()
        for node in self.nodes:
            mod=self._module_for(node)
            if mod is not None:self.mods[node["id"]]=mod
        self._apply_weight_tying()

    def _module_for(self,node):
        # MLBricks symbols are imported lazily per component. Adding/using one
        # component must not depend on unrelated top-level exports being present.
        t=node.get("type"); p=deepcopy(node.get("params") or {})
        device=str(self.runtime.get("device") or "auto")
        backend=str(self.runtime.get("backend") or "pytorch")
        precision=str(self.runtime.get("precision") or "fp16")
        if precision=="auto": precision="fp16" if resolve_device(device).type=="cuda" else "fp32"
        contract = API_COMPONENTS.get(t)
        if contract is not None:
            try:
                return contract.instantiate(node, {**self.runtime, "device": device, "backend": backend, "precision": precision})
            except (TypeError, ValueError) as exc:
                raise ModelCompileError(f"Could not construct {node.get('name') or t} from its MLBricks API contract: {exc}") from exc
        if t=="api_step":
            binding=deepcopy(node.get("api_binding") or {})
            step_params=deepcopy(node.get("params") or {})
            return _APIOperation(
                binding=binding,
                params=step_params,
                runtime=self.runtime,
                label=node.get("name") or "API Function",
                object_registry=self.api_object_registry,
            )
        if t in {"text_input","image_input","audio_input","text_output","logits_output"}: return _Identity()
        if t=="classifier":
            default_dim = runtime_int(
                self.runtime.get("model_dim"),
                384,
                f"{node.get('name','Classifier Head')} runtime model dim",
                minimum=1,
            )
            return _ClassifierHead(
                dim=runtime_int(
                    p.get("dim") or p.get("hidden_size"),
                    default_dim,
                    f"{node.get('name','Classifier Head')} hidden dim",
                    minimum=1,
                ),
                classes=runtime_int(
                    p.get("classes"),
                    10,
                    f"{node.get('name','Classifier Head')} classes",
                    minimum=1,
                ),
            )
        if t=="value_buffer":
            return _PreviousValueBuffer(
                mode=str(p.get("mode") or "hold"),
                width=runtime_int(p.get("width"),0,f"{node.get('name','Previous Value Buffer')} output width",minimum=0),
            )
        if t=="embedding":
            Embedding = IMPORT_POOL.resolve_component("embedding")
            vocab=int(self.vocab_override or p.get("vocab_size") or p.get("num_embeddings") or 32000)
            dim=int(p.get("embedding_dim") or p.get("hidden_size") or 384)
            return Embedding(vocab,dim)
        if t=="lm_head":
            LMHead = IMPORT_POOL.resolve_component("lm_head")
            vocab=int(self.vocab_override or p.get("vocab_size") or 32000)
            hidden=int(p.get("hidden_size") or p.get("dim") or 384)
            # ``tie_to`` is a module reference in the MLBricks Python API. Builder
            # resolves that reference after the graph modules have been created.
            return LMHead(hidden,vocab,bias=_bool(p.get("bias",False)))
        if t=="learned_position":
            LearnedPosition = IMPORT_POOL.resolve_component("learned_position")
            return LearnedPosition(
                runtime_int(p.get("dim"),384,f"{node.get('name','Learned Position')} dim",minimum=1),
                runtime_int(p.get("max_seq_len"),65536,f"{node.get('name','Learned Position')} max sequence length",minimum=1),
            )
        if t=="sinusoidal_position":
            SinusoidalPosition = IMPORT_POOL.resolve_component("sinusoidal_position")
            return SinusoidalPosition(
                runtime_int(p.get("dim"),384,f"{node.get('name','Sinusoidal Position')} dim",minimum=1),
                runtime_int(p.get("max_seq_len"),65536,f"{node.get('name','Sinusoidal Position')} max sequence length",minimum=1),
                base=runtime_float(p.get("base"),10000.0,f"{node.get('name','Sinusoidal Position')} base",minimum=1e-9),
            )
        if t=="esa":
            ESA = IMPORT_POOL.resolve_component("esa")
            return ESA(
                embd=runtime_int(p.get("embd"),384,f"{node.get('name','ESA')} embedding size",minimum=1),
                head=runtime_int(p.get("head"),4,f"{node.get('name','ESA')} head count",minimum=1),
                batch=_none(p.get("batch")), block=_none(p.get("block")),
                backend=backend, precision=precision, compass=p.get("compass","auto"),
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','ESA')} dropout",minimum=0.0),
                gate_min=runtime_float(p.get("gate_min"),0.8,f"{node.get('name','ESA')} gate min"),
                gate_max=runtime_float(p.get("gate_max"),0.995,f"{node.get('name','ESA')} gate max"),
                eps=runtime_float(p.get("eps"),1e-5,f"{node.get('name','ESA')} epsilon",minimum=0.0),
                device=device, auto_compile=False, auto_move_input=True,
                strict_checks=_bool(p.get("strict_checks",False)),
            )
        if t=="stateaware_esa_stack":
            return _StateAwareESAStack(
                dim=runtime_int(p.get("dim"),384,f"{node.get('name','StateAware ESA')} model dim",minimum=1),
                state_dim=runtime_int(p.get("state_dim"),2749,f"{node.get('name','StateAware ESA')} state dim",minimum=1),
                layers=runtime_int(p.get("layers"),8,f"{node.get('name','StateAware ESA')} layer count",minimum=1),
                heads=runtime_int(p.get("heads"),6,f"{node.get('name','StateAware ESA')} head count",minimum=1),
                block=runtime_int(p.get("block"),256,f"{node.get('name','StateAware ESA')} block size",minimum=1),
                batch=runtime_int(p.get("batch"),16,f"{node.get('name','StateAware ESA')} batch",minimum=1),
                depth_dim=runtime_int(p.get("depth_dim"),64,f"{node.get('name','StateAware ESA')} depth embedding dim",minimum=1),
                compass=runtime_int(p.get("compass"),16,f"{node.get('name','StateAware ESA')} compass",minimum=1),
                backend=backend, precision=precision,
                update_ratio_start=runtime_float(p.get("update_ratio_start"),0.20,f"{node.get('name','StateAware ESA')} update ratio start",minimum=1e-9),
                update_ratio_end=runtime_float(p.get("update_ratio_end"),0.14,f"{node.get('name','StateAware ESA')} update ratio end",minimum=1e-9),
                stream_ratio=runtime_float(p.get("stream_ratio"),1.08,f"{node.get('name','StateAware ESA')} stream ratio",minimum=1e-9),
            )
        if t=="soup":
            SOUP = IMPORT_POOL.resolve_component("soup")
            depth = runtime_int(p.get("depth"), 2, f"{node.get('name','SOUP')} depth", minimum=1)
            width = _scalar_or_sequence(
                p.get("width", 1116), cast=int, label=f"{node.get('name','SOUP')} state width"
            )
            mixer = _scalar_or_sequence(
                p.get("mixer", "esa"), cast=lambda v: str(v).strip().lower(),
                label=f"{node.get('name','SOUP')} mixer",
            )
            ffn = _scalar_or_sequence(
                p.get("ffn", "saffn"), cast=lambda v: str(v).strip().lower(),
                label=f"{node.get('name','SOUP')} FFN",
            )
            return SOUP(
                dim=runtime_int(p.get("dim"), 512, f"{node.get('name','SOUP')} model dim", minimum=1),
                width=width,
                depth=depth,
                mixer=mixer,
                ffn=ffn,
                mixer_config=_config_value(p.get("mixer_config"), label=f"{node.get('name','SOUP')} mixer config"),
                ffn_config=_config_value(p.get("ffn_config"), label=f"{node.get('name','SOUP')} FFN config"),
                backend=backend,
                precision=precision,
                memory_dim=runtime_int(p.get("memory_dim"), 128, f"{node.get('name','SOUP')} memory dim", minimum=1),
                fusion_hidden=runtime_int(p.get("fusion_hidden"), 768, f"{node.get('name','SOUP')} fusion hidden", minimum=1),
            )
        if t=="rmsnorm":
            RMSNorm = IMPORT_POOL.resolve_component("rmsnorm")
            shape=p.get("normalized_shape",p.get("hidden_size",384))
            shape=runtime_int(shape,384,f"{node.get('name','RMSNorm')} normalized shape",minimum=1)
            return RMSNorm(shape,eps=runtime_float(p.get("eps"),1e-6,f"{node.get('name','RMSNorm')} epsilon",minimum=0.0),elementwise_affine=_bool(p.get("elementwise_affine",True)))
        if t=="layernorm":
            LayerNorm = IMPORT_POOL.resolve_component("layernorm")
            shape=p.get("normalized_shape",p.get("hidden_size",p.get("dim",384)))
            shape=runtime_int(shape,384,f"{node.get('name','LayerNorm')} normalized shape",minimum=1)
            return LayerNorm(shape,eps=runtime_float(p.get("eps"),1e-5,f"{node.get('name','LayerNorm')} epsilon",minimum=0.0),elementwise_affine=_bool(p.get("elementwise_affine",True)),bias=_bool(p.get("bias",True)))
        if t=="linear":
            Linear = IMPORT_POOL.resolve_component("linear")
            return Linear(
                runtime_int(p.get("in_features"),384,f"{node.get('name','Linear')} input features",minimum=1),
                runtime_int(p.get("out_features"),384,f"{node.get('name','Linear')} output features",minimum=1),
                bias=_bool(p.get("bias",True)),
            )
        if t=="ffn":
            FFN = IMPORT_POOL.resolve_component("ffn")
            return FFN(
                runtime_int(p.get("hidden_size"),384,f"{node.get('name','FFN')} hidden size",minimum=1),
                runtime_int(
                    p.get("intermediate_size"),
                    4*runtime_int(p.get("hidden_size"),384,f"{node.get('name','FFN')} hidden size",minimum=1),
                    f"{node.get('name','FFN')} intermediate size",minimum=1,
                ),
                activation=p.get("activation") or "gelu",
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','FFN')} dropout",minimum=0.0),
                bias=_bool(p.get("bias",True)), gated=_bool(p.get("gated",False)),
            )
        if t=="residual":
            Residual = IMPORT_POOL.resolve_component("residual")
            return Residual(dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','Residual')} dropout",minimum=0.0))
        if t=="dropout":
            probability=p.get("p") if p.get("p") is not None else p.get("dropout")
            return nn.Dropout(runtime_float(probability,0.1,f"{node.get('name','Dropout')} probability",minimum=0.0,maximum=1.0))
        if t=="layer_block":
            dim=runtime_int(p.get("dim") or p.get("hidden_size"),384,f"{node.get('name','Layer Block')} hidden dim",minimum=1)
            return _LayerBlock(
                dim=dim,
                heads=runtime_int(p.get("heads") or p.get("head"),4,f"{node.get('name','Layer Block')} ESA heads",minimum=1),
                ffn_dim=runtime_int(p.get("ffn_dim") or p.get("intermediate_size"),4*dim,f"{node.get('name','Layer Block')} FFN hidden dim",minimum=1),
                batch=runtime_int(p.get("batch"),16,f"{node.get('name','Layer Block')} batch",minimum=1),
                block=runtime_int(p.get("block"),512,f"{node.get('name','Layer Block')} block size",minimum=1),
                backend=backend, precision=precision,
                compass=runtime_int(p.get("compass"),16,f"{node.get('name','Layer Block')} compass",minimum=1),
                activation=str(p.get("activation") or "gelu"),
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','Layer Block')} dropout",minimum=0.0,maximum=1.0),
                norm_eps=runtime_float(p.get("norm_eps"),1e-5,f"{node.get('name','Layer Block')} norm epsilon",minimum=0.0),
                device=device,
            )
        if t=="custom":
            did=node.get("definition_id"); definition=deepcopy(self.custom_components.get(did) or {})
            if not definition: raise ModelCompileError(f"Custom component definition not found for {node.get('name')}.")
            if did in self._custom_stack:
                chain=" -> ".join([*self._custom_stack,did])
                raise ModelCompileError(f"Circular custom component dependency detected: {chain}")
            if str(definition.get("implementation") or "graph") == "api":
                return _APIBoundComponent(
                    definition=definition, params=p, runtime=self.runtime,
                    custom_components=self.custom_components,
                    _custom_stack=(*self._custom_stack,did),
                )
            by_id={n["id"]:n for n in definition.get("nodes") or []}
            for exposed in definition.get("exposed_api") or []:
                key=exposed.get("key"); sid=exposed.get("source_node")
                if key in p and sid in by_id: by_id[sid].setdefault("params",{})[key]=p[key]
            return TensorGraph(
                nodes=list(by_id.values()),edges=definition.get("edges") or [],custom_components=self.custom_components,
                runtime=self.runtime,vocab_override=self.vocab_override,_custom_stack=(*self._custom_stack,did),
            )
        # Not silently faking execution for unsupported advanced blocks.
        raise ModelCompileError(
            f"Training compiler does not yet support component {node.get('name')!r} ({t}). "
            "Supported today: declarative MLBricks API components (including BOLT and ResController), API Function, Text/Image/Audio Input/Output, "
            "Embedding, Learned/Sinusoidal Position, ESA, StateAware ESA Stack, SOUP, RMSNorm, LayerNorm, Linear, FFN, Residual, "
            "Dropout, Previous Value Buffer, LM Head, nested custom components, and API-bound custom components. "
            "ElasticBit is a post-training/inference runtime component, not a differentiable training layer."
        )

    def _nearest_upstream_embedding(self, node_id, expected_shape):
        def upstream_ids(nid):
            ids=[]
            ids.extend(self.in_main.get(nid,[]))
            ids.extend(self.in_skip.get(nid,[]))
            ids.extend(self.in_extra.get(nid,[]))
            ids.extend(
                edge.get("source")
                for edge in self.in_named.get(nid,[])
                if edge.get("source") in self.by_id
            )
            # Preserve graph order while avoiding duplicate traversal caused by
            # one source feeding multiple named API arguments.
            return list(dict.fromkeys(ids))

        queue=upstream_ids(node_id)
        seen=set()
        while queue:
            current=queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node=self.by_id.get(current) or {}
            if node.get("type")=="embedding" and current in self.mods:
                weight=getattr(self.mods[current],"weight",None)
                if weight is not None and tuple(weight.shape)==tuple(expected_shape):
                    return self.mods[current]
            queue.extend(upstream_ids(current))
        return None

    def _apply_weight_tying(self):
        """Resolve Builder LM-head tying to the nearest compatible embedding.

        MLBricks exposes ``LMHead(..., tie_to=<Embedding>)``. A visual graph
        cannot serialize a live module reference, so Builder stores the
        equivalent ``tie_embeddings`` flag and resolves the module here.
        """
        for node in self.nodes:
            if node.get("type")!="lm_head" or node.get("id") not in self.mods:
                continue
            p=node.get("params") or {}
            explicit=p.get("tie_embeddings")
            if explicit is None:
                tie_to=p.get("tie_to")
                enabled=tie_to not in {None,"","none","None","null"}
            else:
                enabled=_bool(explicit)
            if not enabled:
                continue
            head=self.mods[node["id"]]
            expected=(getattr(head,"vocab_size",head.out_features),getattr(head,"hidden_size",head.in_features))
            embedding=self._nearest_upstream_embedding(node["id"],expected)
            if embedding is None:
                raise ModelCompileError(
                    f"{node.get('name','LM Head')} has weight tying enabled but no compatible upstream Embedding "
                    f"with weight shape {expected} was found."
                )
            head.tie_weights(embedding)

    def forward(self, graph_input, graph_skip=None, graph_extra=None):
        values={}
        def edge_value(edge, lane):
            source_id=edge.get("source")
            source_port=str(edge.get("source_port") or "")
            if source_port.startswith("named_out:"):
                return _named_output(values[source_id], source_port.replace("named_out:","",1))
            return _lane_output(values[source_id], lane)
        for node in self.order:
            nid=node["id"]; t=node.get("type"); mod=self.mods[nid]
            main_sources=self.in_main[nid]; skip_sources=self.in_skip[nid]; extra_sources=self.in_extra[nid]; named_edges=self.in_named[nid]
            named_inputs={}
            for named_edge in named_edges:
                source_id=named_edge.get("source")
                source_port=str(named_edge.get("source_port") or "main_out")
                if source_port.startswith("named_out:"):
                    source_key=source_port.replace("named_out:","",1)
                elif "skip" in source_port:
                    source_key="skip"
                elif "extra" in source_port:
                    source_key="extra"
                else:
                    source_key="main"
                target_key=str(named_edge.get("target_port") or "").replace("named_in:","",1)
                if source_id not in values:
                    raise ModelCompileError(f"Named input for {node.get('name')} is not available from upstream node {source_id!r}.")
                named_inputs[target_key]=_named_output(values[source_id],source_key)
            if main_sources:
                if len(main_sources)!=1: raise ModelCompileError(f"{node.get('name')} has {len(main_sources)} Main inputs; merge execution is not implemented.")
                x=edge_value(self.in_main_edges[nid][0], "main")
            else: x=graph_input
            stored_value = None
            contract = API_COMPONENTS.get(t)
            if contract is not None:
                declared=set(contract.input_ports)
                contract_inputs={}

                if "main" in declared:
                    contract_inputs["main"] = x
                elif main_sources:
                    raise ModelCompileError(
                        f"{node.get('name')} received a Main input, but its MLBricks API contract does not declare a Main port."
                    )

                if "skip" in declared:
                    if len(skip_sources)>1:
                        raise ModelCompileError(f"{node.get('name')} has {len(skip_sources)} Skip inputs; exactly one is allowed.")
                    skip_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else graph_skip
                    if skip_value is None:
                        api_arg=contract.input_ports.get("skip", "skip")
                        raise ModelCompileError(
                            f"{node.get('name')} requires the Skip input for MLBricks argument {api_arg!r}."
                        )
                    contract_inputs["skip"] = skip_value
                elif skip_sources:
                    raise ModelCompileError(
                        f"{node.get('name')} received a Skip input, but its MLBricks API contract does not declare a Skip port."
                    )

                if "extra" in declared:
                    if len(extra_sources)>1:
                        raise ModelCompileError(f"{node.get('name')} has {len(extra_sources)} Extra inputs; exactly one is allowed.")
                    extra_value=edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else graph_extra
                    if extra_value is None:
                        api_arg=contract.input_ports.get("extra", "extra")
                        raise ModelCompileError(
                            f"{node.get('name')} requires the Extra input for MLBricks argument {api_arg!r}."
                        )
                    contract_inputs["extra"] = extra_value
                elif extra_sources:
                    raise ModelCompileError(
                        f"{node.get('name')} received an Extra input, but its MLBricks API contract does not declare an Extra port."
                    )

                undeclared_named=sorted(set(named_inputs)-declared)
                if undeclared_named:
                    raise ModelCompileError(
                        f"{node.get('name')} received undeclared named input port(s): {', '.join(undeclared_named)}."
                    )
                named_declared = declared - {"main", "skip", "extra"}
                # Resolve every explicitly connected named input first so a
                # declarative initializer can reference another logical input
                # regardless of Python set iteration order.
                for port in named_declared:
                    if port in named_inputs:
                        contract_inputs[port]=named_inputs[port]

                for port in named_declared:
                    if port in contract_inputs:
                        continue
                    default_value, initialized = contract.initialize_input(
                        port, contract_inputs, node, mod
                    )
                    if initialized:
                        contract_inputs[port]=default_value
                        continue

                    api_arg=contract.input_ports.get(port, port)
                    raise ModelCompileError(
                        f"{node.get('name')} requires named input {port!r} for MLBricks argument {api_arg!r}."
                    )

                result = contract.execute(mod, contract_inputs)
                y = result.get("main")
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat):
                    if "main" not in contract_inputs or "main" not in result:
                        raise ModelCompileError(
                            f"{node.get('name')} cannot Repeat because its MLBricks API contract has no Main input/output feedback path."
                        )
                    contract_inputs["main"] = y
                    result = contract.execute(mod, contract_inputs)
                    y = result.get("main")
                stored_value = result if len(result) > 1 or set(result) != {"main"} else y
            elif t=="layer_block":
                signal_value=named_inputs.get("signal", x)
                residual_value=named_inputs.get("residual")
                if residual_value is None:
                    if len(skip_sources)>1:
                        raise ModelCompileError(f"Layer Block {node.get('name')} has multiple Residual inputs.")
                    residual_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else signal_value
                y=mod(signal_value,residual_value)
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat):
                    feedback=_named_output(y,"signal")
                    residual_feedback=_named_output(y,"residual")
                    y=mod(feedback,residual_feedback)
                stored_value=y
            elif t=="residual":
                if len(skip_sources)!=1: raise ModelCompileError(f"Residual {node.get('name')} needs exactly one Skip input.")
                if extra_sources: raise ModelCompileError(f"Residual {node.get('name')} does not accept an Extra input.")
                y=mod(edge_value(self.in_skip_edges[nid][0], "skip"),x)
            elif t=="api_step":
                if len(skip_sources)>1 or len(extra_sources)>1:
                    raise ModelCompileError(f"API function {node.get('name')} accepts at most one Skip and one Extra tensor lane.")
                skip_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else graph_skip
                extra_value=edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else graph_extra
                y=mod(x,skip=skip_value,extra=extra_value,named_inputs=named_inputs)
                repeat=max(1,int(node.get("repeat") or 1))
                if repeat>1 and getattr(mod,"port_mode","standard")=="named":
                    raise ModelCompileError(f"Named-port User Function {node.get('name')} cannot use Repeat > 1; connect another function node instead.")
                for _ in range(1,repeat):
                    repeat_input=_lane_output(y,"main")
                    y=mod(repeat_input,skip=skip_value,extra=extra_value,named_inputs=named_inputs)
            elif t=="custom" and str((self.custom_components.get(node.get("definition_id")) or {}).get("implementation") or "graph") == "api":
                if len(skip_sources)>1 or len(extra_sources)>1:
                    raise ModelCompileError(f"API component {node.get('name')} accepts at most one Skip and one Extra tensor lane.")
                skip_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else None
                extra_value=edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else None
                y=mod(x,skip=skip_value,extra=extra_value)
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat): y=mod(y,skip=skip_value,extra=extra_value)
            else:
                if skip_sources: raise ModelCompileError(f"{node.get('name')} has a Skip input but this component does not consume Skip tensors.")
                if extra_sources: raise ModelCompileError(f"{node.get('name')} has an Extra input but this component does not consume Extra tensors.")
                y=mod(x)
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat): y=mod(y)
            values[nid]=stored_value if stored_value is not None else y
        sinks=[n for n in self.order if not self.outgoing[n["id"]]]
        if not sinks: raise ModelCompileError("Graph has no output node.")
        if len(sinks)>1: raise ModelCompileError("Training compiler currently requires one tensor output.")
        return _lane_output(values[sinks[0]["id"]], "main")

    def recurrent_generation_support(self):
        """Return whether this visual graph has an exact token-step execution path.

        Generation is deliberately capability based.  Sequence mixers must expose
        an explicit prefill/decode contract; pointwise layers are safe to run on
        one token; arbitrary API/user functions are not guessed to be causal.
        """
        pointwise = {
            "text_input", "image_input", "audio_input", "text_output", "logits_output",
            "embedding", "lm_head", "learned_position", "sinusoidal_position",
            "rmsnorm", "layernorm", "linear", "ffn", "residual", "dropout",
            "value_buffer",
        }
        for node in self.order:
            nid=node["id"]
            component_type=str(node.get("type") or "")
            module=self.mods[nid]
            if self.in_named.get(nid) and component_type != "layer_block":
                return False, f"{node.get('name', component_type)} uses named state ports"
            if component_type == "esa":
                if not callable(getattr(module,"prefill",None)) or not callable(getattr(module,"decode_step",None)):
                    return False, f"{node.get('name','ESA')} does not expose prefill/decode_step"
            elif component_type == "bolt":
                required=("prefill_with_cache","project_decode_state","decode_append_projected")
                if not all(callable(getattr(module,name,None)) for name in required):
                    return False, f"{node.get('name','BOLT')} does not expose fixed-cache decoding"
            elif component_type == "soup":
                if not callable(getattr(module,"prefill",None)) or not callable(getattr(module,"decode_step",None)):
                    return False, f"{node.get('name','SOUP')} does not expose recurrent generation"
            elif component_type == "layer_block":
                if not callable(getattr(module,"prefill",None)) or not callable(getattr(module,"decode_step",None)):
                    return False, f"{node.get('name','Layer Block')} does not expose recurrent ESA generation"
            elif component_type == "custom" and isinstance(module,TensorGraph):
                if self.in_skip.get(nid) or self.in_extra.get(nid):
                    return False, f"{node.get('name','Module')} uses external Skip/Extra generation inputs"
                supported,reason=module.recurrent_generation_support()
                if not supported:
                    return False, f"{node.get('name','Module')}: {reason}"
            elif component_type in pointwise:
                pass
            elif component_type == "rescontroller":
                # ResController is a pointwise residual/update merge.  BOLT is
                # the only other declarative API component with a recurrent path.
                pass
            else:
                return False, f"{node.get('name', component_type)} has no verified token-step contract"
        return True, None

    def recurrent_generation_algorithms(self):
        algorithms=[]
        for node in self.order:
            component_type=str(node.get("type") or "")
            module=self.mods[node["id"]]
            if component_type == "esa": algorithms.append("ESA Thunder prefill + Lightning decode")
            elif component_type == "bolt": algorithms.append("BOLT fixed C/rho cache decode")
            elif component_type == "soup": algorithms.append("SOUP recurrent generation")
            elif component_type == "layer_block": algorithms.append("ESA Layer Block recurrent generation")
            elif component_type == "learned_position": algorithms.append("Cyclic learned-position continuation")
            elif component_type == "custom" and isinstance(module,TensorGraph):
                algorithms.extend(module.recurrent_generation_algorithms())
        return list(dict.fromkeys(algorithms))

    def _terminal_generation_head(self,nid):
        """True when every consumer after an LM head is only an output identity."""
        pending=list(self.outgoing.get(nid) or [])
        seen=set()
        while pending:
            current=pending.pop()
            if current in seen: continue
            seen.add(current)
            node=self.by_id[current]
            if node.get("type") not in {"text_output","logits_output"}: return False
            pending.extend(self.outgoing.get(current) or [])
        return True

    @staticmethod
    def _bolt_prefill(module,x,capacity,start_pos=0):
        y,(prefix_c,prefix_rho)=module.prefill_with_cache(x,start_pos=int(start_pos))
        prefix=int(prefix_c.size(2))
        capacity=max(prefix,int(capacity))
        c=prefix_c.new_empty(prefix_c.size(0),prefix_c.size(1),capacity,prefix_c.size(3))
        rho=prefix_rho.new_empty(prefix_rho.size(0),prefix_rho.size(1),capacity)
        c[:,:,:prefix,:].copy_(prefix_c)
        rho[:,:,:prefix].copy_(prefix_rho)
        return y,{"c":c,"rho":rho,"length":prefix}

    @staticmethod
    def _bolt_decode(module,x,state,position):
        q,c_now,rho_now=module.project_decode_state(x,start_pos=int(position))
        y=module.decode_append_projected(
            q,c_now,rho_now,state["c"],state["rho"],position=int(position)
        )[:,None,:]
        state["length"]=max(int(state.get("length",0)),int(position)+1)
        return y,state

    def _generation_transform(self,node,module,x,run):
        phase=run["phase"]
        nid=node["id"]
        component_type=str(node.get("type") or "")
        repeat=max(1,int(node.get("repeat") or 1))
        states=[] if phase=="prefill" else list(run["states"].get(nid) or [])
        output=x
        next_states=[]
        for index in range(repeat):
            state=states[index] if index<len(states) else None
            if component_type == "esa":
                if phase=="prefill": output,state=module.prefill(output)
                else: output,state=module.decode_step(output,state)
                next_states.append(state)
            elif component_type == "bolt":
                if phase=="prefill":
                    output,state=self._bolt_prefill(module,output,run["capacity"],run["position"])
                else:
                    output,state=self._bolt_decode(module,output,state,run["position"])
                next_states.append(state)
            elif component_type == "soup":
                if phase=="prefill":
                    prepare=getattr(module,"prepare_generation",None)
                    if callable(prepare): prepare(fast=True)
                    output,state=module.prefill(output)
                else: output,state=module.decode_step(output,state)
                next_states.append(state)
            elif component_type == "custom" and isinstance(module,TensorGraph):
                if phase=="prefill":
                    output,state=module.prefill(output,capacity=run["capacity"])
                else: output,state=module.decode_step(output,state,position=run["position"])
                next_states.append(state)
            elif component_type in {"learned_position","sinusoidal_position"}:
                start_pos=int(run["position"])
                if component_type=="learned_position":
                    # Recurrent state mixers can decode beyond the training
                    # context without replaying it. Absolute learned-position
                    # tables are finite, so cycle their index rather than doing
                    # a full 512-token rebuild for every overflow token.
                    params=node.get("params") or {}
                    max_positions=int(
                        params.get("max_seq_len") or params.get("max_length")
                        or params.get("num_embeddings") or 0
                    )
                    if max_positions>0: start_pos%=max_positions
                output=module(output,start_pos=start_pos)
            elif component_type == "lm_head" and self._terminal_generation_head(nid):
                # The full-sequence hidden states are useful to earlier layers,
                # but generation only needs vocabulary scores for the last one.
                output=module(output[:,-1:,:])
            else:
                output=module(output)
        if next_states: run["next_states"][nid]=next_states
        return output

    def _generation_execute(self,graph_input,run,graph_skip=None,graph_extra=None):
        values={}
        def edge_value(edge,lane):
            source_id=edge.get("source")
            source_port=str(edge.get("source_port") or "")
            if source_port.startswith("named_out:"):
                return _named_output(values[source_id], source_port.replace("named_out:","",1))
            return _lane_output(values[source_id],lane)
        for node in self.order:
            nid=node["id"]; component_type=node.get("type"); module=self.mods[nid]
            main_sources=self.in_main[nid]; skip_sources=self.in_skip[nid]; extra_sources=self.in_extra[nid]
            named_inputs={}
            for named_edge in self.in_named[nid]:
                source_id=named_edge.get("source")
                source_port=str(named_edge.get("source_port") or "main_out")
                if source_port.startswith("named_out:"):
                    source_key=source_port.replace("named_out:","",1)
                elif "skip" in source_port:
                    source_key="skip"
                elif "extra" in source_port:
                    source_key="extra"
                else:
                    source_key="main"
                target_key=str(named_edge.get("target_port") or "").replace("named_in:","",1)
                named_inputs[target_key]=_named_output(values[source_id],source_key)
            if main_sources:
                if len(main_sources)!=1: raise ModelCompileError(f"{node.get('name')} has {len(main_sources)} Main inputs; merge execution is not implemented.")
                x=edge_value(self.in_main_edges[nid][0],"main")
            else: x=graph_input

            contract=API_COMPONENTS.get(component_type)
            if component_type=="bolt":
                if skip_sources or extra_sources: raise ModelCompileError("BOLT recurrent generation accepts only its Main input.")
                y=self._generation_transform(node,module,x,run)
            elif component_type=="layer_block":
                signal_value=named_inputs.get("signal",x)
                residual_value=named_inputs.get("residual")
                if residual_value is None:
                    residual_value=edge_value(self.in_skip_edges[nid][0],"skip") if skip_sources else signal_value
                states=[] if run["phase"]=="prefill" else list(run["states"].get(nid) or [])
                state=states[0] if states else None
                if run["phase"]=="prefill":
                    y,state=module.prefill(signal_value,residual_value)
                else:
                    y,state=module.decode_step(signal_value,state,residual_value)
                run["next_states"][nid]=[state]
            elif contract is not None:
                declared=set(contract.input_ports); inputs={}
                if "main" in declared: inputs["main"]=x
                if "skip" in declared:
                    if len(skip_sources)>1: raise ModelCompileError(f"{node.get('name')} has multiple Skip inputs.")
                    value=edge_value(self.in_skip_edges[nid][0],"skip") if skip_sources else graph_skip
                    if value is None: raise ModelCompileError(f"{node.get('name')} requires its Skip input.")
                    inputs["skip"]=value
                if "extra" in declared:
                    if len(extra_sources)>1: raise ModelCompileError(f"{node.get('name')} has multiple Extra inputs.")
                    value=edge_value(self.in_extra_edges[nid][0],"extra") if extra_sources else graph_extra
                    if value is None: raise ModelCompileError(f"{node.get('name')} requires its Extra input.")
                    inputs["extra"]=value
                result=contract.execute(module,inputs)
                y=result.get("main")
                for _ in range(1,max(1,int(node.get("repeat") or 1))):
                    inputs["main"]=y; result=contract.execute(module,inputs); y=result.get("main")
            elif component_type=="residual":
                if len(skip_sources)!=1: raise ModelCompileError(f"Residual {node.get('name')} needs exactly one Skip input.")
                y=module(edge_value(self.in_skip_edges[nid][0],"skip"),x)
            elif component_type=="custom" and isinstance(module,TensorGraph):
                y=self._generation_transform(node,module,x,run)
            else:
                if skip_sources or extra_sources:
                    raise ModelCompileError(f"{node.get('name')} has unsupported auxiliary inputs during recurrent generation.")
                y=self._generation_transform(node,module,x,run)
            values[nid]=y
        sinks=[n for n in self.order if not self.outgoing[n["id"]]]
        if len(sinks)!=1: raise ModelCompileError("Recurrent generation requires exactly one tensor output.")
        return _lane_output(values[sinks[0]["id"]],"main")

    @torch.no_grad()
    def prefill(self,graph_input,*,capacity=None,graph_skip=None,graph_extra=None):
        supported,reason=self.recurrent_generation_support()
        if not supported: raise ModelCompileError(reason or "Graph has no recurrent generation path.")
        if graph_input.ndim<2 or graph_input.size(1)<1: raise ValueError("prefill expects a non-empty [B,T,...] input")
        run={"phase":"prefill","states":{},"next_states":{},"position":0,
             "capacity":max(int(capacity or graph_input.size(1)),int(graph_input.size(1)))}
        output=self._generation_execute(graph_input,run,graph_skip,graph_extra)
        cache={"states":run["next_states"],"position":int(graph_input.size(1)),"capacity":run["capacity"]}
        return output,cache

    @torch.no_grad()
    def decode_step(self,graph_input,cache,*,position=None,graph_skip=None,graph_extra=None):
        if graph_input.ndim<2 or graph_input.size(1)!=1: raise ValueError("decode_step expects one token per batch")
        position=int(cache.get("position",0) if position is None else position)
        if position>=int(cache.get("capacity",position+1)): raise ValueError("generation cache capacity exceeded")
        run={"phase":"decode","states":cache.get("states") or {},"next_states":{},
             "position":position,"capacity":int(cache.get("capacity",position+1))}
        output=self._generation_execute(graph_input,run,graph_skip,graph_extra)
        cache={"states":run["next_states"],"position":position+1,"capacity":run["capacity"]}
        return output,cache



@dataclass
class CompiledModel:
    # ``model`` is the inference/evaluation model. ``training_model`` is a
    # loss-wrapped whole-model graph used only for training when requested.
    model: nn.Module
    raw_model: nn.Module
    training_model: nn.Module | None
    device: torch.device
    precision: str
    vocab_size: int
    parameter_count: int
    compile_used: bool
    compile_error: str | None


class _CausalLMTrainingGraph(nn.Module):
    """Whole language-model training graph including cross entropy.

    Keeping the loss inside this wrapper lets ``torch.compile`` capture the
    model + LM head + loss as one graph, matching the benchmark notebook rather
    than compiling only the logits-producing portion.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )


def _root_model(state):
    ws=(state.get("workspaces") or {}).get("model") or {}
    root_id=ws.get("root_component_id") or state.get("root_component_id")
    comp=(state.get("components") or {}).get(root_id)
    if not comp: raise ModelCompileError("Model Builder graph was not found.")
    return comp


def _graph_vocab(model_graph):
    sizes=[]
    for n in model_graph.get("nodes") or []:
        p=n.get("params") or {}
        if n.get("type") in {"embedding","lm_head"} and p.get("vocab_size"):
            sizes.append(int(p["vocab_size"]))
    return max(sizes) if sizes else 0


def _tokenizer_for(meta, *, local_only_first=True, tokenizer_path=None):
    tok_cfg=((meta or {}).get("pipeline") or {}).get("tokenizer") or {}
    name=tok_cfg.get("tokenizer_name") or "gpt2"
    preferred=None
    if tokenizer_path:
        try:
            candidate=Path(str(tokenizer_path)).expanduser()
            if candidate.exists(): preferred=str(candidate.resolve())
        except Exception:
            preferred=None
    cache_key=preferred or str(name)
    cached=_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Training/generation needs transformers. Install transformers in the notebook.") from exc
    errors=[]
    tok=None
    if preferred:
        try:
            tok=AutoTokenizer.from_pretrained(preferred,local_files_only=True)
        except Exception as exc:
            errors.append(exc)
    if tok is None and local_only_first:
        try: tok=AutoTokenizer.from_pretrained(name,local_files_only=True)
        except Exception as exc: errors.append(exc); tok=None
    if tok is None:
        try: tok=AutoTokenizer.from_pretrained(name)
        except Exception as exc:
            detail=str(errors[-1]) if errors else ""
            raise RuntimeError(f"Tokenizer {name!r} is unavailable. {detail} {exc}") from exc
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None: tok.pad_token=tok.eos_token
        elif tok.unk_token_id is not None: tok.pad_token=tok.unk_token
        else: tok.add_special_tokens({"pad_token":"<|pad|>"})
    _TOKENIZER_CACHE[cache_key]=tok
    # Also cache by canonical tokenizer name when a saved tokenizer directory was
    # used. A later runtime with the same tokenizer can reuse it immediately.
    _TOKENIZER_CACHE.setdefault(str(name),tok)
    return tok


def compile_builder_model(state, model_entry, dataset_meta, runtime, *, progress=None, for_training=False):
    """Build a Builder model against the current public MLBricks runtime.

    Training compilation wraps the *entire* causal-LM forward including loss
    and uses one ``torch.compile`` call with ``fullgraph=True`` and
    ``dynamic=False`` so the requested benchmark flags reach Dynamo unchanged. Inference/generation compilation remains shape-friendly
    and compiles only the model forward.
    """
    graph=copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
    custom_components=copy.deepcopy(state.get("custom_components") or {})
    custom_components.update(
        copy.deepcopy((model_entry or {}).get("custom_components_snapshot") or {})
    )
    device=resolve_device(runtime.get("device","auto"))
    precision,dtype=resolve_precision(runtime.get("precision","fp16"),device)
    if progress:
        progress({
            "status":"running","runtime_kind":"train" if for_training else "generate",
            "phase":"tokenizer","overall":0,
            "message":f"Loading tokenizer for {device}…",
        })
    tokenizer=_tokenizer_for(dataset_meta, tokenizer_path=(model_entry or {}).get("tokenizer_path"))
    graph_vocab=_graph_vocab(graph)
    tokenizer_vocab=len(tokenizer)
    effective_vocab=max(graph_vocab,tokenizer_vocab)
    if progress:
        msg=f"Building model on {device}"
        if effective_vocab!=graph_vocab: msg+=f" · vocab {graph_vocab:,} → {effective_vocab:,} to match tokenizer"
        progress({"status":"running","runtime_kind":"train" if for_training else "generate","phase":"compile","overall":1,"message":msg})

    # Preflight only the MLBricks APIs actually referenced by this graph.
    # This produces a single clear import error before model construction and
    # warms the same cache later used by each component constructor.
    import_checks = IMPORT_POOL.ensure_graph(
        graph.get("nodes") or [], custom_components
    )
    import_failures = [
        status for status in import_checks.values() if not status.get("ok")
    ]
    if import_failures:
        details = "; ".join(
            f"{item.get('component_type')}: {item.get('error')}"
            for item in import_failures
        )
        raise RuntimeError(f"MLBricks component import preflight failed: {details}")

    try:
        model_settings = copy.deepcopy((model_entry or {}).get("model_settings") or {})
        runtime_context = {
            **runtime,
            "device": str(device),
            "precision": precision,
            "model_dim": model_settings.get("embedding_size") or model_settings.get("model_dim"),
            "heads": model_settings.get("heads"),
            "context_length": (model_entry or {}).get("context_length") or runtime.get("context_length"),
            "batch_size": runtime.get("batch_size") or (model_entry or {}).get("batch_size"),
        }
        raw=TensorGraph(
            nodes=graph.get("nodes") or [],
            edges=graph.get("edges") or [],
            custom_components=custom_components,
            runtime=runtime_context,
            vocab_override=effective_vocab,
        )
    except RuntimeError as exc:
        if str(runtime.get("backend") or "pytorch").lower()=="native" and "native extension is unavailable" in str(exc).lower():
            raise RuntimeError(
                "Backend 'native' is selected, but the optional MLBricks native extension is unavailable. "
                "Open Training Setup and set Backend to 'pytorch', or install/build the native MLBricks extension."
            ) from exc
        raise
    # AMP training needs FP32 master parameters; autocast supplies reduced-
    # precision activations and GradScaler safely scales their gradients. Moving
    # trainable parameters themselves to FP16 makes GradScaler fail during
    # unscale_. Inference has no optimizer/scaler, so it can use the requested
    # reduced parameter dtype directly.
    parameter_dtype = (
        torch.float32
        if for_training and precision in {"fp16", "bf16"}
        else dtype
    )
    raw.to(device=device,dtype=parameter_dtype)
    params=sum(p.numel() for p in raw.parameters())
    inference_model=raw
    training_model=_CausalLMTrainingGraph(raw) if for_training else None
    compile_used=False
    compile_error=None

    if str(runtime.get("execution_mode","eager"))=="compiled":
        if not hasattr(torch,"compile"):
            raise RuntimeError("Compiled execution was selected, but torch.compile is unavailable in this PyTorch build.")
        mode=str(runtime.get("compile_mode") or "default")
        # TensorGraph and the training wrapper are ordinary nn.Modules. Use one
        # explicit torch.compile wrapper here so the requested mode/fullgraph/
        # dynamic flags are guaranteed to reach Dynamo exactly as configured.
        # (MLBricks' package-level compile API remains useful for models with
        # their own compile hook; Builder's visual TensorGraph has no such hook.)
        # No eager fallback: a compile failure is surfaced to the user.
        if for_training:
            training_model=torch.compile(
                training_model, mode=mode, dynamic=False, fullgraph=True
            )
        else:
            inference_model=torch.compile(
                raw, mode=mode, dynamic=None, fullgraph=False
            )
        compile_used=True

    return CompiledModel(
        inference_model,raw,training_model,device,precision,effective_vocab,
        params,compile_used,compile_error,
    ),tokenizer


class _PackedLMBatcher:
    """Produce exact ``[batch, context]`` causal-LM batches without padding.

    Tokenized examples are concatenated into a stream with an EOS separator and
    sliced into ``context+1`` blocks. Eager and compiled execution therefore see
    identical fixed shapes and every reported token corresponds to real compute.
    """
    def __init__(self, dataset, *, context, separator_id, rng):
        if len(dataset)<=0:
            raise RuntimeError("Selected split has no rows.")
        self.dataset=dataset
        self.context=int(context)
        self.separator_id=int(separator_id)
        self.rng=rng
        self.buffer=[]
        self.offset=0

    def _append_row(self):
        attempts=0
        while attempts<100:
            attempts+=1
            row=self.dataset[self.rng.randrange(len(self.dataset))]
            ids=row.get("input_ids") if isinstance(row,dict) else None
            if ids is None:
                raise RuntimeError("Prepared data has no input_ids. Add Tokenize Text to the Data Processing pipeline.")
            ids=[int(v) for v in list(ids)]
            # Prepared tokenizer max length is not the model training context.
            # If the dataset was padded during tokenization, remove padding via
            # attention_mask before appending the row to the repackable stream.
            mask=row.get("attention_mask") if isinstance(row,dict) else None
            if mask is not None:
                mask=list(mask)
                if len(mask)==len(ids):
                    ids=[token_id for token_id,keep in zip(ids,mask) if int(keep)!=0]
            if not ids:
                continue
            self.buffer.extend(ids)
            if ids[-1]!=self.separator_id:
                self.buffer.append(self.separator_id)
            return
        raise RuntimeError("Could not read a non-empty tokenized row from the selected split.")

    def batch(self,batch_size,device):
        batch_size=int(batch_size)
        block=self.context+1
        needed=batch_size*block
        while len(self.buffer)-self.offset<needed:
            self._append_row()
        flat=self.buffer[self.offset:self.offset+needed]
        self.offset+=needed
        if self.offset>262144:
            self.buffer=self.buffer[self.offset:]
            self.offset=0
        packed=torch.tensor(flat,dtype=torch.long).view(batch_size,block)
        x=packed[:,:-1].contiguous()
        y=packed[:,1:].contiguous()
        if device.type=="cuda":
            try:
                x=x.pin_memory(); y=y.pin_memory()
            except RuntimeError:
                pass
        return x.to(device,non_blocking=True),y.to(device,non_blocking=True),batch_size*self.context


def _sample_batch(dataset,batch_size,context,pad_id,device,rng,*,fixed_length=True):
    # Backward-compatible helper used by callers outside the training loop.
    # New LM training always uses packed fixed-shape data for both eager and
    # compiled modes; ``pad_id`` is used as the stream separator when EOS is not
    # otherwise available.
    del fixed_length
    return _PackedLMBatcher(
        dataset,context=context,separator_id=pad_id,rng=rng
    ).batch(batch_size,device)


def _autocast_context(device,precision):
    if device.type=="cuda" and precision in {"fp16","bf16"}:
        return torch.autocast(device_type="cuda",dtype=torch.float16 if precision=="fp16" else torch.bfloat16)
    if device.type=="cpu" and precision=="bf16": return torch.autocast(device_type="cpu",dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def _perplexity(loss_value):
    if loss_value is None:
        return None
    try:
        value=float(loss_value)
    except (TypeError,ValueError,OverflowError):
        return None
    if not math.isfinite(value):
        return None
    # exp(20) is already ~4.85e8; cap only to keep telemetry finite.
    return math.exp(min(value,20.0))


def _sync_device(device):
    if device.type=="cuda":
        try: torch.cuda.synchronize(device)
        except Exception: pass


def _memory_snapshot(device):
    empty={
        "memory_allocated_gb":None,"memory_reserved_gb":None,
        "memory_peak_gb":None,"memory_total_gb":None,
    }
    if device.type!="cuda":
        return empty
    try:
        scale=float(1024**3)
        props=torch.cuda.get_device_properties(device)
        return {
            "memory_allocated_gb":torch.cuda.memory_allocated(device)/scale,
            "memory_reserved_gb":torch.cuda.memory_reserved(device)/scale,
            "memory_peak_gb":torch.cuda.max_memory_allocated(device)/scale,
            "memory_total_gb":float(props.total_memory)/scale,
        }
    except Exception:
        return empty


def _optimizer(model,config):
    name=str(config.get("optimizer") or "adamw").lower()
    lr=runtime_float(config.get("learning_rate"),5e-4,"Learning Rate",minimum=0.0)
    wd=runtime_float(config.get("weight_decay"),0.1,"Weight Decay",minimum=0.0)
    beta1=runtime_float(config.get("beta1"),0.9,"Adam Beta 1",minimum=0.0,maximum=1.0)
    beta2=runtime_float(config.get("beta2"),0.95,"Adam Beta 2",minimum=0.0,maximum=1.0)
    if name in {"adamw","adam"}:
        try:
            cls = IMPORT_POOL.resolve_api("optim.AdamW" if name == "adamw" else "optim.Adam")
        except ImportError as exc:
            raise RuntimeError("Current MLBricks installation does not expose Adam/AdamW through the import pool.") from exc
        return cls(model.parameters(),lr=lr,betas=(beta1,beta2),weight_decay=wd)
    if name=="sgd":
        return torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd,momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def _evaluate(
    loss_model, raw_model, batcher, *, steps, batch_size, device, precision,
    progress_callback=None, stop_event=None,
):
    """Evaluate a bounded number of validation batches with live progress.

    Validation used to be a silent block between the final training step and the
    terminal event. On short smoke/retrain runs that made Studio sit at 99% while
    20 validation batches (and then sample generation) were still executing. The
    work was finite, but the UI looked hung. Keep evaluation bounded and surface
    each completed batch without loading any additional dataset state.
    """
    if batcher is None:
        return None
    total=max(1,int(steps))
    raw_model.eval();loss_model.eval();losses=[]
    try:
        with torch.no_grad():
            for index in range(total):
                if stop_event is not None and stop_event.is_set():
                    raise TrainingStopped("Training stopped.")
                x,y,_=batcher.batch(batch_size,device)
                _sync_device(device)
                with _autocast_context(device,precision):
                    loss=loss_model(x,y)
                losses.append(float(loss.detach().float().cpu()))
                if progress_callback is not None:
                    try:
                        progress_callback(index+1,total,sum(losses)/len(losses))
                    except TrainingStopped:
                        raise
                    except Exception:
                        # Telemetry must never make validation fail.
                        pass
    finally:
        raw_model.train();loss_model.train()
    return sum(losses)/len(losses)


def _sample_next(logits,temperature,top_k,top_p,generator=None):
    temperature=max(runtime_float(temperature,0.8,"Temperature",minimum=1e-5),1e-5)
    logits=logits/temperature
    top_k=runtime_int(top_k,50,"Top K",minimum=0)
    candidate_ids=None
    if 0<top_k<logits.size(-1):
        # Restrict first, then run nucleus sampling inside this small candidate
        # set.  The old path masked to top-k but still sorted/scattered the whole
        # vocabulary every token.
        logits,candidate_ids=torch.topk(logits,top_k,dim=-1,sorted=True)
    top_p=runtime_float(top_p,0.95,"Top P",minimum=0.0,maximum=1.0)
    if 0<top_p<1:
        if candidate_ids is None:
            logits,sorted_ids=torch.sort(logits,descending=True)
            candidate_ids=sorted_ids
        probs=torch.softmax(logits,dim=-1)
        cum=torch.cumsum(probs,dim=-1)
        mask=cum>float(top_p)
        mask[...,1:]=mask[...,:-1].clone();mask[...,0]=False
        logits=logits.masked_fill(mask,float('-inf'))
    selected=torch.multinomial(torch.softmax(logits,dim=-1),1,generator=generator)
    return selected if candidate_ids is None else candidate_ids.gather(-1,selected)


def generate_text(model,tokenizer,prompt,*,max_new_tokens,context,device,precision,temperature=.8,top_k=50,top_p=.95,seed=42,progress=None,stop_event=None,stream_every_token=False):
    ids=tokenizer.encode(str(prompt),add_special_tokens=True)
    if not ids: ids=[tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
    generated=list(ids)
    was_training=bool(model.training)
    model.eval()
    try:
        generator_device=device if device.type in {"cpu","cuda"} else torch.device("cpu")
        seed=runtime_int(seed,42,"Seed")
        max_new_tokens=runtime_int(max_new_tokens,128,"New Token Count",minimum=1)
        context=runtime_int(context,512,"Model Context",minimum=2)
        gen=torch.Generator(device=generator_device);gen.manual_seed(seed)

        # torch.compile wrappers expose the original TensorGraph through
        # ``_orig_mod``.  Recurrent methods live on that graph, while ordinary
        # unsupported models keep using the possibly-compiled full forward.
        recurrent_model=model
        while isinstance(getattr(recurrent_model,"_orig_mod",None),nn.Module):
            recurrent_model=recurrent_model._orig_mod
        support=getattr(recurrent_model,"recurrent_generation_support",None)
        supported=False; fallback_reason=None; algorithms=[]
        if callable(support):
            supported,fallback_reason=support()
            if supported:
                report=getattr(recurrent_model,"recurrent_generation_algorithms",None)
                algorithms=report() if callable(report) else []
        mode="recurrent-cache" if supported else "full-context"
        mode_label=("cached prefill/decode" if supported else "full-context compatibility")
        started=time.perf_counter()
        last_stream_at=0.0
        # Notebook widget transports cannot consume hundreds of growing JSON
        # payloads per second. Coalesce only the UI transport; token generation
        # itself remains unthrottled and the browser still updates at 12.5 FPS.
        stream_interval_seconds=0.08
        active_ids=list(ids[-context:])

        with torch.inference_mode(),_autocast_context(device,precision):
            if supported:
                x=torch.tensor([active_ids],dtype=torch.long,device=device)
                recurrent_capacity=max(context,len(active_ids)+max_new_tokens)
                logits,cache=recurrent_model.prefill(x,capacity=recurrent_capacity)
            else:
                cache=None
                x=torch.tensor([active_ids],dtype=torch.long,device=device)
                logits=model(x)
            if isinstance(logits,(tuple,list)): logits=logits[0]

            if progress:
                progress({
                    "status":"running","runtime_kind":"generate","phase":"prefill","overall":0,
                    "generated_tokens":0,"prefill_tokens":len(active_ids),"generation_mode":mode,
                    "generation_algorithms":algorithms,"fallback_reason":fallback_reason,
                    "message":f"Prompt ready · {mode_label}",
                    "generated_text":tokenizer.decode(generated,skip_special_tokens=True),
                })

            for i in range(max_new_tokens):
                if stop_event is not None and stop_event.is_set(): raise TrainingStopped("Generation stopped.")
                next_token=_sample_next(logits[:,-1,:].float(),temperature,top_k,top_p,generator=gen)
                next_id=int(next_token[0,0].item())
                generated.append(next_id)
                active_ids.append(next_id)
                now=time.perf_counter()
                elapsed=max(now-started,1e-9)
                terminal=(
                    (tokenizer.eos_token_id is not None and next_id==tokenizer.eos_token_id)
                    or i+1==max_new_tokens
                )
                if progress and not terminal and (stream_every_token or i==0 or now-last_stream_at>=stream_interval_seconds):
                    last_stream_at=now
                    progress({
                        "status":"running","runtime_kind":"generate","phase":"generate",
                        "overall":round((i+1)/max_new_tokens*100),"generated_tokens":i+1,
                        "tokens_per_sec":float(i+1)/elapsed,"generation_mode":mode,
                        "generation_algorithms":algorithms,"fallback_reason":fallback_reason,
                        "message":f"Generated {i+1}/{max_new_tokens} tokens · {mode_label}",
                        "generated_text":tokenizer.decode(generated,skip_special_tokens=True),
                    })
                if terminal: break

                if supported:
                    logits,cache=recurrent_model.decode_step(
                        next_token,cache,position=len(active_ids)-1
                    )
                else:
                    active_ids=active_ids[-context:]
                    x=torch.tensor([active_ids],dtype=torch.long,device=device)
                    logits=model(x)
                if isinstance(logits,(tuple,list)): logits=logits[0]
        return tokenizer.decode(generated,skip_special_tokens=True),len(generated)-len(ids)
    finally:
        if was_training: model.train()


def train_builder_model(*,state,model_entry,dataset,dataset_meta,config,progress,stop_event,resume_from=None,resume_mode=None):
    """Train a Builder causal LM with notebook-equivalent throughput semantics.

    Eager and compiled runs use the same packed fixed-shape token batches. When
    compiled execution is selected, one whole training graph (model + LM head +
    cross entropy) is compiled with ``fullgraph=True`` and ``dynamic=False``.
    """
    seed=runtime_int(config.get("seed"),42,"Seed")
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

    compiled,tokenizer=compile_builder_model(
        state,model_entry,dataset_meta,config,progress=progress,for_training=True
    )
    device=compiled.device
    model=compiled.model
    raw=compiled.raw_model
    loss_model=(compiled.training_model if compiled.training_model is not None else _CausalLMTrainingGraph(raw))
    precision=compiled.precision

    resume_path=Path(str(resume_from)).expanduser() if resume_from else None
    resume_kind=str(resume_mode or "weights").lower() if resume_path is not None else None
    resume_payload=None
    resume_step=0
    resume_tokens=0
    if resume_path is not None:
        progress({
            "status":"running","runtime_kind":"train","phase":"resume_weights","overall":0,
            "step":0,"tokens_seen":0,
            "message":f"Loading existing trained parameters from {resume_path.name or resume_path}…",
            "resume_from":str(resume_path),"resume_mode":resume_kind,
        })
        try:
            if resume_path.is_dir() and (resume_path/"model.pt").exists():
                mlbricks_load=IMPORT_POOL.resolve_api("lifecycle.load")
                loaded=mlbricks_load(resume_path,device=device,strict=True)
                if not isinstance(loaded,nn.Module):
                    raise RuntimeError(f"loaded artifact is not a torch.nn.Module: {type(loaded)!r}")
                raw.load_state_dict(loaded.state_dict(),strict=True)
                del loaded
                if resume_kind=="checkpoint" and (resume_path/"training_state.pt").is_file():
                    resume_payload=safe_torch_load(resume_path/"training_state.pt",map_location="cpu",allow_unsafe_pickle=False)
            elif resume_path.is_file():
                payload=safe_torch_load(resume_path,map_location="cpu",allow_unsafe_pickle=_bool(config.get("allow_unsafe_legacy_checkpoint",False)))
                if not isinstance(payload,dict) or "model_state" not in payload:
                    raise RuntimeError("legacy checkpoint does not contain model_state")
                raw.load_state_dict(payload["model_state"],strict=True)
                resume_payload=payload if resume_kind=="checkpoint" else None
            else:
                raise FileNotFoundError(f"model artifact was not found: {resume_path}")
        except Exception as exc:
            raise ExistingModelArtifactError(
                resume_path,
                f"Existing model parameters at {resume_path} could not be loaded safely: {type(exc).__name__}: {exc}",
            ) from exc

    train=dataset["train"] if isinstance(dataset,dict) or hasattr(dataset,"keys") else dataset
    val_name=str(config.get("validation_split") or "validation")
    val=dataset.get(val_name) if hasattr(dataset,"get") else None

    context=runtime_int(
        model_entry.get("context_length") or state.get("project",{}).get("context_length"),
        512,"Model Context",minimum=2,
    )
    batch=runtime_int(config.get("batch_size"),16,"Batch Size",minimum=1)
    accum=runtime_int(config.get("gradient_accumulation"),1,"Gradient Accumulation",minimum=1)
    pad=runtime_int(tokenizer.pad_token_id,0,"Tokenizer Pad Token ID",minimum=0)
    separator=int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad)
    opt=_optimizer(raw,config)
    warm=runtime_int(config.get("warmup_steps"),0,"Warmup Steps",minimum=0)
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
    if resume_payload and resume_kind=="checkpoint":
        resume_step=max(0,int(resume_payload.get("step") or 0))
        resume_tokens=max(0,int(resume_payload.get("tokens_seen") or 0))
        try:
            if resume_payload.get("optimizer"):
                opt.load_state_dict(resume_payload["optimizer"])
            if scaler is not None and resume_payload.get("scaler"):
                scaler.load_state_dict(resume_payload["scaler"])
        except Exception as exc:
            # Optimizer/scaler state is optional. The learned model weights are
            # still valuable, so an optimizer mismatch must not force a fresh
            # untrained model. Continue from checkpoint parameters with a new
            # optimizer and a new run counter instead.
            opt=_optimizer(raw,config)
            scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
            resume_step=0
            resume_tokens=0
            progress({
                "status":"running","runtime_kind":"train","phase":"resume_optimizer_reset","overall":0,
                "step":0,"tokens_seen":0,"resume_from":str(resume_path),
                "message":f"Checkpoint weights loaded, but optimizer state is incompatible ({type(exc).__name__}). Continuing retraining with a fresh optimizer…",
            })

    budget=str(config.get("budget_type") or "steps").lower()
    max_steps=runtime_int(config.get("max_steps"),1000,"Training Steps",minimum=1)
    max_tokens=runtime_int(config.get("max_tokens"),1000000,"Token Budget",minimum=1)
    epochs=runtime_float(config.get("epochs"),1.0,"Epochs",minimum=0.000001)
    if budget not in {"steps","tokens","epochs"}:
        raise ValueError(f"Budget By must be steps, tokens, or epochs; received {budget!r}.")
    if budget=="epochs":
        max_steps=max(1,math.ceil(len(train)/batch*epochs))

    # A checkpoint saved exactly at the old budget has no remaining work. In
    # that case treat its weights as the starting point for a new retraining run
    # instead of immediately finishing with no loss value.
    exhausted_resume=(
        resume_kind=="checkpoint" and (
            (budget in {"steps","epochs"} and resume_step>=max_steps)
            or (budget=="tokens" and resume_tokens>=max_tokens)
        )
    )
    if exhausted_resume:
        resume_kind="weights"
        resume_step=0
        resume_tokens=0
        opt=_optimizer(raw,config)
        scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
        progress({
            "status":"running","runtime_kind":"train","phase":"resume_budget_reset","overall":0,
            "step":0,"tokens_seen":0,"resume_from":str(resume_path),
            "message":"Existing checkpoint already reached the configured budget. Starting a new retraining run from its learned weights…",
        })

    validate_every=runtime_int(config.get("validate_every"),100,"Validate Every N Steps",minimum=0)
    val_steps=runtime_int(config.get("validation_steps"),20,"Validation Steps",minimum=1)
    checkpoint_every=runtime_int(config.get("checkpoint_every"),500,"Checkpoint Every N Steps",minimum=0)

    def _train_overall(current_step, current_tokens):
        """Reserve the last 5% for validation, save and artifact commit.

        The old runtime reported 99% as soon as the last optimizer step ended,
        even though final validation, sample generation and model serialization
        were still pending. That made healthy retrains look permanently stuck.
        """
        if budget=="tokens":
            ratio=min(1.0,max(0.0,float(current_tokens)/max(float(max_tokens),1.0)))
        else:
            ratio=min(1.0,max(0.0,float(current_step)/max(float(max_steps),1.0)))
        return min(95,max(2,round(ratio*95)))

    def _is_final_training_point(current_step,current_tokens):
        if budget=="tokens":
            return current_tokens>=max_tokens
        return current_step>=max_steps
    output=Path(str(config.get("output_dir") or "mlbricks_workspace/models"))/_safe_name(model_entry.get("name","model"))
    output.mkdir(parents=True,exist_ok=True)
    (output/'checkpoints').mkdir(exist_ok=True)
    saved_training_config=copy.deepcopy(config)
    logical_output_dir=saved_training_config.pop("_studio_logical_output_dir",None)
    if logical_output_dir:
        saved_training_config["output_dir"]=str(logical_output_dir)

    architecture=copy.deepcopy(model_entry.get("architecture") or _root_model(state))
    custom_components=copy.deepcopy(state.get("custom_components") or {})
    custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {}))
    builder_package={
        "format":"mlb-studio-model-v2",
        "builder_version":__version__,
        "project":copy.deepcopy(state.get("project") or {}),
        "model_component":architecture,
        "custom_components":custom_components,
        "model_entry":copy.deepcopy(model_entry),
        "dataset_meta":copy.deepcopy(dataset_meta or {}),
    }

    train_batcher=_PackedLMBatcher(
        train,context=context,separator_id=separator,rng=random.Random(seed ^ 0x54524149)
    )
    val_batcher=(
        _PackedLMBatcher(val,context=context,separator_id=separator,rng=random.Random(seed ^ 0x56414C49))
        if val is not None else None
    )
    # Validation uses the same raw weights but does not need another compiled
    # training graph variant under no_grad().
    eval_loss_model=_CausalLMTrainingGraph(raw)

    tokens_seen=resume_tokens;run_tokens_seen=0;best_val=float('inf');last_val=None;last_val_ppl=None
    model.train();raw.train();loss_model.train()

    # torch.compile is lazy. Prepare two exact-shape batches first, then force
    # two forward+backward passes so compilation and first-use autotuning are
    # excluded from the throughput timer. No optimizer step is taken.
    compile_seconds=0.0
    if compiled.compile_used:
        progress({
            "status":"running","runtime_kind":"train","phase":"compile_warmup","overall":1,
            "step":resume_step,"max_steps":max_steps,"tokens_seen":tokens_seen,"tokens_per_sec":None,
            "avg_tokens_per_sec":None,"end_to_end_tokens_per_sec":None,
            "avg_end_to_end_tokens_per_sec":None,"loss":None,"ppl":None,
            "val_loss":None,"val_ppl":None,**_memory_snapshot(device),
            "message":f"Compiling one whole training graph on {device} · fixed shape [{batch}, {context}] · 2 warm-up passes…",
        })
        warmup_batcher=_PackedLMBatcher(
            train,context=context,separator_id=separator,rng=random.Random(seed ^ 0x4D4C4252)
        )
        warm_batches=[warmup_batcher.batch(batch,device) for _ in range(2)]
        _sync_device(device)
        if device.type=="cuda":
            try: torch.cuda.reset_peak_memory_stats(device)
            except Exception: pass
        compile_started=time.perf_counter()
        for xw,yw,_ in warm_batches:
            opt.zero_grad(set_to_none=True)
            with _autocast_context(device,precision):
                warm_loss=loss_model(xw,yw)/accum
            if scaler is not None and scaler.is_enabled(): scaler.scale(warm_loss).backward()
            else: warm_loss.backward()
        _sync_device(device)
        compile_seconds=max(time.perf_counter()-compile_started,0.0)
        opt.zero_grad(set_to_none=True)
        progress({
            "status":"running","runtime_kind":"train","phase":"compile_done","overall":2,
            "step":resume_step,"max_steps":max_steps,"tokens_seen":tokens_seen,"tokens_per_sec":None,
            "avg_tokens_per_sec":None,"end_to_end_tokens_per_sec":None,
            "avg_end_to_end_tokens_per_sec":None,"loss":None,"ppl":None,
            "val_loss":None,"val_ppl":None,**_memory_snapshot(device),
            "compile_seconds":compile_seconds,
            "message":f"Whole-model compilation complete · {compile_seconds:.1f}s · throughput timer starts now",
        })

    if device.type=="cuda":
        try: torch.cuda.reset_peak_memory_stats(device)
        except Exception: pass
    wall_start=time.perf_counter()
    gpu_train_seconds=0.0
    e2e_train_seconds=0.0
    progress({
        "status":"running","runtime_kind":"train","phase":"train","overall":2,
        "step":resume_step,"max_steps":max_steps,"tokens_seen":tokens_seen,"tokens_per_sec":None,
        "avg_tokens_per_sec":None,"end_to_end_tokens_per_sec":None,
        "avg_end_to_end_tokens_per_sec":None,"loss":None,"ppl":None,
        "val_loss":None,"val_ppl":None,**_memory_snapshot(device),
        "compile_seconds":compile_seconds,
        "message":f"Training started on {device} · {compiled.parameter_count:,} parameters · packed [{batch}, {context}]"+
                  (f" · retraining existing weights from {resume_path.name}" if resume_path is not None and resume_kind!="checkpoint" else "")+
                  (f" · resumed checkpoint step {resume_step}" if resume_path is not None and resume_kind=="checkpoint" else "")+
                  (f" · whole-model compiled ({compile_seconds:.1f}s warm-up)" if compiled.compile_used else " · eager"),
        "compile_warning":compiled.compile_error,
    })

    step=resume_step;loss_value=None;sample=None
    while True:
        if stop_event.is_set(): raise TrainingStopped("Training stopped.")
        if budget=="steps" and step>=max_steps:break
        if budget=="tokens" and tokens_seen>=max_tokens:break
        if budget=="epochs" and step>=max_steps:break
        step+=1

        # Prepare packed CPU batches and finish H2D transfer before GPU timing.
        # This makes Tok/s a model-training metric, while E2E Tok/s separately
        # reports data preparation + transfer + model training.
        e2e_started=time.perf_counter()
        microbatches=[train_batcher.batch(batch,device) for _ in range(accum)]
        _sync_device(device)
        step_started=time.perf_counter()

        opt.zero_grad(set_to_none=True)
        step_tokens=0
        detached_losses=[]
        for x,y,toks in microbatches:
            step_tokens+=toks
            with _autocast_context(device,precision):
                loss=loss_model(x,y)/accum
            if scaler is not None and scaler.is_enabled():scaler.scale(loss).backward()
            else:loss.backward()
            detached_losses.append(loss.detach())

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0)
            scaler.step(opt);scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0)
            opt.step()

        if warm>0 and step<=warm:
            factor=step/warm
            base_lr=runtime_float(config.get('learning_rate'),5e-4,'Learning Rate',minimum=0.0)
            for group in opt.param_groups:group['lr']=base_lr*factor

        _sync_device(device)
        gpu_elapsed=max(time.perf_counter()-step_started,1e-9)
        e2e_elapsed=max(time.perf_counter()-e2e_started,1e-9)
        gpu_train_seconds+=gpu_elapsed
        e2e_train_seconds+=e2e_elapsed
        tokens_seen+=step_tokens
        run_tokens_seen+=step_tokens
        loss_value=sum(float(v.float().cpu()) for v in detached_losses)
        tokens_per_sec=float(step_tokens)/gpu_elapsed
        avg_tokens_per_sec=float(run_tokens_seen)/max(gpu_train_seconds,1e-9)
        end_to_end_tokens_per_sec=float(step_tokens)/e2e_elapsed
        avg_end_to_end_tokens_per_sec=float(run_tokens_seen)/max(e2e_train_seconds,1e-9)
        ppl=_perplexity(loss_value)
        mem=_memory_snapshot(device)
        lr=float(opt.param_groups[0].get('lr',0.0)) if opt.param_groups else None
        do_val=validate_every>0 and (step%validate_every==0 or (budget=="steps" and step==max_steps))
        sample=None

        base_event={
            "step":step,"max_steps":max_steps,"tokens_seen":tokens_seen,
            "tokens_per_sec":tokens_per_sec,"avg_tokens_per_sec":avg_tokens_per_sec,
            "end_to_end_tokens_per_sec":end_to_end_tokens_per_sec,
            "avg_end_to_end_tokens_per_sec":avg_end_to_end_tokens_per_sec,
            "loss":loss_value,"ppl":ppl,"val_loss":last_val,"val_ppl":last_val_ppl,
            "lr":lr,"compile_seconds":compile_seconds,
        }

        if do_val and val_batcher is not None:
            final_validation=_is_final_training_point(step,tokens_seen)
            validation_base_overall=96 if final_validation else _train_overall(step,tokens_seen)
            progress({
                "status":"running","runtime_kind":"train","phase":"validation",
                "overall":validation_base_overall,
                **base_event,**mem,
                "validation_step":0,"validation_steps":val_steps,
                "message":f"Validating at step {step} · 0/{val_steps} batches…",
            })

            def emit_validation_batch(done,total,running_val):
                if stop_event.is_set():
                    raise TrainingStopped("Training stopped.")
                if final_validation:
                    # Final validation owns 96-98%. Keep 99% for sample/save.
                    validation_overall=min(98,96+round(2*done/max(total,1)))
                else:
                    validation_overall=validation_base_overall
                progress({
                    "status":"running","runtime_kind":"train","phase":"validation",
                    "overall":validation_overall,
                    **base_event,**_memory_snapshot(device),
                    "validation_step":done,"validation_steps":total,
                    "validation_running_loss":running_val,
                    "message":f"Validating at step {step} · {done}/{total} batches…",
                })

            last_val=_evaluate(
                eval_loss_model,raw,val_batcher,steps=val_steps,batch_size=batch,
                device=device,precision=precision,
                progress_callback=emit_validation_batch,stop_event=stop_event,
            )
            best_val=min(best_val,last_val)
            last_val_ppl=_perplexity(last_val)
            base_event.update({"val_loss":last_val,"val_ppl":last_val_ppl})

            if _bool(config.get("generate_on_validation",True)):
                sample_tokens=runtime_int(
                    config.get("validation_generate_tokens"),64,
                    "Validation Sample Tokens",minimum=1,
                )
                progress({
                    "status":"running","runtime_kind":"train","phase":"validation_generation",
                    "overall":98 if final_validation else validation_base_overall,
                    **base_event,**_memory_snapshot(device),
                    "validation_step":val_steps,"validation_steps":val_steps,
                    "validation_generated_tokens":0,
                    "validation_generate_tokens":sample_tokens,
                    "message":f"Validation complete · generating sample 0/{sample_tokens} tokens…",
                })

                def emit_validation_generation(event):
                    generated_count=int((event or {}).get("generated_tokens") or 0)
                    progress({
                        "status":"running","runtime_kind":"train","phase":"validation_generation",
                        "overall":98 if final_validation else validation_base_overall,
                        **base_event,**_memory_snapshot(device),
                        "validation_step":val_steps,"validation_steps":val_steps,
                        "validation_generated_tokens":generated_count,
                        "validation_generate_tokens":sample_tokens,
                        "message":f"Validation complete · generating sample {generated_count}/{sample_tokens} tokens…",
                    })

                try:
                    sample,_=generate_text(
                        model,tokenizer,config.get("validation_prompt","Once upon a time"),
                        max_new_tokens=sample_tokens,
                        context=context,device=device,precision=precision,temperature=.8,top_k=50,top_p=.95,
                        seed=seed+step,stop_event=stop_event,progress=emit_validation_generation,
                    )
                except TrainingStopped:
                    raise
                except Exception as exc:
                    sample=f"[sample generation skipped: {exc}]"
            mem=_memory_snapshot(device)
            progress({
                "status":"running","runtime_kind":"train","phase":"validation_done",
                "overall":98 if final_validation else validation_base_overall,
                **base_event,
                "best_val_loss":None if best_val==float('inf') else best_val,
                "sample_text":sample,"elapsed_seconds":time.perf_counter()-wall_start,**mem,
                "validation_step":val_steps,"validation_steps":val_steps,
                "message":f"Validation complete · val loss {last_val:.4f} · val ppl {last_val_ppl:.2f}",
            })

        if checkpoint_every>0 and step%checkpoint_every==0:
            checkpoint_path=output/'checkpoints'/f'step_{step:06d}'
            metadata={
                "kind":"training_checkpoint","step":step,"tokens_seen":tokens_seen,
                "vocab_size":compiled.vocab_size,"training_config":copy.deepcopy(saved_training_config),
                "builder_package":builder_package,
            }
            mlbricks_save = IMPORT_POOL.resolve_api("lifecycle.save")
            mlbricks_save(raw,checkpoint_path,metadata=metadata)
            # Optimizer/scaler state is supplemental training state; the model
            # itself is always stored through the public MLBricks lifecycle API.
            torch.save({
                "optimizer":opt.state_dict(),"scaler":scaler.state_dict() if scaler is not None else None,
                "step":step,"tokens_seen":tokens_seen,
            },checkpoint_path/'training_state.pt')
            progress({
                "status":"running","runtime_kind":"train","phase":"checkpoint",
                "overall":_train_overall(step,tokens_seen),
                **base_event,"best_val_loss":None if best_val==float('inf') else best_val,
                "sample_text":sample,"checkpoint_path":str(checkpoint_path),
                "elapsed_seconds":time.perf_counter()-wall_start,**_memory_snapshot(device),
                "message":f"MLBricks checkpoint saved · step {step}",
            })

        overall=(
            98 if (do_val and val_batcher is not None and _is_final_training_point(step,tokens_seen))
            else _train_overall(step,tokens_seen)
        )
        mem=_memory_snapshot(device)
        mem_text=(f" · mem {mem['memory_allocated_gb']:.2f} GB" if mem.get('memory_allocated_gb') is not None else "")
        val_text=(f" · val {last_val:.4f} · val ppl {last_val_ppl:.2f}" if last_val is not None else "")
        progress({
            "status":"running","runtime_kind":"train","phase":"train","overall":overall,
            **base_event,"best_val_loss":None if best_val==float('inf') else best_val,
            "sample_text":sample,"elapsed_seconds":time.perf_counter()-wall_start,**mem,
            "message":f"Step {step} · {tokens_per_sec:,.0f} GPU tok/s · {end_to_end_tokens_per_sec:,.0f} E2E tok/s · loss {loss_value:.4f} · ppl {ppl:.2f}"+val_text+mem_text,
        })

    final=output/'last'
    mlbricks_save = IMPORT_POOL.resolve_api("lifecycle.save")
    tok_cfg=((dataset_meta or {}).get("pipeline") or {}).get("tokenizer") or {}
    progress({
        "status":"running","runtime_kind":"train","phase":"final_save","overall":99,
        "step":step,"max_steps":max_steps,"tokens_seen":tokens_seen,
        "tokens_per_sec":tokens_per_sec if 'tokens_per_sec' in locals() else None,
        "avg_tokens_per_sec":float(run_tokens_seen)/max(gpu_train_seconds,1e-9) if run_tokens_seen else None,
        "end_to_end_tokens_per_sec":end_to_end_tokens_per_sec if 'end_to_end_tokens_per_sec' in locals() else None,
        "avg_end_to_end_tokens_per_sec":float(run_tokens_seen)/max(e2e_train_seconds,1e-9) if run_tokens_seen else None,
        "loss":loss_value,"ppl":_perplexity(loss_value) if loss_value is not None else None,
        "val_loss":last_val,"val_ppl":last_val_ppl,
        "best_val_loss":None if best_val==float('inf') else best_val,
        "compile_seconds":compile_seconds,"elapsed_seconds":time.perf_counter()-wall_start,
        **_memory_snapshot(device),
        "message":"Training steps complete · saving final MLBricks model artifact…",
    })
    final_metadata={
        "kind":"trained_model","step":step,"tokens_seen":tokens_seen,
        "best_val_loss":None if best_val==float('inf') else best_val,
        "training_config":copy.deepcopy(saved_training_config),"builder_package":builder_package,
        "tokenizer_name":tok_cfg.get("tokenizer_name") or "gpt2",
        "execution":"whole-model compiled" if compiled.compile_used else "eager",
        "compile_mode":str(config.get("compile_mode") or "default") if compiled.compile_used else None,
        "compile_fullgraph":True if compiled.compile_used else None,
        "compile_dynamic":False if compiled.compile_used else None,
    }
    mlbricks_save(raw,final,metadata=final_metadata)
    progress({
        "status":"running","runtime_kind":"train","phase":"finalize","overall":99,
        "step":step,"max_steps":max_steps,"tokens_seen":tokens_seen,
        "loss":loss_value,"ppl":_perplexity(loss_value) if loss_value is not None else None,
        "val_loss":last_val,"val_ppl":last_val_ppl,
        "best_val_loss":None if best_val==float('inf') else best_val,
        "compile_seconds":compile_seconds,"elapsed_seconds":time.perf_counter()-wall_start,
        **_memory_snapshot(device),
        "message":"Model weights saved · finalizing tokenizer and artifact metadata…",
    })
    tokenizer_dir=final/'tokenizer'
    try:
        tokenizer.save_pretrained(str(tokenizer_dir))
    except Exception:
        tokenizer_dir=None

    final_mem=_memory_snapshot(device)
    update={
        "training_status":"trained","weights_ready":True,"path":str(final),"checkpoint_path":str(final),
        "trained_steps":step,"tokens_seen":tokens_seen,"last_loss":loss_value,"last_ppl":_perplexity(loss_value),
        "best_val_loss":None if best_val==float('inf') else best_val,"last_val_loss":last_val,"last_val_ppl":last_val_ppl,
        "avg_tokens_per_sec":float(run_tokens_seen)/max(gpu_train_seconds,1e-9),
        "avg_end_to_end_tokens_per_sec":float(run_tokens_seen)/max(e2e_train_seconds,1e-9),
        "memory_peak_gb":final_mem.get("memory_peak_gb"),"parameter_count":compiled.parameter_count,
        "effective_vocab_size":compiled.vocab_size,"execution_mode_used":"compiled" if compiled.compile_used else "eager",
        "compile_warning":compiled.compile_error,"compile_seconds":compile_seconds,
        "trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        "format":"MLBricks model artifact","artifact_format":"mlbricks.model",
        "tokenizer_path":str(tokenizer_dir) if tokenizer_dir is not None else None,
        "retrained_from":str(resume_path) if resume_path is not None and resume_kind!="checkpoint" else None,
        "resumed_checkpoint":str(resume_path) if resume_path is not None and resume_kind=="checkpoint" else None,
    }
    return {"compiled":compiled,"tokenizer":tokenizer,"model_update":update,"last_sample":sample}


def load_trained_for_generation(*,state,model_entry,dataset_meta,config,checkpoint_path=None,progress=None):
    """Load a trained runtime with a fast path for unified MLBricks artifacts.

    Studio training saves ``raw`` TensorGraph itself with ``mlbricks.save``.  The
    older generation path rebuilt the entire 50M+ graph, then called
    ``mlbricks.load`` (which loaded a second full graph), and finally copied the
    second graph's state_dict into the first.  On Windows/CPU this can dominate
    first-token latency by tens of seconds.

    Unified artifacts already contain the complete trained module graph, so the
    normal eager/auto generation path can load that graph directly, cast/move it
    once, and reuse it in Builder's runtime cache.  Explicit backend overrides
    still use the reconstruction path because they intentionally re-instantiate
    components with a different backend.
    """
    path=Path(str(checkpoint_path or model_entry.get("checkpoint_path") or model_entry.get("path") or ""))
    if not path.exists():
        raise RuntimeError("Trained model artifact was not found. Train the model in this session or select a valid MLBricks model artifact.")

    backend=str(config.get("backend") or "auto").lower()
    unified=path.is_dir() and (path/"model.pt").exists()
    direct_ok=unified and backend in {"", "auto"}

    if direct_ok:
        device=resolve_device(config.get("device","auto"))
        precision,dtype=resolve_precision(config.get("precision","auto"),device)
        if progress:
            progress({
                "status":"running","runtime_kind":"generate","phase":"tokenizer","overall":0,
                "message":f"Loading tokenizer for {device}…",
            })
        tokenizer=_tokenizer_for(
            dataset_meta, tokenizer_path=(model_entry or {}).get("tokenizer_path")
        )
        if progress:
            progress({
                "status":"running","runtime_kind":"generate","phase":"weights","overall":0,
                "message":f"Loading trained model directly from {path.name or path}…",
            })
        try:
            mlbricks_load=IMPORT_POOL.resolve_api("lifecycle.load")
        except ImportError as exc:
            raise RuntimeError("Current MLBricks installation does not expose mlbricks.load().") from exc
        loaded=mlbricks_load(path,device=device,strict=True)
        if not isinstance(loaded,nn.Module):
            raise RuntimeError(f"Loaded MLBricks artifact is not a torch.nn.Module: {type(loaded)!r}")
        # mlbricks.load already places the model on the requested device. Cast
        # floating tensors once to the generation precision instead of creating
        # and moving a second graph first.
        loaded.to(device=device,dtype=dtype)
        graph=copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
        effective_vocab=max(_graph_vocab(graph),len(tokenizer))
        params=sum(p.numel() for p in loaded.parameters())
        inference_model=loaded
        compile_used=False
        if str(config.get("execution_mode") or "eager")=="compiled":
            if not hasattr(torch,"compile"):
                raise RuntimeError("Compiled execution was selected, but torch.compile is unavailable in this PyTorch build.")
            inference_model=torch.compile(
                loaded,mode=str(config.get("compile_mode") or "default"),dynamic=None,fullgraph=False
            )
            compile_used=True
        return CompiledModel(
            inference_model,loaded,None,device,precision,effective_vocab,
            params,compile_used,None,
        ),tokenizer

    # Compatibility path for legacy checkpoints and explicit backend overrides.
    compiled,tokenizer=compile_builder_model(
        state,model_entry,dataset_meta,config,progress=progress,for_training=False
    )

    if progress:
        progress({
            "status":"running","runtime_kind":"generate","phase":"weights",
            "overall":0,
            "message":f"Loading trained weights from {path.name or path}…",
        })

    if path.is_dir() and (path/"model.pt").exists():
        try:
            mlbricks_load = IMPORT_POOL.resolve_api("lifecycle.load")
        except ImportError as exc:
            raise RuntimeError("Current MLBricks installation does not expose mlbricks.load().") from exc
        loaded=mlbricks_load(path,device=compiled.device,strict=True)
        compiled.raw_model.load_state_dict(loaded.state_dict(),strict=True)
        del loaded
    else:
        # Legacy MLB Studio checkpoint format.
        payload=safe_torch_load(path,map_location="cpu",allow_unsafe_pickle=_bool(config.get("allow_unsafe_legacy_checkpoint",False)))
        if not isinstance(payload,dict) or "model_state" not in payload:
            raise RuntimeError("Selected file is neither an MLBricks model artifact nor a legacy Builder checkpoint.")
        compiled.raw_model.load_state_dict(payload["model_state"],strict=True)

    compiled.raw_model.to(compiled.device)
    return compiled,tokenizer
