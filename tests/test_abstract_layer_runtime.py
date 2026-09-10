from __future__ import annotations

import torch
import torch.nn as nn

from mlb_studio import model_runtime
from mlb_studio.graph import tinystories_30m_project
from mlb_studio.model_runtime import (
    _AbstractLayerComponent,
    _lane_output,
    _named_output,
)


class _IdentityNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class _FakeESA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x + 1

    def prefill(self, x):
        return x + 1, {"seen": int(x.size(1))}

    def decode_step(self, x, state):
        return x + 1, state


class _FakeFFN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x * 2


class _FakeResidual(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, skip, main):
        return skip + main


def _runtime():
    return {
        "device": "cpu",
        "backend": "pytorch",
        "precision": "fp32",
        "model_dim": 4,
        "heads": 2,
    }


def _patch_components(monkeypatch):
    original = model_runtime.IMPORT_POOL.resolve_component

    def resolve(key):
        if key == "layernorm":
            return _IdentityNorm
        if key == "esa":
            return _FakeESA
        if key == "ffn":
            return _FakeFFN
        if key == "residual":
            return _FakeResidual
        return original(key)

    monkeypatch.setattr(model_runtime.IMPORT_POOL, "resolve_component", resolve)


def _standard_50m_definition():
    state = tinystories_30m_project()
    root = state["components"][state["root_component_id"]]
    layer = next(node for node in root["nodes"] if node.get("type") == "custom")
    return state["custom_components"][layer["definition_id"]]


def test_standard_abs_layer_executes_dual_residual_graph(monkeypatch):
    _patch_components(monkeypatch)
    definition = _standard_50m_definition()
    layer = _AbstractLayerComponent(
        definition=definition,
        params={},
        runtime=_runtime(),
        custom_components={definition["id"]: definition},
    )

    x = torch.ones(1, 2, 4)
    result = layer(x, skip=x)

    # ESA update = 2; first residual = 1 + 2 = 3;
    # FFN update = 6; second residual = 3 + 6 = 9.
    assert torch.equal(_lane_output(result, "main"), torch.full_like(x, 9))
    assert torch.equal(_lane_output(result, "skip"), torch.full_like(x, 9))


def test_abs_layer_fixed_and_custom_ports_are_additive():
    definition = {
        "id": "abs_test",
        "name": "ABS Test",
        "implementation": "abstract_layer",
        "interface": {
            "input_ports": [{"id": "state_in", "name": "state_in", "side": "top", "order": 0}],
            "output_ports": [{"id": "state_out", "name": "state_out", "side": "bottom", "order": 0}],
        },
        "nodes": [
            {"id": "in", "type": "abstract_input", "name": "Layer Inputs", "params": {}},
            {"id": "out", "type": "abstract_output", "name": "Layer Outputs", "params": {}},
        ],
        "edges": [
            {"id": "main", "source": "in", "target": "out", "kind": "main", "source_port": "main_out", "target_port": "main_in"},
            {"id": "skip", "source": "in", "target": "out", "kind": "residual", "source_port": "skip_out", "target_port": "skip_in"},
            {"id": "extra", "source": "in", "target": "out", "kind": "aux", "source_port": "extra_out", "target_port": "extra_in"},
            {"id": "named", "source": "in", "target": "out", "kind": "named", "source_port": "named_out:state_in", "target_port": "named_in:state_out"},
        ],
    }
    layer = _AbstractLayerComponent(
        definition=definition,
        params={},
        runtime=_runtime(),
        custom_components={definition["id"]: definition},
    )

    main = torch.tensor([[1.0]])
    skip = torch.tensor([[2.0]])
    extra = torch.tensor([[3.0]])
    state = torch.tensor([[4.0]])
    result = layer(main, skip=skip, extra=extra, named_inputs={"state_in": state})

    assert torch.equal(_lane_output(result, "main"), main)
    assert torch.equal(_lane_output(result, "skip"), skip)
    assert torch.equal(_lane_output(result, "extra"), extra)
    assert torch.equal(_named_output(result, "state_out"), state)


def test_abs_layer_recurrent_generation_preserves_boundary_lanes_and_state(monkeypatch):
    _patch_components(monkeypatch)
    definition = _standard_50m_definition()
    layer = _AbstractLayerComponent(
        definition=definition,
        params={},
        runtime=_runtime(),
        custom_components={definition["id"]: definition},
    )
    supported, reason = layer.recurrent_generation_support()
    assert supported, reason

    x = torch.ones(1, 2, 4)
    result, cache = layer.prefill(x, skip=x, capacity=4)
    assert torch.equal(_lane_output(result, "main"), torch.full_like(x, 9))
    assert cache["position"] == 2

    one = torch.ones(1, 1, 4)
    result, cache = layer.decode_step(one, cache, skip=one)
    assert torch.equal(_lane_output(result, "main"), torch.full_like(one, 9))
    assert cache["position"] == 3
