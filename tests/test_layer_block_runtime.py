import torch
import torch.nn as nn

from mlb_studio import model_runtime
from mlb_studio.graph import primitive_catalog, tinystories_30m_project, esa_200m_project
from mlb_studio.model_runtime import TensorGraph


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


def _patch_components(monkeypatch):
    original = model_runtime.IMPORT_POOL.resolve_component

    def resolve(key):
        if key == "layernorm":
            return _IdentityNorm
        if key == "esa":
            return _FakeESA
        if key == "ffn":
            return _FakeFFN
        return original(key)

    monkeypatch.setattr(model_runtime.IMPORT_POOL, "resolve_component", resolve)


def _runtime():
    return {
        "device": "cpu",
        "backend": "pytorch",
        "precision": "fp32",
        "model_dim": 4,
        "heads": 2,
    }


def _layer_params():
    return {
        "dim": 4,
        "heads": 2,
        "ffn_dim": 8,
        "block": 8,
        "batch": 1,
        "compass": 2,
        "activation": "gelu",
        "dropout": 0.0,
        "norm_eps": 1e-5,
    }


def test_layer_block_catalog_exposes_signal_and_residual_ports():
    item = next(item for item in primitive_catalog() if item["type"] == "layer_block")
    assert [(p["id"], p["name"]) for p in item["runtime_ports"]["inputs"]] == [
        ("signal", "Signal In"),
        ("residual", "Residual In"),
    ]
    assert [(p["id"], p["name"]) for p in item["runtime_ports"]["outputs"]] == [
        ("signal", "Signal Out"),
        ("residual", "Residual Out"),
    ]


def test_layer_block_routes_explicit_signal_and_residual_lanes(monkeypatch):
    _patch_components(monkeypatch)
    nodes = [
        {"id": "input", "type": "text_input", "name": "Input", "params": {}},
        {"id": "layer", "type": "layer_block", "name": "Layer 1", "params": _layer_params()},
        {"id": "output", "type": "text_output", "name": "Output", "params": {}},
    ]
    edges = [
        {"id": "e1", "source": "input", "target": "layer", "source_port": "main_out", "target_port": "named_in:signal", "kind": "named"},
        {"id": "e2", "source": "input", "target": "layer", "source_port": "main_out", "target_port": "named_in:residual", "kind": "named"},
        {"id": "e3", "source": "layer", "target": "output", "source_port": "named_out:signal", "target_port": "main_in", "kind": "main"},
    ]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime=_runtime())
    x = torch.ones(1, 2, 4)
    y = graph(x)
    # residual_mid = 1 + (1 + 1) = 3; FFN = 6; residual_out = 9
    assert torch.equal(y, torch.full_like(x, 9))


def test_layer_block_recurrent_generation_keeps_explicit_lanes(monkeypatch):
    _patch_components(monkeypatch)
    nodes = [
        {"id": "layer", "type": "layer_block", "name": "Layer 1", "params": _layer_params()},
    ]
    graph = TensorGraph(nodes=nodes, edges=[], custom_components={}, runtime=_runtime())
    supported, reason = graph.recurrent_generation_support()
    assert supported, reason
    x = torch.ones(1, 2, 4)
    y, cache = graph.prefill(x, capacity=4)
    assert torch.equal(y, torch.full_like(x, 9))
    one = torch.ones(1, 1, 4)
    y2, cache = graph.decode_step(one, cache)
    assert torch.equal(y2, torch.full_like(one, 9))
    assert cache["position"] == 3


def test_release_presets_use_editable_abstract_layers_for_standard_esa_depth():
    for state, expected in ((tinystories_30m_project(), 10), (esa_200m_project(), 12)):
        graph = state["components"][state["root_component_id"]]
        layers = [n for n in graph["nodes"] if n.get("type") == "custom"]
        assert len(layers) == expected
        assert not [n for n in graph["nodes"] if n.get("type") == "layer_block"]
        definition_ids = {n.get("definition_id") for n in layers}
        assert len(definition_ids) == 1
        definition = state["custom_components"][next(iter(definition_ids))]
        assert definition.get("implementation") == "abstract_layer"
        assert [n.get("type") for n in definition.get("nodes", [])][0] == "abstract_input"
        assert [n.get("type") for n in definition.get("nodes", [])][-1] == "abstract_output"
        # First ABS gets both fixed lanes from embedding dropout.
        first = layers[0]
        incoming = [e for e in graph["edges"] if e.get("target") == first["id"]]
        assert {e.get("target_port") for e in incoming} == {"main_in", "skip_in"}
