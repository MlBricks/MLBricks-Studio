import torch
import torch.nn as nn

from mlb_studio.api_graph_runtime import APIComponentContract, API_COMPONENTS
from mlb_studio.model_runtime import TensorGraph


class _FakeBolt(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim=32, backend="auto", **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.backend = backend
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(x)


class _Echo(nn.Module):
    def forward(self, x):
        return x + 1


def _runtime():
    return {
        "device": "cpu",
        "backend": "pytorch",
        "precision": "fp32",
        "model_dim": 16,
        "heads": 4,
    }


def test_bolt_uses_declarative_api_contract_and_legacy_aliases(monkeypatch):
    from mlb_studio import api_graph_runtime

    original = api_graph_runtime.IMPORT_POOL.resolve_component
    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeBolt if key == "bolt" else original(key),
    )

    graph = TensorGraph(
        nodes=[{
            "id": "bolt-1",
            "type": "bolt",
            "name": "BOLT",
            # Old Studio graphs used dim/kernel.  The contract maps these to
            # the current original MLBricks d_model/backend API.
            "params": {"dim": 16, "num_heads": 4, "latent_dim": 8, "kernel": "pytorch"},
        }],
        edges=[],
        custom_components={},
        runtime=_runtime(),
    )

    module = graph.mods["bolt-1"]
    assert isinstance(module, _FakeBolt)
    assert module.d_model == 16
    assert module.num_heads == 4
    assert module.latent_dim == 8
    assert module.backend == "pytorch"

    x = torch.randn(2, 5, 16, requires_grad=True)
    y = graph(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_new_direct_component_can_join_executor_without_tensorgraph_branch(monkeypatch):
    """Registration alone is enough for a simple one-input/one-output API."""
    from mlb_studio import api_graph_runtime

    component_type = "__test_echo__"
    API_COMPONENTS.register(APIComponentContract(
        component_type=component_type,
        import_key=component_type,
        input_ports={"main": "x"},
        output_ports={"main": None},
    ))

    original = api_graph_runtime.IMPORT_POOL.resolve_component
    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _Echo if key == component_type else original(key),
    )

    graph = TensorGraph(
        nodes=[{"id": "e", "type": component_type, "name": "Echo", "params": {}}],
        edges=[],
        custom_components={},
        runtime=_runtime(),
    )
    x = torch.zeros(1, 2, 3)
    assert torch.equal(graph(x), torch.ones_like(x))


class _FakeResController(nn.Module):
    def __init__(self, update_ratio, stream_ratio=1.08, backend="auto", **kwargs):
        super().__init__()
        self.update_ratio = float(update_ratio)
        self.stream_ratio = float(stream_ratio)
        self.backend = backend

    def forward(self, residual, update):
        return residual + self.update_ratio * update


def test_rescontroller_routes_main_and_skip_into_original_api_without_tensorgraph_branch(monkeypatch):
    """Main -> update and Skip -> residual are routed by the declarative contract."""
    from mlb_studio import api_graph_runtime

    component_type = "__test_echo_rescontroller__"
    API_COMPONENTS.register(APIComponentContract(
        component_type=component_type,
        import_key=component_type,
        input_ports={"main": "x"},
        output_ports={"main": None},
    ))

    original = api_graph_runtime.IMPORT_POOL.resolve_component
    def resolve(key):
        if key == "rescontroller":
            return _FakeResController
        if key == component_type:
            return _Echo
        return original(key)
    monkeypatch.setattr(api_graph_runtime.IMPORT_POOL, "resolve_component", resolve)

    graph = TensorGraph(
        nodes=[
            {"id": "echo-rc", "type": component_type, "name": "Update", "params": {}},
            {
                "id": "res",
                "type": "rescontroller",
                "name": "ResController",
                "params": {"update_ratio": 0.25, "stream_ratio": 1.08, "backend": "pytorch"},
            },
        ],
        edges=[
            {
                "id": "main-edge",
                "source": "echo-rc",
                "target": "res",
                "kind": "main",
                "source_port": "main_out",
                "target_port": "main_in",
            },
            {
                "id": "skip-edge",
                "source": "echo-rc",
                "target": "res",
                "kind": "residual",
                "source_port": "skip_out",
                "target_port": "skip_in",
            },
        ],
        custom_components={},
        runtime=_runtime(),
    )

    x = torch.zeros(1, 2, 3, requires_grad=True)
    y = graph(x)
    assert torch.allclose(y, torch.full_like(y, 1.25))
    y.sum().backward()
    assert x.grad is not None


def test_rescontroller_requires_skip_residual_input(monkeypatch):
    from mlb_studio import api_graph_runtime
    from mlb_studio.model_runtime import ModelCompileError

    original = api_graph_runtime.IMPORT_POOL.resolve_component
    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeResController if key == "rescontroller" else original(key),
    )

    graph = TensorGraph(
        nodes=[{
            "id": "res",
            "type": "rescontroller",
            "name": "ResController",
            "params": {"update_ratio": 0.25, "backend": "pytorch"},
        }],
        edges=[],
        custom_components={},
        runtime=_runtime(),
    )

    x = torch.randn(1, 2, 3)
    try:
        graph(x)
    except ModelCompileError as exc:
        assert "requires the Skip input" in str(exc)
        assert "residual" in str(exc)
    else:
        raise AssertionError("ResController should require the residual Skip input")
