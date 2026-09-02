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

class _FakeSAFFN(nn.Module):
    def __init__(self, d_model, state_dim=3, depth_embedding_dim=2, layer_index=0, total_layers=1, backend="auto", **kwargs):
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.layer_index = int(layer_index)
        self.total_layers = int(total_layers)
        self.backend = backend
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, esa_update, previous_esa, previous_state):
        main = self.scale * (x + esa_update - previous_esa)
        state = previous_state + self.scale
        return main, state


def test_saffn_routes_four_named_inputs_and_state_output_without_tensorgraph_branch(monkeypatch):
    """SAFFN's original 4-input / 2-output API is described only by its contract."""
    from mlb_studio import api_graph_runtime

    source_type = "__test_saffn_source__"
    API_COMPONENTS.register(APIComponentContract(
        component_type=source_type,
        import_key=source_type,
        input_ports={"main": "x"},
        output_ports={"main": None},
    ))

    original = api_graph_runtime.IMPORT_POOL.resolve_component
    def resolve(key):
        if key == "saffn":
            return _FakeSAFFN
        if key == source_type:
            return _Echo
        return original(key)
    monkeypatch.setattr(api_graph_runtime.IMPORT_POOL, "resolve_component", resolve)

    nodes = [
        {"id": "x", "type": source_type, "name": "X", "params": {}},
        {"id": "esa", "type": source_type, "name": "ESA", "params": {}},
        {"id": "prev", "type": source_type, "name": "Previous ESA", "params": {}},
        {"id": "state0", "type": source_type, "name": "State Init", "params": {}},
        {"id": "s1", "type": "saffn", "name": "SAFFN 1", "params": {"d_model": 3, "state_dim": 3, "layer_index": 0, "total_layers": 2, "backend": "pytorch"}},
        {"id": "s2", "type": "saffn", "name": "SAFFN 2", "params": {"d_model": 3, "state_dim": 3, "layer_index": 1, "total_layers": 2, "backend": "pytorch"}},
    ]

    def named(edge_id, source, target, target_key, source_key="main"):
        source_port = f"named_out:{source_key}" if source.startswith("s") else "main_out"
        return {
            "id": edge_id,
            "source": source,
            "target": target,
            "kind": "named",
            "source_port": source_port,
            "target_port": f"named_in:{target_key}",
        }

    edges = [
        named("1", "x", "s1", "x"),
        named("2", "esa", "s1", "esa_update"),
        named("3", "prev", "s1", "previous_esa"),
        named("4", "state0", "s1", "previous_state"),
        named("5", "s1", "s2", "x", "main"),
        named("6", "esa", "s2", "esa_update"),
        named("7", "prev", "s2", "previous_esa"),
        named("8", "s1", "s2", "previous_state", "state"),
    ]

    graph = TensorGraph(
        nodes=nodes,
        edges=edges,
        custom_components={},
        runtime={**_runtime(), "model_dim": 3},
    )

    x = torch.zeros(1, 2, 3, requires_grad=True)
    y = graph(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert graph.mods["s1"].scale.grad is not None
    assert graph.mods["s2"].scale.grad is not None


class _FakeEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))

    def forward(self, input_ids):
        return torch.nn.functional.embedding(input_ids, self.weight)


class _FakeLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.in_features = self.hidden_size
        self.out_features = self.vocab_size
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))
        self.bias = None
        self.tied_to = None

    def tie_weights(self, embedding):
        self.weight = embedding.weight
        self.tied_to = embedding

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


def test_weight_tying_traverses_named_api_edges(monkeypatch):
    """LM-head tying must see embeddings upstream through named SAFFN ports."""
    from mlb_studio import api_graph_runtime, model_runtime

    original = model_runtime.IMPORT_POOL.resolve_component

    def resolve(key):
        if key == "embedding":
            return _FakeEmbedding
        if key == "lm_head":
            return _FakeLMHead
        if key == "saffn":
            return _FakeSAFFN
        return original(key)

    monkeypatch.setattr(model_runtime.IMPORT_POOL, "resolve_component", resolve)
    # Both modules reference the same import pool singleton, but keep the intent
    # explicit in case that implementation detail changes later.
    monkeypatch.setattr(api_graph_runtime.IMPORT_POOL, "resolve_component", resolve)

    nodes = [
        {"id": "emb", "type": "embedding", "name": "Embedding", "params": {"vocab_size": 11, "embedding_dim": 3}},
        {"id": "s", "type": "saffn", "name": "SAFFN", "params": {"d_model": 3, "state_dim": 3, "layer_index": 0, "total_layers": 1, "backend": "pytorch"}},
        {"id": "head", "type": "lm_head", "name": "Head", "params": {"hidden_size": 3, "vocab_size": 11, "tie_embeddings": True}},
    ]
    edges = [
        {"id": "x", "source": "emb", "target": "s", "kind": "named", "source_port": "main_out", "target_port": "named_in:x"},
        {"id": "eu", "source": "emb", "target": "s", "kind": "named", "source_port": "main_out", "target_port": "named_in:esa_update"},
        {"id": "pe", "source": "emb", "target": "s", "kind": "named", "source_port": "main_out", "target_port": "named_in:previous_esa"},
        {"id": "ps", "source": "emb", "target": "s", "kind": "named", "source_port": "main_out", "target_port": "named_in:previous_state"},
        {"id": "out", "source": "s", "target": "head", "kind": "main", "source_port": "named_out:main", "target_port": "main_in"},
    ]

    graph = TensorGraph(
        nodes=nodes,
        edges=edges,
        custom_components={},
        runtime={**_runtime(), "model_dim": 3},
    )

    assert graph.mods["head"].tied_to is graph.mods["emb"]
    assert graph.mods["head"].weight is graph.mods["emb"].weight


def test_previous_value_buffer_holds_or_zero_initializes_without_parameters():
    held = TensorGraph(
        nodes=[{"id": "buf", "type": "value_buffer", "name": "Previous Value", "params": {"mode": "hold", "width": 0}}],
        edges=[], custom_components={}, runtime=_runtime(),
    )
    x = torch.randn(2, 5, 7, requires_grad=True)
    y = held(x)
    assert y is x
    y.sum().backward()
    assert x.grad is not None
    assert sum(p.numel() for p in held.parameters()) == 0

    zero_same = TensorGraph(
        nodes=[{"id": "buf", "type": "value_buffer", "name": "Previous ESA Init", "params": {"mode": "zero_init", "width": 0}}],
        edges=[], custom_components={}, runtime=_runtime(),
    )
    z = zero_same(torch.randn(2, 5, 7))
    assert z.shape == (2, 5, 7)
    assert torch.count_nonzero(z).item() == 0

    zero_state = TensorGraph(
        nodes=[{"id": "buf", "type": "value_buffer", "name": "Previous State Init", "params": {"mode": "zero_init", "width": 3}}],
        edges=[], custom_components={}, runtime=_runtime(),
    )
    state = zero_state(torch.randn(2, 5, 7))
    assert state.shape == (2, 5, 3)
    assert torch.count_nonzero(state).item() == 0


def test_single_layer_saffn_accepts_zero_initialized_previous_value_buffers(monkeypatch):
    from mlb_studio import api_graph_runtime

    original = api_graph_runtime.IMPORT_POOL.resolve_component
    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeSAFFN if key == "saffn" else original(key),
    )

    nodes = [
        {"id": "prev_esa", "type": "value_buffer", "name": "Previous ESA Init", "params": {"mode": "zero_init", "width": 0}},
        {"id": "prev_state", "type": "value_buffer", "name": "Previous State Init", "params": {"mode": "zero_init", "width": 3}},
        {"id": "s", "type": "saffn", "name": "SAFFN", "params": {"d_model": 3, "state_dim": 3, "layer_index": 0, "total_layers": 1, "backend": "pytorch"}},
    ]
    edges = [
        {"id":"pe_in","source":"prev_esa","target":"s","kind":"named","source_port":"main_out","target_port":"named_in:previous_esa"},
        {"id":"ps_in","source":"prev_state","target":"s","kind":"named","source_port":"main_out","target_port":"named_in:previous_state"},
    ]
    # x and esa_update deliberately come from graph input through two explicit
    # named edges; buffer nodes also receive graph input because they have no
    # upstream Main source.
    edges += [
        {"id":"x","source":"prev_esa","target":"s","kind":"named","source_port":"main_out","target_port":"named_in:x"},
        {"id":"eu","source":"prev_esa","target":"s","kind":"named","source_port":"main_out","target_port":"named_in:esa_update"},
    ]
    graph = TensorGraph(nodes=nodes, edges=edges, custom_components={}, runtime={**_runtime(), "model_dim": 3})
    x = torch.randn(1, 2, 3)
    # This is only a shape/routing test; the Gallery's real graph feeds x and
    # esa_update from Embedding/ESA while these buffers initialize prior depth.
    y = graph(x)
    assert y.shape == x.shape
