from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mlb_studio.api_graph_runtime import API_COMPONENTS
from mlb_studio.model_runtime import TensorGraph


class _FakeRoPE(nn.Module):
    def __init__(self, dim=None, base=10000.0):
        super().__init__()
        self.dim = None if dim is None else int(dim)
        self.base = float(base)

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("RoPE expects [B,H,T,D]")

        width = x.size(-1) if self.dim is None else min(self.dim, x.size(-1))
        width -= width % 2
        if width <= 0:
            return x

        dtype = x.dtype
        xf = x[..., :width].float()

        positions = torch.arange(
            start_pos,
            start_pos + x.size(-2),
            device=x.device,
            dtype=torch.float32,
        )
        inv = self.base ** (
            -torch.arange(
                0,
                width,
                2,
                device=x.device,
                dtype=torch.float32,
            ) / width
        )

        phase = positions[:, None] * inv[None, :]
        cos = torch.cos(phase)[None, None, :, :]
        sin = torch.sin(phase)[None, None, :, :]

        even = xf[..., 0::2]
        odd = xf[..., 1::2]

        rotated = torch.empty_like(xf)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos

        if width == x.size(-1):
            return rotated.to(dtype)

        return torch.cat(
            [rotated.to(dtype), x[..., width:]],
            dim=-1,
        )


def _runtime():
    return {
        "device": "cpu",
        "backend": "pytorch",
        "precision": "fp32",
        "model_dim": 16,
        "heads": 4,
    }


def _node(params=None):
    return [{
        "id": "rope-1",
        "type": "rope",
        "name": "RoPE",
        "params": dict(params or {}),
    }]


def test_rope_is_registered_as_universal_model_runtime_component():
    contract = API_COMPONENTS.get("rope")
    assert contract is not None
    assert contract.import_key == "rope"
    assert dict(contract.input_ports) == {"main": "x"}
    assert dict(contract.output_ports) == {"main": None}


def test_rope_executes_4d_qk_tensor_and_preserves_gradient(monkeypatch):
    from mlb_studio import api_graph_runtime

    original = api_graph_runtime.IMPORT_POOL.resolve_component

    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeRoPE if key == "rope" else original(key),
    )

    graph = TensorGraph(
        nodes=_node({
            "dim": 8,
            "base": 10000.0,
        }),
        edges=[],
        custom_components={},
        runtime=_runtime(),
    )

    x = torch.randn(2, 4, 6, 8, requires_grad=True)
    y = graph(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert not torch.equal(y[..., 1:, :], x[..., 1:, :])

    y.square().mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_rope_rejects_non_qk_4d_tensor(monkeypatch):
    from mlb_studio import api_graph_runtime

    original = api_graph_runtime.IMPORT_POOL.resolve_component

    monkeypatch.setattr(
        api_graph_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeRoPE if key == "rope" else original(key),
    )

    graph = TensorGraph(
        nodes=_node({"dim": 8}),
        edges=[],
        custom_components={},
        runtime=_runtime(),
    )

    with pytest.raises(ValueError, match=r"\[B,H,T,D\]"):
        graph(torch.randn(2, 6, 8))
