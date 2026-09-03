from __future__ import annotations

import torch
import pytest

from mlb_studio.model_runtime import (
    ModelCompileError,
    _APIOperation,
    _lane_output,
    _named_output,
)


def _binding():
    return {
        "call_type": "user_function",
        "user_function_name": "custom_function",
        "user_code": (
            "def custom_function(x, gain):\n"
            "    return x * gain, x + gain\n"
        ),
        "port_mode": "extended",
        "auto_main_input": True,
        "input_ports": [
            {
                "id": "gain_in",
                "name": "gain",
                "parameter": "gain",
                "required": True,
                "positional": False,
                "side": "top",
                "order": 0,
            }
        ],
        "output_ports": [
            {
                "id": "sum_out",
                "name": "sum",
                "selector": "1",
                "side": "bottom",
                "order": 0,
            }
        ],
        "output_selector": "0",
        "parameters": [],
    }


def test_extended_custom_terminals_keep_universal_main_and_named_output():
    op = _APIOperation(
        binding=_binding(),
        params={},
        runtime={"allow_user_code": True},
        label="Custom Function",
    )
    x = torch.tensor([[1.0, 2.0]])
    result = op(x, named_inputs={"gain_in": torch.tensor(3.0)})

    # Universal Main output remains available.
    assert torch.equal(_lane_output(result, "main"), torch.tensor([[3.0, 6.0]]))
    # Extra output terminal independently maps the second return value.
    assert torch.equal(_named_output(result, "sum_out"), torch.tensor([[4.0, 5.0]]))


def test_required_custom_input_terminal_must_be_connected():
    op = _APIOperation(
        binding=_binding(),
        params={},
        runtime={"allow_user_code": True},
        label="Custom Function",
    )
    with pytest.raises(ModelCompileError, match="input terminal 'gain' is not connected"):
        op(torch.tensor([[1.0, 2.0]]), named_inputs={})


def test_tensor_graph_can_feed_universal_main_and_extra_terminal_together():
    from mlb_studio.model_runtime import TensorGraph

    source = {
        "id": "source",
        "type": "api_step",
        "name": "Source",
        "params": {},
        "api_binding": {
            "call_type": "user_function",
            "user_function_name": "source",
            "user_code": "def source(x):\n    return x + 1\n",
            "port_mode": "standard",
            "auto_main_input": True,
            "parameters": [],
        },
    }
    target_binding = _binding()
    target_binding["user_code"] = "def custom_function(x, gain):\n    return x + gain\n"
    target_binding["output_ports"] = []
    target_binding["output_selector"] = "auto"
    target = {
        "id": "target",
        "type": "api_step",
        "name": "Target",
        "params": {},
        "api_binding": target_binding,
    }
    graph = TensorGraph(
        nodes=[source, target],
        edges=[
            {"id": "e1", "source": "source", "target": "target", "kind": "main", "source_port": "main_out", "target_port": "main_in"},
            {"id": "e2", "source": "source", "target": "target", "kind": "named", "source_port": "main_out", "target_port": "named_in:gain_in"},
        ],
        custom_components={},
        runtime={"device": "cpu", "precision": "fp32", "backend": "pytorch", "allow_user_code": True},
    )
    x = torch.tensor([[1.0, 2.0]])
    assert torch.equal(graph(x), torch.tensor([[4.0, 6.0]]))


def test_tensor_graph_can_route_custom_output_terminal_into_fixed_main_input():
    from mlb_studio.model_runtime import TensorGraph

    binding = _binding()
    binding["user_code"] = "def custom_function(x, gain=2):\n    return x + 1, x * gain\n"
    binding["input_ports"][0]["required"] = False
    source = {"id": "source", "type": "api_step", "name": "Source", "params": {}, "api_binding": binding}
    sink = {
        "id": "sink",
        "type": "api_step",
        "name": "Sink",
        "params": {},
        "api_binding": {
            "call_type": "user_function",
            "user_function_name": "sink",
            "user_code": "def sink(x):\n    return x + 5\n",
            "port_mode": "standard",
            "auto_main_input": True,
            "parameters": [],
        },
    }
    graph = TensorGraph(
        nodes=[source, sink],
        edges=[
            {"id": "e1", "source": "source", "target": "sink", "kind": "main", "source_port": "named_out:sum_out", "target_port": "main_in"},
        ],
        custom_components={},
        runtime={"device": "cpu", "precision": "fp32", "backend": "pytorch", "allow_user_code": True},
    )
    x = torch.tensor([[1.0, 2.0]])
    assert torch.equal(graph(x), torch.tensor([[7.0, 9.0]]))
