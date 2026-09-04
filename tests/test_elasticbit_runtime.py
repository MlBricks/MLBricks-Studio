from __future__ import annotations

import numpy as np
import pytest
import torch

from mlb_studio.elasticbit_runtime import build_elasticbit_runtime


class _FakeRuntimeMatrix:
    calls = []

    def __init__(self):
        self.forward_calls = []

    @classmethod
    def from_auto(
        cls,
        weights,
        calibration,
        threshold,
        runtime_mode="compact",
        min_bits=4,
        max_bits=32,
    ):
        instance = cls()
        cls.calls.append({
            "weights_shape": tuple(weights.shape),
            "calibration_shape": tuple(calibration.shape),
            "threshold": threshold,
            "runtime_mode": runtime_mode,
            "min_bits": min_bits,
            "max_bits": max_bits,
        })
        return instance

    def forward(self, value):
        self.forward_calls.append(value.copy())
        return np.asarray([float(value.sum())], dtype=np.float32)


class _FakeElasticBit:
    RuntimeMatrix = _FakeRuntimeMatrix
    analyser_calls = []

    @staticmethod
    def native_runtime_available():
        return True

    @staticmethod
    def bitsAnaliser(
        weights,
        calibration,
        threshold,
        min_bits=4,
        max_bits=32,
    ):
        _FakeElasticBit.analyser_calls.append({
            "weights_shape": tuple(weights.shape),
            "calibration_shape": tuple(calibration.shape),
            "threshold": threshold,
            "min_bits": min_bits,
            "max_bits": max_bits,
        })
        return {"selected_bits": min_bits}


def test_elasticbit_builds_analysis_and_runtime_matrix(monkeypatch):
    from mlb_studio import elasticbit_runtime

    monkeypatch.setattr(
        elasticbit_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeElasticBit,
    )

    weights = torch.randn(3, 4)
    calibration = torch.randn(8, 4)

    session = build_elasticbit_runtime(
        weights,
        calibration,
        threshold=0.02,
        runtime_mode="fast",
        min_bits=6,
        max_bits=24,
    )

    assert session.analysis == {"selected_bits": 6}
    assert session.runtime_mode == "fast"
    assert session.min_bits == 6
    assert session.max_bits == 24

    matrix_call = _FakeRuntimeMatrix.calls[-1]
    assert matrix_call["weights_shape"] == (3, 4)
    assert matrix_call["calibration_shape"] == (8, 4)
    assert matrix_call["threshold"] == 0.02
    assert matrix_call["runtime_mode"] == "fast"


def test_elasticbit_runtime_session_executes_forward(monkeypatch):
    from mlb_studio import elasticbit_runtime

    monkeypatch.setattr(
        elasticbit_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeElasticBit,
    )

    session = build_elasticbit_runtime(
        torch.ones(2, 4),
        torch.ones(3, 4),
    )

    result = session.forward(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    assert result.shape == (1,)
    assert float(result[0]) == pytest.approx(10.0)
    assert session.matrix.forward_calls[-1].dtype == np.float32


def test_elasticbit_reports_native_runtime_unavailable(monkeypatch):
    from mlb_studio import elasticbit_runtime

    class _Unavailable(_FakeElasticBit):
        @staticmethod
        def native_runtime_available():
            return False

    monkeypatch.setattr(
        elasticbit_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _Unavailable,
    )

    with pytest.raises(RuntimeError, match="native runtime is unavailable"):
        build_elasticbit_runtime(
            torch.ones(2, 4),
            torch.ones(3, 4),
        )


def test_elasticbit_validates_shapes_modes_and_bit_range(monkeypatch):
    from mlb_studio import elasticbit_runtime

    monkeypatch.setattr(
        elasticbit_runtime.IMPORT_POOL,
        "resolve_component",
        lambda key: _FakeElasticBit,
    )

    with pytest.raises(ValueError, match="weights must be 2D"):
        build_elasticbit_runtime(torch.ones(4), torch.ones(2, 4))

    with pytest.raises(ValueError, match="calibration width"):
        build_elasticbit_runtime(torch.ones(2, 4), torch.ones(2, 5))

    with pytest.raises(ValueError, match="runtime_mode"):
        build_elasticbit_runtime(
            torch.ones(2, 4),
            torch.ones(2, 4),
            runtime_mode="invalid",
        )

    with pytest.raises(ValueError, match="bit range"):
        build_elasticbit_runtime(
            torch.ones(2, 4),
            torch.ones(2, 4),
            min_bits=3,
            max_bits=33,
        )
