from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .import_pool import IMPORT_POOL


def _numpy_float32(value: Any, name: str, *, ndim: int):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "ElasticBit runtime preparation requires NumPy. "
            "Install MLB Studio with the data extra or install numpy."
        ) from exc

    if torch.is_tensor(value):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D.")
    return np.ascontiguousarray(array)


@dataclass
class ElasticBitRuntimeSession:
    analysis: Any
    matrix: Any
    threshold: float
    runtime_mode: str
    min_bits: int
    max_bits: int

    def forward(self, value: Any):
        input_array = _numpy_float32(value, "ElasticBit input", ndim=1)
        return self.matrix.forward(input_array)


def build_elasticbit_runtime(
    weights: Any,
    calibration: Any,
    *,
    threshold: float = 0.01,
    runtime_mode: str = "compact",
    min_bits: int = 4,
    max_bits: int = 32,
) -> ElasticBitRuntimeSession:
    """Build the post-training ElasticBit native runtime represented by Studio.

    ElasticBit is deliberately not a differentiable TensorGraph layer. Studio
    prepares it from a 2D weight matrix plus 2D calibration inputs and returns
    a runtime session whose ``forward`` method invokes RuntimeMatrix.forward().
    """

    threshold = float(threshold)
    min_bits = int(min_bits)
    max_bits = int(max_bits)
    runtime_mode = str(runtime_mode or "compact").strip().lower()

    if threshold < 0:
        raise ValueError("ElasticBit threshold must be >= 0.")
    if runtime_mode not in {"compact", "fast"}:
        raise ValueError("ElasticBit runtime_mode must be 'compact' or 'fast'.")
    if min_bits < 4 or max_bits > 32 or min_bits > max_bits:
        raise ValueError("ElasticBit bit range must satisfy 4 <= min_bits <= max_bits <= 32.")

    weights_array = _numpy_float32(weights, "ElasticBit weights", ndim=2)
    calibration_array = _numpy_float32(calibration, "ElasticBit calibration", ndim=2)

    if calibration_array.shape[1] != weights_array.shape[1]:
        raise ValueError(
            "ElasticBit calibration width must match the weight matrix input width."
        )

    ElasticBit = IMPORT_POOL.resolve_component("elasticbit_runtime")

    checker = getattr(ElasticBit, "native_runtime_available", None)
    if callable(checker) and not bool(checker()):
        raise RuntimeError(
            "ElasticBit native runtime is unavailable in this environment. "
            "Build/install the MLBricks ElasticBit CUDA runtime on a supported "
            "Linux + NVIDIA CUDA system."
        )

    analysis = ElasticBit.bitsAnaliser(
        weights_array,
        calibration_array,
        threshold,
        min_bits=min_bits,
        max_bits=max_bits,
    )
    matrix = ElasticBit.RuntimeMatrix.from_auto(
        weights_array,
        calibration_array,
        threshold,
        runtime_mode=runtime_mode,
        min_bits=min_bits,
        max_bits=max_bits,
    )

    return ElasticBitRuntimeSession(
        analysis=analysis,
        matrix=matrix,
        threshold=threshold,
        runtime_mode=runtime_mode,
        min_bits=min_bits,
        max_bits=max_bits,
    )
