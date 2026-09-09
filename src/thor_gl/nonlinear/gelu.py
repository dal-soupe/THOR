from __future__ import annotations

import numpy as np

from ..gl import FheData, GLEngine
from .polynomial import evaluate_poly_deg4


# Fourth-order Taylor approximation of x * Phi(x) around zero.
_GELU_COEFFICIENTS = np.array(
    [0.0, 0.5, 0.3989422804014327, 0.0, -0.06649038006690546]
)


def _gelu_one(engine: GLEngine, value: FheData) -> FheData:
    return evaluate_poly_deg4(engine, _GELU_COEFFICIENTS, value)


def he_gelu(engine: GLEngine, x, sk=None, initial_btp=False):
    """Evaluate GELU with a degree-four, bootstrap-free approximation."""
    del sk, initial_btp
    # x = engine.bootstrap(x)  # GL bootstrapping is intentionally disabled.
    if isinstance(x, np.ndarray):
        result = np.empty_like(x, dtype=object)
        for index in np.ndindex(x.shape):
            result[index] = _gelu_one(engine, x[index])
        return result
    return _gelu_one(engine, x)


def he_tanh_single(engine: GLEngine, x: FheData) -> FheData:
    """Compatibility helper using the cubic tanh approximation."""
    return evaluate_poly_deg4(
        engine, [0.0, 1.0, 0.0, -1.0 / 3.0], x
    )


__all__ = ["he_gelu", "he_tanh_single"]
