from __future__ import annotations

import numpy as np

from ..gl import FheData, GLEngine
from .polynomial import evaluate_poly_deg4


_TANH_COEFFICIENTS = np.array([0.0, 1.0, 0.0, -1.0 / 3.0])


def he_tanh(engine: GLEngine, x, min_x=-1.0, max_x=1.0, scale=1.0):
    """Evaluate tanh elementwise with a degree-3 approximation."""
    del min_x, max_x
    if scale <= 0:
        raise ValueError("scale must be positive")
    if isinstance(x, np.ndarray):
        result = np.empty_like(x, dtype=object)
        for index in np.ndindex(x.shape):
            result[index] = he_tanh_single(engine, x[index], scale=scale)
        return result
    return he_tanh_single(engine, x, scale=scale)


def he_tanh_single(
    engine: GLEngine,
    x: FheData,
    min_x=-1.0,
    max_x=1.0,
    scale=1.0,
) -> FheData:
    del min_x, max_x
    normalized = x if scale == 1 else engine.mult_scalar(x, 1.0 / scale)
    # normalized = engine.bootstrap(normalized)  # GL bootstrap is unavailable.
    result = evaluate_poly_deg4(
        engine, _TANH_COEFFICIENTS, normalized
    )
    return result if scale == 1 else engine.mult_scalar(result, scale)


__all__ = ["he_tanh", "he_tanh_single"]
