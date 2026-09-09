from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..gl import FheData, GLEngine


def evaluate_poly_deg4(
    engine: GLEngine,
    p: Sequence[float] | np.ndarray,
    x: FheData,
) -> FheData:
    """Evaluate an increasing-order polynomial of degree at most 4.

    The name is retained for API compatibility. The GL engine owns the
    multiplication schedule and consumes the configured Hadamard key.
    """
    coefficients = np.asarray(p, dtype=float).reshape(-1)
    if coefficients.size == 0:
        raise ValueError("p must contain at least one coefficient")
    
    if coefficients.size > 5:
        raise ValueError("GL polynomial approximations are limited to degree 4")
    if engine.hmult_key is None:
        raise ValueError("A hadamard multiplication key is required")
    return engine.evaluate_polynomial(x, coefficients, engine.hmult_key)


# Compatibility with the former CKKS-oriented helper name.
evaluate_polynomial_stockmeyer = evaluate_poly_deg4


__all__ = ["evaluate_poly_deg4", "evaluate_polynomial_stockmeyer"]
