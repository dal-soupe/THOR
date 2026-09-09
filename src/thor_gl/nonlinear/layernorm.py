from __future__ import annotations

import math

import numpy as np

from ..gl import FheData, GLEngine
from .polynomial import evaluate_poly_deg4


# Third-order Taylor approximation to 1/sqrt(x) around x=1.
_INV_SQRT_COEFFICIENTS = np.array(
    [2.1875, -2.1875, 1.3125, -0.3125]
)


def _payload(value):
    return getattr(value, "payload", value)


def _sum_last_axis(engine: GLEngine, value: FheData) -> FheData:
    width = int(engine.shape[-1])
    if width <= 0 or width & (width - 1):
        raise ValueError("the last GL dimension must be a positive power of two")
    result = value
    shift = 1
    while shift < width:
        result = engine.cc_add(
            result, engine.rotate_left(result, shift, axis=2)
        )
        shift *= 2
    return result


def he_layernorm1(
    engine: GLEngine,
    x,
    gamma,
    beta,
    var_e=1e-5,
    min_var=0.15,
    max_var=10.0,
    debug=False,
    sk=None,
):
    return he_layernorm(
        engine, x, gamma, beta, var_e, min_var, max_var, debug=debug, sk=sk
    )


def he_layernorm2(
    engine: GLEngine,
    x,
    gamma,
    beta,
    var_e=1e-5,
    min_var=0.2,
    max_var=150.0,
    debug=False,
    sk=None,
):
    return he_layernorm(
        engine, x, gamma, beta, var_e, min_var, max_var, debug=debug, sk=sk
    )


def he_layernorm3(
    engine: GLEngine,
    x,
    gamma,
    beta,
    var_e=1e-5,
    min_var=0.75,
    max_var=2500.0,
    debug=False,
    sk=None,
):
    return he_layernorm(
        engine, x, gamma, beta, var_e, min_var, max_var, debug=debug, sk=sk
    )


def he_layernorm(
    engine: GLEngine,
    x,
    gamma,
    beta,
    var_e,
    min_var,
    max_var,
    n=768,
    debug=False,
    sk=None,
):
    """Normalize each row of every physical GL matrix."""
    del n
    if debug and sk is None:
        raise ValueError("sk must be provided for debug mode")
    if min_var <= 0 or max_var <= min_var:
        raise ValueError("variance bounds must satisfy 0 < min_var < max_var")

    value = _payload(x)
    width = int(engine.shape[-1])
    mean = engine.mult_scalar(_sum_last_axis(engine, value), 1.0 / width)
    value, mean = engine.auto_level(value, mean)
    centered = engine.cc_sub(value, mean)

    squared = engine.square_elts(centered)
    variance = engine.mult_scalar(
        _sum_last_axis(engine, squared), 1.0 / width
    )
    variance = engine.add_scalar(variance, var_e)

    reference_variance = math.sqrt(min_var * max_var)
    normalized_variance = engine.mult_scalar(
        variance, 1.0 / reference_variance
    )
    inverse_std = he_invsqrt(
        engine,
        normalized_variance,
        reference_variance=reference_variance,
    )
    # inverse_std = engine.bootstrap(inverse_std)  # GL bootstrap disabled.

    normalized = engine.auto_ct_ct_hmult(centered, inverse_std)
    scaled = engine.hmult(normalized, _payload(gamma))
    return engine.pc_add(_payload(beta), scaled)


def he_invsqrt(
    engine: GLEngine,
    numerator: FheData,
    denominator: FheData | None = None,
    e=None,
    alpha=None,
    mask=None,
    *,
    reference_variance: float = 1.0,
):
    """Evaluate a degree-three inverse-square-root approximation."""
    del e, alpha, mask
    value = numerator if denominator is None else denominator
    result = evaluate_poly_deg4(
        engine, _INV_SQRT_COEFFICIENTS, value
    )
    return engine.mult_scalar(result, 1.0 / math.sqrt(reference_variance))


__all__ = [
    "he_invsqrt",
    "he_layernorm",
    "he_layernorm1",
    "he_layernorm2",
    "he_layernorm3",
]
