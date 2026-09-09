from __future__ import annotations

import math
import numpy as np

from ..gl import FheData, GLEngine
from .polynomial import evaluate_poly_deg4


_EXP_COEFFICIENTS = np.array([1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0])
# Cubic geometric-series approximation to 1/x around x=1.
_RECIPROCAL_COEFFICIENTS = np.array([4.0, -6.0, 4.0, -1.0])


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


def he_softmax1(
    engine: GLEngine,
    x,
    attention_mask,
    rescale=False,
    debug=False,
    sk=None,
):
    return he_softmax(
        engine,
        x,
        attention_mask,
        rescale=rescale,
        min_x=-27.2493,
        max_x=21.72692,
        n=1,
        l=1,
        inv_epsilon=2**-11,
        output_alpha=0.01,
        debug=debug,
        sk=sk,
    )


def he_softmax2(
    engine: GLEngine,
    x,
    attention_mask,
    rescale=False,
    debug=False,
    sk=None,
):
    return he_softmax(
        engine,
        x,
        attention_mask,
        rescale=rescale,
        min_x=-70.0,
        max_x=70.0,
        n=1,
        l=1,
        inv_epsilon=2**-18,
        output_alpha=0.01,
        debug=debug,
        sk=sk,
    )


def he_softmax(
    engine: GLEngine,
    u,
    attention_mask,
    rescale,
    min_x,
    max_x,
    n,
    l,
    inv_epsilon,
    output_alpha,
    debug,
    sk,
):
    """Approximate row-wise softmax within every physical GL matrix."""
    del rescale, min_x, max_x, n, l, inv_epsilon, output_alpha
    if debug and sk is None:
        raise ValueError("sk must be provided for debug mode")

    scores = _payload(u)
    mask = _payload(attention_mask)
    exp_scores = evaluate_poly_deg4(
        engine, _EXP_COEFFICIENTS, scores
    )
    exp_scores = engine.hmult(exp_scores, mask)

    width = int(engine.shape[-1])
    denominator = _sum_last_axis(engine, exp_scores)
    normalized_denominator = engine.mult_scalar(denominator, 1.0 / width)
    reciprocal = evaluate_poly_deg4(
        engine, _RECIPROCAL_COEFFICIENTS, normalized_denominator
    )
    probabilities = engine.auto_ct_ct_hmult(exp_scores, reciprocal)
    # probabilities = engine.bootstrap(probabilities)  # GL bootstrap disabled.
    return engine.mult_scalar(probabilities, 1.0 / width)


def he_inv(
    engine: GLEngine,
    numerator: FheData,
    denominator: FheData,
    epsilon: float,
    alpha: float,
    delta=1,
):
    """Compatibility inverse using a degree-three approximation near one."""
    del epsilon, alpha
    reciprocal = evaluate_poly_deg4(
        engine, _RECIPROCAL_COEFFICIENTS, denominator
    )
    result = engine.auto_ct_ct_hmult(numerator, reciprocal)
    return result, delta, 0.0


def he_exp1(engine: GLEngine, enc_x, min_x, max_x, n):
    del min_x, max_x
    result = evaluate_poly_deg4(
        engine, _EXP_COEFFICIENTS, enc_x
    )
    for _ in range(int(math.log2(n)) if n > 1 else 0):
        result = engine.square_elts(result)
    return result


def he_exp2(engine: GLEngine, enc_x, min_x, max_x, n):
    return he_exp1(engine, enc_x, min_x, max_x, n)


class Ciphertext:
    def __init__(self, ciphertext: FheData, delta: float):
        self.ciphertext = ciphertext
        self.delta = delta


__all__ = [
    "Ciphertext",
    "he_exp1",
    "he_exp2",
    "he_inv",
    "he_softmax",
    "he_softmax1",
    "he_softmax2",
]
