"""Small test for GL BERT evaluator.

uses identity block weights so it runs without downloading
the model or GLUE data. Replace make_weights with encoded model weights
when running a real BERT checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from thor_gl import GLEngine, GLTensor, ThorBert, ThorLinearEvaluator


SHAPE = (256, 64, 64)
LAYER_PREFIX = "bert.encoder.layer.0"


def make_weights(engine: GLEngine) -> dict[str, object]:
    identity = np.broadcast_to(0.05 * np.eye(64), SHAPE).copy()
    zeros = np.zeros(SHAPE)
    ones = np.ones(SHAPE)

    def encode(values: np.ndarray):
        return engine.encode(values, level=engine.max_level)

    return {
        f"{LAYER_PREFIX}.intermediate.dense.weight": encode(identity),
        f"{LAYER_PREFIX}.intermediate.dense.bias": encode(zeros),
        f"{LAYER_PREFIX}.output.dense.weight": encode(identity),
        f"{LAYER_PREFIX}.output.dense.bias": encode(zeros),
        f"{LAYER_PREFIX}.output.LayerNorm.weight": encode(ones),
        f"{LAYER_PREFIX}.output.LayerNorm.bias": encode(zeros),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=int, default=15)
    args = parser.parse_args()

    engine = GLEngine(shape=SHAPE, mode=args.mode)
    secret_key = engine.create_secret_key()
    engine.sk = secret_key
    engine.add_mmult_key(engine.create_matrix_multiplication_key(secret_key))
    engine.add_hmult_key(engine.create_hadamard_multiplication_key(secret_key))
    engine.add_rot_key(engine.create_rotation_key(secret_key))

    rng = np.random.default_rng(args.seed)
    values = np.zeros(SHAPE)
    values.reshape(-1)[: np.prod((1, 128, 768))] = rng.normal(
        0.0, 0.1, np.prod((1, 128, 768))
    )
    encrypted = engine.encode_and_encrypt(values, secret_key, level=args.level)
    hidden = GLTensor(encrypted, (1, 128, 768), SHAPE)

    model = ThorBert(
        ThorLinearEvaluator(engine),
        make_weights(engine),
        max_layer_batch=1,
        n_layers=1,
        debug=True,
    )
    output = model.ffs[0].forward(hidden)
    decoded = output.to_logical(engine.decrypt(output.payload, secret_key))

    print(f"output shape: {decoded.shape}")
    print(f"output level: {output.payload.level}")
    print(f"finite values: {np.isfinite(decoded.real).all()}")
    print(f"mean/std: {decoded.real.mean():.6f}/{decoded.real.std():.6f}")


if __name__ == "__main__":
    main()
