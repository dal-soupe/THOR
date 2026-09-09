"""Small test for GL BERT evaluator.

uses identity block weights so it runs without downloading
the model or GLUE data. Replace make_weights with encoded model weights
when running a real BERT checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from thor_gl import GLEngine, GLTensor, ThorBert, ThorLinearEvaluator


def gpu_memory():
    """
    Queries the amount of memory currently used by each NVIDIA GPU.
    @return: List of used GPU memory values in MiB.
    """
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [int(x.strip()) for x in output.splitlines()]

def print_gpu_memory(label):
    """
    Prints the current memory usage for every NVIDIA GPU.
    @param label: Description of the inference stage being measured.
    @return: None.
    """
    print(f"[GPU] {label}: {gpu_memory()} MiB", flush=True)


SHAPE = (256, 64, 64)
LAYER_PREFIX = "bert.encoder.layer.0"


def make_weights(engine: GLEngine) -> dict[str, object]:
    identity = np.broadcast_to(0.05 * np.eye(64), SHAPE).copy()
    zeros = np.zeros(SHAPE)
    ones = np.ones(SHAPE)

    def encode(values: np.ndarray):
        return engine.encode(values, level=engine.max_level)

    return {
        f"{LAYER_PREFIX}.attention.self.query.weight": encode(identity),
        f"{LAYER_PREFIX}.attention.self.query.bias": encode(zeros),
        f"{LAYER_PREFIX}.attention.self.key.weight": encode(identity),
        f"{LAYER_PREFIX}.attention.self.key.bias": encode(zeros),
        f"{LAYER_PREFIX}.attention.self.value.weight": encode(identity),
        f"{LAYER_PREFIX}.attention.self.value.bias": encode(zeros),
        f"{LAYER_PREFIX}.attention.output.dense.weight": encode(identity),
        f"{LAYER_PREFIX}.attention.output.dense.bias": encode(zeros),
        f"{LAYER_PREFIX}.attention.output.LayerNorm.weight": encode(ones),
        f"{LAYER_PREFIX}.attention.output.LayerNorm.bias": encode(zeros),
        f"{LAYER_PREFIX}.intermediate.dense.weight": encode(identity),
        f"{LAYER_PREFIX}.intermediate.dense.bias": encode(zeros),
        f"{LAYER_PREFIX}.output.dense.weight": encode(identity),
        f"{LAYER_PREFIX}.output.dense.bias": encode(zeros),
        f"{LAYER_PREFIX}.output.LayerNorm.weight": encode(ones),
        f"{LAYER_PREFIX}.output.LayerNorm.bias": encode(zeros),
    }
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=int, default=15)
    parser.add_argument(
        "--rotation-key-level",
        type=int,
        default=0,
        help="Lower levels produce substantially smaller GL rotation keys",
    )
    args = parser.parse_args()

    engine = GLEngine(shape=SHAPE, mode=args.mode)
    print_gpu_memory("After creating GLEngine")
    secret_key = engine.create_secret_key()
    print_gpu_memory("After creating secret key")
    engine.sk = secret_key
    engine.add_mmult_key(engine.create_matrix_multiplication_key(secret_key))
    print_gpu_memory("After creating matrix multiplication key")
    engine.add_hmult_key(engine.create_hadamard_multiplication_key(secret_key))
    print_gpu_memory("After creating hadamard multiplication key")
    engine.add_rot_key(
        engine.create_rotation_key(secret_key, args.rotation_key_level)
    )
    print_gpu_memory("After creating rotation key")

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
    )
    output = model.ffs[0].forward(hidden, debug=True, sk=secret_key)
    decoded = output.to_logical(engine.decrypt(output.payload, secret_key))

    print(f"output shape: {decoded.shape}")
    print(f"output level: {output.payload.level}")
    print(f"finite values: {np.isfinite(decoded.real).all()}")
    print(f"mean/std: {decoded.real.mean():.6f}/{decoded.real.std():.6f}")


if __name__ == "__main__":
    main()
