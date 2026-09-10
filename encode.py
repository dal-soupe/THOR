"""Encode BERT model weights for CKKS or GL inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("ckks", "gl"),
        default="ckks",
        help="Encryption backend used for encoding (default: ckks)",
    )
    parser.add_argument(
        "--dataset-type",
        nargs="+",
        default=["mrpc"],
        help="Fine-tuned model subdirectories to encode",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("~/data/THOR/"),
        help="Root directory containing <dataset-type>/model.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination; defaults to <data-dir>/encoded_models_<backend>",
    )
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu"),
        default=None,
        help="Desilo execution mode (default: THOR_FHE_MODE or backend default)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device used by the GL backend",
    )
    parser.add_argument(
        "--all-ff",
        action="store_true",
        help="Write one combined FF file instead of one file per layer",
    )
    return parser.parse_args()


def encode_ckks(args: argparse.Namespace, output_dir: Path) -> None:
    from thor import CkksEngine, ThorModelEncoder

    mode = args.mode or os.environ.get("THOR_FHE_MODE", "gpu")
    engine = CkksEngine(mode=mode, use_bootstrap_to_17_levels=True)
    data_dir = args.data_dir.expanduser()
    for dataset_type in args.dataset_type:
        model_path = data_dir / dataset_type / "model.safetensors"
        out_dir = output_dir / dataset_type
        out_dir.mkdir(parents=True, exist_ok=True)
        encoder = ThorModelEncoder(engine, str(model_path))

        print(f"Encoding CKKS model: {dataset_type}")
        encoder.encode_pooler()
        encoder.save(out_dir / "pooler.pkl")

        if args.all_ff:
            encoder = ThorModelEncoder(engine, str(model_path))
            for layer in range(12):
                encoder.encode_ff(layer)
            encoder.save(out_dir / "ff.pkl")
        else:
            for layer in range(12):
                encoder = ThorModelEncoder(engine, str(model_path))
                encoder.encode_ff(layer)
                prefix = f"bert.encoder.layer.{layer}."
                encoder.weights_pt = {
                    name: weight
                    for name, weight in encoder.weights_pt.items()
                    if name.startswith(prefix) and weight is not None
                }
                encoder.save(out_dir / f"ff_layer_{layer}.pkl")

        encoder = ThorModelEncoder(engine, str(model_path))
        for layer in range(12):
            encoder.encode_att(layer)
        encoder.save(out_dir / "att.pkl")
        encoder.encode_cls()
        encoder.save(out_dir / "cls.pkl")


def encode_gl(args: argparse.Namespace, output_dir: Path) -> None:
    from thor_gl import GLEngine, ThorModelEncoder

    mode = args.mode or os.environ.get("THOR_FHE_MODE", "cpu")
    engine = GLEngine(
        shape=(256, 64, 64),
        mode=mode,
        device_id=args.device_id,
    )
    data_dir = args.data_dir.expanduser()
    for dataset_type in args.dataset_type:
        model_path = data_dir / "finetuned_models" / dataset_type / "model.safetensors"
        out_dir = output_dir / dataset_type
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Encoding GL model: {dataset_type}")
        encoder = ThorModelEncoder(engine, str(model_path))
        encoder.encode_pooler()
        encoder.save(out_dir / "pooler.pkl")

        if args.all_ff:
            encoder = ThorModelEncoder(engine, str(model_path))
            for layer in range(12):
                encoder.encode_ff(layer)
            encoder.save(out_dir / "ff.pkl")
        else:
            for layer in range(12):
                encoder = ThorModelEncoder(engine, str(model_path))
                encoder.encode_ff(layer)
                prefix = f"bert.encoder.layer.{layer}."
                encoder.weights_pt = {
                    name: weight
                    for name, weight in encoder.weights_pt.items()
                    if name.startswith(prefix) and weight is not None
                }
                encoder.save(out_dir / f"ff_layer_{layer}.pkl")

        encoder = ThorModelEncoder(engine, str(model_path))
        for layer in range(12):
            encoder.encode_att(layer)
        encoder.save(out_dir / "att.pkl")
        encoder.encode_cls()
        encoder.save(out_dir / "cls.pkl")


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser()
    elif args.backend == "gl":
        output_dir = Path("~/data/THOR/encoded_models_gl").expanduser()
    else:
        output_dir = Path("encoded_models_split17_new")

    if args.backend == "gl":
        encode_gl(args, output_dir)
    else:
        encode_ckks(args, output_dir)


if __name__ == "__main__":
    main()
