"""Run one encrypted CKKS BERT encoder layer.

The script uses the encoded model and dataset layout consumed by forward.py,
but does not create plots or use a custom forward-layer implementation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-type", default="mrpc")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Dataset saved with datasets.save_to_disk (default: datasets/<type>)",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path("encoded_models_split17_new"),
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--checkpoint", default="bert-base-uncased")
    args = parser.parse_args()

    # These imports are intentionally lazy so --help works without the ML stack.
    import torch
    from transformers import BertForNextSentencePrediction

    from thor import CkksEngine, ThorDataEncryptor, ThorLinearEvaluator
    from thor.bert import ThorBertAttention, ThorBertFF

    mode = args.mode or os.environ.get("THOR_FHE_MODE", "cpu")
    engine = CkksEngine(
        mode=mode,
        device_id=args.device_id,
        use_bootstrap_to_17_levels=True,
    )

    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    engine.add_pk(public_key)
    engine.add_evk(engine.create_relinearization_key(secret_key))
    engine.add_gk(engine.create_rotation_key(secret_key))
    engine.add_conj_key(engine.create_conjugation_key(secret_key))
    engine.add_bs_key(engine.create_bootstrap_key(secret_key))

    dataset_path = args.dataset_path or Path("datasets") / args.dataset_type
    embedding_model = BertForNextSentencePrediction.from_pretrained(
        args.checkpoint
    ).bert.embeddings
    data = ThorDataEncryptor(
        args.dataset_type,
        str(dataset_path),
        embedding_model=embedding_model,
        ckks_engine=engine,
        test=True,
    )
    batch = None
    for sample_index, candidate in enumerate(data.eval_dataloader):
        if sample_index == args.sample:
            batch = candidate
            break
    if batch is None:
        raise SystemExit(f"Validation split has no sample at index {args.sample}")

    inputs = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "token_type_ids"}
    }
    with torch.no_grad():
        embedding = data.embed_data(inputs)
    encrypted_hidden = data.encrypt_embedding(embedding, public_key, level=9)
    attention_mask = data.encode_attention_mask(
        batch["attention_mask"].numpy().squeeze(), level=13
    )

    model_dir = args.encoded_dir / args.dataset_type
    attention_weights = engine.load_plaintext_weights(model_dir / "att.pkl")
    ff_path = model_dir / f"ff_layer_{args.layer}.pkl"
    if not ff_path.exists():
        ff_path = model_dir / "ff.pkl"
    ff_weights = engine.load_plaintext_weights(ff_path)

    evaluator = ThorLinearEvaluator(engine)
    attention = ThorBertAttention(evaluator, attention_weights, args.layer)
    feed_forward = ThorBertFF(evaluator, ff_weights, args.layer)
    hidden = attention.forward(encrypted_hidden, attention_mask)
    output = feed_forward.forward(hidden)

    values = np.concatenate(
        [np.asarray(engine.decrypt(value, secret_key)).real[:8] for value in output]
    )
    print(f"dataset: {args.dataset_type}; layer: {args.layer}; sample: {args.sample}")
    print(f"output ciphertexts: {len(output)}")
    print(f"output level: {output[0].level}")
    print(f"finite values: {np.isfinite(values).all()}")
    print(f"mean/std (first slots): {values.mean():.6f}/{values.std():.6f}")


if __name__ == "__main__":
    main()
