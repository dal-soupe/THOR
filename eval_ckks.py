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
import torch
from transformers import BertForNextSentencePrediction

from thor import CkksEngine, ThorDataEncryptor, ThorLinearEvaluator
from thor.bert import ThorBertAttention, ThorBertFF

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-type", default="mrpc")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Dataset saved with datasets.save_to_disk (default: ~/data/THOR/datasets/<type>)",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path("encoded_models_split17_new"),
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of token rows to print (default: 128)",
    )
    parser.add_argument("--mode", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--checkpoint", default="bert-base-uncased")
    args = parser.parse_args()

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

    dataset_path = args.dataset_path or Path("~/data/THOR/datasets") / args.dataset_type
    embedding_model = BertForNextSentencePrediction.from_pretrained(
        args.checkpoint
    ).bert.embeddings
    data = ThorDataEncryptor(
        args.dataset_type,
        str(dataset_path),
        embedding_model=embedding_model,
        ckks_engine=engine,
        test=False,
    )
    batch = None
    for sample_index, candidate in enumerate(data.eval_dataloader):
        if sample_index == args.sample:
            batch = candidate
            break
    if batch is None:
        raise SystemExit(f"Validation split has no sample at index {args.sample}")
    raw_sample = data.dataset["validation"][args.sample]

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

    decoded_output = [
        np.asarray(engine.decrypt(value, secret_key)).real for value in output
    ]
    values = np.concatenate([decoded[:8] for decoded in decoded_output])

    print("\nInput example")
    if "sentence1" in raw_sample:
        print(f"sentence1: {raw_sample['sentence1']}")
        print(f"sentence2: {raw_sample['sentence2']}")
    else:
        print(f"sentence: {raw_sample['sentence']}")
    print(f"label: {raw_sample['label']}")

    token_ids = batch["input_ids"][0].tolist()
    token_mask = batch["attention_mask"][0].tolist()
    tokens = data.tokenizer.convert_ids_to_tokens(token_ids)
    print("\nTokens and representative decoded output slots")
    print(
        "Each row uses packed slot index*16; these are hidden-state values, "
        "not decoded text."
    )
    print("index token id mask packed-slot output[0:8]")
    for index, (token, token_id, mask_value) in enumerate(
        zip(tokens, token_ids, token_mask)
    ):
        if index >= args.max_tokens:
            break
        slot = index * 16
        packed_values = [decoded[slot] for decoded in decoded_output]
        print(
            f"{index:5d} {token:>16s} {token_id:6d} {mask_value:4d} "
            f"{np.asarray(packed_values)}"
        )
    print(f"dataset: {args.dataset_type}; layer: {args.layer}; sample: {args.sample}")
    print(f"output ciphertexts: {len(output)}")
    print(f"output level: {output[0].level}")
    print(f"finite values: {np.isfinite(values).all()}")
    print(f"mean/std (first slots): {values.mean():.6f}/{values.std():.6f}")


if __name__ == "__main__":
    main()
