import sys
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from pathlib import Path

from thor import ThorModelEncoder, CkksEngine

# Use THOR_FHE_MODE=gpu with the desilofhe-cu121 distribution on a CUDA host.
engine = CkksEngine(
    mode=os.environ.get("THOR_FHE_MODE", "gpu"),
    use_bootstrap_to_17_levels=True,
)

#Encode model
for dataset_type in ['mrpc']:
    model_dir = f"./finetuned_models/{dataset_type}/model.safetensors"
    encoder = ThorModelEncoder(engine, model_dir)
    out_dir = Path(f"encoded_models_new/{dataset_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("-" * 50)
    print(f"Encoding start for {dataset_type}")
    encoder.encode_pooler()
    encoder.save(str(out_dir / "pooler.pkl"))
    print(f"Encoding complete for {dataset_type} Pooler")

    print(f"Encoding start for {dataset_type} FF (per-layer files)")
    for layer in range(12):
        print(layer)
        encoder_ff = ThorModelEncoder(engine, model_dir)
        encoder_ff.encode_ff(layer)
        layer_prefix = f"bert.encoder.layer.{layer}."
        encoder_ff.weights_pt = {
            name: weight
            for name, weight in encoder_ff.weights_pt.items()
            if name.startswith(layer_prefix) and weight is not None
        }
        encoder_ff.save(str(out_dir / f"ff_layer_{layer}.pkl"))
    print(f"Encoding complete for {dataset_type} FF")

    print("-" * 50)
    encoder = ThorModelEncoder(engine, model_dir)
    print(f"Encoding start for {dataset_type} Attention")
    for layer in range(12):
        print(layer)
        encoder.encode_att(layer)
    encoder.save(str(out_dir / "att.pkl"))
    encoder.encode_cls()
    encoder.save(str(out_dir / "cls.pkl"))
    print(f"Encoding complete for {dataset_type}")
    print("-" * 50)
