import sys
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), './src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from thor import ThorModelEncoder, CkksEngine

# Use THOR_FHE_MODE=gpu with the desilofhe-cu121 distribution on a CUDA host.
engine = CkksEngine(
    mode=os.environ.get("THOR_FHE_MODE", "cpu"),
    use_bootstrap_to_17_levels=True,
)

#Encode model
for dataset_type in ['mrpc']:
    model_dir = f"./finetuned_models/{dataset_type}/model.safetensors"
    encoder = ThorModelEncoder(engine, model_dir)
    print("-" * 50)
    print(f"Encoding start for {dataset_type}")
    encoder.encode_pooler()
    encoder.save(f"encoded_models_new/{dataset_type}/pooler.pkl")
    print(f"Encoding complete for {dataset_type} Pooler")
    for layer in range(12):
        print(layer)
        encoder.encode_ff(layer)
    encoder.save(f"encoded_models_new/{dataset_type}/ff.pkl")
    print(f"Encoding complete for {dataset_type} FF")
    print("-" * 50)
    encoder = ThorModelEncoder(engine, model_dir)
    print(f"Encoding start for {dataset_type} Attention")
    for layer in range(12):
        print(layer)
        encoder.encode_att(layer)
    encoder.save(f"encoded_models_new/{dataset_type}/att.pkl")
    encoder.encode_cls()
    encoder.save(f"encoded_models_new/{dataset_type}/cls.pkl")
    print(f"Encoding complete for {dataset_type}")
    print("-" * 50)
