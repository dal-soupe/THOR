# THOR

THOR runs BERT inference with CKKS homomorphic encryption using
[DesiloFHE](https://fhe.desilo.dev/latest/).

## Data files

The following directories are not stored in Git because of their size:

- `datasets/`
- `encoded_models_new/`
- `finetuned_models/`
- `keys/`

Download them from the
[THOR Google Drive folder](https://drive.google.com/drive/folders/1mWBkNdsu3JCQPrSuedyeN_3WJD7h-6RO)
and place the directories in the repository root. Existing Liberate keys and
encoded plaintexts are not compatible with DesiloFHE. Regenerate keys with the
key cell in `forward.ipynb` and regenerate encoded models with `encode.py`.

## Installation

DesiloFHE supports Python 3.10 through 3.14. For CPU execution:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The CPU and CUDA distributions both provide the same `desilofhe` module, so
install one backend at a time. For CUDA 12.1, replace the CPU distribution:

```bash
pip uninstall -y desilofhe
pip install desilofhe-cu121
```

Create the engine with the matching mode:

```python
from thor import CkksEngine

cpu_engine = CkksEngine(mode="cpu", use_bootstrap_to_17_levels=True)
gpu_engine = CkksEngine(mode="gpu", use_bootstrap_to_17_levels=True)
```

`encode.py` uses CPU mode by default. Set `THOR_FHE_MODE=gpu` when the CUDA
distribution is installed:

```bash
THOR_FHE_MODE=gpu python encode.py
```

## Running inference

Run `forward.ipynb`. Its key setup uses DesiloFHE's typed key serialization and
creates keys under `keys/desilofhe/` when they do not exist. Bootstrap-key
generation is expensive and the resulting key is large, so retain that
directory between runs.
