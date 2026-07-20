from .ckks import CkksEngine

__all__ = [
    "CkksEngine",
    "ThorBert",
    "ThorDataEncryptor",
    "ThorLinearEvaluator",
    "ThorModelEncoder",
]


def __getattr__(name):
    if name == "ThorModelEncoder":
        from .model_encoder import ThorModelEncoder

        return ThorModelEncoder
    if name == "ThorLinearEvaluator":
        from .linear import ThorLinearEvaluator

        return ThorLinearEvaluator
    if name == "ThorDataEncryptor":
        from .data import ThorDataEncryptor

        return ThorDataEncryptor
    if name == "ThorBert":
        from .bert import ThorBert

        return ThorBert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
