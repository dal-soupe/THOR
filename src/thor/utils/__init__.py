__all__ = ["ThorDataEncryptor", "inspect_model_weights", "load_model"]


def __getattr__(name):
    if name == "ThorDataEncryptor":
        from ..data import ThorDataEncryptor

        return ThorDataEncryptor
    if name in {"inspect_model_weights", "load_model"}:
        from .model import inspect_model_weights, load_model

        return {
            "inspect_model_weights": inspect_model_weights,
            "load_model": load_model,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
