from .gl import GLEngine
from .tensor import GLTensor, SUPPORTED_GL_SHAPES

__all__ = [
    "BertGLLayout",
    "GLEngine",
    "GLTensor",
    "SUPPORTED_GL_SHAPES",
    "ThorBert",
    "ThorBertAttention",
    "ThorBertClassifier",
    "ThorDataEncryptor",
    "ThorBertFF",
    "ThorBertPooler",
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
    if name in {
        "BertGLLayout",
        "ThorBert",
        "ThorBertAttention",
        "ThorBertClassifier",
        "ThorBertFF",
        "ThorBertPooler",
    }:
        from .bert import (
            BertGLLayout,
            ThorBert,
            ThorBertAttention,
            ThorBertClassifier,
            ThorBertFF,
            ThorBertPooler,
        )

        return {
            "BertGLLayout": BertGLLayout,
            "ThorBert": ThorBert,
            "ThorBertAttention": ThorBertAttention,
            "ThorBertClassifier": ThorBertClassifier,
            "ThorBertFF": ThorBertFF,
            "ThorBertPooler": ThorBertPooler,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
