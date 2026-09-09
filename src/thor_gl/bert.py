from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod
from typing import Any, Mapping
import numpy as np
from functools import partial
import time

try:
    import torch
except ImportError:  # Torch is optional for CPU-only GL use.
    torch = None

from .linear import ThorLinearEvaluator
from .tensor import GLTensor
from .nonlinear.gelu import he_gelu#, he_gelu2
from .nonlinear.softmax import he_softmax1, he_softmax2
from .nonlinear.layernorm import he_layernorm1, he_layernorm2, he_layernorm3
from .nonlinear.tanh import he_tanh


@dataclass(frozen=True, slots=True)
class BertGLLayout:
    """Logical BERT shapes mapped onto GL's deepest supported layout."""

    physical_shape: tuple[int, int, int] = (256, 64, 64)

    HIDDEN_SHAPE = (1, 128, 768)
    HEAD_SHAPE = (12, 128, 64)
    ATTENTION_SCORE_SHAPE = (12, 128, 128)
    ATTENTION_MASK_SHAPE = ATTENTION_SCORE_SHAPE
    INTERMEDIATE_SHAPE = (1, 128, 3072)
    PROJECTION_WEIGHT_SHAPE = (1, 768, 768)
    FEED_FORWARD_WEIGHT_SHAPE = (1, 768, 3072)

    def __post_init__(self) -> None:
        normalized = tuple(int(dimension) for dimension in self.physical_shape)
        object.__setattr__(self, "physical_shape", normalized)
        if normalized != (256, 64, 64):
            raise ValueError(
                "BERT GL layout requires physical shape (256, 64, 64) to "
                "provide 64x64 head blocks and 16 multiplication levels"
            )

    def require_shape(
        self,
        tensor: GLTensor,
        logical_shape: tuple[int, int, int],
        name: str,
    ) -> None:
        if not isinstance(tensor, GLTensor):
            raise TypeError(f"{name} must be a GLTensor")
        if tensor.physical_shape != self.physical_shape:
            raise ValueError(
                f"{name} physical shape {tensor.physical_shape} does not match "
                f"BERT shape {self.physical_shape}"
            )
        if tensor.logical_shape != logical_shape:
            raise ValueError(
                f"{name} logical shape {tensor.logical_shape} does not match "
                f"{logical_shape}"
            )

    def split_heads(self, hidden_state: GLTensor) -> GLTensor:
        """
        Rejects metadata-only head splitting for Desilo's contiguous GL packing.
        @param hidden_state: GL tensor with logical shape (1, 128, 768).
        @return: This method does not return until encrypted head permutation exists.
        """
        self.require_shape(hidden_state, self.HIDDEN_SHAPE, "hidden_state")
        raise NotImplementedError(
            "split_heads requires an encrypted permutation from token-major "
            "hidden data to head-major matrices"
        )

    def merge_heads(self, heads: GLTensor) -> GLTensor:
        """
        Rejects metadata-only head merging for Desilo's contiguous GL packing.
        @param heads: GL tensor with logical shape (12, 128, 64).
        @return: This method does not return until encrypted head permutation exists.
        """
        self.require_shape(heads, self.HEAD_SHAPE, "heads")
        raise NotImplementedError(
            "merge_heads requires an encrypted permutation from head-major "
            "matrices to token-major hidden data"
        )

    def block_count(self, logical_shape: tuple[int, int, int]) -> int:
        """
        Calculates the physical matrix slots occupied by contiguous logical data.
        @param logical_shape: Logical tensor shape packed by Desilo.
        @return: Number of physical square matrix slots containing logical values.
        """
        return ceil(prod(logical_shape) / prod(self.physical_shape[1:]))


class ThorModule:
    """Base class for GL BERT components"""

    def __init__(
        self,
        evaluator: ThorLinearEvaluator,
        weights: Mapping[str, Any],
    ) -> None:
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.weights = weights
        self._configured = bool(weights)
        self.devices = []

    def to(self, devices: list[int] | None = None) -> ThorModule:
        """GL objects remain owned by the mode/device selected by their engine."""
        self.devices = devices

    def cpu(self) -> ThorModule:
        """GL objects cannot be transferred between engine backends in place."""
        return self

    @staticmethod
    def _deferred(operation: str) -> None:
        raise NotImplementedError(
            f"{operation} requires the GL block-multiplication scheduler and "
            "bootstrap-free nonlinear approximations"
        )


class ThorBert:
    """Twelve-layer BERT evaluator for GL tensors."""

    n_layers = 12

    def __init__(
        self,
        evaluator: ThorLinearEvaluator,
        weights: Mapping[str, Any],
        max_layer_batch: int = 2,
        n_layers: int | None = None,
    ) -> None:
        if max_layer_batch <= 0:
            raise ValueError("max_layer_batch must be positive")
        if n_layers is not None and not 1 <= n_layers <= ThorBert.n_layers:
            raise ValueError(
                f"n_layers must be between 1 and {ThorBert.n_layers}"
            )
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.layout = BertGLLayout(tuple(int(value) for value in self.engine.shape))
        self.weights = weights
        self._configured = bool(weights)
        self.max_layer_batch = max_layer_batch
        self.active_layers = n_layers or ThorBert.n_layers
        self.attentions = [
            ThorBertAttention(evaluator, weights, layer)
            for layer in range(self.active_layers)
        ]
        self.ffs = [
            ThorBertFF(evaluator, weights, layer)
            for layer in range(self.active_layers)
        ]
        self.pooler = ThorBertPooler(evaluator, weights)
        self.classifier = ThorBertClassifier(evaluator, weights)
        self.devices = []

    def to(self, devices: list[int] | None = None) -> None:
        self.devices = devices

    def split_heads(self, hidden_state: GLTensor) -> GLTensor:
        return self.layout.split_heads(hidden_state)

    def merge_heads(self, heads: GLTensor) -> GLTensor:
        return self.layout.merge_heads(heads)

    def forward(
        self,
        x: GLTensor,
        attention_mask: GLTensor,
        devices: list[int] | None = None,
        debug: bool = False,
        sk: Any = None,
    ) -> GLTensor:
        self.layout.require_shape(x, self.layout.HIDDEN_SHAPE, "hidden_state")
        self.layout.require_shape(
            attention_mask,
            self.layout.ATTENTION_MASK_SHAPE,
            "attention_mask",
        )
        if not self._configured:
            ThorModule._deferred("BERT forward")
        if debug:
            if sk is None:
                raise ValueError("Please provide secret key")
        if devices == [] and self.engine.mode == "gpu":
            if self.devices == []:
                raise ValueError("Please set GPU devices")
            devices = self.devices
    
        start_time = time.time()
        # if x.shape != (4,):
        #     raise ValueError("Input of bert should be (8,)")
        
        for i in range(
            (self.active_layers + self.max_layer_batch - 1)
            // self.max_layer_batch
        ):
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                if j >= self.active_layers:
                    break
                self.attentions[j].to(devices)
                self.ffs[j].to(devices)
                if self.engine.mode == "gpu" and torch is not None:
                    print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3:.2f} GB after layer {j} allocation: ")
                
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                if j >= self.active_layers:
                    break
                print(f"Forwarding layer: {j}")
                temp = time.time()
                x = self.attentions[j].forward(x, attention_mask, debug=debug, sk=sk)
                x = self.ffs[j].forward(x, debug=debug, sk=sk)
                print(f"Time taken for layer {j} forward: {time.time() - temp:.2f} seconds")
            
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                if j >= self.active_layers:
                    break
                self.attentions[j].cpu()
                self.ffs[j].cpu()
                if self.engine.mode == "gpu" and torch is not None:
                    print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3:.2f} GB after layer {j} release: ")
                
        self.pooler.to(devices)
        self.classifier.to(devices)
        x = self.pooler.forward(x)
        x = self.classifier.forward(x)
        print(f"Total time taken: {time.time() - start_time}")
        return x


class ThorBertAttention(ThorModule):
    def __init__(
        self,
        evaluator: ThorLinearEvaluator,
        weights: Mapping[str, Any],
        layer_idx: int,
    ) -> None:
        super().__init__(evaluator, weights)
        if not 0 <= layer_idx < ThorBert.n_layers:
            raise ValueError(f"layer_idx must be between 0 and {ThorBert.n_layers - 1}")
        self.engine = evaluator.engine
        self.layout = BertGLLayout(tuple(int(value) for value in self.engine.shape))
        # Keys are owned by the caller/engine and must match the input
        # ciphertext's secret key. Component construction must not replace them.
        self.secret_key = getattr(self.engine, "sk", None)
        self.layer_idx = layer_idx
        self.weights = {}

        for qkv in ['query', 'key', 'value']:
            self.weights[f"{qkv}.weight"] = weights[f'bert.encoder.layer.{layer_idx}.attention.self.{qkv}.weight']
            self.weights[f"{qkv}.bias"] = weights[f'bert.encoder.layer.{layer_idx}.attention.self.{qkv}.bias']
        self.weights['dense.weight'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.dense.weight']
        self.weights['dense.bias'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.dense.bias']
        self.weights['LayerNorm.weight'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.weight']
        self.weights['LayerNorm.bias'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.bias']
        
        self.keys = ['query.weight', 'query.bias', 
                            'key.weight', 'key.bias', 
                            'value.weight', 'value.bias', 
                            'dense.weight', 'dense.bias',
                            'LayerNorm.weight', 'LayerNorm.bias']

        self.softmax = partial(he_softmax2, engine=self.engine) if layer_idx == 2 else partial(he_softmax1, engine=self.engine)
        self.layernorm = partial(
            he_layernorm1,
            engine=self.engine,
            gamma=self.weights['LayerNorm.weight'],
            beta=self.weights['LayerNorm.bias'],
        )
        self.devices = []

    def forward(self, x: GLTensor, attention_mask: GLTensor, debug=False, sk=None) -> GLTensor:
        self.layout.require_shape(x, self.layout.HIDDEN_SHAPE, "hidden_state")
        self.layout.require_shape(
            attention_mask,
            self.layout.ATTENTION_MASK_SHAPE,
            "attention_mask",
        )
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        att_score = self.calculate_attention_score(q, k)
        att_prob = GLTensor(
            self.softmax(
                x=att_score.payload,
                attention_mask=attention_mask.payload,
                rescale=False,
                debug=debug,
                sk=sk,
            ),
            self.layout.ATTENTION_SCORE_SHAPE,
            self.layout.physical_shape,
        )

        att_context = self.calculate_attention_context(att_prob, v)
        dense_output = self.dense(att_context)

        residual, projected = self.engine.auto_level(
            x.payload, dense_output.payload
        )
        x=self.engine.cc_add(residual, projected)
        normalized = self.layernorm(x=x, debug=debug, sk=sk)
        return GLTensor(
            normalized, self.layout.HIDDEN_SHAPE, self.layout.physical_shape
        )

    def query(self, hidden_state: GLTensor) -> GLTensor:
        self.layout.require_shape(
            hidden_state, self.layout.HIDDEN_SHAPE, "hidden_state"
        )
        wx = self.engine.mmult(self.weights['query.weight'], hidden_state.payload)
        q = self.engine.pc_add(self.weights['query.bias'], wx)
        return GLTensor(q, self.layout.HEAD_SHAPE, self.layout.physical_shape)

    def key(self, hidden_state: GLTensor) -> GLTensor:
        self.layout.require_shape(
            hidden_state, self.layout.HIDDEN_SHAPE, "hidden_state"
        )
        wx = self.engine.mmult(self.weights['key.weight'], hidden_state.payload)
        k = self.engine.pc_add(self.weights['key.bias'], wx)
        return GLTensor(k, self.layout.HEAD_SHAPE, self.layout.physical_shape)

    def value(self, hidden_state: GLTensor) -> GLTensor:
        self.layout.require_shape(
            hidden_state, self.layout.HIDDEN_SHAPE, "hidden_state"
        )
        wx = self.engine.mmult(self.weights['value.weight'], hidden_state.payload)
        v = self.engine.pc_add(self.weights['value.bias'], wx)
        return GLTensor(v, self.layout.HEAD_SHAPE, self.layout.physical_shape)

    def calculate_attention_score(self, query: GLTensor, key: GLTensor) -> GLTensor:
        """
        Calculates the attention score(qk^T/sqrt(d_k))
        @param k: key, GLTensor with logical shape (12, 128, 64)
        @param q: query, GLTensor with logical shape (12, 128, 64)
        @return: output, GLTensor with logical shape (12, 128, 128)
        """
        k_transposed = self.engine.transpose(key.payload)

        probabilities = self.engine.mmult(query.payload, k_transposed)
        fact = 1 / np.sqrt(query.logical_shape[-1])
        probabilities = self.engine.mult_scalar(probabilities, fact)

        return GLTensor(
            probabilities,
            self.layout.ATTENTION_SCORE_SHAPE,
            self.layout.physical_shape,
        )
    
    def calculate_attention_context(
        self,
        probabilities: GLTensor,
        value: GLTensor,
    ) -> GLTensor:
        """
        Calculates the attention context by concatenating the attention heads and projecting them to the original hidden size.
        @param probabilities: attention probabilities, GLTensor with logical shape 
        @param value: value, GLTensor with logical shape (12, 128, 64) 
        @return: output, GLTensor with logical shape 
        """
        output = self.engine.mmult(probabilities.payload, value.payload)
        return GLTensor(
            output, self.layout.HEAD_SHAPE, self.layout.physical_shape
        )

    def dense(self, context: GLTensor) -> GLTensor:
        self.layout.require_shape(context, self.layout.HEAD_SHAPE, "context")
        wx = self.engine.mmult(self.weights['dense.weight'], context.payload)
        wx = self.engine.pc_add(self.weights['dense.bias'], wx)
        return GLTensor(wx, self.layout.HIDDEN_SHAPE, self.layout.physical_shape)


class ThorBertFF(ThorModule):
    def __init__(
        self,
        evaluator: ThorLinearEvaluator,
        weights: Mapping[str, Any],
        layer_idx: int,
    ) -> None:
        super().__init__(evaluator, weights)
        if not 0 <= layer_idx < ThorBert.n_layers:
            raise ValueError(f"layer_idx must be between 0 and {ThorBert.n_layers - 1}")
        self.layout = BertGLLayout(tuple(int(value) for value in self.engine.shape))
        self.layer_idx = layer_idx
        prefix = f"bert.encoder.layer.{layer_idx}"
        self.weights = (
            {
                "dense1.weight": weights[f"{prefix}.intermediate.dense.weight"],
                "dense1.bias": weights[f"{prefix}.intermediate.dense.bias"],
                "dense2.weight": weights[f"{prefix}.output.dense.weight"],
                "dense2.bias": weights[f"{prefix}.output.dense.bias"],
                "LayerNorm.weight": weights[f"{prefix}.output.LayerNorm.weight"],
                "LayerNorm.bias": weights[f"{prefix}.output.LayerNorm.bias"],
            }
            if weights
            else {}
        )
        self.keys = list(self.weights)
        self.gelu = partial(he_gelu, engine=self.engine)
        layernorm = he_layernorm3 if layer_idx in (9, 10) else he_layernorm2
        self.layernorm = partial(
            layernorm,
            engine=self.engine,
            gamma=self.weights.get("LayerNorm.weight"),
            beta=self.weights.get("LayerNorm.bias"),
        )
        self.devices = []

    def forward(
        self,
        hidden_state: GLTensor,
        debug: bool = False,
        sk: Any = None,
        **_: Any,
    ) -> GLTensor:
        """Evaluate dense, GELU, dense, residual, and layer normalization."""
        self.layout.require_shape(
            hidden_state, self.layout.HIDDEN_SHAPE, "hidden_state"
        )
        intermediate = self.dense1(hidden_state)
        activated = GLTensor(
            self.gelu(x=intermediate.payload, sk=sk),
            self.layout.INTERMEDIATE_SHAPE,
            self.layout.physical_shape,
        )
        projected = self.dense2(activated)

        residual, update = self.engine.auto_level(
            hidden_state.payload, projected.payload
        )
        normalized = self.layernorm(
            x=self.engine.cc_add(residual, update),
            debug=debug,
            sk=sk,
        )
        return GLTensor(
            normalized, self.layout.HIDDEN_SHAPE, self.layout.physical_shape
        )

    def dense1(self, hidden_state: GLTensor) -> GLTensor:
        """Project hidden states from 768 to 3072 features and add bias."""
        self.layout.require_shape(
            hidden_state, self.layout.HIDDEN_SHAPE, "hidden_state"
        )
        projected = self.engine.mmult(
            self.weights["dense1.weight"], hidden_state.payload
        )
        projected = self.engine.pc_add(
            self.weights["dense1.bias"], projected
        )
        return GLTensor(
            projected,
            self.layout.INTERMEDIATE_SHAPE,
            self.layout.physical_shape,
        )

    def dense2(self, intermediate: GLTensor, **_: Any) -> GLTensor:
        """Project intermediate states from 3072 back to 768 features."""
        self.layout.require_shape(
            intermediate,
            self.layout.INTERMEDIATE_SHAPE,
            "intermediate",
        )
        projected = self.engine.mmult(
            self.weights["dense2.weight"], intermediate.payload
        )
        projected = self.engine.pc_add(
            self.weights["dense2.bias"], projected
        )
        return GLTensor(
            projected, self.layout.HIDDEN_SHAPE, self.layout.physical_shape
        )


class ThorBertPooler(ThorModule):
    def forward(self, hidden_state: GLTensor) -> GLTensor:
        self._deferred("pooler forward")

    def dense(self, hidden_state: GLTensor, scale: float = 1.0) -> GLTensor:
        self._deferred("pooler projection")


class ThorBertClassifier(ThorModule):
    def forward(self, pooled: GLTensor) -> GLTensor:
        self._deferred("classifier forward")

    def dense(self, pooled: GLTensor) -> GLTensor:
        self._deferred("classifier projection")


__all__ = [
    "BertGLLayout",
    "ThorBert",
    "ThorBertAttention",
    "ThorBertClassifier",
    "ThorBertFF",
    "ThorBertPooler",
    "ThorModule",
]
