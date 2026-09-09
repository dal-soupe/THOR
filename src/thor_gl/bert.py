from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod
from typing import Any, Mapping
import numpy as np
from functools import partial
import time
import torch

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

    def to(self, devices: list[int] | None = None) -> ThorModule:
        """GL objects remain owned by the mode/device selected by their engine."""
        return self

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
    """Twelve-layer BERT layout scaffold for GL tensors.

    The logical layouts are defined and validated here, but encrypted inference
    is intentionally unavailable until block multiplication and low-depth
    nonlinear evaluation replace the former CKKS/bootstrap implementation.
    """

    n_layers = 12

    def __init__(
        self,
        evaluator: ThorLinearEvaluator,
        weights: Mapping[str, Any],
        max_layer_batch: int = 2,
    ) -> None:
        if max_layer_batch <= 0:
            raise ValueError("max_layer_batch must be positive")
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.layout = BertGLLayout(tuple(int(value) for value in self.engine.shape))
        self.weights = weights
        self.max_layer_batch = max_layer_batch
        self.attentions = [
            ThorBertAttention(evaluator, weights, layer)
            for layer in range(self.n_layers)
        ]
        self.ffs = [
            ThorBertFF(evaluator, weights, layer)
            for layer in range(self.n_layers)
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
        
        for i in range(self.n_layers//self.max_layer_batch):
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                self.attentions[j].to(devices)
                self.ffs[j].to(devices)
                if self.engine.mode == "gpu":
                    print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3:.2f} GB after layer {j} allocation: ")
                
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                print(f"Forwarding layer: {j}")
                temp = time.time()
                x = self.attentions[j].forward(x, attention_mask, debug=debug, sk=sk)
                x = self.ffs[j].forward(x, debug=debug, sk=sk)
                print(f"Time taken for layer {j} forward: {time.time() - temp:.2f} seconds")
            
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                self.attentions[j].cpu()
                self.ffs[j].cpu()
                if self.engine.mode == "gpu":
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
        self.secret_key = self.engine.create_secret_key()
        self.engine.mmult_key = self.engine.create_matrix_multiplication_key(self.secret_key)
        self.engine.transp_key = self.engine.create_transposition_key(self.secret_key)
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
        self.layernorm = partial(he_layernorm1, engine=self.engine, gamma=self.weights['LayerNorm.weight'], beta=self.weights['LayerNorm.bias'])
        self.devices = []

    def forward(self, x: GLTensor, attention_mask: GLTensor, debug=False, sk=None) -> GLTensor:
        input_lev = x[0].level
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        l_k = self.evaluator.transpose_upper_to_lower(k)
        
        sftmx_scale = 1
        att_score = self.calculate_attention_score(q, k)
        att_prob = self.softmax(x=att_score, attention_mask=attention_mask, rescale=False, debug=debug, sk=sk) #Returns att_prob, already copied.

        att_context = self.calculate_attention_context(v, att_prob)
        dense_output = self.dense(att_context)
        
        x_out_sum = np.full((8,), None, dtype=object)

        x = self.layernorm(x=dense_output, debug=False, sk=sk)
        return x

    def query(self, hidden_state: GLTensor) -> GLTensor:
        
        wx = self.engine.mmult(self.weights['query.weight'], hidden_state.payload)
        
        q = np.full(wx.shape, 0.0)
        self.engine.add(self.weights['query.bias'], wx)
        return q

    def key(self, hidden_state: GLTensor) -> GLTensor:
        
        wx = self.engine.mmult(self.weights['key.weight'], hidden_state.payload)
        
        k = np.full(wx.shape, 0.0)
        self.engine.add(self.weights['key.bias'], wx)
        return k

    def value(self, hidden_state: GLTensor) -> GLTensor:
        
        wx = self.engine.mmult(self.weights['value.weight'], hidden_state.payload)
        
        v = np.full(wx.shape, 0.0)
        self.engine.add(self.weights['value.bias'], wx)
        return v

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

        return probabilities
    
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
        return output

    def dense(self, context: GLTensor) -> GLTensor:
        wx = self.engine.mmult(self.weights['dense.weight'], context.payload)
        
        if wx.shape[0] != ???:
            raise ValueError("Shape of wx should be ?? but got {}".format(wx.shape))
        
        wx = self.engine.pc_add(self.weights['dense.bias'], wx)
        return wx


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
        self.layer_idx = layer_idx

    def forward(self, hidden_state: GLTensor, **_: Any) -> GLTensor:
        self._deferred("feed-forward layer")

    def dense1(self, hidden_state: GLTensor) -> GLTensor:
        self._deferred("feed-forward expansion")

    def dense2(self, intermediate: GLTensor, **_: Any) -> GLTensor:
        self._deferred("feed-forward contraction")


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
