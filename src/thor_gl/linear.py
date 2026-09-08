from typing import Any

from desilofhe import (
    GLCiphertext,
    GLMatrixMultiplcationKey,
    GLPlaintext,
    GLTranspositionKey,
)

from .gl import FheData, GLEngine


class ThorLinearEvaluator:
    """GL-native linear operations for full three-dimensional FHE tensors."""

    def __init__(self, engine: GLEngine):
        self.engine = engine

    def pt_ct_matmul(
        self,
        weight: GLPlaintext,
        ciphertext: GLCiphertext,
        matrix_key: GLMatrixMultiplcationKey | None = None,
    ) -> GLCiphertext:
        """Return the per-slice matrix product ``weight @ ciphertext``."""
        if not isinstance(weight, GLPlaintext):
            raise TypeError("weight must be a GLPlaintext")
        if not isinstance(ciphertext, GLCiphertext):
            raise TypeError("ciphertext must be a GLCiphertext")
        return self.engine.mmult(weight, ciphertext, matrix_key)

    def parallel_diagonal_pt_ct_mult(self, w_t: Any, x_t: Any) -> GLCiphertext:
        """Reject the CKKS diagonal algorithm; GL uses ``pt_ct_matmul``."""
        raise NotImplementedError(
            "GL matrices do not use packed CKKS diagonals; call pt_ct_matmul "
            "with one GLPlaintext and one GLCiphertext"
        )

    def transpose_upper_to_lower(
        self,
        value: FheData,
        transposition_key: GLTranspositionKey | None = None,
    ) -> FheData:
        """Transpose every matrix slice using Desilo's native operation."""
        return self.engine.transpose(value, transposition_key)

    def make_rotated_copies(self, ciphertexts: Any) -> Any:
        """Defer copy layout until BERT uses GL batch and matrix axes."""
        raise NotImplementedError(
            "pack independent matrices on GL's batch axis, or use axis-specific "
            "roll operations after the BERT tensor layout is defined"
        )

    def make_copies(self, ciphertexts: Any, scale: float = 0.5) -> Any:
        """Defer ciphertext expansion until the GL tensor layout is defined."""
        raise NotImplementedError(
            "broadcast copies while packing plaintext data where possible; "
            "otherwise use 3D Hadamard masks and batch-axis rolls"
        )

    def rotate_internal(
        self,
        ciphertext: FheData,
        delta: int = 0,
        l_delta: int = 0,
        r_delta: int = 0,
        mask: GLPlaintext | None = None,
        mode: str | None = None,
    ) -> FheData:
        """Defer block-local rotation until logical matrix axes are fixed."""
        raise NotImplementedError(
            "replace cyclic shifts with axis-specific roll; for non-cyclic "
            "block shifts, Hadamard-mask both regions, roll them, and add"
        )

    def rotsum(self, ciphertext: GLCiphertext, interval: int) -> GLCiphertext:
        """Delegate the temporary compatibility API to the GL engine."""
        return self.engine.rotsum(ciphertext, interval)

    def pre_encode_masks(self) -> None:
        """Reject flattened CKKS masks, which cannot encode GL tensor layout."""
        raise NotImplementedError(
            "precompute 3D masks only after BERT defines its GL batch, row, "
            "and column layout"
        )
