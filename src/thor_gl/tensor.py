from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from desilofhe import GLCiphertext, GLPlaintext


SUPPORTED_GL_SHAPES = frozenset(
    {
        (256, 16, 16),
        (16, 256, 256),
        (4, 1024, 1024),
        (256, 32, 32),
        (16, 512, 512),
        (4, 2048, 2048),
        (256, 64, 64),
    }
)

GLPayload = GLPlaintext | GLCiphertext


@dataclass(frozen=True, slots=True)
class GLTensor:
    """Logical-shape metadata for one encoded or encrypted Desilo GL payload."""

    payload: GLPayload
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        """
        Validates the logical shape against the physical GL payload capacity.
        @return: None.
        """
        logical_shape = tuple(int(dimension) for dimension in self.logical_shape)
        physical_shape = tuple(int(dimension) for dimension in self.physical_shape)
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "physical_shape", physical_shape)

        if physical_shape not in SUPPORTED_GL_SHAPES:
            raise ValueError(f"Unsupported GL physical shape: {physical_shape}")
        if not logical_shape or any(dimension <= 0 for dimension in logical_shape):
            raise ValueError("logical_shape must contain positive dimensions")
        logical_size = prod(logical_shape)
        physical_size = prod(physical_shape)
        if logical_size > physical_size:
            raise ValueError(
                f"Logical shape {logical_shape} contains {logical_size} values, but "
                f"physical shape {physical_shape} holds only {physical_size}"
            )

    def to_logical(self, decoded: np.ndarray) -> np.ndarray:
        """
        Restores Desilo's decoded physical array to the original logical shape.
        @param decoded: Decoded GL array with this tensor's physical shape.
        @return: Leading decoded values reshaped to the stored logical shape.
        """
        physical = np.asarray(decoded)
        if physical.shape != self.physical_shape:
            raise ValueError(
                f"Decoded shape {physical.shape} does not match physical shape "
                f"{self.physical_shape}"
            )
        logical_size = prod(self.logical_shape)
        return physical.reshape(-1)[:logical_size].reshape(self.logical_shape)


__all__ = ["GLTensor", "SUPPORTED_GL_SHAPES"]
