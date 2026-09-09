from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from desilofhe import (
    GLEngine,
    GLPlaintext,
    GLCiphertext,
    GLSecretKey,
    GLHadamardMultiplicationKey,
    GLMatrixMultiplcationKey,
    GLConjugationKey,
    GLTranspositionKey,
    GLConjugateTranspositionKey,
    GLRotationKey,
)

FheData = GLCiphertext | GLPlaintext


class GLEngine(GLEngine):
    """THOR's GL helpers implemented on top of the public DesiloFHE API."""

    _WEIGHT_FORMAT = "desilofhe-plaintext-v1"

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        mode: str | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        config = self._translate_params(params or {})
        config.update(kwargs)
        config["mode"] = mode or config.get("mode", "cpu")

        # DesiloFHE configures bootstrapping with a dedicated parameter set.
        # if any(
        #     config.get(flag)
        #     for flag in (
        #         "use_bootstrap",
        #         "use_bootstrap_to_14_levels",
        #         "use_bootstrap_to_17_levels",
        #     )
        # ):
        #     config.pop("log_coeff_count", None)
        #     config.pop("special_prime_count", None)

        super().__init__(**config)
        self.mode = config["mode"]
        self.verbose = verbose
        self.sk: GLSecretKey | None = None
        self.hmult_key: GLHadamardMultiplicationKey | None = None
        self.mmult_key: GLMatrixMultiplcationKey | None = None
        self.rot_key: GLRotationKey | None = None
        self.conj_key: GLConjugationKey | None = None
        self.transp_key: GLTranspositionKey | None = None
        self.conj_transp_key: GLConjugateTranspositionKey | None = None

    @staticmethod
    def _translate_params(params: Mapping[str, Any]) -> dict[str, Any]:
        """Accept the old constructor dictionary while callers migrate."""
        legacy = dict(params)
        translated: dict[str, Any] = {}

        if "logN" in legacy:
            translated["log_coeff_count"] = legacy.pop("logN")
        if "num_special_primes" in legacy:
            translated["special_prime_count"] = legacy.pop("num_special_primes")
        devices = legacy.pop("devices", None)
        if devices:
            translated["mode"] = "gpu"
            translated["device_id"] = devices[0]

        # Scale and security parameters are selected by DesiloFHE's presets.
        for obsolete in ("scale_bits", "num_scales", "quantum"):
            legacy.pop(obsolete, None)
        translated.update(legacy)
        return translated

    @property
    def num_slots(self) -> int:
        return self.slot_count

    @property
    def num_levels(self) -> int:
        return self.max_level

    def add_hmult_key(self, hadamard_multiplication_key: GLHadamardMultiplicationKey) -> None:
        self.hmult_key = hadamard_multiplication_key

    def add_mmult_key(self, matrix_multiplication_key: GLMatrixMultiplcationKey) -> None:
        self.mmult_key = matrix_multiplication_key

    def add_rot_key(self, rot_key: GLRotationKey) -> None:
        self.rot_key = rot_key

    def add_conj_key(self, conjugation_key: GLConjugationKey) -> None:
        self.conj_key = conjugation_key

    def add_bs_key(self, bootstrap_key) -> None:
        return "bootstrapping not supported"

    # unused
    def _add_rot_key_from_sk(self, deltas: list[int], secret_key: GLSecretKey) -> None:
        del deltas
        if self.rot_key is None:
            self.rot_key = super().create_rotation_key(secret_key)

    def encode_and_encrypt(
        self,
        message: Any,
        sk: GLSecretKey | None = None,
        level: int | None = None,
    ) -> GLCiphertext:
        key = sk or self.sk
        if key is None:
            raise ValueError("An encryption key has not been configured")
        if level is None:
            return super().encrypt(message, key)
        return super().encrypt(message, key, level)

    def decrypt_and_decode(self, ciphertext: GLCiphertext, secret_key: GLSecretKey, **_: Any) -> np.ndarray:
        return super().decrypt(ciphertext, secret_key)

    def mmult(
        self,
        a: Any,
        b: Any,
        evk: GLMatrixMultiplcationKey | None = None,
    ) -> GLCiphertext:
        """ Classic matrix multiplication
        a or b can be pt, ct or 3d matrix
        a or b must be ciphertext
        Consumes 1 level
        """
        key = evk or self.mmult_key
        if key is None:
            raise ValueError("A matrix multiplication key has not been configured")
        try:
            return super().matrix_multiply(a, b, key)
        except Exception as e:
            # this tries to catch if it needs to be releveled down (auto_level)
            raise RuntimeError(f"Matrix multiplication failed: {e}") from e

    def cmult(
        self,
        a: Any,
        b: Any,
    ) -> GLCiphertext:
        """ multiply by a constant
        a or b can be ct, int or double
        one among a or b must be ciphertext
        Consumes 1 level usually
        """
        if isinstance(a, GLCiphertext) and isinstance(b, GLCiphertext):
            raise ValueError("Both arguments cannot be ciphertexts for cmult")
        return super().multiply(a, b)

    def hmult(
        self,
        a: Any,
        b: Any,
        evk: GLHadamardMultiplicationKey | None = None,
    ) -> GLCiphertext:
        """
        a or b can be pt, ct or 3d matrix
        a or b must be ciphertext
        Consumes 1 level
        """
        if isinstance(a, GLCiphertext) and isinstance(b, GLCiphertext):
            return self.ct_ct_hmult(a, b, evk=evk)
        return super().hadamard_multiply(a, b)

    def auto_ct_ct_hmult(
        self,
        ct0: GLCiphertext,
        ct1: GLCiphertext,
        evk: GLHadamardMultiplicationKey | None = None,
    ) -> GLCiphertext:
        ct0, ct1 = self.auto_level(ct0, ct1)
        return self.ct_ct_hmult(ct0, ct1, evk=evk)

    def ct_ct_hmult(
        self,
        a: GLCiphertext,
        b: GLCiphertext,
        evk: GLMatrixMultiplcationKey | None = None,
    ) -> GLCiphertext:
        a, b = self.auto_level(a, b)
        key = evk or self.hmult_key
        if key is None:
            raise ValueError("A key has not been configured")
        return super().hadamard_multiply(a, b, key)

    def relinearize(
        self,
        ciphertext: GLCiphertext,
        is_fast: bool = True,
    ) -> GLCiphertext:
        raise NotImplementedError("GL does not support relinearization alone")

    def square_elts(
        self,
        ciphertext: GLCiphertext,
        evk: GLHadamardMultiplicationKey | None = None,
        is_fast: bool = True,
    ) -> GLCiphertext:
        del is_fast
        key = evk or self.hmult_key
        if key is None:
            raise ValueError("A relinearization key has not been configured")
        return super().hadamard_multiply(ciphertext, ciphertext, key)

    def imult(self, ciphertext: GLCiphertext) -> GLCiphertext:
        """
        multiplies ciphertext with 1j
        """
        raise NotImplementedError("GL does not support imaginary multiplication")

    def _minus_imult(self, ciphertext: GLCiphertext) -> GLCiphertext:
        """
        multiplies ciphertext with -1j
        """
        raise NotImplementedError("GL does not support imaginary multiplication")

    def pt_ct_hmult(self, plaintext: GLPlaintext, ciphertext: GLCiphertext) -> GLCiphertext:
        return super().hadamard_multiply(ciphertext, plaintext)

    def _mult_int_scalar_triplet(self, ciphertext: GLCiphertext, scalar: int, **_: Any) -> GLCiphertext:
        return super().hadamard_multiply(ciphertext, int(scalar))

    def mult_int_scalar(self, ciphertext: GLCiphertext, scalar: int, **_: Any) -> GLCiphertext:
        return self.cmult(ciphertext, int(scalar))

    def mult_scalar(self, ciphertext: GLCiphertext, scalar: float, **_: Any) -> GLCiphertext:
        return self.cmult(ciphertext, float(scalar))

    def scalar_hmult(self, scalar: float, ciphertext: GLCiphertext, **kwargs: Any) -> GLCiphertext:
        return self.mult_scalar(ciphertext, scalar, **kwargs)

    def mc_hmult(self, message: Any, ciphertext: GLCiphertext, **_: Any) -> GLCiphertext:
        return super().hadamard_multiply(message, ciphertext)

    def cm_hmult(self, ciphertext: GLCiphertext, message: Any, **_: Any) -> GLCiphertext:
        return super().hadamard_multiply(ciphertext, message)

    def add_scalar(self, ciphertext: GLCiphertext, scalar: float) -> GLCiphertext:
        shape = self.shape
        message = np.ones(shape, dtype=np.float64) * scalar
        return super().add(ciphertext, message)

    def pc_add(self, plaintext: GLPlaintext, ciphertext: GLCiphertext) -> GLCiphertext:
        return super().add(plaintext, ciphertext)

    def cc_add(self, a: GLCiphertext | None, b: GLCiphertext | None) -> GLCiphertext:
        if a is None:
            if b is None:
                raise ValueError("At least one ciphertext is required")
            return b
        if b is None:
            return a
        return super().add(a, b)

    def auto_cc_add(self, a: GLCiphertext, b: GLCiphertext) -> GLCiphertext:
        return self.cc_add(a, b)

    def mc_sub(self, message: Any, ciphertext: GLCiphertext) -> GLCiphertext:
        return super().subtract(message, ciphertext)

    def cm_sub(self, ciphertext: GLCiphertext, message: Any) -> GLCiphertext:
        return super().subtract(ciphertext, message)

    def sub(self, a: Any, b: Any) -> GLCiphertext:
        """
        a or b can be pt, ct or 3d matrix
        one among a or b must be ciphertext
        """
        return super().subtract(a, b)

    def cc_sub(self, a: GLCiphertext, b: GLCiphertext) -> GLCiphertext:
        return super().subtract(a, b)

    def auto_level(self, a: FheData, b: FheData) -> tuple[FheData, FheData]: # type: ignore
        target = min(a.level, b.level)
        if a.level != target:
            a = self._change_data_level(a, target)
        if b.level != target:
            b = self._change_data_level(b, target)
        return a, b

    def _change_data_level(self, value: FheData, level: int) -> FheData: # type: ignore
        return super().level_down(value, level)

    def conjugate(
        self,
        a: GLCiphertext | GLPlaintext,
        conjugation_key: GLConjugationKey | None = None,
    ) -> GLCiphertext | GLPlaintext:
        if isinstance(a, GLPlaintext):
            if conjugation_key is not None:
                raise ValueError("A key cannot be used when conjugating plaintext")
            return super().conjugate(a)
        key = conjugation_key or self.conj_key
        if key is None:
            raise ValueError("A conjugation key has not been configured")
        return super().conjugate(a, key)

    def transpose(
        self,
        a: GLCiphertext | GLPlaintext,
        transp_key: GLTranspositionKey | None = None,
    ) -> GLCiphertext | GLPlaintext:
        if isinstance(a, GLPlaintext):
            if transp_key is not None:
                raise ValueError("A key cannot be used when transposing plaintext")
            return super().transpose(a)
        key = transp_key or self.transp_key
        if key is None:
            raise ValueError("A transposition key has not been configured")
        return super().transpose(a, key)

    def conj_transpose(
        self,
        a: GLCiphertext | GLPlaintext,
        evk: GLConjugateTranspositionKey | None = None,
    ) -> GLCiphertext | GLPlaintext:
        if isinstance(a, GLPlaintext):
            if evk is not None:
                raise ValueError("A key cannot be used when conjugate-transposing plaintext")
            return super().conjugate_transpose(a)
        key = evk or self.conj_transp_key
        if key is None:
            raise ValueError("A conjugation-transposition key has not been configured")
        return super().conjugate_transpose(a, key)

    def rotate_left(
        self,
        a: GLCiphertext | GLPlaintext,
        delta: int,
        rot_key: GLRotationKey | None = None, axis: int = 2,
    ) -> GLCiphertext | GLPlaintext:
        if delta == 0:
            return a
        if isinstance(a, GLPlaintext):
            if rot_key is not None:
                raise ValueError("A key cannot be used when rotating plaintext")
            return super().roll(a, -int(delta), axis)
        key = rot_key or self.rot_key
        if key is None:
            raise ValueError("A rotation key has not been configured")
        # DesiloFHE follows numpy.roll: positive deltas rotate right (to the higher indices).
        return super().roll(a, key, -int(delta), axis)

    def rotate_hoist(
        self,
        a: GLCiphertext | GLPlaintext,
        delta: int,
        rot_key: GLRotationKey | None = None, axis: int = 2,
    ) -> GLCiphertext | GLPlaintext:
        return self.rotate_left(a, delta, rot_key)

    def rotate_right(
        self,
        a: GLCiphertext | GLPlaintext,
        delta: int,
        rot_key: GLRotationKey | None = None, axis: int = 2,
    ) -> GLCiphertext | GLPlaintext:
        if delta == 0:
            return a
        if isinstance(a, GLPlaintext):
            if rot_key is not None:
                raise ValueError("A key cannot be used when rotating plaintext")
            return super().roll(a, int(delta), axis)
        key = rot_key or self.rot_key
        if key is None:
            raise ValueError("A rotation key has not been configured")
        return super().roll(a, key, int(delta), axis)

    def rotsum(self, ciphertext: GLCiphertext, interval: int) -> GLCiphertext:
        if interval <= 0 or self.num_slots % interval:
            raise ValueError("interval must be a positive divisor of the slot count")
        result = ciphertext
        for i in range(int(np.log2(self.num_slots // interval))):
            result = self.cc_add(result, self.rotate_left(result, interval * 2**i))
        return result

    def save_plaintext_weights(self, weights: Mapping[str, Any], filename: str | Path) -> None:
        raise NotImplementedError("Desilo GLEngine does not expose plaintext serialization")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self._WEIGHT_FORMAT,
            "engine_hash": self.build_hash,
            "weights": self._map_plaintexts(weights, serialize=True),
        }
        with path.open("wb") as stream:
            pickle.dump(payload, stream)

    def load_plaintext_weights(self, filename: str | Path) -> dict[str, Any]:
        raise NotImplementedError("Desilo GLEngine does not expose plaintext serialization")
        with path.open("rb") as stream:
            payload = pickle.load(stream)

        if payload.get("format") != self._WEIGHT_FORMAT:
            raise ValueError(f"Unsupported plaintext weight format in {path}")

        stored_hash = payload.get("engine_hash")
        if stored_hash != self.build_hash:
            raise ValueError(
                f"Plaintext weights in {path} are incompatible with this GL engine "
                f"(stored engine hash: {stored_hash!r}, current engine hash: "
                f"{self.build_hash!r}). Regenerate this file with the same engine "
                "parameters used for inference."
            )

        return self._map_plaintexts(payload["weights"], serialize=False)

    def _map_plaintexts(self, value: Any, *, serialize: bool) -> Any:
        if serialize and isinstance(value, GLPlaintext):
            return {"__plaintext__": bytes(super().serialize_plaintext(value))}
        if not serialize and isinstance(value, dict) and set(value) == {"__plaintext__"}:
            return super().deserialize_plaintext(value["__plaintext__"])
        if isinstance(value, np.ndarray):
            mapped = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                mapped[index] = self._map_plaintexts(value[index], serialize=serialize)
            return mapped
        if isinstance(value, dict):
            return {
                key: self._map_plaintexts(item, serialize=serialize)
                for key, item in value.items()
            }
        return value
