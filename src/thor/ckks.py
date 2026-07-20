from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from desilofhe import (
    BootstrapKey,
    Ciphertext,
    ConjugationKey,
    Engine,
    LightPlaintext,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
    SmallBootstrapKey,
)

FheData = Ciphertext | Plaintext | LightPlaintext


class CkksEngine(Engine):
    """THOR's CKKS helpers implemented on top of the public DesiloFHE API."""

    _WEIGHT_FORMAT = "desilofhe-plaintext-v1"

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        mode: str | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        config = self._translate_liberate_params(params or {})
        config.update(kwargs)
        config["mode"] = mode or config.get("mode", "cpu")

        # DesiloFHE configures bootstrapping with a dedicated parameter set.
        if any(
            config.get(flag)
            for flag in (
                "use_bootstrap",
                "use_bootstrap_to_14_levels",
                "use_bootstrap_to_17_levels",
            )
        ):
            config.pop("log_coeff_count", None)
            config.pop("special_prime_count", None)

        super().__init__(**config)
        self.mode = config["mode"]
        self.verbose = verbose
        self.pk: PublicKey | None = None
        self.evk: RelinearizationKey | None = None
        self.rotation_key: RotationKey | None = None
        self.conj_key: ConjugationKey | None = None
        self.bs_key: BootstrapKey | SmallBootstrapKey | None = None

    @staticmethod
    def _translate_liberate_params(params: Mapping[str, Any]) -> dict[str, Any]:
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

    def add_pk(self, public_key: PublicKey) -> None:
        self.pk = public_key

    def add_evk(self, relinearization_key: RelinearizationKey) -> None:
        self.evk = relinearization_key

    def add_gk(self, rotation_key: RotationKey) -> None:
        self.rotation_key = rotation_key

    def add_conj_key(self, conjugation_key: ConjugationKey) -> None:
        self.conj_key = conjugation_key

    def add_bs_key(self, bootstrap_key: BootstrapKey | SmallBootstrapKey) -> None:
        self.bs_key = bootstrap_key

    def add_rot_keys_from_sk(self, deltas: list[int], secret_key: SecretKey) -> None:
        del deltas
        if self.rotation_key is None:
            self.rotation_key = super().create_rotation_key(secret_key)

    def add_hrot_keys_from_sk(self, deltas: list[int], secret_key: SecretKey) -> None:
        self.add_rot_keys_from_sk(deltas, secret_key)

    def encode_and_encrypt(
        self,
        message: Any,
        public_key: PublicKey | SecretKey | None = None,
        level: int | None = None,
        padding: bool = True,
    ) -> Ciphertext:
        del padding
        key = public_key or self.pk
        if key is None:
            raise ValueError("An encryption key has not been configured")
        if level is None:
            return super().encrypt(message, key)
        return super().encrypt(message, key, level)

    def encodecrypt(
        self,
        message: Any,
        public_key: PublicKey | SecretKey | None = None,
        level: int | None = None,
        padding: bool = True,
    ) -> Ciphertext:
        return self.encode_and_encrypt(message, public_key, level, padding)

    def decrode(self, ciphertext: Ciphertext, secret_key: SecretKey, **_: Any) -> np.ndarray:
        return super().decrypt(ciphertext, secret_key)

    def bootstrap(self, ciphertext: Ciphertext, *keys: Any) -> Ciphertext:
        if keys:
            return super().bootstrap(ciphertext, *keys)
        if self.evk is None or self.conj_key is None or self.bs_key is None:
            raise ValueError(
                "Bootstrapping requires configured relinearization, conjugation, "
                "and bootstrap keys"
            )
        if isinstance(self.bs_key, SmallBootstrapKey):
            if self.rotation_key is None:
                raise ValueError("Small-key bootstrapping also requires a rotation key")
            return super().bootstrap(
                ciphertext,
                self.evk,
                self.conj_key,
                self.rotation_key,
                self.bs_key,
            )
        return super().bootstrap(ciphertext, self.evk, self.conj_key, self.bs_key)

    def mult(
        self,
        a: Any,
        b: Any,
        evk: RelinearizationKey | None = None,
        relin: bool = True,
        auto_level: bool = False,
    ) -> Ciphertext:
        del auto_level
        if isinstance(a, Ciphertext) and isinstance(b, Ciphertext):
            return self.ct_ct_mult(a, b, evk=evk, relin=relin)
        return super().multiply(a, b)

    def auto_ct_ct_mult(
        self,
        ct0: Ciphertext,
        ct1: Ciphertext,
        evk: RelinearizationKey | None = None,
        relin: bool = True,
        rescale: bool = True,
    ) -> Ciphertext:
        ct0, ct1 = self.auto_level(ct0, ct1)
        return self.ct_ct_mult(ct0, ct1, evk=evk, relin=relin, rescale=rescale)

    def ct_ct_mult(
        self,
        a: Ciphertext,
        b: Ciphertext,
        evk: RelinearizationKey | None = None,
        relin: bool = True,
        rescale: bool = True,
    ) -> Ciphertext:
        del rescale
        a, b = self.auto_level(a, b)
        if not relin:
            return super().multiply(a, b)
        key = evk or self.evk
        if key is None:
            raise ValueError("A relinearization key has not been configured")
        return super().multiply(a, b, key)

    def relinearize(
        self,
        ciphertext: Ciphertext,
        evk: RelinearizationKey | None = None,
        is_fast: bool = True,
    ) -> Ciphertext:
        del is_fast
        key = evk or self.evk
        if key is None:
            raise ValueError("A relinearization key has not been configured")
        return super().relinearize(ciphertext, key)

    def square(
        self,
        ciphertext: Ciphertext,
        evk: RelinearizationKey | None = None,
        relin: bool = True,
        is_fast: bool = True,
    ) -> Ciphertext:
        del is_fast
        if not relin:
            return super().square(ciphertext)
        key = evk or self.evk
        if key is None:
            raise ValueError("A relinearization key has not been configured")
        return super().square(ciphertext, key)

    def imult(self, ciphertext: Ciphertext) -> Ciphertext:
        return super().multiply_imaginary_integer(ciphertext, 1)

    def minus_imult(self, ciphertext: Ciphertext) -> Ciphertext:
        return super().multiply_imaginary_integer(ciphertext, -1)

    def pt_ct_mult(self, plaintext: Plaintext | LightPlaintext, ciphertext: Ciphertext) -> Ciphertext:
        return super().multiply(ciphertext, plaintext)

    def pt_ct_mult_extended(self, plaintext: Plaintext | LightPlaintext, ciphertext: Ciphertext) -> Ciphertext:
        return super().multiply(ciphertext, plaintext)

    def mult_int_scalar_triplet(self, ciphertext: Ciphertext, scalar: int, **_: Any) -> Ciphertext:
        return super().multiply(ciphertext, int(scalar))

    def mult_int_scalar(self, ciphertext: Ciphertext, scalar: int, **_: Any) -> Ciphertext:
        return super().multiply(ciphertext, int(scalar))

    def mult_scalar(self, ciphertext: Ciphertext, scalar: float, **_: Any) -> Ciphertext:
        return super().multiply(ciphertext, float(scalar))

    def scalar_mult(self, scalar: float, ciphertext: Ciphertext, **kwargs: Any) -> Ciphertext:
        return self.mult_scalar(ciphertext, scalar, **kwargs)

    def add_scalar(self, ciphertext: Ciphertext, scalar: float) -> Ciphertext:
        return super().add(ciphertext, scalar)

    def pc_add(self, plaintext: Plaintext | LightPlaintext, ciphertext: Ciphertext) -> Ciphertext:
        return super().add(plaintext, ciphertext)

    def mc_mult(self, message: Any, ciphertext: Ciphertext, **_: Any) -> Ciphertext:
        return super().multiply(message, ciphertext)

    def cm_mult(self, ciphertext: Ciphertext, message: Any, **_: Any) -> Ciphertext:
        return super().multiply(ciphertext, message)

    def mc_sub(self, message: Any, ciphertext: Ciphertext) -> Ciphertext:
        return super().subtract(message, ciphertext)

    def cm_sub(self, ciphertext: Ciphertext, message: Any) -> Ciphertext:
        return super().subtract(ciphertext, message)

    def sub(self, a: Any, b: Any) -> Ciphertext:
        return super().subtract(a, b)

    def cc_add(self, a: Ciphertext | None, b: Ciphertext | None) -> Ciphertext:
        if a is None:
            if b is None:
                raise ValueError("At least one ciphertext is required")
            return b
        if b is None:
            return a
        return super().add(a, b)

    def cc_sub(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        return super().subtract(a, b)

    def auto_cc_add(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        return self.cc_add(a, b)

    def auto_level(self, a: FheData, b: FheData) -> tuple[FheData, FheData]:
        target = min(a.level, b.level)
        if a.level != target:
            a = self._change_data_level(a, target)
        if b.level != target:
            b = self._change_data_level(b, target)
        return a, b

    def _change_data_level(self, value: FheData, level: int) -> FheData:
        if isinstance(value, LightPlaintext):
            return super().change_level(value, level)
        return super().level_down(value, level)

    def rotate_left(
        self,
        ciphertext: Ciphertext,
        delta: int,
        rotation_key: RotationKey | None = None,
    ) -> Ciphertext:
        if delta == 0:
            return ciphertext
        key = rotation_key or self.rotation_key
        if key is None:
            raise ValueError("A rotation key has not been configured")
        # DesiloFHE follows numpy.roll: positive deltas rotate right.
        return super().rotate(ciphertext, key, -int(delta))

    def rotate_hoist(
        self,
        ciphertext: Ciphertext,
        delta: int,
        rotation_key: RotationKey | None = None,
    ) -> Ciphertext:
        return self.rotate_left(ciphertext, delta, rotation_key)

    def conjugate(
        self,
        ciphertext: Ciphertext,
        conjugation_key: ConjugationKey | None = None,
    ) -> Ciphertext:
        key = conjugation_key or self.conj_key
        if key is None:
            raise ValueError("A conjugation key has not been configured")
        return super().conjugate(ciphertext, key)

    def rotsum(self, ciphertext: Ciphertext, interval: int) -> Ciphertext:
        if interval <= 0 or self.num_slots % interval:
            raise ValueError("interval must be a positive divisor of the slot count")
        result = ciphertext
        for i in range(int(np.log2(self.num_slots // interval))):
            result = self.cc_add(result, self.rotate_left(result, interval * 2**i))
        return result

    def save_plaintext_weights(self, weights: Mapping[str, Any], filename: str | Path) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self._WEIGHT_FORMAT,
            "engine_hash": self.build_hash,
            "weights": self._map_plaintexts(weights, serialize=True),
        }
        with path.open("wb") as stream:
            pickle.dump(payload, stream)

    def load_plaintext_weights(self, filename: str | Path) -> dict[str, Any]:
        with Path(filename).open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("format") != self._WEIGHT_FORMAT:
            raise ValueError(
                "This is not a DesiloFHE THOR weight file; re-run encode.py to migrate it"
            )
        return self._map_plaintexts(payload["weights"], serialize=False)

    def _map_plaintexts(self, value: Any, *, serialize: bool) -> Any:
        if serialize and isinstance(value, Plaintext):
            return {"__plaintext__": bytes(super().serialize_plaintext(value))}
        if serialize and isinstance(value, LightPlaintext):
            return {"__light_plaintext__": bytes(super().serialize_light_plaintext(value))}
        if not serialize and isinstance(value, dict) and set(value) == {"__plaintext__"}:
            return super().deserialize_plaintext(value["__plaintext__"])
        if not serialize and isinstance(value, dict) and set(value) == {"__light_plaintext__"}:
            return super().deserialize_light_plaintext(value["__light_plaintext__"])
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
