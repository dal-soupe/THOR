import tempfile
import unittest
from pathlib import Path

import numpy as np
from desilofhe import LightPlaintext

from thor import CkksEngine


class CkksEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CkksEngine(max_level=4, mode="cpu")
        cls.secret_key = cls.engine.create_secret_key()
        cls.public_key = cls.engine.create_public_key(cls.secret_key)
        cls.relinearization_key = cls.engine.create_relinearization_key(cls.secret_key)
        cls.rotation_key = cls.engine.create_rotation_key(cls.secret_key)
        cls.conjugation_key = cls.engine.create_conjugation_key(cls.secret_key)
        cls.engine.add_pk(cls.public_key)
        cls.engine.add_evk(cls.relinearization_key)
        cls.engine.add_gk(cls.rotation_key)
        cls.engine.add_conj_key(cls.conjugation_key)

    def decrypt(self, ciphertext):
        return self.engine.decrypt(ciphertext, self.secret_key)

    def test_encrypt_and_public_operations(self):
        message = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        ciphertext = self.engine.encode_and_encrypt(message, level=4)

        multiplied = self.engine.auto_ct_ct_mult(ciphertext, ciphertext)
        conjugated = self.engine.conjugate(ciphertext)
        rotated = self.engine.rotate_left(ciphertext, 1)

        np.testing.assert_allclose(self.decrypt(multiplied)[:3], message**2, atol=1e-6)
        np.testing.assert_allclose(self.decrypt(conjugated)[:3], message.conjugate(), atol=1e-6)
        np.testing.assert_allclose(self.decrypt(rotated)[:2], message[1:], atol=1e-6)
        self.assertEqual(multiplied.level, 3)
        self.assertEqual(multiplied.polynomial_count, 2)

    def test_plaintext_multiply_and_level_alignment(self):
        high = self.engine.encode_and_encrypt([1.0, 2.0, 3.0], level=4)
        low = self.engine.encode_and_encrypt([4.0, 5.0, 6.0], level=3)
        high, low = self.engine.auto_level(high, low)
        plaintext = self.engine.encode([2.0, 3.0, 4.0], level=3)

        result = self.engine.pt_ct_mult(plaintext, high)
        summed = self.engine.cc_add(high, low)

        self.assertEqual(high.level, 3)
        self.assertEqual(low.level, 3)
        self.assertEqual(result.level, 2)
        np.testing.assert_allclose(self.decrypt(result)[:3], [2.0, 6.0, 12.0], atol=1e-6)
        np.testing.assert_allclose(self.decrypt(summed)[:3], [5.0, 7.0, 9.0], atol=1e-6)

    def test_masked_triplet_uses_desilo_scale(self):
        left = self.engine.encode_and_encrypt([2.0, 3.0], level=4)
        right = self.engine.encode_and_encrypt([5.0, 7.0], level=4)
        triplet = self.engine.ct_ct_mult(left, right, relin=False)
        mask = self.engine.encode([1.0, 0.0], level=triplet.level)
        masked = self.engine.pt_ct_mult_extended(mask, triplet)
        aligned = self.engine.level_down(triplet, masked.level)
        complement = self.engine.cc_sub(aligned, masked)
        result = self.engine.relinearize(self.engine.cc_add(masked, complement))

        self.assertEqual(triplet.polynomial_count, 3)
        np.testing.assert_allclose(self.decrypt(result)[:2], [10.0, 21.0], atol=1e-5)

    def test_plaintext_weight_serialization(self):
        weights = {
            "weight": np.array(
                [
                    self.engine.encode_to_light_plaintext([1.0], level=4),
                    self.engine.encode_to_light_plaintext([2.0], level=3),
                ],
                dtype=object,
            ),
            "unused": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pkl"
            self.engine.save_plaintext_weights(weights, path)
            loaded = self.engine.load_plaintext_weights(path)

        self.assertIsInstance(loaded["weight"][0], LightPlaintext)
        self.assertEqual(loaded["weight"][0].level, 4)
        self.assertEqual(loaded["weight"][1].level, 3)
        ciphertext = self.engine.encode_and_encrypt([2.0], level=4)
        first = self.engine.pt_ct_mult(loaded["weight"][0], ciphertext)
        second = self.engine.pt_ct_mult(loaded["weight"][1], ciphertext)
        self.assertEqual(first.level, 3)
        self.assertEqual(second.level, 3)
        np.testing.assert_allclose(self.decrypt(first)[:1], [2.0], atol=1e-6)
        np.testing.assert_allclose(self.decrypt(second)[:1], [4.0], atol=1e-6)

    def test_bootstrap_requires_configured_keys(self):
        ciphertext = self.engine.encode_and_encrypt([1.0])
        with self.assertRaisesRegex(ValueError, "Bootstrapping requires"):
            self.engine.bootstrap(ciphertext)


if __name__ == "__main__":
    unittest.main()
