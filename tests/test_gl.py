import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from desilofhe import GLCiphertext, GLEngine, GLPlaintext
from thor_gl import GLEngine


class GLEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GLEngine(shape=(256, 16, 16), mode="cpu")
        cls.secret_key = cls.engine.create_secret_key()
        cls.hadamard_key = cls.engine.create_hadamard_multiplication_key(cls.secret_key)
        cls.matrix_key = cls.engine.create_matrix_multiplication_key(cls.secret_key)
        cls.rotation_key = cls.engine.create_rotation_key(cls.secret_key)
        cls.conjugation_key = cls.engine.create_conjugation_key(cls.secret_key)
        cls.transposition_key = cls.engine.create_transposition_key(cls.secret_key)
        cls.conjugate_transposition_key = cls.engine.create_conjugate_transposition_key(cls.secret_key)
        cls.engine.sk, cls.engine.mmult_key = cls.secret_key, cls.matrix_key
        cls.engine.add_hmult_key(cls.hadamard_key); cls.engine.add_rot_key(cls.rotation_key); cls.engine.add_conj_key(cls.conjugation_key)

    def decrypt(self, ciphertext):
        return self.engine.decrypt(ciphertext, self.secret_key)

    def test_encrypt_and_hadamard_operations(self):
        message = (np.broadcast_to(np.arange(16), self.engine.shape).astype(complex) + 2j).copy()
        ciphertext = self.engine.encode_and_encrypt(message, level=1)

        multiplied = self.engine.auto_ct_ct_hmult(ciphertext, ciphertext)
        conjugated = self.engine.conjugate(ciphertext)
        rotated = self.engine.rotate_left(ciphertext, 1)

        np.testing.assert_allclose(self.decrypt(multiplied), message**2, atol=1e-5)
        np.testing.assert_allclose(self.decrypt(conjugated), message.conjugate(), atol=1e-6)
        np.testing.assert_allclose(self.decrypt(rotated), np.roll(message, -1, axis=-1), atol=1e-6)
        self.assertEqual(multiplied.level, 0)
        self.assertEqual(multiplied.polynomial_count, 2)

    def test_plaintext_multiply_and_level_alignment(self):
        high = self.engine.encode_and_encrypt(np.ones(self.engine.shape), level=1)
        low = self.engine.encode_and_encrypt(np.full(self.engine.shape, 4.0), level=0)
        high, low = self.engine.auto_level(high, low)
        plaintext = self.engine.encode(np.full(self.engine.shape, 2.0), level=1)

        result = self.engine.pt_ct_hmult(plaintext, self.engine.encode_and_encrypt(np.ones(self.engine.shape), level=1))
        summed = self.engine.cc_add(high, low)

        self.assertEqual(high.level, 0)
        self.assertEqual(low.level, 0)
        self.assertEqual(result.level, 0)
        np.testing.assert_allclose(self.decrypt(result), 2.0, atol=1e-6)
        np.testing.assert_allclose(self.decrypt(summed), 5.0, atol=1e-6)

    def test_matrix_multiply_and_square_elements(self):
        left = self.engine.encode_and_encrypt(np.broadcast_to(np.eye(16), self.engine.shape), level=1)
        right = np.full(self.engine.shape, 2.0)
        triplet = self.engine.mmult(left, right, self.matrix_key)
        mask = self.engine.encode(np.ones(self.engine.shape), level=1)
        masked = self.engine.pt_ct_hmult(mask, self.engine.encode_and_encrypt(right, level=1))
        aligned = self.engine.square_elts(self.engine.encode_and_encrypt(right, level=1))
        complement = self.engine.cc_sub(aligned, masked)
        result = self.engine.cc_add(masked, complement)

        self.assertEqual(triplet.polynomial_count, 2)
        np.testing.assert_allclose(self.decrypt(triplet), 2.0, atol=1e-5); np.testing.assert_allclose(self.decrypt(result), 4.0, atol=1e-5)

    def test_transpose_and_conjugate_transpose(self):
        weights = {
            "weight": np.array(
                [
                    self.engine.encode(np.broadcast_to(np.eye(16), self.engine.shape), level=1),
                    self.engine.encode(np.ones(self.engine.shape) * 1j, level=1),
                ],
                dtype=object,
            ),
            "unused": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self.engine.transpose(weights["weight"][0])
            self.engine.transp_key = self.transposition_key
            loaded = {"weight": [path, self.engine.conj_transpose(weights["weight"][1])]}

        self.assertIsInstance(loaded["weight"][0], GLPlaintext)
        self.assertEqual(loaded["weight"][0].level, 1)
        self.assertEqual(loaded["weight"][1].level, 1)
        ciphertext = self.engine.encode_and_encrypt(weights["weight"][1])
        first = self.engine.transpose(ciphertext)
        second = self.engine.conj_transpose(ciphertext, self.conjugate_transposition_key)
        self.assertEqual(first.level, 1)
        self.assertEqual(second.level, 1)
        np.testing.assert_allclose(self.decrypt(first), 1j, atol=1e-6)
        np.testing.assert_allclose(self.decrypt(second), -1j, atol=1e-6)

    def test_plaintext_weight_serialization_is_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pkl"
            with self.assertRaises(NotImplementedError):
                self.engine.save_plaintext_weights({}, path)
            with self.assertRaises(NotImplementedError):
                self.engine.load_plaintext_weights(path)
            payload = self.engine.cmult(self.engine.encode_and_encrypt(np.ones(self.engine.shape)), 3)
            stream = self.decrypt(payload)

            np.testing.assert_allclose(stream, 3.0, atol=1e-6)


    def test_bootstrap_is_unsupported(self):
        ciphertext = self.engine.encode_and_encrypt(np.ones(self.engine.shape))
        with self.assertRaises(NotImplementedError):
            self.engine.relinearize(ciphertext)


if __name__ == "__main__":
    unittest.main()
