import unittest
from unittest.mock import Mock, sentinel

import numpy as np
from desilofhe import GLCiphertext, GLPlaintext

from thor_gl import GLEngine, ThorLinearEvaluator


class GLLinearEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GLEngine(shape=(256, 16, 16), mode="cpu")
        cls.secret_key = cls.engine.create_secret_key()
        cls.matrix_key = cls.engine.create_matrix_multiplication_key(cls.secret_key)
        cls.transposition_key = cls.engine.create_transposition_key(cls.secret_key)
        cls.engine.add_mmult_key(cls.matrix_key)
        cls.engine.transp_key = cls.transposition_key
        cls.evaluator = ThorLinearEvaluator(cls.engine)

    def decrypt(self, ciphertext):
        return self.engine.decrypt(ciphertext, self.secret_key)

    def test_plaintext_ciphertext_matrix_multiply(self):
        weight_matrix = np.arange(256, dtype=float).reshape(16, 16) / 256
        input_matrix = np.diag(np.arange(1, 17, dtype=float))
        weight_values = np.broadcast_to(weight_matrix, self.engine.shape).copy()
        input_values = np.broadcast_to(input_matrix, self.engine.shape).copy()
        weight = self.engine.encode(weight_values, level=1)
        ciphertext = self.engine.encrypt(input_values, self.secret_key, level=1)

        result = self.evaluator.pt_ct_matmul(weight, ciphertext)

        self.assertIsInstance(result, GLCiphertext)
        np.testing.assert_allclose(
            self.decrypt(result),
            np.matmul(weight_values, input_values),
            atol=1e-5,
        )

    def test_plaintext_ciphertext_matrix_multiply_validates_operands(self):
        values = np.ones(self.engine.shape)
        plaintext = self.engine.encode(values)
        ciphertext = self.engine.encrypt(values, self.secret_key)

        with self.assertRaisesRegex(TypeError, "weight must be a GLPlaintext"):
            self.evaluator.pt_ct_matmul(values, ciphertext)
        with self.assertRaisesRegex(TypeError, "ciphertext must be a GLCiphertext"):
            self.evaluator.pt_ct_matmul(plaintext, values)

    def test_matrix_multiply_requires_a_configured_key(self):
        engine = GLEngine(shape=(256, 16, 16), mode="cpu")
        secret_key = engine.create_secret_key()
        values = np.ones(engine.shape)
        evaluator = ThorLinearEvaluator(engine)

        with self.assertRaisesRegex(ValueError, "matrix multiplication key"):
            evaluator.pt_ct_matmul(
                engine.encode(values), engine.encrypt(values, secret_key)
            )

    def test_transpose_plaintext_and_ciphertext(self):
        values = np.arange(np.prod(self.engine.shape), dtype=float).reshape(
            self.engine.shape
        )
        plaintext = self.engine.encode(values)
        ciphertext = self.engine.encrypt(values, self.secret_key)

        transposed_plaintext = self.evaluator.transpose_upper_to_lower(plaintext)
        transposed_ciphertext = self.evaluator.transpose_upper_to_lower(ciphertext)

        self.assertIsInstance(transposed_plaintext, GLPlaintext)
        np.testing.assert_allclose(
            self.engine.decode(transposed_plaintext),
            np.swapaxes(values, -1, -2),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.decrypt(transposed_ciphertext),
            np.swapaxes(values, -1, -2),
            atol=1e-5,
        )

    def test_rotsum_delegates_to_engine(self):
        engine = Mock()
        engine.rotsum.return_value = sentinel.result
        evaluator = ThorLinearEvaluator(engine)

        result = evaluator.rotsum(sentinel.ciphertext, 8)

        self.assertIs(result, sentinel.result)
        engine.rotsum.assert_called_once_with(sentinel.ciphertext, 8)

    def test_ckks_packing_helpers_are_deferred(self):
        deferred_calls = (
            lambda: self.evaluator.parallel_diagonal_pt_ct_mult(None, None),
            lambda: self.evaluator.make_rotated_copies(None),
            lambda: self.evaluator.make_copies(None),
            lambda: self.evaluator.rotate_internal(None),
            self.evaluator.pre_encode_masks,
        )

        for call in deferred_calls:
            with self.subTest(call=call), self.assertRaises(NotImplementedError):
                call()


if __name__ == "__main__":
    unittest.main()
