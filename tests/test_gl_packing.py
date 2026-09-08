import unittest

import numpy as np

from thor_gl import GLEngine, GLTensor


class TestGLEngineMatrixMultiplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Creates one GL engine and its matrix multiplication key for all tests.
        @return: None.
        """
        cls.engine = GLEngine(shape=(256, 16, 16))
        cls.sk = cls.engine.create_secret_key()
        cls.mult_key = cls.engine.create_matrix_multiplication_key(cls.sk)
        cls.tolerance = 1e-5

    def setUp(self):
        """
        Resets NumPy's random generator before each packing test.
        @return: None.
        """
        np.random.seed(42)

    def _physical_array(self, logical):
        """
        Reproduces Desilo's contiguous padding into the physical GL shape.
        @param logical: Logical NumPy array passed to encode or encrypt.
        @return: Zero-padded array with the engine's physical shape.
        """
        logical = np.asarray(logical)
        physical = np.zeros(self.engine.shape, dtype=logical.dtype)
        physical.reshape(-1)[: logical.size] = logical.reshape(-1)
        return physical

    def _multiply(self, a, b):
        """
        Encodes, encrypts, and multiplies two arrays through the GL engine.
        @param a: Plaintext-side logical array.
        @param b: Ciphertext-side logical array.
        @return: Decrypted physical GL multiplication result.
        """
        encoded_a = self.engine.encode(a, level=self.engine.max_level)
        encrypted_b = self.engine.encrypt(b, self.sk, self.engine.max_level)
        result = self.engine.mmult(encoded_a, encrypted_b, self.mult_key)
        return self.engine.decrypt(result, self.sk)

    def test_single_slot_multiplication(self):
        """
        Verifies logical matrix multiplication when both inputs fit one slot.
        @return: None.
        """
        a = np.random.rand(16, 16)
        b = np.random.rand(16, 16)

        decoded = self._multiply(a, b)
        tensor = GLTensor(object(), (16, 16), self.engine.shape)

        np.testing.assert_allclose(
            tensor.to_logical(decoded),
            a @ b,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

    def test_native_padding_round_trip(self):
        """
        Verifies that Desilo restores a non-physical input through GLTensor metadata.
        @return: None.
        """
        logical = np.random.rand(40, 24)
        plaintext = self.engine.encode(logical)
        decoded = self.engine.decode(plaintext)
        tensor = GLTensor(plaintext, logical.shape, self.engine.shape)

        self.assertEqual(decoded.shape, self.engine.shape)
        np.testing.assert_allclose(
            tensor.to_logical(decoded),
            logical,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

    def test_both_matrices_split(self):
        """
        Verifies independent physical-slot products when both inputs are split.
        @return: None.
        """
        a = np.random.rand(40, 24)
        b = np.random.rand(24, 40)
        expected = np.matmul(self._physical_array(a), self._physical_array(b))

        decoded = self._multiply(a, b)

        np.testing.assert_allclose(
            decoded,
            expected,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

    def test_asymmetric_split_uses_independent_slots(self):
        """
        Verifies physical-slot multiplication for asymmetric logical input shapes.
        @return: None.
        """
        a = np.random.rand(1000, 30)
        b = np.random.rand(30, 41)
        expected = np.matmul(self._physical_array(a), self._physical_array(b))

        decoded = self._multiply(a, b)

        np.testing.assert_allclose(
            decoded,
            expected,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

    def test_split_inputs_do_not_form_one_logical_matrix_product(self):
        """
        Documents that native GL multiplication does not accumulate across slots.
        @return: None.
        """
        a = np.random.rand(24, 16)
        b = np.random.rand(16, 24)
        logical_product = a @ b
        decoded = self._multiply(a, b)
        restored = GLTensor(
            object(), logical_product.shape, self.engine.shape
        ).to_logical(decoded)

        self.assertFalse(
            np.allclose(
                restored,
                logical_product,
                rtol=self.tolerance,
                atol=self.tolerance,
            )
        )

    def test_physical_slot_boundaries(self):
        """
        Verifies products immediately before and after physical slot boundaries.
        @return: None.
        """
        a = np.random.rand(33, 17)
        b = np.random.rand(17, 33)
        expected = np.matmul(self._physical_array(a), self._physical_array(b))

        decoded = self._multiply(a, b)

        for slot in range(4):
            with self.subTest(slot=slot):
                np.testing.assert_allclose(
                    decoded[slot],
                    expected[slot],
                    rtol=self.tolerance,
                    atol=self.tolerance,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
