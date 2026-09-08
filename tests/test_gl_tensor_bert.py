import unittest
from unittest.mock import Mock

import numpy as np
from desilofhe import GLCiphertext, GLPlaintext

from thor_gl import (
    BertGLLayout,
    GLEngine,
    GLTensor,
    SUPPORTED_GL_SHAPES,
    ThorBert,
    ThorLinearEvaluator,
)


class GLTensorTest(unittest.TestCase):
    def test_supported_shapes_are_declared(self):
        self.assertEqual(
            SUPPORTED_GL_SHAPES,
            {
                (256, 16, 16),
                (16, 256, 256),
                (4, 1024, 1024),
                (256, 32, 32),
                (16, 512, 512),
                (4, 2048, 2048),
                (256, 64, 64),
            },
        )
        with self.assertRaisesRegex(ValueError, "Unsupported GL physical shape"):
            GLTensor(object(), (2, 2), (1, 2, 2))

    def test_native_padding_restores_rank_two_logical_shape(self):
        engine = GLEngine(shape=(256, 16, 16), mode="cpu")
        logical = np.arange(24 * 16, dtype=float).reshape(24, 16) / 100
        plaintext = engine.encode(logical)
        tensor = GLTensor(plaintext, logical.shape, engine.shape)

        decoded = engine.decode(plaintext)

        self.assertEqual(decoded.shape, engine.shape)
        np.testing.assert_allclose(tensor.to_logical(decoded), logical, atol=1e-6)

    def test_native_padding_restores_rank_three_logical_shape(self):
        engine = GLEngine(shape=(256, 16, 16), mode="cpu")
        logical = np.arange(2 * 10 * 20, dtype=float).reshape(2, 10, 20) / 100
        secret_key = engine.create_secret_key()
        ciphertext = engine.encrypt(logical, secret_key, level=1)
        tensor = GLTensor(ciphertext, logical.shape, engine.shape)

        decrypted = engine.decrypt(ciphertext, secret_key)

        self.assertIsInstance(ciphertext, GLCiphertext)
        np.testing.assert_allclose(tensor.to_logical(decrypted), logical, atol=1e-6)

    def test_plaintext_and_ciphertext_payload_metadata(self):
        engine = GLEngine(shape=(256, 16, 16), mode="cpu")
        values = np.ones((3, 5))
        secret_key = engine.create_secret_key()
        plaintext = GLTensor(engine.encode(values), values.shape, engine.shape)
        ciphertext = GLTensor(
            engine.encrypt(values, secret_key), values.shape, engine.shape
        )

        self.assertIsInstance(plaintext.payload, GLPlaintext)
        self.assertIsInstance(ciphertext.payload, GLCiphertext)

    def test_physical_capacity_and_decoded_shape_are_enforced(self):
        physical_shape = (256, 64, 64)
        logical_shape = (1, 768, 3072)

        with self.assertRaisesRegex(ValueError, "holds only"):
            GLTensor(object(), logical_shape, physical_shape)
        tensor = GLTensor(object(), (24, 16), (256, 16, 16))
        with self.assertRaisesRegex(ValueError, "does not match physical shape"):
            tensor.to_logical(np.zeros((24, 16)))


class BertGLLayoutTest(unittest.TestCase):
    def setUp(self):
        self.layout = BertGLLayout()

    def test_expected_bert_slot_counts(self):
        self.assertEqual(self.layout.block_count(self.layout.HIDDEN_SHAPE), 24)
        self.assertEqual(self.layout.block_count(self.layout.HEAD_SHAPE), 24)
        self.assertEqual(self.layout.block_count(self.layout.ATTENTION_SCORE_SHAPE), 48)
        self.assertEqual(self.layout.block_count(self.layout.INTERMEDIATE_SHAPE), 96)
        self.assertEqual(
            self.layout.block_count(self.layout.FEED_FORWARD_WEIGHT_SHAPE), 576
        )

    def test_hidden_and_head_permutations_are_deferred(self):
        hidden = GLTensor(object(), self.layout.HIDDEN_SHAPE, self.layout.physical_shape)
        heads = GLTensor(object(), self.layout.HEAD_SHAPE, self.layout.physical_shape)

        with self.assertRaisesRegex(NotImplementedError, "split_heads"):
            self.layout.split_heads(hidden)
        with self.assertRaisesRegex(NotImplementedError, "merge_heads"):
            self.layout.merge_heads(heads)

    def test_bert_preserves_twelve_layers_and_defers_forward(self):
        engine = Mock()
        engine.shape = self.layout.physical_shape
        evaluator = ThorLinearEvaluator(engine)
        model = ThorBert(evaluator, {})
        hidden = GLTensor(object(), self.layout.HIDDEN_SHAPE, engine.shape)
        mask = GLTensor(object(), self.layout.ATTENTION_MASK_SHAPE, engine.shape)

        self.assertEqual(model.n_layers, 12)
        self.assertEqual(len(model.attentions), 12)
        self.assertEqual(len(model.ffs), 12)
        with self.assertRaisesRegex(NotImplementedError, "BERT forward"):
            model.forward(hidden, mask)

    def test_bert_rejects_a_different_physical_shape(self):
        engine = Mock()
        engine.shape = (16, 512, 512)

        with self.assertRaisesRegex(ValueError, "requires physical shape"):
            ThorBert(ThorLinearEvaluator(engine), {})


if __name__ == "__main__":
    unittest.main()
