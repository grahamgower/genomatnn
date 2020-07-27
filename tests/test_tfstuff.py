import unittest
import tempfile

import numpy as np
import tensorflow as tf

from genomatnn import (
    convert,
    tfstuff,
)
import tests


class TestModelConstruction(unittest.TestCase):
    def setUp(self):
        sample_counts = tests.HashableDict(YRI=10, CHB=20, CEU=30, Papuan=40)
        ts, model = tests.basic_sim(sample_counts)
        _, pop_indices = convert.ts_pop_counts_indices(ts)
        rng = np.random.default_rng(seed=31415)
        maf_thres = 0.05
        num_rows = 32
        num_inds = ts.num_samples
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_file = f"{tmpdir}/foo.trees"
            ts.dump(ts_file)
            A, _ = convert.ts_genotype_matrix(
                ts_file, pop_indices, 0, num_rows, num_inds, maf_thres, rng
            )
        self.assertEqual(A.shape, (num_rows, num_inds))
        self.A = A[np.newaxis, :, :, np.newaxis]
        self.pop_indices = pop_indices

    def test_basic_cnn(self):
        n_conv, n_dense = 3, 1
        cnn = tfstuff.basic_cnn(
            input_shape=self.A.shape[1:],
            output_shape=1,
            n_conv=n_conv,
            n_conv_filt=16,
            filt_size_x=4,
            filt_size_y=4,
            n_dense=n_dense,
            dense_size=4,
        )
        self.assertGreater(len(cnn.layers), 3)
        self.assertTrue(isinstance(cnn.layers[0], tf.keras.layers.InputLayer))
        self.assertTrue(isinstance(cnn.layers[1], tf.keras.layers.BatchNormalization))
        self.assertTrue(isinstance(cnn.layers[-1], tf.keras.layers.Dense))
        conv_layers = [
            layer for layer in cnn.layers if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        self.assertEqual(len(conv_layers), n_conv)
        dense_layers = [
            layer for layer in cnn.layers if isinstance(layer, tf.keras.layers.Dense)
        ]
        # expect n_dense + 1 because the output layer is a dense layer
        self.assertEqual(len(dense_layers), n_dense + 1)

    def test_permutation_invariant_cnn(self):
        n_conv, n_dense = 3, 1
        cnn = tfstuff.permutation_invariant_cnn(
            input_shape=self.A.shape[1:],
            output_shape=1,
            n_conv=n_conv,
            n_conv_filt=16,
            filt_size=4,
            n_dense=n_dense,
            dense_size=4,
        )
        self.assertGreater(len(cnn.layers), 3)
        self.assertTrue(isinstance(cnn.layers[0], tf.keras.layers.InputLayer))
        self.assertTrue(isinstance(cnn.layers[1], tf.keras.layers.BatchNormalization))
        self.assertTrue(isinstance(cnn.layers[-1], tf.keras.layers.Dense))

    def test_per_population_permutation_invariant_cnn(self):
        n_conv, n_dense = 3, 1
        num_inds = self.A.shape[2]
        pop_starts = list(self.pop_indices.values())
        pop_ends = pop_starts[1:] + [num_inds]
        cnn = tfstuff.per_population_permutation_invariant_cnn(
            input_shape=self.A.shape[1:],
            output_shape=1,
            pop_starts=pop_starts,
            pop_ends=pop_ends,
            n_conv=n_conv,
            n_conv_filt=16,
            filt_size=6,
            n_dense=n_dense,
            dense_size=4,
        )
        self.assertGreater(len(cnn.layers), 2)
        self.assertTrue(isinstance(cnn.layers[0], tf.keras.layers.InputLayer))
        self.assertTrue(isinstance(cnn.layers[-1], tf.keras.layers.Dense))
