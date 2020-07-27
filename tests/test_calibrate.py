import unittest

import numpy as np

from genomatnn import calibrate


class TestUpsample(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def test_upsample_indexes(self):
        for a in (
            [0],
            [1] * 4 + [2] * 4,
            [0] * 4 + [1] * 2 + [2] * 4 + [40] * 40,
        ):
            b = np.array(a)
            self.rng.shuffle(b)
            upidx = calibrate.resample_indexes(b)
            np.testing.assert_array_equal(np.unique(a), np.unique(b[upidx]))
            _, counts = np.unique(b[upidx], return_counts=True)
            self.assertEqual(len(np.unique(counts)), 1)

    def test_upsample_indexes_weighted(self):
        a = [1] * 100 + [2] * 200 + [5] * 500
        for weights in (
            {1: 0.1, 2: 0.1, 5: 1},
            {1: 1, 2: 0.1, 5: 0.1},
            {1: 0.1, 2: 0.2, 5: 0.3},
        ):
            b = np.array(a)
            self.rng.shuffle(b)
            upidx = calibrate.resample_indexes(b, weights)
            unique, counts = np.unique(b[upidx], return_counts=True)
            np.testing.assert_array_equal(np.unique(a), unique)
            w = np.array([weights[u] for u in unique])
            np.testing.assert_almost_equal(w / w.sum(), counts / counts.sum())
            print(weights, counts)
