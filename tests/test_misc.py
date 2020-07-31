import unittest

import numpy as np

from genomatnn.misc import gt_bytes2vec


class TestMisc(unittest.TestCase):
    def test_gt_bytes2vec(self):
        for b, v, allele_counts in [
            (b"0/0\n", [0, 0], [2, 0, 0]),
            (b"0|0\n", [0, 0], [2, 0, 0]),
            (b"1/0\n", [1, 0], [1, 1, 0]),
            (b"0|1\n", [0, 1], [1, 1, 0]),
            (b"./0\n", [2, 0], [1, 0, 1]),
            (b"0|.\n", [0, 2], [1, 0, 1]),
            (
                b"0/0\t0|0\t1/0\t0|1\t./0\t0|.\n",
                [0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2],
                [8, 2, 2],
            ),
        ]:
            gt, ac = gt_bytes2vec(b)
            np.testing.assert_array_equal(gt, v)
            np.testing.assert_array_equal(ac, allele_counts)
