import math
import unittest
from decimal import Decimal

import numpy as np

from aggregationslib.decimalaggregations import exponential
from aggregationslib.aggregation import exponential as exponential2
from aggregationslib.purepython import exponential as exponential3


class TestDecimalExponentialMean(unittest.TestCase):
    def setUp(self):
        pass

    def test_r(self):
        bad_r = 0
        good_r_list = [-10, -5.0, -0.0003, 0.00000000000000000000000000000003, 1, 2, 5.5555, 10000]
        data1 = [1, 2, 3]
        with self.assertRaises(ValueError):
            exponential(y=data1, r=bad_r)
        for r in good_r_list:
            exponential(y=data1, r=r)

    def test_xd_arrays(self):
        _1D = [1, 23, 4]
        _2D = [[1], [2], [3]]
        _3D = [[[3]], [[3]], [[3]]]
        np_1D = np.array(_1D)

        exponential(y=_1D)
        exponential(y=np_1D)
        self.assertEqual(exponential(y=_1D), exponential(y=np_1D))
        with self.assertRaises(ValueError):
            exponential(y=_2D)
        with self.assertRaises(ValueError):
            exponential(y=_3D)

    def test_aggregation_results(self):
        # Should return the same value
        t_1_1 = [0, 0, 0, 0]
        self.assertEqual(exponential(t_1_1), 0)
        t_1_2 = [0.0, 0.0, 0, 0]
        self.assertEqual(exponential(t_1_2), 0)
        t_1_3 = [1, 1, 1, 1]
        # Rouding result return 1
        # self.assertEqual(exponential(t_1_3), Decimal(str(1)))
        self.assertEqual(float(exponential(t_1_3)), Decimal(str(1)))
        t_1_4 = [1.543, 1.543, 1.543, 1.543]
        self.assertEqual(exponential(t_1_4), Decimal(str(1.543)))
        t_1_5 = [-10, -10, -10, - 10]
        self.assertEqual(exponential(t_1_5), -10)


if __name__ == '__main__':
    unittest.main()
