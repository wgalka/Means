import math
import unittest

import numpy as np

from aggregationslib import aggregation
from aggregationslib.aggregations import A_ex
from aggregationslib.purepython import exponential as pure_exponential


class Data:
    x = [
        [0, 0, 1, 1],
        [0.5, 0.5, 0.5],
        [0, 1]
    ]


def data_type_test_case(function):
    class TestDatatypes(unittest.TestCase):
        list1d = [1, 2, 3, 4]
        list2d = [[1, 1], [2, 2], [3, 3]]
        tuple1d = (1, 2, 3, 4)
        tuple2d = ((1, 1), (2, 2))

        variable = 1

        nparray1d = np.array(list1d)
        nparray2d = np.array(list2d)

        def test(self):
            self.assertTrue(isinstance(function(self.list1d), float))
            self.assertTrue(isinstance(function(self.tuple1d), float))
            self.assertTrue(isinstance(function(self.nparray1d), float))

            self.assertRaises(ValueError, function, self.variable)

            with self.assertRaises(ValueError):
                function(self.list2d)
                function(self.tuple2d)
                function(self.nparray2d)

    return TestDatatypes


class TestArithmetic(data_type_test_case(aggregation.arithmetic), unittest.TestCase):
    pass


class TestQuadratic(data_type_test_case(aggregation.quadratic)):
    pass


class TestQuasiArithmetic(unittest.TestCase):
    array = [1, 2, 3, 4]
    expected1 = aggregation.sin_aggregation(array)
    result1 = aggregation.quasi_arithmeric(array, math.sin)

    def test(self):
        self.assertAlmostEqual(self.expected1, self.result1, places=9)


class TestExponential(unittest.TestCase):
    def test_normal_values(self):
        y = [1, 2, 3]
        r = 0.5
        # Expected value using original formula manually calculated:
        # A_ex = 1/0.5 * ln( (e^0.5 + e^1.0 + e^1.5) / 3 )
        #      = 2 * ln( (1.6487212707 + 2.7182818285 + 4.4816890703) / 3 )
        #      = 2 * ln( 8.8486921695 / 3 ) = 2.1633147639472496
        expected = 2.1633147639472496
        
        # Test class A_ex
        ae = A_ex(r)
        self.assertAlmostEqual(ae(y), expected, places=7)

        # Test functional exponential
        self.assertAlmostEqual(aggregation.exponential(y, r), expected, places=7)

        # Test pure Python exponential
        self.assertAlmostEqual(pure_exponential(y, r), expected, places=7)

    def test_overflow_stability(self):
        # Without Log-Sum-Exp, r*y_i > 709.78 raises Overflow or returns inf/nan
        y = [800.0, 800.0, 800.0]
        r = 1.0
        # The exponential mean of identical values should be that value (800.0)
        ae = A_ex(r)
        self.assertAlmostEqual(ae(y), 800.0, places=7)
        self.assertAlmostEqual(aggregation.exponential(y, r), 800.0, places=7)
        self.assertAlmostEqual(pure_exponential(y, r), 800.0, places=7)

    def test_underflow_stability(self):
        # Without Log-Sum-Exp, r*y_i < -709.78 returns -inf (ln of 0)
        y = [-800.0, -800.0, -800.0]
        r = 1.0
        # The exponential mean of identical values should be that value (-800.0)
        ae = A_ex(r)
        self.assertAlmostEqual(ae(y), -800.0, places=7)
        self.assertAlmostEqual(aggregation.exponential(y, r), -800.0, places=7)
        self.assertAlmostEqual(pure_exponential(y, r), -800.0, places=7)

    def test_invalid_r(self):
        y = [1, 2, 3]
        with self.assertRaises(ValueError):
            A_ex(0)(y)
        with self.assertRaises(ValueError):
            aggregation.exponential(y, 0)
        with self.assertRaises(ValueError):
            pure_exponential(y, 0)


if __name__ == '__main__':
    unittest.main()
