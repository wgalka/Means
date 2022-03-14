import math
import unittest

import numpy as np

from aggregationslib import aggregation


class Data:
    x = [
        [0, 0, 1, 1],
        [0.5, 0.5, 0.5],
        [0, 1]
    ]


def test_data_types(function):
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


class TestArithmetic(test_data_types(aggregation.arithmetic), unittest.TestCase):
    pass


class TestQuadratic(test_data_types(aggregation.quadratic)):
    pass


class TestQuasiArithmetic(unittest.TestCase):
    array = [1, 2, 3, 4]
    expected1 = aggregation.sin_aggregation(array)
    result1 = aggregation.quasi_arithmeric(array, math.sin)

    def test(self):
        self.assertAlmostEqual(self.expected1, self.result1, places=9)
