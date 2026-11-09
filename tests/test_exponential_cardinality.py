from decimal import Decimal, localcontext
from itertools import combinations
import unittest

from aggregationslib import aggregation as f, decimalaggregations as d


class TestExponentialCardinality(unittest.TestCase):
    def test_invalid_size_exponent_and_float_policy(self):
        for module in (f, d):
            for n in (0, -1, True, 1.5):
                with self.subTest(module=module.__name__, n=n), self.assertRaises(ValueError):
                    module.exponential_cardinality_measure(n, 1)
            for q in ('0', '-1', 'NaN', 'Infinity', '-Infinity', 'abc'):
                with self.subTest(module=module.__name__, q=q), self.assertRaises(ValueError):
                    module.exponential_cardinality_measure(3, q)
            with self.assertRaises(TypeError):
                module.exponential_cardinality_measure(3, True)
        with self.assertRaises(TypeError):
            d.exponential_cardinality_measure(3, .5)
        with self.assertRaises(TypeError):
            d.exponential_cardinality_measure(3, '0.5', allow_float=1)
        self.assertEqual(d.exponential_cardinality_measure(3, .5, allow_float=True)({0}),
                         d.exponential_cardinality_measure(3, '0.5')({0}))

    def test_q_one_is_uniform_and_measure_is_symmetric(self):
        with localcontext() as ctx:
            ctx.prec = 50
            for module in (f, d):
                for n in (1, 2, 5):
                    uniform = module.uniform_measure(n)
                    power = module.exponential_cardinality_measure(n, 1)
                    half = module.exponential_cardinality_measure(n, '0.5')
                    for size in range(n+1):
                        expected = half(range(size))
                        for subset in combinations(range(n), size):
                            self.assertEqual(power(subset), uniform(subset))
                            self.assertEqual(half(subset), expected)
                    self.assertEqual(half(()), 0)
                    self.assertEqual(half(range(n)), 1)

    def test_integrals_decrease_with_q_and_sugeno_has_plateau(self):
        x = [.2, .5, .9]
        for function in (f.choquet_integral, f.sugeno_integral):
            values = [function(x, f.exponential_cardinality_measure(3, q))
                      for q in (.01, .25, .5, .75, 1, 1.5, 2, 10, 100)]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertTrue(all(min(x) <= value <= max(x) for value in values))
        for q in (.75, 1, 1.5):
            self.assertEqual(f.sugeno_integral(x, f.exponential_cardinality_measure(3, q)), .5)
        measure = f.exponential_cardinality_measure(3, 1)
        self.assertAlmostEqual(f.choquet_integral(x, measure), sum(x)/3)
        self.assertNotAlmostEqual(f.sugeno_integral(x, measure), sum(x)/3)
