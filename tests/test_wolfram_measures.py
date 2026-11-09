"""Independent, saved Wolfram results; running tests does not require Wolfram.

Fixtures use exact rational inputs and symbolic lambda roots. Choquet was
computed through distinct levels and Sugeno by enumerating all subsets.
Float comparisons allow 1e-12 absolute error; Decimal comparisons use 50
working digits with 1e-44 absolute error, leaving six guard digits.
"""

import json
from decimal import Decimal, localcontext
from pathlib import Path
import unittest

import numpy as np

from aggregationslib import aggregation as functional
from aggregationslib import aggregations as vector
from aggregationslib import decimalaggregations as exact
from aggregationslib import ndarray as nd


REFERENCE = Path(__file__).parent / 'data' / 'wolfram_measure_reference.json'
FLOAT_ATOL = 1e-12
DECIMAL_ATOL = Decimal('1e-44')


def make_measure(module, record, convert):
    constructor = getattr(module, record['kind'] + '_measure')
    if record['kind'] == 'exponential_cardinality':
        return constructor(record['n'], convert(record['q']))
    if record['kind'] in ('uniform', 'relative'):
        return constructor(record['n'])
    return constructor([convert(value) for value in record['weights']])


class TestWolframMeasures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
        cls.records = cls.reference['measures']

    def test_reference_contains_all_families_and_boundary_regimes(self):
        self.assertEqual(self.reference['referenceEngine'], 'Wolfram Language')
        self.assertEqual({r['kind'] for r in self.records},
                         {'uniform', 'relative', 'weighted', 'owa', 'k_fuzzy', 'exponential_cardinality'})
        ids = {r['id'] for r in self.records}
        self.assertTrue({'lambda_positive', 'lambda_negative', 'lambda_additive',
                         'lambda_minus_one_extension', 'lambda_dirac',
                         'lambda_near_additive_above', 'lambda_near_additive_below'} <= ids)
        for record in self.records:
            self.assertEqual(len(record['capacityByMask']), 2 ** record['n'])
            self.assertGreater(len(record['cases']), 0)

    def test_measure_values_for_every_subset_float_and_decimal(self):
        with localcontext() as context:
            context.prec = 50
            for record in self.records:
                for module, convert, tolerance in ((functional, float, FLOAT_ATOL),
                                                    (exact, Decimal, DECIMAL_ATOL)):
                    with self.subTest(measure=record['id'], arithmetic=convert.__name__):
                        measure = make_measure(module, record, convert)
                        for mask, expected in enumerate(record['capacityByMask']):
                            subset = frozenset(i for i in range(record['n']) if mask & (1 << i))
                            with self.subTest(mask=mask):
                                actual = measure(subset)
                                if convert is Decimal:
                                    self.assertIsInstance(actual, Decimal)
                                self.assertLessEqual(abs(actual - convert(expected)), tolerance)

    def test_integral_functions_and_vector_objects_float_and_decimal(self):
        with localcontext() as context:
            context.prec = 50
            for record in self.records:
                for module, objects, convert, tolerance in (
                    (functional, vector, float, FLOAT_ATOL),
                    (exact, exact, Decimal, DECIMAL_ATOL),
                ):
                    measure = make_measure(module, record, convert)
                    for index, case in enumerate(record['cases']):
                        values = [convert(value) for value in case['values']]
                        for field, cls in [('choquet', 'A_ch'), ('sugeno', 'A_sg')]:
                            with self.subTest(measure=record['id'], case=index,
                                              integral=field, arithmetic=convert.__name__):
                                expected = convert(case[field])
                                function_result = getattr(module, field + '_integral')(values, measure)
                                object_result = getattr(objects, cls)(measure)(values)
                                for result in (function_result, object_result):
                                    if convert is Decimal:
                                        self.assertIsInstance(result, Decimal)
                                    self.assertLessEqual(abs(result - expected), tolerance)

    def test_ndarray_integrals_on_rows_columns_and_3d(self):
        for record in self.records:
            measure = make_measure(functional, record, float)
            x = np.array([[float(v) for v in case['values']] for case in record['cases']])
            for field, cls in [('choquet', nd.A_ch), ('sugeno', nd.A_sg)]:
                expected = np.array([float(case[field]) for case in record['cases']])
                with self.subTest(measure=record['id'], integral=field):
                    np.testing.assert_allclose(cls(measure, axis=1)(x), expected,
                                               rtol=0, atol=FLOAT_ATOL)
                    np.testing.assert_allclose(cls(measure, axis=0)(x.T), expected,
                                               rtol=0, atol=FLOAT_ATOL)
                    batch = np.stack([x, x[::-1]], axis=1)
                    result = cls(measure, axis=-1, keepdims=True)(batch)
                    expected_batch = np.stack([expected, expected[::-1]], axis=1)[..., None]
                    self.assertEqual(result.shape, expected_batch.shape)
                    np.testing.assert_allclose(result, expected_batch, rtol=0, atol=FLOAT_ATOL)


if __name__ == '__main__':
    unittest.main()
