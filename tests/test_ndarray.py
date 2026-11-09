import itertools
import json
from pathlib import Path
import unittest

import numpy as np

from aggregationslib import aggregations as vector
from aggregationslib import ndarray as nd
from aggregationslib import (
    k_fuzzy_measure, owa_measure, relative_measure, uniform_measure, weighted_measure,
)
from aggregationslib.aggregation import FuzzyMeasure


class TestBoundaryNdarray(unittest.TestCase):
    """Boundary and numerical regression cases, run before ordinary cases."""

    def test_zero_sensitive_means_keep_rows_independent(self):
        x = np.array([[0., 0., 0.], [0., .25, 1.], [.25, .5, 1.]])
        for name, kwargs in [('A_hm', {}), ('A_pw', {'r': -2}), ('A_lm', {'r': 1})]:
            with self.subTest(name=name):
                expected = [getattr(vector, name)(**kwargs)(row) for row in x]
                result = getattr(nd, name)(axis=1, **kwargs)(x)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(result[:2], [0., 0.])
                self.assertGreater(result[2], 0.)

    def test_exponential_extremes_with_positive_and_negative_r(self):
        x = np.array([[800., 800., 800.], [-800., -800., -800.],
                      [-800., 0., 800.], [1e-12, 2e-12, 3e-12]])
        for r in (1., -1.):
            with self.subTest(r=r):
                expected = [vector.A_ex(r)(row) for row in x]
                actual = nd.A_ex(r, axis=1)(x)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.all(np.isfinite(actual)))
                np.testing.assert_allclose(actual[:2], [800., -800.], rtol=0, atol=1e-12)

    def test_trimming_thresholds_and_logarithmic_warning(self):
        for name, kwargs, widths in [('A_ol', {}, (1, 2, 3, 5)),
                                     ('A_oln', {'p': 2}, (1, 4, 5, 7))]:
            for width in widths:
                row = np.linspace(.1, .9, width)
                x = np.stack([row[::-1], np.full(width, .5), row])
                with self.subTest(name=name, width=width):
                    np.testing.assert_array_equal(
                        getattr(nd, name)(axis=1, **kwargs)(x),
                        [getattr(vector, name)(**kwargs)(v) for v in x])
        # Preserve the documented 1D fallback rather than inventing a new
        # value for the singular logarithmic expression in the array wrapper.
        with self.assertWarnsRegex(RuntimeWarning, 'zero denominator'):
            expected = [vector.A_lo()(row) for row in np.array([[2., 2.], [2., 8.]])]
        with self.assertWarnsRegex(RuntimeWarning, 'zero denominator'):
            actual = nd.A_lo(axis=1)(np.array([[2., 2.], [2., 8.]]))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0], 0.)

    def test_integral_boundaries_ties_and_all_measure_formats(self):
        full = {frozenset(i for i in range(3) if mask & (1 << i)): value
                for mask, value in enumerate([0., .2, .3, .6, .4, .7, .8, 1.])}
        measures = [None, uniform_measure(3), relative_measure(3),
                    weighted_measure([.5, .3, .2]), owa_measure([.6, .3, .1]),
                    k_fuzzy_measure([.2, .2, .2]),
                    k_fuzzy_measure([.6, .6, .6]),
                    full, FuzzyMeasure(3, full), [0., .1, .6, 1.]]
        x = np.array([[0., 0., 0.], [1., 1., 1.], [.37, .37, .37],
                      *itertools.permutations([.2, .2, .9]),
                      [0., np.nextafter(0., 1.), 1.],
                      [.5, np.nextafter(.5, 1.), np.nextafter(1., 0.)]])
        for name in ('A_ch', 'A_sg'):
            for index, measure in enumerate(measures):
                with self.subTest(name=name, measure=index):
                    expected = [getattr(vector, name)(measure)(row) for row in x]
                    result = getattr(nd, name)(measure, axis=1)(x)
                    np.testing.assert_array_equal(result, expected)
                    np.testing.assert_allclose(result[:3], [0., 1., .37], rtol=0, atol=1e-15)
                    self.assertTrue(np.all(result >= x.min(axis=1) - 1e-15))
                    self.assertTrue(np.all(result <= x.max(axis=1) + 1e-15))
        # Independent analytic oracle: integral of an indicator equals μ(A).
        indicators = np.array([[float(bool(mask & (1 << i))) for i in range(3)]
                               for mask in range(8)])
        for name in ('A_ch', 'A_sg'):
            np.testing.assert_array_equal(getattr(nd, name)(full, axis=1)(indicators),
                                          list(full.values()))

    def test_integrals_match_saved_wolfram_results(self):
        path = Path(__file__).parent / 'data' / 'wolfram_integral_reference.json'
        cases = json.loads(path.read_text(encoding='utf-8'))['cases']
        for record in cases:
            n = record['n']
            measure = {frozenset(i for i in range(n) if mask & (1 << i)): value
                       for mask, value in enumerate(record['capacityByMask'])}
            # Each reference has its own measure; put its vector in a batch
            # with the two normalized endpoints, rather than mixing measures.
            x = np.array([record['values'], [0.] * n, [1.] * n])
            for name, field in [('A_ch', 'choquet'), ('A_sg', 'sugeno')]:
                with self.subTest(case=record['id'], integral=name):
                    actual = getattr(nd, name)(measure, axis=1)(x)
                    np.testing.assert_array_equal(actual, [getattr(vector, name)(measure)(row) for row in x])
                    np.testing.assert_allclose(actual, [record[field], 0., 1.], rtol=0, atol=1e-12)

    def test_invalid_row_propagates_vector_error(self):
        for name in ('A_ch', 'A_sg'):
            for bad in (np.nan, np.inf, -np.inf, -.01, 1.01, 1j):
                x = np.array([[.1, .2, .3], [.1, bad, .3]])
                with self.subTest(name=name, value=bad):
                    with self.assertRaises(ValueError) as error:
                        getattr(vector, name)()(x[1])
                    with self.assertRaises(ValueError) as array_error:
                        getattr(nd, name)(axis=1)(x)
                    self.assertEqual(str(error.exception), str(array_error.exception))
        for name in ('A_ex', 'A_pw'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                getattr(nd, name)(r=0, axis=1)(np.ones((2, 3)))

    def test_nonlinear_scalar_callback_and_callback_error(self):
        def overlap(x, mu):
            # Scalar branching must work; this differs from the default x*μ.
            return x if x < mu else mu
        x = np.array([[[.2, .8, .5], [.9, .1, .7]],
                      [[.4, .4, .4], [0., 1., .6]]])
        measure = weighted_measure([.5, .3, .2])
        expected = [[vector.A_ch(measure, quasi_overlap=overlap)(row)
                     for row in plane] for plane in x]
        result = nd.A_ch(measure, quasi_overlap=overlap, axis=-1)(x)
        np.testing.assert_array_equal(result, expected)
        self.assertFalse(np.array_equal(result, nd.A_ch(measure, axis=-1)(x)))
        def broken_overlap(x, mu):
            raise RuntimeError('callback failure')
        with self.assertRaisesRegex(RuntimeError, 'callback failure'):
            nd.A_ch(quasi_overlap=broken_overlap, axis=-1)(x)

    def test_flatten_order_dtypes_and_input_not_modified(self):
        for dtype in (np.int64, np.float32, np.float64):
            x = np.arange(1, 25).astype(dtype).reshape(2, 3, 4)[:, ::-1, ::-1]
            original = x.copy()
            for name in ('A_ar', 'A_pr', 'A_md'):
                for axis in (None, 0, 1, -1):
                    for keepdims in (False, True):
                        with self.subTest(dtype=dtype, name=name, axis=axis, keepdims=keepdims):
                            operation = getattr(vector, name)()
                            expected = (operation(x.flatten(order='C')) if axis is None
                                        else np.apply_along_axis(operation, axis, x))
                            actual = getattr(nd, name)(axis=axis, keepdims=keepdims)(x)
                            shape = ((1,) * x.ndim if axis is None else
                                     tuple(1 if i == axis % x.ndim else s for i, s in enumerate(x.shape)))
                            if keepdims:
                                expected = np.asarray(expected).reshape(shape)
                            self.assertEqual(np.shape(actual), np.shape(expected))
                            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(x, original)
        x = np.asfortranarray([[.1, .2], [.3, .4]])
        mu = weighted_measure([.1, .2, .3, .4])
        self.assertEqual(nd.A_ch(mu)(x), vector.A_ch(mu)(x.flatten(order='C')))


class TestNdarray(unittest.TestCase):
    def test_shapes_and_axes_match_numpy(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        for axis in (None, 0, 1, 2, -1, -3):
            for keepdims in (False, True):
                with self.subTest(axis=axis, keepdims=keepdims):
                    actual = nd.A_ar(axis=axis, keepdims=keepdims)(x)
                    expected = np.mean(x, axis=axis, keepdims=keepdims)
                    self.assertEqual(np.shape(actual), np.shape(expected))
                    np.testing.assert_allclose(actual, expected)
        self.assertEqual(nd.A_ar(axis=1)(x[0:1, :, 0]).shape, (1,))
        self.assertTrue(np.isscalar(nd.A_ar(axis=0)(x[0, 0])))
        self.assertEqual(nd.A_ar()(3.0), 3.0)

    def test_all_vector_objects_are_available_and_reused(self):
        params = dict(A_pw={'r': 2}, A_ex={'r': 1}, A_ex2={'p': 2, 'q': 2},
                      A_ex3={'p': 2, 'q': 1.5}, A_lm={'r': 2},
                      A_amn={'p': .3}, A_amx={'p': .3}, A_oln={'p': 1})
        names = [name for name in vars(vector) if name.startswith('A_')]
        x = np.linspace(.1, .9, 24).reshape(2, 3, 4)
        for name in names:
            kwargs = params.get(name, {})
            operation = getattr(vector, name)(**kwargs)
            with self.subTest(name=name):
                # A middle-axis reduction on a non-contiguous view checks
                # which coordinates form each vector and their output order.
                view = x[:, :, ::-1]
                expected = np.array([[operation(view[i, :, j])
                                      for j in range(4)] for i in range(2)])
                np.testing.assert_array_equal(
                    getattr(nd, name)(axis=1, **kwargs)(view), expected)

    def test_zero_in_one_row_does_not_affect_other_rows(self):
        x = np.array([[0., 2., 4.], [1., 2., 4.]])
        np.testing.assert_allclose(nd.A_hm(axis=1)(x), [0., 12 / 7])

    def test_integrals_keep_original_coordinate_weights(self):
        measure = weighted_measure([.5, .3, .2])
        x = np.array([[.2, .6, .9], [.9, .2, .6]])
        np.testing.assert_allclose(nd.A_ch(measure, axis=1)(x), [.46, .63])
        np.testing.assert_allclose(nd.A_sg(measure, axis=1)(x), [.5, .6])
        with self.assertRaises(ValueError):
            nd.A_ch(measure, axis=0)(x)
        with self.assertRaises(ValueError):
            nd.A_sg(measure)(x)

    def test_scalar_callback_is_supported(self):
        def scalar_overlap(x, mu):
            if x == 0:
                return 0.0
            return x * mu
        x = np.array([[.2, .6, .9], [.9, .2, .6]])
        np.testing.assert_allclose(
            nd.A_ch(axis=1, quasi_overlap=scalar_overlap)(x), x.mean(axis=1))

    def test_combination_uses_outer_axis(self):
        x = np.arange(12.0).reshape(3, 4)
        operation = nd.Combine2Aggregations(nd.A_ar(axis=1), min, p=.25,
                                           axis=1, keepdims=True)
        expected = .25 * x.mean(axis=1) + .75 * x.min(axis=1)
        np.testing.assert_allclose(operation(x), expected[:, None])

    def test_empty_dimensions_and_invalid_configuration(self):
        for x, axis in [(np.empty((2, 0)), 1), (np.empty((0, 2)), None)]:
            with self.assertRaises(ValueError):
                nd.A_ar(axis=axis)(x)
        self.assertEqual(nd.A_ar(axis=1)(np.empty((0, 3))).shape, (0,))
        self.assertEqual(nd.A_ar(axis=0, keepdims=True)(np.empty((3, 0))).shape, (1, 0))
        for axis in [2, -3]:
            with self.assertRaises(ValueError):
                nd.A_ar(axis=axis)(np.ones((2, 3)))
        for axis in [True, 1.5, (0, 1)]:
            with self.assertRaises(TypeError):
                nd.A_ar(axis=axis)
        with self.assertRaises(TypeError):
            nd.A_ar(keepdims='yes')

    def test_metadata(self):
        operation = nd.A_pw(r=2, axis=-1, keepdims=True)
        self.assertEqual(operation.name, 'pw')
        self.assertEqual(operation.toLatex('x'), vector.A_pw(2).toLatex('x'))
        self.assertIn('axis=-1', repr(operation))


if __name__ == '__main__':
    unittest.main()
