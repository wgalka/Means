"""Verify exported example code against the app, including CSV round trips."""
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from apps.shiny.core import AGGREGATIONS, calculate, header_weight_vector, grid_view


class TestShinyApp(unittest.TestCase):
    def test_sort_all_rows_before_head_tail_and_toggle_full_view(self):
        df = pd.DataFrame({'x': [8, 7, 6, 5, 4, 0, 19, 1, 18, 2, 17, 3, 16, 9, 10, 11, 12, 13, 14, 15]})
        for direction in ('asc', 'desc'):
            with self.subTest(direction=direction):
                order = {'index': 1, 'direction': direction}
                short = grid_view(df, sort=order)
                expected = sorted(df.x, reverse=direction == 'desc')
                self.assertEqual(list(short.x), expected[:5] + ['…'] + expected[-5:])
                self.assertEqual(list(grid_view(df, full=True, sort=order).x), expected)
                self.assertEqual(list(short.iloc[5]), ['…', '…'])
        self.assertEqual(list(grid_view(df).x), list(df.x[:5]) + ['…'] + list(df.x[-5:]))
        self.assertEqual(len(grid_view(df.head(10))), 10)
        self.assertEqual(len(df), 20)

    def test_header_weights_follow_selected_columns_and_owa_ranks(self):
        values = {'z': '.7', 'x': '.3', 'unused': '.9'}
        self.assertEqual(header_weight_vector('weighted', ['x', 'z'], values), '0.3 0.7')
        self.assertEqual(header_weight_vector('k_fuzzy', ['z'], values), '0.7')
        self.assertEqual(header_weight_vector('owa', ['z', 'x'], {'1': '.8', '2': '.2'}), '0.8 0.2')
        for value in ('', 'nan', 'inf', '-.1', '1.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                header_weight_vector('weighted', ['x'], {'x': value})
        with self.assertRaisesRegex(ValueError, 'column x'):
            header_weight_vector('weighted', ['x'], {})
        with self.assertRaisesRegex(ValueError, 'rank 1'):
            header_weight_vector('owa', ['x'], {})

    def test_example_calculates_all_rows(self):
        sample = Path(__file__).parents[1] / 'apps' / 'shiny' / 'example.csv'
        df = pd.read_csv(sample)
        result, _ = calculate(df, list(df.columns), '', 'A_ar', {})
        self.assertEqual(len(result), 20)
        np.testing.assert_allclose(result.A_ar, df.mean(axis=1))

    def test_generated_code_matches_every_aggregation(self):
        df = pd.DataFrame({'id': ['a', 'b'], 'x': [.2, .4], 'y': [.7, .8]})
        for name in AGGREGATIONS:
            with self.subTest(name=name):
                params = {'r': 2, 'p': 2, 'q': 2}
                if name in ('A_amn', 'A_amx'):
                    params['p'] = .4
                if name == 'A_oln':
                    params['p'] = 1
                result, code = calculate(df, ['x', 'y'], '.1 .9', name, params)
                captured = []
                with patch.object(pd.DataFrame, 'to_csv', lambda self, *a, **k: captured.append(self.copy())):
                    exec(code, {})
                pd.testing.assert_frame_equal(captured[0], result)

    def test_integrals_with_all_measures_and_semicolon_csv(self):
        df = pd.DataFrame({'x': [.2, .7], 'y': [.8, .4]})
        for name in ('A_ch', 'A_sg'):
            for measure in ('uniform', 'relative', 'weighted', 'owa', 'k_fuzzy', 'exponential_cardinality'):
                with self.subTest(name=name, measure=measure):
                    result, code = calculate(df, ['x', 'y'], '', name, {'measure_q': .5}, measure, '.3 .7', ';', ',', True, True)
                    with tempfile.TemporaryDirectory() as directory:
                        source = Path(directory) / 'input.csv'
                        target = Path(directory) / 'results.csv'
                        df.to_csv(source, sep=';', decimal=',', index=False)
                        code = code.replace("'input.csv'", repr(str(source))).replace("'results.csv'", repr(str(target)))
                        exec(code, {})
                        pd.testing.assert_frame_equal(pd.read_csv(target, sep=';', decimal=','), result)

    def test_invalid_data_and_parameters(self):
        df = pd.DataFrame({'x': [.2], 'y': [.8]})
        for columns, extra, name, params, kind, weights in [
            ([], '', 'A_ar', {}, 'uniform', ''),
            (['x', 'y'], '.1', 'A_ar', {}, 'uniform', ''),
            (['x'], 'nan', 'A_ar', {}, 'uniform', ''),
            (['x'], '', 'A_pw', {'r': 0}, 'uniform', ''),
            (['x'], '', 'A_oln', {'p': .5}, 'uniform', ''),
            (['x', 'y'], '', 'A_ch', {}, 'weighted', '.5'),
            (['x', 'y'], '', 'A_ch', {}, 'weighted', '.5 .8'),
        ]:
            with self.subTest(name=name, columns=columns, weights=weights, extra=extra):
                with self.assertRaises(ValueError):
                    calculate(df, columns, extra, name, params, kind, weights)

    def test_output_preserves_existing_result_column(self):
        df = pd.DataFrame({'x': [1., 2.], 'A_ar': ['a', 'b']})
        result, _ = calculate(df, ['x'], '', 'A_ar', {})
        self.assertEqual(list(result.A_ar), ['a', 'b'])
        np.testing.assert_array_equal(result.A_ar_result, df.x)


if __name__ == '__main__':
    unittest.main()
