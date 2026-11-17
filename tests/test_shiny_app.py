"""Verify exported example code against the app, including CSV round trips."""
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from apps.shiny.core import (
    AGGREGATIONS,
    calculate,
    calculate_many,
    calculate_many_per_class,
    detect_csv_has_header,
    estimate_workload,
    grid_view,
    header_weight_vector,
    validate_csv_upload,
)


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

    def test_calculate_reports_chunked_progress_and_matches_result(self):
        df = pd.DataFrame({'x': np.linspace(0.1, 0.9, 11), 'y': np.linspace(0.9, 0.1, 11)})
        updates = []
        chunked, _ = calculate(df, ['x', 'y'], '', 'A_pw', {'r': 2},
                               on_progress=lambda done, total: updates.append((done, total)),
                               chunk_size=4)
        expected, _ = calculate(df, ['x', 'y'], '', 'A_pw', {'r': 2})
        pd.testing.assert_frame_equal(chunked, expected)
        self.assertEqual(updates, [(0, 11), (4, 11), (8, 11), (11, 11)])

    def test_validate_csv_upload_enforces_size_and_row_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / 'sample.csv'
            csv_path.write_text('x,y\n0.1,0.2\n0.3,0.4\n0.5,0.6\n', encoding='utf-8')
            validate_csv_upload(csv_path, max_bytes=1024, max_rows=3)
            with self.assertRaisesRegex(ValueError, 'MB limit'):
                validate_csv_upload(csv_path, max_bytes=10, max_rows=10)
            with self.assertRaisesRegex(ValueError, 'too many rows'):
                validate_csv_upload(csv_path, max_bytes=1024, max_rows=2)

    def test_estimate_workload_and_calculate_reject_oversized_jobs(self):
        df = pd.DataFrame({'x': np.linspace(0.1, 0.9, 50), 'y': np.linspace(0.9, 0.1, 50)})
        info = estimate_workload(df, ['x', 'y'], '', 'A_ar', 'uniform')
        self.assertEqual(info['rows'], 50)
        self.assertEqual(info['cols'], 2)

        with self.assertRaisesRegex(ValueError, 'Too many rows'):
            estimate_workload(df, ['x', 'y'], '', 'A_ar', 'uniform', max_rows=40)
        with self.assertRaisesRegex(ValueError, 'Estimated memory usage'):
            estimate_workload(df, ['x', 'y'], '', 'A_ar', 'uniform', max_ram_bytes=100)
        with self.assertRaisesRegex(ValueError, 'too computationally expensive'):
            estimate_workload(df, ['x', 'y'], '', 'A_ex2', 'uniform', max_complexity=10_000)

        large_df = pd.DataFrame({'x': np.linspace(0.2, 0.8, 6000), 'y': np.linspace(0.8, 0.2, 6000)})
        with self.assertRaisesRegex(ValueError, 'too computationally expensive'):
            calculate(large_df, ['x', 'y'], '', 'A_ex2', {'p': 2, 'q': 3})

    def test_grid_view_caps_oversized_full_view(self):
        df = pd.DataFrame({'x': range(20)})
        capped = grid_view(df, full=True, max_full_rows=10)
        self.assertEqual(len(capped), 11)
        self.assertEqual(list(capped.x), [0, 1, 2, 3, 4, '…', 15, 16, 17, 18, 19])

    def test_detect_csv_has_header(self):
        with tempfile.TemporaryDirectory() as directory:
            with_header = Path(directory) / 'with_header.csv'
            with_header.write_text('x1,x2,label\n0.1,0.2,low\n0.3,0.4,high\n', encoding='utf-8')
            self.assertTrue(detect_csv_has_header(with_header))

            no_header = Path(directory) / 'no_header.csv'
            no_header.write_text('0.1,0.2,low\n0.3,0.4,high\n', encoding='utf-8')
            self.assertFalse(detect_csv_has_header(no_header))

            semi_header = Path(directory) / 'semi_header.csv'
            semi_header.write_text('a;b\n0,1;0,2\n0,3;0,4\n', encoding='utf-8')
            self.assertTrue(detect_csv_has_header(semi_header, sep=';', decimal=','))

            semi_no_header = Path(directory) / 'semi_no_header.csv'
            semi_no_header.write_text('0,1;0,2\n0,3;0,4\n', encoding='utf-8')
            self.assertFalse(detect_csv_has_header(semi_no_header, sep=';', decimal=','))

    def test_calculate_many_runs_multiple_configs_and_generates_code(self):
        df = pd.DataFrame({'x': [0.2, 0.4, 0.7], 'y': [0.8, 0.6, 0.9]})
        methods = [
            {'name': 'A_sg', 'params': {}, 'measure': 'uniform'},
            {'name': 'A_sg', 'params': {}, 'measure': 'relative'},
            {'name': 'A_pw', 'params': {'r': 2}, 'measure': 'uniform'},
        ]
        updates = []
        result, code, cols = calculate_many(
            df, ['x', 'y'], methods,
            on_progress=lambda done, total: updates.append((done, total)),
            chunk_size=2,
        )
        self.assertEqual(cols, ['A_sg(measure=uniform)', 'A_sg(measure=relative)', 'A_pw(r=2)'])
        self.assertEqual(list(result.columns), ['x', 'y'] + cols)
        self.assertEqual(updates[-1], (9, 9))
        captured = []
        with patch.object(pd.DataFrame, 'to_csv', lambda self, *a, **k: captured.append(self.copy())):
            exec(code, {})
        pd.testing.assert_frame_equal(captured[0], result)

    def test_calculate_many_per_class_fusion(self):
        df0 = pd.DataFrame({'m1': [0.8, 0.2, 0.1], 'm2': [0.7, 0.1, 0.2]})
        df1 = pd.DataFrame({'m1': [0.1, 0.7, 0.2], 'm2': [0.2, 0.8, 0.1]})
        df2 = pd.DataFrame({'m1': [0.1, 0.1, 0.7], 'm2': [0.1, 0.1, 0.7]})
        methods = [{'name': 'A_ar', 'params': {}, 'measure': 'uniform'}]
        table, code, cols, matrices = calculate_many_per_class([df0, df1, df2], ['A', 'B', 'C'], methods)
        self.assertEqual(cols, ['A_ar'])
        self.assertEqual(list(table['A_ar [pred]']), ['A', 'B', 'C'])
        self.assertEqual(matrices['A_ar'].shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
