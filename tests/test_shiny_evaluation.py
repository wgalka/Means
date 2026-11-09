import unittest

import numpy as np

from apps.shiny.evaluation import evaluate, text_decisions


class TestEvaluation(unittest.TestCase):
    def test_manual_threshold_updates_metrics_and_roc_point(self):
        result = evaluate([0, 0, 1, 1], [.1, .4, .35, .8], threshold=.35)
        self.assertEqual(result['optimal_threshold'], .8)
        self.assertEqual(result['threshold'], .35)
        self.assertEqual(result['metrics']['Youden J'], .5)
        self.assertEqual(result['point'], (.5, 1.))
        np.testing.assert_array_equal(result['matrix'], [[1, 1], [0, 2]])
        all_positive = evaluate([0, 0, 1, 1], [.1, .4, .35, .8], threshold=0)
        self.assertEqual(all_positive['metrics']['Youden J'], 0.)
        self.assertEqual(all_positive['metrics']['ACC'], .5)
        lower = evaluate([0, 0, 1, 1], [.9, .8, .1, .2], direction='lower', threshold=.15)
        np.testing.assert_array_equal(lower['predicted'], [0, 0, 1, 0])
        with self.assertRaises(ValueError):
            evaluate([0, 1], [.1, .9], threshold=float('inf'))

    def test_known_roc_auc_and_youden_threshold(self):
        result = evaluate(['0', '0', '1', '1'], [.1, .4, .35, .8])
        self.assertEqual(result['metrics']['ROC AUC'], .75)
        # Equal Youden maxima at .8 and .35; prefer fewer positives.
        self.assertEqual(result['threshold'], .8)
        np.testing.assert_array_equal(result['matrix'], [[2, 0], [1, 1]])
        self.assertEqual(result['metrics']['ACC'], .75)
        self.assertEqual(result['metrics']['Balanced ACC'], .75)
        self.assertAlmostEqual(result['metrics']['F1 (+)'], 2/3)
        self.assertEqual(result['metrics']['F1 micro'], .75)
        self.assertAlmostEqual(result['metrics']['F1 macro'], (2/3 + .8)/2)

    def test_f1_criterion_and_lower_score_direction(self):
        result = evaluate([0, 0, 1, 1], [.1, .4, .35, .8], criterion='f1')
        self.assertEqual(result['threshold'], .35)
        self.assertEqual(result['metrics']['F1 (+)'], .8)
        lower = evaluate(['no', 'yes', 'no', 'yes'], [.9, .1, .8, .2], positive='yes', direction='lower')
        self.assertEqual(lower['threshold'], .2)
        self.assertEqual(lower['metrics']['ROC AUC'], 1.)
        np.testing.assert_array_equal(lower['matrix'], [[2, 0], [0, 2]])

    def test_constant_scores_and_all_negative_threshold(self):
        result = evaluate([0, 0, 0, 1], [.5]*4, criterion='accuracy')
        self.assertTrue(np.isposinf(result['threshold']))
        self.assertEqual(result['metrics']['ROC AUC'], .5)
        self.assertEqual(result['metrics']['ACC'], .75)
        self.assertEqual(result['metrics']['Precision (+)'], 0.)
        np.testing.assert_allclose(result['normalized'].sum(axis=1), [1, 1])

    def test_invalid_decisions(self):
        for labels, scores in [([0], [.2, .3]), ([0, 0], [.2, .3]),
                               ([0, 1, 2], [.1, .2, .3]), (['', '1'], [.2, .3]),
                               ([0, 1], [float('nan'), .3])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate(labels, scores)
        self.assertEqual(text_decisions('0,1;0\n1'), ['0', '1', '0', '1'])


if __name__ == '__main__':
    unittest.main()
