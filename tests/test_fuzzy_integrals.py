import json
import unittest
from pathlib import Path

from aggregationslib import (
    A_ch,
    A_sg,
)
from aggregationslib.aggregation import (
    FuzzyMeasure,
    choquet_integral,
    sugeno_integral,
)


REFERENCE_PATH = Path(__file__).parent / "data" / "wolfram_integral_reference.json"


def capacity_from_record(record):
    n = record["n"]
    by_mask = record["capacityByMask"]
    return {
        frozenset(index for index in range(n) if mask & (1 << index)): value
        for mask, value in enumerate(by_mask)
    }


def example_measure():
    return FuzzyMeasure(3, {
        frozenset(): 0.0,
        frozenset({0}): 0.2,
        frozenset({1}): 0.3,
        frozenset({2}): 0.4,
        frozenset({0, 1}): 0.6,
        frozenset({0, 2}): 0.7,
        frozenset({1, 2}): 0.8,
        frozenset({0, 1, 2}): 1.0,
    })


class TestBoundaryCases(unittest.TestCase):
    """Analytical boundary properties; kept first in unittest discovery."""

    def setUp(self):
        self.measure = example_measure()

    def test_zero_one_constant_and_singleton_vectors(self):
        for integral in (choquet_integral, sugeno_integral):
            for value in (0.0, 0.1, 0.5, 1.0):
                with self.subTest(integral=integral.__name__, constant=value):
                    self.assertAlmostEqual(
                        integral([value, value, value], self.measure), value
                    )

            self.assertEqual(integral([0.0, 0.0, 0.0], self.measure), 0.0)
            self.assertEqual(integral([1.0, 1.0, 1.0], self.measure), 1.0)
            self.assertEqual(integral([0.37]), 0.37)

    def test_indicator_vectors_return_measure_of_active_subset(self):
        for mask in range(1 << self.measure.n):
            active = frozenset(
                index for index in range(self.measure.n) if mask & (1 << index)
            )
            indicator = [float(index in active) for index in range(self.measure.n)]
            expected = self.measure(active)

            self.assertAlmostEqual(choquet_integral(indicator, self.measure), expected)
            self.assertAlmostEqual(sugeno_integral(indicator, self.measure), expected)

    def test_tied_values_with_asymmetric_measure(self):
        measure = FuzzyMeasure(2, {
            frozenset(): 0.0,
            frozenset({0}): 0.9,
            frozenset({1}): 0.1,
            frozenset({0, 1}): 1.0,
        })
        for integral in (choquet_integral, sugeno_integral):
            self.assertAlmostEqual(integral([0.5, 0.5], measure), 0.5)


class TestAggregationProperties(unittest.TestCase):
    def setUp(self):
        self.measure = example_measure()

    def test_default_cardinality_measure_results(self):
        values = [0.2, 0.7, 0.9]
        self.assertAlmostEqual(choquet_integral(values), sum(values) / len(values))
        self.assertAlmostEqual(A_ch()(values), sum(values) / len(values))
        self.assertAlmostEqual(sugeno_integral(values), 2 / 3)
        self.assertAlmostEqual(A_sg()(values), 2 / 3)

    def test_internality_and_monotonicity(self):
        low = [0.1, 0.4, 0.6]
        high = [0.2, 0.4, 0.9]
        aggregations = (
            (lambda values: choquet_integral(values, self.measure)),
            (lambda values: sugeno_integral(values, self.measure)),
            A_ch(self.measure),
            A_sg(self.measure),
        )
        for aggregate in aggregations:
            low_result = aggregate(low)
            high_result = aggregate(high)
            self.assertGreaterEqual(low_result, min(low))
            self.assertLessEqual(low_result, max(low))
            self.assertLessEqual(low_result, high_result)

    def test_relabeling_inputs_and_capacity_preserves_results(self):
        values = [0.15, 0.55, 0.85]
        permutation = [2, 0, 1]
        permuted_values = [values[index] for index in permutation]
        permuted_capacity = {}

        for mask in range(1 << self.measure.n):
            subset = frozenset(
                index for index in range(self.measure.n) if mask & (1 << index)
            )
            relabeled_subset = frozenset(
                new_index
                for new_index, old_index in enumerate(permutation)
                if old_index in subset
            )
            permuted_capacity[relabeled_subset] = self.measure(subset)

        relabeled_measure = FuzzyMeasure(self.measure.n, permuted_capacity)
        for integral in (choquet_integral, sugeno_integral):
            self.assertAlmostEqual(
                integral(values, self.measure),
                integral(permuted_values, relabeled_measure),
            )

    def test_choquet_reduces_to_weighted_mean_for_additive_measure(self):
        weights = [0.2, 0.3, 0.5]
        capacity = {
            frozenset(index for index in range(3) if mask & (1 << index)):
            sum(weights[index] for index in range(3) if mask & (1 << index))
            for mask in range(1 << 3)
        }
        measure = FuzzyMeasure(3, capacity)
        values = [0.25, 0.75, 0.4]
        expected = sum(weight * value for weight, value in zip(weights, values))

        self.assertAlmostEqual(choquet_integral(values, measure), expected)
        self.assertAlmostEqual(A_ch(measure)(values), expected)


class TestWolframReferenceRecords(unittest.TestCase):
    def test_functions_and_objects_match_all_reference_cases(self):
        with REFERENCE_PATH.open(encoding="utf-8") as reference_file:
            reference = json.load(reference_file)

        self.assertEqual(reference["caseCount"], 100)
        self.assertEqual(len(reference["cases"]), 100)

        for record in reference["cases"]:
            with self.subTest(case=record["id"]):
                measure = FuzzyMeasure(record["n"], capacity_from_record(record))
                values = record["values"]

                self.assertAlmostEqual(
                    choquet_integral(values, measure), record["choquet"], places=12
                )
                self.assertAlmostEqual(
                    sugeno_integral(values, measure), record["sugeno"], places=12
                )
                self.assertAlmostEqual(
                    A_ch(measure)(values), record["choquet"], places=12
                )
                self.assertAlmostEqual(
                    A_sg(measure)(values), record["sugeno"], places=12
                )


if __name__ == "__main__":
    unittest.main()
