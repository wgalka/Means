import json
import unittest
from decimal import Decimal
from pathlib import Path

from aggregationslib.decimalaggregations import (
    A_ch as DecimalA_ch,
    A_sg as DecimalA_sg,
    DecimalFuzzyMeasure,
    choquet_integral as decimal_choquet_integral,
    k_fuzzy_measure as decimal_k_fuzzy_measure,
    owa_measure as decimal_owa_measure,
    relative_measure as decimal_relative_measure,
    sugeno_integral as decimal_sugeno_integral,
    uniform_measure as decimal_uniform_measure,
    weighted_measure as decimal_weighted_measure,
)


REFERENCE_PATH = Path(__file__).parent / "data" / "wolfram_integral_reference.json"
D = Decimal


def capacity_from_record(record):
    n = record["n"]
    return {
        frozenset(index for index in range(n) if mask & (1 << index)):
        D(str(value))
        for mask, value in enumerate(record["capacityByMask"])
    }


class TestDecimalBoundaryCases(unittest.TestCase):
    def setUp(self):
        self.measure = DecimalFuzzyMeasure(3, {
            frozenset(): D("0"),
            frozenset({0}): D("0.2"),
            frozenset({1}): D("0.3"),
            frozenset({2}): D("0.4"),
            frozenset({0, 1}): D("0.6"),
            frozenset({0, 2}): D("0.7"),
            frozenset({1, 2}): D("0.8"),
            frozenset({0, 1, 2}): D("1"),
        })

    def test_boundary_constant_singleton_and_indicator_properties(self):
        aggregations = (
            lambda values: decimal_choquet_integral(values, self.measure),
            lambda values: decimal_sugeno_integral(values, self.measure),
            DecimalA_ch(self.measure),
            DecimalA_sg(self.measure),
        )
        for aggregate in aggregations:
            for value in (D("0"), D("0.1"), D("0.5"), D("1")):
                self.assertEqual(aggregate([value, value, value]), value)

        self.assertEqual(DecimalA_ch()([D("0.37")]), D("0.37"))
        self.assertEqual(DecimalA_sg()([D("0.37")]), D("0.37"))

        for mask in range(1 << self.measure.n):
            active = frozenset(
                index for index in range(self.measure.n) if mask & (1 << index)
            )
            indicator = [
                D("1") if index in active else D("0")
                for index in range(self.measure.n)
            ]
            expected = self.measure(active)
            self.assertEqual(decimal_choquet_integral(indicator, self.measure), expected)
            self.assertEqual(decimal_sugeno_integral(indicator, self.measure), expected)

    def test_internality_and_monotonicity(self):
        low = [D("0.1"), D("0.4"), D("0.6")]
        high = [D("0.2"), D("0.4"), D("0.9")]
        for aggregate in (
            DecimalA_ch(self.measure),
            DecimalA_sg(self.measure),
        ):
            low_result = aggregate(low)
            high_result = aggregate(high)
            self.assertGreaterEqual(low_result, min(low))
            self.assertLessEqual(low_result, max(low))
            self.assertGreaterEqual(high_result, min(high))
            self.assertLessEqual(high_result, max(high))
            self.assertLessEqual(low_result, high_result)


class TestDecimalAggregationProperties(unittest.TestCase):
    def test_rejects_float_inputs_to_preserve_decimal_precision(self):
        with self.assertRaisesRegex(TypeError, "must not be float"):
            decimal_choquet_integral([0.1, D("0.5")])

        with self.assertRaisesRegex(TypeError, "must not be float"):
            DecimalFuzzyMeasure(1, {
                frozenset(): D("0"),
                frozenset({0}): 1.0,
            })

    def test_allow_float_converts_inputs_and_measures_to_decimal(self):
        expected_mean = D("0.4")
        self.assertEqual(
            decimal_choquet_integral([0.1, 0.7], allow_float=True), expected_mean
        )
        self.assertEqual(DecimalA_ch(allow_float=True)([0.1, 0.7]), expected_mean)

        symmetric_measure = [0.0, 0.25, 1.0]
        self.assertEqual(
            decimal_sugeno_integral(
                [0.2, 0.8], measure=symmetric_measure, allow_float=True
            ),
            D("0.25"),
        )
        self.assertEqual(
            DecimalA_sg(measure=symmetric_measure, allow_float=True)([0.2, 0.8]),
            D("0.25"),
        )

        capacity = DecimalFuzzyMeasure(1, {
            frozenset(): 0.0,
            frozenset({0}): 1.0,
        }, allow_float=True)
        self.assertEqual(capacity(frozenset({0})), D("1.0"))

    def test_allow_float_must_be_bool(self):
        with self.assertRaisesRegex(TypeError, "allow_float must be a bool"):
            decimal_choquet_integral([D("0.5")], allow_float=1)

        with self.assertRaisesRegex(TypeError, "allow_float must be a bool"):
            DecimalA_sg(allow_float="yes")

        with self.assertRaisesRegex(TypeError, "allow_float must be a bool"):
            DecimalFuzzyMeasure(1, {
                frozenset(): D("0"),
                frozenset({0}): D("1"),
            }, allow_float=None)

    def test_cardinality_and_additive_measure_results(self):
        values = [D("0.2"), D("0.7"), D("0.9")]
        expected_mean = D("0.6")
        expected_sugeno = D("2") / D("3")
        self.assertEqual(decimal_choquet_integral(values), expected_mean)
        self.assertEqual(DecimalA_ch()(values), expected_mean)
        self.assertEqual(decimal_sugeno_integral(values), expected_sugeno)
        self.assertEqual(DecimalA_sg()(values), expected_sugeno)

        weights = [D("0.2"), D("0.3"), D("0.5")]
        capacity = {
            frozenset(index for index in range(3) if mask & (1 << index)):
            sum((weights[index] for index in range(3) if mask & (1 << index)), D("0"))
            for mask in range(1 << 3)
        }
        additive_measure = DecimalFuzzyMeasure(3, capacity)
        weighted_values = [D("0.25"), D("0.75"), D("0.4")]
        weighted_mean = sum(
            (weight * value for weight, value in zip(weights, weighted_values)), D("0")
        )
        self.assertEqual(
            decimal_choquet_integral(weighted_values, additive_measure), weighted_mean
        )
        self.assertEqual(DecimalA_ch(additive_measure)(weighted_values), weighted_mean)

    def test_paper_measure_constructors_work_with_both_integrals(self):
        measures = (
            decimal_uniform_measure(3),
            decimal_relative_measure(3),
            decimal_k_fuzzy_measure([D("0.5"), D("0.25")]),
            decimal_weighted_measure([D("0.2"), D("0.3"), D("0.5")]),
            decimal_owa_measure([D("0.5"), D("0.3"), D("0.2")]),
        )

        for measure in measures:
            for mask in range(1 << measure.n):
                active = frozenset(
                    index for index in range(measure.n) if mask & (1 << index)
                )
                indicator = [
                    D("1") if index in active else D("0")
                    for index in range(measure.n)
                ]
                with self.subTest(measure=measure, active=active):
                    self.assertEqual(
                        decimal_choquet_integral(indicator, measure),
                        measure(active),
                    )
                    self.assertEqual(
                        decimal_sugeno_integral(indicator, measure),
                        measure(active),
                    )

        self.assertEqual(
            decimal_uniform_measure(3)(frozenset({0, 2})), D(2) / D(3)
        )
        self.assertEqual(
            decimal_relative_measure(3)(frozenset({0, 2})), D(4) / D(6)
        )
        self.assertEqual(
            decimal_k_fuzzy_measure([D("0.5"), D("0.25")])(
                frozenset({0, 1})
            ),
            D("1"),
        )
        self.assertEqual(
            decimal_weighted_measure([D("0.2"), D("0.3"), D("0.5")])(
                frozenset({0, 2})
            ),
            D("0.7"),
        )

        owa = decimal_owa_measure([D("0.5"), D("0.3"), D("0.2")])
        expected_owa = D("0.5") * D("0.9") + D("0.3") * D("0.4") + D("0.2") * D("0.1")
        self.assertEqual(
            decimal_choquet_integral([D("0.1"), D("0.9"), D("0.4")], owa),
            expected_owa,
        )

    def test_k_fuzzy_measure_handles_negative_parameter_and_impossible_weights(self):
        measure = decimal_k_fuzzy_measure([D("0.8"), D("0.5")])
        self.assertEqual(measure(frozenset({0})), D("0.8"))
        self.assertEqual(measure(frozenset({1})), D("0.5"))
        self.assertEqual(measure(frozenset({0, 1})), D("1"))

        positive_parameter_measure = decimal_k_fuzzy_measure(
            [D("0.125"), D("0.25"), D("0.3")]
        )
        self.assertEqual(
            positive_parameter_measure(frozenset({0, 1})), D("0.4375")
        )
        negative_one_measure = decimal_k_fuzzy_measure(
            [D("1"), D("0.3"), D("0.4")]
        )
        self.assertEqual(negative_one_measure(frozenset({1, 2})), D("0.58"))

        with self.assertRaisesRegex(ValueError, "do not define"):
            decimal_k_fuzzy_measure([D("0.3"), D("0")])

    def test_weighted_and_owa_measures_require_normalized_weights(self):
        for constructor in (decimal_weighted_measure, decimal_owa_measure):
            with self.subTest(constructor=constructor.__name__):
                with self.assertRaisesRegex(ValueError, "sum to 1"):
                    constructor([D("0.2"), D("0.2")])

    def test_quasi_overlap_choquet_formula_and_callable_object(self):
        measure = decimal_uniform_measure(2)
        values = [D("0.2"), D("0.7")]
        expected = D("0.5")

        self.assertEqual(
            decimal_choquet_integral(values, measure, quasi_overlap=min), expected
        )
        self.assertEqual(
            DecimalA_ch(measure, quasi_overlap=min)(values), expected
        )
        self.assertEqual(decimal_choquet_integral(values, measure), D("0.45"))

    def test_quasi_overlap_argument_is_validated(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            decimal_choquet_integral([D("0.2")], quasi_overlap=1)
        with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
            decimal_choquet_integral(
                [D("0.2")], quasi_overlap=lambda x, mu: D("2")
            )


class TestDecimalWolframReferenceRecords(unittest.TestCase):
    def test_functions_and_objects_match_all_reference_cases(self):
        with REFERENCE_PATH.open(encoding="utf-8") as reference_file:
            reference = json.load(reference_file)

        self.assertEqual(reference["caseCount"], 100)
        self.assertEqual(len(reference["cases"]), 100)

        tolerance = D("1e-12")
        for record in reference["cases"]:
            with self.subTest(case=record["id"]):
                values = [D(str(value)) for value in record["values"]]
                measure = DecimalFuzzyMeasure(record["n"], capacity_from_record(record))
                expected_choquet = D(str(record["choquet"]))
                expected_sugeno = D(str(record["sugeno"]))

                actual_results = (
                    (decimal_choquet_integral(values, measure), expected_choquet),
                    (DecimalA_ch(measure)(values), expected_choquet),
                    (decimal_sugeno_integral(values, measure), expected_sugeno),
                    (DecimalA_sg(measure)(values), expected_sugeno),
                )
                for actual, expected in actual_results:
                    self.assertIsInstance(actual, Decimal)
                    self.assertLessEqual(abs(actual - expected), tolerance)


if __name__ == "__main__":
    unittest.main()
