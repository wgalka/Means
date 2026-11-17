from decimal import Decimal

import numpy as np
import pytest

from aggregationslib import aggregation, aggregations, decimalaggregations, ndarray
from aggregationslib import A_lm, A_lm1
from apps.shiny.core import configuration


@pytest.mark.parametrize('r,values,expected', [
    (1, [0, 1], '0.5'),
    (1, [0, 0], '0'),
    (1, [1, 2, 3], '2'),
    (0, [0, 1], '0'),
    (0.5, [0, 1], '0'),
    (2, [0, 1], '1'),
    (2, [0, 0], '0'),
])
def test_continuous_lehmer_zero_boundary_across_apis(r, values, expected):
    vector = np.array(values, dtype=float)
    assert aggregation.lehmer1(vector, r) == pytest.approx(float(expected))
    assert A_lm1(r)(vector) == pytest.approx(float(expected))
    assert ndarray.A_lm1(r, axis=1)(vector[None, :])[0] == pytest.approx(float(expected))
    decimal_values = [Decimal(str(value)) for value in values]
    assert decimalaggregations.lehmer1(decimal_values, r) == Decimal(expected)


def test_decimal_lehmer_arithmetic_accepts_plain_numbers():
    assert decimalaggregations.lehmer1([0, 1], 1) == Decimal('0.5')


def test_default_lehmer_uses_zero_div_zero_convention():
    assert aggregation.lehmer([0, 1], 1) == 0
    assert decimalaggregations.lehmer([0, 1], 1) == 0
    assert aggregations.A_lm(1)([0, 1]) == 0
    assert ndarray.A_lm(1, axis=1)([[0, 1]])[0] == 0
    operator, _, _ = configuration('A_lm', {'r': 1}, 'uniform', '', 2)
    np.testing.assert_array_equal(operator([[0, 1]]), [0])


def test_integer_inputs_for_negative_exponents_and_quadratic_overflow():
    assert ndarray.A_pw(-1)([1, 2]) == pytest.approx(4 / 3)
    assert aggregations.A_pw(-1)([1, 2]) == pytest.approx(4 / 3)
    assert aggregations.A_lm(0)([1, 2]) == pytest.approx(4 / 3)
    assert aggregations.A_lm1(0)([1, 2]) == pytest.approx(4 / 3)
    assert ndarray.A_qd()([2**32, 2**32]) == pytest.approx(float(2**32))
    assert aggregations.A_gm()([1, 4]) == pytest.approx(2.0)
    assert aggregations.A_ex2(2, 2)([0, -1]) == pytest.approx(aggregations.A_ex2(2, 2)([0.0, -1.0]))
    assert aggregations.A_ex3(2, 2)([0, -1]) == pytest.approx(aggregations.A_ex3(2, 2)([0.0, -1.0]))


@pytest.mark.parametrize('r,values,expected', [
    (0.5, [0, 1], (0, 0)),
    (1, [0, 1], (0, 0.5)),
    (1, [1, 3], (2, 2)),
    (2, [0, 1], (1, 1)),
    (2, [0, 0], (0, 0)),
])
def test_named_lehmer_variants(r, values, expected):
    decimals = [Decimal(value) for value in values]
    variants = zip(
        (A_lm, A_lm1),
        (aggregation.lehmer, aggregation.lehmer1),
        (decimalaggregations.lehmer, decimalaggregations.lehmer1),
        expected,
    )
    for cls, functional, exact, result in variants:
        assert cls(r)(values) == result
        assert functional(values, r) == result
        assert exact(decimals, r) == Decimal(str(result))
        operator, _, _ = configuration(cls.__name__, {'r': r}, 'uniform', '', len(values))
        np.testing.assert_array_equal(operator([values]), [result])
