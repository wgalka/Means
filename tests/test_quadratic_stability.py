import json
from fractions import Fraction
from pathlib import Path
import re

import numpy as np
import pytest

from aggregationslib import aggregation, aggregations, ndarray


REFERENCE = json.loads((Path(__file__).parent / 'data' /
                        'wolfram_quadratic_reference.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda case: case['id'])
def test_quadratic_matches_wolfram(case):
    assert case['equal'] is True
    # Wolfram InputForm annotates precision with a backtick and uses *^ for exponents.
    expected = float(re.sub(r'`[0-9.]+', '', case['expected']).replace('*^', 'e'))
    values = [float(Fraction(value)) for value in case['values']]
    for operator in (aggregation.quadratic, aggregations.A_qd(), ndarray.A_qd()):
        assert operator(values) == pytest.approx(expected, rel=1e-14, abs=0)
    result = ndarray.A_qd(axis=1)([values, values])
    np.testing.assert_allclose(result, [expected, expected], rtol=1e-14, atol=0)


def test_integer_overflow_regression():
    values = np.array([2**32, 2**32], dtype=np.int64)
    for operator in (aggregation.quadratic, aggregations.A_qd(), ndarray.A_qd()):
        assert operator(values) == 2**32


@pytest.mark.parametrize('values,expected', [([0, 0], 0), ([np.inf, 1], np.inf),
                                            ([np.nan, 1], np.nan), ([], np.nan)])
def test_quadratic_special_values(values, expected):
    for operator in (aggregation.quadratic, aggregations.A_qd()):
        np.testing.assert_equal(operator(values), expected)
