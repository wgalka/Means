from collections.abc import Iterable, Mapping
import math
import warnings
from decimal import Decimal
from math import isclose, isfinite
from numbers import Integral

import numpy as np
from pynverse.inverse import inversefunc


# 1 Arithmetic aggregation
def arithmetic(y):
    """
    Arithmetic aggregation
    :param y: 1 dimensional array
    :return: aggregation value
    """
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    return np.mean(y)


# 2
def quadratic(y):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    y = np.array(y)
    size = len(y)
    with np.errstate(divide='ignore'):
        return np.sqrt(np.sum(y ** 2) / size)


# 3
def geometric(y):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")

    size = len(y)
    wk = 1 / size
    with np.errstate(divide='ignore'):
        return np.prod(y ** wk)


# 4
def harmonic(y):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    if type(y).__module__ != np.__name__:
        y = np.array(y)
    if np.any(y == 0):
        return 0
    size = len(y)
    return size / (np.sum(1 / y))


# 5
def power(y, r=1):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    if r == 0:
        raise ValueError("parameter r cannot be equal to 0 ")
    # If one 0 occur in array then mean is 0
    if 0 in y and r < 0:
        return 0
    else:
        result = (np.sum(y ** r) / y.shape[0]) ** (1 / r)
        return result


# 6
def exponential(y, r=1):
    """
    Exponential mean (numerically stable implementation using Log-Sum-Exp trick).

    The exponential mean for a sequence y = (y_1, ..., y_n) and shape parameter r != 0 is defined as:
        A_ex(y) = 1/r * ln( 1/n * sum( e^(r * y_i) ) )

    To avoid numerical instability issues:
    - Overflow (when r * y_i > 709.78), making the exponential term +inf
    - Underflow (when all r * y_i < -709.78), making the sum of exponentials 0
    the Log-Sum-Exp (LSE) trick is applied by shifting the exponents by a_max = max(r * y_j):
        A_ex(y) = 1/r * [ a_max + ln( sum( e^(r * y_i - a_max) ) ) - ln(n) ]
    """
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    if r == 0:
        raise ValueError("parameter r should be != 0 ")
    size = len(y)
    y = np.array(y)
    r_y = y * r
    a_max = np.max(r_y)
    with np.errstate(divide='ignore'):
        sum_exp = np.sum(np.exp(r_y - a_max))
        return (1 / r) * (a_max + np.lib.scimath.log(sum_exp) - np.lib.scimath.log(size))


# 7
def lehmer(y, r=0):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    # if  array and r below 0 return 0. 0 to power of 0.
    if 0 in y and r - 1 <= 0:
        return 0
    else:
        r2 = (r - 1)
        nominator = np.sum(y ** r)
        with np.errstate(divide='ignore'):  #
            denominator = np.sum(y ** r2)
        if denominator == 0:
            return 0
        else:
            return nominator / denominator


# 8
def arithmetic_min(y, p=0):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    min = np.min(y)
    return p * np.mean(y) + (1 - p) * min


# 9
def arithmetic_max(y, p=0):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    max = np.max(y)
    return p * np.mean(y) + (1 - p) * max


# 10
def median(y):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    return np.median(y)


# 11
def olimpic(y):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    # b = np.max(a, 0)  # max
    # c = np.min(a, 0)  # min
    if len(y) < 3:
        return np.mean(y)
    else:
        d = np.sort(y)[1:-1]
        return np.mean(d)


def exponential2(y, p=0, q=0):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    y = np.array(y)
    mean = np.mean(p ** (q ** y))
    firstlog = math.log(mean, p)
    result = math.log(firstlog, q)
    return result


def exponential3(y, p=0, q=0):
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    y = np.array(y)
    result = (1 / q) * math.log(np.mean(p ** (q * y)), p)
    return result


def sin_aggregation(y):
    '''

    :param y:
    :return:
    '''
    # Check if 1 dimensional array
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    n = len(y)
    return math.asin(np.sum(np.sin(y)) / n)


def quasi_arithmeric(y, function):
    """
    Quasi arithmetic mean.

    If function f(x)=x to then quasi-arithmetic mean is aritmetic mean.
    If function f(x)=x^p (p != 0) then quasi-arithmetic mean is power mean to p power.
    If function f(x)=log x then quasi-arithmetic mean is geometric mean.

    :param y: 1 dimmensional array
    :param function: function which maps an interval I of the real line to the real numbers, and is both continuous and injective
    :return: aggregation value
    """
    # Check if 1 dimensional array
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    y = np.asarray(y)
    n = len(y)
    vectorized_function = np.vectorize(function)
    apply_func = vectorized_function(y)
    value = np.sum(apply_func) / n
    return inversefunc(vectorized_function, value)


# Fuzzy-integral aggregation functions
_TOLERANCE = 1e-12


def _subset_key(subset: object, n: int) -> frozenset[int]:
    if isinstance(subset, (str, bytes)) or not isinstance(subset, Iterable):
        raise TypeError("subset keys must be iterables of zero-based integer indices")

    result = frozenset(subset)
    if any(
        isinstance(index, bool)
        or not isinstance(index, Integral)
        or index < 0
        or index >= n
        for index in result
    ):
        raise ValueError(f"subset indices must be integers between 0 and {n - 1}")
    return frozenset(int(index) for index in result)


class FuzzyMeasure:
    """A normalized monotone capacity on a finite set of vector positions.

    ``values`` must contain one entry for every subset. Keys are iterables of
    zero-based positions, for example ``frozenset({0, 2})``.
    """

    def __init__(self, n: int, values: Mapping[object, float]):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(values, Mapping):
            raise TypeError("values must map every subset to its measure")

        n = int(n)
        normalized: dict[frozenset[int], float] = {}
        for subset, raw_value in values.items():
            key = _subset_key(subset, n)
            if key in normalized:
                raise ValueError(f"duplicate subset measure for {sorted(key)}")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TypeError("measure values must be real numbers") from exc
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("measure values must be finite and in [0, 1]")
            normalized[key] = value

        if len(normalized) != 1 << n:
            raise ValueError(f"values must define all {1 << n} subsets")

        empty_set = frozenset()
        full_set = frozenset(range(n))
        if not isclose(normalized[empty_set], 0.0, abs_tol=_TOLERANCE, rel_tol=0.0):
            raise ValueError("a fuzzy measure must assign 0 to the empty set")
        if not isclose(normalized[full_set], 1.0, abs_tol=_TOLERANCE, rel_tol=0.0):
            raise ValueError("a fuzzy measure must assign 1 to the full set")
        normalized[empty_set] = 0.0
        normalized[full_set] = 1.0

        for subset, value in normalized.items():
            for index in full_set - subset:
                superset = subset | {index}
                if value > normalized[superset] + _TOLERANCE:
                    raise ValueError(
                        "measure must be monotone: adding an element cannot "
                        "decrease its value"
                    )

        self._n = n
        self._values = normalized

    @property
    def n(self) -> int:
        """Number of elements in the measured set."""
        return self._n

    def __call__(self, subset: Iterable[int]) -> float:
        key = _subset_key(subset, self._n)
        return self._values[key]

    def __repr__(self) -> str:
        return f"FuzzyMeasure(n={self._n})"


def _as_unit_interval_vector(values: object) -> np.ndarray:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("values must be real numbers")
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be a one-dimensional numeric vector") from exc

    if vector.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if vector.size == 0:
        raise ValueError("values must contain at least one element")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("values must be in [0, 1]")
    return vector


def _resolve_measure(measure: object, n: int):
    if measure is None:
        return lambda subset: len(subset) / n

    if isinstance(measure, FuzzyMeasure):
        if measure.n != n:
            raise ValueError(f"measure is defined for {measure.n} elements, not {n}")
        return measure

    if isinstance(measure, Mapping):
        return FuzzyMeasure(n, measure)

    try:
        cardinality_values = np.array(measure, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "measure must be a FuzzyMeasure, a full subset mapping, or "
            "a symmetric measure vector"
        ) from exc

    if cardinality_values.ndim != 1 or cardinality_values.size != n + 1:
        raise ValueError(
            "a symmetric measure vector must contain n + 1 values: "
            "[mu(empty), ..., mu(full)]"
        )
    if not np.all(np.isfinite(cardinality_values)):
        raise ValueError("measure values must be finite")
    if np.any((cardinality_values < 0.0) | (cardinality_values > 1.0)):
        raise ValueError("measure values must be in [0, 1]")
    if not isclose(cardinality_values[0], 0.0, abs_tol=_TOLERANCE, rel_tol=0.0):
        raise ValueError("a fuzzy measure must assign 0 to the empty set")
    if not isclose(cardinality_values[-1], 1.0, abs_tol=_TOLERANCE, rel_tol=0.0):
        raise ValueError("a fuzzy measure must assign 1 to the full set")
    if np.any(np.diff(cardinality_values) < -_TOLERANCE):
        raise ValueError("a symmetric fuzzy measure must be nondecreasing by set size")

    cardinality_values[0] = 0.0
    cardinality_values[-1] = 1.0
    return lambda subset: float(cardinality_values[len(subset)])


def choquet_integral(values: object, measure: object = None) -> float:
    """Return the scalar discrete Choquet integral of a vector in ``[0, 1]``.

    Values are sorted in ascending order. ``measure`` can be:

    * ``None`` for the cardinality measure ``mu(A) = |A| / n``;
    * a symmetric measure vector ``[mu(empty), ..., mu(full)]`` of length
      ``n + 1``;
    * a :class:`FuzzyMeasure` or a mapping that defines every subset.

    Subset keys in mappings use zero-based vector positions. The result is a
    Python ``float``. The Choquet integral is defined here on ``[0, 1]^n``.
    """
    vector = _as_unit_interval_vector(values)
    capacity = _resolve_measure(measure, vector.size)
    order = np.argsort(vector, kind="stable")
    ordered_values = vector[order]

    result = 0.0
    previous_value = 0.0
    for rank, value in enumerate(ordered_values):
        upper_level_set = frozenset(int(index) for index in order[rank:])
        result += (float(value) - previous_value) * capacity(upper_level_set)
        previous_value = float(value)
    return float(result)


def sugeno_integral(values: object, measure: object = None) -> float:
    """Return the scalar discrete Sugeno integral of a vector in ``[0, 1]``.

    The accepted measure formats are the same as for :func:`choquet_integral`.
    The implementation computes ``max_i min(x_(i), mu(A_i))`` using ascending
    order statistics and their upper-level sets ``A_i``.
    """
    vector = _as_unit_interval_vector(values)
    capacity = _resolve_measure(measure, vector.size)
    order = np.argsort(vector, kind="stable")
    ordered_values = vector[order]

    result = 0.0
    for rank, value in enumerate(ordered_values):
        upper_level_set = frozenset(int(index) for index in order[rank:])
        result = max(result, min(float(value), capacity(upper_level_set)))
    return float(result)


if __name__ == "__main__":
    print(quasi_arithmeric.__str__())
    print(arithmetic_min([0.2, 0.4, 0.6], 0))
