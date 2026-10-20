from collections.abc import Iterable, Mapping
import logging
import numbers
import warnings

from decimal import Decimal, InvalidOperation
from numbers import Integral

# 1 Arithmetic aggregation


def arithmetic(y):
    """
    Arithmetic aggregation
    :param y: 1 dimensional array
    :return: aggregation value
    """
    _sum = Decimal('0')
    n_elems = len(y)
    for num in y:
        try:
            _sum += num
        except:
            _sum += Decimal(str(num))
            warnings.warn("Numbers in iterable are converted to Decimal automatically.")
    return _sum / n_elems


# 2
def quadratic(y):
    n = len(y)
    _sum = Decimal('0')
    for num in y:
        try:
            _sum += num ** 2
        except:
            _sum += Decimal(str(num)) ** 2
            warnings.warn("Numbers in iterable are converted to Decimal automatically.")
    return Decimal.sqrt(_sum / n)


# 3
def geometric(y):
    """
    y should contain only positive numbers [0, inf]
    :param y:
    :return:
    """
    _prod = Decimal('1')
    n = Decimal(len(y))

    for num in y:
        try:
            _prod *= num ** (1 / n)
        except:
            _prod *= Decimal(str(num)) ** (1 / n)
            warnings.warn("Numbers in iterable are converted to Decimal automatically.")
    return _prod


# 4
def harmonic(y):
    """
    WARNING if sum of denominator is  0 eg. [-10, 0 ,10] then function return 0.
    WARNING if y cointains 0 then return 0.
    :param y:
    :return:
    """
    n = Decimal(len(y))

    y_dec = []
    for x in y:
        if isinstance(x, numbers.Number):
            val = Decimal(str(x))
            y_dec.append(val)
        else:
            raise ValueError("y should contain numbers in 1D iterable object.")

    _sum = Decimal('0')
    for num in y_dec:
        if num == 0:
            warnings.warn("y contains 0 for hearmonic(" + str(y) + ")")
            return Decimal('0')
        _sum += Decimal('1') / num


    if _sum == 0:
        warnings.warn("sum of denominator is equal to 0 for hearmonic(" + str(y) + ")")
        return Decimal('0')
    return n / _sum


# 5
def power(y, r=1):
    if r == 0:
        raise ValueError("parameter r cannot be equal to 0 ")
    # If one 0 occur in array then mean is 0
    if 0 in y and r < 0:
        return 0
    else:
        n = Decimal(len(y))

        _sum = Decimal('0')
        for num in y:
            try:
                _sum += num ** r
            except:
                _sum += Decimal(str(num)) ** r
                warnings.warn("Numbers in iterable are converted to Decimal automatically.")

        return (_sum / n) ** (1 / Decimal(r))


# 6
def exponential(y, r=1):
    r"""
    Mean is given by equation:
    $A_6^{(r)}(x_1,...,x_n)= \frac{1}{r}\ln
    \Big(\frac{1}{n} \sum \limits_{k=1}^{n} e^{rx_k}\Big)$, where
    $r \in \mathbb{R}$, $r \neq 0$

    :param y: array of values
    :param r: r != 0
    :return: value of exponential mean
    """
    size = Decimal(str(len(y)))
    r = Decimal(r)
    if r == 0:
        raise ValueError("parameter r should be != 0 ")

    # Convert y to Decimal
    y_dec = []
    for x in y:
        if isinstance(x, numbers.Number):
            val = Decimal(str(x))
            y_dec.append(val)
        else:
            raise ValueError("y should contain numbers in 1D iterable object.")

    _sum = Decimal('0')
    for idx, num in enumerate(y_dec):
        _sum += (num * r).exp()

    return (1 / r) * (_sum / size).ln()


# 7
def lehmer(y, r=0):
    r = Decimal(r)
    if 0 in y and r - 1 <= 0:
        return 0
    else:
        r2 = (r - 1)
        _nominator = Decimal('0')
        _denominator = Decimal('0')
        for num in y:
            try:
                _nominator += num ** r
                _denominator += num ** r2
            except:
                _nominator += Decimal(str(num)) ** r
                _denominator += Decimal(str(num)) ** r2
                warnings.warn("Numbers in iterable are converted to Decimal automatically.")

        if _denominator == 0:
            return 0
        else:
            return _nominator / _denominator


# 8
def arithmetic_min(y, p=0):
    p = Decimal(p)
    _min = Decimal(min(y))
    return p * arithmetic(y) + (1 - p) * _min


# 9
def arithmetic_max(y, p=0):
    p = Decimal(p)
    _max = Decimal(max(y))
    return p * arithmetic(y) + (1 - p) * _max


# 10
def median(y):
    n = len(y)
    if n % 2 == 0:
        return arithmetic([y[(n / 2) - 1], y[n / 2]])
    else:
        return y[n // 2]


# 11
def olimpic(y):
    if len(y) < 3:
        return arithmetic(y)
    else:
        d = sorted(y)
        d = d[1:-1]
        return arithmetic(d)


_UNIT_ZERO = Decimal("0")
_UNIT_ONE = Decimal("1")


def _validate_allow_float(allow_float):
    if not isinstance(allow_float, bool):
        raise TypeError("allow_float must be a bool")
    return allow_float


def _decimal_unit_value(value, name="value", allow_float=False):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if isinstance(value, float) and not allow_float:
        raise TypeError(f"{name} must not be float; use Decimal for exact decimal input")
    if not isinstance(value, (Decimal, Integral, str, float)):
        raise TypeError(f"{name} must be a Decimal, integer, or decimal string")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    if not _UNIT_ZERO <= decimal_value <= _UNIT_ONE:
        raise ValueError(f"{name} must be in [0, 1]")
    return decimal_value


def _decimal_vector(values, allow_float=False):
    _validate_allow_float(allow_float)
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError("values must be a one-dimensional numeric iterable")
    vector = [
        _decimal_unit_value(value, "values", allow_float=allow_float)
        for value in values
    ]
    if not vector:
        raise ValueError("values must contain at least one element")
    return vector


def _decimal_subset_key(subset, n):
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


class DecimalFuzzyMeasure:
    """Normalized monotone capacity that stores values as ``Decimal``."""

    def __init__(self, n, values, *, allow_float=False):
        _validate_allow_float(allow_float)
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(values, Mapping):
            raise TypeError("values must map every subset to its measure")

        n = int(n)
        normalized = {}
        for subset, raw_value in values.items():
            key = _decimal_subset_key(subset, n)
            if key in normalized:
                raise ValueError(f"duplicate subset measure for {sorted(key)}")
            normalized[key] = _decimal_unit_value(
                raw_value, "measure values", allow_float=allow_float
            )

        if len(normalized) != 1 << n:
            raise ValueError(f"values must define all {1 << n} subsets")

        empty_set = frozenset()
        full_set = frozenset(range(n))
        if normalized[empty_set] != _UNIT_ZERO:
            raise ValueError("a fuzzy measure must assign 0 to the empty set")
        if normalized[full_set] != _UNIT_ONE:
            raise ValueError("a fuzzy measure must assign 1 to the full set")

        for subset, value in normalized.items():
            for index in full_set - subset:
                if value > normalized[subset | {index}]:
                    raise ValueError(
                        "measure must be monotone: adding an element cannot "
                        "decrease its value"
                    )

        self._n = n
        self._values = normalized

    @property
    def n(self):
        return self._n

    def __call__(self, subset):
        return self._values[_decimal_subset_key(subset, self._n)]

    def __repr__(self):
        return f"DecimalFuzzyMeasure(n={self._n})"


def _decimal_measure_from_function(n, capacity_function):
    values = {}
    for mask in range(1 << n):
        subset = frozenset(index for index in range(n) if mask & (1 << index))
        values[subset] = capacity_function(subset)
    values[frozenset()] = _UNIT_ZERO
    values[frozenset(range(n))] = _UNIT_ONE
    return DecimalFuzzyMeasure(n, values)


def _decimal_measure_size(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    return int(n)


def _decimal_weights(weights, *, allow_float=False, require_sum_one=False):
    vector = _decimal_vector(weights, allow_float=allow_float)
    if require_sum_one and sum(vector, _UNIT_ZERO) != _UNIT_ONE:
        raise ValueError("weights must sum to 1")
    return vector


def uniform_measure(n):
    """Return the cardinality capacity ``mu(A) = |A| / n``."""
    n = _decimal_measure_size(n)
    return _decimal_measure_from_function(
        n, lambda subset: Decimal(len(subset)) / Decimal(n)
    )


def relative_measure(n):
    """Return the relative capacity with singleton weights 1, ..., n."""
    n = _decimal_measure_size(n)
    denominator = Decimal(n * (n + 1)) / Decimal(2)
    return _decimal_measure_from_function(
        n,
        lambda subset: sum(
            (Decimal(index + 1) for index in subset), _UNIT_ZERO
        ) / denominator,
    )


def weighted_measure(weights, *, allow_float=False):
    """Return the additive capacity induced by normalized positional weights."""
    vector = _decimal_weights(
        weights, allow_float=allow_float, require_sum_one=True
    )
    return _decimal_measure_from_function(
        len(vector),
        lambda subset: sum(
            (vector[index] for index in subset), _UNIT_ZERO
        ),
    )


def owa_measure(weights, *, allow_float=False):
    """Return a symmetric capacity that yields OWA weights on sorted values.

    ``weights[0]`` is applied to the largest input, ``weights[1]`` to the
    second largest, and so on. The weights must sum to one.
    """
    vector = _decimal_weights(
        weights, allow_float=allow_float, require_sum_one=True
    )
    cumulative_weights = [_UNIT_ZERO]
    for weight in vector:
        cumulative_weights.append(cumulative_weights[-1] + weight)
    return _decimal_measure_from_function(
        len(vector), lambda subset: cumulative_weights[len(subset)]
    )


def _decimal_sugeno_lambda(weights):
    total = sum(weights, _UNIT_ZERO)
    if total == _UNIT_ONE:
        return _UNIT_ZERO

    positive_count = sum(weight > _UNIT_ZERO for weight in weights)
    if total < _UNIT_ONE:
        if positive_count < 2:
            raise ValueError(
                "singleton weights do not define a normalized k-fuzzy measure"
            )

        def gap(value):
            product = _UNIT_ONE
            for weight in weights:
                product *= _UNIT_ONE + value * weight
            return product - _UNIT_ONE - value

        lower, upper = _UNIT_ZERO, _UNIT_ONE
        while gap(upper) <= _UNIT_ZERO:
            upper *= 2
        for _ in range(200):
            middle = (lower + upper) / Decimal(2)
            if middle == lower or middle == upper:
                break
            middle_gap = gap(middle)
            if middle_gap == _UNIT_ZERO:
                return middle
            if middle_gap < _UNIT_ZERO:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / Decimal(2)

    if any(weight == _UNIT_ONE for weight in weights):
        return -_UNIT_ONE

    def gap(value):
        product = _UNIT_ONE
        for weight in weights:
            product *= _UNIT_ONE + value * weight
        return product - _UNIT_ONE - value

    lower, upper = -_UNIT_ONE, _UNIT_ZERO
    for _ in range(200):
        middle = (lower + upper) / Decimal(2)
        if middle == lower or middle == upper:
            break
        middle_gap = gap(middle)
        if middle_gap == _UNIT_ZERO:
            return middle
        if middle_gap > _UNIT_ZERO:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / Decimal(2)


def k_fuzzy_measure(singleton_weights, *, allow_float=False):
    """Return the normalized Sugeno ``k``-fuzzy capacity.

    The values are determined from the singleton weights by
    ``mu(A) = (prod(1 + k*w_i) - 1) / k``. The parameter ``k`` is selected
    so that the measure of the full set is one; when ``k = 0``, this reduces
    to the additive measure. Some singleton-weight vectors cannot define a
    normalized ``k``-fuzzy measure and are rejected.
    """
    vector = _decimal_weights(singleton_weights, allow_float=allow_float)
    parameter = _decimal_sugeno_lambda(vector)

    def capacity(subset):
        if not subset:
            return _UNIT_ZERO
        subset_weights = [vector[index] for index in subset]
        if parameter == _UNIT_ZERO:
            return sum(subset_weights, _UNIT_ZERO)
        if parameter == -_UNIT_ONE:
            product = _UNIT_ONE
            for weight in subset_weights:
                product *= _UNIT_ONE - weight
            return _UNIT_ONE - product
        product = _UNIT_ONE
        for weight in subset_weights:
            product *= _UNIT_ONE + parameter * weight
        return (product - _UNIT_ONE) / parameter

    return _decimal_measure_from_function(len(vector), capacity)


def _decimal_overlap_value(quasi_overlap, value, measure_value, allow_float):
    result = quasi_overlap(value, measure_value)
    return _decimal_unit_value(
        result, "quasi_overlap result", allow_float=allow_float
    )


def _resolve_decimal_measure(measure, n, allow_float=False):
    if measure is None:
        size = Decimal(n)
        return lambda subset: Decimal(len(subset)) / size

    if isinstance(measure, DecimalFuzzyMeasure):
        if measure.n != n:
            raise ValueError(f"measure is defined for {measure.n} elements, not {n}")
        return measure

    if isinstance(measure, Mapping):
        return DecimalFuzzyMeasure(n, measure, allow_float=allow_float)

    if isinstance(measure, (str, bytes)) or not isinstance(measure, Iterable):
        raise TypeError(
            "measure must be a DecimalFuzzyMeasure, full subset "
            "mapping, or symmetric measure vector"
        )
    cardinality_values = [
        _decimal_unit_value(value, "measure values", allow_float=allow_float)
        for value in measure
    ]
    if len(cardinality_values) != n + 1:
        raise ValueError(
            "a symmetric measure vector must contain n + 1 values: "
            "[mu(empty), ..., mu(full)]"
        )
    if cardinality_values[0] != _UNIT_ZERO:
        raise ValueError("a fuzzy measure must assign 0 to the empty set")
    if cardinality_values[-1] != _UNIT_ONE:
        raise ValueError("a fuzzy measure must assign 1 to the full set")
    if any(
        left > right
        for left, right in zip(cardinality_values, cardinality_values[1:])
    ):
        raise ValueError("a symmetric fuzzy measure must be nondecreasing by set size")
    return lambda subset: cardinality_values[len(subset)]


def choquet_integral(
    values, measure=None, *, quasi_overlap=None, allow_float=False
):
    """Return the discrete Choquet or quasi-overlap Choquet aggregation.

    Values and custom measure entries must be Decimal-compatible exact inputs:
    ``Decimal``, integers, or decimal strings. ``float`` is rejected unless
    ``allow_float=True``, in which case it is converted through ``str``.
    A full mapping can describe a nonsymmetric capacity. ``None`` selects the
    cardinality capacity. When ``quasi_overlap`` is supplied, it must be a
    callable ``O(x, mu)`` and the CQO formula
    ``sum(O(x_i, mu(A_i)) - O(x_(i-1), mu(A_i)))`` is used. Omitting it uses
    the classical product formula.
    """
    if quasi_overlap is not None and not callable(quasi_overlap):
        raise TypeError("quasi_overlap must be callable")
    vector = _decimal_vector(values, allow_float=allow_float)
    if measure is None and quasi_overlap is None:
        return sum(vector, _UNIT_ZERO) / Decimal(len(vector))

    capacity = _resolve_decimal_measure(
        measure, len(vector), allow_float=allow_float
    )
    order = sorted(range(len(vector)), key=vector.__getitem__)

    result = _UNIT_ZERO
    previous_value = _UNIT_ZERO
    for rank, index in enumerate(order):
        upper_level_set = frozenset(order[rank:])
        value = vector[index]
        measure_value = capacity(upper_level_set)
        if quasi_overlap is None:
            result += (value - previous_value) * measure_value
        else:
            result += _decimal_overlap_value(
                quasi_overlap, value, measure_value, allow_float
            ) - _decimal_overlap_value(
                quasi_overlap, previous_value, measure_value, allow_float
            )
        previous_value = value
    return result


def sugeno_integral(values, measure=None, *, allow_float=False):
    """Return the discrete Sugeno aggregation as a ``Decimal``.

    The accepted measure formats are the same as for :func:`choquet_integral`.
    """
    vector = _decimal_vector(values, allow_float=allow_float)
    capacity = _resolve_decimal_measure(
        measure, len(vector), allow_float=allow_float
    )
    order = sorted(range(len(vector)), key=vector.__getitem__)

    result = _UNIT_ZERO
    for rank, index in enumerate(order):
        upper_level_set = frozenset(order[rank:])
        result = max(result, min(vector[index], capacity(upper_level_set)))
    return result


class A_ch:
    """Callable Decimal Choquet-integral aggregation object."""

    name = "ch"

    def __init__(self, measure=None, *, quasi_overlap=None, allow_float=False):
        self.__measure = measure
        if quasi_overlap is not None and not callable(quasi_overlap):
            raise TypeError("quasi_overlap must be callable")
        self.__quasi_overlap = quasi_overlap
        self.__allow_float = _validate_allow_float(allow_float)

    def __call__(self, values):
        return choquet_integral(
            values,
            measure=self.__measure,
            quasi_overlap=self.__quasi_overlap,
            allow_float=self.__allow_float,
        )

    def __repr__(self):
        if self.__measure is None and self.__quasi_overlap is None:
            return f"A_ch(allow_float=True)" if self.__allow_float else "A_ch()"
        arguments = []
        if self.__measure is not None:
            arguments.append(f"measure={self.__measure!r}")
        if self.__quasi_overlap is not None:
            arguments.append(f"quasi_overlap={self.__quasi_overlap!r}")
        if self.__allow_float:
            arguments.append("allow_float=True")
        return f"A_ch({', '.join(arguments)})"

    def __str__(self):
        return self.__repr__()

    def toLatex(self, params):
        return f"C_{{\\mu}}({params})"


class A_sg:
    """Callable Decimal Sugeno-integral aggregation object."""

    name = "sg"

    def __init__(self, measure=None, *, allow_float=False):
        self.__measure = measure
        self.__allow_float = _validate_allow_float(allow_float)

    def __call__(self, values):
        return sugeno_integral(
            values, measure=self.__measure, allow_float=self.__allow_float
        )

    def __repr__(self):
        if self.__measure is None:
            return f"A_sg(allow_float=True)" if self.__allow_float else "A_sg()"
        if self.__allow_float:
            return f"A_sg(measure={self.__measure!r}, allow_float=True)"
        return f"A_sg(measure={self.__measure!r})"

    def __str__(self):
        return self.__repr__()

    def toLatex(self, params):
        return f"S_{{\\mu}}({params})"


def exponential2(y, p=0, q=0):
    pass


def exponential3(y, p=0, q=0):
    pass


def sin_aggregation(y):
    pass


def quasi_arithmeric(y, function):
    pass


if __name__ == "__main__":
    y = [-10, -22, 10]
    # 1
    print(arithmetic(y))

    # 2
    print(quadratic(y))

    # 3
    print(geometric(y))

    # # 4
    print(harmonic(y))

    # 5
    print(power(y, r=1))

    # 6
    print(exponential(y, r=1))

    # 7
    print(lehmer(y, r=0))

    # 8
    print(arithmetic_min(y, p=0))

    # 9
    print(arithmetic_max(y, p=0))

    # 10
    print(median(y))

    # 11
    print(olimpic(y))
