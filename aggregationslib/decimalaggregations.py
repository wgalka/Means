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


def choquet_integral(values, measure=None, *, allow_float=False):
    """Return the discrete Choquet aggregation as a ``Decimal``.

    Values and custom measure entries must be Decimal-compatible exact inputs:
    ``Decimal``, integers, or decimal strings. ``float`` is rejected unless
    ``allow_float=True``, in which case it is converted through ``str``.
    A full mapping can describe a nonsymmetric capacity. ``None`` selects the
    cardinality capacity. The input and result lie in ``[0, 1]``.
    """
    vector = _decimal_vector(values, allow_float=allow_float)
    if measure is None:
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
        result += (value - previous_value) * capacity(upper_level_set)
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

    def __init__(self, measure=None, *, allow_float=False):
        self.__measure = measure
        self.__allow_float = _validate_allow_float(allow_float)

    def __call__(self, values):
        return choquet_integral(
            values, measure=self.__measure, allow_float=self.__allow_float
        )

    def __repr__(self):
        if self.__measure is None:
            return f"A_ch(allow_float=True)" if self.__allow_float else "A_ch()"
        if self.__allow_float:
            return f"A_ch(measure={self.__measure!r}, allow_float=True)"
        return f"A_ch(measure={self.__measure!r})"

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
