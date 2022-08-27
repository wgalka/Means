import logging
import numbers
import warnings

from decimal import Decimal

# 1 Arithmetic aggregation
import numpy as np


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
    """
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

    # # Comparasion to numpy
    #     # np_sum = np.sum(np.exp(y * int(r)))
    #     # np_sum_log = np.lib.scimath.log(np_sum / len(y))

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
