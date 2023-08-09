import math
import warnings
from decimal import Decimal

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
    if np.ndim(y) != 1:
        raise ValueError("y must be 1 dimensional array")
    if r == 0:
        raise ValueError("parameter r should be != 0 ")
    size = len(y)
    y = np.array(y)
    with np.errstate(divide='ignore'):
        return (1 / r) * np.lib.scimath.log(np.sum(np.exp(y * r)) / size)


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
    n = len(y)
    apply_func = np.array([function(x) for x in y])
    value = np.sum(apply_func) / n
    return inversefunc(function, value)


if __name__ == "__main__":
    print(quasi_arithmeric.__str__())
    print(arithmetic_min([0.2, 0.4, 0.6], 0))
