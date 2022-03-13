import math

import numpy as np


# 1 Arithmetic aggregation
def arithmetic(a):
    return sum(a) / len(a)


# 2
def quadratic(a):
    size = len(a)
    with np.errstate(divide='ignore'):
        return np.sqrt(np.sum(a ** 2) / size)


# 3
def geometric(self, a, axis=0, dtype=None):
    # If a is not np class then try to convert
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=dtype)

    if isinstance(a, np.ma.MaskedArray):
        size = a.count(axis)
    else:
        if axis is None:
            a = a.ravel()
            size = a.shape[0]
        else:
            size = a.shape[axis]

    wk = 1 / size
    with np.errstate(divide='ignore'):
        return np.prod(a ** wk, axis=axis, dtype=dtype)


# 4
def harmonic(a):
    if type(a).__module__ != np.__name__:
        a = np.array(a)
    size = len(a)
    return size / (np.sum(1 / a))


# 5
def power(a, r=1):
    if r == 0:
        raise ValueError("parameter r should not be = 0 ")
    if 0 in a and r < 0:
        return 0
    else:
        result = (np.sum(a ** r) / a.shape[0]) ** (1 / r)
        return result


# 6
def exponential(a, r=1, axis=0, dtype=None):
    if r == 0:
        raise ValueError("parameter r should be >= 0 ")
    # If a is not np class then try to convert
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=dtype)

    if isinstance(a, np.ma.MaskedArray):
        size = a.count(axis)
    else:
        if axis is None:
            a = a.ravel()
            size = a.shape[0]
        else:
            size = a.shape[axis]
    with np.errstate(divide='ignore'):
        return (1 / r) * np.lib.scimath.log(np.sum(np.exp(a * r), axis=axis, dtype=dtype) / size)


# 7
def lehmer(a, r=0):
    if 0 in a and r - 1 <= 0:
        return 0
    else:
        r2 = (r - 1)
        nominator = np.sum(a ** r)
        with np.errstate(divide='ignore'):  #
            denominator = np.sum(a ** r2)
        if denominator == 0:
            return 0
        else:
            return nominator / denominator


# 8
# poprawiona 21.03.2021
def arithmetic_min(a, p=0):
    if len(a.shape) != 1:
        raise Exception("arithmetic_min function given array should be 1D")
    min = np.min(a)
    return p * np.mean(a) + (1 - p) * min


# 9
def arithmetic_max(a, p=0):
    if len(a.shape) != 1:
        raise Exception("arithmetic_max function given array should be 1D")
    max = np.max(a)
    return p * np.mean(a) + (1 - p) * max


# 10
def median(a):
    if len(a.shape) != 1:
        raise Exception("median function given array should be 1D")
    return np.median(a)


# 11
def olimpic(a):
    if len(a.shape) != 1:
        raise Exception("olimpic function given array should be 1D")
    # b = np.max(a, 0)  # max
    # c = np.min(a, 0)  # min
    if len(a) < 3:
        return np.mean(a)
    else:
        d = np.sort(a)[1:-1]
        return np.mean(d)


def exponential2(a, p=0, q=0):
    mean = np.mean(p ** (q ** a))
    firstlog = math.log(mean, p)
    result = math.log(firstlog, q)
    return result


def exponential3(a, p=0, q=0):
    result = (1 / q) * math.log(np.mean(p ** (q * a)), p)
    return result
