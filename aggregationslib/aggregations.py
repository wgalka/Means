import math

import numpy as np
import re


class A_ar:
    """
    Arithmetic mean
    """

    def __init__(self):
        pass

    def __call__(self, array):
        """
        Arithmetic aggregation
        :param y: 1 dimensional array
        :return: aggregation value
        """
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        return np.mean(array)

    def __repr__(self):
        return f"A_ar()"

    def __str__(self):
        return f"A_ar"


class A_qd:
    """
    Quadratic mean
    """

    def __init__(self):
        pass

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        y = np.array(array)
        size = len(y)
        with np.errstate(divide='ignore'):
            return np.sqrt(np.sum(y ** 2) / size)

    def __repr__(self):
        return f"A_qd()"

    def __str__(self):
        return f"A_qd"


class A_gm:
    """
    Geometric mean
    """

    def __init__(self):
        pass

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        size = len(array)
        wk = 1 / size
        with np.errstate(divide='ignore'):
            return np.prod(array ** wk)

    def __repr__(self):
        return f"A_gm()"

    def __str__(self):
        return f"A_gm"


class A_hm:
    """
    Harmonic mean
    """

    def __init__(self):
        pass

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        y = np.array(array)
        size = len(y)
        return size / (np.sum(1 / y))

    def __repr__(self):
        return f"A_hm()"

    def __str__(self):
        return f"A_hm"


class A_pw:
    """
    Power mean
    """
    valid_params = ["r"]

    def __init__(self, r):
        self.__r = r

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        if self.__r == 0:
            raise ValueError("parameter r cannot be equal to 0 ")
        # If one 0 occur in array then mean is 0
        if 0 in array and self.__r < 0:
            return 0
        else:
            result = (np.sum(array ** self.__r) / array.shape[0]) ** (1 / self.__r)
            return result

    def __repr__(self):
        return f"A_pw(r={self.__r})"

    def __str__(self):
        return f"A_pw({self.__r})"


class A_ex:
    """
    Exponential mean
    """
    valid_params = ["r"]

    def __init__(self, r):
        self.__r = r

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        if self.__r == 0:
            raise ValueError("parameter r should be != 0 ")
        size = len(array)
        y = np.array(array)
        with np.errstate(divide='ignore'):
            return (1 / self.__r) * np.lib.scimath.log(np.sum(np.exp(y * self.__r)) / size)

    def __repr__(self):
        return f"A_ex(r={self.__r})"

    def __str__(self):
        return f"A_ex({self.__r})"


class A_ex2:
    """
    Exponential2 mean
    """
    valid_params = ["p", "q"]

    def __init__(self, p, q):
        self.__p = p
        self.__q = q

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        y = np.array(array)
        mean = np.mean(self.__p ** (self.__q ** y))
        firstlog = math.log(mean, self.__p)
        result = math.log(firstlog, self.__q)
        return result

    def __repr__(self):
        return f"A_ex2(p={self.__p}, q={self.__q})"

    def __str__(self):
        return f"A_ex2({self.__p}, {self.__q})"


class A_ex3:
    """
    Exponential3 mean
    """
    valid_params = ["p", "q"]

    def __init__(self, p, q):
        self.__p = p
        self.__q = q

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        y = np.array(array)
        result = (1 / self.__q) * math.log(np.mean(self.__p ** (self.__q * y)), self.__p)
        return result

    def __repr__(self):
        return f"A_ex3(p={self.__p}, q={self.__q})"

    def __str__(self):
        return f"A_ex3({self.__p}, {self.__q})"


class A_lm:
    """
    Lehmer mean
    """
    valid_params = ["r"]

    def __init__(self, r):
        self.__r = r

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        # if  array and r below 0 return 0. 0 to power of 0.
        if 0 in array and self.__r - 1 <= 0:
            return 0
        else:
            r2 = (self.__r - 1)
            nominator = np.sum(array ** self.__r)
            with np.errstate(divide='ignore'):  #
                denominator = np.sum(array ** r2)
            if denominator == 0:
                return 0
            else:
                return nominator / denominator

    def __repr__(self):
        return f"A_lm(r={self.__r})"

    def __str__(self):
        return f"A_lm({self.__r})"


class A_amn:
    """
    Arithmetic minimum mean
    """
    valid_params = ["p"]

    def __init__(self, p):
        self.__p = p

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        min_val = np.min(array)
        return self.__p * np.mean(array) + (1 - self.__p) * min_val

    def __repr__(self):
        return f"A_amn(p={self.__p})"

    def __str__(self):
        return f"A_amn({self.__p})"


class A_amx:
    """
    Arithmetic maximum mean
    """
    valid_params = ["p"]

    def __init__(self, p):
        self.__p = p

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        max_val = np.max(array)
        return self.__p * np.mean(array) + (1 - self.__p) * max_val

    def __repr__(self):
        return f"A_amx(p={self.__p})"

    def __str__(self):
        return f"A_amx({self.__p})"


class A_md:
    """
    Median (Ordered weighted aggregation)
    """

    def __init__(self):
        pass

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        return np.median(array)

    def __repr__(self):
        return f"A_md()"

    def __str__(self):
        return f"A_md"


class A_ol:
    """
    Olimpic mean
    """

    def __init__(self):
        pass

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        if len(array) < 3:
            return np.mean(array)
        else:
            d = np.sort(array)[1:-1]
            return np.mean(d)

    def __repr__(self):
        return f"A_ol()"

    def __str__(self):
        return f"A_ol"


class A_oln:
    """
    Generalized olimpic mean
    """
    valid_params = ['n']

    def __init__(self, n):
        self.__n = n

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        if len(array) <= self.__n * 2:
            return np.mean(array)
        else:
            d = np.sort(array)[self.__n:-self.__n]
            return np.mean(d)

    def __repr__(self):
        return f"A_oln(n={self.__n})"

    def __str__(self):
        return f"A_oln({self.__n})"


class Combine2Aggregations:
    def __init__(self, aggregation_function1, aggregation_function2, p=0.5):
        self.__agg1 = aggregation_function1
        self.__agg2 = aggregation_function2
        self.__p = p

    def __call__(self, array):
        if np.ndim(array) != 1:
            raise ValueError("y must be 1 dimensional array")
        return self.__p * self.__agg1(array) + (1 - self.__p) * self.__agg2(array)

    def __str__(self):
        pattern = r'A_(.*?)\('
        params_pattern = r'((.*?)\)'

        agg1 = re.search(pattern, self.__agg1.__str__).group(1)
        agg1_param = re.search(params_pattern, self.__agg1.__str__).group(1)

        agg2 = re.search(pattern, self.__agg2.__str__).group(1)
        agg2_param = re.search(params_pattern, self.__agg2.__str__).group(1)

        return f"A_{agg1}{agg2}({agg1_param}, {agg2_param})"

    def __repr__(self):
        pattern = r'A_(.*?)\('
        params_pattern = r'((.*?)\)'

        agg1 = re.search(pattern, self.__agg1.__repr__).group(1)
        agg1_param = re.search(params_pattern, self.__agg1.__repr__).group(1)

        agg2 = re.search(pattern, self.__agg2.__repr__).group(1)
        agg2_param = re.search(params_pattern, self.__agg2.__repr__).group(1)

        return f"A_{agg1}{agg2}({agg1_param}, {agg2_param})"
