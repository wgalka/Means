import math

import numpy as np


def exponential(y, r=1):
    r"""
    Mean is given by equation:
    $A_6^{(r)}(x_1,...,x_n)= \frac{1}{r}\ln
    \Big(\frac{1}{n} \sum \limits_{k=1}^{n} e^{rx_k}\Big)$, where
    $r \in \mathbb{R}$, $r \neq 0$

    Numerically stable implementation using Log-Sum-Exp trick:
    A_ex(y) = 1/r * [ a_max + ln( sum( e^(r * y_i - a_max) ) ) - ln(n) ]
    where a_max = max(r * y_j)

    :param y: array of values
    :param r: r != 0
    :return: value of exponential mean
    """
    size = len(y)
    if r == 0:
        raise ValueError("parameter r should be != 0 ")

    r_y = [num * r for num in y]
    a_max = max(r_y)
    _sum = 0
    for val in r_y:
        _sum += math.exp(val - a_max)

    return (1 / r) * (a_max + math.log(_sum) - math.log(size))
