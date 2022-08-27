import math

import numpy as np


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
    size = len(y)
    if r == 0:
        raise ValueError("parameter r should be != 0 ")

    _sum = 0
    for idx, num in enumerate(y):
        _sum += math.exp(num * r)

    return (1 / r) * math.log(_sum / size)
