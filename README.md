# Means
Means, Aggregation functions...



#### Example 1:
Mix numpy and pure python example:
```python
import numpy as np
t1 = [0.0, 0.0, 0.0, 0.0, 0.1]
t2 = [1 - x for x in t1]
mean1 = np.mean(t1)
mean2 = np.mean(t2)
print(t1, mean1)
print(t2, mean2)
```
In output we can see that returned value does not sum to 1. It have impact on comparasion
```pycon
>>> [0.0, 0.0, 0.0, 0.0, 0.1] 0.02
>>> [1.0, 1.0, 1.0, 1.0, 0.9] 0.9800000000000001
```


#### Using 1:
```python
from aggregationslib.aggregation import arithmetic

t1 = [0.0, 0.0, 0.0, 0.0, 0.1]
t2 = [1 - x for x in t1]
mean1 = arithmetic_(t1)
mean2 = arithmetic_(t2)
print(t1, mean1)
print(t2, mean2)
```
In implementation we obtain exact number:
```pycon
>>> [0.0, 0.0, 0.0, 0.0, 0.1] 0.02
>>> [1.0, 1.0, 1.0, 1.0, 0.9] 0.98
```

`exponential(y, r=1)` is given by equation $A_6^{(r)}(x_1,...,x_n)= \frac{1}{r}\ln
    \Big(\frac{1}{n} \sum \limits_{k=1}^{n} e^{rx_k}\Big)$, where
    $r \in \mathbb{R}$, $r \neq 0$


# 1
arithmetic(y)

# 2
quadratic(y)

# 3
geometric(y)

# 4
harmonic(y)

# 5
power(y, r=1)

# 6
exponential(y, r=1)

# 7
lehmer(y, r=0)

# 8
arithmetic_min(y, p=0)

# 9
arithmetic_max(y, p=0)

# 10
median(y)

# 11
olimpic(y)