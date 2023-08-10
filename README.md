# Means
Means, Aggregation functions...

#### Example 1:

```pycon
# example data
data = [0.2, 0.6, 0.7]
# configure function parameters
func1 = A_amn(p=0.5)
# use aggregation funciton
print(func1(data))

# Combine two aggregations - arithmetic mean and minimum
func2 = Combine2Aggregations(A_ar(), min)
# use combination of aggregation funciton
print(func2(data))
```

#### Example2:
To get information about aggregation function you can use `__str__()` or '__repr__()' methods.

```pycon
func1 = A_amn(p=0.5)
print(func1)
>>>A_amn(0.5)

func2 = Combine2Aggregations(A_ar(), A_md())
print(func2)
>>>A_armd

func3 = Combine2Aggregations(A_ar(), A_pw(r=3))
print(func3.__repr__()) # function parameters are printed in order: func1, func2
>>>A_arpw(r=3)
```

`exponential(y, r=1)` is given by equation 

$$
A_6^{(r)}(x_1,...,x_n)= \frac{1}{r}\ln
    \Big(\frac{1}{n} \sum \limits_{k=1}^{n} e^{rx_k}\Big), where
    r \in \mathbb{R}, r \neq 0
$$


# A_ar - Arithmetic mean


# A_qd - Quadratic mean


# A_gm - Geometric mean


# A_hm - Harmonic mean


# A_pw - Power mean


# A_ex, A_ex2, A_ex3 - Exponential mean


# A_lm - Lehmer mean


# A_amn - Arithmetic minimum mean


# A_amx - Arithmetic maximum mean


# A_md - Median - ordered weighted aggregation


# A_ol - Olimpic aggregation

# A_oln - Olimpic aggregation
We can specify how many greatest and smallest records remove

# Combine2Aggregations - Combine aggregation functions
Amn, Amx, Aar , Aex , Amd,
Aow1, Aow1