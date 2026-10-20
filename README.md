# Means
Means, Aggregation functions...

Documentation:

- English: https://wgalka.github.io/Means/en/
- Polski: https://wgalka.github.io/Means/pl/

Aggregations functions are:
- Conjuctive - the final aggregated value will always be influenced by the smallest value among the inputs.
- Disjnuctive - the aggregated value will always be influenced by the largest value among the inputs.

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



## Aggregation functions

An aggregation function on the unit interval [1] is an increasing mapping
$A:[0,1]^n\to[0,1]$, $n \in \mathbb{N}$, $n\ge 2$, satisfying
$A(0,\dots,0) = 0$ and $A(1,\dots,1) = 1$. For inputs on another interval
$[a,b]\subset\mathbb{R}$, an $[a,b]$-aggregation is an increasing mapping
$A:[a,b]^n\to[a,b]$ satisfying $A(a,\dots,a)=a$ and $A(b,\dots,b)=b$.

The following aggregation functions are implemented in this library. Tags
describe properties of each formula. `averaging` means the result lies between
the minimum and maximum inputs; `conjunctive` means it does not exceed the
minimum; `symmetric` means permuting the inputs does not change the result.
`aggregation interval` states the intervals on which the formula is an
aggregation function: an increasing map $A:[a,b]^n\to[a,b]$ satisfying
$A(a,\dots,a)=a$ and $A(b,\dots,b)=b$. `any [a,b] ⊂ ℝ` includes intervals
with negative values; `any [a,b] ⊆ [0,∞)` is restricted to nonnegative
intervals. `[0,1]` means only the unit interval. These are mathematical
properties; most callable objects do not validate input bounds at runtime.
Some tags depend on parameters or the fuzzy measure.

$\mathbf{A_{ch}^{(\mu)}}$ **- Choquet integral [1]**

Tags: `averaging` `aggregation interval: [0,1]` `symmetric if μ is symmetric`

For an input vector $(x_1,\dots,x_n)\in[0,1]^n$, let $\sigma$
sort its indices so that $x_{\sigma(1)} \leq \dots \leq x_{\sigma(n)}$.
Define $A_{\sigma(i)}=\{\sigma(i),\dots,\sigma(n)\}$ and $x_{\sigma(0)}=0$:

$$A_{ch}^{(\mu)}(x_1,\dots,x_n)=\sum_{i=1}^{n}(x_{\sigma(i)}-x_{\sigma(i-1)})\,\mu(A_{\sigma(i)})$$

With the default cardinality measure $\mu(A)=|A|/n$, this is the arithmetic
mean, $A_{ch}^{(\mu)}(x_1,\dots,x_n)=\frac{1}{n}\sum_{i=1}^{n}x_i$.

$\mathbf{A_{sg}^{(\mu)}}$ **- Sugeno integral [1]**

Tags: `averaging` `aggregation interval: [0,1]` `symmetric if μ is symmetric`

For an input vector $(x_1,\dots,x_n)\in[0,1]^n$, let $\sigma$ sort its indices
so that $x_{\sigma(1)} \leq \dots \leq x_{\sigma(n)}$, and define
$A_{\sigma(i)}=\{\sigma(i),\dots,\sigma(n)\}$:

$$A_{sg}^{(\mu)}(x_1,\dots,x_n)=\max_{1\leq i\leq n}\min(x_{\sigma(i)},\mu(A_{\sigma(i)}))$$

$\mathbf{A_{ar}}$ **- Arithmetic mean**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊂ ℝ`

$$A_{ar}(x_1,...,x_n)=\frac{1}{n}\sum \limits_{k=1}^{n} x_k$$

$\mathbf{A_{qd}}$ **- Quadratic mean**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊆ [0,∞)`

$$A_{qd}(x_1,...,x_n)=\sqrt{\frac{1}{n}\sum
\limits_{k=1}^{n} x_k^2}$$

$\mathbf{A_{pr}}$ **- Product aggregation**

Tags: `conjunctive` `symmetric` `aggregation interval: [0,1]`

$$A_{pr}(x_1, x_2, \dots, x_n) = \prod_{k=1}^n x_k$$

$\mathbf{A_{gm}}$ **- Geometric mean**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊆ [0,∞)`

$$A_{gm}(x_1,...,x_n)=\sqrt[n]{\prod
\limits_{k=1}^{n} x_k}$$

$\mathbf{A_{hm}}$ **- Harmonic mean**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊆ [0,∞)`

$$
A_{hm}(x_1,...,x_n) =
\begin{cases}
0, & \text{if } \exists k,\, 1 \leq k \leq n : x_k = 0 \\
\frac{n}{\sum\limits_{k=1}^{n} \frac{1}{x_k}}, & \text{otherwise}
\end{cases}
$$


$\mathbf{A_{pw}^{(r)}}$ **- Power mean**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊆ [0,∞)` `r ≠ 0`

$$
A_{pw}^{(r)}(x_1,...,x_n) =
\begin{cases}
0, & r < 0, \; \exists k,\, 1 \leq k \leq n : x_k = 0 \\
\Big(\frac{1}{n} \sum\limits_{k=1}^{n} x_k^r \Big)^{\frac{1}{r}}, & \text{otherwise}
\end{cases}
$$


special cases of power means $A_{\text{pw}}^r$ [1] are:
- Arithmetic Mean where r=1;
- Quadratic Mean where r=2;
- Harmonic Mean where r=-1;
- Geometric Mean  obtained as the limit $\lim_{r \to 0} A_{\text{pw}}^r$.


----

$\mathbf{A_{ex}^{(r)}}$ **- Exponential mean**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊂ ℝ` `r ≠ 0`

$$A_{ex}^{(r)}(x_1,...,x_n)= \frac{1}{r}\ln
\Big(\frac{1}{n} \sum \limits_{k=1}^{n} e^{rx_k}\Big), \text{where
} r \in \mathbb{R}, r \neq 0$$

For computer implementations, to avoid numerical instability (overflow when $r x_k > 709.78$ or underflow when all $r x_k < -709.78$), we use a numerically stable Log-Sum-Exp formulation by shifting the exponents by $a_{max} = \max_{j} (r x_j)$:

$$A_{ex}^{(r)}(x_1,...,x_n) = \frac{1}{r} \left[ a_{max} + \ln \left( \sum_{i=1}^n e^{r x_i - a_{max}} \right) - \ln(n) \right]$$

$\mathbf{A_{lm}}$ **- Lehmer mean**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊆ [0,∞) if 0 ≤ r ≤ 1`

$$A_{lm}(x_1,...,x_n)= \begin{cases}
0, & r\leq 1, \exists_{1 \leq k \leq n} x^{k} = 0 \\
\frac{\sum \limits_{k=1}^{n}x_k^r}{\sum \limits_{k=1}^{n}x_k^{r-1}}, & \text{otherwise}
\end{cases}$$

$\mathbf{A_{amn}^{(p)}}$ **- Arithmetic minimum mean**

Tags: `averaging` `symmetric` `conjunctive if p = 0` `aggregation interval: any [a,b] ⊂ ℝ` `p ∈ [0,1]`

$$A_{amn}^{(p)}(x_1,...,x_n)=\frac{p}{n}\sum \limits_{k=1}^{n} x_k+
(1-p) \min \limits_{1 \leq k \leq n}x_k, p \in [0, 1]$$

$\mathbf{A_{amx}^{(p)}}$ **- Arithmetic maximum mean**

Tags: `averaging` `symmetric` `disjunctive if p = 0` `aggregation interval: any [a,b] ⊂ ℝ` `p ∈ [0,1]`

$$A_{amx}^{(p)}(x_1,...,x_n)=\frac{p}{n}\sum \limits_{k=1}^{n} x_k+
(1-p) \max \limits_{1\leq k \leq n}x_k, p \in [0, 1]$$


### OWA

In the case of n = 2 (when only two values are aggregated) the
calculation of the median and the Olympic aggregation are reduced to the case of calculating the
arithmetic mean of the given values.

$\mathbf{A_{md}}$ **- Median - ordered weighted aggregation**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊂ ℝ`

$$A_{md}(x_1,\dots,x_n) =
\begin{cases}
y_{(n+1)/2},&\text{if $n$ is odd}\\
\frac{y_{n/2}+y_{(n/2)+1}}{2},&\text{if $n$ is even}
\end{cases},$$

$\mathbf{A_{ol}}$ **- Olimpic aggregation**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊂ ℝ`

$$ A_{ol}(x_1,\dots,x_n) = \frac{1}{n-2} \sum \limits_{k=2}^{n-1}  y_k, \quad \text{where } \{y_1, \dots, y_n\} = \{x_1, \dots, x_n\},\ y_1 \leq y_2 \leq \dots \leq y_n.$$

$\mathbf{A_{oln}^{(p)}}$ **- Olimpic aggregation**

Tags: `averaging` `symmetric` `aggregation interval: any [a,b] ⊂ ℝ` `p ∈ ℕ, p ≥ 1`

We can specify how many greatest and smallest records remove

$$ A_{oln}^{(p)}(x_1,\dots,x_n) = \frac{1}{n-2p} \sum \limits_{k=p+1}^{n-p}  y_k, \quad \text{where } \{y_1, \dots, y_n\} = \{x_1, \dots, x_n\},\ y_1 \leq y_2 \leq \dots \leq y_n, p \in \mathbb{N}, p\geq 1$$

--------------------

$\mathbf{A_{ln}}$ **- Logaritmic aggregation [2]**

Tags: `averaging (continuous formula)` `symmetric` `aggregation interval (after continuous extension): any [a,b] ⊂ (0,∞)`

$$
A_{ln}\left(x_1, x_2, \ldots, x_n\right)=(n-1) ! \sum_{i=1}^n \frac{x_i}{\prod_{\substack{j=1 \\ j \neq i}}^n \log \left(x_i / x_j\right)}
$$

The formula has a continuous averaging extension when input values coincide,
but the current implementation does not handle ties: a zero denominator makes
it return `0`.

### $\mathbf{A^{(p)}_{A_1,A_2}}$ - Convex combinations of aggregation functions

Tags: `aggregation interval: common interval of A₁ and A₂` `averaging if both inputs are averaging` `p ∈ [0,1]`

We can construct covex combination of aggregation functions as $A^{(p)}_{A_1,A_2}=pA_1+(1-p)A_2$, where $A_1, A_2$ are aggregation functions and $p \in [0,1]$.



## References

1. Beliakov, G., Bustince, H., and Calvo, T.: *A Practical Guide to Averaging Functions*. Berlin: Springer, vol. 329, 2016.
2. [Mustonen, Seppo. (2010). Logarithmic mean for several arguments. ](https://www.researchgate.net/publication/228886844_Logarithmic_mean_for_several_arguments)
