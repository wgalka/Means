# Wolfram reference data

`wolfram_measure_reference.json` contains results generated with Wolfram
Engine 15.0, independently of the Python implementations. Tests load the
saved results and do not require Wolfram. Generator scripts are kept in
the ignored `tmp/wolfram/` directory, outside the tracked source files.

The fixture covers 22 configurations of six measures and 166 input cases
(each with a Choquet and a Sugeno result). Boundary vectors come first:
zeros, ones, constant values and an indicator vector, followed by ties,
permuted coordinates and values close to the interval endpoints.
Every measure also includes its value on every subset.

## Definitions used for the reference

For the universe $N=\{1,\ldots,n\}$ and $A\subseteq N$:

| Measure | Definition |
| --- | --- |
| Uniform | $\mu(A)=|A|/n$ |
| Exponential cardinality | $\mu(A)=(|A|/n)^q$, $q>0$ |
| Relative | $\mu(A)=\sum_{i\in A}i/\sum_{i=1}^n i$ |
| Weighted | $\mu(A)=\sum_{i\in A}w_i$, where $\sum_i w_i=1$ |
| OWA | $\mu(A)=\sum_{j=1}^{|A|}w_j$, with weights starting at the largest input |
| k-fuzzy (Sugeno lambda) | $\mu(A)=(\prod_{i\in A}(1+\lambda w_i)-1)/\lambda$ |

For k-fuzzy, the reference solves the normalization equation symbolically
after cancelling its trivial zero factor. When the singleton weights sum
to one, the measure is additive ($\lambda=0$). Cases include positive and
negative parameters, both sides of zero, and the explicit boundary extension
$\lambda=-1$, giving $1-\prod_{i\in A}(1-w_i)$.

Choquet is computed using distinct positive input levels
$0=t_0<t_1<\cdots<t_m$:

$$C_\mu(x_1,\ldots,x_n)=\sum_{j=1}^m(t_j-t_{j-1})\mu(\{i:x_i\geq t_j\}).$$

Sugeno is computed by enumerating all nonempty subsets:

$$S_\mu(x_1,\ldots,x_n)=\max_{\emptyset\ne A\subseteq N}
\min\left(\mu(A),\min_{i\in A}x_i\right).$$

Inputs are exact rationals in Wolfram; symbolic results are exported as
decimal strings with 85 significant digits. `capacityByMask` uses bit 0
for the first coordinate. The fixture records the engine version and method.
Exponential cardinality references cover q=1/4, 1/2, 1, 2, 10, 100 and a
singleton universe. The `q` field stores its exponent as a decimal string.

`tests/test_wolfram_measures.py` checks measures, scalar functions, callable
objects (float and Decimal), and ndarray reductions over rows, columns and
three-dimensional arrays. Float comparisons use absolute tolerance
`1e-12`; Decimal uses precision 50 and absolute tolerance `1e-44`.

`wolfram_integral_reference.json` is the earlier, separate integral fixture.
