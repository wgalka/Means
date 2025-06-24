Means
=====

Means, Aggregation functions…

Aggregations functions are: - Conjuctive - the final aggregated value
will always be influenced by the smallest value among the inputs. -
Disjnuctive - the aggregated value will always be influenced by the
largest value among the inputs.

Example 1:
^^^^^^^^^^

.. code:: pycon

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

Example2:
^^^^^^^^^

To get information about aggregation function you can use ``__str__()``
or ‘**repr**\ ()’ methods.

.. code:: pycon

   func1 = A_amn(p=0.5)
   print(func1)
   >>>A_amn(0.5)

   func2 = Combine2Aggregations(A_ar(), A_md())
   print(func2)
   >>>A_armd

   func3 = Combine2Aggregations(A_ar(), A_pw(r=3))
   print(func3.__repr__()) # function parameters are printed in order: func1, func2
   >>>A_arpw(r=3)

``exponential(y, r=1)`` is given by equation

.. math::


   A_6^{(r)}(x_1,...,x_n)= \frac{1}{r}\ln
   \Big(\frac{1}{n} \sum \limits_{k=1}^{n} e^{rx_k}\Big), where
   r \in \mathbb{R}, r \neq 0

Aggregation functions
---------------------

Aggregation function [1] is a mapping :math:`A:[0,1]^n\to[0,1]`,
:math:`n \in \mathbb{N}`, :math:`n\ge 2` which is is increasing and
fulfills two boundary conditions, namely :math:`A(0,\dots,0) = 0`,
:math:`A(1,\dots,1) = 1`.

The floowoing aggregation functions are implemented in this library:

:math:`\mathbf{A_{ar}}` **- Arithmetic mean**

.. math:: A_{ar}(x_1,...,x_n)=\frac{1}{n}\sum \limits_{k=1}^{n} x_k

:math:`\mathbf{A_{qd}}` **- Quadratic mean**

.. math::

   A_{qd}(x_1,...,x_n)=\sqrt{\frac{1}{n}\sum
   \limits_{k=1}^{n} x_k^2}

:math:`\mathbf{A_{pr}}` **- Product aggregation**

.. math:: A_{pr}(x_1, x_2, \dots, x_n) = \prod_{k=1}^n x_k

:math:`\mathbf{A_{gm}}` **- Geometric mean**

.. math::

   A_{gm}(x_1,...,x_n)=\sqrt[n]{\prod
   \limits_{k=1}^{n} x_k}

:math:`\mathbf{A_{hm}}` **- Harmonic mean**

.. math::


   A_{hm}(x_1,...,x_n) =
   \begin{cases}
   0, & \text{if } \exists k,\, 1 \leq k \leq n : x_k = 0 \\
   \frac{n}{\sum\limits_{k=1}^{n} \frac{1}{x_k}}, & \text{otherwise}
   \end{cases}

:math:`\mathbf{A_{pw}^{(r)}}` **- Power mean**

.. math::


   A_{pw}^{(r)}(x_1,...,x_n) =
   \begin{cases}
   0, & r < 0, \; \exists k,\, 1 \leq k \leq n : x_k = 0 \\
   \Big(\frac{1}{n} \sum\limits_{k=1}^{n} x_k^r \Big)^{\frac{1}{r}}, & \text{otherwise}
   \end{cases}

special cases of power means :math:`A_{\text{pw}}^r` [1] are: -
Arithmetic Mean where r=1; - Quadratic Mean where r=2; - Harmonic Mean
where r=-1; - Geometric Mean obtained as the limit
:math:`\lim_{r \to 0} A_{\text{pw}}^r`.

--------------

:math:`\mathbf{A_{ex}^{(r)}}` **- Exponential mean**

.. math::

   A_{ex}^{(r)}(x_1,...,x_n)= \frac{1}{r}\ln
   \Big(\frac{1}{n} \sum \limits_{k=1}^{n} e^{rx_k}\Big), \text{where
   } r \in \mathbb{R}$, $r \neq 0

:math:`\mathbf{A_{lm}}` **- Lehmer mean**

.. math:: A_{lm}(x_1,...,x_n)= \frac{\sum \limits_{k=1}^{n}x_k^r}{\sum \limits_{k=1}^{n}x_k^{r-1}} 

:math:`\mathbf{A_{amn}^{(p)}}` **- Arithmetic minimum mean**

.. math::

   A_{amn}^{(p)}(x_1,...,x_n)=\frac{p}{n}\sum \limits_{k=1}^{n} x_k+
   (1-p) \min \limits_{1 \leq k \leq n}x_k,

:math:`\mathbf{A_{amx}^{(p)}}` **- Arithmetic maximum mean**

.. math::

   A_{amx}^{(p)}(x_1,...,x_n)=\frac{p}{n}\sum \limits_{k=1}^{n} x_k+
   (1-p) \max \limits_{1\leq k \leq n}x_k,

OWA
~~~

In the case of n = 2 (when only two values are aggregated) the
calculation of the median and the Olympic aggregation are reduced to the
case of calculating the arithmetic mean of the given values.

:math:`\mathbf{A_{md}}` **- Median - ordered weighted aggregation**

.. math::

   A_{md}(x_1,\dots,x_n) =
   \begin{cases}
   y_{(n+1)/2},&\text{if $n$ is odd}\\
   \frac{y_{n/2}+y_{(n/2)+1}}{2},&\text{if $n$ is even}
   \end{cases},

:math:`\mathbf{A_{ol}}` **- Olimpic aggregation**

.. math::  A_{ol}(x_1,\dots,x_n) = \frac{1}{n-2} \sum \limits_{k=2}^{n-1}  y_k, \quad \text{where } \{y_1, \dots, y_n\} = \{x_1, \dots, x_n\},\ y_1 \leq y_2 \leq \dots \leq y_n.

:math:`\mathbf{A_{oln}^{(p)}}` **- Olimpic aggregation**

We can specify how many greatest and smallest records remove

.. math::  A_{oln}^{(p)}(x_1,\dots,x_n) = \frac{1}{n-2p} \sum \limits_{k=p}^{n-p}  y_k, \quad \text{where } \{y_1, \dots, y_n\} = \{x_1, \dots, x_n\},\ y_1 \leq y_2 \leq \dots \leq y_n.

--------------

:math:`\mathbf{A_{ln}}` **- Logaritmic aggregation**

https://www.researchgate.net/publication/228886844_Logarithmic_mean_for_several_arguments

.. math::


   A_{ln}\left(x_1, x_2, \ldots, x_n\right)=(n-1) ! \sum_{i=1}^n \frac{x_i}{\prod_{\substack{j=1 \\ j \neq i}}^n \log \left(x_i / x_j\right)}

:math:`\mathbf{A^{(p)}_{A_1,A_2}}` - Convex combinations of aggregation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can construct covex combination of aggregation functions as
:math:`A^{(p)}_{A_1,A_2}=pA_1+(1-p)A_2`, where :math:`A_1, A_2` are
aggregation functions and :math:`p \in [0,1]`.

References
----------

1. Beliakov, G., Bustince, H., and Calvo, T.: A practical Guide to
   Averaging Functions. Berlin: Springer Vol. 329, 2016.
