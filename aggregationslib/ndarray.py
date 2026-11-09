"""Apply existing vector aggregations along an axis of a NumPy array.

Constructor parameters are forwarded to the vector aggregation. ``axis``
and ``keepdims`` configure the reduction, for example ``A_pw(2, axis=1)``.
``axis=None`` flattens the input in C order. An integer axis reduces each
vector independently. This module reuses the scalar algorithms; it is not
a vectorized implementation of their formulas.
"""

from numbers import Integral

import numpy as np

from . import aggregations as _vector


__all__ = [
    "A_mn", "A_mx", "A_pr", "A_ch", "A_sg", "A_ar", "A_qd", "A_gm",
    "A_hm", "A_pw", "A_ex", "A_ex2", "A_ex3", "A_lm", "A_amn",
    "A_amx", "A_md", "A_ol", "A_oln", "A_lo", "Combine2Aggregations",
]


class _ArrayAggregation:
    """Shared axis handling; subclasses select an existing vector object."""

    def __init__(self, *args, axis=None, keepdims=False, **kwargs):
        if axis is not None and (isinstance(axis, (bool, np.bool_))
                                 or not isinstance(axis, Integral)):
            raise TypeError("axis must be an integer or None")
        if not isinstance(keepdims, (bool, np.bool_)):
            raise TypeError("keepdims must be a boolean")
        self.axis = int(axis) if axis is not None else None
        self.keepdims = bool(keepdims)
        self._aggregation = self._vector_type(*args, **kwargs)

    def __call__(self, array):
        values = np.asarray(array)
        if self.axis is None:
            if values.size == 0:
                raise ValueError("cannot aggregate an empty vector")
            result = np.asarray(self._aggregation(values.reshape(-1)))
            if result.ndim != 0:
                raise ValueError("a vector aggregation must return a scalar")
            return result.reshape((1,) * values.ndim) if self.keepdims else result[()]

        axis = self.axis
        if not -values.ndim <= axis < values.ndim:
            raise ValueError(f"axis {axis} is out of bounds for an array of dimension {values.ndim}")
        axis %= values.ndim
        if values.shape[axis] == 0:
            raise ValueError("cannot aggregate an empty vector")
        vectors = np.moveaxis(values, axis, -1)
        shape = vectors.shape[:-1]
        # ndindex also visits the single vector of a one-dimensional input.
        results = [self._aggregation(vectors[index]) for index in np.ndindex(shape)]
        result = np.asarray(results)
        if result.ndim != 1:
            raise ValueError("a vector aggregation must return a scalar")
        result = result.reshape(shape)
        if self.keepdims:
            return np.expand_dims(result, axis)
        return result[()] if result.ndim == 0 else result

    @property
    def name(self):
        return self._aggregation.name

    def toLatex(self, params):
        return self._aggregation.toLatex(params)

    def __repr__(self):
        original = repr(self._aggregation)
        separator = "" if original.endswith("()") else ", "
        return f"{original[:-1]}{separator}axis={self.axis!r}, keepdims={self.keepdims!r})"

    def __str__(self):
        return repr(self)


class A_mn(_ArrayAggregation):
    """Minimum along the selected axis."""
    _vector_type = _vector.A_mn


class A_mx(_ArrayAggregation):
    """Maximum along the selected axis."""
    _vector_type = _vector.A_mx


class A_pr(_ArrayAggregation):
    """Product along the selected axis."""
    _vector_type = _vector.A_pr


class A_ar(_ArrayAggregation):
    """Arithmetic mean along the selected axis."""
    _vector_type = _vector.A_ar


class A_qd(_ArrayAggregation):
    """Quadratic mean along the selected axis."""
    _vector_type = _vector.A_qd


class A_gm(_ArrayAggregation):
    """Geometric mean along the selected axis."""
    _vector_type = _vector.A_gm


class A_hm(_ArrayAggregation):
    """Harmonic mean along the selected axis."""
    _vector_type = _vector.A_hm


class A_pw(_ArrayAggregation):
    """Power mean; pass the parameter r to the constructor."""
    _vector_type = _vector.A_pw


class A_ex(_ArrayAggregation):
    """Exponential mean; pass the parameter r to the constructor."""
    _vector_type = _vector.A_ex


class A_ex2(_ArrayAggregation):
    """Second exponential mean; accepts parameters p and q."""
    _vector_type = _vector.A_ex2


class A_ex3(_ArrayAggregation):
    """Third exponential mean; accepts parameters p and q."""
    _vector_type = _vector.A_ex3


class A_lm(_ArrayAggregation):
    """Lehmer mean; accepts the parameter r."""
    _vector_type = _vector.A_lm


class A_amn(_ArrayAggregation):
    """Arithmetic-minimum mean; accepts the parameter p."""
    _vector_type = _vector.A_amn


class A_amx(_ArrayAggregation):
    """Arithmetic-maximum mean; accepts the parameter p."""
    _vector_type = _vector.A_amx


class A_md(_ArrayAggregation):
    """Median along the selected axis."""
    _vector_type = _vector.A_md


class A_ol(_ArrayAggregation):
    """Olympic mean along the selected axis."""
    _vector_type = _vector.A_ol


class A_oln(_ArrayAggregation):
    """Generalized Olympic mean; accepts the trimming parameter p."""
    _vector_type = _vector.A_oln


class A_lo(_ArrayAggregation):
    """Generalized logarithmic mean along the selected axis."""
    _vector_type = _vector.A_lo


class A_ch(_ArrayAggregation):
    """Discrete Choquet integral; accepts measure and scalar quasi_overlap callbacks.

    The measure size must equal the reduced axis length, or the total input
    size for axis=None. The same measure is used for every vector.
    """
    _vector_type = _vector.A_ch


class A_sg(_ArrayAggregation):
    """Discrete Sugeno integral; accepts a measure matching the reduced vector size."""
    _vector_type = _vector.A_sg


class Combine2Aggregations(_ArrayAggregation):
    """Combine vector callables; ndarray objects contribute their vector rule.

    Only this object's axis and keepdims determine the array reduction.
    """
    _vector_type = _vector.Combine2Aggregations

    def __init__(self, aggregation_function1, aggregation_function2, p=0.5,
                 *, axis=None, keepdims=False):
        first = (aggregation_function1._aggregation
                 if isinstance(aggregation_function1, _ArrayAggregation)
                 else aggregation_function1)
        second = (aggregation_function2._aggregation
                  if isinstance(aggregation_function2, _ArrayAggregation)
                  else aggregation_function2)
        super().__init__(first, second, p, axis=axis, keepdims=keepdims)
