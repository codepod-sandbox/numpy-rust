"""numpy._core._methods - method implementations for ndarray."""
import numpy


def _mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return numpy.mean(a, axis=axis)


def _sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return numpy.sum(a, axis=axis)


def _prod(a, axis=None, dtype=None, out=None, keepdims=False):
    return numpy.prod(a, axis=axis)


def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return numpy.var(a, axis=axis, ddof=ddof)


def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return numpy.std(a, axis=axis, ddof=ddof)


def _any(a, axis=None, dtype=None, out=None, keepdims=False):
    return numpy.any(a, axis=axis)


def _all(a, axis=None, dtype=None, out=None, keepdims=False):
    return numpy.all(a, axis=axis)


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
