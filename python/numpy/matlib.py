"""numpy.matlib — matrix-oriented creation and utility functions.

Provides matrix-specific creation functions and repmat.
Everything else is re-exported from numpy.
"""
import sys as _sys
import numpy as _np

# Re-export all of numpy's namespace (matlib is a superset)
from numpy import *  # noqa: F401, F403


def repmat(a, m, n):
    """Repeat a 0-D to 2-D array or matrix M x N times.

    Parameters
    ----------
    a : array_like
        The input array.
    m, n : int
        The number of times ``a`` is repeated along the first and second axes.

    Returns
    -------
    out : ndarray
        The result of repeating ``a`` tiled ``m x n`` times.
    """
    a = _np.asarray(a)
    if a.ndim == 0:
        a = a.reshape((1, 1))
    elif a.ndim == 1:
        a = a.reshape((1, a.shape[0]))
    elif a.ndim > 2:
        raise ValueError("input must be 0, 1, or 2-D")
    return _np.tile(a, (int(m), int(n)))


def empty(shape, dtype=float, order='C'):
    """Return an empty matrix of the given shape and type."""
    return _np.empty(shape, dtype=dtype, order=order)


def zeros(shape, dtype=float, order='C'):
    """Return a matrix of given shape and type, filled with zeros."""
    return _np.zeros(shape, dtype=dtype, order=order)


def ones(shape, dtype=None, order='C'):
    """Matrix of ones."""
    return _np.ones(shape, dtype=dtype or float, order=order)


def eye(n, M=None, k=0, dtype=float, order='C'):
    """Return a matrix with ones on the diagonal and zeros elsewhere."""
    return _np.eye(n, M=M, k=k, dtype=dtype, order=order)


def identity(n, dtype=None):
    """Return the square identity matrix of given size."""
    return _np.identity(n, dtype=dtype)


def rand(*args):
    """Return a matrix of random values with given shape."""
    return _np.random.rand(*args)


def randn(*args):
    """Return a matrix of random values from a standard normal distribution."""
    return _np.random.randn(*args)


# Register as numpy.matlib
_sys.modules['numpy.matlib'] = _sys.modules[__name__]
