"""numpy.lib._arraypad_impl - array padding helpers."""
import numpy as np


def _as_pairs(x, ndim, as_index=False):
    """
    Broadcast `x` to an array with shape ``(ndim, 2)``.

    Parameters
    ----------
    x : scalar, tuple, list, or array-like
        The value(s) to broadcast.
    ndim : int
        Number of pairs to create.
    as_index : bool, optional
        If True, values are rounded up to nearest int and negative values raise
        ValueError.

    Returns
    -------
    result : ndarray of shape (ndim, 2)
    """
    if x is None:
        # Return tuple of (None, None) pairs
        return tuple((None, None) for _ in range(ndim))

    # Convert to array for broadcasting
    x = np.array(x)

    if as_index:
        # Round up (ceil) and convert to int
        import math
        def _ceil(v):
            iv = int(math.ceil(float(v)))
            if iv < 0:
                raise ValueError("index can't contain negative values")
            return iv

    # Handle various shapes by broadcasting to (ndim, 2)
    if x.ndim == 0:
        # scalar -> broadcast to (ndim, 2)
        if as_index:
            v = _ceil(x)
            return np.array([[v, v]] * ndim, dtype=np.intp)
        v = x.flatten()[0] if x.size > 0 else 0
        return np.array([[v, v]] * ndim)
    elif x.ndim == 1:
        if x.size == 1:
            # [v] -> (ndim, 2) with v repeated
            if as_index:
                v = _ceil(x[0])
                return np.array([[v, v]] * ndim, dtype=np.intp)
            v = x[0]
            return np.array([[v, v]] * ndim)
        elif x.size == 2:
            # [a, b] -> (ndim, 2) with [a, b] repeated
            if as_index:
                a, b = _ceil(x[0]), _ceil(x[1])
                return np.array([[a, b]] * ndim, dtype=np.intp)
            return np.array([[x[0], x[1]]] * ndim)
        else:
            raise ValueError(
                "Unable to create correctly shaped pairs from %s" % repr(x))
    elif x.ndim == 2:
        if x.shape == (1, 2):
            # [[a, b]] -> broadcast to (ndim, 2)
            if as_index:
                a, b = _ceil(x[0][0]), _ceil(x[0][1])
                return np.array([[a, b]] * ndim, dtype=np.intp)
            return np.array([[x[0][0], x[0][1]]] * ndim)
        elif x.shape == (1, 1):
            # [[v]] -> broadcast to (ndim, 2)
            if as_index:
                v = _ceil(x[0][0])
                return np.array([[v, v]] * ndim, dtype=np.intp)
            v = x[0][0]
            return np.array([[v, v]] * ndim)
        elif x.shape[0] == ndim and x.shape[1] == 2:
            # Already correct shape
            if as_index:
                rows = []
                for i in range(ndim):
                    rows.append([_ceil(x[i][0]), _ceil(x[i][1])])
                return np.array(rows, dtype=np.intp)
            return x
        elif x.shape[1] == 1 and x.shape[0] == ndim:
            # (ndim, 1) -> broadcast to (ndim, 2)
            if as_index:
                rows = []
                for i in range(ndim):
                    v = _ceil(x[i][0])
                    rows.append([v, v])
                return np.array(rows, dtype=np.intp)
            rows = []
            for i in range(ndim):
                v = x[i][0]
                rows.append([v, v])
            return np.array(rows)
        else:
            raise ValueError(
                "Unable to create correctly shaped pairs from %s" % repr(x))
    else:
        raise ValueError(
            "Unable to create correctly shaped pairs from %s" % repr(x))


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._arraypad_impl' has no attribute {name!r}")
