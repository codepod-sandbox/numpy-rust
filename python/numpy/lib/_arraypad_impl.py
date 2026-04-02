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
        If True, values are rounded to nearest int and negative values raise
        ValueError.

    Returns
    -------
    result : ndarray of shape (ndim, 2)
    """
    if x is None:
        # Return tuple of (None, None) pairs
        return tuple((None, None) for _ in range(ndim))

    # Convert to array for broadcasting, with object dtype fallback for mixed types
    try:
        x = np.array(x)
    except (TypeError, ValueError):
        try:
            x = np.array(x, dtype=object)
        except Exception:
            x = np.array(list(x) if hasattr(x, '__iter__') else [x], dtype=object)

    if x.ndim > 2:
        raise ValueError(
            "'x' has more dimensions than allowed"
        )

    is_object = (x.dtype == object or str(x.dtype) == 'object')

    if as_index:
        def _convert(v):
            # Round to nearest int (round half up)
            import math as _m
            fv = float(v)
            iv = int(_m.floor(fv + 0.5))
            if iv < 0:
                raise ValueError("index can't contain negative values")
            return iv

    def _make_pairs(pair_vals, n):
        """Make n copies of a 2-element pair."""
        return np.array([[pair_vals[0], pair_vals[1]]] * n)

    def _make_single(v, n):
        """Make n copies of [v, v]."""
        return np.array([[v, v]] * n)

    if x.ndim == 0:
        # scalar
        if as_index:
            v = _convert(x.flat[0])
            return np.array([[v, v]] * ndim, dtype=np.intp)
        return _make_single(x.flat[0], ndim)

    elif x.ndim == 1:
        if x.size == 1:
            if as_index:
                v = _convert(x[0])
                return np.array([[v, v]] * ndim, dtype=np.intp)
            return _make_single(x[0], ndim)
        elif x.size == 2:
            if as_index:
                a, b = _convert(x[0]), _convert(x[1])
                return np.array([[a, b]] * ndim, dtype=np.intp)
            return _make_pairs([x[0], x[1]], ndim)
        else:
            raise ValueError(
                "could not be broadcast to shape (%d, 2)" % ndim
            )

    else:  # x.ndim == 2
        if x.shape == (1, 2):
            if as_index:
                a, b = _convert(x[0][0]), _convert(x[0][1])
                return np.array([[a, b]] * ndim, dtype=np.intp)
            return _make_pairs([x[0][0], x[0][1]], ndim)
        elif x.shape == (1, 1):
            if as_index:
                v = _convert(x[0][0])
                return np.array([[v, v]] * ndim, dtype=np.intp)
            return _make_single(x[0][0], ndim)
        elif x.shape[0] == ndim and x.shape[1] == 2:
            # Already correct shape
            if as_index:
                rows = []
                for i in range(ndim):
                    rows.append([_convert(x[i][0]), _convert(x[i][1])])
                return np.array(rows, dtype=np.intp)
            return x
        elif x.shape[1] == 1 and x.shape[0] == ndim:
            # (ndim, 1) -> broadcast to (ndim, 2)
            if as_index:
                rows = []
                for i in range(ndim):
                    v = _convert(x[i][0])
                    rows.append([v, v])
                return np.array(rows, dtype=np.intp)
            rows = []
            for i in range(ndim):
                # Use _data directly for _ObjectArray to get the scalar at row i
                if hasattr(x, '_data'):
                    row_val = x._data[i]
                    # row_val may be a list (e.g. ['a']) if created without dtype=object
                    v = row_val[0] if isinstance(row_val, (list, tuple)) else row_val
                else:
                    v = x[i][0]
                rows.append([v, v])
            return np.array(rows)
        else:
            raise ValueError(
                "could not be broadcast to shape (%d, 2)" % ndim
            )


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._arraypad_impl' has no attribute {name!r}")
