"""numpy.lib._stride_tricks_impl - stride tricks implementation."""
from numpy.lib.stride_tricks import as_strided, broadcast_shapes, _broadcast_shape


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """Create a sliding window view into the array.

    Simplified implementation that returns copies (not true views) since
    RustPython ndarray doesn't support arbitrary strides.
    """
    import numpy as np
    x = np.asarray(x)

    if axis is None:
        if isinstance(window_shape, int):
            window_shape = (window_shape,)
        if len(window_shape) != x.ndim:
            raise ValueError(
                "window_shape length {} must match x.ndim {}".format(
                    len(window_shape), x.ndim))
        # Multi-dimensional sliding window
        out_shape = tuple(s - w + 1 for s, w in zip(x.shape, window_shape))
        for s, w in zip(x.shape, window_shape):
            if w < 0:
                raise ValueError(
                    "window_shape cannot contain negative values")
            if w > s:
                raise ValueError(
                    "window size {} too large for axis of size {}".format(w, s))
        if x.ndim == 1:
            w = window_shape[0]
            n = x.shape[0] - w + 1
            result = []
            for i in range(n):
                result.append(x[i:i + w].tolist())
            return np.array(result)
        # General N-d case: iterate over all window positions
        import itertools
        ranges = [range(s - w + 1) for s, w in zip(x.shape, window_shape)]
        slices_list = []
        for pos in itertools.product(*ranges):
            sl = tuple(slice(p, p + w) for p, w in zip(pos, window_shape))
            slices_list.append(x[sl])
        result = np.array([s.tolist() for s in slices_list])
        # Reshape to (*out_shape, *window_shape)
        return result.reshape(out_shape + tuple(window_shape))
    else:
        # Single axis
        if isinstance(window_shape, (list, tuple)):
            if len(window_shape) != 1:
                raise ValueError("axis specified but window_shape has multiple elements")
            w = window_shape[0]
        else:
            w = int(window_shape)
        if axis < 0:
            axis = x.ndim + axis
        n = x.shape[axis] - w + 1
        if n <= 0:
            raise ValueError("window size {} too large for axis of size {}".format(
                w, x.shape[axis]))
        slices = []
        for i in range(n):
            sl = [slice(None)] * x.ndim
            sl[axis] = slice(i, i + w)
            slices.append(x[tuple(sl)])
        return np.stack(slices, axis=axis)


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._stride_tricks_impl' has no attribute {name!r}")
