"""numpy.lib._stride_tricks_impl - stride tricks implementation."""
from numpy.lib.stride_tricks import as_strided, broadcast_shapes, _broadcast_shape


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """Create a sliding window view into the array.

    Simplified implementation that returns copies (not true views) since
    RustPython ndarray doesn't support arbitrary strides.
    """
    import numpy as np
    x = np.asarray(x)

    if isinstance(window_shape, int):
        window_shape = (window_shape,)
    else:
        window_shape = tuple(window_shape)

    if axis is not None:
        # Normalize axis to tuple
        if isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)

        if len(window_shape) != len(axis):
            raise ValueError(
                "Must provide matching length window_shape and axis; "
                "got {} window_shape entries and {} axes".format(
                    len(window_shape), len(axis)))

        # Normalize negative axes
        axis = tuple(a if a >= 0 else x.ndim + a for a in axis)

        # Apply sliding windows sequentially along each axis
        result = x
        # Track how many new dims have been appended (each step adds one at end)
        for w, ax in zip(window_shape, axis):
            n = result.shape[ax]
            if w > n:
                raise ValueError(
                    "window shape cannot be larger than input array shape")
            n_windows = n - w + 1
            slices = []
            for i in range(n_windows):
                sl = [slice(None)] * result.ndim
                sl[ax] = slice(i, i + w)
                slices.append(result[tuple(sl)])
            result = np.stack(slices, axis=ax)
        return result
    else:
        if len(window_shape) != x.ndim:
            raise ValueError(
                "Since axis is `None`, must provide window_shape for all "
                "dimensions of `x`; got {} window_shape entries for array "
                "with {} dimensions.".format(len(window_shape), x.ndim))
        # Multi-dimensional sliding window
        for s, w in zip(x.shape, window_shape):
            if w < 0:
                raise ValueError(
                    "window_shape cannot contain negative values")
            if w > s:
                raise ValueError(
                    "window shape cannot be larger than input array shape")

        out_shape = tuple(s - w + 1 for s, w in zip(x.shape, window_shape))

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


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._stride_tricks_impl' has no attribute {name!r}")
