"""numpy.lib.stride_tricks."""
import numpy as np


def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    if shape is not None:
        return np.broadcast_to(x, shape)
    return x


def broadcast_shapes(*shapes):
    """Broadcast shapes together, returning the resulting shape."""
    if not shapes:
        return ()
    # Normalize: integers are treated as 1-d shapes
    shapes = [s if hasattr(s, '__len__') else (s,) for s in shapes]
    ndim = max(len(s) for s in shapes)
    result = [1] * ndim
    for shape in shapes:
        # Right-align
        offset = ndim - len(shape)
        for i, dim in enumerate(shape):
            j = i + offset
            if result[j] == 1:
                result[j] = dim
            elif dim != 1 and dim != result[j]:
                raise ValueError(
                    f"shape mismatch: objects cannot be broadcast to a single shape. "
                    f"Mismatch at dimension {j}."
                )
    return tuple(result)


def _broadcast_shape(*args):
    return broadcast_shapes(*[np.asarray(a).shape for a in args])


from numpy.lib._stride_tricks_impl import sliding_window_view
