"""numpy.lib.stride_tricks."""
import numpy as np


def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    """Create a view into the array with the given shape and strides.

    Since RustPython doesn't support true stride manipulation, this returns
    a copy with the equivalent values computed by simulating stride access.
    """
    x = np.asarray(x)
    if shape is None and strides is None:
        return x
    if shape is None:
        shape = x.shape
    if strides is None:
        # No strides specified, just reshape/broadcast
        return np.broadcast_to(x, shape)
    # Simulate stride-based access on the underlying flat data
    flat_data = x.flatten().tolist()
    itemsize = x.itemsize
    # Compute result by walking through the output shape using strides
    import itertools
    result = []
    for pos in itertools.product(*(range(s) for s in shape)):
        # Compute byte offset from strides
        byte_offset = sum(p * st for p, st in zip(pos, strides))
        idx = byte_offset // itemsize
        if 0 <= idx < len(flat_data):
            result.append(flat_data[idx])
        else:
            result.append(0)
    arr = np.array(result)
    if arr.size > 0:
        arr = arr.reshape(shape)
    else:
        arr = np.zeros(shape, dtype=x.dtype)
    return arr


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
