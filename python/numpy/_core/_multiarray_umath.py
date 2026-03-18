"""numpy._core._multiarray_umath - stub for C extension module."""
import numpy

ALLOW_THREADS = 1
CLIP = 0
WRAP = 1
RAISE = 2


def _reconstruct(subtype, shape, dtype):
    """Reconstruct an array (used by serialization)."""
    return numpy.empty(shape, dtype=dtype)


def _get_ndarray_c_version():
    return 0


# Re-export core items
ndarray = numpy.ndarray
dtype = numpy.dtype
array = numpy.array
empty = numpy.empty
zeros = numpy.zeros


def _arg(x):
    """Return the angle (argument) of complex array elements.

    Works element-wise on complex arrays, returning real array of angles.
    """
    import math
    if hasattr(x, '_data'):
        # _ObjectArray: process element-wise
        result = []
        for v in x._data:
            if isinstance(v, complex):
                result.append(math.atan2(v.imag, v.real))
            elif isinstance(v, tuple) and len(v) == 2:
                result.append(math.atan2(v[1], v[0]))
            elif isinstance(v, (int, float)):
                result.append(math.atan2(0.0, float(v)))
            else:
                result.append(float('nan'))
        return numpy.array(result)
    return numpy.angle(x)


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
