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


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
