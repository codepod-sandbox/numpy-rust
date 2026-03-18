"""numpy._core.multiarray - array creation and manipulation re-exports."""
import numpy

# Flags
_flagdict = {
    'C_CONTIGUOUS': 0x0001,
    'F_CONTIGUOUS': 0x0002,
    'OWNDATA': 0x0004,
    'WRITEABLE': 0x0100,
    'ALIGNED': 0x0200,
    'WRITEBACKIFCOPY': 0x2000,
}

ALLOW_THREADS = 1
CLIP = 0
WRAP = 1
RAISE = 2


def _get_ndarray_c_version():
    return 0


def _reconstruct(subtype, shape, dtype):
    """Reconstruct an array (used by serialization)."""
    return numpy.empty(shape, dtype=dtype)


# Re-export core functions
array = numpy.array
empty = numpy.empty
zeros = numpy.zeros
ones = numpy.ones
arange = numpy.arange
dot = numpy.dot
concatenate = numpy.concatenate
where = numpy.where
asarray = numpy.asarray
ascontiguousarray = getattr(numpy, 'ascontiguousarray', numpy.asarray)
asfortranarray = getattr(numpy, 'asfortranarray', numpy.asarray)
result_type = getattr(numpy, 'result_type', None)
can_cast = getattr(numpy, 'can_cast', None)
promote_types = getattr(numpy, 'promote_types', None)
normalize_axis_index = getattr(numpy, 'normalize_axis_index', None)

# ndarray type
ndarray = numpy.ndarray
dtype = numpy.dtype


def _vec_string(a, dtype, func, args=()):
    """Apply a string function element-wise to an array."""
    arr = numpy.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(func(str(v), *args))
    out = numpy.array(result, dtype=dtype)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
