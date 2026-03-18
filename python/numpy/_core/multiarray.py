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


def _vec_string(a, dtype_arg, method_name, args=()):
    """Apply a string method element-wise to an array.

    Parameters
    ----------
    a : array-like of strings
    dtype_arg : dtype – must be a string/bytes dtype (S or U)
    method_name : str – name of a str/bytes method (e.g. 'strip', 'upper')
    args : tuple – extra arguments forwarded to the method
    """
    # Validate dtype_arg
    dt = numpy.dtype(dtype_arg)
    if dt.kind not in ('S', 'U'):
        raise TypeError("string operation on non-string array")
    # Validate method_name is a string
    if not isinstance(method_name, str):
        raise TypeError("method_name must be a string")
    # Validate args is a tuple
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    # Validate method exists on str
    if not hasattr(str, method_name):
        raise AttributeError(f"'{method_name}' is not a string method")
    # Validate a is array-like with string data
    arr = numpy.asarray(a)
    if arr.dtype.kind not in ('S', 'U', 'O') and not isinstance(arr.tolist(), str):
        raise TypeError("string operation on non-string array")
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        method = getattr(str(v), method_name)
        if isinstance(args, tuple) and len(args) > 0:
            # Broadcast args if needed: each arg may be an array
            cur_args = []
            for arg in args:
                a_arr = numpy.asarray(arg)
                if a_arr.ndim > 0:
                    a_flat = a_arr.flatten()
                    if a_flat.size != flat.size:
                        raise ValueError("shape mismatch in _vec_string args")
                    cur_args.append(a_flat[i])
                else:
                    cur_args.append(a_arr.tolist() if hasattr(a_arr, 'tolist') else arg)
            result.append(method(*cur_args))
        else:
            result.append(method())
    out = numpy.array(result, dtype=dtype_arg)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
