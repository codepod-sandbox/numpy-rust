"""Array utility functions for numpy-rust."""
from _numpy_native import ndarray
from numpy.exceptions import AxisError
from ._helpers import _copy_into, _flat_arraylike_data
from ._creation import array, asarray, arange, where

__all__ = [
    'frompyfunc',
    'take_along_axis', 'put_along_axis',
    'byte_bounds',
    'memmap',
    'matrix',
    'add_newdoc', 'deprecate', 'get_include',
]


def frompyfunc(func, nin, nout):
    """Takes an arbitrary Python function and returns a NumPy ufunc-like object."""
    from ._manipulation import vectorize
    return vectorize(func)


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

def _normalize_along_axis_inputs(arr, indices, axis, *, allow_axis_none):
    arr = asarray(arr)
    indices = asarray(indices)
    dtype_name = str(indices.dtype)
    if dtype_name == 'bool':
        raise IndexError("`indices` must be an integer array")
    if not ('int' in dtype_name or 'uint' in dtype_name):
        if 'float' not in dtype_name:
            raise IndexError("`indices` must be an integer array")
        casted = indices.astype('int64')
        original = _flat_arraylike_data(indices)
        cast_flat = _flat_arraylike_data(casted)
        for before, after in zip(original, cast_flat):
            if float(before) != float(after):
                raise IndexError("`indices` must be an integer array")
        indices = casted
    if axis is None:
        if not allow_axis_none:
            raise ValueError("`axis` must be specified")
        if indices.ndim != 1:
            raise ValueError("when axis=None, `indices` must have a single dimension")
        return arr.reshape((arr.size,)), indices.reshape((indices.size,)), 0
    if axis < -arr.ndim or axis >= arr.ndim:
        raise AxisError(axis, arr.ndim)
    if indices.ndim != arr.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    return arr, indices, axis % arr.ndim


def _linearized_take_setup(arr, indices, axis):
    from ._manipulation import moveaxis
    from ._shape import broadcast_shapes, broadcast_to

    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    arr_lead_shape = arr_m.shape[:-1]
    ind_lead_shape = ind_m.shape[:-1]
    out_lead_shape = broadcast_shapes(arr_lead_shape, ind_lead_shape)
    axis_len = arr_m.shape[-1]
    out_axis_len = ind_m.shape[-1]

    arr_row_count = 1
    for dim in arr_lead_shape:
        arr_row_count *= dim
    out_row_count = 1
    for dim in out_lead_shape:
        out_row_count *= dim

    row_ids = arange(arr_row_count, dtype='int64').reshape(arr_lead_shape if arr_lead_shape else ())
    row_ids_bc = broadcast_to(row_ids, out_lead_shape if out_lead_shape else ()).reshape((out_row_count, 1))
    ind_rows = broadcast_to(ind_m, out_lead_shape + (out_axis_len,)).reshape((out_row_count, out_axis_len)).astype('int64')
    norm_rows = where(ind_rows < 0, ind_rows + axis_len, ind_rows)

    for idx in _flat_arraylike_data(norm_rows):
        if idx < 0 or idx >= axis_len:
            raise IndexError("index out of bounds")

    flat_offsets = row_ids_bc * axis_len
    linear = (flat_offsets + norm_rows).reshape((out_row_count * out_axis_len,))
    arr_rows = arr_m.reshape((arr_row_count, axis_len))
    out_shape = out_lead_shape + (out_axis_len,)
    return arr_m, arr_rows, linear, out_shape

def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1-d index and data slices along the given axis."""
    from ._manipulation import moveaxis

    arr, indices, axis = _normalize_along_axis_inputs(arr, indices, axis, allow_axis_none=True)
    arr_m, arr_rows, linear, out_shape = _linearized_take_setup(arr, indices, axis)
    result = arr_rows.reshape((arr_rows.shape[0] * arr_rows.shape[1],))[linear].reshape(out_shape)
    return moveaxis(result, -1, axis) if arr.ndim > 1 else result


def put_along_axis(arr, indices, values, axis):
    """Put values into the destination array by matching 1-d index and data slices along the given axis."""
    from ._manipulation import moveaxis
    from ._shape import broadcast_to

    arr, indices, axis = _normalize_along_axis_inputs(arr, indices, axis, allow_axis_none=True)
    arr_m, arr_rows, linear, out_shape = _linearized_take_setup(arr, indices, axis)
    values_arr = asarray(values)
    values_src = moveaxis(values_arr, axis, -1) if values_arr.ndim == arr.ndim else values_arr
    values_bc = broadcast_to(values_src, out_shape).reshape((linear.size,))
    flat = _flat_arraylike_data(arr_rows.reshape((arr_rows.shape[0] * arr_rows.shape[1],)))
    for idx, value in zip(_flat_arraylike_data(linear), _flat_arraylike_data(values_bc)):
        flat[int(idx)] = value
    updated = array(flat, dtype=str(arr.dtype)).reshape(arr_m.shape)
    _copy_into(arr, moveaxis(updated, -1, axis))
    return arr


# ---------------------------------------------------------------------------
# byte_bounds
# ---------------------------------------------------------------------------

def byte_bounds(a):
    """Return low and high byte pointers (stub returns (0, nbytes))."""
    arr = asarray(a)
    return (0, arr.nbytes)


# ---------------------------------------------------------------------------
# memmap stub
# ---------------------------------------------------------------------------

class memmap:
    """Memory-mapped file stub (not supported in sandboxed environment)."""
    def __new__(cls, filename, dtype=None, mode='r+', offset=0, shape=None, order='C'):
        raise NotImplementedError("memmap not supported in sandboxed environment")


# ---------------------------------------------------------------------------
# matrix class
# ---------------------------------------------------------------------------

class matrix:
    """Simplified matrix class (deprecated in NumPy, but still used)."""
    def __init__(self, data, dtype=None, copy=True):
        from ._manipulation import atleast_2d
        if isinstance(data, str):
            rows = data.split(";")
            parsed = []
            for row in rows:
                parsed.append([float(x) for x in row.strip().split()])
            self.A = array(parsed)
        else:
            self.A = atleast_2d(asarray(data))
        if dtype is not None:
            self.A = self.A.astype(str(dtype))

    @property
    def T(self):
        return matrix(self.A.T)

    @property
    def I(self):
        from _numpy_native import linalg as _linalg
        return matrix(_linalg.inv(self.A))

    @property
    def shape(self):
        return self.A.shape

    @property
    def ndim(self):
        return self.A.ndim

    def __mul__(self, other):
        from _numpy_native import dot
        if isinstance(other, matrix):
            return matrix(dot(self.A, other.A))
        return matrix(self.A * asarray(other))

    def __add__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A + other.A)
        return matrix(self.A + asarray(other))

    def __sub__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A - other.A)
        return matrix(self.A - asarray(other))

    def __getitem__(self, key):
        return self.A[key]

    def tolist(self):
        return self.A.tolist()

    def __repr__(self):
        return "matrix({})".format(self.A.tolist())


# ---------------------------------------------------------------------------
# Misc stubs
# ---------------------------------------------------------------------------

def add_newdoc(place, obj, doc):
    """Add documentation (no-op in our implementation)."""
    pass

def deprecate(func=None, oldname=None, newname=None, message=None):
    """Deprecation decorator (no-op)."""
    if func is not None:
        return func
    def decorator(f):
        return f
    return decorator

def get_include():
    """Return the numpy include directory."""
    import os
    return os.path.dirname(os.path.abspath(__file__))
