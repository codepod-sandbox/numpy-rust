"""Array utility functions for numpy-rust."""
from _numpy_native import ndarray
from ._helpers import _flat_arraylike_data
from ._creation import array, asarray

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

def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1-d index and data slices along the given axis."""
    from ._manipulation import moveaxis
    arr = asarray(arr)
    indices = asarray(indices)
    if arr.ndim == 1:
        result = []
        for i in range(indices.size):
            result.append(arr[int(indices[i])])
        return array(result)
    if arr.ndim == 2:
        if axis == 0:
            rows = []
            for j in range(arr.shape[1]):
                col = []
                for i in range(indices.shape[0]):
                    col.append(arr[int(indices[i][j])][j])
                rows.append(col)
            result = []
            for i in range(indices.shape[0]):
                row = [rows[j][i] for j in range(arr.shape[1])]
                result.append(row)
            return array(result)
        else:
            rows = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(indices.shape[1]):
                    row.append(arr[i][int(indices[i][j])])
                rows.append(row)
            return array(rows)
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    out_shape = ind_m.shape
    n_axis = arr_m.shape[-1]
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    arr_flat = arr_m.reshape((lead, n_axis))
    ind_flat = ind_m.reshape((lead, ind_m.shape[-1]))
    arr_list = [_flat_arraylike_data(arr_flat[i]) for i in range(lead)]
    ind_list = [_flat_arraylike_data(ind_flat[i]) for i in range(lead)]
    result = []
    for i in range(lead):
        row = arr_list[i]
        idxs = ind_list[i]
        result.append([row[int(j)] for j in idxs])
    result_arr = array(result).reshape(out_shape)
    return moveaxis(result_arr, -1, axis)


def put_along_axis(arr, indices, values, axis):
    """Put values into the destination array by matching 1-d index and data slices along the given axis."""
    from ._manipulation import moveaxis
    arr = asarray(arr)
    indices = asarray(indices)
    values = asarray(values)
    if arr.ndim == 1:
        result = [arr[i] for i in range(arr.size)]
        vals_flat = values.flatten()
        for i in range(indices.size):
            result[int(indices[i])] = vals_flat[i % vals_flat.size]
        return array(result)
    if arr.ndim == 2 and axis == 1:
        rows = []
        for i in range(arr.shape[0]):
            row = [arr[i][j] for j in range(arr.shape[1])]
            for j in range(indices.shape[1]):
                idx = int(indices[i][j])
                row[idx] = values[i][j] if values.ndim == 2 else values[j]
            rows.append(row)
        return array(rows)
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    val_m = moveaxis(values, axis, -1)
    out_shape = arr_m.shape
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    n_axis = arr_m.shape[-1]
    arr_flat = [_flat_arraylike_data(arr_m.reshape((lead, n_axis))[i]) for i in range(lead)]
    ind_flat = [_flat_arraylike_data(ind_m.reshape((lead, ind_m.shape[-1]))[i]) for i in range(lead)]
    val_flat = [_flat_arraylike_data(val_m.reshape((lead, val_m.shape[-1]))[i]) for i in range(lead)]
    for i in range(lead):
        for j in range(len(ind_flat[i])):
            arr_flat[i][int(ind_flat[i][j])] = val_flat[i][j]
    result = array(arr_flat).reshape(out_shape)
    return moveaxis(result, -1, axis)


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
