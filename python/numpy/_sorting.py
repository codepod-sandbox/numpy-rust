"""Sorting, searching, and counting: sort, argsort, lexsort, unique, etc."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray, _coerce_native_boxed_operand,
    _builtin_range, _builtin_min, _builtin_max,
)

_builtin_sorted = sorted
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate

__all__ = [
    'sort', '_sort_complex_axis', 'sort_complex', 'argsort',
    'lexsort', 'partition', 'argpartition',
    'unique', 'unique_values', 'unique_counts', 'unique_inverse', 'unique_all',
]


def _sort_complex_axis(a, axis):
    """Sort a complex ndarray along the given axis, lexicographically by (real, imag)."""
    from ._creation import _make_complex_array
    if a.ndim == 1:
        flat = a.tolist()
        def _cx_key(v):
            if isinstance(v, tuple):
                return (v[0], v[1])
            return (v.real if hasattr(v, 'real') else float(v), 0.0)
        flat.sort(key=_cx_key)
        return _make_complex_array(flat, (len(flat),))
    # Multi-dim: iterate over slices along the axis
    n = a.shape[axis]
    slices_in = []
    for i in range(n):
        idx = [slice(None)] * a.ndim
        idx[axis] = i
        slices_in.append(a[tuple(idx)])
    # Build rows and sort each 1D slice orthogonal to axis
    # Actually, we need to sort ALONG the axis for each combination of other indices
    # Simpler: moveaxis to front, iterate over remaining axes
    import numpy as _np
    moved = _np.moveaxis(a, axis, 0)  # shape: (n, ...)
    out_shape = moved.shape
    flat_outer = 1
    for s in out_shape[1:]:
        flat_outer *= s
    result_flat = []
    for i in range(flat_outer):
        # Get the 1D slice for this combination
        idx = list(_np.unravel_index(i, out_shape[1:]))
        vec_idx = tuple([slice(None)] + idx)
        vec = moved[vec_idx].tolist()
        def _cx_key(v):
            if isinstance(v, tuple):
                return (v[0], v[1])
            return (v.real if hasattr(v, 'real') else float(v), 0.0)
        vec.sort(key=_cx_key)
        result_flat.append(vec)
    # Reconstruct
    result_arr = _make_complex_array([v for row in result_flat for v in row], (flat_outer * n,))
    # Reshape and moveaxis back
    result_arr = result_arr.reshape(out_shape)
    return _np.moveaxis(result_arr, 0, axis)


def sort(a, axis=-1, kind=None, order=None):
    from ._helpers import _ObjectArray
    if not isinstance(a, ndarray):
        a = asarray(a)
    if isinstance(a, _ObjectArray):
        import copy
        a_copy = copy.copy(a)
        a_copy.sort(axis, kind=kind, order=order)
        return a_copy
    original_dtype = str(a.dtype)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    # Complex arrays: sort lexicographically by (real, imag)
    if original_dtype in ('complex64', 'complex128'):
        if axis is None:
            flat = a.flatten().tolist()
            # Each element from complex ndarray is a (re, im) tuple
            def _cx_key(v):
                if isinstance(v, tuple):
                    return (v[0], v[1])
                return (v.real if hasattr(v, 'real') else float(v), 0.0)
            flat.sort(key=_cx_key)
            from ._creation import _make_complex_array
            return _make_complex_array(flat, (len(flat),))
        ax = axis if axis is not None else a.ndim - 1
        # Sort along axis
        _sorted = _sort_complex_axis(a, ax)
        return _sorted
    result = a.copy()
    result.sort(axis)
    # Preserve original dtype (Rust sort may convert to float64)
    if original_dtype in ('int32', 'int64') and str(result.dtype) != original_dtype:
        result = result.astype(original_dtype)
    return result


def sort_complex(a):
    """Sort a complex array using the real part first, then the imaginary part."""
    a = asarray(a)
    dt = str(a.dtype)
    # Map real/integer types to the appropriate complex output dtype
    _type_map = {
        'int64': 'complex128', 'int16': 'complex64', 'uint16': 'complex64',
        'int8': 'complex64', 'uint8': 'complex64',
        'float32': 'complex64', 'float64': 'complex128',
        'complex64': 'complex64', 'complex128': 'complex128',
    }
    out_dtype = _type_map.get(dt, 'complex128')
    c = a.astype(out_dtype)
    n = len(c.flatten())
    # Extract as Python complex numbers (elements may be tuples in RustPython)
    vals = []
    for i in _builtin_range(n):
        v = c.flatten()[i]
        if isinstance(v, tuple) and len(v) == 2:
            vals.append(complex(v[0], v[1]))
        else:
            try:
                vals.append(complex(v))
            except (TypeError, ValueError):
                vals.append(complex(float(v), 0.0))
    vals.sort(key=lambda x: (x.real, x.imag))
    return array(vals, dtype=out_dtype)


def argsort(a, axis=-1, kind=None, order=None):
    if isinstance(a, (tuple, list)):
        if len(a) == 0:
            return _native.zeros((0,), 'int64')
        a = _native.array([float(x) for x in a])
    else:
        a = asarray(a)
        if isinstance(a, _ObjectArray):
            vals = [float(x) if isinstance(x, (int, float)) else 0. for x in (a._data or [])]
            inds = sorted(range(len(vals)), key=lambda i: vals[i])
            return _native.array([float(i) for i in inds]) if inds else _native.zeros((0,), 'int64')
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    return a.argsort(axis)


def lexsort(keys):
    """Perform an indirect stable sort using a sequence of keys.
    Last key is used as the primary sort key."""
    keys_list = [asarray(k) for k in keys]
    n = keys_list[0].size
    indices = list(range(n))
    # Flatten all keys
    flat_keys = []
    for k in keys_list:
        fk = k.flatten()
        flat_keys.append([fk[i] for i in range(n)])
    # Sort indices using keys (last key = primary, first key = least significant)
    # Python sort is stable, so we sort from least significant to most significant
    for ki in range(len(flat_keys)):
        vals = flat_keys[ki]
        indices.sort(key=lambda idx, v=vals: v[idx])
    return array(indices)


def partition(a, kth, axis=-1):
    """Return a partitioned copy of an array.
    Creates a copy where the element at kth position is where it would be in a sorted array.
    Elements before kth are <= element at kth, elements after are >=."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        flat = a.flatten()
        n = flat.size
        vals = [flat[i] for i in range(n)]
        vals.sort()
        return array(vals)
    # For multi-dim, sort along axis (full sort, which satisfies partition contract)
    return sort(a, axis=axis)


def argpartition(a, kth, axis=-1):
    """Return indices that would partition the array.
    Same as argsort but only guarantees kth element is correct."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        return argsort(a)
    return argsort(a, axis=axis)


def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True):
    """Return sorted unique elements of an array.

    Parameters
    ----------
    a : array_like
    return_index : bool
        If True, return indices of first occurrences.
    return_inverse : bool
        If True, return indices to reconstruct original from unique.
    return_counts : bool
        If True, return count of each unique value.
    axis : int, optional
        The axis along which to find unique slices. If None, flatten first.

    Returns
    -------
    unique : ndarray
    unique_indices : ndarray (optional)
    unique_inverse : ndarray (optional)
    unique_counts : ndarray (optional)
    """
    from ._helpers import _ObjectArray
    from ._shape import swapaxes, moveaxis
    from ._join import stack
    if not isinstance(a, ndarray) and type(a).__name__ not in ('StructuredArray',):
        a = asarray(a)

    def _get_slice(arr, ax, i):
        if ax == 0:
            return arr[i]
        tmp = swapaxes(arr, 0, ax)
        return tmp[i]

    def _slice_key(sl):
        def _to_sortable(v):
            # Complex scalars from Rust backend are (re, im) tuples
            if isinstance(v, tuple) and len(v) == 2:
                return v  # already (re, im), sortable as tuple
            # Python complex objects
            if isinstance(v, complex):
                return (v.real, v.imag)
            # _NumpyComplexScalar: has .real and .imag
            if hasattr(v, 'real') and hasattr(v, 'imag') and not isinstance(v, (int, float)):
                try:
                    return (float(v.real), float(v.imag))
                except Exception:
                    pass
            return v
        if isinstance(sl, ndarray):
            flat = sl.flatten().tolist()
            return tuple(_to_sortable(v) for v in flat)
        if isinstance(sl, _ObjectArray):
            return tuple(_to_sortable(v) for v in sl._data)
        if type(sl).__name__ == 'StructuredArray' and hasattr(sl, 'tolist'):
            def _deep_flatten(x):
                if isinstance(x, (list, tuple)):
                    result = []
                    for v in x:
                        result.extend(_deep_flatten(v))
                    return result
                return [x]
            return tuple(_deep_flatten(sl.tolist()))
        return (sl,)

    def _is_nan_val(v):
        try:
            return v != v
        except TypeError:
            return False

    # Singleton to replace NaN in dict keys so equal_nan dedup works
    class _NanKey:
        def __eq__(self, other): return type(other) is type(self)
        def __hash__(self): return hash('__NAN_KEY__')
        def __lt__(self, other): return False
        def __le__(self, other): return type(other) is type(self)
        def __gt__(self, other): return True
        def __ge__(self, other): return True

    _NAN_KEY = _NanKey()

    def _normalize_key(key):
        """Replace NaN floats with _NAN_KEY for equal_nan-aware dict lookup."""
        if not equal_nan:
            return key
        result = []
        for v in key:
            try:
                if v != v:
                    result.append(_NAN_KEY)
                    continue
            except TypeError:
                pass
            result.append(v)
        return tuple(result)

    if axis is not None:
        # Validate axis is an integer
        try:
            import operator as _op
            axis = _op.index(axis)
        except TypeError:
            raise TypeError("integer argument expected, got {}".format(type(axis).__name__))
        # Validate axis bounds
        _ndim = a.ndim
        if axis < -_ndim or axis >= _ndim:
            raise AxisError("axis {} is out of bounds for array of dimension {}".format(axis, _ndim))
        if axis < 0:
            axis += _ndim
        # Reject object dtype (cannot be used with axis unique)
        _dtype_str = str(a.dtype) if hasattr(a, 'dtype') else ''
        if _dtype_str == 'object':
            raise TypeError("cannot use axis argument on arrays with object dtype")
        # Reject structured dtype with object fields
        if hasattr(a, 'dtype') and hasattr(a.dtype, 'names') and a.dtype.names:
            for _fn in a.dtype.names:
                if 'object' in str(a.dtype.fields[_fn][0]):
                    raise TypeError("cannot use axis argument on structured arrays with object fields")
        # For axis=0: find unique rows (or slices along axis 0)
        n_slices = a.shape[axis]
        # Extract each slice as a tuple for hashing
        seen = {}
        unique_indices_list = []
        for i in range(n_slices):
            sl = _get_slice(a, axis, i)
            key = _normalize_key(_slice_key(sl))
            if key not in seen:
                seen[key] = i
                unique_indices_list.append(i)
        # Build result array from unique indices
        if axis == 0:
            rows = [a[i] for i in unique_indices_list]
        else:
            tmp = swapaxes(a, 0, axis)
            rows = [tmp[i] for i in unique_indices_list]
        # Sort by the first element of each row for consistent ordering
        # When equal_nan=True, NaN values sort after all regular values
        def _axis_sort_key(p):
            key = _slice_key(p[1])
            if not equal_nan:
                return key
            result = []
            for v in key:
                try:
                    if v != v:  # NaN
                        result.append((1, 0.0))
                    else:
                        result.append((0, v))
                except TypeError:
                    result.append((0, str(v)))
            return result
        pairs = list(zip(unique_indices_list, rows))
        try:
            pairs.sort(key=_axis_sort_key)
        except TypeError:
            pairs.sort(key=lambda p: _slice_key(p[1]))
        unique_indices_list = [p[0] for p in pairs]
        rows = [p[1] for p in pairs]
        # For 1D arrays, rows are scalars — use array() directly
        if len(rows) > 0 and type(rows[0]).__name__ == 'StructuredArray':
            # Stack StructuredArray rows: flatten all scalars then reshape
            from ._creation import _create_structured_array
            from ._core_types import dtype as _dtype_cls
            _sdt = a.dtype if hasattr(a.dtype, 'names') else _dtype_cls(str(a.dtype))
            flat_tuples = []
            row_shape = rows[0].shape  # shape of each row (may be ND)
            for r in rows:
                # Flatten ND row to list of scalar tuples
                def _flatten_sa(sa):
                    if sa.ndim == 0:
                        v = sa.tolist()
                        return [v if isinstance(v, tuple) else (v,)]
                    if sa.ndim == 1:
                        lst = sa.tolist()
                        return [x if isinstance(x, tuple) else (x,) for x in lst]
                    result_inner = []
                    for i in range(sa.shape[0]):
                        result_inner.extend(_flatten_sa(sa[i]))
                    return result_inner
                flat_tuples.extend(_flatten_sa(r))
            total_shape = (len(rows),) + row_shape
            sa_flat = _create_structured_array(flat_tuples, _sdt)
            import functools, operator
            total_size = functools.reduce(operator.mul, total_shape, 1)
            result = sa_flat.reshape(list(total_shape))
            if axis != 0:
                result = moveaxis(result, 0, axis)
        elif len(rows) > 0 and not isinstance(rows[0], (ndarray, _ObjectArray)):
            result = array(rows, dtype=a.dtype)
        elif len(rows) == 0:
            # No unique rows — return empty with correct shape and dtype
            result_shape = list(a.shape)
            result_shape[axis] = 0
            import numpy as _np_mod
            result = _np_mod.empty(result_shape, dtype=a.dtype)
        else:
            result = stack(rows, axis=0)
            if axis != 0:
                result = moveaxis(result, 0, axis)
            # Preserve original dtype (stack may not preserve narrow dtypes like int8)
            if str(result.dtype) != str(a.dtype):
                try:
                    result = result.astype(a.dtype)
                except Exception:
                    pass
        extras = return_index or return_inverse or return_counts
        if not extras:
            return result
        ret = (result,)
        if return_index:
            ret = ret + (array(unique_indices_list, dtype='int64'),)
        if return_inverse:
            # Map each original slice index to its position in unique result
            key_to_pos = {}
            for pos, idx in enumerate(unique_indices_list):
                sl = _get_slice(a, axis, idx)
                key = _normalize_key(_slice_key(sl))
                key_to_pos[key] = pos
            inv = []
            for i in range(n_slices):
                sl = _get_slice(a, axis, i)
                key = _normalize_key(_slice_key(sl))
                inv.append(key_to_pos[key])
            ret = ret + (array(inv, dtype='int64'),)
        if return_counts:
            # Count occurrences of each unique
            count_map = {}
            for i in range(n_slices):
                sl = _get_slice(a, axis, i)
                key = _normalize_key(_slice_key(sl))
                if key not in count_map:
                    count_map[key] = 0
                count_map[key] += 1
            counts_list = []
            for idx in unique_indices_list:
                sl = _get_slice(a, axis, idx)
                key = _normalize_key(_slice_key(sl))
                counts_list.append(count_map[key])
            ret = ret + (array(counts_list, dtype='int64'),)
        return ret
    flat = a.flatten()
    n = flat.shape[0]
    vals = flat.tolist()

    # If dtype is complex, convert (re, im) tuples from tolist() to Python complex
    _dt = str(a.dtype)
    if 'complex' in _dt:
        new_vals = []
        for v in vals:
            if isinstance(v, tuple) and len(v) == 2:
                new_vals.append(complex(v[0], v[1]))
            elif isinstance(v, complex):
                new_vals.append(v)
            else:
                new_vals.append(complex(v))
        vals = new_vals

    # Sort key that handles complex, tuples, and mixed types
    def _sort_key(t):
        v = t[1]
        if isinstance(v, complex):
            return (v.real, v.imag)
        if isinstance(v, tuple):
            return v
        try:
            return (v,)
        except TypeError:
            return (str(v),)

    # Build sorted unique with tracking info
    # When equal_nan=True, NaN values must sort together (at end) so the dedup
    # loop can detect consecutive NaN values and treat them as equal.
    def _flat_sort_key(t):
        v = t[1]
        if equal_nan:
            try:
                if _is_nan_val(v):
                    if isinstance(v, complex):
                        # For complex NaN, sort by real part (using inf if real is NaN),
                        # then by inf for imag (so NaN-imag sorts after non-NaN-imag).
                        # This matches NumPy's sort order: 0+nanj < 1+nanj < nan+0j.
                        import math
                        real_key = float('inf') if math.isnan(v.real) else v.real
                        imag_key = float('inf') if math.isnan(v.imag) else v.imag
                        return (1, real_key, imag_key)
                    # Non-complex NaN (float): group at the end
                    return (1, float('inf'))
            except (TypeError, ValueError):
                pass
        if isinstance(v, complex):
            return (0, v.real, v.imag)
        if isinstance(v, tuple):
            return (0,) + v
        try:
            return (0, v)
        except TypeError:
            return (0, str(v))
    try:
        indexed = _builtin_sorted(enumerate(vals), key=_flat_sort_key)
    except TypeError:
        indexed = _builtin_sorted(enumerate(vals), key=_sort_key)
    unique_vals = []
    first_indices = []
    counts = []
    _in_nan_group = False  # True when deduplicating via NaN equality
    _sentinel = object()
    prev = _sentinel
    for orig_idx, v in indexed:
        if prev is _sentinel:
            _same = False
            _nan_eq = False
        else:
            try:
                _same = bool(v == prev)
                _nan_eq = False
            except (TypeError, ValueError):
                _same = False
                _nan_eq = False
            if not _same and equal_nan:
                try:
                    _nan_eq = _is_nan_val(v) and _is_nan_val(prev)
                    if _nan_eq:
                        _same = True
                except (TypeError, ValueError):
                    pass
        if not _same:
            unique_vals.append(v)
            first_indices.append(orig_idx)
            counts.append(1)
            _in_nan_group = bool(_nan_eq)
            prev = v
        else:
            counts[-1] += 1
            if _nan_eq:
                _in_nan_group = True
            # For NaN groups: keep index from first-in-sorted-order (initial append).
            # For non-NaN groups: keep smallest original index (= first occurrence).
            if not _in_nan_group and orig_idx < first_indices[-1]:
                first_indices[-1] = orig_idx

    # For complex ndarray inputs, build result as regular ndarray (not _ObjectArray)
    if not isinstance(a, _ObjectArray) and 'complex' in str(a.dtype) and unique_vals:
        n = len(unique_vals)
        result_unique = _native.zeros((n,), str(a.dtype))
        for _i, _v in enumerate(unique_vals):
            if isinstance(_v, complex):
                result_unique[_i] = (_v.real, _v.imag)
            else:
                result_unique[_i] = (float(_v), 0.0)
    else:
        result_unique = array(unique_vals, dtype=a.dtype)
    extras = return_index or return_inverse or return_counts
    if not extras:
        return result_unique
    ret = (result_unique,)
    if return_index:
        ret = ret + (array(first_indices, dtype='int64'),)
    if return_inverse:
        # For each element in the original flat array, find its position in unique_vals
        # NaN needs special handling since nan != nan in dict lookups
        val_to_pos = {}
        nan_pos = None
        for i, v in enumerate(unique_vals):
            if _is_nan_val(v):
                nan_pos = i
            else:
                try:
                    val_to_pos[v] = i
                except TypeError:
                    val_to_pos[str(v)] = i
        inverse = []
        for v in vals:
            if _is_nan_val(v) and nan_pos is not None:
                inverse.append(nan_pos)
            else:
                try:
                    inverse.append(val_to_pos[v])
                except (KeyError, TypeError):
                    inverse.append(val_to_pos.get(str(v), 0))
        inv_arr = array(inverse, dtype='int64')
        # NumPy 2.x: reshape inverse to original input shape when axis=None
        if a.ndim > 1:
            try:
                inv_arr = inv_arr.reshape(a.shape)
            except Exception:
                pass
        ret = ret + (inv_arr,)
    if return_counts:
        ret = ret + (array(counts, dtype='int64'),)
    return ret


# ---------------------------------------------------------------------------
# NumPy 2.0 Array API unique functions
# Result types are namedtuple subclasses so isinstance(result, tuple) is True
# ---------------------------------------------------------------------------

from collections import namedtuple as _namedtuple

_UniqueCountsResult = _namedtuple('UniqueCountsResult', ['values', 'counts'])
_UniqueInverseResult = _namedtuple('UniqueInverseResult', ['values', 'inverse_indices'])
_UniqueAllResult = _namedtuple('UniqueAllResult', ['values', 'indices', 'inverse_indices', 'counts'])


def unique_values(x):
    """Return the unique values in x (no auxiliary outputs). equal_nan=False."""
    return unique(x, equal_nan=False)


def unique_counts(x):
    """Return unique values and their counts. equal_nan=False."""
    vals, counts = unique(x, return_counts=True, equal_nan=False)
    return _UniqueCountsResult(vals, counts)


def unique_inverse(x):
    """Return unique values and inverse indices. equal_nan=False; inverse shape matches input."""
    x = asarray(x)
    input_shape = x.shape
    vals, inv = unique(x, return_inverse=True, equal_nan=False)
    inv = inv.reshape(input_shape)
    return _UniqueInverseResult(vals, inv)


def unique_all(x):
    """Return unique values, indices, inverse indices, and counts. equal_nan=False."""
    x = asarray(x)
    input_shape = x.shape
    vals, idx, inv, counts = unique(
        x, return_index=True, return_inverse=True, return_counts=True, equal_nan=False)
    inv = inv.reshape(input_shape)
    return _UniqueAllResult(vals, idx, inv, counts)
