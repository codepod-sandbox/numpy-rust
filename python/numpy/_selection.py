"""Selection and placement: choose, compress, extract, select, take, put, delete, insert, etc."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray,
    _builtin_range, _builtin_min, _builtin_max,
    _flat_arraylike_data,
)
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate

__all__ = [
    'choose', 'compress', 'extract', 'select', 'piecewise',
    'take', '_take_structured', '_take_along_axis',
    'copyto', 'place', 'put', 'putmask',
    'delete', 'insert',
]


def _normalize_edit_axis(arr, axis, none_message):
    """Normalize axis handling shared by delete/insert style edits."""
    if axis is None:
        return arr.flatten(), 0
    if not isinstance(axis, int):
        try:
            axis = int(axis)
        except (TypeError, ValueError):
            raise TypeError(none_message)
    if arr.ndim == 0:
        raise AxisError(
            "axis {} is out of bounds for array of dimension 0".format(axis)
        )
    if axis < 0:
        axis = arr.ndim + axis
    if axis < 0 or axis >= arr.ndim:
        raise AxisError(
            "axis {} is out of bounds for array of dimension {}".format(axis, arr.ndim)
        )
    return arr, axis


def _delete_index_type_error():
    return IndexError(
        "only integers, slices (`:`), ellipsis (`...`), "
        "numpy.newaxis (`None`) and integer or boolean arrays are "
        "valid indices"
    )


def _normalize_delete_obj(obj, axis, n):
    """Normalize delete obj into integer positions on the target axis."""
    obj_dtype = str(obj.dtype) if isinstance(obj, ndarray) else ''

    if isinstance(obj, slice):
        idx_arr = arange(n)[obj]
        return [int(v) for v in _flat_arraylike_data(idx_arr)]
    if isinstance(obj, bool):
        raise ValueError(
            "in the future, boolean array-likes will be handled as a "
            "boolean array index"
        )
    if isinstance(obj, ndarray):
        if 'float' in obj_dtype or 'complex' in obj_dtype:
            raise IndexError("arrays used as indices must be of integer (or boolean) type")
        if obj_dtype == 'object' or 'timedelta' in obj_dtype or 'datetime' in obj_dtype:
            raise _delete_index_type_error()
        flat_list = _flat_arraylike_data(obj)
        if obj_dtype == 'bool':
            if len(flat_list) != n:
                raise ValueError(
                    "boolean index did not match indexed array along "
                    "dimension {}; dimension is {} but corresponding boolean "
                    "dimension is {}".format(axis, n, len(flat_list))
                )
            return [i for i, v in enumerate(flat_list) if v]
        values = flat_list
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []
        if all(isinstance(x, bool) for x in obj):
            if len(obj) != n:
                raise ValueError(
                    "boolean index did not match indexed array along "
                    "dimension {}; dimension is {} but corresponding boolean "
                    "dimension is {}".format(axis, n, len(obj))
                )
            return [i for i, v in enumerate(obj) if v]
        values = list(obj)
    elif isinstance(obj, int):
        values = [obj]
    else:
        try:
            values = [int(obj)]
        except (TypeError, ValueError):
            raise _delete_index_type_error()

    indices = []
    for value in values:
        index = int(value)
        if index < -n or index >= n:
            raise IndexError(
                "index {} is out of bounds for axis {} with size {}".format(index, axis, n)
            )
        indices.append(index)
    return indices


def _normalize_insert_obj(obj, n):
    """Normalize insert obj into integer indices and scalar-ness."""
    scalar_obj = False
    if isinstance(obj, slice):
        return list(_builtin_range(*obj.indices(n))), False
    if isinstance(obj, ndarray) and obj.dtype.kind == 'b':
        return [i for i in _builtin_range(obj.size) if bool(obj.flat[i])], False
    if isinstance(obj, ndarray) and obj.dtype.kind == 'f':
        raise IndexError("index {} is not a valid index for insertion".format(obj))
    if isinstance(obj, ndarray) and obj.size == 0:
        return [], False
    if isinstance(obj, (list, tuple)):
        return [int(v) for v in obj], False
    if isinstance(obj, ndarray) and obj.ndim > 0:
        if obj.dtype.kind == 'f':
            raise IndexError("index {} is not a valid index for insertion".format(obj))
        return [int(v) for v in _flat_arraylike_data(obj)], False
    if isinstance(obj, ndarray) and obj.ndim == 0:
        if obj.dtype.kind == 'f':
            raise IndexError("index {} is not a valid index for insertion".format(obj))
        return [int(float(obj[()]))], True
    if isinstance(obj, float):
        raise IndexError("index {} is not a valid index for insertion".format(obj))
    scalar_obj = True
    return [int(obj)], scalar_obj


def choose(a, choices, out=None, mode="raise"):
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Convert choices: complex scalars become float (real part) to match Rust clip behavior
    def _to_choice_array(c):
        if isinstance(c, complex):
            return asarray(c.real)
        if isinstance(c, ndarray):
            return c
        return asarray(c)
    choice_arrays = [_to_choice_array(c) for c in choices]
    result = _native.choose(a, choice_arrays)
    if out is not None and isinstance(out, ndarray):
        flat = _flat_arraylike_data(result)
        for i in range(len(flat)):
            out.flat[i] = flat[i]
        return out
    return result


def compress(condition, a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    cond = condition if isinstance(condition, ndarray) else asarray(condition)
    return _native.compress(cond, a, axis)


def extract(condition, arr):
    """Return elements of arr where condition is True."""
    condition = asarray(condition).flatten()
    arr = asarray(arr).flatten()
    return _native.compress(condition, arr, None)


def select(condlist, choicelist, default=0):
    """Return array drawn from elements in choicelist, depending on conditions."""
    from ._shape import broadcast_to
    if len(condlist) != len(choicelist):
        raise ValueError("condlist and choicelist must be the same length")
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")
    condlist = [asarray(c) for c in condlist]
    # Validate all conditions are bool dtype
    for cond in condlist:
        if str(cond.dtype) not in ('bool',):
            raise TypeError(
                "condlist must be a list of bool arrays")
    choicelist = [asarray(c) for c in choicelist]
    # Determine output shape by broadcasting all arrays together
    def _bcast_shapes(s1, s2):
        """Compute broadcast shape of two shape tuples."""
        if len(s1) < len(s2):
            s1 = (1,) * (len(s2) - len(s1)) + s1
        elif len(s2) < len(s1):
            s2 = (1,) * (len(s1) - len(s2)) + s2
        out = []
        for a_dim, b_dim in zip(s1, s2):
            if a_dim == 1:
                out.append(b_dim)
            elif b_dim == 1:
                out.append(a_dim)
            elif a_dim == b_dim:
                out.append(a_dim)
            else:
                raise ValueError("shape mismatch")
        return tuple(out)
    # Also include default in shape broadcasting if it's array-like
    _default_arr = None
    if not isinstance(default, (int, float, complex)):
        try:
            _default_arr = asarray(default)
        except Exception:
            pass
    all_arrays = condlist + choicelist
    if _default_arr is not None:
        all_arrays = all_arrays + [_default_arr]
    out_shape = condlist[0].shape
    for arr in all_arrays[1:]:
        try:
            out_shape = _bcast_shapes(out_shape, arr.shape)
        except Exception:
            pass
    # Determine output dtype
    _is_complex = isinstance(default, complex)
    for ch in choicelist:
        if 'complex' in str(ch.dtype):
            _is_complex = True
    if _is_complex:
        _default_val = complex(default) if not isinstance(default, complex) else default
        out_dtype = 'complex128'
    elif isinstance(default, float) or any('float' in str(ch.dtype) for ch in choicelist):
        _default_val = float(default)
        out_dtype = 'float64'
    else:
        try:
            _default_val = float(default)
        except (TypeError, ValueError):
            _default_val = 0.0
        out_dtype = None
        for ch in choicelist:
            _dt = str(ch.dtype)
            if 'int' in _dt or 'float' in _dt:
                out_dtype = _dt
                break
        if out_dtype is None:
            out_dtype = 'float64'
    # Build result by broadcasting conditions and choices
    # Start with default value broadcast to output shape
    _size = 1
    for s in out_shape:
        _size *= s
    if _is_complex:
        _re = _default_val.real
        _im = _default_val.imag
        result_re = [_re] * _size
        result_im = [_im] * _size
    else:
        result_flat = [float(_default_val)] * _size
    # Broadcast each array to out_shape
    def _broadcast_to_flat(arr, shape):
        """Broadcast arr to shape and return flat list."""
        if arr.shape == shape:
            return _flat_arraylike_data(arr)
        # Simple broadcast: expand dimensions
        _arr = arr
        while _arr.ndim < len(shape):
            _arr = _arr.reshape([1] + list(_arr.shape))
        return _flat_arraylike_data(broadcast_to(_arr, shape))
    def _broadcast_cond_flat(cond, shape):
        """Broadcast boolean cond to flat list of ints."""
        if cond.shape == shape:
            vals = _flat_arraylike_data(cond)
        elif cond.size == 1:
            v = _flat_arraylike_data(cond)[0]
            _size2 = 1
            for s in shape:
                _size2 *= s
            vals = [v] * _size2
        else:
            _c = cond
            while _c.ndim < len(shape):
                _c = _c.reshape([1] + list(_c.shape))
            _c = broadcast_to(_c, shape)
            vals = _flat_arraylike_data(_c)
        return [bool(v) for v in vals]
    # Process in reverse order so first matching condition wins
    for i in _builtin_range(len(condlist) - 1, -1, -1):
        cond_flat = _broadcast_cond_flat(condlist[i], out_shape)
        choice_flat = _broadcast_to_flat(choicelist[i], out_shape)
        if _is_complex:
            ch_arr = choicelist[i]
            if 'complex' in str(ch_arr.dtype):
                ch_re = ch_arr.real
                ch_im = ch_arr.imag
                if not isinstance(ch_re, ndarray):
                    ch_re = array([float(ch_re)])
                    ch_im = array([float(ch_im)])
                ch_re_flat = _broadcast_to_flat(ch_re, out_shape)
                ch_im_flat = _broadcast_to_flat(ch_im, out_shape)
            else:
                ch_re_flat = choice_flat
                ch_im_flat = [0.0] * len(choice_flat)
            for j in _builtin_range(_size):
                if cond_flat[j]:
                    result_re[j] = ch_re_flat[j]
                    result_im[j] = ch_im_flat[j]
        else:
            for j in _builtin_range(_size):
                if cond_flat[j]:
                    result_flat[j] = float(choice_flat[j])
    if _is_complex:
        _vals = [complex(result_re[j], result_im[j]) for j in _builtin_range(_size)]
        result_arr = array(_vals, dtype='complex128')
    else:
        result_arr = array(result_flat, dtype=out_dtype)
    if len(out_shape) != 1 or out_shape != result_arr.shape:
        result_arr = result_arr.reshape(list(out_shape))
    return result_arr


def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function."""
    orig_x = x
    x = asarray(x)
    subtype = type(orig_x) if isinstance(orig_x, ndarray) and type(orig_x) is not ndarray else None

    def _restore_piecewise_subclass(value):
        if subtype is None:
            return value
        if isinstance(value, ndarray) and type(value) is ndarray:
            return value.view(subtype)
        return value

    n_conditions = len(condlist)
    n_funcs = len(funclist)

    # Normalize condlist: if it's a 1D boolean/int array or a flat list of scalars,
    # treat it as a single condition (wrap in a list)
    _is_flat_condlist = False
    if isinstance(condlist, ndarray):
        if condlist.ndim <= 1:
            _is_flat_condlist = True
            condlist = [condlist]
    elif isinstance(condlist, list) and n_conditions > 0:
        # Check if each element is a scalar bool/int (not an array or list)
        _all_scalar = True
        for c in condlist:
            if isinstance(c, (list, ndarray)):
                _all_scalar = False
                break
        if _all_scalar and len(condlist) == x.size and x.ndim > 0:
            _is_flat_condlist = True
            condlist = [asarray(condlist)]

    n_conditions = len(condlist)
    n_funcs = len(funclist)

    # Validate funclist length
    if n_funcs not in (n_conditions, n_conditions + 1):
        raise ValueError(
            "{} or {} functions are expected, got {}".format(
                n_conditions, n_conditions + 1, n_funcs
            )
        )

    # Support 0-d arrays
    if x.ndim == 0:
        scalar_x = float(x)
        for j, cond in enumerate(condlist):
            c = asarray(cond)
            c_val = False
            if c.ndim == 0:
                c_val = bool(float(c))
            elif c.size == 1:
                c_val = bool(float(c.flatten()[0]))
            else:
                # Condition is array, treat scalar x against first element
                c_val = bool(float(c.flatten()[0]))
            if c_val:
                if callable(funclist[j]):
                    val = funclist[j](x)
                    if not isinstance(val, ndarray):
                        val = asarray(val).reshape(())
                    return _restore_piecewise_subclass(val)
                else:
                    return _restore_piecewise_subclass(asarray(funclist[j]).reshape(()))
        # No condition matched - use "otherwise" or default 0
        if n_funcs == n_conditions + 1:
            if callable(funclist[-1]):
                val = funclist[-1](x)
                if not isinstance(val, ndarray):
                    val = asarray(val).reshape(())
                return _restore_piecewise_subclass(val)
            else:
                return _restore_piecewise_subclass(asarray(funclist[-1]).reshape(()))
        return _restore_piecewise_subclass(asarray(0.0).reshape(()))

    flat_x = x.flatten()
    n = x.size
    has_otherwise = n_funcs == n_conditions + 1
    result = [0.0] * n

    for i in range(n):
        try:
            val = float(flat_x[i])
        except (TypeError, ValueError):
            val = flat_x[i]
        matched = False
        for j, cond in enumerate(condlist):
            c = asarray(cond)
            # Handle scalar condition (broadcast to all elements)
            if c.ndim == 0 or c.size == 1:
                c_val = bool(float(c.flatten()[0]))
            else:
                c_flat = c.flatten()
                if i >= c_flat.size:
                    c_val = False
                else:
                    c_val = float(c_flat[i]) != 0.0
            if c_val:
                if callable(funclist[j]):
                    result[i] = float(funclist[j](val))
                else:
                    result[i] = float(funclist[j])
                matched = True
                break
        if not matched and has_otherwise:
            if callable(funclist[-1]):
                result[i] = float(funclist[-1](val))
            else:
                result[i] = float(funclist[-1])

    result_arr = array(result)
    if x.ndim > 1:
        result_arr = result_arr.reshape(list(x.shape))
    elif x.ndim == 0:
        result_arr = result_arr.reshape(())
    return _restore_piecewise_subclass(result_arr)


def _take_structured(a, indices, axis):
    """Pure-Python take for StructuredArray."""
    from ._creation import _create_structured_array
    import itertools
    idx_list = _flat_arraylike_data(indices) if hasattr(indices, 'tolist') else list(indices)
    shape = list(a.shape)
    if axis is None:
        # Flatten then index: result is 1D
        flat_tuples = []
        for i in range(a.size):
            coords = []
            rem = i
            for s in reversed(a.shape):
                coords.insert(0, rem % s)
                rem //= s
            val = a[tuple(coords)]
            flat_tuples.append(val if isinstance(val, tuple) else tuple(_flat_arraylike_data(val)))
        picked = [flat_tuples[j] for j in idx_list]
        return _create_structured_array(picked, a.dtype)
    # axis-based take
    ax = axis if axis >= 0 else len(shape) + axis
    leading = shape[:ax]
    trailing = shape[ax+1:]
    new_shape = leading + [len(idx_list)] + trailing
    # Build result by iterating all combinations of leading and trailing indices
    _field_names = a.dtype.names if hasattr(a.dtype, 'names') else None
    flat_records = []
    for lead_idx in itertools.product(*[range(s) for s in leading]):
        for k in idx_list:
            for trail_idx in itertools.product(*[range(s) for s in trailing]):
                coords = lead_idx + (k,) + trail_idx
                val = a[coords] if len(coords) > 1 else a[coords[0]]
                # Convert void or structured scalar to plain tuple
                if _field_names and type(val).__name__ == 'void':
                    flat_records.append(tuple(val[n] for n in _field_names))
                elif isinstance(val, tuple):
                    flat_records.append(val)
                elif hasattr(val, 'tolist'):
                    v = _flat_arraylike_data(val)
                    flat_records.append(v if isinstance(v, tuple) else tuple(v) if isinstance(v, list) else (v,))
                else:
                    flat_records.append((val,))
    sa_flat = _create_structured_array(flat_records, a.dtype)
    return sa_flat.reshape(new_shape)


def take(a, indices, axis=None, out=None, mode="raise"):
    if type(a).__name__ == 'StructuredArray' and hasattr(a, 'shape'):
        if not hasattr(indices, 'tolist'):
            indices = asarray(indices).astype('int64')
        return _take_structured(a, indices, axis)
    if not isinstance(a, ndarray):
        a = asarray(a)
    if not isinstance(indices, ndarray):
        indices = asarray(indices).astype('int64')
    # Handle empty arrays or empty indices
    if axis is not None:
        ax = axis if axis >= 0 else a.ndim + axis
        dim_size = a.shape[ax]
        if dim_size == 0 and indices.size > 0:
            raise IndexError("cannot take from a 0-length dimension")
    elif a.size == 0 and indices.size > 0:
        raise IndexError("cannot take from a 0-length dimension")
    if indices.size == 0:
        if axis is None:
            return zeros(indices.shape, dtype=a.dtype)
        shape = list(a.shape)
        ax = axis if axis >= 0 else a.ndim + axis
        shape[ax] = 0
        return zeros(shape, dtype=a.dtype)
    # Normalize negative axis before passing to Rust (Rust uses unsigned int)
    native_axis = axis
    if native_axis is not None and native_axis < 0:
        native_axis = a.ndim + native_axis
    result = _native.take(a, indices, native_axis)
    # When axis=None, NumPy returns a result with the same shape as indices
    if axis is None and indices.ndim > 1 and result.ndim != indices.ndim:
        result = result.reshape(indices.shape)
    return result


def _take_along_axis(arr, indices, axis):
    """Take slices along an axis by indices list, return concatenated result."""
    parts = []
    for idx in indices:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(idx, idx + 1)
        parts.append(arr[tuple(slices)])
    return concatenate(parts, axis=axis)


def copyto(dst, src, casting='same_kind', where=True):
    """Copy values from one array to another, broadcasting as necessary."""
    from ._shape import broadcast_to
    from ._creation import where as _where
    src = asarray(src)
    dst = asarray(dst)
    src_b = broadcast_to(src, dst.shape)
    if where is True:
        return src_b
    mask = asarray(where)
    return _where(mask, src_b, dst)


def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values."""
    if not isinstance(arr, ndarray):
        raise TypeError(
            "argument 1 must be numpy.ndarray, not {}".format(type(arr).__name__)
        )
    mask_arr = asarray(mask).flatten()
    vals_arr = asarray(vals).flatten()
    n = arr.size
    nv = vals_arr.size
    # Count True positions to validate
    count_true = 0
    for i in _builtin_range(n):
        if bool(mask_arr[i]):
            count_true += 1
    if count_true > 0 and nv == 0:
        raise ValueError("Cannot insert from an empty array!")
    flat_a = arr.flatten()
    vi = 0
    for i in _builtin_range(n):
        if bool(mask_arr[i]):
            flat_a[i] = vals_arr[vi % nv]
            vi += 1
    # Write back in-place
    if arr.ndim == 1:
        for i in _builtin_range(n):
            arr[i] = flat_a[i]
    else:
        _shape = arr.shape
        _strides = [1] * arr.ndim
        for _d in _builtin_range(arr.ndim - 2, -1, -1):
            _strides[_d] = _strides[_d + 1] * _shape[_d + 1]
        for i in _builtin_range(n):
            _idx = []
            _rem = i
            for _d in _builtin_range(arr.ndim):
                _idx.append(_rem // _strides[_d])
                _rem %= _strides[_d]
            arr[tuple(_idx)] = flat_a[i]


def put(a, ind, v, mode='raise'):
    """Replaces specified elements of an array with given values.

    Modifies 'a' in-place and returns None.
    """
    from ._helpers import _ObjectArray
    if not isinstance(a, (ndarray, _ObjectArray)):
        raise TypeError("argument 1 must be numpy.ndarray")
    ind_arr = asarray(ind).flatten()
    v_arr = asarray(v).flatten()
    n = a.size
    ni = ind_arr.size
    nv = v_arr.size
    if nv == 0:
        # Empty values: no-op (numpy behavior)
        return None
    for idx in range(ni):
        i = int(ind_arr[idx])
        if mode == 'wrap':
            if n == 0:
                return None
            i = i % n
        elif mode == 'clip':
            if n == 0:
                return None
            if i < 0:
                i = 0
            elif i >= n:
                i = n - 1
        elif mode == 'raise':
            if i < 0:
                i = n + i
            if i < 0 or i >= n:
                raise IndexError("index {} is out of bounds for axis 0 with size {}".format(
                    int(ind_arr[idx]), n))
        if isinstance(a, _ObjectArray):
            a._data[i] = v_arr[idx % nv] if isinstance(v_arr, _ObjectArray) else float(v_arr[idx % nv])
        else:
            a.flat[i] = float(v_arr[idx % nv])
    return None


def putmask(a, mask, values):
    """Changes elements of an array based on conditional and input values.

    Modifies 'a' in-place and returns None.
    """
    mask = asarray(mask)
    values = asarray(values)
    flat_m = mask.flatten()
    flat_v = values.flatten()
    n = a.size
    nv = flat_v.size
    if nv == 0:
        return None
    vi = 0
    for i in range(n):
        if flat_m[i]:
            a.flat[i] = float(flat_v[vi % nv])
            vi += 1
    return None


def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted."""
    orig_arr = arr
    arr = asarray(arr)

    def _restore_subclass(result):
        if type(orig_arr) is not ndarray and isinstance(orig_arr, ndarray):
            try:
                return result.view(type(orig_arr))
            except Exception:
                return result
        return result

    if axis is not None and not isinstance(axis, int):
        try:
            axis = int(axis)
        except (TypeError, ValueError):
            raise TypeError(
                "axis must be an integer, not {}".format(type(axis).__name__)
            )

    if axis is None:
        arr = arr.flatten()
        axis = 0
    elif arr.ndim == 0:
        raise AxisError(
            "Axis {} is out of bounds for array of dimension 0".format(axis)
        )
    elif axis < 0:
        axis = arr.ndim + axis
        if axis < 0:
            raise AxisError(
                "axis {} is out of bounds for array of dimension {}".format(axis, arr.ndim)
            )
    elif axis >= arr.ndim:
        raise AxisError(
            "axis {} is out of bounds for array of dimension {}".format(axis, arr.ndim)
        )

    n = arr.shape[axis]
    del_indices = _normalize_delete_obj(obj, axis, n)

    # Normalize negative indices and build keep list
    del_set = set(i if i >= 0 else n + i for i in del_indices)
    keep = [i for i in _builtin_range(n) if i not in del_set]

    # Build result by selecting kept slices along axis and concatenating
    if len(keep) == 0:
        new_shape = list(arr.shape)
        new_shape[axis] = 0
        result = zeros(new_shape, dtype=str(arr.dtype))
    else:
        parts = []
        for i in keep:
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + 1)
            parts.append(arr[tuple(sl)])
        result = concatenate(parts, axis=axis)

    result = _restore_subclass(result)

    # Preserve memory order (Fortran or C) from input array
    if hasattr(arr, 'flags') and arr.flags.f_contiguous:
        if hasattr(result, '_mark_fortran'):
            result._mark_fortran()

    return result


def insert(arr, obj, values, axis=None):
    """Insert values along the given axis before the given indices."""
    from ._helpers import AxisError
    from ._shape import transpose
    orig_arr = arr
    arr = asarray(arr)

    def _restore_subclass(result):
        if type(orig_arr) is not ndarray and isinstance(orig_arr, ndarray):
            try:
                return result.view(type(orig_arr))
            except Exception:
                return result
        return result

    arr, axis = _normalize_edit_axis(arr, axis, "an integer is required")

    ndims = arr.ndim
    n = arr.shape[axis]

    indices, scalar_obj = _normalize_insert_obj(obj, n)

    # Normalize negative indices and check bounds
    norm_indices = []
    for idx in indices:
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx > n:
            raise IndexError("index {} is out of bounds for axis {} with size {}".format(idx, axis, n))
        norm_indices.append(idx)

    if not norm_indices:
        return _restore_subclass(arr.copy())

    if getattr(getattr(arr, 'dtype', None), 'names', None) and arr.ndim == 1 and axis == 0:
        from ._creation import _create_structured_array
        from ._shape import broadcast_to

        val_arr = values if type(values).__name__ == 'StructuredArray' else array(values, dtype=arr.dtype)
        if getattr(val_arr, 'ndim', 0) == 0:
            val_arr = val_arr.reshape((1,))

        insert_count = int(getattr(val_arr, 'shape', (1,))[0]) if getattr(val_arr, 'ndim', 0) > 0 else 1
        if scalar_obj and len(norm_indices) == 1 and insert_count > 1:
            norm_indices = [norm_indices[0]] * insert_count
        elif not scalar_obj:
            if insert_count == 1 and len(norm_indices) > 1:
                val_arr = broadcast_to(val_arr, (len(norm_indices),))
                insert_count = len(norm_indices)
            elif len(norm_indices) == 1 and insert_count > 1:
                norm_indices = [norm_indices[0]] * insert_count

        if insert_count != len(norm_indices):
            raise ValueError(
                "could not broadcast input array from shape {} into shape {}".format(
                    getattr(val_arr, 'shape', ()), (len(norm_indices),)
                )
            )

        field_names = arr.dtype.names

        def _record_tuple(v):
            if isinstance(v, tuple):
                return v
            if type(v).__name__ == 'void':
                return tuple(v[name] for name in field_names)
            if hasattr(v, 'tolist'):
                out = _flat_arraylike_data(v)
                if isinstance(out, tuple):
                    return out
                if isinstance(out, list):
                    return tuple(out)
            return tuple(v)

        records = []
        ins_pos = 0
        inserts = sorted(zip(norm_indices, _builtin_range(insert_count)), key=lambda x: x[0])
        for orig_i in _builtin_range(n):
            while ins_pos < len(inserts) and inserts[ins_pos][0] == orig_i:
                records.append(_record_tuple(val_arr[inserts[ins_pos][1]]))
                ins_pos += 1
            records.append(_record_tuple(arr[orig_i]))
        while ins_pos < len(inserts):
            records.append(_record_tuple(val_arr[inserts[ins_pos][1]]))
            ins_pos += 1

        result = _create_structured_array(records, arr.dtype)
        return _restore_subclass(result)

    # Move target axis to front
    perm = [axis] + [i for i in _builtin_range(ndims) if i != axis]
    inv_perm = [0] * ndims
    for i, p in enumerate(perm):
        inv_perm[p] = i
    t = transpose(arr, perm)  # shape: (n, ...)
    sub_shape = t.shape[1:]
    row_size = 1
    for s in sub_shape:
        row_size *= s

    val_arr = asarray(values) if not isinstance(values, ndarray) else values

    def _make_val_row(v):
        """Create a sub-array row from value v (scalar or array)."""
        if not sub_shape:
            if hasattr(v, '__float__'):
                try:
                    return asarray(float(v))
                except Exception:
                    return asarray(v)
            return asarray(v)
        v_arr = asarray(v)
        if v_arr.ndim == 0:
            flat_row = [float(v_arr[()])] * row_size
        elif v_arr.size == row_size:
            flat_row = [float(x) for x in _flat_arraylike_data(v_arr)]
        else:
            flat_row = [float(v_arr.flat[0])] * row_size
        return array(flat_row).reshape(list(sub_shape))

    # Determine val_list matching norm_indices
    if val_arr.ndim == 0:
        # Scalar value: use for all insertions
        val_list = [val_arr[()] for _ in norm_indices]
    elif len(norm_indices) == 1 and val_arr.size == row_size and (
            (scalar_obj and val_arr.ndim < 2) or (not scalar_obj and val_arr.ndim >= 2)):
        # Single insertion with values matching sub_shape exactly:
        # - scalar_obj + 1D values: values fill sub_shape as one column
        # - array_obj + 2D+ values: explicit column values for one insertion
        val_list = [val_arr]
    elif scalar_obj:
        # Scalar obj with multiple values: insert all values at the same position
        if val_arr.ndim >= 2:
            # Multi-dim values: each row is a separate insertion, broadcast to sub_shape
            n_inserts = val_arr.shape[0]
            val_list = []
            for i in _builtin_range(n_inserts):
                val_list.append(val_arr[i])
            norm_indices = [norm_indices[0]] * n_inserts
        elif row_size == 1:
            val_flat = val_arr.flatten()
            val_list = list(_flat_arraylike_data(val_flat))
            norm_indices = [norm_indices[0]] * len(val_list)
        else:
            # 1D values for scalar obj: 1 insertion, values broadcast to sub_shape
            val_list = [val_arr]
    else:
        # Array obj: one value per index (with broadcasting for scalars)
        val_flat = val_arr.flatten()
        if val_flat.size == 1:
            val_list = [val_flat[0] for _ in norm_indices]
        elif val_flat.size == len(norm_indices):
            val_list = [val_flat[i] for i in _builtin_range(len(norm_indices))]
        elif len(norm_indices) == 1:
            # 1 index, multiple values
            if val_arr.ndim == 1:
                # 1D values: each element is a separate scalar insertion (broadcast obj)
                val_list = list(_flat_arraylike_data(val_flat))
                norm_indices = [norm_indices[0]] * len(val_list)
            elif row_size == 1:
                val_list = list(_flat_arraylike_data(val_flat))
                norm_indices = [norm_indices[0]] * len(val_list)
            else:
                # 2D+ values with 1 obj: 1 insertion, values broadcast to sub_shape
                val_list = [val_arr]
        else:
            val_list = [val_flat[i] for i in _builtin_range(len(norm_indices))]

    # Build sorted insertion pairs: (index, value)
    inserts = sorted(zip(norm_indices, val_list), key=lambda x: x[0])

    all_rows = []
    ins_pos = 0
    for orig_i in _builtin_range(n):
        while ins_pos < len(inserts) and inserts[ins_pos][0] == orig_i:
            all_rows.append(_make_val_row(inserts[ins_pos][1]))
            ins_pos += 1
        all_rows.append(t[orig_i])
    while ins_pos < len(inserts):
        all_rows.append(_make_val_row(inserts[ins_pos][1]))
        ins_pos += 1

    # Stack rows
    flat = []
    for row in all_rows:
        r = asarray(row).flatten()
        _rsize = r.size if r.size > 0 else (row_size if row_size > 0 else 1)
        for j in _builtin_range(_rsize):
            if r.ndim > 0:
                flat.append(float(r[j]))
            else:
                flat.append(float(r[()]))
    new_shape = [len(all_rows)] + list(sub_shape)
    result = array(flat).reshape(new_shape) if flat else array([]).reshape(new_shape)
    result = transpose(result, inv_perm)
    return _restore_subclass(result)
