"""Array creation functions."""
import sys as _sys
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray, dot
from _numpy_native import concatenate as _native_concatenate
from ._helpers import (
    AxisError, _ObjectArray, _ComplexResultArray,
    _copy_into, _apply_order, _infer_shape,
    _flatten_nested, _all_bools_nested, _to_float_list, _is_temporal_dtype,
    _temporal_dtype_info, _make_temporal_array, _CLIP_UNSET, _builtin_range,
    _unsupported_numeric_dtype,
)
from ._core_types import (
    dtype, _ScalarType, _normalize_dtype, _normalize_dtype_with_size,
    _string_dtype_info, _NumpyVoidScalar, _DTypeClassMeta,
)
from ._datetime import datetime64, timedelta64

__all__ = [
    # internal helpers
    '_array_core', '_make_complex_array', '_detect_builtin_str_bytes',
    '_make_str_bytes_result', '_make_void_result', '_like_order',
    # core creation
    'array', 'concatenate',
    # zeros/ones family
    'empty', 'full', 'full_like', 'zeros', 'zeros_like',
    'ones', 'ones_like', 'empty_like',
    # sequence
    'eye', 'identity', 'arange', 'linspace', 'logspace', 'geomspace',
    # conversion
    'asarray', 'asanyarray', 'ascontiguousarray', 'asfortranarray',
    'asarray_chkfinite', 'copy', 'require',
    # from-functions
    'frombuffer', 'fromfunction', 'fromfile', 'fromiter', 'fromstring',
    # comparison
    'array_equal', 'array_equiv',
    # conditional
    'where',
]


def _is_fraction_scalar(value):
    return type(value).__name__ == 'Fraction' and type(value).__module__ == 'fractions'


def _is_decimal_scalar(value):
    return type(value).__name__ == 'Decimal' and type(value).__module__ == 'decimal'


def _contains_fraction_values(value):
    if _is_fraction_scalar(value):
        return True
    if isinstance(value, _ObjectArray):
        return any(_contains_fraction_values(v) for v in value.flatten().tolist())
    if isinstance(value, ndarray):
        if str(value.dtype) != 'object':
            return False
        return any(_contains_fraction_values(v) for v in value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        return any(_contains_fraction_values(v) for v in value)
    return False


def _contains_decimal_values(value):
    if _is_decimal_scalar(value):
        return True
    if isinstance(value, _ObjectArray):
        return any(_contains_decimal_values(v) for v in value.flatten().tolist())
    if isinstance(value, ndarray):
        if str(value.dtype) != 'object':
            return False
        return any(_contains_decimal_values(v) for v in value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        return any(_contains_decimal_values(v) for v in value)
    return False


def _flatten_object_nested(value):
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_object_nested(item))
        return out
    return [value]


def _contains_large_python_ints(value):
    if type(value) is int:
        return value < -9223372036854775808 or value > 9223372036854775807
    if isinstance(value, (list, tuple)):
        return any(_contains_large_python_ints(item) for item in value)
    return False


def _native_integer_result(values, dtype_name):
    if dtype_name == 'uint64':
        signed = [
            value if value <= 9223372036854775807 else value - 18446744073709551616
            for value in values
        ]
        result = _native.zeros([len(signed)], 'int64')
        for i, value in enumerate(signed):
            result[i] = int(value)
        return result.astype('uint64')
    result = _native.zeros([len(values)], dtype_name)
    for i, value in enumerate(values):
        result[i] = int(value)
    return result


def _array_from_array_interface(data):
    import ctypes
    import numpy as _np

    ai = getattr(data, "__array_interface__", None)
    if not isinstance(ai, dict):
        return None
    shape = tuple(ai.get("shape", ()))
    typestr = ai.get("typestr")
    data_info = ai.get("data")
    if typestr is None or not isinstance(data_info, tuple) or not data_info:
        return None
    ptr = data_info[0]
    if not ptr:
        return None

    dt_name = str(_np.dtype(typestr))
    ctype_map = {
        "bool": ctypes.c_bool,
        "int8": ctypes.c_int8,
        "int16": ctypes.c_int16,
        "int32": ctypes.c_int32,
        "int64": ctypes.c_int64,
        "uint8": ctypes.c_uint8,
        "uint16": ctypes.c_uint16,
        "uint32": ctypes.c_uint32,
        "uint64": ctypes.c_uint64,
        "float32": ctypes.c_float,
        "float64": ctypes.c_double,
    }
    ctype = ctype_map.get(dt_name)
    if ctype is None:
        return None

    count = 1
    for dim in shape:
        count *= dim
    buffer = (ctype * count).from_address(ptr)
    values = list(buffer)
    result = array(values, dtype=dt_name)
    if shape:
        result = result.reshape(shape)
    return result


def concatenate(arrays, axis=0, out=None, dtype=None, casting='same_kind'):
    """Wrapper around native concatenate that handles tuples and auto-converts to arrays."""
    from ._core_types import can_cast
    _valid_castings = ('no', 'equiv', 'safe', 'same_kind', 'unsafe')
    if casting not in _valid_castings:
        raise ValueError("casting must be one of {}".format(_valid_castings))
    # Cannot specify both out and dtype
    if out is not None and dtype is not None:
        raise TypeError("concatenate() does not support 'out' and 'dtype' together")
    arrs = [asarray(a) if not isinstance(a, ndarray) else a for a in arrays]
    # When axis is None, flatten all inputs and concatenate along axis 0
    if axis is None:
        arrs = [a.flatten() for a in arrs]
        axis = 0
    else:
        # 0-d arrays are not allowed with explicit axis
        for a in arrs:
            if a.ndim == 0:
                raise ValueError(
                    "zero-dimensional arrays cannot be concatenated")
        # All arrays must have the same number of dimensions
        if len(arrs) > 1:
            ndim0 = arrs[0].ndim
            for i, a in enumerate(arrs[1:], 1):
                if a.ndim != ndim0:
                    raise ValueError(
                        "all the input arrays must have same number of dimensions, "
                        "but the array at index 0 has {} dimension(s) and the array "
                        "at index {} has {} dimension(s)".format(ndim0, i, a.ndim))

    # Normalize negative axis and validate
    ndim = arrs[0].ndim if len(arrs) > 0 else 1
    if axis < 0:
        orig_axis = axis
        axis = ndim + axis
        if axis < 0:
            raise AxisError(orig_axis, ndim)
    elif axis >= ndim:
        raise AxisError(axis, ndim)
    # Validate casting against target dtype (from out or dtype parameter)
    target_dtype = None
    if out is not None:
        out_arr = asarray(out) if not isinstance(out, ndarray) else out
        target_dtype = str(out_arr.dtype)
    elif dtype is not None:
        target_dtype = str(dtype)
    # Validate casting against target dtype
    # For unicode string target: allow numeric→string (implicit unsafe) but reject object dtype
    # For bytes/S* target: respect casting parameter normally
    def _is_str_dtype(dt):
        s = str(dt)
        return (s.startswith('U') or s.startswith('<U') or
                s.startswith('>U') or s in ('str', 'unicode'))
    def _is_object_dtype(a):
        return str(a.dtype) == 'object'
    if target_dtype is not None:
        _tgt_is_str = _is_str_dtype(target_dtype)
        _tgt_str = str(target_dtype)
        # Subarray dtypes like '(2,)i' are not supported as concatenate target
        if _tgt_str.startswith('('):
            raise TypeError(
                "output dtype is not allowed to be a subarray dtype")
        for a in arrs:
            if _tgt_is_str:
                # Only raise for object/None-containing arrays
                if _is_object_dtype(a):
                    raise TypeError(
                        "Cannot cast array data from dtype('{}') to dtype('{}') "
                        "according to the rule '{}'".format(a.dtype, target_dtype, casting))
            elif not can_cast(a, target_dtype, casting=casting):
                raise TypeError(
                    "Cannot cast array data from dtype('{}') to dtype('{}') "
                    "according to the rule '{}'".format(a.dtype, target_dtype, casting))
    # Validate shapes match except along concat axis
    if len(arrs) > 1 and arrs[0].ndim > 0:
        ref = arrs[0]
        for i, a in enumerate(arrs[1:], 1):
            for dim in range(ref.ndim):
                if dim == axis:
                    continue
                if ref.shape[dim] != a.shape[dim]:
                    raise ValueError(
                        "all the input array dimensions except for the concatenation "
                        "axis must match exactly, but along dimension {}, the array at "
                        "index 0 has size {} and the array at index {} has size {}".format(
                            dim, ref.shape[dim], i, a.shape[dim]))
    from ._helpers import _ObjectArray
    if any(isinstance(a, _ObjectArray) for a in arrs):
        # Python-level concatenation for _ObjectArray
        result = _concat_object_arrays(arrs, axis)
    else:
        result = _native_concatenate(arrs, axis)
    if target_dtype is not None:
        result = result.astype(target_dtype)
    if out is not None:
        if out_arr.shape != result.shape:
            raise ValueError(
                "Output array shape {} does not match "
                "concatenation result shape {}".format(out_arr.shape, result.shape))
        out_arr[:] = result
        return out_arr
    return result

def _concat_object_arrays(arrs, axis):
    """Concatenate arrays along axis, handling _ObjectArray."""
    from ._helpers import _ObjectArray
    # Convert all to _ObjectArray with common dtype
    dtypes = [str(a.dtype) for a in arrs]
    # Use the first complex dtype if any, else first dtype
    common_dt = dtypes[0]
    for d in dtypes:
        if 'complex' in d:
            common_dt = d
            break
    converted = []
    for a in arrs:
        if isinstance(a, ndarray):
            converted.append(_ObjectArray(a.tolist(), common_dt, shape=a.shape))
        elif isinstance(a, _ObjectArray):
            converted.append(a)
        else:
            converted.append(_ObjectArray([a], common_dt))
    # Gather values along the concat axis
    shapes = [c.shape for c in converted]
    # Verify shapes match except along axis
    ref_shape = list(shapes[0])
    for i, s in enumerate(shapes[1:], 1):
        for d in range(len(ref_shape)):
            if d != axis and ref_shape[d] != s[d]:
                raise ValueError(
                    "all the input array dimensions except for the concatenation axis "
                    "must match exactly")
    new_shape = list(ref_shape)
    new_shape[axis] = sum(s[axis] for s in shapes)
    # For 1-d case, simple concatenation
    if len(new_shape) == 1:
        all_vals = []
        for c in converted:
            all_vals.extend(c.tolist())
        return _ObjectArray(all_vals, common_dt, shape=tuple(new_shape))
    # For n-d, use nested list manipulation
    all_data = []
    for c in converted:
        all_data.append(c.tolist())
    if axis == 0:
        combined = []
        for d in all_data:
            if isinstance(d, list):
                combined.extend(d)
            else:
                combined.append(d)
        return _ObjectArray(combined, common_dt, shape=tuple(new_shape))
    # General axis - flatten, rearrange, and reshape
    # Simple approach: iterate along axis and collect
    import itertools
    ndim = len(new_shape)
    result_data = []
    # Build result by iterating all positions
    for pos in itertools.product(*(range(s) for s in new_shape)):
        # Find which source array this position belongs to
        ax_idx = pos[axis]
        offset = 0
        for src_i, c in enumerate(converted):
            if ax_idx < offset + shapes[src_i][axis]:
                src_pos = list(pos)
                src_pos[axis] = ax_idx - offset
                val = c
                for p in src_pos:
                    val = val[p]
                result_data.append(val)
                break
            offset += shapes[src_i][axis]
    return _ObjectArray(result_data, common_dt, shape=tuple(new_shape))

def _make_complex_array(values, shape):
    """Create a complex128 ndarray from a list of Python complex/float values.
    Uses real+imag decomposition since the Rust array() can't handle complex lists directly."""
    reals = [v.real if isinstance(v, complex) else float(v) for v in values]
    imags = [v.imag if isinstance(v, complex) else 0.0 for v in values]
    # Build real and imag arrays, promote both to complex128, then add
    re_arr = _native.array(reals).reshape(shape).astype("complex128")
    im_arr = _native.array(imags).reshape(shape).astype("complex128")
    # im_arr contains real values that should become imaginary parts.
    # We need to multiply by 1j. The Rust side supports complex arithmetic,
    # so create a scalar 1j array and multiply.
    j_scalar = zeros(shape, dtype="complex128")
    # Workaround: we can construct the complex array correctly by
    # using the real array as base and adding imag * 1j via Rust ops
    # re_arr already has imag=0, im_arr has the imag values as real part
    # We need: result[i] = complex(reals[i], imags[i])
    # Since Rust complex arrays store (re, im) pairs, and re_arr.astype("complex128")
    # gives us (reals[i], 0), we need a way to set the imaginary part.
    # The cleanest way: build via the Rust ops: re_arr + im_arr * 1j
    # But we need a 1j constant array. Let's try creating one:
    ones_arr = ones(shape, dtype="complex128")  # (1+0j)
    # Subtract real part to get (0+0j), then... this doesn't help.
    # Alternative: use Rust-level subtract and multiply
    # Actually, the simplest: create the array by putting values into an _ObjectArray
    # with reshape support, since scimath results are usually small.
    return _ComplexResultArray(values, shape)




def _is_structured_dtype(dt):
    """Return True if dt is a StructuredDtype or a dtype wrapping one, or a comma string."""
    from ._core_types import StructuredDtype, dtype as _dtype_cls
    if isinstance(dt, StructuredDtype):
        return True
    if isinstance(dt, _dtype_cls) and (hasattr(dt, '_structured') and dt._structured is not None):
        return True
    if isinstance(dt, str) and ',' in dt:
        return True
    return False


def _create_structured_array(data, sdt):
    """Create a StructuredArray from a sequence of tuples/void scalars and a StructuredDtype.

    Args:
        data: sequence of tuples or void scalars, e.g. [(1.0, 2), (3.0, 4)]
        sdt: StructuredDtype or dtype wrapping a StructuredDtype
    """
    import json
    from numpy import StructuredArray, void
    from ._core_types import StructuredDtype
    # Unwrap if dtype wraps a StructuredDtype
    if hasattr(sdt, '_structured') and sdt._structured is not None:
        sdt = sdt._structured
    names = sdt.names
    nrows = len(data)
    fields = []
    for i, name in enumerate(names):
        field_dtype_obj, _ = sdt.fields[name]
        col_values = []
        for row in data:
            # Handle void scalars (numpy.void objects) or regular tuples
            if isinstance(row, void):
                # void objects are dict-like: access by field name
                col_values.append(row[name])
            else:
                # Regular tuple: access by index
                col_values.append(row[i])
        col_arr = array(col_values, dtype=field_dtype_obj)
        # Convert _ObjectArray complex columns to native ndarray so Rust
        # StructuredArray can accept them.
        if isinstance(col_arr, _ObjectArray):
            _dt = str(col_arr.dtype) if hasattr(col_arr.dtype, '__str__') else str(col_arr._dtype)
            if _dt.startswith("complex"):
                _dt_name = _dt if _dt in ("complex64", "complex128") else "complex128"
                _n = len(col_values)
                _narr = _native.zeros([_n], _dt_name)
                for _ci, _cv in enumerate(col_values):
                    if isinstance(_cv, complex):
                        _narr[_ci] = (_cv.real, _cv.imag)
                    elif isinstance(_cv, (int, float)):
                        _narr[_ci] = (float(_cv), 0.0)
                    elif isinstance(_cv, tuple) and len(_cv) == 2:
                        _narr[_ci] = (float(_cv[0]), float(_cv[1]))
                col_arr = _narr
            else:
                # Non-numeric field (bytes, str, etc.) — fall back to object array
                # Build void scalars as the records and return an _ObjectArray
                from numpy import dtype as _dtype_cls
                records = []
                for row in data:
                    if isinstance(row, void):
                        records.append(row)
                    else:
                        # Convert tuple to void scalar with field access
                        d = {}
                        for j2, nm2 in enumerate(names):
                            d[nm2] = row[j2]
                        from numpy import void as _void
                        v = _void(d, sdt)
                        records.append(v)
                result = _ObjectArray(records, dt=sdt)
                return result
        fields.append((name, col_arr))
    dtype_json = json.dumps([[nm, str(sdt.fields[nm][0])] for nm in names])
    native_fields = [(name, col._native if hasattr(col, '_native') else col)
                     for name, col in fields]
    native = _native.StructuredArray(native_fields, [nrows], dtype_json)
    return StructuredArray(native)


def _create_empty_structured(nrows, sdt, fill_value=0):
    """Create a zero/fill-filled StructuredArray of shape (nrows,).

    Args:
        nrows: int — number of records
        sdt: StructuredDtype or dtype wrapping one
        fill_value: scalar to fill each column (default 0)
    """
    import json
    from numpy import StructuredArray
    from ._core_types import StructuredDtype
    if hasattr(sdt, '_structured') and sdt._structured is not None:
        sdt = sdt._structured
    names = sdt.names
    fields = []
    for name in names:
        field_dtype_obj, _ = sdt.fields[name]
        col_arr = full(nrows, fill_value, dtype=field_dtype_obj)
        fields.append((name, col_arr))
    dtype_json = json.dumps([[nm, str(sdt.fields[nm][0])] for nm in names])
    native_fields = [(name, col._native if hasattr(col, '_native') else col)
                     for name, col in fields]
    native = _native.StructuredArray(native_fields, [nrows], dtype_json)
    return StructuredArray(native)


def array(data, dtype=None, copy=None, order=None, subok=False, ndmin=0, like=None):
    result = _array_core(data, dtype=dtype, copy=copy, order=order, subok=subok, like=like)
    if ndmin > 0 and isinstance(result, ndarray):
        while result.ndim < ndmin:
            import numpy as _np
            result = _np.expand_dims(result, 0)
    if isinstance(result, ndarray):
        if order == 'F' and hasattr(result, '_mark_fortran'):
            result._mark_fortran()
        elif order == 'C' and hasattr(result, '_mark_c_contiguous'):
            result._mark_c_contiguous()
    return result

def _array_core(data, dtype=None, copy=None, order=None, subok=False, like=None):
    # Handle chararray: unwrap to underlying ndarray (unless subok=True)
    if type(data).__name__ == 'chararray' and hasattr(data, '_arr') and not subok:
        return _array_core(data._arr, dtype=dtype, copy=copy, order=order, subok=subok, like=like)
    # Handle range objects: convert to list
    if isinstance(data, range):
        data = list(data)
    # Handle flatiter: convert to flattened array
    if type(data).__name__ == 'flatiter':
        if hasattr(data, '_arr'):
            # Python flatiter
            arr = data._arr.flatten() if data._arr is not None else ndarray._native([])
            return _array_core(arr, dtype=dtype, copy=copy, order=order, subok=subok, like=like)
        else:
            # Rust flatiter - iterate to list
            vals = list(data)
            return _array_core(vals, dtype=dtype, copy=copy, order=order, subok=subok, like=like)
    if (not isinstance(data, (ndarray, _ObjectArray, list, tuple, int, float, complex, bool, str, bytes))
            and hasattr(data, '__array_interface__')):
        converted = _array_from_array_interface(data)
        if converted is not None:
            if dtype is not None:
                converted = converted.astype(str(dtype))
            if copy:
                converted = converted.copy()
            return converted
    # Support __array__ protocol: objects with __array__() method
    if (not isinstance(data, (ndarray, _ObjectArray, list, tuple, int, float, complex, bool, str, bytes))
            and hasattr(data, '__array__')):
        try:
            converted = data.__array__(dtype=dtype, copy=copy)
        except TypeError:
            try:
                converted = data.__array__(dtype=dtype)
            except TypeError:
                converted = data.__array__()
        if isinstance(converted, (ndarray, _ObjectArray)):
            if copy:
                return converted.copy()
            return converted
        # __array__ returned something else, use it as data
        data = converted
    # Check for structured dtype BEFORE normalization (normalization loses field info)
    # Also handle list-of-tuples dtype specs like [('x', 'i4'), ('y', 'i4')]
    if dtype is not None and isinstance(dtype, list):
        from ._core_types import dtype as _dtype_cls
        parsed = _dtype_cls(dtype)
        if isinstance(data, (list, tuple)) and len(data) > 0:
            nfields = len(parsed.names) if hasattr(parsed, 'names') else 1
            def _is_record(t):
                """True if t is a leaf record: tuple/list of scalars matching field count."""
                if not isinstance(t, (list, tuple)):
                    return False
                if len(t) != nfields:
                    return False
                return all(not isinstance(v, (list, tuple)) for v in t)
            def _get_shape(d):
                """Get shape of nested structure of record tuples."""
                if _is_record(d):
                    return ()  # leaf record
                if not isinstance(d, (list, tuple)):
                    return ()
                if len(d) == 0:
                    return (0,)
                inner = _get_shape(d[0])
                return (len(d),) + inner
            def _flatten_to_records(d):
                if _is_record(d):
                    return [d]
                if isinstance(d, (list, tuple)):
                    result = []
                    for item in d:
                        result.extend(_flatten_to_records(item))
                    return result
                # scalar — wrap in tuple of all same value
                return [tuple([d] * nfields)]
            shape = _get_shape(data)
            if len(shape) == 0 or len(shape) == 1:
                # 0D or 1D: just create directly
                data_seq = data if isinstance(data, (list, tuple)) else [data]
                return _create_structured_array(data_seq, parsed)
            else:
                # Multi-dim: flatten to records, create, reshape
                flat_records = _flatten_to_records(data)
                sa_flat = _create_structured_array(flat_records, parsed)
                return sa_flat.reshape(shape)
        data_seq = data if isinstance(data, (list, tuple)) else [data]
        return _create_structured_array(data_seq, parsed)
    if dtype is not None and _is_structured_dtype(dtype):
        from ._core_types import dtype as _dtype_cls
        try:
            parsed = dtype if isinstance(dtype, _dtype_cls) else _dtype_cls(dtype)
        except Exception:
            parsed = None
        # Only route to StructuredArray when data is list-of-tuples (actual records)
        _is_records = False
        data_seq = None
        if parsed is not None and hasattr(parsed, 'names') and parsed.names:
            nfields = len(parsed.names)
            if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (tuple, list)):
                _is_records = True
                data_seq = data
            elif isinstance(data, ndarray):
                # ndarray with structured dtype — try to convert
                data_seq = [tuple(float(data[i]) for _ in parsed.names) for i in range(data.size)]
                _is_records = True
            elif isinstance(data, (list, tuple)) and len(data) == nfields and not isinstance(data[0], (list, tuple)):
                # Single record as flat tuple/list: (1, 2) with dtype='i4,i4' → [(1, 2)]
                data_seq = [tuple(data)]
                _is_records = True
        if _is_records and data_seq is not None:
            if data_seq and not isinstance(data_seq[0], (tuple, list)):
                converted = []
                for val in data_seq:
                    vals = []
                    for name in parsed.names:
                        field_dt, _ = parsed.fields[name]
                        fd = str(field_dt)
                        if fd in ('float64', 'float32', 'float16') or fd.startswith('f'):
                            vals.append(float(val))
                        elif fd in ('int8', 'int16', 'int32', 'int64') or fd.startswith('i'):
                            vals.append(int(val))
                        elif fd in ('uint8', 'uint16', 'uint32', 'uint64') or fd.startswith('u'):
                            vals.append(int(val))
                        elif fd.startswith('complex') or fd.startswith('c'):
                            vals.append(complex(val))
                        else:
                            vals.append(val)
                    converted.append(tuple(vals))
                data_seq = converted
            return _create_structured_array(data_seq, parsed)
    if dtype is not None:
        dtype = _normalize_dtype(dtype)
    elif hasattr(data, "_numpy_dtype_name"):
        # Preserve numpy scalar wrapper dtype metadata for asarray()/array().
        dtype = _normalize_dtype(str(getattr(data, "_numpy_dtype_name")))
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        # Preserve dtype when building arrays from wrapped numpy scalars.
        _ball = __import__("builtins").all
        wrapped = [getattr(x, "_numpy_dtype_name", None) for x in data]
        if _ball(w is not None for w in wrapped):
            import numpy as _np
            cur = _normalize_dtype(str(wrapped[0]))
            for w in wrapped[1:]:
                cur = str(_np.promote_types(cur, _normalize_dtype(str(w))))
            dtype = cur
            converted = []
            for x, w in zip(data, wrapped):
                wn = _normalize_dtype(str(w)) or ""
                if wn == "bool":
                    converted.append(bool(x))
                elif wn.startswith("int") or wn.startswith("uint"):
                    converted.append(int(x))
                elif wn.startswith("float"):
                    converted.append(float(x))
                elif wn.startswith("complex"):
                    converted.append(complex(x))
                else:
                    converted.append(x)
            data = converted
    if dtype is None and (_contains_fraction_values(data) or _contains_decimal_values(data)):
        if isinstance(data, _ObjectArray):
            return data.copy() if copy else data
        if isinstance(data, (list, tuple)):
            shape = _infer_shape(data)
            if shape is not None:
                flat = _flatten_object_nested(data)
                expected = 1
                for dim in shape:
                    expected *= dim
                if flat is not None and len(flat) == expected:
                    return _ObjectArray(flat, "object", shape=shape)
            return _ObjectArray(list(data), "object")
        return _ObjectArray([data], "object", shape=())
    # Check if dtype forces a non-numeric path
    if dtype is not None:
        dt = str(dtype)
        # Temporal dtypes (datetime64 / timedelta64): route to _ObjectArray
        if _is_temporal_dtype(dt):
            return _make_temporal_array(data, dt)
        # String dtypes: route to Rust native (S-prefixed, U-prefixed, "str")
        if dt.startswith("S") or dt.startswith("U") or dt == "str":
            if isinstance(data, str):
                data = [data]
            if isinstance(data, (list, tuple)):
                # Handle nested lists (2D+ string arrays)
                if len(data) > 0 and isinstance(data[0], (list, tuple)):
                    flat = [str(x) for row in data for x in row]
                    inner_len = len(data[0])
                    result = _native.array(flat)
                    return result.reshape((len(data), inner_len))
                return _native.array([str(x) for x in data])
            return _native.array(data)
        if dt in ("object", "<class 'object'>"):
            # Flatten nested list/tuple for object arrays (e.g. [[1,2],[3,4]])
            # Only flatten if ALL elements are sequences of the same length.
            if (isinstance(data, (list, tuple)) and len(data) > 0
                    and all(isinstance(row, (list, tuple)) for row in data)):
                inner_len = len(data[0])
                if all(len(row) == inner_len for row in data):
                    flat = [x for row in data for x in row]
                    shape = (len(data), inner_len)
                    return _ObjectArray(flat, "object", shape=shape)
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], "object")
        if dt == "void" or dt.startswith("V"):
            data_list = data if isinstance(data, (list, tuple)) else [data]
            kind, itemsize = _string_dtype_info(dt)
            return _make_void_result((len(data_list),), dt, 0, itemsize=itemsize or 0)
        if dt == "bytes":
            data_list = data if isinstance(data, (list, tuple)) else [data]
            return _ObjectArray([bytes(str(x), 'ascii') if not isinstance(x, bytes) else x for x in data_list], "bytes")
        # Structured dtype: route to columnar Rust-backed StructuredArray
        # Note: str(structured_dtype) returns "void", not a comma-separated string,
        # so we must check _is_structured_dtype on the original dtype object.
        if "," in dt or _is_structured_dtype(dtype):
            from ._core_types import dtype as _dtype_cls, StructuredDtype
            parsed = _dtype_cls(dtype) if not isinstance(dtype, _dtype_cls) else dtype
            if _is_structured_dtype(parsed):
                data_seq = data if isinstance(data, (list, tuple)) else [data]
                return _create_structured_array(data_seq, parsed)
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], dt)
    if isinstance(data, _ObjectArray):
        return data.copy() if copy else data
    if isinstance(data, ndarray):
        result = data.copy() if copy else data
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        # subok=False means strip subclass to base ndarray
        if not subok and type(result) is not ndarray:
            result = _native.array(result.tolist())
            if dtype is not None:
                dt = str(dtype)
                if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                    result = result.astype(dt)
            elif hasattr(data, 'dtype'):
                result = result.astype(str(data.dtype))
        # Apply order= flag
        if order == 'F' and hasattr(result, '_mark_fortran'):
            if result is data:
                result = result.copy()
            result._mark_fortran()
        elif order == 'C' and hasattr(result, '_mark_c_contiguous'):
            if result is data:
                result = result.copy()
            result._mark_c_contiguous()
        elif order in ('K', 'A'):
            result = _like_order(result, data, order)
        return result
    if isinstance(data, bool):
        # bool must be checked before int since bool is a subclass of int
        dt_name = str(dtype) if dtype is not None else 'bool'
        result = _native.full([], 1.0 if data else 0.0, 'float64').astype(dt_name)
        return result
    if isinstance(data, int):
        if dtype is None and _contains_large_python_ints(data):
            return _ObjectArray([data], "object", shape=())
        dt_name = str(dtype) if dtype is not None else 'int64'
        if dt_name in ('uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'):
            result = _native_integer_result([int(data)], dt_name).reshape([])
        else:
            result = _native.full([], float(data), 'float64').astype(dt_name)
        return result
    if isinstance(data, float):
        dt_name = str(dtype) if dtype is not None else 'float64'
        result = _native.full([], data, dt_name)
        return result
    if isinstance(data, complex):
        dt_name = str(dtype) if dtype is not None else 'complex128'
        result = _native.zeros([1], dt_name)
        result[0] = (data.real, data.imag)
        return result.reshape([])
    if isinstance(data, (int, float)):
        result = _native.array([float(data)])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    # Handle empty list/tuple before non-empty checks
    if isinstance(data, (list, tuple)) and len(data) == 0:
        result = _native.array([])  # empty float64 array
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    if isinstance(data, str):
        # Single string -> string array
        return _native.array([data])
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], str):
        # List/tuple of strings -> string array
        return _native.array(list(data) if isinstance(data, tuple) else data)
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], bool):
        result = _native.array([1.0 if x else 0.0 for x in data])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        else:
            # Auto-detect: all bools -> bool dtype
            _all = __import__("builtins").all
            if _all(isinstance(x, bool) for x in data):
                result = result.astype("bool")
        return result
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], complex):
        dt_name = str(dtype) if dtype is not None else "complex128"
        return _ObjectArray(data if isinstance(data, list) else list(data), dt_name)
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (int, float)):
        # Check if any element is complex (mixed int/float/complex list)
        _any_complex = __import__("builtins").any(isinstance(x, complex) for x in data)
        if _any_complex:
            return _ObjectArray([complex(x) for x in data], "complex128")
        if dtype is None and _contains_large_python_ints(data):
            return _ObjectArray(list(data), "object")
        dt_name = str(dtype) if dtype is not None else None
        if dt_name in ('uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'):
            return _native_integer_result([int(x) for x in data], dt_name)
        if dtype is None and not __import__("builtins").any(isinstance(x, float) for x in data):
            return _native_integer_result([int(x) for x in data], 'int64')
        result = _native.array([float(x) for x in data])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    # Nested lists/tuples: infer shape, flatten, reshape
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple, ndarray, _ObjectArray)):
        shape = _infer_shape(data)
        if shape is not None:
            # First try: check for complex elements before flattening
            _any = __import__("builtins").any
            _builtins = __import__("builtins")

            def _flatten_complex(d):
                """Flatten nested lists including complex values."""
                if isinstance(d, _ObjectArray):
                    flat = d.flatten().tolist()
                    result = []
                    for v in flat:
                        if isinstance(v, complex):
                            result.append(v)
                        elif isinstance(v, tuple):
                            result.append(complex(v[0], v[1]))
                        else:
                            try:
                                result.append(complex(float(v), 0.0))
                            except (TypeError, ValueError):
                                return None
                    return result
                if isinstance(d, ndarray):
                    flat = d.flatten().tolist()
                    result = []
                    for v in flat:
                        if isinstance(v, tuple):
                            result.append(complex(v[0], v[1]))
                        elif isinstance(v, complex):
                            result.append(v)
                        else:
                            result.append(complex(float(v), 0.0))
                    return result
                if isinstance(d, (list, tuple)):
                    result = []
                    for item in d:
                        sub = _flatten_complex(item)
                        if sub is None:
                            return None
                        result.extend(sub)
                    return result
                if isinstance(d, complex):
                    return [d]
                if isinstance(d, (int, float, bool)):
                    return [complex(d, 0.0)]
                try:
                    return [complex(float(d), 0.0)]
                except (TypeError, ValueError):
                    return None

            # Check if any nested element is complex
            def _has_any_complex(d):
                if isinstance(d, complex):
                    return True
                if isinstance(d, _ObjectArray):
                    dt = str(d.dtype)
                    return dt.startswith("complex") or dt.startswith("c")
                if isinstance(d, ndarray):
                    dt = str(d.dtype)
                    return dt.startswith("complex") or dt.startswith("c")
                if isinstance(d, (list, tuple)):
                    return _any(_has_any_complex(x) for x in d)
                return False

            _is_complex_data = _has_any_complex(data)
            _want_complex = (dtype is not None and str(dtype).startswith("complex"))

            if _is_complex_data or _want_complex:
                flat_c = _flatten_complex(data)
                if flat_c is not None:
                    reals = [x.real for x in flat_c]
                    imags = [x.imag for x in flat_c]
                    re_arr = _native.array(reals).reshape(list(shape)).astype("complex128")
                    im_arr = _native.array(imags).reshape(list(shape)).astype("complex128")
                    _j_arr = _native.zeros([1], "complex128")
                    _j_arr[0] = (0.0, 1.0)
                    result = re_arr + im_arr * _j_arr.reshape([])
                    if dtype is not None:
                        dt = str(dtype)
                        if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                            result = result.astype(dt)
                    return result

            # Check if original data is pure integers (before _flatten_nested converts to float)
            def _is_all_ints_nested(d):
                if isinstance(d, (list, tuple)):
                    return all(_is_all_ints_nested(x) for x in d)
                return isinstance(d, int) and not isinstance(d, bool)
            _orig_all_ints = _is_all_ints_nested(data)
            flat = _flatten_nested(data)
            if flat is not None:
                # Detect source dtype from nested arrays for int preservation
                _src_dtype = None
                if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], ndarray):
                    _src_dtype = str(data[0].dtype)
                # Rust native array requires floats; convert ints before passing
                if _orig_all_ints and dtype is None:
                    result = _native_integer_result([int(x) for x in flat], "int64").reshape(shape)
                elif dtype is not None:
                    if flat and isinstance(flat[0], int) and not isinstance(flat[0], bool):
                        flat = [float(x) for x in flat]
                    result = _native.array(flat)
                    result = result.reshape(shape)
                    dt = str(dtype)
                    if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                        result = result.astype(dt)
                else:
                    if flat and isinstance(flat[0], int) and not isinstance(flat[0], bool):
                        flat = [float(x) for x in flat]
                    result = _native.array(flat)
                    result = result.reshape(shape)
                if not (_orig_all_ints and dtype is None) and _all_bools_nested(data):
                    result = result.astype("bool")
                elif _src_dtype and (_src_dtype.startswith("int") or _src_dtype.startswith("uint")):
                    result = result.astype(_src_dtype)
                return result
    # Try the native array constructor
    try:
        result = _native.array(data)
    except (TypeError, ValueError):
        try:
            result = _native.array(_to_float_list(data))
        except (TypeError, ValueError):
            # Final fallback for non-numeric data
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data])
    if dtype is not None and isinstance(result, ndarray):
        dt = str(dtype)
        if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
            result = result.astype(dt)
    return result


def zeros(shape, dtype=None, order="C", like=None):
    if isinstance(shape, int):
        shape = (shape,)
    # Handle structured dtype
    if dtype is not None:
        from ._core_types import dtype as _dtype_cls
        parsed = _dtype_cls(dtype) if not isinstance(dtype, _dtype_cls) else dtype
        if _is_structured_dtype(parsed):
            if isinstance(shape, (list, tuple)) and len(shape) > 1:
                nrows = 1
                for s in shape:
                    nrows *= s
                arr = _create_empty_structured(nrows, parsed, fill_value=0)
                return arr.reshape(tuple(shape))
            nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
            return _create_empty_structured(nrows, parsed, fill_value=0)
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else None
    if dt is not None and _is_temporal_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_make_temporal_array([0] * n, dt).reshape(shape), order)
    if dt in ("object", "<class 'object'>"):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([0] * n, "object", shape=shape), order)
    if dt is not None and _unsupported_numeric_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        kind, itemsize = _string_dtype_info(dt)
        if kind == 'void':
            return _apply_order(_make_void_result(shape, dt, 0, itemsize=itemsize), order)
        fill = '' if kind == 'str' else b'' if kind == 'bytes' else 0
        return _apply_order(_ObjectArray([fill] * n, dt, shape=shape, itemsize=itemsize), order)
    if dt is not None:
        return _apply_order(_native.zeros(shape, dt), order)
    return _apply_order(_native.zeros(shape), order)

def ones(shape, dtype=None, order="C", like=None):
    if isinstance(shape, int):
        shape = (shape,)
    # Handle structured dtype
    if dtype is not None:
        from ._core_types import dtype as _dtype_cls
        parsed = _dtype_cls(dtype) if not isinstance(dtype, _dtype_cls) else dtype
        if _is_structured_dtype(parsed):
            if isinstance(shape, (list, tuple)) and len(shape) > 1:
                nrows = 1
                for s in shape:
                    nrows *= s
                arr = _create_empty_structured(nrows, parsed, fill_value=1)
                return arr.reshape(tuple(shape))
            nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
            return _create_empty_structured(nrows, parsed, fill_value=1)
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else None
    if dt is not None and _is_temporal_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_make_temporal_array([1] * n, dt).reshape(shape), order)
    if dt in ("object", "<class 'object'>"):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([1] * n, "object", shape=shape), order)
    if dt is not None and _unsupported_numeric_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        kind, itemsize = _string_dtype_info(dt)
        if kind == 'void':
            return _apply_order(_make_void_result(shape, dt, 1, itemsize=itemsize), order)
        fill = '1' if kind == 'str' else b'1' if kind == 'bytes' else 1
        return _apply_order(_ObjectArray([fill] * n, dt, shape=shape, itemsize=itemsize), order)
    if dt is not None:
        return _apply_order(_native.ones(shape, dt), order)
    return _apply_order(_native.ones(shape), order)

def arange(*args, dtype=None, like=None, **kwargs):
    if dtype is datetime64:
        dt = "datetime64[D]"
    elif dtype is timedelta64:
        dt = "timedelta64[D]"
    else:
        dt = _normalize_dtype_with_size(dtype) if dtype is not None else None
    if dt in ("<class 'numpy.datetime64'>", "datetime64"):
        dt = "datetime64[D]"
    elif dt in ("<class 'numpy.timedelta64'>", "timedelta64"):
        dt = "timedelta64[D]"
    if dt == "object" or dt == "<class 'object'>":
        float_args = [float(a) for a in args]
        if len(float_args) == 1:
            vals = list(range(int(float_args[0])))
        elif len(float_args) == 2:
            vals = list(range(int(float_args[0]), int(float_args[1])))
        else:
            vals = list(range(int(float_args[0]), int(float_args[1]), int(float_args[2])))
        return _ObjectArray(vals, "object")
    if dt is not None and _is_temporal_dtype(dt):
        kind, unit, _, _ = _temporal_dtype_info(dt)
        step_dt = "timedelta64[{}]".format(unit)

        def _temporal_int(value, cast_dt):
            if isinstance(value, (str, bytes)) or hasattr(value, '_numpy_dtype_name'):
                return int(array([value], dtype=cast_dt).astype('int64')[0])
            return int(float(value))

        if len(args) == 1:
            start_i = 0
            stop_i = _temporal_int(args[0], dt)
            step_i = 1
        elif len(args) == 2:
            start_i = _temporal_int(args[0], dt)
            stop_i = _temporal_int(args[1], dt)
            step_i = 1
        else:
            start_i = _temporal_int(args[0], dt)
            stop_i = _temporal_int(args[1], dt)
            step_i = _temporal_int(args[2], step_dt)
        vals = list(range(start_i, stop_i, step_i))
        return _make_temporal_array(vals, dt)
    # Detect if all args are integers (for integer output dtype)
    _all_int = all(isinstance(a, (int, bool)) and not isinstance(a, float) for a in args)
    # Also check for numpy int scalars
    if not _all_int:
        _all_int = all(
            (isinstance(a, int) and not isinstance(a, float)) or
            (hasattr(a, '_numpy_dtype_name') and a._numpy_dtype_name in
             ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'))
            for a in args
        )
    float_args = [float(a) for a in args]
    # Normalize to (start, stop, step) form
    if len(float_args) == 1:
        float_args = [0.0, float_args[0], 1.0]
    elif len(float_args) == 2:
        float_args = [float_args[0], float_args[1], 1.0]
    if dtype is not None:
        result = _native.arange(float_args[0], float_args[1], float_args[2], dt)
    else:
        result = _native.arange(float_args[0], float_args[1], float_args[2])
    # Fix arange length for floating-point steps (match NumPy behavior)
    # NumPy uses ceil((stop - start) / step) to determine length
    if not _all_int and float_args[2] != 0:
        import math
        expected_len = max(0, int(math.ceil((float_args[1] - float_args[0]) / float_args[2])))
        if len(result) > expected_len:
            result = result[:expected_len]
    # If all inputs were integers and no explicit dtype, cast to int64
    if _all_int and dt is None:
        result = result.astype('int64')
    return result

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    from ._helpers import _ObjectArray
    # Validate num: must be integer type, non-negative
    if isinstance(num, float):
        if num != int(num):
            raise TypeError(
                "object of type 'float' cannot be safely interpreted as an integer"
            )
        num = int(num)
    num = int(num)
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)
    # Extract scalars from 0-d arrays
    if isinstance(start, ndarray) and start.ndim == 0:
        start = start.flat[0]
        if isinstance(start, tuple) and len(start) == 2:
            start = complex(start[0], start[1])
    if isinstance(stop, ndarray) and stop.ndim == 0:
        stop = stop.flat[0]
        if isinstance(stop, tuple) and len(stop) == 2:
            stop = complex(stop[0], stop[1])
    # Check if start or stop are array-like (not scalar)
    _start_is_array = isinstance(start, (ndarray, _ObjectArray, list, tuple))
    _stop_is_array = isinstance(stop, (ndarray, _ObjectArray, list, tuple))
    if _start_is_array or _stop_is_array:
        import numpy as _np
        start_arr = _np.asarray(start, dtype='float64')
        stop_arr = _np.asarray(stop, dtype='float64')
        if num == 0:
            shape = list(start_arr.shape) if start_arr.ndim > 0 else []
            result = _np.array([], dtype=dtype or 'float64').reshape([0] + shape)
            if retstep:
                return result, _np.nan
            return result
        if num == 1:
            result = _np.expand_dims(start_arr.copy(), axis=axis)
            if dtype is not None:
                result = result.astype(str(dtype))
            step = _np.nan
            if retstep:
                return result, step
            return result
        # Broadcast start and stop to common shape
        try:
            _bcast = start_arr + stop_arr * 0  # force broadcast
            start_arr = start_arr + _bcast * 0
            stop_arr = stop_arr + _bcast * 0
        except Exception:
            pass
        if endpoint:
            step = (stop_arr - start_arr) / (num - 1)
        else:
            step = (stop_arr - start_arr) / num
        # Build the result: each point is start + i * step
        vals = []
        for i in range(num):
            vals.append(start_arr + i * step)
        # If endpoint, force last value to be exactly stop
        if endpoint:
            vals[-1] = stop_arr.copy()
        result = _np.stack(vals, axis=axis)
        if dtype is not None:
            _is_int = dtype is int or (isinstance(dtype, type) and issubclass(dtype, int))
            if not _is_int:
                try:
                    _is_int = _np.issubdtype(dtype, _np.integer)
                except Exception:
                    pass
            if _is_int:
                result = _np.floor(result)
            result = result.astype(str(dtype))
        if retstep:
            return result, step
        return result
    # Handle complex scalars
    if isinstance(start, complex) or isinstance(stop, complex):
        start_c = complex(start)
        stop_c = complex(stop)
        if num == 0:
            result = array([], dtype=dtype or 'complex128')
            if retstep:
                return result, complex(float('nan'), float('nan'))
            return result
        if endpoint:
            step = (stop_c - start_c) / (num - 1) if num > 1 else 0
        else:
            step = (stop_c - start_c) / num if num > 0 else 0
        vals = [start_c + i * step for i in range(num)]
        if endpoint and num > 1:
            vals[-1] = stop_c
        result = array(vals)
        if dtype is not None:
            result = result.astype(str(dtype))
        if retstep:
            return result, step
        return result
    start = float(start)
    stop = float(stop)
    if num == 0:
        result = array([], dtype=dtype or 'float64')
        if retstep:
            return result, float('nan')
        return result
    if endpoint:
        if num == 1:
            result = array([start])
            step = float('nan')
            if dtype is not None:
                result = result.astype(str(dtype))
            if retstep:
                return result, step
            return result
        result = _native.linspace(start, stop, num)
        step = (stop - start) / (num - 1)
    else:
        step = (stop - start) / num
        result = array([start + i * step for i in range(num)])
    if dtype is not None:
        import numpy as _np
        _is_int = dtype is int or (isinstance(dtype, type) and issubclass(dtype, int))
        if not _is_int:
            try:
                _is_int = _np.issubdtype(dtype, _np.integer)
            except Exception:
                pass
        if _is_int:
            result = _np.floor(result)
        result = result.astype(str(dtype))
    if retstep:
        return result, step
    return result

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    import numpy as _np
    _base_is_array = not isinstance(base, (float, int)) and _np.asarray(base).ndim > 0
    if _base_is_array:
        _ndmax = max(_np.asarray(start).ndim, _np.asarray(stop).ndim, _np.asarray(base).ndim)
        if _ndmax > 0:
            start = _np.asarray(start)
            stop = _np.asarray(stop)
            base = _np.asarray(base)
            if start.ndim < _ndmax:
                start = start.reshape((1,) * (_ndmax - start.ndim) + start.shape)
            if stop.ndim < _ndmax:
                stop = stop.reshape((1,) * (_ndmax - stop.ndim) + stop.shape)
            if base.ndim < _ndmax:
                base = base.reshape((1,) * (_ndmax - base.ndim) + base.shape)
            base = _np.expand_dims(base, axis=axis)
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    result = _np.power(base, y)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    import cmath as _cmath
    from ._helpers import _ObjectArray
    # Extract scalars from 0-d arrays
    if isinstance(start, ndarray) and start.ndim == 0:
        start = start.flat[0]
        if isinstance(start, tuple) and len(start) == 2:
            start = complex(start[0], start[1])
    if isinstance(stop, ndarray) and stop.ndim == 0:
        stop = stop.flat[0]
        if isinstance(stop, tuple) and len(stop) == 2:
            stop = complex(stop[0], stop[1])
    # Check if start or stop are array-like
    _start_is_array = isinstance(start, (ndarray, _ObjectArray, list, tuple))
    _stop_is_array = isinstance(stop, (ndarray, _ObjectArray, list, tuple))
    if _start_is_array or _stop_is_array:
        import numpy as _np
        # Element-wise geomspace for array inputs
        start_arr = _np.asarray(start)
        stop_arr = _np.asarray(stop)
        # Broadcast to common shape
        try:
            _bcast = start_arr + stop_arr * 0
            start_arr = start_arr + _bcast * 0
            stop_arr = stop_arr + _bcast * 0
        except Exception:
            pass
        s_flat = start_arr.flatten().tolist()
        e_flat = stop_arr.flatten().tolist()
        # Convert numpy scalars to Python builtins
        def _to_py(v):
            if isinstance(v, complex):
                return complex(v)
            try:
                return complex(v)
            except (TypeError, ValueError):
                return float(v)
        s_flat = [_to_py(v) for v in s_flat]
        e_flat = [_to_py(v) for v in e_flat]
        # Compute element-wise using scalar geomspace
        cols = []
        for sv, ev in zip(s_flat, e_flat):
            col = geomspace(sv, ev, num=num, endpoint=endpoint)
            cols.append(col)
        result = _np.stack(cols, axis=1 if axis == 0 else 0)
        if dtype is not None:
            result = result.astype(str(dtype))
        return result
    # Handle negative values by working in complex domain if needed
    # Also use complex path if dtype is complex (even for real start/stop with sign change)
    _use_complex = isinstance(start, complex) or isinstance(stop, complex) or hasattr(start, 'imag') and start.imag != 0 or hasattr(stop, 'imag') and stop.imag != 0
    if not _use_complex and dtype is not None:
        _dt_str = str(dtype)
        if 'complex' in _dt_str or dtype is complex:
            _use_complex = True
    if _use_complex:
        # Complex geomspace
        start_c = complex(start)
        stop_c = complex(stop)
        if num == 0:
            import numpy as _np
            return _np.array([], dtype=dtype or 'complex128')
        if num == 1:
            import numpy as _np
            return _np.array([start_c], dtype=dtype or 'complex128')
        # Manual geomspace for complex
        ratio = (stop_c / start_c) ** (1.0 / (num - 1 if endpoint else num))
        vals = [start_c * (ratio ** i) for i in range(num)]
        if endpoint:
            vals[-1] = stop_c  # ensure exact endpoint
        import numpy as _np
        return _np.array(vals, dtype=dtype or 'complex128')
    start_f = float(start)
    stop_f = float(stop)
    if start_f == 0 or stop_f == 0:
        raise ValueError("Geometric sequence cannot include zero")
    both_negative = start_f < 0 and stop_f < 0
    sign_change = (start_f < 0) != (stop_f < 0)
    if sign_change:
        # Sign change: compute via complex domain, then take real part
        # This produces NaN for interior points (matching numpy behavior)
        import numpy as _np
        start_c = complex(start_f)
        stop_c = complex(stop_f)
        if num == 0:
            result = _np.array([], dtype='float64')
        elif num == 1:
            result = _np.array([start_f])
        else:
            ratio = (stop_c / start_c) ** (1.0 / (num - 1 if endpoint else num))
            vals = []
            for i in range(num):
                c = start_c * (ratio ** i)
                # If imaginary part is non-negligible, the value is NaN in real domain
                if abs(c.imag) > abs(c.real) * 1e-15:
                    vals.append(float('nan'))
                else:
                    vals.append(c.real)
            if endpoint:
                vals[-1] = stop_f
            vals[0] = start_f
            result = _np.array(vals)
        if dtype is not None:
            result = result.astype(str(dtype))
        return result
    if both_negative:
        # Both negative: compute for abs values, then negate
        log_start = _math.log10(-start_f)
        log_stop = _math.log10(-stop_f)
        result = logspace(log_start, log_stop, num=num, endpoint=endpoint)
        import numpy as _np
        result = -result
    else:
        log_start = _math.log10(start_f)
        log_stop = _math.log10(stop_f)
        result = logspace(log_start, log_stop, num=num, endpoint=endpoint)
    # Force exact start/stop boundaries
    import numpy as _np
    if num > 0:
        flat = result.flatten().tolist()
        flat[0] = start_f
        if endpoint and num > 1:
            flat[-1] = stop_f
        result = _np.array(flat).reshape(result.shape)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def eye(N, M=None, k=0, dtype=None, order="C", like=None):
    if M is None:
        M = N
    dt_str = str(dtype) if dtype is not None else None
    # Handle string/bytes dtypes that Rust doesn't support
    if dt_str is not None and (dt_str.startswith('S') or dt_str.startswith('|S') or
                                dt_str.startswith('U') or dt_str.startswith('<U') or
                                dt_str.startswith('>U') or dt_str == 'bytes' or dt_str == 'str'):
        from ._helpers import _ObjectArray
        # Determine the "one" and "zero" values for the dtype
        if 'S' in dt_str or dt_str == 'bytes':
            one, zero = b'1', b''
        else:
            one, zero = '1', ''
        data = []
        for i in range(N):
            for j in range(M):
                data.append(one if j - i == k else zero)
        return _ObjectArray(data, dt_str, shape=(N, M))
    try:
        if dt_str is not None:
            result = _native.eye(N, M, k, dt_str)
        else:
            result = _native.eye(N, M, k)
    except (TypeError, ValueError):
        # Fallback for unsupported dtypes
        result = _native.eye(N, M, k)
        if dt_str is not None:
            result = result.astype(dt_str)
    if order == 'F' and hasattr(result, '_mark_fortran'):
        result._mark_fortran()
    return result

def where(condition, x=None, y=None):
    if x is None and y is None:
        import numpy as _np
        return _np.nonzero(condition)
    if not isinstance(condition, ndarray):
        condition = asarray(condition)
    # Ensure condition is boolean dtype (native where_ requires bool)
    if str(condition.dtype) != "bool":
        condition = condition.astype("bool")
    from ._helpers import _ObjectArray
    # Handle _ObjectArray inputs — element-wise selection
    if isinstance(x, _ObjectArray) or isinstance(y, _ObjectArray):
        cond_flat = condition.flatten().tolist()
        x_arr = asarray(x) if not isinstance(x, _ObjectArray) else x
        y_arr = asarray(y) if not isinstance(y, _ObjectArray) else y
        x_flat = x_arr._data if isinstance(x_arr, _ObjectArray) else x_arr.flatten().tolist()
        y_flat = y_arr._data if isinstance(y_arr, _ObjectArray) else y_arr.flatten().tolist()
        # Broadcast scalar-like
        if len(x_flat) == 1 and len(cond_flat) > 1:
            x_flat = x_flat * len(cond_flat)
        if len(y_flat) == 1 and len(cond_flat) > 1:
            y_flat = y_flat * len(cond_flat)
        result_data = [xv if c else yv for c, xv, yv in zip(cond_flat, x_flat, y_flat)]
        dt = x_arr._dtype if isinstance(x_arr, _ObjectArray) else (y_arr._dtype if isinstance(y_arr, _ObjectArray) else 'object')
        result = _ObjectArray(result_data, dt)
        if len(condition.shape) > 1:
            result._shape = tuple(condition.shape)
        return result
    if not isinstance(x, ndarray):
        x = full(condition.shape, float(x))
    if not isinstance(y, ndarray):
        y = full(condition.shape, float(y))
    return _native.where_(condition, x, y)

def empty(shape, dtype=None, order="C"):
    """Stub: returns zeros instead of uninitialized."""
    return zeros(shape, dtype=dtype, order=order)

def _like_order(arr, source, order):
    """Apply ordering to result of a like function. K=keep source order, A=fortran if source is, else C."""
    import numpy as _np
    src_flags = getattr(source, 'flags', None)
    src_is_f = getattr(src_flags, 'f_contiguous', False)
    src_is_c = getattr(src_flags, 'c_contiguous', True)
    if order == 'F':
        if hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
    elif order in ('K', 'A'):
        # Only use Fortran layout if source is *strictly* F-order (not also C-contiguous).
        # 0-d and 1-D arrays are always both C and F, so they default to C.
        if src_is_f and not src_is_c and hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
        elif order == 'K' and not src_is_c and not src_is_f:
            # Non-contiguous source with order='K': match the stride ordering.
            # Applies when both have the same ndim (even if shapes differ).
            src_strides = getattr(source, 'strides', None)
            arr_ndim = len(arr.shape) if hasattr(arr, 'shape') else 0
            src_ndim = getattr(source, 'ndim', 0)
            if (isinstance(arr, ndarray) and src_strides is not None
                    and arr_ndim == src_ndim and arr_ndim > 1):
                ndim = arr_ndim
                # Sort axes by descending absolute stride of source (outermost first)
                axes_by_stride = sorted(range(ndim), key=lambda i: -abs(src_strides[i]))
                perm_shape = tuple(arr.shape[i] for i in axes_by_stride)
                inv_perm = [0] * ndim
                for i, ax in enumerate(axes_by_stride):
                    inv_perm[ax] = i
                fill = arr.flat[0] if arr.size > 0 else 0
                new_arr = _np.full(perm_shape, fill, dtype=arr.dtype)
                arr = new_arr.transpose(inv_perm)
    # order='C' is the default (no fortran)
    return arr

def _detect_builtin_str_bytes(dtype):
    """Return ('str', 'bytes', or None) if dtype is Python builtin str or bytes."""
    import builtins
    if dtype is builtins.str:
        return 'str'
    if dtype is builtins.bytes:
        return 'bytes'
    return None

def _make_str_bytes_result(shape, dtype_kind, fill_value=None, itemsize=None):
    """Create an _ObjectArray with correct strides for dtype=str or dtype=bytes."""
    # itemsize: Unicode char = 4 bytes, byte = 1 byte
    if itemsize is None:
        itemsize = 4 if dtype_kind == 'str' else 1
    n = 1
    for s in shape:
        n *= s
    data = [fill_value if fill_value is not None else ''] * n
    dt_name = 'str' if dtype_kind == 'str' else 'bytes'
    return _ObjectArray(data, dt_name, shape=shape, itemsize=itemsize)

def _make_void_result(shape, dtype_str, fill_value=0, itemsize=None):
    """Create an _ObjectArray for void/record dtypes (no field support)."""
    n = 1
    for s in shape:
        n *= s
    data = [_NumpyVoidScalar(fill_value)] * n
    return _ObjectArray(data, dtype_str, shape=shape, itemsize=itemsize)

def full(shape, fill_value, dtype=None, order="C"):
    if isinstance(shape, int):
        shape = (shape,)
    # Handle structured dtype
    if dtype is not None:
        from ._core_types import dtype as _dtype_cls
        parsed = _dtype_cls(dtype) if not isinstance(dtype, _dtype_cls) else dtype
        if _is_structured_dtype(parsed):
            if isinstance(shape, (list, tuple)) and len(shape) > 1:
                nrows = 1
                for s in shape:
                    nrows *= s
                arr = _create_empty_structured(nrows, parsed, fill_value=fill_value)
                return arr.reshape(tuple(shape))
            nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
            return _create_empty_structured(nrows, parsed, fill_value=fill_value)
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else None
    if dt in ("object", "<class 'object'>"):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([fill_value] * n, "object", shape=shape), order)
    if dt is not None and _unsupported_numeric_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        kind, itemsize = _string_dtype_info(dt)
        if kind == 'void':
            return _apply_order(_make_void_result(shape, dt, fill_value, itemsize=itemsize), order)
        # Coerce fill_value to appropriate type for bytes/str dtypes
        if kind == 'bytes' and not isinstance(fill_value, (bytes, str)):
            fill_val = str(fill_value)
        elif kind == 'str' and not isinstance(fill_value, str):
            fill_val = str(fill_value)
        else:
            fill_val = fill_value
        return _apply_order(_ObjectArray([fill_val] * n, dt, shape=shape, itemsize=itemsize), order)
    if dt is not None:
        return _apply_order(_native.full(shape, float(fill_value), dt), order)
    # If fill_value is not numeric (e.g. None, object), use object dtype
    if not isinstance(fill_value, (int, float, complex, bool)):
        n = 1
        for s in (shape if isinstance(shape, (list, tuple)) else (shape,)):
            n *= s
        return _apply_order(_ObjectArray([fill_value] * n, "object", shape=shape if isinstance(shape, tuple) else tuple(shape)), order)
    return _apply_order(_native.full(shape, float(fill_value)), order)

def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    s = tuple(shape) if shape is not None else a.shape
    # Check for invalid dtype like "S-1"
    if isinstance(dtype, str) and dtype.startswith('S') and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    # Detect Python builtin str/bytes
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk, fill_value), a, order)
    # Determine effective dtype
    src_dt = _normalize_dtype_with_size(a.dtype) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else src_dt
    kind, itemsize = _string_dtype_info(dt)
    if kind == 'void':
        return _like_order(_make_void_result(s, dt, fill_value, itemsize=itemsize), a, order)
    if kind is not None:
        return _like_order(_make_str_bytes_result(s, kind, fill_value, itemsize=itemsize), a, order)
    # OverflowError: check if fill_value (plain Python int) fits in target integer dtype
    if type(fill_value) is int:
        _int_dtypes_info = {
            'int8': (-128, 127), 'int16': (-32768, 32767),
            'int32': (-2147483648, 2147483647), 'int64': (-9223372036854775808, 9223372036854775807),
            'uint8': (0, 255), 'uint16': (0, 65535), 'uint32': (0, 4294967295),
            'uint64': (0, 18446744073709551615),
        }
        if dt in _int_dtypes_info:
            lo, hi = _int_dtypes_info[dt]
            if fill_value < lo or fill_value > hi:
                raise OverflowError("Python integer {} out of bounds for {}".format(fill_value, dt))
    arr = _native.full(s, float(fill_value), dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def zeros_like(a, dtype=None, order="K", subok=True, shape=None):
    s = tuple(shape) if shape is not None else a.shape
    if isinstance(dtype, str) and dtype.startswith('S') and len(dtype) > 1 and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk), a, order)
    src_dt = _normalize_dtype_with_size(a.dtype) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else src_dt
    kind, itemsize = _string_dtype_info(dt)
    if kind == 'void':
        return _like_order(_make_void_result(s, dt, 0, itemsize=itemsize), a, order)
    if kind is not None:
        return _like_order(_make_str_bytes_result(s, kind, itemsize=itemsize), a, order)
    arr = _native.zeros(s, dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def ones_like(a, dtype=None, order="K", subok=True, shape=None):
    s = tuple(shape) if shape is not None else a.shape
    if isinstance(dtype, str) and dtype.startswith('S') and len(dtype) > 1 and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk, 1), a, order)
    src_dt = _normalize_dtype_with_size(a.dtype) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else src_dt
    kind, itemsize = _string_dtype_info(dt)
    if kind == 'void':
        return _like_order(_make_void_result(s, dt, 1, itemsize=itemsize), a, order)
    if kind is not None:
        return _like_order(_make_str_bytes_result(s, kind, 1, itemsize=itemsize), a, order)
    arr = _native.ones(s, dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def empty_like(a, dtype=None, order="K", subok=True, shape=None):
    if not hasattr(a, 'shape'):
        a = asarray(a)
    s = tuple(shape) if shape is not None else a.shape
    if isinstance(dtype, str) and dtype.startswith('S') and len(dtype) > 1 and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk), a, order)
    src_dt = _normalize_dtype_with_size(a.dtype) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else src_dt
    kind, itemsize = _string_dtype_info(dt)
    if kind == 'void':
        return _like_order(_make_void_result(s, dt, 0, itemsize=itemsize), a, order)
    if kind is not None:
        return _like_order(_make_str_bytes_result(s, kind, itemsize=itemsize), a, order)
    arr = _native.zeros(s, dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def asarray(a, dtype=None, order=None):
    from numpy import StructuredArray
    if isinstance(a, ndarray):
        if dtype is not None:
            return a.astype(str(dtype))
        return a
    if isinstance(a, StructuredArray):
        return a
    # MaskedArray → extract underlying data via filled()
    if hasattr(a, 'filled') and hasattr(a, 'mask') and hasattr(a, 'data'):
        result = a.filled()
        if dtype is not None:
            return result.astype(str(dtype))
        return result
    return array(a, dtype=dtype)

asanyarray = asarray  # In our implementation, same as asarray

def ascontiguousarray(a, dtype=None):
    from numpy import StructuredArray
    if isinstance(a, StructuredArray):
        return a
    return asarray(a)

def copy(a, order="K", subok=False):
    from numpy.ma import MaskedArray as _MA
    if subok and isinstance(a, _MA):
        return a.copy()
    if isinstance(a, _MA):
        # subok=False: return base ndarray
        return array(a.data if hasattr(a, 'data') else a, copy=True, order=order)
    return array(a, copy=True, order=order, subok=subok)

def fromiter(iterable, dtype=None, count=-1):
    if dtype is not None:
        _dt = _normalize_dtype(str(dtype))
        if _dt in ("str", "bytes") or str(dtype) in ("S", "S0", "V0", "U0"):
            raise ValueError("Must specify length when using variable-length dtypes")
    subarray_len = None
    if dtype is not None:
        # Minimal support for subarray dtype validation, e.g. dtype((int, 2)).
        if isinstance(dtype, tuple) and len(dtype) == 2 and isinstance(dtype[1], int):
            subarray_len = int(dtype[1])
        else:
            _dtype_name = str(getattr(dtype, "name", dtype))
            if _dtype_name.startswith("(<class '") and _dtype_name.endswith(")"):
                _comma = _dtype_name.rfind(",")
                if _comma != -1:
                    try:
                        subarray_len = int(_dtype_name[_comma + 1 : -1].strip())
                    except (TypeError, ValueError):
                        subarray_len = None
    if count > 0:
        data = []
        for val in iterable:
            data.append(val)
            if len(data) >= count:
                break
        if len(data) < count:
            raise ValueError(
                "iterator too short: Expected %d but iterator had only %d items." % (count, len(data))
            )
    else:
        data = list(iterable)
    if subarray_len is not None:
        for val in data:
            if not isinstance(val, (list, tuple)):
                raise ValueError("setting an array element with a sequence")
            if len(val) != subarray_len:
                raise ValueError("setting an array element with a sequence")
    import numpy as _np
    return _np.array(data, dtype=dtype)

def fromstring(string, dtype=None, count=-1, sep=''):
    """Parse array from a string."""
    if sep:
        parts = string.split(sep)
        data = [float(p.strip()) for p in parts if p.strip()]
    else:
        data = [float(x) for x in string.split()]
    if count > 0:
        data = data[:count]
    return array(data, dtype=dtype)

def array_equal(a1, a2, equal_nan=False):
    """True if two arrays have the same shape and elements."""
    try:
        # Handle StructuredArray comparison field-by-field
        import numpy as _np_mod
        if isinstance(a1, _np_mod.StructuredArray) or isinstance(a2, _np_mod.StructuredArray):
            if not (isinstance(a1, _np_mod.StructuredArray) and isinstance(a2, _np_mod.StructuredArray)):
                return False
            if a1.shape != a2.shape:
                return False
            if a1.dtype != a2.dtype:
                return False
            for name in a1.dtype.names:
                col1 = a1[name]
                col2 = a2[name]
                if not array_equal(col1, col2, equal_nan=equal_nan):
                    return False
            return True
        if not isinstance(a1, (ndarray, _ObjectArray)):
            a1 = array(a1)
        if not isinstance(a2, (ndarray, _ObjectArray)):
            a2 = array(a2)
        if a1.shape != a2.shape:
            return False
        if equal_nan:
            try:
                import numpy as _np
                both_nan = _np.logical_and(_np.isnan(a1), _np.isnan(a2))
                neither_nan = _np.logical_and(_np.logical_not(_np.isnan(a1)), _np.logical_not(_np.isnan(a2)))
                return bool(_np.all(_np.logical_or(both_nan, _np.logical_and(neither_nan, a1 == a2))))
            except Exception:
                return bool((a1 == a2).all())
        try:
            import numpy as _np
            has_nan = bool(_np.any(_np.logical_or(_np.isnan(a1), _np.isnan(a2))))
        except Exception:
            has_nan = False
        if has_nan:
            return False
        return bool((a1 == a2).all())
    except Exception:
        return False

def array_equiv(a1, a2):
    """True if arrays are shape-consistent and element-wise equal, with broadcasting."""
    a1 = asarray(a1)
    a2 = asarray(a2)
    try:
        import numpy as _np
        bshape = _np.broadcast_shapes(a1.shape, a2.shape)
        b1 = _np.broadcast_to(a1, bshape)
        b2 = _np.broadcast_to(a2, bshape)
        return bool(_np.all(b1 == b2))
    except Exception:
        return False

def require(a, dtype=None, requirements=None):
    """Return an ndarray that satisfies the given requirements."""
    _flag_alias = {
        'C': 'C_CONTIGUOUS', 'C_CONTIGUOUS': 'C_CONTIGUOUS', 'CONTIGUOUS': 'C_CONTIGUOUS',
        'F': 'F_CONTIGUOUS', 'F_CONTIGUOUS': 'F_CONTIGUOUS', 'FORTRAN': 'F_CONTIGUOUS',
        'A': 'ALIGNED', 'ALIGNED': 'ALIGNED',
        'W': 'WRITEABLE', 'WRITEABLE': 'WRITEABLE',
        'O': 'OWNDATA', 'OWNDATA': 'OWNDATA',
        'E': 'ENSUREARRAY', 'ENSUREARRAY': 'ENSUREARRAY',
    }
    if requirements is None:
        requirements = []
    if isinstance(requirements, str):
        requirements = [requirements]
    # Normalize requirements
    reqs = set()
    for r in requirements:
        r_upper = r.upper()
        if r_upper not in _flag_alias:
            raise KeyError(r)
        reqs.add(_flag_alias[r_upper])
    if 'C_CONTIGUOUS' in reqs and 'F_CONTIGUOUS' in reqs:
        raise ValueError("Cannot specify both C and Fortran contiguous.")
    # Convert input
    order = 'C'
    if 'F_CONTIGUOUS' in reqs:
        order = 'F'
    if dtype is not None:
        arr = array(a, dtype=dtype, order=order, copy=False)
    else:
        arr = array(a, order=order, copy=False)
    if not isinstance(arr, ndarray):
        arr = asarray(arr)
    # E (ENSUREARRAY) means return base ndarray, not subclass
    if 'ENSUREARRAY' in reqs and type(arr) is not ndarray:
        arr = array(arr, subok=False)
    # W (WRITEABLE) - ensure writable
    if 'WRITEABLE' in reqs:
        if not arr.flags['WRITEABLE']:
            arr = arr.copy()
    # Copy if OWNDATA required (asarray may share memory)
    if 'OWNDATA' in reqs:
        arr = arr.copy()
    return arr

def identity(n, dtype=None):
    return eye(n, dtype=dtype)

def fromfunction(function, shape, dtype=float, **kwargs):
    """Construct an array by executing a function over each coordinate."""
    import numpy as _np
    coords = _np.indices(shape, dtype=dtype)
    return asarray(function(*coords, **kwargs))

def asfortranarray(a):
    """Return an array laid out in Fortran order (simplified: just copy)."""
    from numpy import StructuredArray
    if isinstance(a, StructuredArray):
        # Structured arrays don't have meaningful Fortran order — return as-is
        return a
    from ._helpers import _ObjectArray
    if isinstance(a, _ObjectArray):
        return a.copy()
    return array(a, copy=True)


def asarray_chkfinite(a, dtype=None, order=None):
    """Convert to array, checking for NaN and Inf."""
    if dtype is not None:
        a = asarray(a, dtype=dtype)
    else:
        a = asarray(a)
    if isinstance(a, ndarray):
        vals = a.flatten().tolist()
        for v in vals:
            try:
                if _math.isinf(v) or _math.isnan(v):
                    raise ValueError("array must not contain infs or NaNs")
            except TypeError:
                pass
    return a

def frombuffer(buffer, dtype=None, count=-1, offset=0):
    """Interpret a buffer as a 1-D array (simplified: treats as uint8)."""
    if isinstance(buffer, (bytes, bytearray)):
        data = list(buffer[offset:])
        if count != -1:
            data = data[:count]
        result = array([float(x) for x in data])
        if dtype is not None:
            result = result.astype(str(dtype))
        return result
    return asarray(buffer)

def fromfile(file, dtype=float, count=-1, sep='', offset=0, like=None):
    """Read array from file (stub - not supported in sandbox)."""
    raise NotImplementedError("fromfile not supported in sandboxed environment")
