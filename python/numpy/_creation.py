"""Array creation functions."""
import sys as _sys
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray, dot
from _numpy_native import concatenate as _native_concatenate
from ._helpers import (
    AxisError, _ObjectArray, _ComplexResultArray,
    _copy_into, _apply_order, _infer_shape,
    _flatten_nested, _to_float_list, _is_temporal_dtype,
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


def concatenate(arrays, axis=0, out=None, dtype=None, casting='same_kind'):
    """Wrapper around native concatenate that handles tuples and auto-converts to arrays."""
    arrs = [asarray(a) if not isinstance(a, ndarray) else a for a in arrays]
    return _native_concatenate(arrs, axis)

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
    """Return True if dt is a StructuredDtype or a dtype wrapping one."""
    from ._core_types import StructuredDtype, dtype as _dtype_cls
    if isinstance(dt, StructuredDtype):
        return True
    if isinstance(dt, _dtype_cls) and (hasattr(dt, '_structured') and dt._structured is not None):
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
    return result

def _array_core(data, dtype=None, copy=None, order=None, subok=False, like=None):
    # Check for structured dtype BEFORE normalization (normalization loses field info)
    if dtype is not None and _is_structured_dtype(dtype):
        from ._core_types import dtype as _dtype_cls
        parsed = dtype if isinstance(dtype, _dtype_cls) else _dtype_cls(dtype)
        data_seq = data if isinstance(data, (list, tuple)) else [data]
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
                return _native.array([str(x) for x in data])
            return _native.array(data)
        if dt == "object":
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], dt)
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
        return result
    if isinstance(data, bool):
        # bool must be checked before int since bool is a subclass of int
        dt_name = str(dtype) if dtype is not None else 'bool'
        result = _native.full([], 1.0 if data else 0.0, 'float64').astype(dt_name)
        return result
    if isinstance(data, int):
        dt_name = str(dtype) if dtype is not None else 'int64'
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
        return _ObjectArray(data if isinstance(data, list) else list(data), "complex128")
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (int, float)):
        # Check if any element is complex (mixed int/float/complex list)
        _any_complex = __import__("builtins").any(isinstance(x, complex) for x in data)
        if _any_complex:
            return _ObjectArray([complex(x) for x in data], "complex128")
        result = _native.array([float(x) for x in data])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    # Nested lists/tuples: infer shape, flatten, reshape
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple, ndarray)):
        shape = _infer_shape(data)
        if shape is not None:
            flat = _flatten_nested(data)
            if flat is not None:
                result = _native.array(flat)
                result = result.reshape(shape)
                if dtype is not None:
                    dt = str(dtype)
                    if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                        result = result.astype(dt)
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
                raise ValueError(
                    "structured arrays only support 1D shape in this implementation; "
                    "got shape {}".format(tuple(shape))
                )
            nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
            return _create_empty_structured(nrows, parsed, fill_value=0)
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else None
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
    dt = _normalize_dtype_with_size(dtype) if dtype is not None else None
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
    dt = _normalize_dtype(str(dtype)) if dtype is not None else None
    if dt == "object" or dt == "<class 'object'>":
        float_args = [float(a) for a in args]
        if len(float_args) == 1:
            vals = list(range(int(float_args[0])))
        elif len(float_args) == 2:
            vals = list(range(int(float_args[0]), int(float_args[1])))
        else:
            vals = list(range(int(float_args[0]), int(float_args[1]), int(float_args[2])))
        return _ObjectArray(vals, "object")
    float_args = [float(a) for a in args]
    # Normalize to (start, stop, step) form
    if len(float_args) == 1:
        float_args = [0.0, float_args[0], 1.0]
    elif len(float_args) == 2:
        float_args = [float_args[0], float_args[1], 1.0]
    if dtype is not None:
        return _native.arange(float_args[0], float_args[1], float_args[2], dt)
    return _native.arange(float_args[0], float_args[1], float_args[2])

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start = float(start)
    stop = float(stop)
    num = int(num)
    if endpoint:
        result = _native.linspace(start, stop, num)
        step = (stop - start) / (num - 1) if num > 1 else 0.0
    else:
        step = (stop - start) / num if num > 0 else 0.0
        result = array([start + i * step for i in range(num)])
    if dtype is not None:
        result = result.astype(str(dtype))
    if retstep:
        return result, step
    return result

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    import numpy as _np
    y = linspace(start, stop, num=num, endpoint=endpoint)
    result = _np.power(base, y)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    log_start = _math.log10(start)
    log_stop = _math.log10(stop)
    result = logspace(log_start, log_stop, num=num, endpoint=endpoint)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def eye(N, M=None, k=0, dtype=None, order="C", like=None):
    if dtype is not None:
        if M is not None:
            return _native.eye(N, M, k, str(dtype))
        return _native.eye(N, N, k, str(dtype))
    if M is not None:
        return _native.eye(N, M, k)
    if k != 0:
        return _native.eye(N, N, k)
    return _native.eye(N)

def where(condition, x=None, y=None):
    if x is None and y is None:
        import numpy as _np
        return _np.nonzero(condition)
    if not isinstance(condition, ndarray):
        condition = asarray(condition)
    # Ensure condition is boolean dtype (native where_ requires bool)
    if str(condition.dtype) != "bool":
        condition = condition.astype("bool")
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
    src_is_f = getattr(getattr(source, 'flags', None), 'f_contiguous', False)
    if order == 'F':
        if hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
    elif order == 'K':
        if src_is_f and hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
    elif order == 'A':
        if src_is_f and hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
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
                raise ValueError(
                    "structured arrays only support 1D shape in this implementation; "
                    "got shape {}".format(tuple(shape))
                )
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
        # StructuredArray is already an array-like, return as-is
        return a
    return array(a, dtype=dtype)

asanyarray = asarray  # In our implementation, same as asarray

def ascontiguousarray(a, dtype=None):
    return asarray(a)

def copy(a, order="K"):
    if isinstance(a, ndarray):
        return a.copy()
    return array(a)

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
        if not isinstance(a1, ndarray):
            a1 = array(a1)
        if not isinstance(a2, ndarray):
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
    return array(a, copy=True)


def asarray_chkfinite(a):
    """Convert to array, checking for NaN and Inf."""
    a = asarray(a)
    if isinstance(a, ndarray):
        vals = a.flatten().tolist()
        for v in vals:
            if _math.isinf(v) or _math.isnan(v):
                raise ValueError("array must not contain infs or NaNs")
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
