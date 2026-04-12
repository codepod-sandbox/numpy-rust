import _numpy_native as _native
from _numpy_native import ndarray

from ._creation import array, asarray
from ._helpers import _ObjectArray


def _unwrap_chararray(value):
    from ._string_ops import chararray

    if isinstance(value, chararray):
        return value._arr
    return value


def _shape_of(value):
    if isinstance(value, ndarray):
        return value.shape
    if isinstance(value, _ObjectArray):
        return value.shape
    if isinstance(value, (list, tuple)):
        return asarray(value, dtype=object).shape
    return ()


def _restore_shape(result, shape):
    if isinstance(result, ndarray) and shape and result.shape != shape:
        return result.reshape(shape)
    return result


def native_string_unary(value, native_op, *, wrap_chararray=False):
    raw = _unwrap_chararray(value)
    shape = _shape_of(raw)
    if isinstance(raw, _ObjectArray):
        flat = raw.tolist()
        arr = asarray(flat).reshape(shape or (len(flat),))
    else:
        arr = asarray(raw)
    out = _restore_shape(native_op(arr), shape)
    if wrap_chararray:
        from ._string_ops import chararray

        return chararray._from_array(out)
    return out


def python_string_map(value, func, *, result_kind="string", wrap_chararray=False):
    raw = _unwrap_chararray(value)
    shape = _shape_of(raw)
    arr = asarray(raw, dtype=object)
    flat = [func(item) for item in arr.flatten().tolist()]
    out = array(flat, dtype=object).reshape(shape or (len(flat),))
    if result_kind == "bool":
        out = out.astype("bool")
    elif result_kind == "int":
        out = out.astype("int64")
    if wrap_chararray:
        from ._string_ops import chararray

        return chararray._from_array(out)
    return out
