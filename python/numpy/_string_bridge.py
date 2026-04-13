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


def normalize_native_string_input(value):
    raw = _unwrap_chararray(value)
    shape = _shape_of(raw)
    if isinstance(raw, _ObjectArray):
        flat = raw.tolist()
        arr = asarray(flat).reshape(shape or (len(flat),))
    else:
        arr = asarray(raw)
    return arr, shape


def native_string_unary(value, native_op, *, wrap_chararray=False):
    arr, shape = normalize_native_string_input(value)
    out = _restore_shape(native_op(arr), shape)
    if wrap_chararray:
        from ._string_ops import chararray

        return chararray._from_array(out)
    return out


def python_string_items(value):
    raw = _unwrap_chararray(value)
    shape = _shape_of(raw)
    if isinstance(raw, ndarray):
        return raw.flatten().tolist(), shape
    if isinstance(raw, _ObjectArray):
        return list(raw._data), shape
    if isinstance(raw, (list, tuple)):
        arr = asarray(raw, dtype=object)
        return arr.flatten().tolist(), arr.shape
    return [raw], shape


def python_string_broadcast(value, size):
    raw = _unwrap_chararray(value)
    if isinstance(raw, (ndarray, _ObjectArray)):
        values, _ = python_string_items(value)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) == 1 and size > 1:
        values = values * size
    return values


def _string_method_target(item):
    if isinstance(item, (str, bytes)):
        return item
    return str(item)


def _string_method_arg(value):
    if isinstance(value, (str, bytes)):
        return value
    return str(value)


def python_string_map(
    value,
    mapper,
    *,
    result_kind="string",
    wrap_chararray=False,
    extra_shape=(),
):
    items, shape = python_string_items(value)
    result = [mapper(item) for item in items]
    out_shape = tuple(shape) + tuple(extra_shape)
    if result_kind == "bool":
        out = array([1.0 if item else 0.0 for item in result]).astype("bool")
    elif result_kind == "object":
        data = []
        if extra_shape:
            for item in result:
                if isinstance(item, (list, tuple)):
                    data.extend(item)
                else:
                    data.append(item)
        else:
            data = result
        target_shape = out_shape or (len(data),)
        out = _ObjectArray(data, "object", shape=target_shape)
    else:
        out = array(result)
    if out_shape and result_kind != "object":
        out = out.reshape(out_shape)
    if wrap_chararray:
        from ._string_ops import chararray

        return chararray._from_array(out)
    return out


def python_string_predicate(value, predicate):
    return python_string_map(
        value,
        lambda item: predicate(_string_method_target(item)),
        result_kind="bool",
    )


def python_string_search(value, needle, method_name, start=0, end=None):
    items, _ = python_string_items(value)
    needles = python_string_broadcast(needle, len(items))
    return python_string_map(
        value,
        lambda item, state=iter(needles): getattr(_string_method_target(item), method_name)(
            _string_method_arg(next(state)),
            start,
        )
        if end is None
        else getattr(_string_method_target(item), method_name)(
            _string_method_arg(next(state)),
            start,
            end,
        ),
        result_kind="int",
    )


def python_string_strip(
    value,
    chars=None,
    *,
    method_name="strip",
    wrap_chararray=False,
):
    if chars is None:
        return python_string_map(
            value,
            lambda item: getattr(_string_method_target(item), method_name)(),
            wrap_chararray=wrap_chararray,
        )
    items, _ = python_string_items(value)
    chars_values = python_string_broadcast(chars, len(items))
    chars_iter = iter(chars_values)
    return python_string_map(
        value,
        lambda item: getattr(_string_method_target(item), method_name)(
            _string_method_arg(next(chars_iter))
        ),
        wrap_chararray=wrap_chararray,
    )


def python_string_pad(
    value,
    width,
    method_name,
    fillchar=" ",
    *,
    wrap_chararray=False,
):
    items, _ = python_string_items(value)
    widths = python_string_broadcast(width, len(items))
    width_iter = iter(widths)
    return python_string_map(
        value,
        lambda item: getattr(_string_method_target(item), method_name)(
            int(next(width_iter)),
            b" "
            if fillchar is None and isinstance(_string_method_target(item), bytes)
            else (" " if fillchar is None else _string_method_arg(fillchar)),
        ),
        wrap_chararray=wrap_chararray,
    )


def python_string_replace(value, old, new, count=None, *, wrap_chararray=False):
    items, _ = python_string_items(value)
    olds = python_string_broadcast(old, len(items))
    news = python_string_broadcast(new, len(items))
    old_iter = iter(olds)
    new_iter = iter(news)
    if count is None:
        return python_string_map(
            value,
            lambda item: _string_method_target(item).replace(
                _string_method_arg(next(old_iter)),
                _string_method_arg(next(new_iter)),
            ),
            wrap_chararray=wrap_chararray,
        )
    counts = python_string_broadcast(count, len(items))
    count_iter = iter(counts)
    return python_string_map(
        value,
        lambda item: _string_method_target(item).replace(
            _string_method_arg(next(old_iter)),
            _string_method_arg(next(new_iter)),
            int(next(count_iter)),
        ),
        wrap_chararray=wrap_chararray,
    )


def python_string_split(value, sep=None, maxsplit=-1):
    return python_string_map(
        value,
        lambda item: _string_method_target(item).split(
            None if sep is None else _string_method_arg(sep),
            maxsplit,
        ),
        result_kind="object",
    )


def python_string_rsplit(value, sep=None, maxsplit=-1):
    return python_string_map(
        value,
        lambda item: _string_method_target(item).rsplit(
            None if sep is None else _string_method_arg(sep),
            maxsplit,
        ),
        result_kind="object",
    )


def python_string_splitlines(value):
    return python_string_map(
        value,
        lambda item: _string_method_target(item).splitlines(),
        result_kind="object",
    )


def python_string_partition(
    value,
    sep,
    *,
    method_name="partition",
    wrap_chararray=False,
):
    items, _ = python_string_items(value)
    seps = python_string_broadcast(sep, len(items))
    sep_iter = iter(seps)
    result = [
        list(
            getattr(_string_method_target(item), method_name)(
                _string_method_arg(next(sep_iter))
            )
        )
        for item in items
    ]
    out = array(result)
    if wrap_chararray:
        from ._string_ops import chararray

        return chararray._from_array(out)
    return out


def python_string_transform(value, method_name, *args, wrap_chararray=False):
    return python_string_map(
        value,
        lambda item: getattr(_string_method_target(item), method_name)(*args),
        wrap_chararray=wrap_chararray,
    )


def python_string_join(seq, sep, *, wrap_chararray=False):
    items, shape = python_string_items(seq)
    seps = python_string_broadcast(sep, len(items))
    sep_iter = iter(seps)
    return python_string_map(
        seq,
        lambda item: _string_method_arg(next(sep_iter)).join(
            [
                _string_method_target(part)
                for part in item
            ]
            if isinstance(item, (list, tuple))
            else _string_method_target(item)
        ),
        wrap_chararray=wrap_chararray,
    ).reshape(shape or (len(items),))
