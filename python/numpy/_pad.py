"""Array padding: pad and all its helpers."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate

__all__ = [
    'pad',
    '_validate_pad_width_type', '_normalize_pad_width', '_normalize_stat_length',
    '_normalize_per_axis_val',
    '_pad_constant', '_pad_edge', '_reflect_index', '_pad_reflect',
    '_pad_symmetric', '_pad_wrap', '_pad_linear_ramp', '_pad_stat',
    '_compute_stat', '_pad_empty', '_take_along_axis', '_PadVector',
    '_pad_callable',
]


def _validate_pad_width_type(pad_width):
    """Raise TypeError if pad_width contains non-integral values."""
    def _check_scalar(v):
        if v is None or isinstance(v, str):
            raise TypeError("`pad_width` must be of integral type.")
        if isinstance(v, complex):
            raise TypeError("`pad_width` must be of integral type.")
        if isinstance(v, float):
            if v != int(v) or _math.isnan(v) or _math.isinf(v):
                raise TypeError("`pad_width` must be of integral type.")
            # float that equals its int (e.g. 3.0) is technically also
            # rejected by NumPy >= 1.25
            raise TypeError("`pad_width` must be of integral type.")
        if isinstance(v, bool):
            return  # bool is a subclass of int, accepted
        if isinstance(v, int):
            return  # plain int is fine
        # ndarray / _ObjectArray scalar
        if isinstance(v, ndarray):
            if v.ndim == 0:
                _check_scalar_dtype(v)
            else:
                for i in range(v.size):
                    _check_scalar(v.flatten()[i])
            return
        # Unknown type
        raise TypeError("`pad_width` must be of integral type.")

    def _check_scalar_dtype(arr):
        """Check a 0-d ndarray for integral dtype."""
        dt = str(arr.dtype) if hasattr(arr, 'dtype') else ''
        if 'float' in dt or 'complex' in dt or 'str' in dt or 'object' in dt:
            raise TypeError("`pad_width` must be of integral type.")

    def _walk(pw):
        if isinstance(pw, dict):
            for v in pw.values():
                _walk(v)
            return
        if isinstance(pw, ndarray):
            dt = str(pw.dtype) if hasattr(pw, 'dtype') else ''
            if 'float' in dt or 'complex' in dt or 'str' in dt or 'object' in dt or 'bytes' in dt:
                raise TypeError("`pad_width` must be of integral type.")
            return
        if hasattr(pw, '_data'):
            # _ObjectArray
            for v in pw._data:
                _check_scalar(v)
            return
        if isinstance(pw, (list, tuple)):
            for item in pw:
                _walk(item)
            return
        _check_scalar(pw)

    _walk(pad_width)


def pad(a, pad_width, mode='constant', **kwargs):
    """Pad an array.

    Parameters
    ----------
    a : array_like
    pad_width : int, sequence, or array_like
        Number of values padded to the edges of each axis.
    mode : str or callable
        Padding mode.
    **kwargs : keyword arguments for the mode.
    """
    if not isinstance(a, ndarray):
        a = asarray(a)
    _orig_dtype = str(a.dtype)  # remember for dtype preservation

    # Handle callable mode (legacy vector functionality)
    if callable(mode):
        result = _pad_callable(a, pad_width, mode, kwargs)
        if str(result.dtype) != _orig_dtype:
            try:
                result = result.astype(_orig_dtype)
            except Exception:
                pass
        return result

    # Validate pad_width contains only integral types
    _validate_pad_width_type(pad_width)

    # Normalise pad_width to array of shape (ndim, 2)
    pw = _normalize_pad_width(pad_width, a.ndim)

    # Validate no negative values
    for _ax in range(a.ndim):
        if pw[_ax][0] < 0 or pw[_ax][1] < 0:
            raise ValueError("index can't contain negative values")

    # Check for empty axes that need padding with non-constant/empty modes
    if mode not in ('constant', 'empty'):
        for ax in range(a.ndim):
            if a.shape[ax] == 0 and (pw[ax][0] > 0 or pw[ax][1] > 0):
                raise ValueError(
                    "can't extend empty axis %d using modes other than "
                    "'constant' or 'empty'" % ax
                )

    # For minimum/maximum stat modes, validate stat_length even with zero padding
    if mode in ('minimum', 'maximum') and 'stat_length' in kwargs:
        _sl_check = kwargs['stat_length']
        if _sl_check is not None:
            _sl_arr = _normalize_stat_length(_sl_check, a.ndim)
            for _ax in range(a.ndim):
                _sb, _sa = _sl_arr[_ax]
                if (_sb is not None and int(_sb) == 0) or (_sa is not None and int(_sa) == 0):
                    raise ValueError("stat_length of 0 yields no value for padding")

    # Shortcut: no padding needed
    total_pad = sum(pw[ax][0] + pw[ax][1] for ax in range(a.ndim))
    if total_pad == 0:
        return a.copy() if hasattr(a, 'copy') else array(a)

    # Validate kwargs: each mode only accepts specific keyword arguments
    _allowed_kwargs = {
        'constant': {'constant_values'},
        'edge': set(),
        'linear_ramp': {'end_values'},
        'reflect': {'reflect_type'},
        'symmetric': {'reflect_type'},
        'wrap': set(),
        'maximum': {'stat_length'},
        'minimum': {'stat_length'},
        'mean': {'stat_length'},
        'median': {'stat_length'},
        'empty': set(),
    }
    if isinstance(mode, str) and mode in _allowed_kwargs:
        _bad = set(kwargs) - _allowed_kwargs[mode]
        if _bad:
            raise ValueError(
                "unsupported keyword arguments for mode '%s'" % mode
            )

    if mode == 'constant':
        result = _pad_constant(a, pw, kwargs.get('constant_values', 0))
    elif mode == 'edge':
        result = _pad_edge(a, pw)
    elif mode == 'linear_ramp':
        result = _pad_linear_ramp(a, pw, kwargs.get('end_values', 0))
    elif mode == 'reflect':
        result = _pad_reflect(a, pw, kwargs.get('reflect_type', 'even'))
    elif mode == 'symmetric':
        result = _pad_symmetric(a, pw, kwargs.get('reflect_type', 'even'))
    elif mode == 'wrap':
        result = _pad_wrap(a, pw)
    elif mode in ('maximum', 'minimum', 'mean', 'median'):
        result = _pad_stat(a, pw, mode, kwargs.get('stat_length', None))
    elif mode == 'empty':
        result = _pad_empty(a, pw)
    else:
        raise ValueError("mode '%s' is not supported" % (mode,))

    # Preserve the original dtype (padding may upcast narrow types like uint8→int32)
    if isinstance(result, ndarray) and str(result.dtype) != _orig_dtype:
        try:
            _int_dtypes = ('int8', 'int16', 'int32', 'int64',
                           'uint8', 'uint16', 'uint32', 'uint64')
            if _orig_dtype in _int_dtypes and 'float' in str(result.dtype):
                # Use Python's built-in round() for banker's rounding (round-half-to-even)
                import builtins as _builtins
                _flat = [_builtins.round(float(v)) for v in result.flatten().tolist()]
                result = asarray(_flat).reshape(result.shape).astype(_orig_dtype)
            else:
                result = result.astype(_orig_dtype)
        except Exception:
            pass

    # Preserve memory layout flag
    if isinstance(result, ndarray) and hasattr(a, 'flags'):
        try:
            if a.flags["F_CONTIGUOUS"] and not a.flags["C_CONTIGUOUS"]:
                result._mark_fortran()
        except Exception:
            pass

    return result


def _normalize_pad_width(pad_width, ndim):
    """Normalize pad_width to list of (before, after) tuples, one per axis."""
    import math

    def _to_int(x):
        return int(math.floor(float(x) + 0.5)) if isinstance(x, float) else int(x)

    if isinstance(pad_width, (int, float)):
        v = _to_int(pad_width)
        return [(v, v)] * ndim

    # Dict form: {axis: (before, after)} or {axis: scalar}
    if isinstance(pad_width, dict):
        result = [(0, 0)] * ndim
        for axis, val in pad_width.items():
            ax = int(axis)
            if ax < 0:
                ax += ndim
            if isinstance(val, (int, float)):
                v = _to_int(val)
                result[ax] = (v, v)
            elif isinstance(val, (list, tuple)):
                if len(val) == 1:
                    v = _to_int(val[0])
                    result[ax] = (v, v)
                elif len(val) == 2:
                    result[ax] = (_to_int(val[0]), _to_int(val[1]))
                else:
                    raise ValueError("operands could not be broadcast together")
            else:
                v = _to_int(val)
                result[ax] = (v, v)
        return result

    # Convert to a flat structure
    if hasattr(pad_width, 'tolist'):
        pad_width = pad_width.tolist()

    if isinstance(pad_width, (list, tuple)):
        # Check for nested structure
        if len(pad_width) == 0:
            raise ValueError("pad_width must not be empty")

        first = pad_width[0]
        if isinstance(first, (int, float)):
            # 1D: either (before, after) broadcast to all axes, or (val,) broadcast
            if len(pad_width) == 1:
                v = _to_int(first)
                return [(v, v)] * ndim
            elif len(pad_width) == 2:
                return [(_to_int(pad_width[0]), _to_int(pad_width[1]))] * ndim
            else:
                raise ValueError(
                    "operands could not be broadcast together with shape "
                    "(%d,) and (%d, 2)" % (len(pad_width), ndim)
                )
        elif isinstance(first, (list, tuple)):
            # List of tuples: one per axis, or one broadcast to all axes
            if len(pad_width) == 1:
                # Single pair broadcast to all axes
                p = pad_width[0]
                if len(p) == 1:
                    v = _to_int(p[0])
                    return [(v, v)] * ndim
                elif len(p) == 2:
                    return [(_to_int(p[0]), _to_int(p[1]))] * ndim
                else:
                    raise ValueError(
                        "operands could not be broadcast together"
                    )
            elif len(pad_width) == ndim:
                result = []
                for p in pad_width:
                    if isinstance(p, (int, float)):
                        v = _to_int(p)
                        result.append((v, v))
                    elif isinstance(p, (list, tuple)):
                        # Check if elements are themselves nested sequences (too many dims)
                        if len(p) > 0 and isinstance(p[0], (list, tuple)):
                            raise ValueError(
                                "input operand has more dimensions than "
                                "allowed by the axis remapping"
                            )
                        if len(p) == 1:
                            v = _to_int(p[0])
                            result.append((v, v))
                        elif len(p) == 2:
                            result.append((_to_int(p[0]), _to_int(p[1])))
                        else:
                            # Too many scalar elements in a pair
                            raise ValueError(
                                "operands could not be broadcast together"
                            )
                    else:
                        v = _to_int(p)
                        result.append((v, v))
                return result
            else:
                raise ValueError(
                    "operands could not be broadcast together with shape "
                    "(%d,) and (%d, 2)" % (len(pad_width), ndim)
                )
        else:
            v = _to_int(first)
            return [(v, v)] * ndim
    else:
        v = _to_int(pad_width)
        return [(v, v)] * ndim


def _normalize_stat_length(stat_length, ndim):
    """Normalize stat_length to list of (before, after) tuples."""
    from numpy.lib._arraypad_impl import _as_pairs
    if stat_length is None:
        return [(None, None)] * ndim
    return _as_pairs(stat_length, ndim, as_index=True)


def _normalize_per_axis_val(val, ndim):
    """Normalize a per-axis value (constant_values or end_values) to (ndim, 2) shape."""
    if isinstance(val, (int, float)):
        return [(val, val)] * ndim
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return [(0, 0)] * ndim
        first = val[0]
        if isinstance(first, (int, float)):
            if len(val) == 1:
                return [(first, first)] * ndim
            elif len(val) == 2:
                return [(val[0], val[1])] * ndim
            else:
                return [(val[0], val[1])] * ndim
        elif isinstance(first, (list, tuple)):
            if len(val) == ndim:
                result = []
                for p in val:
                    if isinstance(p, (list, tuple)):
                        result.append((p[0], p[1]))
                    else:
                        result.append((p, p))
                return result
            elif len(val) == 1:
                p = val[0]
                if isinstance(p, (list, tuple)):
                    return [(p[0], p[1])] * ndim
                return [(p, p)] * ndim
            else:
                result = []
                for p in val:
                    if isinstance(p, (list, tuple)):
                        result.append((p[0], p[1]))
                    else:
                        result.append((p, p))
                return result
        else:
            # Non-numeric elements: treat like a pair if len==2, else single value
            if len(val) == 2:
                return [(val[0], val[1])] * ndim
            return [(first, first)] * ndim
    return [(val, val)] * ndim


def _pad_constant(a, pw, constant_values):
    """Pad with constant values."""
    cv = _normalize_per_axis_val(constant_values, a.ndim)

    # Build the new shape
    new_shape = []
    for ax in range(a.ndim):
        new_shape.append(a.shape[ax] + pw[ax][0] + pw[ax][1])

    # Handle empty dimensions
    has_empty = False
    for ax in range(a.ndim):
        if a.shape[ax] == 0 and (pw[ax][0] > 0 or pw[ax][1] > 0):
            has_empty = True

    if has_empty or a.size == 0:
        result = zeros(tuple(new_shape), dtype=a.dtype)
        return result

    # Pad axis by axis using concatenate
    result = a
    for ax in range(a.ndim):
        before, after = pw[ax]
        before_val, after_val = cv[ax]

        if before > 0:
            before_shape = list(result.shape)
            before_shape[ax] = before
            if str(a.dtype) == 'object':
                from ._helpers import _ObjectArray
                n = 1
                for s in before_shape:
                    n *= s
                before_arr = _ObjectArray([before_val] * n, 'object', shape=tuple(before_shape))
            else:
                before_arr = ones(tuple(before_shape), dtype=a.dtype) * asarray(before_val).astype(a.dtype)
            result = concatenate([before_arr, result], axis=ax)

        if after > 0:
            after_shape = list(result.shape)
            after_shape[ax] = after
            if str(a.dtype) == 'object':
                from ._helpers import _ObjectArray
                n = 1
                for s in after_shape:
                    n *= s
                after_arr = _ObjectArray([after_val] * n, 'object', shape=tuple(after_shape))
            else:
                after_arr = ones(tuple(after_shape), dtype=a.dtype) * asarray(after_val).astype(a.dtype)
            result = concatenate([result, after_arr], axis=ax)

    return result


def _pad_edge(a, pw):
    """Pad with edge values."""
    result = a
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before == 0 and after == 0:
            continue

        n = result.shape[ax]
        if n == 0:
            continue

        parts = []
        if before > 0:
            # Take first slice along axis and repeat
            slices = [slice(None)] * result.ndim
            slices[ax] = slice(0, 1)
            edge = result[tuple(slices)]
            # Tile to get `before` copies
            reps = [1] * result.ndim
            reps[ax] = before
            from ._iteration import tile
            parts.append(tile(edge, reps))

        parts.append(result)

        if after > 0:
            slices = [slice(None)] * result.ndim
            slices[ax] = slice(n - 1, n)
            edge = result[tuple(slices)]
            reps = [1] * result.ndim
            reps[ax] = after
            from ._iteration import tile
            parts.append(tile(edge, reps))

        result = concatenate(parts, axis=ax)

    return result


def _reflect_index(i, n, reflect_type='even'):
    """Get reflected index for padding. i is the distance from edge (1-based)."""
    if n <= 1:
        return 0
    period = 2 * (n - 1)
    idx = i % period
    if idx >= n:
        idx = period - idx
    return idx


def _pad_reflect(a, pw, reflect_type='even'):
    """Pad with reflected values."""
    result = a
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before == 0 and after == 0:
            continue
        n = result.shape[ax]
        if n == 0:
            continue

        parts = []
        if before > 0:
            # Build before-padding indices
            indices = []
            for i in range(before, 0, -1):
                idx = _reflect_index(i, n)
                indices.append(idx)
            before_arr = _take_along_axis(result, indices, ax)
            if reflect_type == 'odd':
                # odd reflect: iterative to handle pads > n-1
                before_arr = _odd_reflect_before(result, ax, n, before)
            parts.append(before_arr)

        parts.append(result)

        if after > 0:
            indices = []
            for i in range(1, after + 1):
                idx = n - 1 - _reflect_index(i, n)
                indices.append(idx)
            after_arr = _take_along_axis(result, indices, ax)
            if reflect_type == 'odd':
                after_arr = _odd_reflect_after(result, ax, n, after)
            parts.append(after_arr)

        result = concatenate(parts, axis=ax)

    return result


def _reverse_along_ax(arr, ax):
    """Reverse array along one axis."""
    slices = [slice(None)] * arr.ndim
    slices[ax] = slice(None, None, -1)
    return arr[tuple(slices)]


def _odd_reflect_before(result, ax, n, before):
    """Compute odd-reflect before-padding iteratively (reflect: period 2*(n-1))."""
    from ._iteration import tile
    slices = [slice(None)] * result.ndim

    slices[ax] = slice(0, 1)
    edge = result[tuple(slices)]

    period_len = n - 1
    if period_len == 0:
        reps = [1] * result.ndim
        reps[ax] = before
        return tile(edge, reps)

    # Period source: elements 1..n-1 (excludes edge, period=n-1)
    slices[ax] = slice(1, n)
    period_src = result[tuple(slices)]

    parts = []  # nearest-first, each chunk nearest-first within
    remaining = before
    while remaining > 0:
        n_this = _builtin_min(period_len, remaining)
        slices[ax] = slice(0, n_this)
        src = period_src[tuple(slices)]

        reps = [1] * result.ndim
        reps[ax] = n_this
        edge_broad = tile(edge, reps)
        new_arr = edge_broad * 2 - src
        parts.append(new_arr)

        slices[ax] = slice(n_this - 1, n_this)
        edge = new_arr[tuple(slices)]

        # Reverse for next period (n_this elements)
        period_src = _reverse_along_ax(new_arr, ax)
        remaining -= n_this

    # Reverse chunk order and each chunk to get furthest-first
    return concatenate([_reverse_along_ax(p, ax) for p in reversed(parts)], axis=ax)


def _odd_reflect_after(result, ax, n, after):
    """Compute odd-reflect after-padding iteratively (reflect: period 2*(n-1))."""
    from ._iteration import tile
    slices = [slice(None)] * result.ndim

    slices[ax] = slice(n - 1, n)
    edge = result[tuple(slices)]

    period_len = n - 1
    if period_len == 0:
        reps = [1] * result.ndim
        reps[ax] = after
        return tile(edge, reps)

    # Period source: elements n-2..0 (excludes edge, period=n-1)
    rev = _reverse_along_ax(result, ax)
    slices[ax] = slice(1, n)
    period_src = rev[tuple(slices)]

    parts = []  # nearest-first
    remaining = after
    while remaining > 0:
        n_this = _builtin_min(period_len, remaining)
        slices[ax] = slice(0, n_this)
        src = period_src[tuple(slices)]

        reps = [1] * result.ndim
        reps[ax] = n_this
        edge_broad = tile(edge, reps)
        new_arr = edge_broad * 2 - src
        parts.append(new_arr)

        slices[ax] = slice(n_this - 1, n_this)
        edge = new_arr[tuple(slices)]

        period_src = _reverse_along_ax(new_arr, ax)
        remaining -= n_this

    return concatenate(parts, axis=ax)


def _odd_sym_before(result, ax, n, before):
    """Compute odd-symmetric before-padding iteratively (handles large pads)."""
    from ._iteration import tile
    slices = [slice(None)] * result.ndim

    # Initial edge slice (shape with ax-size=1)
    slices[ax] = slice(0, 1)
    edge = result[tuple(slices)]

    # Initial period source: first n elements along ax
    slices[ax] = slice(0, n)
    period_src = result[tuple(slices)]

    parts = []  # nearest-first
    remaining = before
    while remaining > 0:
        n_this = _builtin_min(n, remaining)
        # Take first n_this along ax from period_src
        slices[ax] = slice(0, n_this)
        src = period_src[tuple(slices)]

        # Broadcast edge to n_this along ax
        reps = [1] * result.ndim
        reps[ax] = n_this
        edge_broad = tile(edge, reps)

        new_arr = edge_broad * 2 - src
        parts.append(new_arr)

        # Update edge: last element of new_arr along ax
        slices[ax] = slice(n_this - 1, n_this)
        edge = new_arr[tuple(slices)]

        # Reverse new_arr along ax for next period
        slices[ax] = slice(None, None, -1)
        period_src = new_arr[tuple(slices)]

        remaining -= n_this

    # parts are nearest-first; reverse order AND each chunk (furthest-first)
    return concatenate([_reverse_along_ax(p, ax) for p in reversed(parts)], axis=ax)


def _odd_sym_after(result, ax, n, after):
    """Compute odd-symmetric after-padding iteratively (handles large pads)."""
    from ._iteration import tile
    slices = [slice(None)] * result.ndim

    # Edge slice: last element
    slices[ax] = slice(n - 1, n)
    edge = result[tuple(slices)]

    # Period source: last n elements reversed (going outward from edge)
    slices[ax] = slice(None, None, -1)
    rev = result[tuple(slices)]
    slices[ax] = slice(0, n)
    period_src = rev[tuple(slices)]

    parts = []  # left-to-right order (first element closest to original)
    remaining = after
    while remaining > 0:
        n_this = _builtin_min(n, remaining)
        slices[ax] = slice(0, n_this)
        src = period_src[tuple(slices)]

        reps = [1] * result.ndim
        reps[ax] = n_this
        edge_broad = tile(edge, reps)

        new_arr = edge_broad * 2 - src
        parts.append(new_arr)

        slices[ax] = slice(n_this - 1, n_this)
        edge = new_arr[tuple(slices)]

        slices[ax] = slice(None, None, -1)
        period_src = new_arr[tuple(slices)]

        remaining -= n_this

    return concatenate(parts, axis=ax)


def _pad_symmetric(a, pw, reflect_type='even'):
    """Pad with symmetric (mirror) values."""
    result = a
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before == 0 and after == 0:
            continue
        n = result.shape[ax]
        if n == 0:
            continue

        parts = []
        if before > 0:
            indices = []
            for i in range(before, 0, -1):
                idx = (i - 1) % (2 * n)
                if idx >= n:
                    idx = 2 * n - 1 - idx
                indices.append(idx)
            before_arr = _take_along_axis(result, indices, ax)
            if reflect_type == 'odd':
                before_arr = _odd_sym_before(result, ax, n, before)
            parts.append(before_arr)

        parts.append(result)

        if after > 0:
            indices = []
            for i in range(1, after + 1):
                idx = (i - 1) % (2 * n)
                if idx >= n:
                    idx = 2 * n - 1 - idx
                indices.append(n - 1 - idx)
            after_arr = _take_along_axis(result, indices, ax)
            if reflect_type == 'odd':
                after_arr = _odd_sym_after(result, ax, n, after)
            parts.append(after_arr)

        result = concatenate(parts, axis=ax)

    return result


def _pad_wrap(a, pw):
    """Pad with wrapped values."""
    result = a
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before == 0 and after == 0:
            continue
        n = result.shape[ax]
        if n == 0:
            continue

        parts = []
        if before > 0:
            indices = []
            for i in range(before, 0, -1):
                idx = (n - (i % n)) % n
                indices.append(idx)
            parts.append(_take_along_axis(result, indices, ax))

        parts.append(result)

        if after > 0:
            indices = []
            for i in range(after):
                idx = i % n
                indices.append(idx)
            parts.append(_take_along_axis(result, indices, ax))

        result = concatenate(parts, axis=ax)

    return result


def _pad_linear_ramp(a, pw, end_values):
    """Pad with linear ramp to end values."""
    ev = _normalize_per_axis_val(end_values, a.ndim)
    result = a.astype('float64') if a.dtype in ('int8', 'int16', 'int32', 'int64',
                                                  'uint8', 'uint16', 'uint32', 'uint64',
                                                  'bool') else a.copy() if hasattr(a, 'copy') else asarray(a)
    orig_dtype = a.dtype

    for ax in range(a.ndim):
        before, after = pw[ax]
        if before == 0 and after == 0:
            continue
        n = result.shape[ax]

        parts = []
        if before > 0:
            # Get edge value (first slice along this axis)
            slices = [slice(None)] * result.ndim
            slices[ax] = slice(0, 1)
            edge_val = result[tuple(slices)]
            end_val = ev[ax][0]

            # Create ramp: from end_val (at position 0) to edge_val (at position before)
            # positions: 0, 1, ..., before-1 map to end_val ... (approaching edge_val)
            ramp_parts = []
            for i in range(before):
                # t goes from 0 to 1 as i goes from 0 to before
                t = float(i) / float(before)
                val = asarray(end_val) + (edge_val - asarray(end_val)) * t
                ramp_parts.append(val)
            before_arr = concatenate(ramp_parts, axis=ax)
            parts.append(before_arr)

        parts.append(result)

        if after > 0:
            slices = [slice(None)] * result.ndim
            slices[ax] = slice(n - 1, n)
            edge_val = result[tuple(slices)]
            end_val = ev[ax][1]

            ramp_parts = []
            for i in range(1, after + 1):
                t = float(i) / float(after)
                val = edge_val + (asarray(end_val) - edge_val) * t
                ramp_parts.append(val)
            after_arr = concatenate(ramp_parts, axis=ax)
            parts.append(after_arr)

        result = concatenate(parts, axis=ax)

    return result.astype(orig_dtype)


def _pad_stat(a, pw, mode, stat_length):
    """Pad with statistical values (mean, median, minimum, maximum)."""
    sl = _normalize_stat_length(stat_length, a.ndim)

    result = a
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before == 0 and after == 0:
            continue
        n = result.shape[ax]
        if n == 0:
            continue

        sl_before, sl_after = sl[ax]

        # Compute stat for before-padding: use first sl_before elements along axis
        if before > 0:
            if sl_before is None:
                chunk_before = result  # use all
            else:
                sl_b = _builtin_min(int(sl_before), n)
                if sl_b == 0 and mode in ('minimum', 'maximum'):
                    raise ValueError("stat_length of 0 yields no value for padding")
                slices = [slice(None)] * result.ndim
                slices[ax] = slice(0, _builtin_max(sl_b, 0))
                chunk_before = result[tuple(slices)]

            stat_before = _compute_stat(chunk_before, ax, mode)
            # Broadcast to before-pad shape
            reps = [1] * result.ndim
            reps[ax] = before
            from ._iteration import tile
            before_arr = tile(stat_before, reps)
        else:
            before_arr = None

        if after > 0:
            if sl_after is None:
                chunk_after = result
            else:
                sl_a = _builtin_min(int(sl_after), n)
                if sl_a == 0 and mode in ('minimum', 'maximum'):
                    raise ValueError("stat_length of 0 yields no value for padding")
                slices = [slice(None)] * result.ndim
                slices[ax] = slice(n - _builtin_max(sl_a, 0) if sl_a > 0 else n, n)
                chunk_after = result[tuple(slices)]

            stat_after = _compute_stat(chunk_after, ax, mode)
            reps = [1] * result.ndim
            reps[ax] = after
            from ._iteration import tile
            after_arr = tile(stat_after, reps)
        else:
            after_arr = None

        parts = []
        if before_arr is not None:
            parts.append(before_arr)
        parts.append(result)
        if after_arr is not None:
            parts.append(after_arr)
        result = concatenate(parts, axis=ax)

    return result


def _compute_stat(chunk, axis, mode):
    """Compute a statistic along an axis, keeping dims."""
    import numpy as np
    # Guard against empty chunks (any dimension zero)
    if chunk.shape[axis] == 0 or chunk.size == 0:
        shape = list(chunk.shape)
        shape[axis] = 1
        return np.full(tuple(shape), float('nan'))
    if mode == 'mean':
        return np.mean(chunk, axis=axis, keepdims=True)
    elif mode == 'median':
        # np.median via quantile doesn't support complex; handle separately
        try:
            return np.median(chunk, axis=axis, keepdims=True)
        except (TypeError, ValueError):
            # Complex: compute median on real and imag separately
            re = np.median(chunk.real, axis=axis, keepdims=True)
            im = np.median(chunk.imag, axis=axis, keepdims=True)
            return (re + im * 1j).astype(chunk.dtype)
    elif mode == 'minimum':
        return np.min(chunk, axis=axis, keepdims=True)
    elif mode == 'maximum':
        return np.max(chunk, axis=axis, keepdims=True)


def _pad_empty(a, pw):
    """Pad with uninitialized values (zeros in our case)."""
    new_shape = tuple(a.shape[ax] + pw[ax][0] + pw[ax][1] for ax in range(a.ndim))
    result = zeros(new_shape, dtype=a.dtype)
    # Copy original data into the right position
    if a.size > 0:
        slices = tuple(slice(pw[ax][0], pw[ax][0] + a.shape[ax]) for ax in range(a.ndim))
        # Build result by overlaying original data
        # Since we can't do result[slices] = a, we build it with concatenate
        result = _pad_constant(a, pw, 0)
    return result


def _take_along_axis(arr, indices, axis):
    """Take slices along an axis by indices list, return concatenated result."""
    parts = []
    for idx in indices:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(idx, idx + 1)
        parts.append(arr[tuple(slices)])
    return concatenate(parts, axis=axis)


class _PadVector:
    """Mutable list-like that supports numpy-style slice assignment (broadcast scalar)."""
    def __init__(self, flat_data, indices):
        self._flat = flat_data
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            idxs = range(*key.indices(len(self._indices)))
            return [self._flat[self._indices[i]] for i in idxs]
        return self._flat[self._indices[key]]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            idxs = list(range(*key.indices(len(self._indices))))
            if not hasattr(value, '__len__'):
                # Broadcast scalar to slice
                for i in idxs:
                    self._flat[self._indices[i]] = value
            else:
                for j, i in enumerate(idxs):
                    self._flat[self._indices[i]] = value[j]
        else:
            self._flat[self._indices[key]] = value


def _pad_callable(a, pad_width, func, kwargs):
    """Pad using a user-supplied callable."""
    pw = _normalize_pad_width(pad_width, a.ndim)

    # Build padded array shape
    new_shape = tuple(a.shape[ax] + pw[ax][0] + pw[ax][1] for ax in range(a.ndim))

    # Create output filled with edge-padded values initially
    padded = _pad_edge(a, pw)

    # Use a flattened approach for in-place mutation
    import itertools
    result_flat = padded.flatten().tolist()
    strides = []
    stride = 1
    for d in range(a.ndim - 1, -1, -1):
        strides.insert(0, stride)
        stride *= new_shape[d]

    for ax in range(a.ndim):
        other_axes = [i for i in range(a.ndim) if i != ax]
        ranges = [range(new_shape[i]) for i in other_axes]
        for idx_combo in itertools.product(*ranges):
            # Compute flat indices for this 1D slice
            base = 0
            oi = 0
            for i in range(a.ndim):
                if i != ax:
                    base += idx_combo[oi] * strides[i]
                    oi += 1

            # Build a 1D ndarray-like mutable wrapper
            vec_len = new_shape[ax]
            flat_indices = [base + j * strides[ax] for j in range(vec_len)]
            vector = _PadVector(result_flat, flat_indices)
            func(vector, (pw[ax][0], pw[ax][1]), ax, kwargs)

    return array(result_flat, dtype=a.dtype).reshape(new_shape)
