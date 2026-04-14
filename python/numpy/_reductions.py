"""Aggregation, statistics, NaN-aware reductions, set operations."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray, _copy_into, _coerce_native_boxed_operand,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._core_types import dtype, _ScalarType, _normalize_dtype
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate, linspace, full

def _check_mean_var_dtype(dtype_arg, out):
    """Raise TypeError if dtype or out.dtype is not valid for mean/var/std."""
    import numpy as _np
    _invalid = (bool, _np.bool_, _np.bool, _np.int_, _np.object_,
                _np.int8, _np.int16, _np.int32, _np.int64,
                _np.uint8, _np.uint16, _np.uint32, _np.uint64)
    if dtype_arg is not None:
        try:
            dt = _np.dtype(dtype_arg)
        except Exception:
            dt = None
        if dt is not None and (not _np.issubdtype(dt, _np.floating) and
                               not _np.issubdtype(dt, _np.complexfloating)):
            raise TypeError(
                f"Cannot cast array data from dtype('{dt}') to dtype('float64') "
                "according to the rule 'safe'"
            )
    if out is not None and hasattr(out, 'dtype'):
        try:
            import numpy as _np2
            out_dt = _np2.dtype(str(out.dtype))
            if (not _np2.issubdtype(out_dt, _np2.floating) and
                    not _np2.issubdtype(out_dt, _np2.complexfloating)):
                raise TypeError(
                    f"Cannot cast array data from dtype('float64') to "
                    f"dtype('{out_dt}') according to the rule 'same_kind'"
                )
        except TypeError:
            raise


def _scalar_result(result, input_arr, force_dtype=None):
    """Wrap a scalar result so it preserves dtype as a 0-d array."""
    if isinstance(result, ndarray):
        return result
    dt = force_dtype
    if dt is None and isinstance(input_arr, ndarray):
        dt = str(input_arr.dtype)
    # Complex scalars from Rust arrive as (re, im) tuples — convert to Python complex first
    if isinstance(result, tuple) and len(result) == 2 and dt is not None and dt.startswith('complex'):
        try:
            result = complex(result[0], result[1])
        except Exception:
            pass
    if dt is not None:
        try:
            if _is_boxed_dtype_name(dt):
                return array(result, dtype=dt).reshape(())
            return array(result).astype(dt).reshape(())
        except Exception:
            pass
    return result


def _mean_result_dtype(input_arr):
    """Get the result dtype for mean/var/std operations (int → float64)."""
    try:
        dt = str(input_arr.dtype)
    except AttributeError:
        return 'float64'
    if dt.startswith(('int', 'uint', 'bool')):
        return 'float64'
    return dt


def _complex_to_real_dtype(dt):
    """If dt is a complex dtype, return its real counterpart; else return None."""
    if dt == 'complex64':
        return 'float32'
    if dt in ('complex128', 'complex256', 'clongdouble'):
        return 'float64'
    return None


def _nan_result_like(a, axis, keepdims):
    """Return NaN-filled result matching what a reduction on `a` would produce."""
    dt = str(a.dtype) if isinstance(a, ndarray) else 'float64'
    if axis is None:
        return _scalar_result(float('nan'), a, dt)
    ax = axis if axis >= 0 else a.ndim + axis
    shape = list(a.shape)
    if keepdims:
        shape[ax] = 1
    else:
        shape.pop(ax)
    if not shape:
        return _scalar_result(float('nan'), a, dt)
    result = full(shape, float('nan'))
    if str(result.dtype) != dt:
        try:
            result = result.astype(dt)
        except Exception:
            pass
    return result


# Keep builtin sum reference before it gets shadowed
_builtin_sum = __builtins__["sum"] if isinstance(__builtins__, dict) else __import__("builtins").sum

__all__ = [
    # Internal helper
    '_dtype_cast',
    # Basic reductions
    'sum', 'prod', 'cumsum', 'cumprod', 'diff', 'ediff1d', 'gradient',
    'trapz', 'trapezoid', 'cumulative_trapezoid',
    # Statistics
    'mean', 'std', 'var', 'median', 'average', 'cov', 'corrcoef',
    # Extrema
    'max', 'min', 'amax', 'amin', 'argmax', 'argmin', 'ptp',
    # NaN-aware
    'nansum', 'nanmean', 'nanstd', 'nanvar', 'nanmin', 'nanmax',
    'nanargmin', 'nanargmax', 'nanprod', 'nancumsum', 'nancumprod',
    'nanmedian', 'nanpercentile', 'nanquantile',
    # Quantile
    'quantile', 'percentile',
    # Boolean
    'all', 'any', 'count_nonzero',
    # Search
    'nonzero', 'flatnonzero', 'argwhere', 'searchsorted',
    # Set operations
    'intersect1d', 'union1d', 'setdiff1d', 'setxor1d', 'in1d', 'isin',
    # Memory
    'may_share_memory', 'shares_memory',
]


def _dtype_cast(result, dtype):
    """Cast result to dtype if dtype is not None."""
    if dtype is not None:
        result = asarray(result).astype(str(dtype) if not isinstance(dtype, str) else dtype)
    return result


def _ensure_reduction_array(a, fallback_builtin=None):
    """Normalize reduction input to ndarray when appropriate."""
    if isinstance(a, ndarray):
        return a
    if fallback_builtin is not None and not isinstance(a, (list, tuple)):
        return fallback_builtin(a)
    return asarray(a)


def _copy_reduction_out(out, result):
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return None


def _preserve_inexact_result_dtype(result, input_arr):
    input_dt = str(input_arr.dtype)
    if not isinstance(result, ndarray):
        return _scalar_result(result, input_arr)
    if input_dt.startswith(('float', 'complex')) and str(result.dtype) != input_dt:
        try:
            return result.astype(input_dt)
        except Exception:
            return result
    return result


def _is_boxed_dtype_name(dt):
    return (
        dt == "object"
        or dt.startswith("datetime64")
        or dt.startswith("timedelta64")
    )


def _apply_where_mask(a, where, *, false_fill=None, true_identity=False):
    if where is True:
        return a
    w = asarray(where).astype("bool")
    if false_fill is None:
        mask_f = w.astype("float64")
        return a * mask_f + ((1.0 - mask_f) if true_identity else 0.0)
    flat_a = a.flatten().tolist()
    flat_w = w.flatten().tolist()
    masked = [v if m else false_fill for v, m in zip(flat_a, flat_w)]
    return array(masked).reshape(a.shape)


def _call_reduction_method(a, name, *, axis=None, dtype=None, out=None, keepdims=False):
    method = getattr(a, name)
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if dtype is not None:
        kwargs['dtype'] = dtype
    if out is not None:
        kwargs['out'] = out
    if keepdims:
        kwargs['keepdims'] = keepdims
    return method(**kwargs)


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    a = _ensure_reduction_array(a, fallback_builtin=_builtin_sum)
    if not isinstance(a, ndarray):
        return a
    a = _apply_where_mask(a, where)
    result = _call_reduction_method(a, 'sum', axis=axis, keepdims=keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    else:
        result = _preserve_inexact_result_dtype(result, a)
    if initial is not None:
        result = result + initial
    return result


def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    a = _ensure_reduction_array(a)
    a = _apply_where_mask(a, where, true_identity=True)
    result = a.prod(axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    else:
        result = _preserve_inexact_result_dtype(result, a)
    if initial is not None:
        result = result * initial
    return result


def cumsum(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    _in_dt = str(a.dtype)
    result = a.cumsum(axis)
    if dtype is not None:
        return _dtype_cast(result, dtype)
    if _in_dt.startswith(('float', 'complex')) and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    return result


def cumprod(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    _in_dt = str(a.dtype)
    result = a.cumprod(axis)
    if dtype is not None:
        return _dtype_cast(result, dtype)
    if _in_dt.startswith(('float', 'complex')) and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    return result


def diff(a, n=1, axis=-1, prepend=None, append=None):
    from ._helpers import AxisError as _AxisError
    # n=0: return a unchanged (identity)
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got {}".format(n))
    if not isinstance(a, ndarray):
        a = array(a)
    # Validate axis
    if a.ndim == 0:
        raise ValueError("diff requires an array of at least 1 dimension")
    if axis < 0:
        _axis = a.ndim + axis
    else:
        _axis = axis
    if _axis < 0 or _axis >= a.ndim:
        raise _AxisError(
            "Axis {} is out of bounds for array of dimension {}".format(
                axis, a.ndim
            )
        )
    _is_bool = str(a.dtype) == 'bool'
    if _is_bool:
        a = a.astype('int64')
    if prepend is not None:
        prepend = asarray(prepend)
        if prepend.ndim == 0:
            # Expand scalar to match a's shape with size 1 along axis
            shape = list(a.shape)
            shape[_axis] = 1
            prepend = full(shape, float(prepend))
        elif prepend.ndim < a.ndim:
            # Expand dims to match a's ndim (broadcast along axis)
            shape = list(a.shape)
            shape[_axis] = prepend.shape[0] if prepend.ndim > 0 else 1
            try:
                prepend = prepend.reshape(shape)
            except Exception:
                pass
        a = concatenate([prepend, a], axis=_axis)
    if append is not None:
        append = asarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[_axis] = 1
            append = full(shape, float(append))
        elif append.ndim < a.ndim:
            shape = list(a.shape)
            shape[_axis] = append.shape[0] if append.ndim > 0 else 1
            try:
                append = append.reshape(shape)
            except Exception:
                pass
        a = concatenate([a, append], axis=_axis)
    # If n >= axis size, result is empty along that axis
    _axis_len = a.shape[_axis]
    if n >= _axis_len:
        new_shape = list(a.shape)
        _result_len = _axis_len - n
        new_shape[_axis] = _result_len if _result_len > 0 else 0
        _out_dtype = 'bool' if _is_bool else str(a.dtype)
        return zeros(new_shape, dtype=_out_dtype)
    _in_dtype = str(a.dtype)
    result = _native.diff(a, n, _axis)
    if _is_bool:
        result = result.astype('bool')
    elif 'int' in _in_dtype or 'uint' in _in_dtype:
        result = result.astype(_in_dtype)
    return result


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    from numpy.ma import MaskedArray as _MA
    if isinstance(a, _MA):
        import numpy as _np
        filled = a.filled(0.0)
        not_mask = _np.logical_not(a.mask).astype("float64")
        s = sum(filled * not_mask, axis=axis, keepdims=keepdims)
        c = sum(not_mask, axis=axis, keepdims=keepdims)
        result = s / c
        if out is not None and isinstance(out, _MA):
            _copy_into(out.data, result if isinstance(result, ndarray) else asarray(result))
            return out
        if isinstance(result, ndarray):
            return _MA(result, mask=zeros(result.shape).astype("bool"))
        return result
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        import warnings as _w
        _w.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)
        result = float('nan')
        if out is not None:
            out[()] = result
            return out
        return result
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        s = (a * w).sum(axis, keepdims)
        c = w.sum(axis, keepdims)
        result = s / c
    elif axis is not None:
        if isinstance(axis, tuple):
            # Compute n for the tuple axes
            n = 1
            for ax in axis:
                n *= a.shape[ax]
            result = a.sum(axis, False) / n
            if keepdims:
                new_shape = list(a.shape)
                for ax in axis:
                    new_shape[ax] = 1
                result = result.reshape(tuple(new_shape))
        else:
            result = a.mean(axis, keepdims)
    else:
        result = a.mean(None, keepdims)
    if keepdims and not isinstance(result, ndarray):
        result = array([float(result)]).reshape((1,) * a.ndim)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if not isinstance(result, ndarray) and dtype is None:
        result = _scalar_result(result, a, _mean_result_dtype(a))
    elif isinstance(result, ndarray) and dtype is None:
        _in_dt = _mean_result_dtype(a)
        if _in_dt != 'float64' and str(result.dtype) != _in_dt:
            try:
                result = result.astype(_in_dt)
            except Exception:
                pass
    if out is not None:
        if isinstance(result, ndarray):
            _copy_into(out, result)
        elif isinstance(out, ndarray) and out.size == 1:
            out[0] = float(result)
        return out
    return result


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    from numpy.ma import MaskedArray as _MA
    if isinstance(a, _MA):
        v = var(a, axis=axis, ddof=ddof, keepdims=keepdims, where=where, mean=mean)
        import numpy as _np
        if isinstance(v, _MA):
            result = _MA(_np.sqrt(v.data), mask=v.mask)
        else:
            result = _np.sqrt(v) if isinstance(v, ndarray) else _math.sqrt(v)
        if out is not None and isinstance(out, _MA):
            r_arr = result.data if isinstance(result, _MA) else (result if isinstance(result, ndarray) else asarray(result))
            _copy_into(out.data, r_arr)
            return out
        return result
    if isinstance(a, complex) and not isinstance(a, (int, float)):
        return 0.0
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        import warnings as _w
        _w.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        result = float('nan')
        if out is not None:
            if isinstance(out, ndarray) and out.size == 1:
                out[0] = result
            return out
        return result
    import numpy as _np
    result = _np.sqrt(var(a, axis=axis, ddof=ddof, keepdims=keepdims, where=where, mean=mean))
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif isinstance(result, ndarray):
        _in_dt = _mean_result_dtype(a)
        _real_dt = _complex_to_real_dtype(_in_dt)
        _out_dt = _real_dt if _real_dt is not None else _in_dt
        if str(result.dtype) != _out_dt:
            try:
                result = result.astype(_out_dt)
            except Exception:
                pass
    if out is not None:
        if isinstance(result, ndarray):
            _copy_into(out, result)
        elif isinstance(out, ndarray) and out.size == 1:
            out[0] = float(result)
        return out
    if isinstance(a, ndarray) and axis is None and not keepdims and not isinstance(result, ndarray):
        from ._core_types import float64, complex128
        if isinstance(result, complex):
            return complex128(result)
        return float64(result)
    return result


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    from numpy.ma import MaskedArray as _MA
    if isinstance(a, _MA):
        import numpy as _np
        filled = a.filled(0.0)
        not_mask = _np.logical_not(a.mask).astype("float64")
        c = sum(not_mask, axis=axis, keepdims=True)
        if mean is not None:
            m = mean.data if isinstance(mean, _MA) else (mean if isinstance(mean, ndarray) else asarray(mean))
        else:
            m = sum(filled * not_mask, axis=axis, keepdims=True) / c
        diff = (filled - m) ** 2
        c_out = sum(not_mask, axis=axis, keepdims=keepdims)
        result = sum(diff * not_mask, axis=axis, keepdims=keepdims) / (c_out - ddof)
        if out is not None and isinstance(out, _MA):
            _copy_into(out.data, result if isinstance(result, ndarray) else asarray(result))
            return out
        if isinstance(result, ndarray):
            return _MA(result, mask=zeros(result.shape).astype("bool"))
        return result
    if isinstance(a, complex) and not isinstance(a, (int, float)):
        # scalar complex: var of single value is 0
        return 0.0
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        import warnings as _w
        _w.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        result = float('nan')
        if out is not None:
            if isinstance(out, ndarray) and out.size == 1:
                out[0] = result
            return out
        return result
    _is_complex = str(a.dtype).startswith("complex")
    def _sq_dev(diff):
        if _is_complex:
            return abs(diff) ** 2
        return diff ** 2
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        c_full = w.sum(axis, True)  # keepdims for mean computation
        if mean is not None:
            m = asarray(mean)
        else:
            m = (a * w).sum(axis, True) / c_full
        c_out = w.sum(axis, keepdims)  # match output keepdims
        if ddof == 0:
            result = (_sq_dev(a - m) * w).sum(axis, keepdims) / c_out
        else:
            result = (_sq_dev(a - m) * w).sum(axis, keepdims) / (c_out - ddof)
        if not keepdims and isinstance(result, ndarray) and result.ndim > 0:
            result = result.squeeze()
    elif axis is not None:
        # Compute n for the reduction axes
        if isinstance(axis, int):
            n = a.shape[axis]
        elif isinstance(axis, tuple):
            n = 1
            for ax in axis:
                n *= a.shape[ax]
        else:
            n = a.size
        def _sum_keepdims(arr, ax, kd):
            """Sum with keepdims support for tuple axes."""
            r = arr.sum(ax, False)
            if kd and isinstance(ax, tuple):
                new_shape = list(arr.shape)
                for _ax in ax:
                    new_shape[_ax] = 1
                r = r.reshape(tuple(new_shape))
            elif kd:
                r = arr.sum(ax, True)
                return r
            return r
        if mean is not None:
            m = asarray(mean)
        else:
            m = _sum_keepdims(a, axis, True) / n
        result = _sum_keepdims(_sq_dev(a - m), axis, keepdims) / (n - ddof)
        if not keepdims and isinstance(result, ndarray) and result.ndim == 0:
            result = float(result)
    elif mean is not None:
        m = asarray(mean)
        n = a.size
        result = _sq_dev(a - m).sum() / (n - ddof)
        if isinstance(result, ndarray) and result.ndim == 0:
            result = float(result)
    elif _is_complex:
        _s = a.sum(None, True)
        if isinstance(_s, tuple) and len(_s) == 2:
            _s = asarray(complex(_s[0], _s[1])).astype(str(a.dtype))
        m = _s / float(a.size)
        result = _sq_dev(a - m).sum() / (a.size - ddof)
        if isinstance(result, ndarray) and result.ndim == 0:
            result = float(result)
    else:
        if isinstance(ddof, float) and ddof != int(ddof):
            # float ddof not supported by native var — compute in Python
            m = a.sum(None, True) / float(a.size)
            result = _sq_dev(a - m).sum() / (a.size - ddof)
            if isinstance(result, ndarray) and result.ndim == 0:
                result = float(result)
        else:
            result = a.var(None, int(ddof) if isinstance(ddof, float) else ddof, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    else:
        _in_dt = _mean_result_dtype(a)
        _real_dt = _complex_to_real_dtype(_in_dt)
        _out_dt = _real_dt if _real_dt is not None else _in_dt
        if not isinstance(result, ndarray):
            result = _scalar_result(result, a, _out_dt)
        elif isinstance(result, ndarray) and str(result.dtype) != _out_dt:
            try:
                result = result.astype(_out_dt)
            except Exception:
                pass
    if out is not None:
        if isinstance(result, ndarray):
            _copy_into(out, result)
        elif isinstance(out, ndarray) and out.size == 1:
            out[0] = float(result)
        return out
    return result


def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
    if where is not True:
        _wmask = asarray(where).astype("bool")
        a = a.copy()
        a[~_wmask] = 0
    result = _native.nansum(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    elif isinstance(result, ndarray) and _in_dt.startswith(('float', 'complex')) and str(result.dtype) != _in_dt:
        # Preserve dtype for float/complex types (not int — int sums upcast to int64)
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if initial is not None:
        result = result + initial
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    _check_mean_var_dtype(dtype, out)
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        import numpy as _np
        _wmask = asarray(where).astype("bool")
        # Combine where mask with NaN mask to get valid positions
        _valid = _wmask & ~_np.isnan(a)
        _a_clean = a.copy()
        _a_clean[~_valid] = 0
        _s = _a_clean.sum(axis, keepdims)
        # Complex sum may return tuple (re, im) — convert to scalar
        if isinstance(_s, tuple) and len(_s) == 2:
            _s = asarray(complex(_s[0], _s[1])).astype(str(a.dtype))
        _c = _valid.astype('float64').sum(axis, keepdims)
        result = _s / _c
    else:
        try:
            result = _native.nanmean(a, axis, keepdims)
        except TypeError:
            # _ObjectArray (complex/object dtypes) — compute in Python
            import numpy as _np
            _mask = ~_np.isnan(a)
            _count = _mask.sum(axis, keepdims if keepdims else False)
            if not keepdims:
                _count_kd = _mask.sum(axis, False)
            else:
                _count_kd = _count
            _clean = _np.where(_mask, a, 0)
            result = _clean.sum(axis, keepdims) / _count_kd
        except (ValueError, ZeroDivisionError):
            import warnings
            warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
            result = _nan_result_like(a, axis, keepdims)
        else:
            # Warn for all-NaN slices (nanmean returns NaN when all values are NaN)
            import numpy as _npw
            _has_nan = _npw.any(_npw.isnan(result)) if isinstance(result, ndarray) else (result != result)
            if _has_nan:
                import warnings
                warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    elif axis is not None:
        # Preserve narrow float / complex dtype (Rust nanmean always returns float64)
        _in_dt = _mean_result_dtype(a)
        if _in_dt != 'float64' and str(result.dtype) != _in_dt:
            try:
                result = result.astype(_in_dt)
            except Exception:
                pass
    if out is not None:
        if isinstance(out, ndarray):
            _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
            return out
    return result


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    _check_mean_var_dtype(dtype, out)
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        import numpy as _np
        result = _np.sqrt(nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims, where=where))
    elif mean is not None:
        import numpy as _np
        result = _np.sqrt(nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims, mean=mean))
    else:
        try:
            result = _native.nanstd(a, axis, int(ddof) if isinstance(ddof, float) and ddof == int(ddof) else ddof, keepdims)
        except TypeError:
            # ddof is a float — use nanvar + sqrt
            import numpy as _np
            result = _np.sqrt(nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims))
        except (ValueError, ZeroDivisionError):
            import warnings
            warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=2)
            result = _nan_result_like(a, axis, keepdims)
        else:
            import numpy as _npw
            _has_nan = _npw.any(_npw.isnan(result)) if isinstance(result, ndarray) else (result != result)
            if _has_nan:
                import warnings
                warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=2)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    else:
        _in_dt = _mean_result_dtype(a)
        _real_dt = _complex_to_real_dtype(_in_dt)
        if _real_dt is not None:
            # Complex input: var/std result should be real-valued
            if not isinstance(result, ndarray):
                result = _scalar_result(result, a, _real_dt)
            elif str(result.dtype) != _real_dt:
                try:
                    result = result.astype(_real_dt)
                except Exception:
                    pass
        elif not isinstance(result, ndarray):
            result = _scalar_result(result, a, _in_dt)
        elif _in_dt != 'float64' and str(result.dtype) != _in_dt:
            try:
                result = result.astype(_in_dt)
            except Exception:
                pass
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    _check_mean_var_dtype(dtype, out)
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        import numpy as _np
        _wmask = asarray(where).astype("bool")
        # Combine where mask with NaN mask
        _valid = _wmask & ~_np.isnan(a)
        _c = _valid.astype('float64').sum(axis, True)
        _a_clean = a.copy()
        _a_clean[~_valid] = 0
        _m = _a_clean.sum(axis, True) / _c
        _diff = a - _m
        _diff_clean = _a_clean - _m
        _diff_clean[~_valid] = 0
        if str(a.dtype).startswith('complex'):
            _sq = abs(_diff_clean) ** 2
        else:
            _sq = _diff_clean ** 2
        result = _sq.sum(axis, keepdims) / (_c - ddof)
        if not keepdims and isinstance(result, ndarray) and result.ndim > 0:
            result = result.squeeze()
    elif mean is not None:
        # Use provided precomputed mean
        import numpy as _np
        _mask = ~_np.isnan(a)
        _count = _mask.astype('float64').sum(axis, True)
        _diff = a - mean
        _diff_clean = _np.where(_mask, _diff, 0.0)
        if str(a.dtype).startswith('complex'):
            _sq = abs(_diff_clean) ** 2
        else:
            _sq = _diff_clean ** 2
        result = _sq.sum(axis, keepdims) / (_count - ddof)
        if not keepdims and isinstance(result, ndarray) and result.ndim > 0:
            result = result.squeeze()
    else:
        try:
            result = _native.nanvar(a, axis, int(ddof) if isinstance(ddof, float) and ddof == int(ddof) else ddof, keepdims)
        except TypeError:
            # ddof is a float (e.g. 0.5) — compute in Python
            import numpy as _np
            _mask = ~_np.isnan(a)
            _count = _mask.astype('float64').sum(axis, True)
            _m = nanmean(a, axis=axis, keepdims=True)
            _diff = a - _m
            # Replace NaN with 0 for the squared diff
            _diff_clean = _np.where(_mask, _diff, 0.0)
            # For complex, use |diff|^2 (real-valued) instead of diff^2
            if str(a.dtype).startswith('complex'):
                _sq = abs(_diff_clean) ** 2
            else:
                _sq = _diff_clean ** 2
            result = _sq.sum(axis, keepdims) / (_count - ddof)
            if not keepdims and isinstance(result, ndarray) and result.ndim > 0:
                result = result.squeeze()
        except (ValueError, ZeroDivisionError):
            import warnings
            warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=2)
            result = _nan_result_like(a, axis, keepdims)
        else:
            import numpy as _npw
            _has_nan = _npw.any(_npw.isnan(result)) if isinstance(result, ndarray) else (result != result)
            if _has_nan:
                import warnings
                warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=2)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    else:
        _in_dt = _mean_result_dtype(a)
        _real_dt = _complex_to_real_dtype(_in_dt)
        if _real_dt is not None:
            # Complex input: var/std result should be real-valued
            if not isinstance(result, ndarray):
                result = _scalar_result(result, a, _real_dt)
            elif str(result.dtype) != _real_dt:
                try:
                    result = result.astype(_real_dt)
                except Exception:
                    pass
        elif not isinstance(result, ndarray):
            result = _scalar_result(result, a, _in_dt)
        elif _in_dt != 'float64' and str(result.dtype) != _in_dt:
            try:
                result = result.astype(_in_dt)
            except Exception:
                pass
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nanmin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
    if where is not True:
        _wmask = asarray(where).astype("bool")
        a = a.copy()
        a[~_wmask] = float('inf')
    try:
        result = _native.nanmin(a, axis, keepdims)
    except ValueError:
        import warnings
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        result = _nan_result_like(a, axis, keepdims)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    elif isinstance(result, ndarray) and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if initial is not None:
        import numpy as _np
        result = _np.minimum(result, initial)
        if isinstance(result, ndarray) and str(result.dtype) != _in_dt:
            try:
                result = result.astype(_in_dt)
            except Exception:
                pass
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nanmax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
    if where is not True:
        _wmask = asarray(where).astype("bool")
        a = a.copy()
        a[~_wmask] = float('-inf')
    try:
        result = _native.nanmax(a, axis, keepdims)
    except ValueError:
        import warnings
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        result = _nan_result_like(a, axis, keepdims)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    elif isinstance(result, ndarray) and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if initial is not None:
        import numpy as _np
        result = _np.maximum(result, initial)
        if isinstance(result, ndarray) and str(result.dtype) != _in_dt:
            try:
                result = result.astype(_in_dt)
            except Exception:
                pass
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nanargmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    # Handle empty arrays
    if a.size == 0:
        if axis is not None:
            ax = axis if axis >= 0 else a.ndim + axis
            if a.shape[ax] == 0:
                # The axis being reduced is empty → can't find argmin
                raise ValueError("attempt to get argmin of an empty sequence")
            # Other axes are empty → result is empty array
            result_shape = tuple(s for i, s in enumerate(a.shape) if i != ax)
            result = zeros(result_shape, dtype='intp')
            if keepdims:
                result = _apply_keepdims(result, a, ax)
            if out is not None and isinstance(out, ndarray):
                _copy_into(out, result)
                return out
            return result
        raise ValueError("attempt to get argmin of an empty sequence")
    result = _native.nanargmin(a, axis)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, 'intp')
    if keepdims:
        if axis is None:
            shape = tuple(1 for _ in a.shape)
            v = int(result)
            result = array([v], dtype='intp').reshape(shape)
        else:
            result = _apply_keepdims(result, a, axis)
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nanargmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    # Handle empty arrays
    if a.size == 0:
        if axis is not None:
            ax = axis if axis >= 0 else a.ndim + axis
            if a.shape[ax] == 0:
                # The axis being reduced is empty → can't find argmax
                raise ValueError("attempt to get argmax of an empty sequence")
            # Other axes are empty → result is empty array
            result_shape = tuple(s for i, s in enumerate(a.shape) if i != ax)
            result = zeros(result_shape, dtype='intp')
            if keepdims:
                result = _apply_keepdims(result, a, ax)
            if out is not None and isinstance(out, ndarray):
                _copy_into(out, result)
                return out
            return result
        raise ValueError("attempt to get argmax of an empty sequence")
    result = _native.nanargmax(a, axis)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, 'intp')
    if keepdims:
        if axis is None:
            shape = tuple(1 for _ in a.shape)
            v = int(result)
            result = array([v], dtype='intp').reshape(shape)
        else:
            result = _apply_keepdims(result, a, axis)
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
    if where is not True:
        _wmask = asarray(where).astype("bool")
        a = a.copy()
        a[~_wmask] = 1
    result = _native.nanprod(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    elif isinstance(result, ndarray) and _in_dt.startswith(('float', 'complex')) and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if initial is not None:
        result = result * initial
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nancumsum(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    result = _native.nancumsum(a, axis)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif isinstance(result, ndarray) and _in_dt.startswith(('float', 'complex')) and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def nancumprod(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    result = _native.nancumprod(a, axis)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif isinstance(result, ndarray) and _in_dt.startswith(('float', 'complex')) and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def _apply_keepdims(result, a, axis):
    """Expand reduced axes back to size-1 when keepdims=True."""
    if not isinstance(result, ndarray):
        result = asarray(result)
    if axis is None:
        shape = tuple(1 for _ in a.shape)
    elif isinstance(axis, (tuple, list)):
        try:
            from numpy._core.numeric import normalize_axis_tuple
            axes = normalize_axis_tuple(axis, a.ndim)
        except Exception:
            axes = tuple(ax if ax >= 0 else a.ndim + ax for ax in axis)
        shape = list(a.shape)
        for ax in axes:
            shape[ax] = 1
        shape = tuple(shape)
    else:
        ax = axis if axis >= 0 else a.ndim + axis
        shape = list(a.shape)
        shape[ax] = 1
        shape = tuple(shape)
    return result.reshape(shape)


def _normalize_axis_arg(axis, ndim):
    """Normalize a single integer axis, raising AxisError if out of bounds."""
    from ._helpers import AxisError
    ax = int(axis)
    if ax < 0:
        ax += ndim
    if ax < 0 or ax >= ndim:
        raise AxisError("axis {} is out of bounds for array of dimension {}".format(int(axis), ndim))
    return ax


def _normalize_axes_tuple(axis, ndim):
    """Normalize a tuple/list of axes, raising AxisError/ValueError on invalid."""
    from ._helpers import AxisError
    axes = []
    for ax in axis:
        ax = int(ax)
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise AxisError("axis {} is out of bounds for array of dimension {}".format(int(ax - (ndim if ax >= 0 else 0)), ndim))
        if ax in axes:
            raise ValueError("duplicate value in 'axis'")
        axes.append(ax)
    return axes


_VALID_QUANTILE_METHODS = frozenset([
    'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation',
    'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear',
    'median_unbiased', 'normal_unbiased', 'nearest', 'lower', 'higher',
    'midpoint'
])


def _compute_quantile_1d(sorted_vals, q, method='linear'):
    """Compute quantile on a sorted 1D list (NaN already at end) using H&F method."""
    _max = _builtin_max
    _min = _builtin_min
    n = len(sorted_vals)
    if n == 0:
        return float('nan')
    # Check for NaN (sorted to end)
    n_nan = sum(1 for v in sorted_vals if v != v)
    n_valid = n - n_nan
    if n_valid == 0:
        return float('nan')
    # NaN propagation: if any NaN exists → result is NaN
    if n_nan > 0:
        return float('nan')

    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[n_valid - 1])

    def _lerp(a, b, t):
        return a + t * (b - a)

    def _interp_at(h):
        h = _max(0.0, _min(h, n_valid - 1.0))
        lo = int(_math.floor(h))
        hi = int(_math.ceil(h))
        if lo == hi:
            return float(sorted_vals[lo])
        return _lerp(float(sorted_vals[lo]), float(sorted_vals[hi]), h - lo)

    if method == 'linear':
        return _interp_at(q * (n_valid - 1))
    elif method == 'lower':
        i = _min(int(_math.floor(q * (n_valid - 1))), n_valid - 1)
        return float(sorted_vals[i])
    elif method == 'higher':
        i = _min(int(_math.ceil(q * (n_valid - 1))), n_valid - 1)
        return float(sorted_vals[i])
    elif method == 'midpoint':
        lo = int(_math.floor(q * (n_valid - 1)))
        hi = _min(int(_math.ceil(q * (n_valid - 1))), n_valid - 1)
        return _lerp(float(sorted_vals[lo]), float(sorted_vals[hi]), 0.5)
    elif method == 'nearest':
        h = q * (n_valid - 1)
        lo = int(_math.floor(h))
        hi = int(_math.ceil(h))
        frac = h - lo
        if frac < 0.5:
            return float(sorted_vals[lo])
        elif frac > 0.5:
            return float(sorted_vals[hi])
        else:  # tie: round half to even
            return float(sorted_vals[lo if lo % 2 == 0 else hi])
    elif method == 'inverted_cdf':
        # H&F 1: 0-indexed = ceil(q*n) - 1
        idx = _max(int(_math.ceil(q * n_valid)) - 1, 0)
        return float(sorted_vals[_min(idx, n_valid - 1)])
    elif method == 'averaged_inverted_cdf':
        # H&F 2: if q*n integer → average neighbors; else like inverted_cdf
        h = q * n_valid
        h_floor = _math.floor(h)
        if h != h_floor:
            idx = _max(int(_math.ceil(h)) - 1, 0)
            return float(sorted_vals[_min(idx, n_valid - 1)])
        else:
            i_lo = _max(int(h) - 1, 0)
            i_hi = _min(int(h), n_valid - 1)
            return _lerp(float(sorted_vals[i_lo]), float(sorted_vals[i_hi]), 0.5)
    elif method == 'closest_observation':
        # H&F 3: round half to odd
        h = q * n_valid
        h_floor = int(_math.floor(h))
        frac = h - h_floor
        if frac == 0.5:
            # prefer odd (1-indexed): h_floor is the lower 1-indexed value
            if h_floor % 2 == 1:  # h_floor is odd (1-indexed) → 0-indexed = h_floor-1
                idx = h_floor - 1
            else:
                idx = h_floor
        else:
            idx = _max(int(_math.ceil(h)) - 1, 0)
        return float(sorted_vals[_min(_max(idx, 0), n_valid - 1)])
    elif method == 'interpolated_inverted_cdf':
        # H&F 4: h (0-indexed) = q*n - 1
        return _interp_at(_max(q * n_valid - 1, 0))
    elif method == 'hazen':
        # H&F 5: h (0-indexed) = q*n - 0.5
        return _interp_at(_max(q * n_valid - 0.5, 0))
    elif method == 'weibull':
        # H&F 6: h (0-indexed) = q*(n+1) - 1
        return _interp_at(_max(_min(q * (n_valid + 1) - 1, n_valid - 1), 0))
    elif method == 'median_unbiased':
        # H&F 8: h (0-indexed) = q*(n+1/3)+1/3 - 1
        return _interp_at(_max(_min(q * (n_valid + 1.0 / 3.0) + 1.0 / 3.0 - 1, n_valid - 1), 0))
    elif method == 'normal_unbiased':
        # H&F 9: h (0-indexed) = q*(n+0.25)+0.375 - 1
        return _interp_at(_max(_min(q * (n_valid + 0.25) + 0.375 - 1, n_valid - 1), 0))
    else:
        raise ValueError("method '{}' is not recognized".format(method))


def _quantile_along_axis(a, q, axis, method):
    """Apply quantile computation along a single axis using Python implementation."""
    from ._manipulation import moveaxis
    import numpy as _np

    n = a.shape[axis]
    # Move reduction axis to last position
    a_moved = moveaxis(a, axis, -1)
    orig_shape = a_moved.shape[:-1]
    # Flatten all but last axis
    if len(orig_shape) == 0:
        # Result is scalar
        vals = a_moved.flatten().tolist()
        vals.sort(key=lambda x: (x != x, x))  # NaN to end
        return asarray(_compute_quantile_1d(vals, q, method))
    a_2d = a_moved.reshape(-1, n)
    m = a_2d.shape[0]
    results = []
    for i in range(m):
        row = a_2d[i].tolist()
        row.sort(key=lambda x: (x != x, x))  # NaN to end
        results.append(_compute_quantile_1d(row, q, method))
    return asarray(results).reshape(orig_shape)


def _quantile_tuple_axis(a, q, axes, method):
    """Apply quantile over multiple axes by collapsing them first."""
    from ._manipulation import moveaxis
    import numpy as _np

    # Move all reduction axes to the end, then reshape to 2D
    ndim = a.ndim
    axes_sorted = sorted(axes)
    other_axes = [i for i in range(ndim) if i not in axes_sorted]
    # Transpose: other_axes first, then reduction axes
    perm = other_axes + axes_sorted
    a_t = a.transpose(perm)
    result_shape = tuple(a.shape[ax] for ax in other_axes)
    n_reduce = 1
    for ax in axes_sorted:
        n_reduce *= a.shape[ax]

    if len(result_shape) == 0:
        vals = a_t.flatten().tolist()
        vals.sort(key=lambda x: (x != x, x))
        return asarray(_compute_quantile_1d(vals, q, method))

    a_2d = a_t.reshape(-1, n_reduce)
    m = a_2d.shape[0]
    results = []
    for i in range(m):
        row = a_2d[i].tolist()
        row.sort(key=lambda x: (x != x, x))
        results.append(_compute_quantile_1d(row, q, method))
    return asarray(results).reshape(result_shape)


_NON_INTERPOLATING_METHODS = frozenset([
    'inverted_cdf', 'closest_observation', 'lower', 'higher', 'nearest'
])


def _get_quantile_result_dtype(a, method):
    """Get the expected output dtype for quantile given input array and method."""
    if method in _NON_INTERPOLATING_METHODS:
        # Preserve input dtype (object → float64)
        if a.dtype.kind == 'O':
            return 'float64'
        return str(a.dtype)
    else:
        # Interpolating methods:
        # integer → float64; float → same; object → float64
        if a.dtype.kind in ('i', 'u'):
            return 'float64'
        elif a.dtype.kind == 'f':
            return str(a.dtype)
        else:
            return 'float64'


def _quantile_core(a, q, axis, method, keepdims, orig_axis_for_keepdims=None):
    """Core quantile computation dispatching to Rust (linear) or Python (other methods).
    axis can be None, int (already normalized), or list of ints (already normalized)."""
    import numpy as _np

    is_tuple_axis = isinstance(axis, list)

    if is_tuple_axis:
        # Tuple axis: must collapse all axes at once (sequential reduction is wrong)
        result = _quantile_tuple_axis(a, q, axis, method)
    elif method == 'linear':
        # Use Rust backend for linear interpolation (fast path)
        result = _native.quantile(a, q, axis)
        if not isinstance(result, ndarray):
            result = asarray(result)
        # Apply NaN mask for float arrays (Rust doesn't always propagate NaN)
        if a.dtype.kind == 'f':
            result = _apply_nan_mask(result, a, axis)
    else:
        # Python implementation for non-linear methods
        if axis is None:
            vals = a.flatten().tolist()
            vals.sort(key=lambda x: (x != x, x))
            result = asarray(_compute_quantile_1d(vals, q, method))
        else:
            result = _quantile_along_axis(a, q, axis, method)

    # Apply result dtype based on method and input dtype
    target_dtype = _get_quantile_result_dtype(a, method)
    if not isinstance(result, ndarray):
        result = asarray(result)
    try:
        result = result.astype(target_dtype)
    except Exception:
        pass

    if keepdims:
        kd_axis = orig_axis_for_keepdims if orig_axis_for_keepdims is not None else axis
        result = _apply_keepdims(result, a, kd_axis)
    return result


def _apply_nan_mask(result, a, axis):
    """Set result positions to NaN where the input slice has any NaN."""
    import numpy as _np
    if a.dtype.kind != 'f':
        return result
    # Compute where NaN exists in input
    isnan_a = _np.isnan(a)
    has_nan = _np.any(isnan_a, axis=axis if not isinstance(axis, list) else tuple(axis))
    if not isinstance(has_nan, ndarray):
        has_nan = asarray(has_nan)
    # If has_nan is a scalar
    if has_nan.ndim == 0:
        if bool(has_nan.tolist()):
            if isinstance(result, ndarray):
                if result.ndim == 0:
                    return asarray(float('nan'))
                return full(result.shape, float('nan'))
            return asarray(float('nan'))
        return result
    # has_nan is an array: set NaN where True
    if not isinstance(result, ndarray):
        result = asarray(result)
    result_f = result.astype('float64')
    # Use fancy indexing to set NaN
    has_nan_list = has_nan.flatten().tolist()
    res_list = result_f.flatten().tolist()
    out_list = [float('nan') if hn else rv for hn, rv in zip(has_nan_list, res_list)]
    return asarray(out_list).reshape(result_f.shape)


def _validate_q_range(q, is_percentile=False):
    """Validate q is in valid range. Raises ValueError if out of bounds."""
    import math as _m
    lo, hi = (0.0, 100.0) if is_percentile else (0.0, 1.0)
    label = "Percentiles" if is_percentile else "Quantiles"
    rng = "[0, 100]" if is_percentile else "[0, 1]"
    def _check(v):
        v = float(v)
        if _m.isnan(v) or v < lo or v > hi:
            raise ValueError("{} must be in the range {}".format(label, rng))
    if hasattr(q, '__iter__') and not isinstance(q, ndarray):
        for qi in q:
            _check(qi)
    elif isinstance(q, ndarray):
        for qi in q.flatten().tolist():
            _check(qi)
    else:
        _check(q)


def _normalize_quantile_axis(axis, ndim):
    orig_axis = axis
    if isinstance(axis, (list, tuple)):
        axis = _normalize_axes_tuple(axis, ndim)
    elif axis is not None:
        axis = _normalize_axis_arg(axis, ndim)
    return axis, orig_axis


def _normalize_q_array(q, *, scale=1.0):
    q_arr = asarray(q, dtype='float64') if not isinstance(q, ndarray) else q.astype('float64')
    q_flat = [float(qi) * scale for qi in q_arr.flatten().tolist()]
    return q_arr, q_flat


def _stack_quantile_results(a, q_arr, results, *, keepdims, orig_axis, target_dtype):
    import numpy as _np

    results = [r if isinstance(r, ndarray) else asarray(r) for r in results]
    if keepdims:
        results = [_apply_keepdims(r, a, orig_axis) for r in results]

    stacked = _np.stack(results)
    try:
        stacked = stacked.astype(target_dtype)
    except Exception:
        pass
    if q_arr.ndim > 1:
        stacked = stacked.reshape(q_arr.shape + stacked.shape[1:])
    return stacked


def _quantile_dispatch(a, q, axis, out, method, keepdims, *, q_scale, result_dtype):
    axis, orig_axis = _normalize_quantile_axis(axis, a.ndim)

    if isinstance(q, (list, tuple, ndarray)):
        q_arr, q_flat = _normalize_q_array(q, scale=q_scale)
        results = [_quantile_core(a, qi, axis, method, keepdims=False) for qi in q_flat]
        stacked = _stack_quantile_results(
            a, q_arr, results, keepdims=keepdims, orig_axis=orig_axis, target_dtype=result_dtype
        )
        if out is not None:
            out[...] = stacked
            return out
        return stacked

    result = _quantile_core(
        a, float(q) * q_scale, axis, method, keepdims=keepdims, orig_axis_for_keepdims=orig_axis
    )
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    if out is not None:
        out[...] = result
        return out
    return result


def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)

    # Validate method
    if method not in _VALID_QUANTILE_METHODS:
        raise ValueError("method '{}' is not recognized".format(method))

    # Validate q range
    _validate_q_range(q, is_percentile=False)

    # Validate weights
    if weights is not None:
        if method != 'inverted_cdf':
            raise ValueError("Only method 'inverted_cdf' supports weights")
    return _quantile_dispatch(
        a, q, axis, out, method, keepdims,
        q_scale=1.0,
        result_dtype=_get_quantile_result_dtype(a, method),
    )


def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)

    # Validate method
    if method not in _VALID_QUANTILE_METHODS:
        raise ValueError("method '{}' is not recognized".format(method))

    # Validate q range (0-100)
    _validate_q_range(q, is_percentile=True)

    # Validate weights
    if weights is not None:
        if method != 'inverted_cdf':
            raise ValueError("Only method 'inverted_cdf' supports weights")
    return _quantile_dispatch(
        a, q, axis, out, method, keepdims,
        q_scale=0.01,
        result_dtype=_get_quantile_result_dtype(a, method),
    )


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    import warnings as _warnings
    if not isinstance(a, ndarray):
        a = array(a)

    # Normalize axis
    orig_axis = axis
    if isinstance(axis, (list, tuple)):
        axis = _normalize_axes_tuple(axis, a.ndim)
    elif axis is not None:
        axis = _normalize_axis_arg(axis, a.ndim)

    # Handle empty array
    if a.size == 0:
        _warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
        _warnings.warn("invalid value encountered in scalar divide", RuntimeWarning, stacklevel=2)
        if axis is None or isinstance(axis, list):
            result = asarray(float('nan'))
        else:
            res_shape = list(a.shape)
            res_shape.pop(int(axis))
            result = full(tuple(res_shape), float('nan')) if tuple(res_shape) else asarray(float('nan'))
        if keepdims:
            result = _apply_keepdims(result, a, orig_axis)
        if out is not None:
            out[...] = result
            return out
        return result

    try:
        result = _quantile_core(a, 0.5, axis, 'linear', keepdims=keepdims,
                                orig_axis_for_keepdims=orig_axis)
    except Exception as e:
        if 'empty' in str(e).lower():
            _warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
            result = asarray(float('nan'))
            if keepdims:
                result = _apply_keepdims(result, a, orig_axis)
        else:
            raise

    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    if out is not None:
        out[...] = result
        return out
    return result


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None):
    if not isinstance(m, ndarray):
        m = array(m)
    _ddof = ddof if ddof is not None else (0 if bias else 1)

    # Handle empty array: return NaN
    if m.size == 0:
        import warnings as _w
        _w.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        if m.ndim <= 1 or m.size == 0:
            return float('nan')
        # shape (p, 0) → return (p, p) NaN matrix
        p = m.shape[0] if rowvar else m.shape[1]
        result = full((p, p), float('nan'))
        if dtype is not None:
            result = result.astype(str(dtype))
        return result

    # Handle fweights/aweights in Python since Rust backend doesn't support them
    if fweights is not None or aweights is not None:
        result = _cov_weighted(m, y=y, rowvar=rowvar, bias=bias, ddof=_ddof,
                               fweights=fweights, aweights=aweights)
        if dtype is not None:
            result = result.astype(str(dtype))
        return result

    # Determine number of observations to check ddof validity
    _m2 = m if rowvar else m.T
    _num_obs = _m2.shape[1] if _m2.ndim >= 2 else _m2.shape[0]
    if _ddof >= _num_obs:
        import warnings as _w
        _w.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)

    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        result = _native.cov(m, y, rowvar, _ddof)
    else:
        result = _native.cov(m, None, rowvar, _ddof)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result


def _cov_weighted(m, y=None, rowvar=True, bias=False, ddof=1, fweights=None, aweights=None):
    """Weighted covariance following NumPy's algorithm."""
    X = array(m)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif not rowvar:
        X = X.T
    if y is not None:
        y = array(y)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        elif not rowvar:
            y = y.T
        X = concatenate([X, y], axis=0)

    num_vars, num_obs = X.shape[0], X.shape[1]

    # Build combined weight vector
    w = None
    w_type = 'f'  # 'f'=fweights, 'a'=aweights, 'fa'=both

    if fweights is not None:
        fw = array(fweights).flatten()
        if str(fw.dtype) not in ('int32', 'int64'):
            # Check if they're integer-valued
            fw_list = fw.tolist()
            for v in fw_list:
                if v != int(v):
                    raise TypeError("fweights must be integer")
            fw = fw.astype('int64')
        if fw.shape[0] != num_obs:
            raise RuntimeError("incompatible numbers of samples and fweights")
        fw_list = fw.tolist()
        _fw_neg = False
        for _fwv in fw_list:
            if _fwv < 0:
                _fw_neg = True
                break
        if _fw_neg:
            raise ValueError("fweights cannot be negative")
        w = fw.astype('float64')

    if aweights is not None:
        aw = array(aweights).flatten()
        if aw.ndim != 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aw.shape[0] != num_obs:
            raise RuntimeError("incompatible numbers of samples and aweights")
        aw_list = aw.tolist()
        _aw_neg = False
        for _awv in aw_list:
            if _awv < 0:
                _aw_neg = True
                break
        if _aw_neg:
            raise ValueError("aweights cannot be negative")
        aw = aw.astype('float64')
        if w is None:
            w = aw
        else:
            w = w * aw

    # Compute weighted mean of each row using np-level functions
    import numpy as _np_local
    if w is None:
        avg = _np_local.mean(X, axis=1)
    else:
        w_sum = float(_np_local.sum(w))
        avg = _np_local.sum(X * w, axis=1) / w_sum

    # Center: subtract row means
    X_centered = X - avg.reshape((-1, 1))

    # Compute effective weight and denominator
    # Use conjugate transpose for complex Hermitian covariance
    _is_complex = 'complex' in str(X.dtype)
    if w is None:
        norm = float(num_obs - ddof)
        if _is_complex:
            X_T = X_centered.conj().T
        else:
            X_T = X_centered.T
    else:
        w_sum = float(_np_local.sum(w))
        if aweights is None:
            norm = float(w_sum - ddof)
        else:
            aw_arr = array(aweights).flatten().astype('float64')
            norm = float(w_sum - ddof * float(_np_local.sum(w * aw_arr)) / w_sum)
        if _is_complex:
            X_T = (X_centered * w).conj().T
        else:
            X_T = (X_centered * w).T

    # C = X_centered @ X_T / norm  (use matmul)
    c = _np_local.dot(X_centered, X_T) / norm
    return c


def corrcoef(x, y=None, rowvar=True, dtype=None):
    if not isinstance(x, ndarray):
        x = array(x)

    # Handle empty
    if x.size == 0:
        import warnings as _w
        _w.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        if x.ndim <= 1:
            return float('nan')
        p = x.shape[0] if rowvar else x.shape[1]
        if p == 0:
            return array([]).reshape(0, 0)
        result = full((p, p), float('nan'))
        if dtype is not None:
            result = result.astype(str(dtype))
        return result

    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        result = _native.corrcoef(x, y, rowvar)
    else:
        result = _native.corrcoef(x, None, rowvar)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result


def average(a, axis=None, weights=None, returned=False, keepdims=False):
    """Compute the weighted average along the specified axis."""
    a = asarray(a)
    if weights is None:
        avg = mean(a, axis=axis, keepdims=keepdims)
        if returned:
            if keepdims:
                if axis is None:
                    wt_shape = tuple(1 for _ in range(a.ndim)) if a.ndim > 0 else ()
                    return avg, full(wt_shape, float(a.size))
                else:
                    return avg, full(avg.shape, float(a.shape[axis]))
            else:
                if axis is None:
                    return avg, float(a.size)
                return avg, full(avg.shape, float(a.shape[axis]))
        return avg
    weights = asarray(weights)
    # Validate and broadcast weights against a
    if weights.shape != a.shape:
        if axis is None:
            raise TypeError(
                "Axis must be specified when shapes of a and weights differ.")
        # Normalize axis (may be tuple or int)
        _axis = axis
        if isinstance(_axis, int):
            ax = _axis if _axis >= 0 else _axis + a.ndim
            # 1D weights matching the axis dim: reshape to broadcast
            if weights.ndim == 1 and weights.shape[0] == a.shape[ax]:
                new_shape = [1] * a.ndim
                new_shape[ax] = weights.shape[0]
                weights = weights.reshape(new_shape)
            # weights.ndim matches the axes being reduced: broadcast to a.shape
            try:
                import numpy as _np_local
                weights = _np_local.broadcast_to(weights, a.shape)
            except Exception:
                pass
        elif isinstance(_axis, tuple):
            # Multi-axis averaging: weights must be broadcastable to a.shape
            try:
                import numpy as _np_local
                weights = _np_local.broadcast_to(weights, a.shape)
            except Exception:
                raise ValueError(
                    "Shape of weights must be consistent with shape of a "
                    "along specified axis")
    wsum = sum(a * weights, axis=axis, keepdims=keepdims)
    wt = sum(weights, axis=axis, keepdims=keepdims)
    avg = wsum / wt
    if returned:
        return avg, wt
    return avg


def _is_complex_nan(v):
    """Check if v is a complex NaN (stored as tuple)."""
    if isinstance(v, tuple) and len(v) == 2:
        re, im = v
        return (re != re) or (im != im)
    return False


def _nan_quantile_1d(vals_flat, q):
    """Compute quantile of a 1D array, ignoring NaNs. Returns scalar."""
    # Check if this is a complex dtype
    is_complex = False
    if vals_flat.size > 0:
        v0 = vals_flat[0] if vals_flat.ndim > 0 else vals_flat[()]
        is_complex = isinstance(v0, tuple) and len(v0) == 2

    if is_complex:
        # For complex: only support all-NaN case (return complex NaN)
        all_nan = True
        for i in range(vals_flat.size):
            v = vals_flat[i]
            if not _is_complex_nan(v) and v == v:
                all_nan = False
                break
        if all_nan:
            return complex(float('nan'), float('nan'))
        raise TypeError("quantile not supported for complex arrays")

    vals = []
    for i in range(vals_flat.size):
        v = vals_flat[i]
        if v == v:  # not NaN
            vals.append(float(v))
    if len(vals) == 0:
        return float('nan')
    vals.sort()
    idx = q * (len(vals) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(vals):
        return vals[lo]
    frac = idx - lo
    if frac == 0.0:  # avoid inf * 0.0 = nan
        return vals[lo]
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _nan_quantile_impl(a, q, axis):
    """Helper for nanmedian/nanpercentile/nanquantile with axis support.
    Supports integer axis, tuple axis, and axis=None.
    Returns Python float for scalar results, ndarray for array results."""
    import numpy as _np
    a = asarray(a)

    if axis is None:
        flat = a.flatten()
        if flat.size == 0:
            return float('nan')
        return _nan_quantile_1d(flat, q)

    # Normalize axis to a tuple of non-negative ints
    if isinstance(axis, (tuple, list)):
        try:
            from numpy._core.numeric import normalize_axis_tuple
            axes = tuple(normalize_axis_tuple(axis, a.ndim))
        except Exception as e:
            from numpy.exceptions import AxisError
            raise AxisError(str(e)) from e
    else:
        orig_ax = int(axis)
        ax = orig_ax
        if ax < 0:
            ax += a.ndim
        if ax < 0 or ax >= a.ndim:
            from numpy.exceptions import AxisError
            raise AxisError(
                f"axis {orig_ax} is out of bounds for array of dimension {a.ndim}"
            )
        axes = (ax,)

    if a.ndim == 0:
        return _nan_quantile_1d(a.flatten(), q)

    # Move all target axes to the end
    a_moved = _np.moveaxis(a, list(axes), list(range(a.ndim - len(axes), a.ndim)))
    other_shape = a_moved.shape[:a.ndim - len(axes)]
    axis_len = 1
    for s in a_moved.shape[a.ndim - len(axes):]:
        axis_len *= s

    # Compute number of output slices (product of non-reduced dims)
    other_size = 1
    for s in other_shape:
        other_size *= s

    if axis_len == 0:
        # Reducing along empty axis — all slices yield NaN
        if not other_shape:
            return float('nan')
        return _np.full(other_shape, float('nan'))

    if other_size == 0:
        # No slices to iterate — return appropriately-shaped empty array
        if not other_shape:
            return float('nan')
        return _np.zeros(other_shape)

    flat_2d = a_moved.reshape((other_size, axis_len))
    results = []
    for i in range(other_size):
        row = flat_2d[i]
        results.append(
            _nan_quantile_1d(row if isinstance(row, ndarray) else asarray(row), q)
        )

    if not other_shape:
        return results[0]  # scalar result

    result = array(results)
    if len(other_shape) > 1:
        result = result.reshape(other_shape)
    return result


def _nanmedian_result_dtype(a):
    """Get the result dtype for nanmedian (preserves float/complex, int→float64)."""
    if isinstance(a, ndarray):
        dt = str(a.dtype)
        if dt.startswith(('int', 'uint', 'bool')):
            return 'float64'
        return dt
    return 'float64'


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Compute the median along the specified axis, ignoring NaNs."""
    import warnings as _warnings
    import numpy as _npw
    a = asarray(a)
    _result_dt = _nanmedian_result_dtype(a)
    result = _nan_quantile_impl(a, 0.5, axis)
    # Use hasattr check to handle both ndarray and _ObjectArray (not isinstance)
    _is_array_like = hasattr(result, 'ndim') and result.ndim > 0
    if not _is_array_like:
        # scalar result
        _v = result[()] if hasattr(result, '__getitem__') and hasattr(result, 'ndim') else result
        _is_nan = (_v != _v) if not isinstance(_v, complex) else (
            _v.real != _v.real or _v.imag != _v.imag)
        if _is_nan:
            _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
            # NaN result: 0-d ndarray so np.isnan(result).all() works
            result = _scalar_result(result, a, _result_dt)
        elif axis is None:
            # axis=None: return 0-d ndarray (matches np.median behavior)
            result = _scalar_result(_v, a, _result_dt)
        else:
            # axis specified: return numpy scalar (isscalar=True, has .ndim/.dtype)
            _ctor = getattr(_npw, _result_dt, None)
            if _ctor is not None:
                try:
                    result = _ctor(_v)
                except Exception:
                    result = _scalar_result(_v, a, _result_dt)
            else:
                result = _scalar_result(_v, a, _result_dt)
    else:
        # array result - emit one warning per all-NaN slice
        # For empty arrays, emit at most 1 warning (numpy semantics)
        try:
            _nan_mask = _npw.isnan(result)
            _nan_count = int(_npw.count_nonzero(_nan_mask))
        except TypeError:
            _nan_count = 0
        _warn_count = 1 if (a.size == 0 and _nan_count > 0) else _nan_count
        for _ in range(_warn_count):
            _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        if str(result.dtype) != _result_dt:
            try:
                result = result.astype(_result_dt)
            except Exception:
                pass
    if keepdims:
        result = _apply_keepdims(result if isinstance(result, ndarray) else asarray(result), a, axis)
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def _nanq_warn_and_wrap(result, a, _nan_result_dtype, axis=None):
    """Emit All-NaN warning if needed and wrap scalar result.
    axis=None → return 0-d ndarray (matches np.quantile behavior).
    axis given → return numpy scalar (isscalar=True).
    NaN → always return 0-d ndarray (for np.isnan(x).all() check)."""
    import warnings as _w
    import numpy as _npw
    _is_array_like = hasattr(result, 'ndim') and result.ndim > 0
    if not _is_array_like:
        _v = result[()] if hasattr(result, '__getitem__') and hasattr(result, 'ndim') else result
        _is_nan = (_v != _v) if not isinstance(_v, complex) else (
            _v.real != _v.real or _v.imag != _v.imag)
        if _is_nan:
            _w.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
            # NaN result: return 0-d ndarray so np.isnan(result).all() works
            result = _scalar_result(_v, a, _nan_result_dtype)
        elif axis is None:
            # axis=None: return 0-d ndarray (matches np.quantile behavior)
            result = _scalar_result(_v, a, _nan_result_dtype)
        else:
            # axis specified: return numpy scalar (isscalar=True, has .ndim/.dtype)
            _ctor = getattr(_npw, _nan_result_dtype, None)
            if _ctor is not None:
                try:
                    result = _ctor(_v)
                except Exception:
                    result = _scalar_result(_v, a, _nan_result_dtype)
            else:
                result = _scalar_result(_v, a, _nan_result_dtype)
    else:
        # array result - emit one warning per all-NaN slice
        # For empty input arrays, emit at most 1 warning (numpy semantics)
        try:
            _nan_mask = _npw.isnan(result)
            _nan_count = int(_npw.count_nonzero(_nan_mask))
        except TypeError:
            _nan_count = 0
        _warn_count = 1 if (a.size == 0 and _nan_count > 0) else _nan_count
        for _ in range(_warn_count):
            _w.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
        if str(result.dtype) != _nan_result_dtype:
            try:
                result = result.astype(_nan_result_dtype)
            except Exception:
                pass
    return result


def _nanq_impl(a, q_scale, axis, keepdims, out, _rdt):
    """Shared implementation for nanpercentile / nanquantile (scalar q)."""
    result = _nan_quantile_impl(a, q_scale, axis)
    result = _nanq_warn_and_wrap(result, a, _rdt, axis=axis)
    if keepdims:
        result = _apply_keepdims(result if isinstance(result, ndarray) else asarray(result), a, axis)
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def _nanq_list_impl(a, q_arr, q_scale, axis, keepdims, out, _rdt):
    """Shared implementation for nanpercentile / nanquantile (array q).
    q_arr: original q ndarray (for shape info).
    q_scale: q values already in [0,1] range.
    """
    import numpy as _np
    q_flat = q_scale.flatten().tolist()
    results = [_nanq_warn_and_wrap(_nan_quantile_impl(a, float(qi), axis), a, _rdt, axis=axis)
               for qi in q_flat]
    # Wrap results as ndarrays
    results = [r if isinstance(r, ndarray) else asarray(r) for r in results]
    if keepdims:
        results = [_apply_keepdims(r, a, axis) for r in results]
        # Stack along new leading axis
        result = _np.stack(results)
        # Reshape leading dim to q_arr.shape
        if q_arr.ndim > 1:
            result = result.reshape(q_arr.shape + result.shape[1:])
    else:
        if axis is None:
            result = array([float(r) if r.ndim == 0 else r for r in results])
            if q_arr.ndim > 1:
                result = result.reshape(q_arr.shape)
        else:
            result = _np.stack(results)
            if q_arr.ndim > 1:
                result = result.reshape(q_arr.shape + result.shape[1:])
    if out is not None and isinstance(out, ndarray):
        _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
        return out
    return result


def _nan_quantile_dispatch(a, q, axis, out, keepdims, *, q_scale):
    rdt = _nanmedian_result_dtype(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_arr, q_flat = _normalize_q_array(q, scale=q_scale)
        q_scaled = asarray(q_flat, dtype='float64').reshape(q_arr.shape)
        return _nanq_list_impl(a, q_arr, q_scaled, axis, keepdims, out, rdt)
    return _nanq_impl(a, float(q) * q_scale, axis, keepdims, out, rdt)


def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    """Compute the qth percentile, ignoring NaNs."""
    a = asarray(a)
    return _nan_quantile_dispatch(a, q, axis, out, keepdims, q_scale=0.01)


def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    """Compute the qth quantile, ignoring NaNs."""
    a = asarray(a)
    return _nan_quantile_dispatch(a, q, axis, out, keepdims, q_scale=1.0)


def ediff1d(ary, to_end=None, to_begin=None):
    """The differences between consecutive elements of an array."""
    from ._type_promotion import can_cast
    ary = asarray(ary).flatten()
    dtype_str = str(ary.dtype)
    n = ary.size
    # Validate to_begin/to_end casting compatibility (same_kind rule)
    if to_begin is not None:
        tb = asarray(to_begin).flatten()
        if not can_cast(tb, dtype_str, casting='same_kind'):
            raise TypeError(
                "dtype of `to_begin` must be compatible (same kind) with "
                "the dtype of `ary`. "
                "dtype of `to_begin` must be compatible with int")
    if to_end is not None:
        te = asarray(to_end).flatten()
        if not can_cast(te, dtype_str, casting='same_kind'):
            raise TypeError(
                "dtype of `to_end` must be compatible (same kind) with "
                "the dtype of `ary`. "
                "dtype of `to_end` must be compatible with int")
    parts = []
    if to_begin is not None:
        parts.append(asarray(to_begin).flatten().astype(dtype_str))
    if n > 1:
        diff_vals = [ary[i] - ary[i - 1] for i in range(1, n)]
        parts.append(array(diff_vals))
    elif n == 1:
        parts.append(array([], dtype=dtype_str))
    else:
        parts.append(array([], dtype=dtype_str))
    if to_end is not None:
        parts.append(asarray(to_end).flatten().astype(dtype_str))
    import numpy as _np
    result = _np.concatenate(parts) if parts else array([], dtype=dtype_str)
    return result.astype(dtype_str)


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    a = _ensure_reduction_array(a)
    input_dt = str(a.dtype)
    a = _apply_where_mask(a, where, false_fill=float('-inf'))
    if axis is not None:
        result = a.max(axis, keepdims)
    else:
        result = a.max(None, keepdims)
    result = _scalar_result(result, a) if not isinstance(result, ndarray) else result
    if isinstance(result, ndarray) and str(result.dtype) != input_dt and not _is_boxed_dtype_name(input_dt):
        try:
            result = result.astype(input_dt)
        except Exception:
            pass
    if initial is not None:
        import numpy as _np
        result = _np.maximum(result, initial)
        if isinstance(result, ndarray) and str(result.dtype) != input_dt and not _is_boxed_dtype_name(input_dt):
            try:
                result = result.astype(input_dt)
            except Exception:
                pass
    copied = _copy_reduction_out(out, result)
    if copied is not None:
        return copied
    return result


amax = max


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    a = _ensure_reduction_array(a)
    input_dt = str(a.dtype)
    a = _apply_where_mask(a, where, false_fill=float('inf'))
    if axis is not None:
        result = a.min(axis, keepdims)
    else:
        result = a.min(None, keepdims)
    result = _scalar_result(result, a) if not isinstance(result, ndarray) else result
    if isinstance(result, ndarray) and str(result.dtype) != input_dt and not _is_boxed_dtype_name(input_dt):
        try:
            result = result.astype(input_dt)
        except Exception:
            pass
    if initial is not None:
        import numpy as _np
        result = _np.minimum(result, initial)
        if isinstance(result, ndarray) and str(result.dtype) != input_dt and not _is_boxed_dtype_name(input_dt):
            try:
                result = result.astype(input_dt)
            except Exception:
                pass
    copied = _copy_reduction_out(out, result)
    if copied is not None:
        return copied
    return result


amin = min


def argmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    result = a.argmax(axis)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, 'int64')
    return result


def argmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    result = a.argmin(axis)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, 'int64')
    return result


def ptp(a, axis=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return a.max(axis=axis, keepdims=keepdims) - a.min(axis=axis, keepdims=keepdims)


def argwhere(a):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.argwhere(a)


def _unravel_linear_indices(indices, shape):
    if not shape:
        return (array(indices, dtype="int64"),)
    coords = [[] for _ in shape]
    for linear_idx in indices:
        rem = int(linear_idx)
        coord = [0] * len(shape)
        for axis in range(len(shape) - 1, -1, -1):
            dim = shape[axis]
            coord[axis] = rem % dim
            rem //= dim
        for axis, value in enumerate(coord):
            coords[axis].append(value)
    return tuple(array(values, dtype="int64") for values in coords)


def _flatten_truth_values(values):
    if isinstance(values, (list, tuple)):
        result = []
        for value in values:
            result.extend(_flatten_truth_values(value))
        return result
    return [values]


def _stable_truth_vector(values):
    flat_values = _flatten_truth_values(values)
    first_pass = [bool(v) for v in flat_values]
    second_pass = [bool(v) for v in flat_values]
    if first_pass != second_pass:
        raise RuntimeError("number of non-zero array elements changed during function execution")
    return first_pass


def _structured_nonzero_linear_indices(a):
    flat_size = 1
    for dim in a.shape:
        flat_size *= dim
    truth = [False] * flat_size
    for name in a.dtype.names:
        field_values = a[name].flatten().tolist()
        for idx, value in enumerate(field_values):
            truth[idx] = truth[idx] or bool(value)
    return [idx for idx, value in enumerate(truth) if value]


def nonzero(a):
    a = _coerce_native_boxed_operand(a)
    if isinstance(a, _ObjectArray):
        if len(a.shape) == 0:
            raise ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(a).nonzero() instead.")
        truth = _stable_truth_vector(a.flatten().tolist())
        indices = [i for i, value in enumerate(truth) if value]
        return _unravel_linear_indices(indices, a.shape)
    import numpy as _np
    if isinstance(a, _np.StructuredArray):
        if len(a.shape) == 0:
            raise ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(a).nonzero() instead.")
        return _unravel_linear_indices(_structured_nonzero_linear_indices(a), a.shape)
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.ndim == 0:
        raise ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(a).nonzero() instead.")
    return _native.nonzero(a)


def count_nonzero(a, axis=None, *, keepdims=False):
    import numpy as _np
    if isinstance(a, _np.StructuredArray) and axis is None:
        result = len(_structured_nonzero_linear_indices(a))
        if keepdims:
            return array([float(result)]).reshape((1,) * len(a.shape)).astype("int64")
        return result
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Empty tuple axis means element-wise (no reduction)
    if isinstance(axis, tuple) and len(axis) == 0:
        return a.astype("bool")
    # Validate axis
    if axis is not None and not isinstance(axis, (int, tuple)):
        if isinstance(axis, ndarray):
            raise TypeError("axis must be an integer or a tuple of integers")
        raise TypeError("'{}' object cannot be interpreted as an integer".format(type(axis).__name__))
    if isinstance(axis, tuple):
        # Check for duplicate axes
        normed = []
        for ax in axis:
            n = ax if ax >= 0 else ax + a.ndim
            if n in normed:
                raise ValueError("duplicate value in 'axis'")
            if n < 0 or n >= a.ndim:
                raise AxisError(ax, a.ndim)
            normed.append(n)
    elif isinstance(axis, int):
        n = axis if axis >= 0 else axis + a.ndim
        if n < 0 or n >= a.ndim:
            raise AxisError(axis, a.ndim)
    # Helper: determine if a scalar value is nonzero
    def _is_nonzero(v):
        if isinstance(v, tuple):  # complex (re, im) representation
            return v[0] != 0.0 or v[1] != 0.0
        if isinstance(v, (str, bytes)):  # empty string/bytes is falsy
            return bool(v)
        if isinstance(v, (int, float, bool)):
            return v != 0
        # Arbitrary Python objects (void, datetime64, etc.) — use __bool__
        return bool(v)
    if axis is None:
        if not isinstance(a, ndarray):
            # _ObjectArray (strings, bytes, etc.) — iterate Python-side
            flat = a.flatten().tolist() if hasattr(a, 'flatten') else list(a)
            truth = _stable_truth_vector(flat)
            result = sum(1 for v in truth if v)
            if keepdims:
                return array([float(result)]).reshape((1,) * len(a.shape)).astype("int64")
            return result
        result = _native.count_nonzero(a)
        if keepdims:
            return array([float(result)]).reshape((1,) * a.ndim).astype("int64")
        return result
    # Build a boolean mask (nonzero -> 1.0, zero -> 0.0), then sum along axis
    flat = a.flatten().tolist()
    mask_data = [1.0 if _is_nonzero(v) else 0.0 for v in flat]
    mask = array(mask_data).reshape(a.shape)
    result = mask.sum(axis, keepdims)
    # Convert to integer values
    if isinstance(result, ndarray):
        return result.astype("int64")
    return int(result)


def flatnonzero(a):
    """Return indices of non-zero elements in the flattened array."""
    a = asarray(a).flatten()
    indices_list = []
    for i in range(len(a)):
        if float(a[i]) != 0.0:
            indices_list.append(i)
    return array(indices_list)


def searchsorted(a, v, side="left", sorter=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    scalar_v = not isinstance(v, (ndarray, list, tuple))
    if not isinstance(v, ndarray):
        v = array([v]) if scalar_v else asarray(v)
    result = _native.searchsorted(a, v, side)
    if scalar_v and isinstance(result, ndarray) and result.size == 1:
        return int(float(result.flatten()[0]))
    return result


def _gradient_scalar_spacing(sp):
    """Extract scalar spacing from scalar, 0-d array, or size-1 array."""
    if isinstance(sp, ndarray):
        if sp.ndim > 1:
            raise ValueError("Spacing must be scalars or 1d, not {}d".format(sp.ndim))
        if sp.size == 1:
            return float(sp.flatten()[0])
        return None  # non-scalar, needs array handling
    return float(sp)


def _coerce_gradient_spacing(sp):
    """Normalize one spacing argument to a float or 1-D ndarray."""
    if isinstance(sp, ndarray):
        arr = sp
    elif isinstance(sp, (list, tuple)):
        arr = asarray(sp)
    else:
        return float(sp)

    if arr.ndim > 1:
        raise ValueError("Spacing must be scalars or 1d, not {}d".format(arr.ndim))
    if arr.size == 1:
        return float(arr.flatten()[0])
    return arr


def _resolve_gradient_axes(axis, ndim):
    """Normalize gradient axes and whether the result should stay scalar-axis shaped."""
    if axis is None:
        return list(_builtin_range(ndim)), False
    if isinstance(axis, int):
        return [_normalize_axis_arg(axis, ndim)], True
    return _normalize_axes_tuple(axis, ndim), False


def _resolve_gradient_spacings(varargs, axes, ndim_shape):
    """Normalize gradient spacing arguments to the runtime shape expected by native code."""
    len_axes = len(axes)

    if len(varargs) == 0:
        spacings = [1.0] * len_axes
    elif len(varargs) == 1:
        sp = _coerce_gradient_spacing(varargs[0])
        is_non_scalar_spacing = isinstance(sp, ndarray)
        if is_non_scalar_spacing and len_axes != 1:
            raise TypeError(
                "gradient() only takes 1 non-scalar spacing argument when "
                "axis is not a single integer, but {} axes were specified".format(len_axes)
            )
        spacings = [sp] * len_axes
    elif len(varargs) == len_axes:
        spacings = [_coerce_gradient_spacing(sp) for sp in varargs]
    else:
        raise TypeError(
            "gradient() takes from 1 to {} positional arguments but {} "
            "were given".format(len(ndim_shape) + 1, len(varargs) + 1)
        )

    for i, sp in enumerate(spacings):
        if isinstance(sp, ndarray) and sp.size != ndim_shape[axes[i]]:
            raise ValueError(
                "Spacing array has wrong size {} for axis {} with size {}".format(
                    sp.size, axes[i], ndim_shape[axes[i]]
                )
            )
    return spacings


def _gradient_1d_nonuniform(f_1d, x, edge_order):
    """Compute 1-D gradient with non-uniform coordinate array x."""
    n = len(f_1d)
    import math as _math
    out = [0.0] * n
    # Interior: central differences
    for i in _builtin_range(1, n - 1):
        dx1 = x[i] - x[i - 1]
        dx2 = x[i + 1] - x[i]
        f0 = float(f_1d[i - 1])
        f1 = float(f_1d[i])
        f2 = float(f_1d[i + 1])
        # Weighted central difference for non-uniform grid
        out[i] = (f2 * dx1 ** 2 + (dx2 ** 2 - dx1 ** 2) * f1 - f0 * dx2 ** 2) / (dx1 * dx2 * (dx1 + dx2))
    # Boundaries
    if edge_order == 1:
        out[0] = (float(f_1d[1]) - float(f_1d[0])) / (x[1] - x[0])
        out[-1] = (float(f_1d[-1]) - float(f_1d[-2])) / (x[-1] - x[-2])
    else:
        # 2nd order one-sided (matches NumPy's formula using consecutive diffs)
        dx1 = x[1] - x[0]  # first spacing
        dx2 = x[2] - x[1]  # second spacing
        a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
        b = (dx1 + dx2) / (dx1 * dx2)
        c = -dx1 / (dx2 * (dx1 + dx2))
        out[0] = a * float(f_1d[0]) + b * float(f_1d[1]) + c * float(f_1d[2])
        dx1 = x[-2] - x[-3]  # second-to-last spacing
        dx2 = x[-1] - x[-2]  # last spacing
        a = dx2 / (dx1 * (dx1 + dx2))
        b = -(dx2 + dx1) / (dx1 * dx2)
        c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
        out[-1] = a * float(f_1d[-3]) + b * float(f_1d[-2]) + c * float(f_1d[-1])
    return out


def _gradient_1d_uniform(f_1d, dx, edge_order):
    """Compute 1-D gradient with uniform spacing dx."""
    n = len(f_1d)
    out = [0.0] * n
    # Interior: central differences
    for i in _builtin_range(1, n - 1):
        out[i] = (float(f_1d[i + 1]) - float(f_1d[i - 1])) / (2.0 * dx)
    # Boundaries
    if edge_order == 1:
        out[0] = (float(f_1d[1]) - float(f_1d[0])) / dx
        out[-1] = (float(f_1d[-1]) - float(f_1d[-2])) / dx
    else:
        # 2nd order one-sided: (-3f[0]+4f[1]-f[2])/(2dx)
        out[0] = (-3.0 * float(f_1d[0]) + 4.0 * float(f_1d[1]) - float(f_1d[2])) / (2.0 * dx)
        out[-1] = (float(f_1d[-3]) - 4.0 * float(f_1d[-2]) + 3.0 * float(f_1d[-1])) / (2.0 * dx)
    return out


def _gradient_along_axis(f, axis, spacing, edge_order):
    """Compute gradient of f along the given axis with given spacing."""
    shape = list(f.shape)
    n = shape[axis]
    min_size = edge_order + 1
    if n < min_size:
        raise ValueError(
            "Shape of array too small to calculate a numerical gradient, "
            "at least {} elements are required for edge_order={}".format(
                min_size, edge_order
            )
        )
    # Resolve spacing to scalar or coordinate list
    use_array_spacing = False
    x_list = None
    dx = None
    if isinstance(spacing, ndarray):
        if spacing.ndim > 1:
            raise ValueError(
                "Spacing must be scalars or 1d, not {}d".format(spacing.ndim)
            )
        if spacing.size == 1:
            dx = float(spacing.flatten()[0])
        elif spacing.size == n:
            x_list = spacing.flatten().tolist()
            use_array_spacing = True
        else:
            raise ValueError(
                "Spacing array has wrong size {}, expected 1 or {}".format(
                    spacing.size, n
                )
            )
    elif isinstance(spacing, (list, tuple)):
        if len(spacing) == n:
            x_list = [float(v) for v in spacing]
            use_array_spacing = True
        else:
            raise ValueError(
                "Spacing list has wrong size {}, expected {}".format(len(spacing), n)
            )
    else:
        dx = float(spacing)
    # Allocate output
    out_flat = zeros(f.size).flatten()
    # Iterate over all slices perpendicular to axis
    # Use reshaping: move axis to last, iterate over all leading dims
    from ._manipulation import moveaxis as _moveaxis
    f_moved = _moveaxis(f, axis, f.ndim - 1)
    # f_moved shape: shape without axis, then n
    outer_shape = list(f_moved.shape[:-1])
    outer_size = 1
    for s in outer_shape:
        outer_size *= s
    f_flat = f_moved.reshape([outer_size, n]) if f_moved.ndim > 1 else f_moved.reshape([1, n])
    result_rows = []
    for row_idx in _builtin_range(outer_size):
        row = f_flat[row_idx]
        row_list = row.tolist()
        if use_array_spacing:
            grad_row = _gradient_1d_nonuniform(row_list, x_list, edge_order)
        else:
            grad_row = _gradient_1d_uniform(row_list, dx, edge_order)
        result_rows.append(grad_row)
    # Reassemble
    out_data = []
    for row in result_rows:
        out_data.extend(row)
    out_arr = array(out_data).reshape(f_moved.shape)
    # Move axis back
    return _moveaxis(out_arr, f_moved.ndim - 1, axis)


def gradient(f, *varargs, axis=None, edge_order=1):
    f = array(f) if not isinstance(f, ndarray) else f
    if f.ndim == 0:
        raise ValueError("f must have at least 1 dimension")
    if edge_order not in (1, 2):
        raise ValueError("'edge_order' must be 1 or 2")
    axes, single_axis = _resolve_gradient_axes(axis, f.ndim)
    spacings = _resolve_gradient_spacings(varargs, axes, f.shape)
    results = _native.gradient(f, spacings, edge_order, axes)
    if single_axis or (axis is None and f.ndim == 1):
        return results[0]
    return tuple(results)


def trapz(y, x=None, dx=1.0, axis=-1):
    """Integrate along the given axis using the composite trapezoidal rule."""
    y = asarray(y)
    if x is not None:
        return _native.trapz_x(y, asarray(x), float(dx), int(axis))
    return _native.trapz(y, float(dx), int(axis))


trapezoid = trapz


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """Cumulatively integrate y(x) using the trapezoidal rule."""
    y = asarray(y)
    if x is not None:
        result = _native.cumulative_trapezoid_x(y, asarray(x), float(dx), int(axis))
    else:
        result = _native.cumulative_trapezoid(y, float(dx), int(axis))
    if initial is not None:
        # Prepend the initial value along the integration axis
        import numpy as _np
        ax = int(axis)
        if ax < 0:
            ax = y.ndim + ax
        init_shape = list(result.shape)
        init_shape[ax] = 1
        init_arr = _np.full(init_shape, float(initial))
        result = concatenate([init_arr, result], axis=ax)
    return result


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    import numpy as _np
    ar1 = _coerce_native_boxed_operand(ar1)
    ar2 = _coerce_native_boxed_operand(ar2)
    if not isinstance(ar1, (ndarray, _ObjectArray)):
        ar1 = asarray(ar1)
    if not isinstance(ar2, (ndarray, _ObjectArray)):
        ar2 = asarray(ar2)
    orig_dtype = ar1.dtype
    if not return_indices:
        if isinstance(ar1, _ObjectArray) or isinstance(ar2, _ObjectArray):
            s1 = set(ar1.flatten().tolist())
            s2 = set(ar2.flatten().tolist())
            return array(sorted(s1 & s2))
        result = _native.intersect1d(ar1, ar2)
        if str(result.dtype) != str(orig_dtype):
            try:
                result = result.astype(orig_dtype)
            except Exception:
                pass
        return result
    # Find intersection with indices
    flat1 = ar1.flatten().tolist()
    flat2 = ar2.flatten().tolist()
    s1 = set(flat1)
    s2 = set(flat2)
    common = sorted(s1 & s2)
    ind1 = [flat1.index(v) for v in common]
    ind2 = [flat2.index(v) for v in common]
    return array(common), array(ind1), array(ind2)


def union1d(ar1, ar2):
    ar1 = _coerce_native_boxed_operand(ar1)
    ar2 = _coerce_native_boxed_operand(ar2)
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    orig_dtype = ar1.dtype
    result = _native.union1d(ar1, ar2)
    if str(result.dtype) != str(orig_dtype):
        try:
            result = result.astype(orig_dtype)
        except Exception:
            pass
    return result


def setdiff1d(ar1, ar2, assume_unique=False):
    import numpy as _np
    ar1 = _coerce_native_boxed_operand(ar1)
    ar2 = _coerce_native_boxed_operand(ar2)
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    orig_dtype = ar1.dtype
    if isinstance(ar1, _ObjectArray) or isinstance(ar2, _ObjectArray):
        s1 = set(ar1.flatten().tolist())
        s2 = set(ar2.flatten().tolist())
        return array(sorted(s1 - s2))
    if assume_unique:
        # Preserve order: elements of ar1 not in ar2 (no sort, no dedup)
        ar2_flat = ar2.flatten()
        b_set = set(ar2_flat.tolist())
        result_list = [x for x in ar1.flatten().tolist() if x not in b_set]
        if not result_list:
            return _np.empty(0, dtype=orig_dtype)
        return array(result_list).astype(orig_dtype)
    result = _native.setdiff1d(ar1, ar2)
    # Preserve the original dtype (Rust always returns float64)
    if str(result.dtype) != str(orig_dtype):
        try:
            result = result.astype(orig_dtype)
        except Exception:
            pass
    return result


def setxor1d(ar1, ar2, assume_unique=False):
    """Return sorted, unique values that are in only one of the input arrays."""
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    import numpy as _np
    u1 = _np.unique(ar1)
    u2 = _np.unique(ar2)
    # Elements in ar1 but not ar2, plus elements in ar2 but not ar1
    diff1 = setdiff1d(u1, u2)
    diff2 = setdiff1d(u2, u1)
    return _np.sort(concatenate([diff1, diff2]))


def isin(element, test_elements, assume_unique=False, invert=False, kind=None):
    element = _coerce_native_boxed_operand(element)
    test_elements = _coerce_native_boxed_operand(test_elements)
    # Validate kind
    if kind is not None and kind not in ('sort', 'table'):
        raise ValueError(
            "Invalid value for `kind` argument: {!r}. "
            "Expected 'sort', 'table', or None.".format(kind)
        )
    if not isinstance(element, (ndarray, _ObjectArray)):
        element = array(element)
    if not isinstance(test_elements, (ndarray, _ObjectArray)):
        test_elements = array(test_elements)
    # Handle _ObjectArray (strings, objects, etc.) in Python
    # (must be before kind='table' check since _ObjectArray raises ValueError for non-int)
    if isinstance(element, _ObjectArray) or isinstance(test_elements, _ObjectArray):
        if kind == 'table':
            raise ValueError(
                "The 'table' method only works for integer inputs. "
                "For other dtypes, use kind='sort'."
            )
        el_flat = element.flatten().tolist() if isinstance(element, (ndarray, _ObjectArray)) else [element]
        te_flat = test_elements.flatten().tolist() if isinstance(test_elements, (ndarray, _ObjectArray)) else [test_elements]
        te_set = set(te_flat)
        if invert:
            res = [x not in te_set for x in el_flat]
        else:
            res = [x in te_set for x in el_flat]
        result = array(res, dtype='bool')
        if hasattr(element, 'shape') and element.shape != result.shape:
            result = result.reshape(element.shape)
        return result
    # kind='table' requires integer dtype
    if kind == 'table':
        el_dtype = str(element.dtype)
        te_dtype = str(test_elements.dtype)
        _integer_dtypes = {'int8','int16','int32','int64','uint8','uint16','uint32','uint64',
                           'bool','int','uint'}
        def _is_int_dtype(dt):
            return any(dt.startswith(x) for x in _integer_dtypes) or dt in _integer_dtypes
        if not (_is_int_dtype(el_dtype) and _is_int_dtype(te_dtype)):
            raise ValueError(
                "The 'table' method only works for integer inputs. "
                "For other dtypes, use kind='sort'."
            )
        # Check for integer overflow (range too large)
        te_flat = test_elements.flatten().tolist()
        if te_flat:
            try:
                _range = _builtin_max(te_flat) - _builtin_min(te_flat)
            except Exception:
                _range = 0
            _MAX_TABLE = 2**26  # 64MB limit (numpy uses similar threshold)
            if _range > _MAX_TABLE:
                raise RuntimeError(
                    "isin with kind='table': the range of values in ar2 exceed the maximum "
                    "supported by the table algorithm. Use kind='sort' instead."
                )
    # Use Python fallback for string dtypes (native isin doesn't handle them correctly)
    _el_dtype_str = str(element.dtype) if hasattr(element, 'dtype') else ''
    _te_dtype_str = str(test_elements.dtype) if hasattr(test_elements, 'dtype') else ''
    # Use Python set-based fallback for types that could lose precision in native float64 ops
    # (uint64, large int64 values) or for non-numeric types.
    _large_int_dtypes = {'uint64', 'uint32'}  # uint32 can exceed float32 precision too
    _use_python = (_el_dtype_str in ('str', 'object', 'bytes') or
                   _te_dtype_str in ('str', 'object', 'bytes') or
                   _el_dtype_str.startswith('U') or _te_dtype_str.startswith('U') or
                   _el_dtype_str.startswith('S') or _te_dtype_str.startswith('S') or
                   _el_dtype_str in _large_int_dtypes or _te_dtype_str in _large_int_dtypes)
    if not _use_python:
        try:
            result = _native.isin(element, test_elements)
            if invert:
                import numpy as _np
                return _np.logical_not(result)
            return result
        except (ValueError, TypeError, IndexError):
            pass  # Fall through to Python implementation
    # Python fallback
    el_flat = element.flatten().tolist()
    te_flat = test_elements.flatten().tolist()
    te_set = set(te_flat)
    if invert:
        res = [x not in te_set for x in el_flat]
    else:
        res = [x in te_set for x in el_flat]
    result = array(res, dtype='bool')
    if element.shape != result.shape:
        result = result.reshape(element.shape)
    return result


def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Test whether each element of ar1 is in ar2. Deprecated: use isin instead."""
    import warnings
    warnings.warn(
        "`in1d` is deprecated. Use `np.isin` instead.",
        DeprecationWarning, stacklevel=2
    )
    return isin(ar1, ar2, assume_unique=assume_unique, invert=invert)


def all(a, axis=None, out=None, keepdims=False, where=True):
    a = _ensure_reduction_array(a)
    a = _apply_where_mask(a, where, true_identity=True)
    if isinstance(a, ndarray):
        return a.all(axis=axis, keepdims=keepdims)
    if axis is None:
        return a.all()
    # Compatibility fallback for _ObjectArray-like arrays until they are deleted.
    m = a.min(axis, keepdims)
    if not isinstance(m, ndarray):
        return bool(m != 0.0)
    flat = m.flatten().tolist()
    result = [v != 0.0 for v in flat]
    return array(result).reshape(m.shape)


def any(a, axis=None, out=None, keepdims=False, where=True):
    a = _ensure_reduction_array(a)
    a = _apply_where_mask(a, where)
    if isinstance(a, ndarray):
        return a.any(axis=axis, keepdims=keepdims)
    if axis is None:
        return a.any()
    # Compatibility fallback for _ObjectArray-like arrays until they are deleted.
    m = a.max(axis, keepdims)
    if not isinstance(m, ndarray):
        return bool(m != 0.0)
    flat = m.flatten().tolist()
    result = [v != 0.0 for v in flat]
    return array(result).reshape(m.shape)


def may_share_memory(a, b, max_work=None):
    """Check if arrays may share memory (conservative)."""
    return shares_memory(a, b, max_work=max_work)


def shares_memory(a, b, max_work=None):
    """Check if arrays share memory via ArcArray buffer pointer equality."""
    if a is b:
        return True
    if isinstance(a, ndarray) and isinstance(b, ndarray) and hasattr(a, '_shares_memory_with'):
        return a._shares_memory_with(b)
    return False
