"""Aggregation, statistics, NaN-aware reductions, set operations."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray, _copy_into,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._core_types import dtype, _ScalarType, _normalize_dtype
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate, linspace, full

def _scalar_result(result, input_arr, force_dtype=None):
    """Wrap a scalar result so it preserves dtype as a 0-d array."""
    if isinstance(result, ndarray):
        return result
    dt = force_dtype
    if dt is None and isinstance(input_arr, ndarray):
        dt = str(input_arr.dtype)
    if dt is not None:
        try:
            return array(result).astype(dt).reshape(())
        except Exception:
            pass
    return result


def _mean_result_dtype(input_arr):
    """Get the result dtype for mean/var/std operations (int → float64)."""
    if isinstance(input_arr, ndarray):
        dt = str(input_arr.dtype)
        if dt.startswith(('int', 'uint', 'bool')):
            return 'float64'
        return dt
    return 'float64'


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


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        # For plain lists/tuples, convert to array; for other iterables use builtin sum
        if isinstance(a, (list, tuple)):
            a = asarray(a)
        else:
            return _builtin_sum(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        a = a * w
    if axis is not None:
        result = a.sum(axis, keepdims)
    else:
        result = a.sum(None, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    if initial is not None:
        result = result + initial
    return result


def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        a = a * w + (1.0 - w)
    result = a.prod(axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    if initial is not None:
        result = result * initial
    return result


def cumsum(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        result = a.cumsum(axis)
    else:
        result = array(a).cumsum(axis)
    return _dtype_cast(result, dtype)


def cumprod(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        result = a.cumprod(axis)
    else:
        result = array(a).cumprod(axis)
    return _dtype_cast(result, dtype)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    if prepend is not None:
        prepend = asarray(prepend)
        if prepend.ndim == 0:
            prepend = array([float(prepend)])
        a = concatenate([prepend, a])
    if append is not None:
        append = asarray(append)
        if append.ndim == 0:
            append = array([float(append)])
        a = concatenate([a, append])
    return _native.diff(a, n, axis)


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
        m = a.sum(None, True) / a.size
        result = _sq_dev(a - m).sum() / (a.size - ddof)
        if isinstance(result, ndarray) and result.ndim == 0:
            result = float(result)
    else:
        if isinstance(ddof, float) and ddof != int(ddof):
            # float ddof not supported by native var — compute in Python
            m = a.sum(None, True) / a.size
            result = _sq_dev(a - m).sum() / (a.size - ddof)
            if isinstance(result, ndarray) and result.ndim == 0:
                result = float(result)
        else:
            result = a.var(None, int(ddof) if isinstance(ddof, float) else ddof, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
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
        w = asarray(where).astype("bool").astype("float64")
        a = a * w
    result = _native.nansum(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    elif isinstance(result, ndarray) and _in_dt.startswith('float') and str(result.dtype) != _in_dt:
        # Only preserve dtype for float types (not int — int sums upcast to int64)
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if initial is not None:
        result = result + initial
    return result


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        s = (a * w).sum(axis, keepdims)
        c = w.sum(axis, keepdims)
        result = s / c
    else:
        try:
            result = _native.nanmean(a, axis, keepdims)
        except (ValueError, ZeroDivisionError):
            import warnings
            warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
            result = _nan_result_like(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    if out is not None:
        if isinstance(out, ndarray):
            _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
            return out
    return result


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        import numpy as _np
        result = _np.sqrt(nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims, where=where))
    else:
        try:
            result = _native.nanstd(a, axis, int(ddof) if isinstance(ddof, float) and ddof == int(ddof) else ddof, keepdims)
        except TypeError:
            # ddof is a float — use nanvar + sqrt
            import numpy as _np
            result = _np.sqrt(nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims))
        except (ValueError, ZeroDivisionError):
            import warnings
            warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
            result = _nan_result_like(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    return result


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        c = w.sum(axis, True)
        m = (a * w).sum(axis, True) / c
        if ddof == 0:
            result = ((a - m) ** 2 * w).sum(axis, keepdims) / c
        else:
            result = ((a - m) ** 2 * w).sum(axis, keepdims) / (c - ddof)
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
            result = (_diff_clean ** 2).sum(axis, keepdims) / (_count - ddof)
            if not keepdims and isinstance(result, ndarray) and result.ndim > 0:
                result = result.squeeze()
        except (ValueError, ZeroDivisionError):
            import warnings
            warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
            result = _nan_result_like(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    return result


def nanmin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
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
    return result


def nanmax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
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
    return result


def nanargmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    if a.size == 0:
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
    return result


def nanargmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    if a.size == 0:
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
    return result


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    _in_dt = str(a.dtype)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        a = a * w + (1.0 - w)
    result = _native.nanprod(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    elif not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    elif isinstance(result, ndarray) and _in_dt.startswith('float') and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    if initial is not None:
        result = result * initial
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
    elif isinstance(result, ndarray) and _in_dt.startswith('float') and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
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
    elif isinstance(result, ndarray) and _in_dt.startswith('float') and str(result.dtype) != _in_dt:
        try:
            result = result.astype(_in_dt)
        except Exception:
            pass
    return result


def _apply_keepdims(result, a, axis):
    """Expand reduced axis back to size-1 when keepdims=True."""
    if isinstance(result, ndarray):
        shape = list(a.shape)
        shape[axis] = 1
        return result.reshape(shape)
    else:
        # scalar result - wrap in array with expanded dims
        shape = [1] * a.ndim
        return array([float(result)]).reshape(shape)


def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_native.quantile(a, float(qi), axis) for qi in q_list]
        import numpy as _np
        return array(results) if axis is None else _np.stack(results)
    result = _native.quantile(a, float(q), axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result


def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_native.percentile(a, float(qi), axis) for qi in q_list]
        import numpy as _np
        return array(results) if axis is None else _np.stack(results)
    result = _native.percentile(a, float(q), axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.quantile(a, 0.5, axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    return result


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    if not isinstance(m, ndarray):
        m = array(m)
    _ddof = ddof if ddof is not None else (0 if bias else 1)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.cov(m, y, rowvar, _ddof)
    return _native.cov(m, None, rowvar, _ddof)


def corrcoef(x, y=None, rowvar=True):
    if not isinstance(x, ndarray):
        x = array(x)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.corrcoef(x, y, rowvar)
    return _native.corrcoef(x, None, rowvar)


def average(a, axis=None, weights=None, returned=False, keepdims=False):
    """Compute the weighted average along the specified axis."""
    a = asarray(a)
    if weights is None:
        avg = mean(a, axis=axis)
        if returned:
            if axis is None:
                return avg, float(a.size)
            return avg, full(avg.shape, float(a.shape[axis]))
        return avg
    weights = asarray(weights)
    wsum = sum(a * weights, axis=axis)
    wt = sum(weights, axis=axis)
    avg = wsum / wt
    if returned:
        return avg, wt
    return avg


def _nan_quantile_1d(vals_flat, q):
    """Compute quantile of a 1D array, ignoring NaNs. Returns scalar."""
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
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _nan_quantile_impl(a, q, axis):
    """Helper for nanmedian/nanpercentile/nanquantile with axis support."""
    a = asarray(a)
    if axis is None:
        return _nan_quantile_1d(a.flatten(), q)
    # Normalize negative axis
    if axis < 0:
        axis = a.ndim + axis
    if a.ndim == 1:
        return _nan_quantile_1d(a.flatten(), q)
    # General n-d: move target axis to last, iterate over remaining dims
    import numpy as _np
    a_moved = _np.moveaxis(a, axis, -1)
    # Reshape to (product_of_other_dims, axis_len)
    other_shape = a_moved.shape[:-1]
    axis_len = a_moved.shape[-1]
    flat_2d = a_moved.reshape((-1, axis_len)) if a_moved.ndim > 1 else a_moved.reshape((1, axis_len))
    n_slices = flat_2d.shape[0]
    results = []
    for i in range(n_slices):
        row = flat_2d[i]
        results.append(_nan_quantile_1d(row if isinstance(row, ndarray) else asarray(row), q))
    result = array(results)
    if len(other_shape) > 1:
        result = result.reshape(other_shape)
    return result


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Compute the median along the specified axis, ignoring NaNs."""
    a = asarray(a)
    result = _nan_quantile_impl(a, 0.5, axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis) if isinstance(result, ndarray) else _apply_keepdims(array([float(result)]), a, axis)
    if out is not None:
        if isinstance(out, ndarray):
            _copy_into(out, result if isinstance(result, ndarray) else asarray(result))
            return out
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    return result


def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    """Compute the qth percentile, ignoring NaNs."""
    a = asarray(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_nan_quantile_impl(a, float(qi) / 100.0, axis) for qi in q_list]
        import numpy as _np
        return array(results) if axis is None else _np.stack(results)
    result = _nan_quantile_impl(a, float(q) / 100.0, axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis) if isinstance(result, ndarray) else _apply_keepdims(array([float(result)]), a, axis)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    return result


def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None):
    """Compute the qth quantile, ignoring NaNs."""
    a = asarray(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_nan_quantile_impl(a, float(qi), axis) for qi in q_list]
        import numpy as _np
        return array(results) if axis is None else _np.stack(results)
    result = _nan_quantile_impl(a, float(q), axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis) if isinstance(result, ndarray) else _apply_keepdims(array([float(result)]), a, axis)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a, _mean_result_dtype(a))
    return result


def ediff1d(ary, to_end=None, to_begin=None):
    """The differences between consecutive elements of an array."""
    ary = asarray(ary).flatten()
    dtype_str = str(ary.dtype)
    n = ary.size
    parts = []
    if to_begin is not None:
        tb = asarray(to_begin).flatten()
        parts.append(tb)
    if n > 1:
        diff_vals = [ary[i] - ary[i - 1] for i in range(1, n)]
        parts.append(array(diff_vals))
    elif n == 1:
        # 0-length diff
        parts.append(array([], dtype=dtype_str))
    else:
        parts.append(array([], dtype=dtype_str))
    if to_end is not None:
        te = asarray(to_end).flatten()
        parts.append(te)
    import numpy as _np
    result = _np.concatenate(parts) if parts else array([], dtype=dtype_str)
    return result.astype(dtype_str)


def max(a, axis=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        fill_val = float('-inf')
        flat_a = a.flatten().tolist()
        flat_w = w.flatten().tolist()
        masked = [v if m else fill_val for v, m in zip(flat_a, flat_w)]
        a = array(masked).reshape(a.shape)
    if axis is not None:
        result = a.max(axis, keepdims)
    else:
        result = a.max(None, keepdims)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a)
    return result


amax = max


def min(a, axis=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        fill_val = float('inf')
        flat_a = a.flatten().tolist()
        flat_w = w.flatten().tolist()
        masked = [v if m else fill_val for v, m in zip(flat_a, flat_w)]
        a = array(masked).reshape(a.shape)
    if axis is not None:
        result = a.min(axis, keepdims)
    else:
        result = a.min(None, keepdims)
    if not isinstance(result, ndarray):
        result = _scalar_result(result, a)
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


def ptp(a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return a.max(axis) - a.min(axis)


def argwhere(a):
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Rust argwhere only handles float64; cast integer/bool arrays
    dt = str(a.dtype)
    if dt in ("int32", "int64", "int8", "int16", "uint8", "uint16", "uint32", "uint64", "bool"):
        a = a.astype("float64")
    return _native.argwhere(a)


def nonzero(a):
    if isinstance(a, _ObjectArray):
        indices = []
        for i, v in enumerate(a._data):
            if bool(v):
                indices.append(i)
        return (array(indices, dtype="int64"),) if indices else (array([], dtype="int64"),)
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.ndim == 0:
        raise ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(a).nonzero() instead.")
    # Rust nonzero only handles float64; cast integer/bool arrays
    dt = str(a.dtype)
    if dt in ("int32", "int64", "int8", "int16", "uint8", "uint16", "uint32", "uint64", "bool"):
        a = a.astype("float64")
    return _native.nonzero(a)


def count_nonzero(a, axis=None, *, keepdims=False):
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
    # Rust count_nonzero only handles float64; cast integer/bool arrays
    dt = str(a.dtype)
    if dt in ("int32", "int64", "int8", "int16", "uint8", "uint16", "uint32", "uint64", "bool"):
        a = a.astype("float64")
    if axis is None:
        if not isinstance(a, ndarray):
            # _ObjectArray (strings, bytes, etc.) — iterate Python-side
            flat = a.flatten().tolist() if hasattr(a, 'flatten') else list(a)
            result = sum(1 for v in flat if _is_nonzero(v))
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


def gradient(f, *varargs, axis=None, edge_order=1):
    if not isinstance(f, ndarray):
        f = array(f)
    if f.ndim == 1:
        # 1D case: single spacing
        spacing = float(varargs[0]) if varargs else 1.0
        return _native.gradient(f, spacing)
    # nD case
    if axis is not None:
        # gradient along specific axis/axes
        if isinstance(axis, int):
            ax = axis
            if ax < 0:
                ax = f.ndim + ax
            sp = float(varargs[0]) if varargs else 1.0
            result = _native.gradient(f, sp)
            if isinstance(result, (list, tuple)):
                return result[ax]
            return result
        # multiple axes
        results = []
        for i, ax in enumerate(axis):
            sp = float(varargs[i]) if i < len(varargs) else 1.0
            grads = _native.gradient(f, sp)
            if isinstance(grads, (list, tuple)):
                a = ax
                if a < 0:
                    a = f.ndim + a
                results.append(grads[a])
            else:
                results.append(grads)
        return results
    # All axes
    if len(varargs) == 0:
        return _native.gradient(f, 1.0)
    elif len(varargs) == 1:
        return _native.gradient(f, float(varargs[0]))
    else:
        # Different spacing per axis
        results = []
        for i in range(f.ndim):
            sp = float(varargs[i]) if i < len(varargs) else 1.0
            grads = _native.gradient(f, sp)
            if isinstance(grads, (list, tuple)):
                results.append(grads[i])
            else:
                results.append(grads)
        return results


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
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    if not return_indices:
        if isinstance(ar1, _ObjectArray) or isinstance(ar2, _ObjectArray):
            s1 = set(ar1.flatten().tolist())
            s2 = set(ar2.flatten().tolist())
            return array(sorted(s1 & s2))
        return _native.intersect1d(ar1, ar2)
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
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.union1d(ar1, ar2)


def setdiff1d(ar1, ar2, assume_unique=False):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    if isinstance(ar1, _ObjectArray) or isinstance(ar2, _ObjectArray):
        s1 = set(ar1.flatten().tolist())
        s2 = set(ar2.flatten().tolist())
        return array(sorted(s1 - s2))
    return _native.setdiff1d(ar1, ar2)


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
    if not isinstance(element, ndarray):
        element = array(element)
    if not isinstance(test_elements, ndarray):
        test_elements = array(test_elements)
    # Handle _ObjectArray (strings, objects, etc.) in Python
    if isinstance(element, _ObjectArray) or isinstance(test_elements, _ObjectArray):
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
    try:
        result = _native.isin(element, test_elements)
    except (ValueError, TypeError, IndexError):
        # Fallback to Python implementation for cases Rust can't handle
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
    if invert:
        import numpy as _np
        return _np.logical_not(result)
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
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        # Masked elements become True (identity for AND)
        mask_f = w.astype("float64")
        a = a * mask_f + (1.0 - mask_f)
    if axis is None:
        return a.all()
    # Reduce along specific axis: all elements nonzero iff min != 0
    m = a.min(axis, keepdims)
    if not isinstance(m, ndarray):
        return bool(m != 0.0)
    flat = m.flatten().tolist()
    result = [v != 0.0 for v in flat]
    return array(result).reshape(m.shape)


def any(a, axis=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        # Masked elements become False (identity for OR / 0.0)
        mask_f = w.astype("float64")
        a = a * mask_f
    if axis is None:
        return a.any()
    # Reduce along specific axis: any element nonzero iff max != 0
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
