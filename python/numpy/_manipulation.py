"""Shape manipulation, stacking, splitting, reordering, selection, broadcasting."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray, _copy_into,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._core_types import dtype, _ScalarType, _normalize_dtype
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate, linspace
from ._math import isnan, isfinite

_builtin_divmod = divmod

from ._shape import *
from ._join import *
from ._sorting import *
from ._selection import *
from ._iteration import *
from ._pad import *

__all__ = [
    # Shape
    'reshape', 'ravel', 'flatten', 'expand_dims', 'squeeze', 'transpose',
    'moveaxis', 'swapaxes', 'resize',
    # At-least
    'atleast_1d', 'atleast_2d', 'atleast_3d',
    # Stacking
    'stack', 'unstack', 'hstack', 'vstack', 'dstack', 'column_stack', 'row_stack',
    # Splitting
    'split', 'array_split', 'hsplit', 'vsplit', 'dsplit',
    # Repetition / manipulation
    'repeat', 'tile', 'append', 'insert', 'delete',
    # Sorting
    'sort', 'argsort', 'lexsort', 'partition', 'argpartition', 'unique',
    'sort_complex',
    # NumPy 2.0 Array API unique functions
    'unique_values', 'unique_counts', 'unique_inverse', 'unique_all',
    # Flipping
    'flip', 'flipud', 'fliplr', 'rot90', 'roll', 'rollaxis',
    # Selection
    'extract', 'select', 'choose', 'take', 'compress', 'put', 'putmask',
    'place', 'piecewise', 'copyto',
    # Broadcasting
    'broadcast', '_BroadcastIter', 'broadcast_shapes', 'broadcast_to',
    'broadcast_arrays',
    # Utility
    'trim_zeros', 'apply_along_axis', 'apply_over_axes', 'vectorize',
    # Size / block
    'ndim', 'size', 'block',
    # Internal helper (used by tensordot)
    '_transpose_with_axes',
    # Array padding
    'pad',
    # Vandermonde matrix
    'vander',
    # Interpolation
    'interp',
    # Bin counting
    'bincount',
]


# ---------------------------------------------------------------------------
# vander
# ---------------------------------------------------------------------------

def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix."""
    x = asarray(x).flatten()
    n = x.size
    if N is None:
        N = n
    if N == 0:
        return empty((n, 0), dtype=x.dtype)
    from ._helpers import _ObjectArray
    if isinstance(x, _ObjectArray):
        # Build result as list of rows for _ObjectArray (complex etc.)
        rows = []
        for i in range(n):
            row = []
            for j in range(N):
                exp = j if increasing else (N - 1 - j)
                row.append(x[i] ** exp)
            rows.append(row)
        return array(rows)
    if increasing:
        cols = []
        for j in range(N):
            col = []
            for i in range(n):
                col.append(x[i] ** j)
            cols.append(array(col))
    else:
        cols = []
        for j in range(N):
            col = []
            for i in range(n):
                col.append(x[i] ** (N - 1 - j))
            cols.append(array(col))
    result = stack(cols, axis=1)
    return result


# ---------------------------------------------------------------------------
# interp
# ---------------------------------------------------------------------------

def interp(x, xp, fp, left=None, right=None, period=None):
    import _numpy_native as _nat
    import math as _mth

    # Normalize inputs
    xp_arr = asarray(xp, dtype='float64').flatten()
    fp_arr = asarray(fp)

    # Validation
    if xp_arr.size == 0:
        raise ValueError("xp and fp must be of at least size 1")
    if xp_arr.size != fp_arr.size:
        raise ValueError("xp and fp must have same length")
    if period is not None:
        period = float(period)
        if period == 0:
            raise ValueError("period must be a non-zero value")

    # Determine if output should be scalar
    x_in = x
    x_arr = asarray(x_in)
    x_is_scalar = (x_arr.ndim == 0) or (not isinstance(x_in, ndarray) and not isinstance(x_in, (list, tuple)))
    x_shape = x_arr.shape
    x_flat = asarray(x_arr, dtype='float64').flatten()

    # Check for complex fp
    fp_dtype = str(fp_arr.dtype)
    is_complex = 'complex' in fp_dtype

    def _complex_real(v):
        if isinstance(v, tuple): return float(v[0])
        return float(v.real) if hasattr(v, 'real') else float(v)

    def _complex_imag(v):
        if isinstance(v, tuple): return float(v[1])
        return float(v.imag) if hasattr(v, 'imag') else 0.0

    def _flat_values(arr):
        return arr.flatten().tolist()

    def _complex_parts(values):
        return (
            [_complex_real(v) for v in values],
            [_complex_imag(v) for v in values],
        )

    def _interp_periodic_fallback(x_values, xp_values, fp_values, period_value):
        pairs = sorted(
            [((float(xpj) % period_value), float(fpj)) for xpj, fpj in zip(xp_values, fp_values)],
            key=lambda pair: pair[0],
        )
        xp_sorted = [p[0] for p in pairs]
        fp_sorted = [p[1] for p in pairs]
        xp_ext = [xp_sorted[-1] - period_value] + xp_sorted + [xp_sorted[0] + period_value]
        fp_ext = [fp_sorted[-1]] + fp_sorted + [fp_sorted[0]]
        x_mod = asarray([(float(xi) % period_value) for xi in x_values], dtype='float64')
        return _nat.interp(x_mod, asarray(xp_ext, dtype='float64'), asarray(fp_ext, dtype='float64'))

    def _rebuild_complex_result(values, shape=None):
        flat_values = list(values)
        result_arr = zeros((len(flat_values),), dtype='complex128')
        result_arr.real = asarray([_complex_real(v) for v in flat_values], dtype='float64')
        result_arr.imag = asarray([_complex_imag(v) for v in flat_values], dtype='float64')
        if shape is not None:
            result_arr = result_arr.reshape(shape)
        return result_arr

    def _interp_nonfinite_xp_lane(values, fill_count):
        lane = [float(v) for v in values]
        head = lane[0]
        if not _mth.isnan(head) and all(v == head for v in lane[1:]):
            return [head] * fill_count
        return [float('nan')] * fill_count

    def _interp_infinite_xp_lane(values, xp_values, fill_count):
        lane = [float(v) for v in values]
        head = lane[0]
        if all(v == head for v in lane[1:]) and not _mth.isnan(head):
            return [head] * fill_count
        if any(not _mth.isfinite(v) for v in lane):
            return [float('nan')] * fill_count
        if all(v == float('inf') for v in xp_values):
            return [lane[0]] * fill_count
        if all(v == float('-inf') for v in xp_values):
            return [lane[-1]] * fill_count
        if xp_values and xp_values[0] == float('-inf') and all(_mth.isfinite(v) for v in xp_values[1:]):
            return [lane[-1]] * fill_count
        if xp_values and xp_values[-1] == float('inf') and all(_mth.isfinite(v) for v in xp_values[:-1]):
            return [lane[0]] * fill_count
        return [float('nan')] * fill_count

    def _interp_two_point_nonfinite_fp_lane(values, fill_count):
        lane = [float(v) for v in values]
        if any(_mth.isnan(v) for v in lane):
            return [float('nan')] * fill_count
        inf_signs = {1 if v > 0 else -1 for v in lane if _mth.isinf(v)}
        if len(inf_signs) > 1:
            return [float('nan')] * fill_count
        if len(inf_signs) == 1:
            sign = next(iter(inf_signs))
            return [float('inf') if sign > 0 else float('-inf')] * fill_count
        return None

    xp_values = xp_arr.tolist()
    has_nan_xp = any(_mth.isnan(v) for v in xp_values)
    has_inf_xp = any(_mth.isinf(v) for v in xp_values)

    if has_nan_xp or has_inf_xp:
        if is_complex:
            re_vals, im_vals = _complex_parts(_flat_values(fp_arr))
            if has_nan_xp:
                re_result = _interp_nonfinite_xp_lane(re_vals, x_flat.size)
                im_result = _interp_nonfinite_xp_lane(im_vals, x_flat.size)
            else:
                re_result = _interp_infinite_xp_lane(re_vals, xp_values, x_flat.size)
                im_result = _interp_infinite_xp_lane(im_vals, xp_values, x_flat.size)
            result = _rebuild_complex_result(
                [
                    complex(re_val, im_val)
                    for re_val, im_val in zip(re_result, im_result)
                ],
                () if x_is_scalar else x_shape,
            )
            return result

        if has_nan_xp:
            result_vals = _interp_nonfinite_xp_lane(_flat_values(fp_arr), x_flat.size)
        else:
            result_vals = _interp_infinite_xp_lane(_flat_values(fp_arr), xp_values, x_flat.size)
        result = array(result_vals)
        if x_is_scalar:
            return float(result.flatten()[0])
        if x_shape != result.shape:
            try:
                result = result.reshape(x_shape)
            except Exception:
                pass
        return result

    if xp_arr.size == 2 and all(_mth.isfinite(v) for v in xp_values):
        if is_complex:
            re_vals, im_vals = _complex_parts(_flat_values(fp_arr))
            re_result = _interp_two_point_nonfinite_fp_lane(re_vals, x_flat.size)
            im_result = _interp_two_point_nonfinite_fp_lane(im_vals, x_flat.size)
            if re_result is not None or im_result is not None:
                if re_result is None:
                    re_result = _nat.interp(
                        x_flat, xp_arr, asarray(re_vals, dtype='float64')
                    ).flatten().tolist()
                if im_result is None:
                    im_result = _nat.interp(
                        x_flat, xp_arr, asarray(im_vals, dtype='float64')
                    ).flatten().tolist()
                result = _rebuild_complex_result(
                    [complex(re_val, im_val) for re_val, im_val in zip(re_result, im_result)],
                    () if x_is_scalar else x_shape,
                )
                return result
        else:
            result_vals = _interp_two_point_nonfinite_fp_lane(_flat_values(fp_arr), x_flat.size)
            if result_vals is not None:
                result = array(result_vals)
                if x_is_scalar:
                    return float(result.flatten()[0])
                if x_shape != result.shape:
                    try:
                        result = result.reshape(x_shape)
                    except Exception:
                        pass
                return result

    if period is not None:
        interp_periodic = getattr(_nat, 'interp_periodic', None)
        if is_complex:
            fp_re, fp_im = _complex_parts(_flat_values(fp_arr))
            xp_list = xp_arr.tolist()
            x_list = x_flat.tolist()
            if interp_periodic is not None:
                result_re = interp_periodic(x_flat, xp_arr, asarray(fp_re, dtype='float64'), period)
                result_im = interp_periodic(x_flat, xp_arr, asarray(fp_im, dtype='float64'), period)
            else:
                result_re = _interp_periodic_fallback(x_list, xp_list, fp_re, period)
                result_im = _interp_periodic_fallback(x_list, xp_list, fp_im, period)
            result = zeros(result_re.shape, dtype='complex128')
            result.real = result_re
            result.imag = result_im
        else:
            if interp_periodic is not None:
                result = interp_periodic(x_flat, xp_arr, fp_arr, period)
            else:
                result = _interp_periodic_fallback(x_flat.tolist(), xp_arr.tolist(), _flat_values(fp_arr), period)
    elif is_complex:
        # Interpolate real and imag parts separately
        fp_flat = _flat_values(fp_arr)
        fp_re, fp_im = _complex_parts(fp_flat)
        fp_re_arr = asarray(fp_re)
        fp_im_arr = asarray(fp_im)
        result_re = _nat.interp(x_flat, xp_arr, fp_re_arr)
        result_im = _nat.interp(x_flat, xp_arr, fp_im_arr)
        # Avoid inf*0=nan by setting .real/.imag directly instead of + 1j*
        result = zeros(result_re.shape, dtype='complex128')
        result.real = result_re
        result.imag = result_im
        result_list = result.flatten().tolist()
        fp_list = [complex(re, im) for re, im in zip(fp_re, fp_im)]
        for i, xi in enumerate(x_flat.tolist()):
            for j, xpj in enumerate(xp_arr.tolist()):
                if xi == xpj:
                    result_list[i] = fp_list[j]
                    break
        result = _rebuild_complex_result(result_list, result.shape)

        # Apply left/right for complex
        if left is not None or right is not None:
            x_list = x_flat.tolist()
            xp_list = xp_arr.tolist()
            xp_min = _builtin_min(xp_list)
            xp_max = _builtin_max(xp_list)
            result_list = result.flatten().tolist()
            for i, xi in enumerate(x_list):
                if left is not None and xi < xp_min:
                    v = left
                    result_list[i] = complex(v.real if hasattr(v, 'real') else float(v),
                                             v.imag if hasattr(v, 'imag') else 0.0)
                if right is not None and xi > xp_max:
                    v = right
                    result_list[i] = complex(v.real if hasattr(v, 'real') else float(v),
                                             v.imag if hasattr(v, 'imag') else 0.0)
            result = _rebuild_complex_result(result_list, result.shape)
    else:
        if left is not None or right is not None:
            result = _nat.interp_with_options(
                x_flat,
                xp_arr,
                fp_arr,
                None if left is None else float(left),
                None if right is None else float(right),
            )
        else:
            result = _nat.interp(x_flat, xp_arr, fp_arr)

    # Restore original shape
    if x_is_scalar:
        if is_complex:
            return result.reshape(())
        # Return scalar float64
        v = result.flatten()[0] if result.size > 0 else 0.0
        return float(v)
    if x_shape != result.shape:
        try:
            result = result.reshape(x_shape)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# bincount
# ---------------------------------------------------------------------------

def bincount(x, weights=None, minlength=0):
    import _numpy_native as _nat
    if not isinstance(x, ndarray):
        x = array(x)
    if x.ndim != 1:
        raise ValueError("object too deep for desired array")
    if not isinstance(minlength, int):
        try:
            minlength = int(minlength)
        except (TypeError, ValueError):
            raise TypeError("'{}' object cannot be interpreted as an integer".format(
                type(minlength).__name__))
    if minlength < 0:
        raise ValueError("minlength must not be negative")
    if weights is not None and not isinstance(weights, ndarray):
        weights = array(weights)
    return _nat.bincount(x, weights, minlength)
