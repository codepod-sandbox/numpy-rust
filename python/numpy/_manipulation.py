"""Shape manipulation, stacking, splitting, reordering, selection, broadcasting."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray, _copy_into, _coerce_native_boxed_operand,
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
    if isinstance(x, _ObjectArray):
        native_x = _coerce_native_boxed_operand(x)
        if isinstance(native_x, ndarray):
            x = native_x.flatten()
    n = x.size
    if N is None:
        N = n
    if N == 0:
        return empty((n, 0), dtype=x.dtype)
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

    if period is not None:
        # Sort xp/fp by xp modulo period, then interpolate
        x_flat_list = x_flat.tolist()
        xp_list = xp_arr.tolist()
        if is_complex:
            fp_re = [_complex_real(v) for v in fp_arr.flatten().tolist()]
            fp_im = [_complex_imag(v) for v in fp_arr.flatten().tolist()]
        else:
            fp_list = [float(v) for v in fp_arr.flatten().tolist()]

        # Normalize x to [0, period), then sort and add wrap-around points
        # Build sorted (xp_norm, fp) pairs
        pairs = list(zip(xp_list, fp_re if is_complex else fp_list))
        if is_complex:
            pairs_im = list(zip(xp_list, fp_im))
        # Normalize xp to [0, period)
        pairs = [((xpi % period), fpi) for xpi, fpi in pairs]
        pairs.sort(key=lambda kv: kv[0])
        if is_complex:
            pairs_im = [((xpi % period), fpi) for xpi, fpi in pairs_im]
            pairs_im.sort(key=lambda kv: kv[0])

        # Add wrap-around: append first point at period and last point at -period
        pairs = [(pairs[-1][0] - period, pairs[-1][1])] + pairs + [(pairs[0][0] + period, pairs[0][1])]
        if is_complex:
            pairs_im = [(pairs_im[-1][0] - period, pairs_im[-1][1])] + pairs_im + [(pairs_im[0][0] + period, pairs_im[0][1])]

        xp_sorted = array([p[0] for p in pairs])
        fp_sorted = array([p[1] for p in pairs])
        x_norm = array([(xi % period) for xi in x_flat_list])

        result_re = _nat.interp(x_norm, xp_sorted, fp_sorted)
        if is_complex:
            xp_sorted_im = array([p[0] for p in pairs_im])
            fp_sorted_im = array([p[1] for p in pairs_im])
            result_im = _nat.interp(x_norm, xp_sorted_im, fp_sorted_im)
            result = zeros(result_re.shape, dtype='complex128')
            result.real = result_re
            result.imag = result_im
        else:
            result = result_re
    elif is_complex:
        # Interpolate real and imag parts separately
        fp_re_arr = asarray([_complex_real(v) for v in fp_arr.flatten().tolist()])
        fp_im_arr = asarray([_complex_imag(v) for v in fp_arr.flatten().tolist()])
        result_re = _nat.interp(x_flat, xp_arr, fp_re_arr)
        result_im = _nat.interp(x_flat, xp_arr, fp_im_arr)
        # Avoid inf*0=nan by setting .real/.imag directly instead of + 1j*
        result = zeros(result_re.shape, dtype='complex128')
        result.real = result_re
        result.imag = result_im

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
            result = array(result_list)
    else:
        result = _nat.interp(x_flat, xp_arr, fp_arr)
        if left is not None or right is not None:
            x_list = x_flat.tolist()
            xp_list = xp_arr.tolist()
            result_list = result.flatten().tolist()
            xp_min = _builtin_min(xp_list)
            xp_max = _builtin_max(xp_list)
            for i, xi in enumerate(x_list):
                if left is not None and xi < xp_min:
                    result_list[i] = float(left)
                if right is not None and xi > xp_max:
                    result_list[i] = float(right)
            result = array(result_list)

    # Restore original shape
    if x_is_scalar:
        # Return scalar float64 (or complex if complex fp)
        v = result.flatten()[0] if result.size > 0 else 0.0
        if is_complex:
            # v might be a tuple (re, im) from native complex array
            if isinstance(v, tuple):
                return complex(v[0], v[1])
            return complex(v)
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
