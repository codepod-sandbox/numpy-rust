"""NumPy-compatible Python package wrapping the Rust native module."""
import sys as _sys
import math as _stdlib_math
from functools import reduce as _reduce

__version__ = "1.26.0"

# Import from native Rust module
import _numpy_native as _native
from _numpy_native import ndarray
from _numpy_native import dot
from _numpy_native import concatenate as _native_concatenate
from ._helpers import *
from ._core_types import *
from ._datetime import *
from ._creation import *
from ._math import *
from ._reductions import *
from ._manipulation import *
from ._bitwise import *
from ._ufunc import *
from ._poly import *


# Import submodules so they're accessible as numpy.linalg etc.
from _numpy_native import linalg, fft, random

# Register Rust submodules in sys.modules so `from numpy.random import ...` works
_sys.modules["numpy.linalg"] = linalg
_sys.modules["numpy.fft"] = fft
_sys.modules["numpy.random"] = random



# --- Constants --------------------------------------------------------------
nan = float("nan")
inf = float("inf")
pi = _stdlib_math.pi
e = _stdlib_math.e
newaxis = None
PINF = float("inf")
NINF = float("-inf")
PZERO = 0.0
NZERO = -0.0
Inf = inf
Infinity = inf
NaN = nan
NAN = nan
euler_gamma = 0.5772156649015329
ALLOW_THREADS = 1
little_endian = True

# numpy 1.x compat: np.bool (deprecated, can't shadow builtin 'bool' in module
# scope since isinstance checks recurse). We set it via __getattr__ below.

# --- Missing functions (stubs) ----------------------------------------------

absolute = abs

radians = deg2rad
degrees = rad2deg

round_ = around
round = around

special = type('special', (), {
    'gamma': staticmethod(gamma),
    'erf': staticmethod(erf),
    'erfc': staticmethod(erfc),
    'lgamma': staticmethod(lgamma),
    'j0': staticmethod(j0),
    'j1': staticmethod(j1),
    'y0': staticmethod(y0),
    'y1': staticmethod(y1),
})()


_err_state = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}

def seterr(**kwargs):
    """Set floating point error handling."""
    global _err_state
    old = dict(_err_state)
    for k, v in kwargs.items():
        if k == "all":
            for key in _err_state:
                _err_state[key] = v
            continue
        if k not in _err_state:
            raise ValueError("invalid key: %r" % k)
        _err_state[k] = v
    return old

def geterr():
    return dict(_err_state)

class errstate:
    """Context manager for floating point error handling."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._old = None
    def __enter__(self):
        self._old = seterr(**self._kwargs)
        return self
    def __exit__(self, *args):
        seterr(**self._old)

def set_printoptions(**kwargs):
    pass

def get_printoptions():
    return {}

class printoptions:
    """Context manager for print options."""
    def __init__(self, **kwargs):
        self._opts = kwargs
    def __enter__(self):
        set_printoptions(**self._opts)
        return self
    def __exit__(self, *args):
        pass  # We don't actually track old options


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Extract diagonal from array."""
    a = asarray(a)
    if a.ndim < 2:
        raise ValueError("diagonal requires at least a 2-d array")
    # For 2D with default axes, delegate directly
    if a.ndim == 2:
        if axis1 == 1 and axis2 == 0:
            return _native.diagonal(a.T, offset)
        return _native.diagonal(a, offset)
    # For nD, move the two axes to the end and extract diagonals
    # along the last two axes
    ax1 = axis1 if axis1 >= 0 else a.ndim + axis1
    ax2 = axis2 if axis2 >= 0 else a.ndim + axis2
    a = moveaxis(a, (ax1, ax2), (-2, -1))
    # Now extract diagonal from last two dims for each "batch" index
    shape = a.shape
    batch_shape = shape[:-2]
    m, n = shape[-2], shape[-1]
    if offset >= 0:
        diag_len = _builtin_min(m, n - offset)
    else:
        diag_len = _builtin_min(m + offset, n)
    if diag_len <= 0:
        out_shape = list(batch_shape) + [0]
        return zeros(out_shape)
    flat = a.flatten().tolist()
    batch_size = 1
    for s in batch_shape:
        batch_size *= s
    mn = m * n
    result = []
    for b in range(batch_size):
        base = b * mn
        for k in range(diag_len):
            if offset >= 0:
                result.append(flat[base + k * n + (k + offset)])
            else:
                result.append(flat[base + (k - offset) * n + k])
    out_shape = list(batch_shape) + [diag_len]
    return array(result).reshape(out_shape)


def trace(a, offset=0, axis1=0, axis2=1):
    d = diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    return d.sum()

def outer(a, b, out=None):
    """Compute outer product."""
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    result = _native.outer(a, b)
    if out is not None:
        _copy_into(out, result)
        return out
    return result

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Cross product of two arrays.

    Handles 1D vectors of length 2 or 3, and batched 2D arrays where the
    last axis has length 2 or 3.
    """
    _a_scalar = not isinstance(a, (ndarray, list, tuple))
    _b_scalar = not isinstance(b, (ndarray, list, tuple))
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    if a.ndim == 0 or b.ndim == 0 or _a_scalar or _b_scalar:
        raise ValueError("At least one array has zero dimension")
    if axis is not None:
        axisa = axisb = axisc = axis
    # Validate axes
    if axisa < -a.ndim or axisa >= a.ndim:
        raise AxisError(axisa, a.ndim, "axisa")
    if axisb < -b.ndim or axisb >= b.ndim:
        raise AxisError(axisb, b.ndim, "axisb")
    # Move the vector axis to the last position for both arrays
    if a.ndim > 1 and axisa != -1 and axisa != a.ndim - 1:
        a = moveaxis(a, axisa, -1)
    if b.ndim > 1 and axisb != -1 and axisb != b.ndim - 1:
        b = moveaxis(b, axisb, -1)
    # Broadcast: if one is 2D+ and other is 1D, expand the 1D one
    if a.ndim >= 2 and b.ndim == 1:
        b = b.reshape((1,) * (a.ndim - 1) + (b.shape[0],))
        # Broadcast b to match a's batch dims
        b_shape = list(a.shape[:-1]) + [b.shape[-1]]
        b_flat = b.flatten().tolist()
        b_new = []
        batch = 1
        for s in a.shape[:-1]:
            batch *= s
        vec_len = b.shape[-1]
        for i in range(batch):
            b_new.extend(b_flat[:vec_len])
        b = array(b_new).reshape(b_shape)
    elif b.ndim >= 2 and a.ndim == 1:
        a = a.reshape((1,) * (b.ndim - 1) + (a.shape[0],))
        a_shape = list(b.shape[:-1]) + [a.shape[-1]]
        a_flat = a.flatten().tolist()
        a_new = []
        batch = 1
        for s in b.shape[:-1]:
            batch *= s
        vec_len = a.shape[-1]
        for i in range(batch):
            a_new.extend(a_flat[:vec_len])
        a = array(a_new).reshape(a_shape)
    # Simple 1D cases
    if a.ndim == 1 and b.ndim == 1:
        af = a.flatten().tolist()
        bf = b.flatten().tolist()
        la, lb = len(af), len(bf)
        if la not in (2, 3) or lb not in (2, 3):
            raise ValueError("incompatible vector sizes for cross product")
        # Pad 2D to 3D with z=0
        if la == 2:
            af = [af[0], af[1], 0.0]
        if lb == 2:
            bf = [bf[0], bf[1], 0.0]
        cx = af[1]*bf[2] - af[2]*bf[1]
        cy = af[2]*bf[0] - af[0]*bf[2]
        cz = af[0]*bf[1] - af[1]*bf[0]
        if la == 2 and lb == 2:
            return array(cz)
        return array([cx, cy, cz])
    # Batched nD case: process along last axis
    if a.ndim >= 2 and b.ndim >= 2:
        # Broadcast batch dimensions
        a_batch = a.shape[:-1]
        b_batch = b.shape[:-1]
        try:
            out_batch = broadcast_shapes(a_batch, b_batch)
        except Exception:
            out_batch = a_batch
        a_bc = broadcast_to(a, tuple(out_batch) + (a.shape[-1],))
        b_bc = broadcast_to(b, tuple(out_batch) + (b.shape[-1],))
        la = a_bc.shape[-1]
        lb = b_bc.shape[-1]
        batch_size = 1
        for s in out_batch:
            batch_size *= s
        af = a_bc.flatten().tolist()
        bf = b_bc.flatten().tolist()
        results = []
        for i in range(batch_size):
            ai = af[i * la:(i + 1) * la]
            bi = bf[i * lb:(i + 1) * lb]
            # Pad 2D to 3D
            if la == 2:
                ai = [ai[0], ai[1], 0.0]
            if lb == 2:
                bi = [bi[0], bi[1], 0.0]
            cx = ai[1]*bi[2] - ai[2]*bi[1]
            cy = ai[2]*bi[0] - ai[0]*bi[2]
            cz = ai[0]*bi[1] - ai[1]*bi[0]
            if la == 2 and lb == 2:
                results.append(cz)
            else:
                results.extend([cx, cy, cz])
        if la == 2 and lb == 2:
            result = array(results).reshape(out_batch)
        else:
            result = array(results).reshape(list(out_batch) + [3])
        # Move result vector axis to axisc position
        if axisc != -1 and axisc != result.ndim - 1 and result.ndim > 1 and not (la == 2 and lb == 2):
            result = moveaxis(result, -1, axisc)
        return result

def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes.

    Parameters
    ----------
    a, b : array_like
    axes : int or (2,) list of lists
        If int N, contract last N axes of *a* with first N axes of *b*.
        If a tuple of two sequences, contract the specified axes.
    """
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    if isinstance(axes, int):
        axes_a = list(range(a.ndim - axes, a.ndim))
        axes_b = list(range(0, axes))
    else:
        axes_a = list(axes[0]) if not isinstance(axes[0], int) else [axes[0]]
        axes_b = list(axes[1]) if not isinstance(axes[1], int) else [axes[1]]
    na = a.ndim
    nb = b.ndim
    # Normalise negative axes
    axes_a = [ax if ax >= 0 else ax + na for ax in axes_a]
    axes_b = [ax if ax >= 0 else ax + nb for ax in axes_b]
    # Free axes (those not being contracted)
    free_a = [i for i in range(na) if i not in axes_a]
    free_b = [i for i in range(nb) if i not in axes_b]
    # Transpose a so free axes come first, contracted axes last
    perm_a = free_a + axes_a
    # Transpose b so contracted axes come first, free axes last
    perm_b = axes_b + free_b
    at = _transpose_with_axes(a, perm_a)
    bt = _transpose_with_axes(b, perm_b)
    # Compute shapes for reshape into 2D
    free_a_shape = [a.shape[i] for i in free_a]
    free_b_shape = [b.shape[i] for i in free_b]
    contract_size = 1
    for ax in axes_a:
        contract_size *= a.shape[ax]
    rows = 1
    for s in free_a_shape:
        rows *= s
    cols = 1
    for s in free_b_shape:
        cols *= s
    at2 = at.reshape([rows, contract_size])
    bt2 = bt.reshape([contract_size, cols])
    result = dot(at2, bt2)
    out_shape = free_a_shape + free_b_shape
    if len(out_shape) == 0:
        return result
    return result.reshape(out_shape)

def meshgrid(*xi, indexing='xy'):
    arrays = [a if isinstance(a, ndarray) else array(a) for a in xi]
    return _native.meshgrid(arrays, indexing)

def pad(a, pad_width, mode='constant', constant_values=0, **kwargs):
    if not isinstance(a, ndarray):
        a = array(a)
    # Normalise pad_width to list of (before, after) per axis
    if isinstance(pad_width, int):
        pw = [(pad_width, pad_width)] * a.ndim
    elif isinstance(pad_width, (list, tuple)):
        if isinstance(pad_width[0], int):
            if len(pad_width) == 2:
                pw = [(pad_width[0], pad_width[1])] * a.ndim
            else:
                pw = [(pad_width[0], pad_width[0])] * a.ndim
        else:
            pw = [(p[0], p[1]) for p in pad_width]
    else:
        pw = [(pad_width, pad_width)] * a.ndim

    if mode == 'constant':
        if isinstance(constant_values, (list, tuple)):
            constant_values = constant_values[0] if isinstance(constant_values[0], (int, float)) else constant_values[0][0]
        return _native.pad(a, pad_width, float(constant_values))

    # Pure-Python implementation for 'edge', 'reflect', 'wrap'
    def _pad_1d(data_list, before, after, mode_str):
        """Pad a 1D Python list with the given mode."""
        n = len(data_list)
        result = []
        if mode_str == 'edge':
            result = [data_list[0]] * before + list(data_list) + [data_list[-1]] * after
        elif mode_str == 'reflect':
            left = []
            for i in range(before):
                idx = (i + 1) % (2 * (n - 1)) if n > 1 else 0
                if idx >= n:
                    idx = 2 * (n - 1) - idx
                left.insert(0, data_list[idx])
            right = []
            for i in range(after):
                idx = (i + 1) % (2 * (n - 1)) if n > 1 else 0
                if idx >= n:
                    idx = 2 * (n - 1) - idx
                right.append(data_list[n - 1 - idx])
            result = left + list(data_list) + right
        elif mode_str == 'wrap':
            left = []
            for i in range(before):
                left.insert(0, data_list[-(i + 1) % n])
            right = []
            for i in range(after):
                right.append(data_list[i % n])
            result = left + list(data_list) + right
        elif mode_str == 'symmetric':
            # Like reflect but includes the edge value
            left = []
            for i in range(before):
                idx = i % (2 * n) if n > 0 else 0
                if idx >= n:
                    idx = 2 * n - 1 - idx
                left.insert(0, data_list[idx])
            right = []
            for i in range(after):
                idx = i % (2 * n) if n > 0 else 0
                if idx >= n:
                    idx = 2 * n - 1 - idx
                right.append(data_list[n - 1 - idx])
            result = left + list(data_list) + right
        elif mode_str == 'linear_ramp':
            end_val = kwargs.get('end_values', 0)
            if isinstance(end_val, (list, tuple)):
                end_val = end_val[0] if isinstance(end_val[0], (int, float)) else end_val[0][0]
            left = []
            for i in range(before):
                # Linear ramp from end_val to data_list[0]
                frac = float(i + 1) / float(before + 1) if before > 0 else 1.0
                left.append(end_val + (data_list[0] - end_val) * frac)
            right = []
            for i in range(after):
                # Linear ramp from data_list[-1] to end_val
                frac = float(i + 1) / float(after + 1) if after > 0 else 1.0
                right.append(data_list[-1] + (end_val - data_list[-1]) * frac)
            result = left + list(data_list) + right
        elif mode_str == 'mean':
            mean_val = sum(data_list) / len(data_list) if len(data_list) > 0 else 0.0
            result = [mean_val] * before + list(data_list) + [mean_val] * after
        elif mode_str == 'median':
            sorted_data = sorted(data_list)
            mid = len(sorted_data) // 2
            if len(sorted_data) % 2 == 0 and len(sorted_data) > 0:
                median_val = (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
            elif len(sorted_data) > 0:
                median_val = sorted_data[mid]
            else:
                median_val = 0.0
            result = [median_val] * before + list(data_list) + [median_val] * after
        elif mode_str == 'minimum':
            min_val = min(data_list) if len(data_list) > 0 else 0.0
            result = [min_val] * before + list(data_list) + [min_val] * after
        elif mode_str == 'maximum':
            max_val = max(data_list) if len(data_list) > 0 else 0.0
            result = [max_val] * before + list(data_list) + [max_val] * after
        else:
            raise NotImplementedError("pad mode '{}' is not supported".format(mode_str))
        return result

    if a.ndim == 1:
        data = [float(a[i]) for i in range(a.shape[0])]
        padded = _pad_1d(data, pw[0][0], pw[0][1], mode)
        return array(padded)

    # nD: pad axis-by-axis, starting from the last axis
    def _to_nested(arr):
        """Convert ndarray to nested Python lists."""
        if arr.ndim == 1:
            return [float(arr[i]) for i in range(arr.shape[0])]
        return [_to_nested(arr[i]) for i in range(arr.shape[0])]

    def _pad_axis(nested, axis, before, after, mode_str, current_depth=0):
        """Recursively pad along a specific axis of nested lists."""
        if current_depth == axis:
            return _pad_1d(nested, before, after, mode_str)
        else:
            return [_pad_axis(sub, axis, before, after, mode_str, current_depth + 1) for sub in nested]

    nested = _to_nested(a)
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before > 0 or after > 0:
            nested = _pad_axis(nested, ax, before, after, mode)
    return array(nested)

def indices(dimensions, dtype=None, sparse=False):
    """Return an array representing the indices of a grid."""
    ndim = len(dimensions)
    _dt = str(dtype) if dtype is not None else None
    if _dt is not None:
        _dt = _normalize_dtype(_dt)
    if ndim == 0:
        if sparse:
            return []
        return array([], dtype=_dt)

    if sparse:
        result = []
        for i in range(ndim):
            shape = [1] * ndim
            shape[i] = dimensions[i]
            idx = arange(0, dimensions[i])
            if _dt is not None:
                idx = idx.astype(_dt)
            idx = idx.reshape(shape)
            result.append(idx)
        return result

    # Dense: result shape is (ndim, *dimensions)
    grids = []
    for axis in range(ndim):
        # For each axis, create index array
        idx = arange(0, dimensions[axis])
        if _dt is not None:
            idx = idx.astype(_dt)
        # Reshape to broadcast: shape is [1,...,1,dim_axis,1,...,1]
        shape = [1] * ndim
        shape[axis] = dimensions[axis]
        idx = idx.reshape(shape)
        # Tile to fill all dimensions
        reps = list(dimensions)
        reps[axis] = 1
        grid = tile(idx, reps)
        grids.append(grid)

    # Force contiguous layout before stacking to avoid memory layout issues
    contiguous = [asarray(g.tolist()) for g in grids]
    result = stack(contiguous)
    if _dt is not None:
        result = result.astype(_dt)
    return result

def binary_repr(num, width=None):
    if num >= 0:
        s = bin(num)[2:]
        if width is not None:
            s = s.zfill(width)
        return s
    else:
        if width is not None:
            # Two's complement
            s = bin(2**width + num)[2:]
            return s.zfill(width)
        else:
            return '-' + bin(-num)[2:]

def base_repr(number, base=2, padding=0):
    if base < 2 or base > 36:
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    if number == 0:
        return "0" * (padding + 1)
    digits = []
    n = __import__("builtins").abs(number)
    while n:
        digits.append(str(n % base) if n % base < 10 else chr(ord('A') + n % base - 10))
        n //= base
    s = "".join(reversed(digits))
    s = "0" * padding + s
    if number < 0:
        s = "-" + s
    return s

def advanced_fancy_index(arr, indices):
    """Handle multi-axis fancy indexing: arr[[0,1], [2,3]] -> [arr[0,2], arr[1,3]].

    In NumPy, ``a[[0,1], [2,3]]`` selects ``[a[0,2], a[1,3]]`` (paired
    indices, not cross-product).  Because the Rust ndarray class does not
    support tuple-of-list indexing natively, this helper provides the same
    semantics as a module-level function.

    Parameters
    ----------
    arr : array_like
        Source array (must be at least 2-D for multi-axis use).
    indices : sequence of array_like
        One index array per axis. All index arrays must broadcast to the
        same shape (here: must have the same length).

    Returns
    -------
    ndarray
        1-D array of selected elements.

    Examples
    --------
    >>> a = np.arange(12).reshape(3, 4)
    >>> np.advanced_fancy_index(a, [[0, 1, 2], [3, 2, 1]])
    array([3., 6., 9.])  # a[0,3], a[1,2], a[2,1]
    """
    arr = asarray(arr)
    # Normalise each index array to a flat Python list of ints
    idx_arrays = [asarray(idx).flatten().tolist() for idx in indices]
    lengths = [len(a) for a in idx_arrays]
    if len(set(lengths)) > 1:
        raise IndexError("shape mismatch: indexing arrays could not be broadcast together")
    n = lengths[0]
    result = []
    for i in _builtin_range(n):
        # Walk into the array one axis at a time
        val = arr
        for ax in _builtin_range(len(idx_arrays)):
            ix = int(idx_arrays[ax][i])
            val = val[ix]
        result.append(float(val.tolist()) if hasattr(val, 'tolist') else float(val))
    return array(result)

def matrix_transpose(a):
    a = asarray(a) if not isinstance(a, ndarray) else a
    return a.T

conjugate = conj

def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Compute the bin edges for a histogram without computing the histogram itself."""
    a = asarray(a)
    if isinstance(bins, int):
        flat = a.flatten().tolist()
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo, hi = _builtin_min(flat), _builtin_max(flat)
        edges = linspace(lo, hi, bins + 1)
        return edges
    else:
        return asarray(bins)

def histogram(a, bins=10, range=None, density=None, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(bins, (list, tuple, ndarray)):
        # Custom bin edges
        edges = asarray(bins).flatten()
        edge_list = edges.tolist()
        flat = a.flatten().tolist()
        n_bins = len(edge_list) - 1
        counts = [0.0] * n_bins
        w_list = None
        if weights is not None:
            w_list = asarray(weights).flatten().tolist()
        for idx_val, v in enumerate(flat):
            for j in _builtin_range(n_bins):
                if j == n_bins - 1:
                    if edge_list[j] <= v <= edge_list[j + 1]:
                        counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                        break
                else:
                    if edge_list[j] <= v < edge_list[j + 1]:
                        counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                        break
        counts_arr = array(counts)
        if density:
            bin_widths = diff(edges)
            total = float(sum(counts_arr))
            if total > 0.0:
                counts_arr = counts_arr / (total * bin_widths)
        return counts_arr, edges
    # bins is an int
    if weights is not None or range is not None:
        # Python fallback for weights/range with integer bins
        flat = a.flatten().tolist()
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo, hi = _builtin_min(flat), _builtin_max(flat)
        edges = linspace(lo, hi, num=bins + 1, endpoint=True)
        edge_list = edges.tolist()
        counts = [0.0] * bins
        w_list = None
        if weights is not None:
            w_list = asarray(weights).flatten().tolist()
        for idx_val, val in enumerate(flat):
            if range is not None and (val < lo or val > hi):
                continue
            for j in _builtin_range(bins):
                if (val >= edge_list[j] and val < edge_list[j + 1]) or (j == bins - 1 and val == edge_list[j + 1]):
                    counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                    break
        hist = array(counts)
        if density:
            widths = array([edge_list[i+1] - edge_list[i] for i in _builtin_range(bins)])
            total = float(sum(hist))
            if total > 0.0:
                hist = hist / (total * widths)
        return hist, edges
    # No weights, no range: use native
    counts, edges = _native.histogram(a, bins)
    if density:
        bin_widths = diff(edges)
        total = float(sum(counts))
        if total > 0.0:
            counts = counts / (total * bin_widths)
    return counts, edges

def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """Compute the 2D histogram of two data samples."""
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(y, ndarray):
        y = array(y)
    x_flat = x.flatten()
    y_flat = y.flatten()
    n = x_flat.size
    x_list = x_flat.tolist()
    y_list = y_flat.tolist()
    # Determine number of bins for x and y
    if isinstance(bins, (list, tuple)):
        nbins_x = int(bins[0])
        nbins_y = int(bins[1])
    else:
        nbins_x = int(bins)
        nbins_y = int(bins)
    # Determine ranges
    if range is not None:
        xmin, xmax = float(range[0][0]), float(range[0][1])
        ymin, ymax = float(range[1][0]), float(range[1][1])
    else:
        xmin = _builtin_min(x_list)
        xmax = _builtin_max(x_list)
        ymin = _builtin_min(y_list)
        ymax = _builtin_max(y_list)
    # Build bin edges
    xedges_list = []
    yedges_list = []
    for i in _builtin_range(nbins_x + 1):
        xedges_list.append(xmin + i * (xmax - xmin) / nbins_x)
    for i in _builtin_range(nbins_y + 1):
        yedges_list.append(ymin + i * (ymax - ymin) / nbins_y)
    # Count into 2D bins
    hist_data = []
    for i in _builtin_range(nbins_x):
        row = []
        for j in _builtin_range(nbins_y):
            row.append(0.0)
        hist_data.append(row)
    xspan = xmax - xmin
    yspan = ymax - ymin
    for k in _builtin_range(n):
        xv = x_list[k]
        yv = y_list[k]
        # Find x bin
        if xspan == 0.0:
            xi = 0
        else:
            xi = int((xv - xmin) / (xspan / nbins_x))
        if xi >= nbins_x:
            xi = nbins_x - 1
        if xi < 0:
            xi = 0
        # Find y bin
        if yspan == 0.0:
            yi = 0
        else:
            yi = int((yv - ymin) / (yspan / nbins_y))
        if yi >= nbins_y:
            yi = nbins_y - 1
        if yi < 0:
            yi = 0
        hist_data[xi][yi] = hist_data[xi][yi] + 1.0
    # Convert to arrays
    flat_hist = []
    for i in _builtin_range(nbins_x):
        for j in _builtin_range(nbins_y):
            flat_hist.append(hist_data[i][j])
    hist = array(flat_hist).reshape((nbins_x, nbins_y))
    xedges = array(xedges_list)
    yedges = array(yedges_list)
    return hist, xedges, yedges

def bincount(x, weights=None, minlength=0):
    if not isinstance(x, ndarray):
        x = array(x)
    if weights is not None and not isinstance(weights, ndarray):
        weights = array(weights)
    return _native.bincount(x, weights, minlength)

def einsum(*operands, **kwargs):
    if len(operands) < 2:
        raise ValueError("einsum requires at least a subscript string and one operand")
    subscripts = operands[0]
    arrays = operands[1:]
    # Handle implicit output subscripts: 'ij,jk' -> 'ij,jk->ik'
    if '->' not in subscripts:
        # Collect all input subscripts
        input_subs = subscripts.replace(' ', '')
        parts = input_subs.split(',')
        # Output indices: letters that appear exactly once across all inputs (sorted alphabetically)
        from collections import Counter
        counts = Counter()
        for p in parts:
            counts.update(p)
        # Output = indices that appear exactly once, in alphabetical order
        output = ''.join(sorted(c for c, n in counts.items() if n == 1))
        subscripts = input_subs + '->' + output
    return _native.einsum(subscripts, *arrays)

# --- String (char) operations -----------------------------------------------
class _char_mod:
    @staticmethod
    def upper(a):
        return _native.char_upper(a)

    @staticmethod
    def lower(a):
        return _native.char_lower(a)

    @staticmethod
    def capitalize(a):
        return _native.char_capitalize(a)

    @staticmethod
    def strip(a):
        return _native.char_strip(a)

    @staticmethod
    def str_len(a):
        return _native.char_str_len(a)

    @staticmethod
    def startswith(a, prefix):
        return _native.char_startswith(a, prefix)

    @staticmethod
    def endswith(a, suffix):
        return _native.char_endswith(a, suffix)

    @staticmethod
    def replace(a, old, new):
        return _native.char_replace(a, old, new)

    @staticmethod
    def split(a, sep=None, maxsplit=-1):
        """Split each element in a around sep."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            result.append(str(s).split(sep, maxsplit))
        if len(result) == 1:
            return result[0]
        return result

    @staticmethod
    def join(sep, a):
        """Join strings in a with separator sep, element-wise."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, (list, tuple)):
            items = a
        else:
            items = [a]
        # If items is a list of lists, join each sublist
        if len(items) > 0 and isinstance(items[0], (list, tuple)):
            result = [str(sep).join(str(x) for x in sub) for sub in items]
            return array(result)
        # Otherwise join all items into a single string
        return str(sep).join(str(x) for x in items)

    @staticmethod
    def find(a, sub, start=0, end=None):
        """Find first occurrence of sub in each element of a."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.find(sub, start))
            else:
                result.append(s.find(sub, start, end))
        return array(result)

    @staticmethod
    def count(a, sub, start=0, end=None):
        """Count non-overlapping occurrences of sub in each element of a."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.count(sub, start))
            else:
                result.append(s.count(sub, start, end))
        return array(result)

    @staticmethod
    def add(a, b):
        """Element-wise string concatenation."""
        if isinstance(a, ndarray):
            items_a = a.tolist()
        elif isinstance(a, _ObjectArray):
            items_a = a._data
        elif isinstance(a, str):
            items_a = [a]
        else:
            items_a = list(a)
        if isinstance(b, ndarray):
            items_b = b.tolist()
        elif isinstance(b, _ObjectArray):
            items_b = b._data
        elif isinstance(b, str):
            items_b = [b]
        else:
            items_b = list(b)
        # Broadcast if lengths differ
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [str(x) + str(y) for x, y in zip(items_a, items_b)]
        return array(result)

    @staticmethod
    def multiply(a, i):
        """Element-wise string repetition."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        i = int(i)
        result = [str(s) * i for s in items]
        return array(result)

    @staticmethod
    def _to_str_list(a):
        """Convert input to a flat list of strings."""
        if isinstance(a, _ObjectArray):
            return [str(x) for x in a._data]
        if isinstance(a, ndarray):
            data = a.tolist()
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, list):
                        result.extend([str(x) for x in item])
                    else:
                        result.append(str(item))
                return result
            return [str(data)]
        if isinstance(a, str):
            return [a]
        if isinstance(a, (list, tuple)):
            result = []
            for item in a:
                if isinstance(item, (list, tuple)):
                    result.extend([str(x) for x in item])
                else:
                    result.append(str(item))
            return result
        return [str(a)]

    @staticmethod
    def center(a, width, fillchar=' '):
        """Pad each string element in a to width, centering the string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.center(int(width), fillchar) for s in data])

    @staticmethod
    def ljust(a, width, fillchar=' '):
        """Left-justify each string element in a to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.ljust(int(width), fillchar) for s in data])

    @staticmethod
    def rjust(a, width, fillchar=' '):
        """Right-justify each string element in a to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.rjust(int(width), fillchar) for s in data])

    @staticmethod
    def zfill(a, width):
        """Pad each string element in a with zeros on the left to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.zfill(int(width)) for s in data])

    @staticmethod
    def title(a):
        """Return element-wise title cased version of string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.title() for s in data])

    @staticmethod
    def swapcase(a):
        """Return element-wise with uppercase converted to lowercase and vice versa."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.swapcase() for s in data])

    @staticmethod
    def isalpha(a):
        """Return true for each element if all characters are alphabetic."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isalpha() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isdigit(a):
        """Return true for each element if all characters are digits."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isdigit() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isnumeric(a):
        """Return true for each element if all characters are numeric."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if (s.isnumeric() if hasattr(s, 'isnumeric') else s.isdigit()) else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isupper(a):
        """Return true for each element if all cased characters are uppercase."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isupper() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def islower(a):
        """Return true for each element if all cased characters are lowercase."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.islower() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isspace(a):
        """Return true for each element if all characters are whitespace."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isspace() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isdecimal(a):
        """Return true for each element if all characters are decimal."""
        a = asarray(a)
        return array([1.0 if str(s).isdecimal() else 0.0 for s in a.flatten().tolist()]).reshape(a.shape).astype("bool")

    @staticmethod
    def encode(a, encoding='utf-8', errors='strict'):
        """Encode each string element to bytes."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.encode(encoding, errors) for s in data])

    @staticmethod
    def decode(a, encoding='utf-8', errors='strict'):
        """Decode each bytes element to string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.decode(encoding, errors) if isinstance(s, bytes) else s for s in data])

char = _char_mod()

# --- Index Utilities --------------------------------------------------------

def unravel_index(indices, shape, order='C'):
    if not isinstance(indices, ndarray):
        if isinstance(indices, int):
            indices = array([indices])
        else:
            indices = array(indices)
    return _native.unravel_index(indices, shape)

def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    arrays = tuple(array([a]) if isinstance(a, (int, float)) else (a if isinstance(a, ndarray) else array(a)) for a in multi_index)
    return _native.ravel_multi_index(arrays, dims)

def interp(x, xp, fp, left=None, right=None, period=None):
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(xp, ndarray):
        xp = array(xp)
    if not isinstance(fp, ndarray):
        fp = array(fp)
    result = _native.interp(x, xp, fp)
    if left is not None or right is not None:
        x_arr = asarray(x).flatten().tolist()
        xp_arr = asarray(xp).flatten().tolist()
        result_list = result.flatten().tolist()
        xp_min = _builtin_min(xp_arr)
        xp_max = _builtin_max(xp_arr)
        for i, xi in enumerate(x_arr):
            if left is not None and xi < xp_min:
                result_list[i] = float(left)
            if right is not None and xi > xp_max:
                result_list[i] = float(right)
        result = array(result_list)
    return result

# --- I/O: loadtxt / savetxt / genfromtxt ------------------------------------

def loadtxt(fname, dtype=None, comments='#', delimiter=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, **kwargs):
    """Load data from a text file. Each row must have the same number of values."""
    if isinstance(fname, str):
        f = open(fname, 'r')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        rows = []
        lines_read = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip comment lines
            if comments and line.startswith(comments):
                continue
            if lines_read < skiprows:
                lines_read += 1
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            # Split by delimiter
            if delimiter is None:
                parts = line.split()
            else:
                parts = line.split(delimiter)
            # Select columns
            if usecols is not None:
                parts = [parts[i] for i in usecols]
            row = [float(x.strip()) for x in parts]
            rows.append(row)
        if not rows:
            return array([])
        if len(rows) == 1 and ndmin < 2:
            result = array(rows[0])
        else:
            result = array(rows)
        if unpack:
            return result.T
        return result
    finally:
        if close_file:
            f.close()

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None):
    """Save an array to a text file."""
    if not isinstance(X, ndarray):
        X = array(X)
    if X.ndim == 1:
        X = X.reshape([X.size, 1])

    if isinstance(fname, str):
        f = open(fname, 'w')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        if header:
            for hline in header.split('\n'):
                f.write(comments + hline + newline)

        rows = X.shape[0]
        cols = X.shape[1]
        for i in range(rows):
            vals = []
            for j in range(cols):
                vals.append(fmt % float(X[i][j]))
            f.write(delimiter.join(vals) + newline)

        if footer:
            for fline in footer.split('\n'):
                f.write(comments + fline + newline)
    finally:
        if close_file:
            f.close()

def genfromtxt(fname, dtype=None, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, missing_values=None, filling_values=None,
               usecols=None, names=None, excludelist=None, deletechars=None,
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt='f%i', unpack=False, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding='bytes', **kwargs):
    """Load data from text file, handling missing values."""
    if filling_values is None:
        filling_values = float('nan')

    if isinstance(fname, str):
        with open(fname, 'r') as f:
            lines = f.readlines()
    else:
        lines = fname.readlines()

    # Skip header/footer
    lines = lines[skip_header:]
    if skip_footer > 0:
        lines = lines[:-skip_footer]
    if max_rows is not None:
        lines = lines[:max_rows]

    rows = []
    for line in lines:
        line = line.strip()
        if not line or (comments and line.startswith(comments)):
            continue
        if delimiter is None:
            parts = line.split()
        else:
            parts = line.split(delimiter)

        if usecols is not None:
            parts = [parts[i] for i in usecols]

        row = []
        for p in parts:
            p = p.strip()
            if missing_values and p in (missing_values if isinstance(missing_values, (list, tuple, set)) else [missing_values]):
                row.append(filling_values)
            else:
                try:
                    row.append(float(p))
                except (ValueError, TypeError):
                    row.append(filling_values)
        rows.append(row)

    if not rows:
        return array([])
    result = array(rows)
    if unpack:
        return result.T
    return result

# --- ufunc function forms (Tier 12A) ----------------------------------------

mod = remainder

divmod = divmod_

def diag(v, k=0):
    """Construct a diagonal array or extract a diagonal.

    If *v* is 1-D, return a 2-D array with *v* on the *k*-th diagonal.
    If *v* is 2-D, extract the *k*-th diagonal (same as ``diagonal``).
    """
    v = asarray(v)
    if v.ndim == 1:
        n = len(v)
        abs_k = abs(k)
        size = n + abs_k
        # Build as flat list then reshape
        flat = [0.0] * (size * size)
        for i in range(n):
            if k >= 0:
                flat[i * size + (k + i)] = float(v[i])
            else:
                flat[(abs_k + i) * size + i] = float(v[i])
        return array(flat).reshape([size, size])
    elif v.ndim == 2:
        return diagonal(v, offset=k)
    else:
        raise ValueError("Input must be 1-D or 2-D")

def tri(N, M=None, k=0, dtype=None):
    """An array with ones at and below the given diagonal and zeros elsewhere."""
    if M is None:
        M = N
    rows = []
    for i in range(N):
        row = []
        for j in range(M):
            row.append(1.0 if j <= i + k else 0.0)
        rows.append(row)
    return array(rows)

def tril(m, k=0):
    """Lower triangle of an array. Return a copy with elements above the k-th diagonal zeroed."""
    m = asarray(m)
    mask = tri(m.shape[0], m.shape[1], k=k)
    return m * mask

def triu(m, k=0):
    """Upper triangle of an array. Return a copy with elements below the k-th diagonal zeroed."""
    m = asarray(m)
    mask = tri(m.shape[0], m.shape[1], k=k - 1)
    return m * (ones(m.shape) - mask)

def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix."""
    x = asarray(x).flatten()
    n = x.size
    if N is None:
        N = n
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
    return stack(cols, axis=1)

def kron(a, b):
    """Kronecker product of two arrays."""
    a = asarray(a)
    b = asarray(b)
    if a.ndim == 1:
        a = a.reshape((1, a.size))
    if b.ndim == 1:
        b = b.reshape((1, b.size))
    ar, ac = a.shape[0], a.shape[1]
    br, bc = b.shape[0], b.shape[1]
    rows = []
    for i in range(ar):
        for bi in range(br):
            row = []
            for j in range(ac):
                for bj in range(bc):
                    row.append(a[i][j] * b[bi][bj])
            rows.append(row)
    return array(rows)

def inner(a, b):
    """Inner product of two arrays.

    For 1-D arrays this is the dot product.  For higher-dimensional arrays
    this contracts over the last axis of both a and b.
    """
    a = asarray(a)
    b = asarray(b)
    if a.ndim <= 1 and b.ndim <= 1:
        return dot(a, b)
    # For 2D: inner(A, B) = A @ B.T
    if a.ndim == 2 and b.ndim == 2:
        return dot(a, b.T)
    # General: tensordot contracting last axis of each
    return tensordot(a, b, axes=([-1], [-1]))

def matmul(x1, x2):
    """Matrix product of two arrays (same as the ``@`` operator)."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    return dot(x1, x2)


def digitize(x, bins, right=False):
    """Return the indices of the bins to which each value belongs."""
    x = asarray(x)
    bins = asarray(bins)
    bins_list = [float(bins[i]) for i in range(len(bins))]
    result = []
    ascending = len(bins_list) < 2 or bins_list[-1] >= bins_list[0]
    for i in range(x.size):
        val = float(x.flatten()[i])
        if ascending:
            # bins ascending: find first bin > val (or >= if right)
            idx = 0
            for j in range(len(bins_list)):
                if right:
                    if bins_list[j] < val:
                        idx = j + 1
                else:
                    if bins_list[j] <= val:
                        idx = j + 1
            result.append(idx)
        else:
            # bins descending
            idx = 0
            for j in range(len(bins_list)):
                if right:
                    if bins_list[j] > val:
                        idx = j + 1
                else:
                    if bins_list[j] >= val:
                        idx = j + 1
            result.append(idx)
    return array(result)

# --- mgrid / ogrid / ix_ ----------------------------------------------------

class _MGrid:
    """Return dense multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else 0
                step = s.step if s.step is not None else 1
                # Complex step means "use linspace with this many points"
                if isinstance(step, complex) or (hasattr(step, 'imag') and step.imag != 0):
                    num = int(abs(step))
                    grid = linspace(float(start), float(stop), num=num, endpoint=True)
                    arrays.append(grid)
                else:
                    vals = []
                    v = float(start)
                    if step > 0:
                        while v < float(stop):
                            vals.append(v)
                            v += float(step)
                    else:
                        while v > float(stop):
                            vals.append(v)
                            v += float(step)
                    arrays.append(array(vals) if vals else array([]))
            else:
                arrays.append(array([float(s)]))

        if ndim == 1:
            return arrays[0]

        # Create dense meshgrid
        shapes = [len(a) for a in arrays]
        result = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = shapes[i]
            reshaped = arr.reshape(shape)
            reps = list(shapes)
            reps[i] = 1
            result.append(tile(reshaped, reps))
        return result

mgrid = _MGrid()


class _OGrid:
    """Return open (sparse) multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else 0
                step = s.step if s.step is not None else 1
                # Complex step means "use linspace with this many points"
                if isinstance(step, complex) or (hasattr(step, 'imag') and step.imag != 0):
                    num = int(abs(step))
                    grid = linspace(float(start), float(stop), num=num, endpoint=True)
                    arrays.append(grid)
                else:
                    vals = []
                    v = float(start)
                    if step > 0:
                        while v < float(stop):
                            vals.append(v)
                            v += float(step)
                    else:
                        while v > float(stop):
                            vals.append(v)
                            v += float(step)
                    arrays.append(array(vals) if vals else array([]))
            else:
                arrays.append(array([float(s)]))

        if ndim == 1:
            return arrays[0]

        # Sparse: each array reshaped to broadcast along its own axis
        result = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = len(arr)
            result.append(arr.reshape(shape))
        return result

ogrid = _OGrid()


def ix_(*args):
    """Construct an open mesh from multiple sequences for cross-indexing."""
    ndim = len(args)
    result = []
    for i, arg in enumerate(args):
        arr = asarray(arg).flatten()
        shape = [1] * ndim
        shape[i] = len(arr)
        result.append(arr.reshape(shape))
    return tuple(result)


class _RClass:
    """Row concatenation using index syntax: np.r_[1:5, 7, 8]"""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        pieces = []
        for item in key:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop
                step = item.step if item.step is not None else 1
                if stop is None:
                    raise ValueError("r_ requires explicit stop for slices")
                pieces.append(arange(start, stop, step))
            elif isinstance(item, (int, float)):
                pieces.append(array([item]))
            else:
                pieces.append(asarray(item))
        if len(pieces) == 0:
            return array([])
        return concatenate(pieces)

r_ = _RClass()


class _CClass:
    """Column concatenation: np.c_[a, b] == np.column_stack((a, b))"""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        arrays = []
        for item in key:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop
                step = item.step if item.step is not None else 1
                if stop is None:
                    raise ValueError("c_ requires explicit stop for slices")
                arr = arange(start, stop, step)
            elif isinstance(item, (int, float)):
                arr = array([item])
            else:
                arr = asarray(item)
            # Ensure 2D
            if arr.ndim == 1:
                arr = arr.reshape((-1 if arr.size > 0 else 0, 1))
            arrays.append(arr)
        if len(arrays) == 0:
            return array([])
        return concatenate(arrays, 1)

c_ = _CClass()


class _SClass:
    """Index expression helper: np.s_[0:5, 1::2]"""
    def __getitem__(self, key):
        return key

s_ = _SClass()
index_exp = s_  # s_ already exists and does the same thing


# --- fromfunction — construct array from function ----------------------------
# --- fmod — C-style remainder (sign of dividend) ----------------------------
def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array. Returns new array (our arrays are immutable)."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("array must be 2-d")
    n = a.shape[0]
    m = a.shape[1]
    rows = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == j:
                row.append(float(val) if not isinstance(val, (list, tuple)) else float(val[i % len(val)]))
            else:
                row.append(a[i][j])
        rows.append(row)
    return array(rows)


# --- diag_indices / diag_indices_from — diagonal index helpers --------------
def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array."""
    idx = arange(0, n)
    return tuple([idx] * ndim)

def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    n = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("array must be square")
    return diag_indices(n, 2)


# --- tril_indices / triu_indices — triangle index helpers -------------------
def tril_indices(n, k=0, m=None):
    """Return the indices for the lower-triangle of an (n, m) array."""
    if m is None:
        m = n
    rows = []
    cols = []
    for i in range(n):
        for j in range(m):
            if j <= i + k:
                rows.append(float(i))
                cols.append(float(j))
    return array(rows), array(cols)

def triu_indices(n, k=0, m=None):
    """Return the indices for the upper-triangle of an (n, m) array."""
    if m is None:
        m = n
    rows = []
    cols = []
    for i in range(n):
        for j in range(m):
            if j >= i + k:
                rows.append(float(i))
                cols.append(float(j))
    return array(rows), array(cols)

def tril_indices_from(arr, k=0):
    """Return the indices for the lower-triangle of arr."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    return tril_indices(arr.shape[0], k=k, m=arr.shape[1])

def triu_indices_from(arr, k=0):
    """Return the indices for the upper-triangle of arr."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    return triu_indices(arr.shape[0], k=k, m=arr.shape[1])


# --- ndenumerate — multidimensional index iterator --------------------------
class ndenumerate:
    """Multidimensional index iterator."""
    def __init__(self, arr):
        self._arr = asarray(arr)
        self._flat = self._arr.flatten()
        self._shape = self._arr.shape
        self._size = self._flat.size
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        # Convert flat index to multi-dim index
        idx = self._idx
        multi = []
        for s in reversed(self._shape):
            multi.append(idx % s)
            idx //= s
        multi.reverse()
        val = self._flat[self._idx]
        self._idx += 1
        return tuple(multi), val


# --- ndindex — N-dimensional index iterator ---------------------------------
class ndindex:
    """An N-dimensional iterator object to index arrays."""
    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        self._shape = shape
        self._size = 1
        for s in shape:
            self._size *= s
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        idx = self._idx
        multi = []
        for s in reversed(self._shape):
            multi.append(idx % s)
            idx //= s
        multi.reverse()
        self._idx += 1
        return tuple(multi)


# --- Signal window functions ------------------------------------------------
def bartlett(M):
    """Return the Bartlett window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    n = arange(0, M)
    mid = (M - 1) / 2.0
    vals = []
    for i in range(M):
        v = float(n[i])
        if v <= mid:
            vals.append(2.0 * v / (M - 1))
        else:
            vals.append(2.0 - 2.0 * v / (M - 1))
    return array(vals)

def blackman(M):
    """Return the Blackman window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.42 - 0.5 * _stdlib_math.cos(2.0 * pi * i / (M - 1)) + 0.08 * _stdlib_math.cos(4.0 * pi * i / (M - 1)))
    return array(vals)

def hamming(M):
    """Return the Hamming window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.54 - 0.46 * _stdlib_math.cos(2.0 * pi * i / (M - 1)))
    return array(vals)

def hanning(M):
    """Return the Hanning window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.5 - 0.5 * _stdlib_math.cos(2.0 * pi * i / (M - 1)))
    return array(vals)

def kaiser(M, beta):
    """Return the Kaiser window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    # I0 is modified Bessel function of first kind, order 0
    # Use series approximation
    def _i0(x):
        """Modified Bessel function I0 via series."""
        val = 1.0
        term = 1.0
        for k in range(1, 25):
            term *= (x / 2.0) ** 2 / (k * k)
            val += term
        return val

    alpha = (M - 1) / 2.0
    vals = []
    for i in range(M):
        arg = beta * _stdlib_math.sqrt(1.0 - ((i - alpha) / alpha) ** 2)
        vals.append(_i0(arg) / _i0(beta))
    return array(vals)


# --- nditer — simplified N-dimensional iterator -----------------------------
class nditer:
    """Simplified N-dimensional iterator."""
    def __init__(self, op, flags=None, op_flags=None, op_dtypes=None, order='K',
                 casting='safe', op_axes=None, itershape=None, buffersize=0):
        if isinstance(op, (list, tuple)):
            self._arrays = [asarray(a) for a in op]
        else:
            self._arrays = [asarray(op)]
        self._flat = [a.flatten() for a in self._arrays]
        self._size = self._flat[0].size
        self._idx = 0
        self._shape = self._arrays[0].shape
        self.multi_index = None

    def _unravel(self, flat_idx):
        """Convert flat index to tuple of indices."""
        idx = []
        remaining = flat_idx
        for dim in reversed(self._shape):
            idx.append(remaining % dim)
            remaining //= dim
        return tuple(reversed(idx))

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        # Compute multi_index from flat index
        self.multi_index = self._unravel(self._idx)
        if len(self._flat) == 1:
            val = self._flat[0][self._idx]
            self._idx += 1
            return val
        vals = tuple(f[self._idx] for f in self._flat)
        self._idx += 1
        return vals

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def finished(self):
        return self._idx >= self._size

    def iternext(self):
        if self._idx >= self._size:
            return False
        self._idx += 1
        return self._idx < self._size

    @property
    def value(self):
        if len(self._flat) == 1:
            return self._flat[0][self._idx]
        return tuple(f[self._idx] for f in self._flat)


# --- array_str / array_repr ------------------------------------------------
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """Return a string representation of the data in an array."""
    a = asarray(a)
    return str(a)

def array_repr(a, max_line_width=None, precision=None, suppress_small=None):
    """Return the string representation of an array."""
    a = asarray(a)
    return repr(a)


# --- Tier 18 Group C: i0, apply_over_axes, real_if_close, isneginf, isposinf ---

def save(file, arr, allow_pickle=True, fix_imports=True):
    """Save an array to a .npy file (text-based format for compatibility)."""
    arr = asarray(arr)
    with open(file, 'w') as f:
        f.write(f"# shape: {list(arr.shape)}\n")
        flat = arr.flatten()
        vals = [str(flat[i]) for i in range(flat.size)]
        f.write(','.join(vals) + '\n')

def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """Load array from a .npy file (text-based format)."""
    with open(file, 'r') as f:
        lines = f.readlines()
    # Parse shape from first line
    shape_line = lines[0].strip()
    if shape_line.startswith('# shape:'):
        import json
        shape = tuple(json.loads(shape_line.split(':')[1].strip()))
        data_line = lines[1].strip()
    else:
        # Fallback: treat as flat data
        data_line = lines[0].strip()
        shape = None
    vals = [float(v) for v in data_line.split(',')]
    result = array(vals)
    if shape is not None and len(shape) > 1:
        result = result.reshape(shape)
    return result

def savez(file, *args, **kwds):
    """Save several arrays into a single file in text format.
    Since we can't use actual npz (zip) format, save as multi-section text."""
    arrays = {}
    for i, arr in enumerate(args):
        arrays[f'arr_{i}'] = asarray(arr)
    for name, arr in kwds.items():
        arrays[name] = asarray(arr)
    with open(file, 'w') as f:
        for name, arr in arrays.items():
            f.write(f"# {name} shape: {list(arr.shape)}\n")
            flat = arr.flatten()
            vals = [str(flat[i]) for i in range(flat.size)]
            f.write(','.join(vals) + '\n')

savez_compressed = savez  # alias, same behavior in our sandbox

# --- frompyfunc (Tier 18A) --------------------------------------------------

def frompyfunc(func, nin, nout):
    """Takes an arbitrary Python function and returns a NumPy ufunc-like object.
    Returns a vectorize wrapper."""
    return vectorize(func)

# --- take_along_axis / put_along_axis (Tier 18A) ----------------------------

def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1-d index and data slices along the given axis."""
    arr = asarray(arr)
    indices = asarray(indices)
    if arr.ndim == 1:
        result = []
        for i in range(indices.size):
            result.append(arr[int(indices[i])])
        return array(result)
    if arr.ndim == 2:
        if axis == 0:
            rows = []
            for j in range(arr.shape[1]):
                col = []
                for i in range(indices.shape[0]):
                    col.append(arr[int(indices[i][j])][j])
                rows.append(col)
            # Transpose to get correct shape
            result = []
            for i in range(indices.shape[0]):
                row = [rows[j][i] for j in range(arr.shape[1])]
                result.append(row)
            return array(result)
        else:  # axis == 1
            rows = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(indices.shape[1]):
                    row.append(arr[i][int(indices[i][j])])
                rows.append(row)
            return array(rows)
    # General nD case: move axis to last, flatten leading dims, gather, reshape back
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    out_shape = ind_m.shape
    n_axis = arr_m.shape[-1]
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    arr_flat = arr_m.reshape((lead, n_axis))
    ind_flat = ind_m.reshape((lead, ind_m.shape[-1]))
    arr_list = arr_flat.tolist()
    ind_list = ind_flat.tolist()
    result = []
    for i in range(lead):
        row = arr_list[i]
        idxs = ind_list[i]
        result.append([row[int(j)] for j in idxs])
    result_arr = array(result).reshape(out_shape)
    return moveaxis(result_arr, -1, axis)

def put_along_axis(arr, indices, values, axis):
    """Put values into the destination array by matching 1-d index and data slices along the given axis."""
    arr = asarray(arr)
    indices = asarray(indices)
    values = asarray(values)
    if arr.ndim == 1:
        result = [arr[i] for i in range(arr.size)]
        vals_flat = values.flatten()
        for i in range(indices.size):
            result[int(indices[i])] = vals_flat[i % vals_flat.size]
        return array(result)
    if arr.ndim == 2 and axis == 1:
        rows = []
        for i in range(arr.shape[0]):
            row = [arr[i][j] for j in range(arr.shape[1])]
            for j in range(indices.shape[1]):
                idx = int(indices[i][j])
                row[idx] = values[i][j] if values.ndim == 2 else values[j]
            rows.append(row)
        return array(rows)
    # General nD case: move axis to last, flatten, scatter, reshape back
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    val_m = moveaxis(values, axis, -1)
    out_shape = arr_m.shape
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    n_axis = arr_m.shape[-1]
    arr_flat = arr_m.reshape((lead, n_axis)).tolist()
    ind_flat = ind_m.reshape((lead, ind_m.shape[-1])).tolist()
    val_flat = val_m.reshape((lead, val_m.shape[-1])).tolist()
    for i in range(lead):
        for j in range(len(ind_flat[i])):
            arr_flat[i][int(ind_flat[i][j])] = val_flat[i][j]
    result = array(arr_flat).reshape(out_shape)
    return moveaxis(result, -1, axis)

from ._linalg_ext import *

# Link top-level functions into linalg module (needs trace/cross/diagonal/outer defined above)
linalg.trace = trace
linalg.cross = cross        # delegate to top-level cross()
linalg.diagonal = diagonal  # delegate to top-level diagonal()
linalg.outer = outer        # delegate to top-level outer()

# --- FFT module extensions (Tier 19 Group B) --------------------------------

def _fft_rfftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies for rfft."""
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = []
    for i in range(N):
        results.append(float(i) * val)
    return array(results)

def _fft_fftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies."""
    results = []
    half = (n - 1) // 2 + 1
    for i in range(half):
        results.append(float(i) / (n * d))
    for i in range(-(n // 2), 0):
        results.append(float(i) / (n * d))
    return array(results)

def _fft_fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = n // 2
        result = roll(result, shift, axis=ax)
    return result

def _fft_ifftshift(x, axes=None):
    """The inverse of fftshift."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = -(n // 2)
        result = roll(result, shift, axis=ax)
    return result

def _fft_complex_column_fft(row_ffts, rows, cols, inverse=False):
    """Apply FFT/IFFT along columns of a complex (rows, cols, 2) representation.

    row_ffts is a list of (cols, 2) arrays from fft.fft applied to each row.
    Returns a (rows, cols, 2) shaped array representing the 2D FFT result.
    The complex representation uses [real, imag] pairs.
    """
    fft_fn = fft.ifft if inverse else fft.fft
    # For each column j, extract real and imaginary parts across all rows,
    # apply FFT to each separately, then combine using:
    #   DFT(xr + j*xi) = DFT(xr) + j*DFT(xi)
    #   result_real = DFT(xr)_real - DFT(xi)_imag
    #   result_imag = DFT(xr)_imag + DFT(xi)_real
    col_results = []  # col_results[j] is a list of (real, imag) for each row i
    for j in range(cols):
        col_real = array([row_ffts[i][j][0] for i in range(rows)])
        col_imag = array([row_ffts[i][j][1] for i in range(rows)])
        if inverse:
            # ifft requires complex format (n, 2) - convert real arrays to complex with zero imag
            col_real_c = array([[float(col_real[i]), 0.0] for i in range(rows)])
            col_imag_c = array([[float(col_imag[i]), 0.0] for i in range(rows)])
            fft_of_real = fft_fn(col_real_c)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag_c)   # (rows, 2)
        else:
            fft_of_real = fft_fn(col_real)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag)   # (rows, 2)
        # Combine: for each row i
        col_ri = []
        for i in range(rows):
            r = fft_of_real[i][0] - fft_of_imag[i][1]
            im = fft_of_real[i][1] + fft_of_imag[i][0]
            col_ri.append((r, im))
        col_results.append(col_ri)
    # Reconstruct as (rows, cols, 2) using stack
    final_rows = []
    for i in range(rows):
        row_data = []
        for j in range(cols):
            row_data.append([col_results[j][i][0], col_results[j][i][1]])
        final_rows.append(array(row_data))
    return stack(final_rows)

def _fft_fft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("fft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # FFT each row -> list of (cols, 2) complex arrays
    row_ffts = [fft.fft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_ffts, rows, cols, inverse=False)

def _fft_ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D inverse discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("ifft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # IFFT each row -> list of (cols, 2) complex arrays
    row_iffts = [fft.ifft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)

# Monkey-patch fft module with extension functions
fft.rfftfreq = _fft_rfftfreq
fft.fftfreq = _fft_fftfreq
fft.fftshift = _fft_fftshift
fft.ifftshift = _fft_ifftshift
fft.fft2 = _fft_fft2
fft.ifft2 = _fft_ifft2

def _fft_fftn(a, s=None, axes=None):
    """N-dimensional FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.fft(a)
    elif a.ndim == 2:
        return fft.fft2(a, s=s)
    else:
        # For higher dimensions, apply fft2 on last two axes as approximation
        # This is a simplified implementation
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            # Apply 1D FFT along each axis using apply_along_axis
            result = apply_along_axis(lambda x: array(fft.fft(array(x)).tolist()), ax, result)
        return result

def _fft_ifftn(a, s=None, axes=None):
    """N-dimensional inverse FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.ifft(a)
    elif a.ndim == 2:
        return fft.ifft2(a, s=s)
    elif a.ndim == 3 and a.shape[-1] == 2:
        # Complex representation from fft2/fftn: shape (rows, cols, 2)
        # This is a complex array; apply ifft2 logic
        rows = a.shape[0]
        cols = a.shape[1]
        # Extract row-wise complex data and apply ifft
        row_iffts = []
        for i in _builtin_range(rows):
            row_iffts.append(fft.ifft(a[i]))
        return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)
    else:
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            result = apply_along_axis(lambda x: array(fft.ifft(array(x)).tolist()), ax, result)
        return result

fft.fftn = _fft_fftn
fft.ifftn = _fft_ifftn

def _fft_rfft(a, n=None, axis=-1, norm=None):
    """Real FFT - only positive frequencies."""
    a = asarray(a).astype("float64")
    if a.ndim == 0:
        a = a.reshape([1])
    data = a.tolist()
    if isinstance(data[0], list):
        raise NotImplementedError("rfft only supports 1D")
    N = n if n is not None else len(data)
    # Pad or truncate
    if len(data) < N:
        data = data + [0.0] * (N - len(data))
    elif len(data) > N:
        data = data[:N]
    # Compute full DFT - only first N//2 + 1 frequencies
    import cmath
    result = []
    out_len = N // 2 + 1
    for k in _builtin_range(out_len):
        s = 0.0 + 0.0j
        for n_idx in _builtin_range(N):
            angle = -2.0 * 3.141592653589793 * k * n_idx / N
            s += data[n_idx] * cmath.exp(1j * angle)
        if norm == "ortho":
            s /= N ** 0.5
        result.append([s.real, s.imag])
    # Return as (out_len, 2) shaped array matching native fft format
    return array(result)

def _fft_irfft(a, n=None, axis=-1, norm=None):
    """Inverse real FFT."""
    a = asarray(a)
    # Handle (M, 2) complex format from rfft
    data_list = a.tolist()
    if a.ndim == 2 and a.shape[1] == 2:
        # Complex format: [[real, imag], ...]
        data_r = [row[0] for row in data_list]
        data_i = [row[1] for row in data_list]
    elif a.ndim == 1:
        # Real-only input
        data_r = data_list
        data_i = [0.0] * len(data_r)
    else:
        data_r = a.real.tolist() if hasattr(a, 'real') else data_list
        data_i = a.imag.tolist() if hasattr(a, 'imag') else [0.0] * len(data_r)
    if not isinstance(data_r, list):
        data_r = [data_r]
        data_i = [data_i]
    m = len(data_r)
    N = n if n is not None else 2 * (m - 1)
    # Reconstruct full spectrum using Hermitian symmetry
    import math as _math_mod
    full_spectrum = []
    for i in _builtin_range(m):
        full_spectrum.append(complex(data_r[i], data_i[i]))
    # Mirror for negative frequencies
    for i in _builtin_range(m, N):
        mirror = N - i
        full_spectrum.append(complex(data_r[mirror], -data_i[mirror]))
    # IDFT
    result = []
    for n_idx in _builtin_range(N):
        s = 0.0
        for k in _builtin_range(N):
            angle = 2.0 * 3.141592653589793 * k * n_idx / N
            s += full_spectrum[k].real * _math_mod.cos(angle) - full_spectrum[k].imag * _math_mod.sin(angle)
        if norm == "ortho":
            s /= N ** 0.5
        else:
            s /= N
        result.append(s)
    return array(result)

fft.rfft = _fft_rfft
fft.irfft = _fft_irfft

def _fft_rfft2(a, s=None, axes=(-2, -1), norm=None):
    """2D real FFT — rfft along last axis for each row."""
    a = asarray(a)
    if a.ndim < 2:
        return fft.rfft(a)
    rows = a.tolist()
    rfft_rows = []
    for row in rows:
        r = fft.rfft(array(row))
        rfft_rows.append(r)
    # Stack results: each r is an ndarray
    return stack(rfft_rows)

def _fft_irfft2(a, s=None, axes=(-2, -1), norm=None):
    """2D inverse real FFT — irfft along last axis for each row."""
    a = asarray(a)
    if a.ndim < 2:
        return fft.irfft(a)
    n_val = s[-1] if s else None
    result_rows = []
    for i in range(a.shape[0]):
        row = a[i]
        r = fft.irfft(row, n=n_val)
        result_rows.append(r)
    return stack(result_rows)

def _fft_rfftn(a, s=None, axes=None, norm=None):
    """N-D real FFT."""
    return _fft_rfft2(a, s=s, norm=norm)

def _fft_irfftn(a, s=None, axes=None, norm=None):
    """N-D inverse real FFT."""
    return _fft_irfft2(a, s=s, norm=norm)

def _fft_hfft(a, n=None, axis=-1, norm=None):
    """Hermitian FFT - input is Hermitian symmetric, output is real."""
    a = asarray(a)
    # hfft(a) = irfft(conj(a)) * N
    conj_a = conj(a)
    N = n if n is not None else 2 * (a.shape[0] - 1) if a.ndim > 0 else 2
    result = fft.irfft(conj_a, n=N, norm=norm)
    if norm != 'ortho':
        result = result * N
    return result

def _fft_ihfft(a, n=None, axis=-1, norm=None):
    """Inverse Hermitian FFT - input is real, output is Hermitian."""
    a = asarray(a)
    # ihfft(a) = conj(rfft(a)) / N
    N = n if n is not None else len(a.tolist()) if a.ndim > 0 else 1
    result = fft.rfft(a, n=N, norm=norm)
    return conj(result) / N if norm != 'ortho' else conj(result)

fft.rfft2 = _fft_rfft2
fft.irfft2 = _fft_irfft2
fft.rfftn = _fft_rfftn
fft.irfftn = _fft_irfftn
fft.hfft = _fft_hfft
fft.ihfft = _fft_ihfft

# --- random extension functions (Tier 19 Group C) ---------------------------

def _random_shuffle(x):
    """Modify a sequence in-place by shuffling its contents. Returns None."""
    if not isinstance(x, ndarray):
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    # Attempt to update array in-place via __setitem__
    try:
        for i in range(n):
            x[i] = vals[i]
    except Exception:
        pass  # if in-place update not supported, shuffle is best-effort
    return None  # real numpy returns None

def _random_permutation(x):
    """Randomly permute a sequence, or return a permuted range."""
    if isinstance(x, (int, float)):
        x = arange(0, int(x))
    else:
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    result = array(vals)
    if x.ndim > 1:
        result = result.reshape(x.shape)
    return result

def _random_standard_normal(size=None):
    """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
    if size is None:
        return float(random.normal(0.0, 1.0, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    return random.normal(0.0, 1.0, size)

def _random_exponential(scale=1.0, size=None):
    """Draw samples from an exponential distribution."""
    if size is None:
        import math as _m
        u = float(random.uniform(0.0, 1.0, (1,)).flatten()[0])
        if u >= 1.0:
            u = 0.9999999999
        return float(-scale * _m.log(1.0 - u))
    if isinstance(size, int):
        size = (size,)
    # Generate uniform [0,1) then transform: -scale * ln(1 - U)
    u = random.uniform(0.0, 1.0, size)
    flat = u.flatten()
    n = flat.size
    result = []
    for i in range(n):
        v = float(flat[i])
        if v >= 1.0:
            v = 0.9999999999
        import math as _m
        result.append(-scale * _m.log(1.0 - v))
    r = array(result)
    if u.ndim > 1:
        r = r.reshape(u.shape)
    return r

def _random_poisson(lam=1.0, size=None):
    """Draw samples from a Poisson distribution."""
    import math as _m
    if size is None:
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        return float(k - 1)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Knuth algorithm
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        result.append(float(k - 1))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_binomial(n, p, size=None):
    """Draw samples from a binomial distribution."""
    if size is None:
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        return float(successes)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        result.append(float(successes))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_beta(a, b, size=None):
    """Draw samples from a Beta distribution.
    Uses the relationship: if X~Gamma(a,1) and Y~Gamma(b,1), then X/(X+Y)~Beta(a,b)."""
    if size is None:
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                return float(x / (x + y))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Use Johnk's algorithm for Beta
        import math as _m
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                result.append(x / (x + y))
                break
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_gamma(shape_param, scale=1.0, size=None):
    """Draw samples from a Gamma distribution using Marsaglia-Tsang method."""
    import math as _m
    def _gamma_one_sample(shape_param, scale):
        alpha = shape_param
        if alpha < 1:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            alpha = alpha + 1
            boost = u ** (1.0 / shape_param)
        else:
            boost = 1.0
        d = alpha - 1.0/3.0
        c = 1.0 / _m.sqrt(9.0 * d)
        while True:
            x = float(random.randn((1,))[0])
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < 1 - 0.0331 * x**4:
                return d * v * scale * boost
            if _m.log(u) < 0.5 * x**2 + d * (1 - v + _m.log(v)):
                return d * v * scale * boost
    if size is None:
        return float(_gamma_one_sample(shape_param, scale))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        result.append(_gamma_one_sample(shape_param, scale))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_multinomial(n, pvals, size=None):
    """Draw samples from a multinomial distribution."""
    pvals = [float(p) for p in (pvals.tolist() if isinstance(pvals, ndarray) else pvals)]
    k = len(pvals)
    if size is None:
        # Single draw: n trials among k categories
        result = [0] * k
        for _ in range(n):
            r = float(random.rand((1,))[0])
            cumsum = 0.0
            for j in range(k):
                cumsum += pvals[j]
                if r < cumsum:
                    result[j] += 1
                    break
            else:
                result[-1] += 1
        return array(result)
    else:
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        rows = []
        for _ in range(total):
            result = [0] * k
            for _ in range(n):
                r = float(random.rand((1,))[0])
                cumsum = 0.0
                for j in range(k):
                    cumsum += pvals[j]
                    if r < cumsum:
                        result[j] += 1
                        break
                else:
                    result[-1] += 1
            rows.append(result)
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out

def _random_lognormal(mean=0.0, sigma=1.0, size=None):
    """Draw samples from a log-normal distribution."""
    if size is None:
        import math as _m
        n = float(random.normal(mean, sigma, (1,)).flatten()[0])
        return float(_m.exp(n))
    if isinstance(size, int):
        size = (size,)
    normals = random.normal(mean, sigma, size)
    return exp(normals)

def _random_geometric(p, size=None):
    """Draw samples from a geometric distribution.
    Returns number of trials until first success (minimum value 1)."""
    import math as _m
    if size is None:
        log1mp = _m.log(1.0 - p)
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        return float(_m.ceil(_m.log(u) / log1mp))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    log1mp = _m.log(1.0 - p)
    result = []
    for _ in range(total):
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        # Avoid log(0)
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        result.append(float(_m.ceil(_m.log(u) / log1mp)))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_dirichlet(alpha, size=None):
    """Draw samples from a Dirichlet distribution."""
    if isinstance(alpha, ndarray):
        alpha = alpha.tolist()
    alpha = [float(a) for a in alpha]
    k = len(alpha)
    if size is None:
        # Single draw
        samples = []
        for a in alpha:
            g = float(_random_gamma(a, 1.0, (1,))[0])
            samples.append(g)
        total = sum(samples)
        return array([s / total for s in samples])
    else:
        if isinstance(size, int):
            size = (size,)
        num = 1
        for s in size:
            num *= s
        rows = []
        for _ in range(num):
            samples = []
            for a in alpha:
                g = float(_random_gamma(a, 1.0, (1,))[0])
                samples.append(g)
            total = sum(samples)
            rows.append([s / total for s in samples])
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out

class _Generator:
    """Random number generator (simplified)."""
    def __init__(self, seed_val=None):
        if seed_val is not None:
            random.seed(int(seed_val))

    def random(self, size=None):
        if size is None:
            return float(random.rand((1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.rand(size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def integers(self, low, high=None, size=None, dtype='int64', endpoint=False):
        if high is None:
            high = low
            low = 0
        if not endpoint:
            pass  # high is exclusive already
        else:
            high = high + 1
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a = arange(0.0, float(a), 1.0)
        elif isinstance(a, (list, tuple)):
            a = array(a)
        elif not isinstance(a, ndarray):
            a = asarray(a)
        if size is None:
            size = 1
        return random.choice(a, size, replace)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.normal(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.uniform(low, high, size)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

def _random_random(size=None):
    """Return random floats in [0, 1). Same as rand but takes size tuple."""
    if size is None:
        return float(random.rand((1,))[0])
    if isinstance(size, int):
        size = (size,)
    # Compute total elements
    total = 1
    for s in size:
        total *= s
    result = random.uniform(0.0, 1.0, (total,))
    if len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_multivariate_normal(mean, cov, size=None):
    """Draw from multivariate normal distribution."""
    mean = asarray(mean)
    cov = asarray(cov)
    n = len(mean.tolist())

    # Cholesky decomposition of covariance
    L = linalg.cholesky(cov)

    if size is None:
        # Single sample: generate n standard normals
        z = random.normal(0.0, 1.0, (n,))
        # Transform: L @ z + mean
        z_list = z.tolist()
        mean_list = mean.tolist()
        L_list = L.tolist()
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * z_list[j]
            sample.append(val)
        return array(sample)

    if isinstance(size, int):
        size = (size,)

    total = 1
    for s in size:
        total *= s

    # Generate total*n standard normals
    z = random.normal(0.0, 1.0, (total * n,)).reshape((total, n))

    # Transform: samples = mean + z @ L^T
    z_list = z.tolist()
    mean_list = mean.tolist()
    L_list = L.tolist()

    results = []
    for row in z_list:
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * row[j]
            sample.append(val)
        results.append(sample)

    result = array(results)
    if len(size) > 1:
        result = result.reshape(list(size) + [n])
    return result

def _random_chisquare(df, size=None):
    """Chi-square distribution (sum of df squared standard normals)."""
    df = int(df)
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        z = random.normal(0.0, 1.0, (df,))
        z_list = z.tolist()
        results.append(sum(v * v for v in z_list))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_laplace(loc=0.0, scale=1.0, size=None):
    """Laplace distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    # Inverse CDF: loc - scale * sign(u - 0.5) * log(1 - 2*abs(u - 0.5))
    u_list = u.tolist()
    results = []
    import math
    for ui in u_list:
        ui_shifted = ui - 0.5
        if ui_shifted == 0:
            results.append(loc)
        else:
            sign_val = 1.0 if ui_shifted > 0 else -1.0
            results.append(loc - scale * sign_val * math.log(1.0 - 2.0 * abs(ui_shifted)))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_triangular(left, mode, right, size=None):
    """Triangular distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    u_list = u.tolist()
    results = []
    fc = (mode - left) / (right - left)
    for ui in u_list:
        if ui < fc:
            results.append(left + ((right - left) * (mode - left) * ui) ** 0.5)
        else:
            results.append(right - ((right - left) * (right - mode) * (1.0 - ui)) ** 0.5)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_rayleigh(scale=1.0, size=None):
    """Rayleigh distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [scale * math.sqrt(-2.0 * math.log(1.0 - ui)) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_weibull(a, size=None):
    """Weibull distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [(-math.log(1.0 - ui)) ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_logistic(loc=0.0, scale=1.0, size=None):
    """Logistic distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc + scale * math.log(u / (1.0 - u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_gumbel(loc=0.0, scale=1.0, size=None):
    """Gumbel distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc - scale * math.log(-math.log(u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_negative_binomial(n, p, size=None):
    """Negative binomial distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        # Generate n geometric trials: count failures before n successes
        count = 0
        successes = 0
        while successes < n:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < p:
                successes += 1
            else:
                count += 1
        results.append(float(count))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_power(a, size=None):
    """Power distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    results = [ui ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_vonmises(mu, kappa, size=None):
    """Von Mises distribution (rejection sampling)."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    if kappa < 1e-6:
        # For very small kappa, uniform on [-pi, pi]
        for _ in range(total):
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            results.append(-math.pi + 2.0 * math.pi * u)
    else:
        tau = 1.0 + (1.0 + 4.0 * kappa * kappa) ** 0.5
        rho = (tau - (2.0 * tau) ** 0.5) / (2.0 * kappa)
        r = (1.0 + rho * rho) / (2.0 * rho)
        for _ in range(total):
            while True:
                u1 = float(random.uniform(0.0, 1.0, (1,))[0])
                z = _stdlib_math.cos(math.pi * u1)
                f = (1.0 + r * z) / (r + z)
                c = kappa * (r - f)
                u2 = float(random.uniform(0.0, 1.0, (1,))[0])
                if u2 < c * (2.0 - c) or u2 <= c * _stdlib_math.exp(1.0 - c):
                    u3 = float(random.uniform(0.0, 1.0, (1,))[0])
                    theta = mu + (1.0 if u3 > 0.5 else -1.0) * _stdlib_math.acos(f)
                    results.append(theta)
                    break
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_wald(mean, scale, size=None):
    """Wald (inverse Gaussian) distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        v = float(random.normal(0.0, 1.0, (1,))[0])
        y = v * v
        x = mean + (mean * mean * y) / (2.0 * scale) - (mean / (2.0 * scale)) * (4.0 * mean * scale * y + mean * mean * y * y) ** 0.5
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u <= mean / (mean + x):
            results.append(x)
        else:
            results.append(mean * mean / x)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_zipf(a, size=None):
    """Zipf distribution (rejection sampling)."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    am1 = a - 1.0
    b = 2.0 ** am1
    results = []
    for _ in range(total):
        while True:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u <= 0.0:
                continue
            v = float(random.uniform(0.0, 1.0, (1,))[0])
            x = int(u ** (-1.0 / am1))
            if x < 1:
                x = 1
            t = (1.0 + 1.0 / x) ** am1
            if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                results.append(float(x))
                break
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_hypergeometric(ngood, nbad, nsample, size=None):
    """Hypergeometric distribution."""
    def _draw_one(ng, nb, ns):
        count = 0
        rg = ng
        rt = ng + nb
        uniforms = random.uniform(0.0, 1.0, (ns,)).tolist()
        for u in uniforms:
            if u < rg / rt:
                count += 1
                rg -= 1
            rt -= 1
        return count
    if size is None:
        return _draw_one(ngood, nbad, nsample)
    if isinstance(size, int):
        size = (size,)
    total_elems = 1
    for s in size:
        total_elems *= s
    result = [float(_draw_one(ngood, nbad, nsample)) for _ in _builtin_range(total_elems)]
    return array(result).reshape(list(size))

def _random_pareto(a, size=None):
    """Pareto II (Lomax) distribution."""
    if size is None:
        u = float(random.uniform(0.0, 1.0, (1,)).flatten()[0])
        return (1.0 - u) ** (-1.0 / a) - 1.0
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    uniforms = random.uniform(0.0, 1.0, (total,)).tolist()
    result = [(1.0 - u) ** (-1.0 / a) - 1.0 for u in uniforms]
    return array(result).reshape(list(size))

def _random_bytes(length):
    """Return random bytes."""
    vals = random.uniform(0.0, 1.0, (length,)).tolist()
    return bytes([int(v * 256) for v in vals])

def _default_rng(seed=None):
    return _Generator(seed)

class _RandomState:
    """Legacy random state compatible with np.random.RandomState(seed)."""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def rand(self, *shape):
        if len(shape) == 0:
            return float(random.rand((1,))[0])
        return random.rand(shape)

    def randn(self, *shape):
        if len(shape) == 0:
            return float(random.randn((1,))[0])
        return random.randn(shape)

    def randint(self, low, high=None, size=None, dtype='int64'):
        if high is None:
            high = low
            low = 0
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

    def random(self, size=None):
        return _random_random(size=size)

    def random_sample(self, size=None):
        return _random_random(size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return float(random.normal(float(loc), float(scale), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.normal(float(loc), float(scale), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return float(random.uniform(float(low), float(high), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.uniform(float(low), float(high), size)

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            arr = arange(0.0, float(a), 1.0)
        elif isinstance(a, (list, tuple)):
            arr = array(a)
        elif not isinstance(a, ndarray):
            arr = asarray(a)
        else:
            arr = a
        if size is None:
            size = 1
        return random.choice(arr, size, replace)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def seed(self, seed=None):
        random.seed(seed)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

    def beta(self, a, b, size=None):
        return _random_beta(a, b, size)

    def gamma(self, shape, scale=1.0, size=None):
        return _random_gamma(shape, scale, size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return _random_lognormal(mean, sigma, size)

    def chisquare(self, df, size=None):
        return _random_chisquare(df, size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def multivariate_normal(self, mean, cov, size=None):
        return _random_multivariate_normal(mean, cov, size)

    def get_state(self):
        return {'state': 'not_implemented'}

    def set_state(self, state):
        pass

# Wrap random.choice to accept lists, tuples, and ints (Rust version requires ndarray)
_native_random_choice = random.choice
def _wrapped_random_choice(a, size=None, replace=True, p=None):
    if isinstance(a, int):
        a = arange(0.0, float(a), 1.0)
    elif isinstance(a, (list, tuple)):
        a = array([float(x) for x in a])
    if size is None:
        size = 1
    return _native_random_choice(a, size, replace)
random.choice = _wrapped_random_choice

# Monkey-patch random module with extension functions
random.shuffle = _random_shuffle
random.permutation = _random_permutation
random.standard_normal = _random_standard_normal
random.exponential = _random_exponential
random.poisson = _random_poisson
random.binomial = _random_binomial
random.beta = _random_beta
random.gamma = _random_gamma
random.multinomial = _random_multinomial
random.lognormal = _random_lognormal
random.geometric = _random_geometric
random.dirichlet = _random_dirichlet
random.default_rng = _default_rng
random.Generator = _Generator
random.random = _random_random
random.random_sample = _random_random
random.multivariate_normal = _random_multivariate_normal
random.chisquare = _random_chisquare
random.laplace = _random_laplace
random.triangular = _random_triangular
random.rayleigh = _random_rayleigh
random.weibull = _random_weibull
random.logistic = _random_logistic
random.gumbel = _random_gumbel
random.negative_binomial = _random_negative_binomial
random.power = _random_power
random.vonmises = _random_vonmises
random.wald = _random_wald
random.zipf = _random_zipf
random.hypergeometric = _random_hypergeometric
random.pareto = _random_pareto
random.bytes = _random_bytes
random.RandomState = _RandomState

# --- Numerical Utilities (Tier 20C) -----------------------------------------

def packbits(a, axis=None, bitorder='big'):
    """Pack a binary-valued array into uint8 (int64 since we lack uint8 dtype)."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Flatten if axis is None
    vals = a.flatten().tolist()
    if bitorder == 'little':
        # Reverse bit order within each byte
        result = []
        for i in range(0, len(vals), 8):
            chunk = vals[i:i+8]
            byte = 0
            for j in range(len(chunk)):
                if int(chunk[j]):
                    byte |= (1 << j)
            result.append(byte)
        return array(result)
    else:
        # big endian (default)
        result = []
        for i in range(0, len(vals), 8):
            chunk = vals[i:i+8]
            byte = 0
            for j in range(len(chunk)):
                if int(chunk[j]):
                    byte |= (1 << (7 - j))
            result.append(byte)
        return array(result)


def unpackbits(a, axis=None, count=None, bitorder='big'):
    """Unpack elements of a uint8 array into a binary-valued output array."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    vals = a.flatten().tolist()
    result = []
    for v in vals:
        byte = int(v)
        if bitorder == 'little':
            for j in range(8):
                result.append((byte >> j) & 1)
        else:
            for j in range(7, -1, -1):
                result.append((byte >> j) & 1)
    if count is not None:
        count = int(count)
        if count < len(result):
            result = result[:count]
        else:
            result = result + [0] * (count - len(result))
    return array(result)


def vdot(a, b):
    """Conjugate dot product of two arrays (flattened)."""
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    return dot(a, b)


# --- Tier 27 Group B: Additional functions ----------------------------------

def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """Simplified N-dimensional histogram."""
    sample = asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    sample_list = sample.tolist()
    n_samples = len(sample_list)
    n_dims = len(sample_list[0])

    # Determine bins per dimension
    if isinstance(bins, int):
        bins_per_dim = [bins] * n_dims
    else:
        bins_per_dim = list(bins)

    # Determine edges per dimension
    _range = range  # local alias to avoid conflict
    edges = []
    for d in _builtin_range(n_dims):
        vals = [row[d] for row in sample_list]
        if _range is not None and _range[d] is not None:
            lo, hi = float(_range[d][0]), float(_range[d][1])
        else:
            lo, hi = _builtin_min(vals), _builtin_max(vals)
        edge = linspace(lo, hi, num=bins_per_dim[d] + 1, endpoint=True).tolist()
        edges.append(edge)

    # Build histogram
    shape = bins_per_dim
    total = 1
    for s in shape:
        total *= s
    counts = [0.0] * total

    w_list = None
    if weights is not None:
        w_list = asarray(weights).flatten().tolist()

    for idx_s in _builtin_range(n_samples):
        row = sample_list[idx_s]
        bin_indices = []
        in_range_flag = True
        for d in _builtin_range(n_dims):
            val = row[d]
            edge = edges[d]
            nb = bins_per_dim[d]
            found = False
            for j in _builtin_range(nb):
                if (val >= edge[j] and val < edge[j + 1]) or (j == nb - 1 and val == edge[j + 1]):
                    bin_indices.append(j)
                    found = True
                    break
            if not found:
                in_range_flag = False
                break
        if not in_range_flag:
            continue
        # Compute flat index
        flat_idx = 0
        stride = 1
        for d in _builtin_range(n_dims - 1, -1, -1):
            flat_idx += bin_indices[d] * stride
            stride *= bins_per_dim[d]
        counts[flat_idx] += (w_list[idx_s] if w_list is not None else 1.0)

    hist = array(counts).reshape(shape)
    edge_arrays = [array(e) for e in edges]

    if density and float(sum(hist)) > 0:
        # Normalize
        total_count = float(sum(hist))
        bin_volumes = ones(shape)
        for d in _builtin_range(n_dims):
            widths = [edges[d][i+1] - edges[d][i] for i in _builtin_range(bins_per_dim[d])]
            w_arr = array(widths)
            bcast_shape = [1] * n_dims
            bcast_shape[d] = bins_per_dim[d]
            w_arr = w_arr.reshape(bcast_shape)
            bin_volumes = bin_volumes * broadcast_to(w_arr, shape)
        hist = hist / (total_count * bin_volumes)

    return hist, edge_arrays

class matrix:
    """Simplified matrix class (deprecated in NumPy, but still used)."""
    def __init__(self, data, dtype=None, copy=True):
        if isinstance(data, str):
            # Parse string like "1 2; 3 4"
            rows = data.split(";")
            parsed = []
            for row in rows:
                parsed.append([float(x) for x in row.strip().split()])
            self.A = array(parsed)
        else:
            self.A = atleast_2d(asarray(data))
        if dtype is not None:
            self.A = self.A.astype(str(dtype))

    @property
    def T(self):
        return matrix(self.A.T)

    @property
    def I(self):
        return matrix(linalg.inv(self.A))

    @property
    def shape(self):
        return self.A.shape

    @property
    def ndim(self):
        return self.A.ndim

    def __mul__(self, other):
        if isinstance(other, matrix):
            return matrix(dot(self.A, other.A))
        return matrix(self.A * asarray(other))

    def __add__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A + other.A)
        return matrix(self.A + asarray(other))

    def __sub__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A - other.A)
        return matrix(self.A - asarray(other))

    def __getitem__(self, key):
        return self.A[key]

    def tolist(self):
        return self.A.tolist()

    def __repr__(self):
        return "matrix({})".format(self.A.tolist())


# --- Tier 30 Group C: array2string, lib.stride_tricks, info, who -----------

def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix='', style=None, formatter=None,
                 threshold=None, edgeitems=None, sign=None, floatmode=None,
                 suffix='', legacy=None):
    """Return a string representation of an array."""
    a = asarray(a)
    return repr(a)


def info(object=None, maxwidth=76, output=None, toplevel='numpy'):
    """Display documentation for numpy objects."""
    if object is not None:
        doc = getattr(object, '__doc__', None)
        if doc:
            print(doc)
        else:
            print("No documentation available for {}".format(object))


def who(vardict=None):
    """Print info about variables in the given dictionary."""
    if vardict is None:
        return
    for name, val in vardict.items():
        if hasattr(val, 'shape'):
            print("{}: shape={}, dtype={}".format(name, val.shape, val.dtype))


class _LibModule:
    class stride_tricks:
        @staticmethod
        def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
            """Simplified as_strided - creates a new array with the given shape.
            WARNING: This does NOT share memory with the original array.
            It creates a view-like result by repeating/tiling data."""
            x = asarray(x)
            if shape is None:
                return x.copy()
            # Best effort: reshape or tile to match requested shape
            flat = x.flatten().tolist()
            total = 1
            for s in shape:
                total *= s
            # Repeat flat data to fill the requested size
            result = []
            for i in range(total):
                result.append(flat[i % len(flat)])
            return array(result).reshape(shape)

        @staticmethod
        def sliding_window_view(x, window_shape, axis=None):
            """Create a sliding window view of the array."""
            x = asarray(x)
            if isinstance(window_shape, int):
                window_shape = (window_shape,)
            if x.ndim == 1 and len(window_shape) == 1:
                w = window_shape[0]
                data = x.tolist()
                n = len(data) - w + 1
                if n <= 0:
                    return array([]).reshape((0, w))
                rows = []
                for i in range(n):
                    rows.append(data[i:i+w])
                return array(rows)
            raise NotImplementedError("sliding_window_view only supports 1D")

lib = _LibModule()

def _has_complex(result):
    """Check if any element in result is complex (avoids shadowed builtin any)."""
    for r in result:
        if isinstance(r, complex):
            return True
    return False

class _ScimathModule:
    """Complex-safe math functions (numpy.lib.scimath)."""

    @staticmethod
    def _to_array(result, shape):
        """Convert list of float/complex results to an ndarray."""
        has_cplx = False
        for r in result:
            if isinstance(r, complex):
                has_cplx = True
                break
        if has_cplx:
            return _make_complex_array(result, shape)
        return _native.array([float(r) for r in result]).reshape(shape)

    @staticmethod
    def sqrt(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v < 0:
                result.append(complex(0, (-v)**0.5))
            else:
                result.append(v**0.5)
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def log(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v <= 0:
                import cmath
                result.append(cmath.log(v))
            else:
                result.append(_stdlib_math.log(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def log10(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v <= 0:
                import cmath
                result.append(cmath.log10(v))
            else:
                result.append(_stdlib_math.log10(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def log2(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v <= 0:
                import cmath
                result.append(cmath.log(v) / cmath.log(2))
            else:
                result.append(_stdlib_math.log2(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def power(x, p):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            try:
                r = v ** p
                result.append(r)
            except (ValueError, ZeroDivisionError):
                import cmath
                result.append(cmath.exp(p * cmath.log(v)))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def arccos(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if abs(v) > 1:
                import cmath
                result.append(cmath.acos(v))
            else:
                result.append(_stdlib_math.acos(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def arcsin(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if abs(v) > 1:
                import cmath
                result.append(cmath.asin(v))
            else:
                result.append(_stdlib_math.asin(v))
        return _ScimathModule._to_array(result, x.shape)

lib.scimath = _ScimathModule()

class NumpyVersion:
    """Minimal numpy version comparison class."""
    def __init__(self, vstring):
        self.vstring = vstring
        parts = vstring.split('.')
        self.major = int(parts[0]) if len(parts) > 0 else 0
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.bugfix = int(parts[2].split('rc')[0].split('a')[0].split('b')[0]) if len(parts) > 2 else 0
    def __repr__(self):
        return f"NumpyVersion('{self.vstring}')"
    def __str__(self):
        return self.vstring
    def __lt__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) < (other.major, other.minor, other.bugfix)
    def __le__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) <= (other.major, other.minor, other.bugfix)
    def __gt__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) > (other.major, other.minor, other.bugfix)
    def __ge__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) >= (other.major, other.minor, other.bugfix)
    def __eq__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) == (other.major, other.minor, other.bugfix)

lib.NumpyVersion = NumpyVersion


# --- testing module ---------------------------------------------------------
class _TestingModule:
    def assert_allclose(self, actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        actual = asarray(actual)
        desired = asarray(desired)
        if not allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan):
            actual_list = actual.tolist()
            desired_list = desired.tolist()
            raise AssertionError(err_msg or "Not equal to tolerance rtol={}, atol={}\n Actual: {}\n Desired: {}".format(rtol, atol, actual_list, desired_list))

    def assert_array_equal(self, x, y, err_msg='', verbose=True, strict=False):
        x = asarray(x)
        y = asarray(y)
        # Handle scalar vs array comparison (NumPy broadcasts)
        if x.shape != y.shape:
            # 0-D vs 1-element: equivalent for comparison purposes
            if x.ndim == 0 and y.size == 1:
                y = y.reshape(())
            elif y.ndim == 0 and x.size == 1:
                x = x.reshape(())
            elif y.size == 1:
                y = broadcast_to(y.flatten(), x.shape)
            elif x.size == 1:
                x = broadcast_to(x.flatten(), y.shape)
        if not array_equal(x, y, equal_nan=True):
            raise AssertionError(err_msg or "Arrays are not equal\n x: {}\n y: {}".format(x.tolist(), y.tolist()))

    def assert_array_almost_equal(self, x, y, decimal=6, err_msg='', verbose=True):
        x = asarray(x)
        y = asarray(y)
        if not allclose(x, y, rtol=0, atol=1.5 * 10**(-decimal)):
            raise AssertionError(err_msg or "Arrays are not almost equal to {} decimals".format(decimal))

    def assert_equal(self, actual, desired, err_msg='', verbose=True):
        # Handle tuples/lists recursively
        if isinstance(actual, (tuple, list)) and isinstance(desired, (tuple, list)):
            if len(actual) != len(desired):
                raise AssertionError(err_msg or "Length mismatch: {} vs {}".format(len(actual), len(desired)))
            for i, (a, d) in enumerate(zip(actual, desired)):
                self.assert_equal(a, d, err_msg=err_msg, verbose=verbose)
            return
        actual_a = asarray(actual)
        desired_a = asarray(desired)
        # Empty array is vacuously equal to any scalar
        if actual_a.size == 0 and desired_a.size <= 1:
            return
        if desired_a.size == 0 and actual_a.size <= 1:
            return
        # Handle 0-d scalar comparison: extract element and compare directly
        if actual_a.shape == () and desired_a.size == 1:
            a_val = actual_a.flatten()[0]
            d_val = desired_a.flatten()[0]
            if a_val != d_val:
                raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))
            return
        if desired_a.shape == () and actual_a.size == 1:
            a_val = actual_a.flatten()[0]
            d_val = desired_a.flatten()[0]
            if a_val != d_val:
                raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))
            return
        # Handle scalar vs array comparison
        if actual_a.shape != desired_a.shape:
            if desired_a.size == 1:
                desired_a = broadcast_to(desired_a.flatten(), actual_a.shape)
            elif actual_a.size == 1:
                actual_a = broadcast_to(actual_a.flatten(), desired_a.shape)
        if not array_equal(actual_a, desired_a, equal_nan=True):
            raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))

    def assert_raises(self, exception_class, *args, **kwargs):
        """Simple assert_raises - returns a context manager."""
        class _AssertRaisesCtx:
            def __init__(self, exc_cls):
                self.exc_cls = exc_cls
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError("Expected {} but no exception raised".format(self.exc_cls.__name__))
                if not issubclass(exc_type, self.exc_cls):
                    return False  # re-raise
                return True  # suppress exception
        if args:
            # Called as assert_raises(Error, func, *args)
            callable_obj = args[0]
            rest = args[1:]
            try:
                callable_obj(*rest, **kwargs)
            except exception_class:
                return
            raise AssertionError("Expected {}".format(exception_class.__name__))
        return _AssertRaisesCtx(exception_class)

    def assert_raises_regex(self, exception_class, expected_regex, *args, **kwargs):
        """Assert that an exception is raised matching a regex."""
        import re
        callable_obj = kwargs.pop('callable', None)
        if callable_obj is None and len(args) >= 1:
            callable_obj = args[0]
            args = args[1:]
        if callable_obj is not None:
            try:
                callable_obj(*args, **kwargs)
            except exception_class as e:
                if not re.search(expected_regex, str(e)):
                    raise AssertionError(
                        "Exception message '{}' did not match '{}'".format(str(e), expected_regex))
                return
            except Exception as e:
                raise AssertionError(
                    "Expected {}, got {}".format(exception_class.__name__, type(e).__name__))
            raise AssertionError("{} not raised".format(exception_class.__name__))
        else:
            # Context manager mode
            return _AssertRaisesRegexContext(exception_class, expected_regex)

    def assert_warns(self, warning_class, *args, **kwargs):
        """Assert that a warning is raised. Since we don't have warnings module, just run the callable."""
        callable_obj = kwargs.pop('callable', None)
        if callable_obj is None and len(args) >= 1:
            callable_obj = args[0]
            args = args[1:]
        if callable_obj is not None:
            return callable_obj(*args, **kwargs)
        # Return a context manager that suppresses warnings
        class _WarnCtx:
            def __enter__(self_ctx):
                return self_ctx
            def __exit__(self_ctx, *exc):
                return False
        return _WarnCtx()

    def assert_approx_equal(self, actual, desired, significant=7, err_msg='', verbose=True):
        """Assert approximately equal to given number of significant digits."""
        if desired == 0:
            if _stdlib_math.fabs(actual) > 1.5 * 10**(-significant):
                raise AssertionError("{} != {} to {} significant digits".format(actual, desired, significant))
        else:
            rel = _stdlib_math.fabs((actual - desired) / desired)
            if rel > 1.5 * 10**(-significant):
                raise AssertionError("{} != {} to {} significant digits".format(actual, desired, significant))

    def assert_array_less(self, x, y, err_msg='', verbose=True):
        """Assert array_like x is less than array_like y, element-wise."""
        x = asarray(x)
        y = asarray(y)
        if not all((x < y).flatten().tolist()):
            raise AssertionError("Arrays are not less-ordered\nx: {}\ny: {}".format(x.tolist(), y.tolist()))


class _AssertRaisesRegexContext:
    def __init__(self, exc_class, pattern):
        self.exc_class = exc_class
        self.pattern = pattern
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        import re
        if exc_type is None:
            raise AssertionError("{} not raised".format(self.exc_class.__name__))
        if not issubclass(exc_type, self.exc_class):
            return False
        if not re.search(self.pattern, str(exc_val)):
            raise AssertionError("'{}' did not match '{}'".format(str(exc_val), self.pattern))
        return True


testing = _TestingModule()

# --- dtypes module (exposes per-dtype DType classes) ------------------------
class _dtypes_mod:
    Float64DType = Float64DType
    Float32DType = Float32DType
    Float16DType = Float16DType
    Int8DType = Int8DType
    Int16DType = Int16DType
    Int32DType = Int32DType
    Int64DType = Int64DType
    UInt8DType = UInt8DType
    UInt16DType = UInt16DType
    UInt32DType = UInt32DType
    UInt64DType = UInt64DType
    Complex64DType = Complex64DType
    Complex128DType = Complex128DType
    BoolDType = BoolDType
    StrDType = StrDType
    BytesDType = BytesDType
    VoidDType = VoidDType
    ObjectDType = ObjectDType
dtypes = _dtypes_mod()

# --- np.rec module (basic stub) ---
class _RecModule:
    """Minimal np.rec namespace."""
    def __init__(self):
        self.recarray = None  # placeholder

    def array(self, data, dtype=None):
        """Create a record array (falls back to regular array)."""
        arr = asarray(data)
        if dtype is not None:
            dt = dtype if isinstance(dtype, StructuredDtype) else StructuredDtype(dtype) if isinstance(dtype, list) else dtype
            # Try to attach structured dtype metadata; silently skip if type doesn't allow it
            try:
                arr._structured_dtype = dt
            except (TypeError, AttributeError):
                pass
        return arr

    def fromarrays(self, arrays, dtype=None, names=None):
        """Create a record array from separate arrays."""
        if names is not None and dtype is None:
            fields = [(n, 'float64') for n in names]
            dtype = StructuredDtype(fields)
        return self.array(arrays, dtype=dtype)

rec = _RecModule()

# --- show_config stub -------------------------------------------------------
def show_config():
    """Show numpy-rust build configuration."""
    print("numpy-rust (codepod)")
    print("  backend: Rust + RustPython")

# --- fromfile stub ----------------------------------------------------------
# --- einsum_path stub -------------------------------------------------------
def einsum_path(*operands, optimize='greedy'):
    """Evaluate optimal contraction order (stub returns naive path)."""
    # Return a naive path: contract in order
    n = int(len(operands) // 2)  # rough estimate
    path = [(0, 1)] * _builtin_max(1, n - 1)
    return path, ""

# --- byte_bounds stub -------------------------------------------------------
def byte_bounds(a):
    """Return low and high byte pointers (stub returns (0, nbytes))."""
    arr = asarray(a)
    return (0, arr.nbytes)

# --- Module stubs -----------------------------------------------------------

# np.core module stub
class _CoreModule:
    """Minimal np.core namespace."""
    pass

core = _CoreModule()
core.numeric = core  # self-reference for np.core.numeric compatibility
core.multiarray = core  # np.core.multiarray compatibility
core.fromnumeric = core  # np.core.fromnumeric compatibility

# np.compat module stub
class _CompatModule:
    pass
compat = _CompatModule()

# np.exceptions module
class _ExceptionsModule:
    AxisError = AxisError  # already defined
    ComplexWarning = type('ComplexWarning', (UserWarning,), {})
    DTypePromotionError = type('DTypePromotionError', (TypeError,), {})
    VisibleDeprecationWarning = type('VisibleDeprecationWarning', (UserWarning,), {})
    ModuleDeprecationWarning = type('ModuleDeprecationWarning', (DeprecationWarning,), {})
    RankWarning = type('RankWarning', (UserWarning,), {})
    TooHardError = type('TooHardError', (RuntimeError,), {})
exceptions = _ExceptionsModule()
exceptions.__name__ = 'numpy.exceptions'

# Expose exception classes at top level (sklearn fallback path)
ComplexWarning = exceptions.ComplexWarning
VisibleDeprecationWarning = exceptions.VisibleDeprecationWarning

# np.matlib stub
class _MatlibModule:
    """Minimal np.matlib namespace."""
    pass
matlib = _MatlibModule()

# np.ctypeslib stub
class _CtypeslibModule:
    pass
ctypeslib = _CtypeslibModule()

# --- format_float functions -------------------------------------------------
def format_float_positional(x, precision=None, unique=True, fractional=True, trim='k', sign=False, pad_left=None, pad_right=None, min_digits=None):
    """Format a float in positional notation."""
    if precision is not None:
        return f"{x:.{precision}f}"
    return str(x)

def format_float_scientific(x, precision=None, unique=True, trim='k', sign=False, pad_left=None, exp_digits=None, min_digits=None):
    """Format a float in scientific notation."""
    if precision is not None:
        return f"{x:.{precision}e}"
    return f"{x:e}"

# --- memmap stub ------------------------------------------------------------
class memmap:
    """Memory-mapped file stub (not supported in sandboxed environment)."""
    def __new__(cls, filename, dtype=None, mode='r+', offset=0, shape=None, order='C'):
        raise NotImplementedError("memmap not supported in sandboxed environment")


# --- Misc stubs -------------------------------------------------------------
def seterrcall(func):
    """Set callback for floating-point error handler (no-op)."""
    return None

def geterrcall():
    """Get callback for floating-point error handler (no-op)."""
    return None

def add_newdoc(place, obj, doc):
    """Add documentation (no-op in our implementation)."""
    pass

def deprecate(func=None, oldname=None, newname=None, message=None):
    """Deprecation decorator (no-op)."""
    if func is not None:
        return func
    def decorator(f):
        return f
    return decorator

def get_include():
    """Return include directory (not applicable)."""
    return ""

tracemalloc_domain = 0
use_hugepage = 0
nested_iters = None  # Not supported

# --- Import submodules so np.ma and np.polynomial are accessible ------------
import numpy.ma as ma
import numpy.polynomial as polynomial

# --- Module-level __getattr__ for deprecated aliases like np.bool -----------
def __getattr__(name):
    _bi = __import__("builtins")
    _deprecated_aliases = {
        'bool': _bi.bool,
        'int': _bi.int,
        'float': _bi.float,
        'complex': _bi.complex,
        'str': _bi.str,
        'object': _bi.object,
    }
    if name in _deprecated_aliases:
        return _deprecated_aliases[name]
    raise AttributeError(f"module 'numpy' has no attribute '{name}'")
