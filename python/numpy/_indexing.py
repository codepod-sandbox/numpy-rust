"""Index generation, iteration, histograms."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import AxisError, _ObjectArray, _builtin_range, _builtin_min, _builtin_max
from ._core_types import _normalize_dtype
from ._creation import array, asarray, zeros, ones, arange, linspace, empty
from ._math import floor as _floor, isnan
from ._reductions import sum, diff
from ._manipulation import reshape, concatenate, moveaxis, tile, stack, broadcast_to, atleast_2d

__all__ = [
    'diagonal', 'trace', 'meshgrid', 'indices', 'matrix_transpose',
    'histogram_bin_edges', 'histogram', 'histogram2d', 'histogramdd',
    'digitize', 'unravel_index', 'ravel_multi_index',
    'diag', 'tri', 'tril', 'triu',
    'fill_diagonal', 'diag_indices', 'diag_indices_from',
    'tril_indices', 'triu_indices', 'tril_indices_from', 'triu_indices_from',
    'ndenumerate', 'ndindex', 'nditer',
    'ix_', 'mgrid', 'ogrid', 'r_', 'c_', 's_', 'index_exp',
    '_MGrid', '_OGrid', '_RClass', '_CClass', '_SClass',
    'advanced_fancy_index',
]


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
    flat = _flat_values(a)
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


def meshgrid(*xi, indexing='xy', sparse=False, copy=True):
    from ._shape import broadcast_to
    if indexing not in ('xy', 'ij'):
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    arrays = [a if isinstance(a, ndarray) else array(a) for a in xi]
    if not arrays:
        return []
    ndim = len(arrays)
    if sparse:
        # Return sparse (open) meshgrids: each output has shape with 1s except along its own axis
        sparse_result = []
        for i, a in enumerate(arrays):
            shape = [1] * ndim
            if indexing == 'xy' and ndim >= 2:
                if i == 0:
                    shape[1] = len(a.flatten())
                elif i == 1:
                    shape[0] = len(a.flatten())
                else:
                    shape[i] = len(a.flatten())
            else:
                shape[i] = len(a.flatten())
            sparse_result.append(a.flatten().reshape(shape))
        if copy:
            sparse_result = [s.copy() for s in sparse_result]
        return sparse_result
    dense_shape = []
    for i, a in enumerate(arrays):
        axis = i
        if indexing == 'xy' and ndim >= 2:
            if i == 0:
                axis = 1
            elif i == 1:
                axis = 0
        while len(dense_shape) <= axis:
            dense_shape.append(1)
        dense_shape[axis] = len(a.flatten())
    while len(dense_shape) < ndim:
        dense_shape.append(1)
    result = []
    for i, a in enumerate(arrays):
        axis = i
        if indexing == 'xy' and ndim >= 2:
            if i == 0:
                axis = 1
            elif i == 1:
                axis = 0
        shape = [1] * ndim
        shape[axis] = len(a.flatten())
        expanded = a.flatten().reshape(shape)
        result.append(broadcast_to(expanded, tuple(dense_shape)))
    if copy:
        result = [r.copy() for r in result]
    return result


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


def matrix_transpose(a):
    a = asarray(a) if not isinstance(a, ndarray) else a
    return a.T


def _histogram_range_from_flat(flat, range):
    """Resolve histogram low/high bounds from explicit range or data."""
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
    elif flat:
        lo, hi = _builtin_min(flat), _builtin_max(flat)
    else:
        lo, hi = 0.0, 1.0
    if not (_math.isfinite(lo) and _math.isfinite(hi)):
        raise ValueError("autodetected range of [{}, {}] is not finite".format(lo, hi))
    if lo == hi:
        lo = lo - 0.5
        hi = hi + 0.5
    return lo, hi


def _coerce_histogram_edges(bins):
    """Normalize explicit 1-D histogram edges."""
    edges = asarray(bins)
    if edges.ndim != 1:
        raise ValueError("bins must be 1d")
    edges = edges.flatten()
    edge_list = edges.tolist()
    n_bins = len(edge_list) - 1
    if n_bins < 1:
        raise ValueError("bins must have at least 2 edges")
    for edge in edge_list:
        if getattr(edge, '_is_nat', False):
            raise ValueError("bins must not contain NaN")
        if isinstance(edge, (int, float)) and _math.isnan(edge):
            raise ValueError("bins must not contain NaN")
    for i in _builtin_range(len(edge_list) - 1):
        if not (edge_list[i] < edge_list[i + 1]):
            raise ValueError("`bins` must increase monotonically, when an array")
    return edges, edge_list, n_bins


def _histogram_count_dtype(weights, density):
    if density:
        return 'float64'
    if weights is None:
        return 'int64'
    w = asarray(weights)
    kind = getattr(w.dtype, 'kind', '')
    if kind == 'f':
        return 'float64'
    if kind == 'c':
        return 'complex128'
    if kind in ('i', 'u', 'b'):
        return 'int64'
    return 'object'


def _coerce_histogram_weight_value(value, dtype_name):
    if dtype_name == 'complex128' and isinstance(value, tuple) and len(value) == 2:
        return complex(value[0], value[1])
    return value


def _histogram_accumulate_counts(flat, edge_list, count_dtype, w_list=None, *, lo=None, hi=None):
    n_bins = len(edge_list) - 1
    if count_dtype == 'float64':
        counts = [0.0] * n_bins
    elif count_dtype == 'complex128':
        counts = [0j] * n_bins
    else:
        counts = [0] * n_bins
    for idx_val, val in enumerate(flat):
        if lo is not None and hi is not None and (val < lo or val > hi):
            continue
        for j in _builtin_range(n_bins):
            is_last = (j == n_bins - 1)
            in_bin = (edge_list[j] <= val <= edge_list[j + 1]) if is_last else (edge_list[j] <= val < edge_list[j + 1])
            if in_bin:
                counts[j] += (_coerce_histogram_weight_value(w_list[idx_val], count_dtype) if w_list is not None else 1.0)
                break
    return counts


def _histogramdd_flat_index(bin_indices, bins_per_dim):
    flat_idx = 0
    stride = 1
    for d in _builtin_range(len(bins_per_dim) - 1, -1, -1):
        flat_idx += bin_indices[d] * stride
        stride *= bins_per_dim[d]
    return flat_idx


def _histogramdd_row_bin_indices(row, edges, bins_per_dim):
    bin_indices = []
    for d in _builtin_range(len(bins_per_dim)):
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
            return None
    return bin_indices


def _flat_int_index_values(arr):
    if hasattr(arr, 'flatten'):
        return [int(v) for v in arr.flatten().tolist()]
    return [int(arr)]


def _flat_values(arr):
    return arr.flatten().tolist() if hasattr(arr, 'flatten') else [arr]


def _flat_weight_values(weights):
    if weights is None:
        return None
    return _flat_values(asarray(weights))


def _normalize_histogramdd_bins(bins, n_dims):
    """Normalize histogramdd-style bins into one spec per dimension."""
    if isinstance(bins, int):
        return [bins] * n_dims
    if isinstance(bins, (list, tuple)):
        if len(bins) == n_dims:
            return list(bins)
        return [bins] * n_dims
    return [int(bins)] * n_dims


def _resolve_histogramdd_edges(bin_spec, values, range_spec):
    """Resolve one histogramdd dimension bin spec to explicit edges."""
    if isinstance(bin_spec, ndarray):
        edge = _flat_values(bin_spec)
        return edge, len(edge) - 1
    if isinstance(bin_spec, (list, tuple)):
        edge = [float(v) for v in bin_spec]
        return edge, len(edge) - 1

    nb = int(bin_spec)
    if range_spec is not None:
        lo, hi = float(range_spec[0]), float(range_spec[1])
    elif values:
        lo, hi = _builtin_min(values), _builtin_max(values)
    else:
        lo, hi = 0.0, 1.0
    edge = linspace(lo, hi, num=nb + 1, endpoint=True).tolist()
    return edge, nb


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Compute the bin edges for a histogram without computing the histogram itself."""
    a = asarray(a)
    if isinstance(bins, str):
        return _histogram_bin_edges_from_method(a, bins, range=range)
    if isinstance(bins, int):
        flat = _flat_values(a)
        lo, hi = _histogram_range_from_flat(flat, range)
        edges = linspace(lo, hi, bins + 1)
        return edges
    else:
        return asarray(bins)


def _histogram_bin_edges_from_method(a, method, range=None):
    """Compute bin edges using an automatic bin-width method (string name)."""
    flat = _flat_values(a)
    n = len(flat)
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
        # Filter data to range
        vals = [float(v) for v in flat if lo <= float(v) <= hi]
        n_eff = len(vals)
    else:
        vals = [float(v) for v in flat]
        n_eff = n
        if n_eff == 0:
            lo, hi = 0.0, 1.0
        else:
            if any(not _math.isfinite(v) for v in vals):
                raise ValueError("autodetected range of [{}, {}] is not finite".format(_builtin_min(vals), _builtin_max(vals)))
            lo = _builtin_min(vals)
            hi = _builtin_max(vals)
    if not (_math.isfinite(lo) and _math.isfinite(hi)):
        raise ValueError("autodetected range of [{}, {}] is not finite".format(lo, hi))
    if lo == hi:
        lo = lo - 0.5
        hi = hi + 0.5
    if n_eff <= 1:
        return linspace(lo, hi, 2, endpoint=True)
    # Compute std
    mean = __import__("builtins").sum(vals) / n_eff
    var = __import__("builtins").sum((v - mean) ** 2 for v in vals) / n_eff
    std = var ** 0.5
    data_range = hi - lo
    def _sturges():
        return data_range / (_math.log2(n_eff) + 1) if n_eff > 0 else 1.0
    def _rice():
        return data_range / (2.0 * n_eff ** (1.0/3.0)) if n_eff > 0 else 1.0
    def _scott():
        return (24.0 * 3.14159265 * var / n_eff) ** 0.5 if std > 0 else 1.0
    def _fd():
        # Freedman-Diaconis: 2 * IQR / n^(1/3)
        sorted_v = sorted(vals)
        q25 = sorted_v[int(0.25 * n_eff)]
        q75 = sorted_v[int(0.75 * n_eff) if int(0.75 * n_eff) < n_eff else n_eff - 1]
        iqr = q75 - q25
        if iqr == 0:
            return _sturges()
        return 2.0 * iqr / (n_eff ** (1.0/3.0))
    def _sqrt():
        return data_range / (n_eff ** 0.5) if n_eff > 0 else 1.0
    def _doane():
        if std == 0:
            return 1.0
        sg1 = __import__("builtins").sum((v - mean) ** 3 for v in vals) / (n_eff * std ** 3)
        sigma_g1 = (6.0 * (n_eff - 2) / ((n_eff + 1) * (n_eff + 3))) ** 0.5
        k = 1.0 + _math.log2(n_eff) + _math.log2(1.0 + __import__("builtins").abs(sg1) / sigma_g1) if sigma_g1 > 0 else _math.log2(n_eff) + 1
        return data_range / k if k > 0 else 1.0
    _methods = {
        'sturges': _sturges,
        'rice': _rice,
        'scott': _scott,
        'fd': _fd,
        'sqrt': _sqrt,
        'doane': _doane,
    }
    if method == 'auto':
        # auto = max(sturges, fd) in terms of bin count
        w_sturges = _sturges()
        w_fd = _fd()
        # Smaller width = more bins; pick the one with more bins (smaller width)
        width = _builtin_min(w_sturges, w_fd) if w_sturges > 0 and w_fd > 0 else (w_sturges if w_sturges > 0 else w_fd)
    elif method == 'stone':
        # Stone's method: try different bin counts, pick optimal via AIC-like
        # Simplified: use Sturges as fallback
        width = _sturges()
    elif method in _methods:
        width = _methods[method]()
    else:
        raise ValueError("Invalid bin estimation method: {}".format(method))
    if width <= 0:
        width = 1.0
    nbins = max(1, int(_math.ceil(data_range / width)))
    return linspace(lo, hi, nbins + 1, endpoint=True)

def histogram(a, bins=10, range=None, density=None, weights=None):
    import warnings as _warnings
    if not isinstance(a, ndarray):
        a = array(a)
    if getattr(a.dtype, 'kind', '') == 'b':
        _warnings.warn(
            "Converting input from bool to <class 'numpy.uint8'> for compatibility.",
            RuntimeWarning,
            stacklevel=2,
        )
        a = a.astype('uint8')
    # Validate weights
    if weights is not None:
        w = asarray(weights)
        if w.shape != a.shape:
            raise ValueError("weights should have the same shape as a")
    # Handle string bin methods
    if isinstance(bins, str):
        edges = _histogram_bin_edges_from_method(a, bins, range=range)
        return histogram(a, bins=edges, range=range, density=density, weights=weights)
    if isinstance(bins, (list, tuple, ndarray, _ObjectArray)):
        edges, edge_list, n_bins = _coerce_histogram_edges(bins)
        flat = _flat_values(a)
        count_dtype = _histogram_count_dtype(weights, density)
        w_list = _flat_weight_values(weights)
        counts = _histogram_accumulate_counts(flat, edge_list, count_dtype, w_list=w_list)
        counts_arr = array(counts, dtype=count_dtype)
        if density:
            bin_widths = diff(edges)
            total = float(sum(counts_arr))
            if total > 0.0:
                counts_arr = counts_arr / (total * bin_widths)
        return counts_arr, edges
    # bins is an int
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
        if not (_math.isfinite(lo) and _math.isfinite(hi)):
            raise ValueError("supplied range of [{}, {}] is not finite".format(lo, hi))
        if lo > hi:
            raise ValueError("max must be larger than min in range parameter")
        if lo == hi:
            lo = lo - 0.5
            hi = hi + 0.5
    if range is None:
        lo, hi = _histogram_range_from_flat(_flat_values(a), None)
    edge_preview = linspace(lo, hi, num=bins + 1, endpoint=True).tolist()
    for i in _builtin_range(len(edge_preview) - 1):
        if not (edge_preview[i] < edge_preview[i + 1]):
            raise ValueError("Too many bins for data range")
    if weights is not None or range is not None:
        # Python fallback for weights/range with integer bins
        flat = _flat_values(a)
        if range is None:
            lo, hi = _histogram_range_from_flat(flat, None)
        edges = linspace(lo, hi, num=bins + 1, endpoint=True)
        edge_list = edges.tolist()
        count_dtype = _histogram_count_dtype(weights, density)
        w_list = _flat_weight_values(weights)
        counts = _histogram_accumulate_counts(flat, edge_list, count_dtype, w_list=w_list, lo=lo, hi=hi)
        hist = array(counts, dtype=count_dtype)
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
    """Compute the 2D histogram of two data samples.

    Delegates to histogramdd for the actual computation.
    """
    import numpy as _np
    x = _np.asarray(x)
    y = _np.asarray(y)

    if x.size != y.size:
        raise ValueError('x and y must have the same length.')

    # Build bins for histogramdd
    if isinstance(bins, (list, tuple)):
        if len(bins) != 2:
            raise ValueError("bins must be a sequence of 1 or 2 elements")
        dd_bins = [bins[0], bins[1]]
    else:
        dd_bins = [bins, bins]

    # Stack x, y into (N, 2) sample
    sample = _np.column_stack([x.flatten(), y.flatten()])

    hist, edges = _np.histogramdd(sample, bins=dd_bins, range=range,
                                  density=density, weights=weights)
    return hist, edges[0], edges[1]


def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """Compute the multidimensional histogram of some data."""
    if isinstance(sample, (list, tuple)) and sample:
        first = sample[0]
        if isinstance(first, (list, tuple, ndarray, _ObjectArray)):
            cols = [asarray(col).flatten() for col in sample]
            lengths = [col.size for col in cols]
            if lengths and all(n == lengths[0] for n in lengths):
                rows = []
                for i in _builtin_range(lengths[0]):
                    rows.append([col[i] for col in cols])
                sample = array(rows)
    sample = asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    n_samples, n_dims = sample.shape[0], sample.shape[1]
    sample_list = sample.tolist()

    _range = range

    # Determine bins per dimension - each can be int or array-like edge list
    bins_spec = _normalize_histogramdd_bins(bins, n_dims)

    # Build edges for each dimension
    edges = []
    bins_per_dim = []
    for d in _builtin_range(n_dims):
        values = [row[d] for row in sample_list] if n_samples > 0 else []
        range_spec = _range[d] if _range is not None and _range[d] is not None else None
        edge, nb = _resolve_histogramdd_edges(bins_spec[d], values, range_spec)
        edges.append(edge)
        bins_per_dim.append(nb)

    # Build histogram
    shape = bins_per_dim
    total = 1
    for s in shape:
        total *= s
        if total > 2**31:
            raise ValueError(
                "Too many bins: total number of bins exceeds maximum"
            )
    counts = [0.0] * total

    w_list = None
    if weights is not None:
        w_list = _flat_weight_values(weights)

    for idx_s in _builtin_range(n_samples):
        row = sample_list[idx_s]
        bin_indices = _histogramdd_row_bin_indices(row, edges, bins_per_dim)
        if bin_indices is None:
            continue
        flat_idx = _histogramdd_flat_index(bin_indices, bins_per_dim)
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


def digitize(x, bins, right=False):
    """Return the indices of the bins to which each value belongs."""
    from numpy._reductions import searchsorted
    from numpy._math import iscomplexobj

    def _contains_unsafe_python_ints(value):
        if type(value) is int:
            return abs(value) > (1 << 53)
        if isinstance(value, (list, tuple)):
            return any(_contains_unsafe_python_ints(item) for item in value)
        return False

    def _is_plain_int_sequence(value):
        return isinstance(value, (list, tuple)) and all(type(item) is int for item in value)

    def _digitize_exact_python_ints(values, edges):
        n_edges = len(edges)
        if n_edges == 0:
            return [] if isinstance(values, (list, tuple)) else 0
        increasing = all(edges[i] <= edges[i + 1] for i in range(n_edges - 1))
        decreasing = all(edges[i] >= edges[i + 1] for i in range(n_edges - 1))
        if not increasing and not decreasing:
            raise ValueError("bins must be monotonically increasing or decreasing")

        def _search_one(v):
            if increasing:
                if right:
                    lo, hi = 0, n_edges
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if v <= edges[mid]:
                            hi = mid
                        else:
                            lo = mid + 1
                    return lo
                lo, hi = 0, n_edges
                while lo < hi:
                    mid = (lo + hi) // 2
                    if v < edges[mid]:
                        hi = mid
                    else:
                        lo = mid + 1
                return lo

            lo, hi = 0, n_edges
            while lo < hi:
                mid = (lo + hi) // 2
                edge = edges[n_edges - 1 - mid]
                if right:
                    if v <= edge:
                        hi = mid
                    else:
                        lo = mid + 1
                else:
                    if v < edge:
                        hi = mid
                    else:
                        lo = mid + 1
            return n_edges - lo

        if type(values) is int:
            return _search_one(values)
        return array([_search_one(v) for v in values], dtype='int64')

    if (
        _is_plain_int_sequence(bins)
        and _contains_unsafe_python_ints(bins)
        and (type(x) is int or _is_plain_int_sequence(x))
    ):
        return _digitize_exact_python_ints(x, bins)

    x = asarray(x)
    bins = asarray(bins)
    if iscomplexobj(x) or iscomplexobj(bins):
        raise TypeError("x and bins must be real")
    n = len(bins)
    if n == 0:
        return zeros(x.shape, dtype='int64')
    # Validate monotonicity
    if n > 1:
        bins_list = [float(bins[i]) for i in range(n)]
        increasing = all(bins_list[i] <= bins_list[i+1] for i in range(n-1))
        decreasing = all(bins_list[i] >= bins_list[i+1] for i in range(n-1))
        if not increasing and not decreasing:
            raise ValueError("bins must be monotonically increasing or decreasing")
        ascending = increasing
    else:
        ascending = True
    if ascending:
        side = 'left' if right else 'right'
        return searchsorted(bins, x, side=side)
    else:
        # Descending: flip bins, search, then flip result
        bins_rev = bins[::-1]
        side = 'left' if right else 'right'
        result = searchsorted(bins_rev, x, side=side)
        return n - result


def _empty_int_sequence_type_error():
    return TypeError(
        "indices must be integral: the provided empty sequence was"
        " inferred as float. Wrap it with np.intp for clarity."
    )


def _normalize_integral_index_array(indices):
    """Normalize one index input to an int64 ndarray plus scalar metadata."""
    if isinstance(indices, float):
        raise TypeError("Expected type 'int' but 'float' found.")

    scalar_input = isinstance(indices, int)
    was_ndarray = isinstance(indices, ndarray)

    if isinstance(indices, (list, tuple)) and len(indices) == 0:
        raise _empty_int_sequence_type_error()

    if was_ndarray:
        arr = indices
        dt = str(arr.dtype)
        if dt.startswith('float'):
            if arr.size == 0:
                raise TypeError("only int indices permitted")
        elif dt not in ('int64',):
            arr = arr.astype('int64')
    else:
        arr = array([indices], dtype='int64') if scalar_input else array(indices, dtype='int64')

    if arr.size == 0:
        dt = str(arr.dtype)
        if dt.startswith('float'):
            raise TypeError("only int indices permitted")

    return arr, scalar_input, was_ndarray


def _normalize_multi_index_sequence(multi_index):
    """Normalize multi_index to a tuple of coordinate arrays."""
    if isinstance(multi_index, ndarray) and multi_index.ndim == 2:
        return tuple(multi_index[i] for i in range(multi_index.shape[0])), False, True

    arrays = []
    all_scalar = True
    for item in multi_index:
        if isinstance(item, float):
            raise TypeError("Expected type 'int' but 'float' found.")
        if isinstance(item, (list, tuple)) and len(item) == 0:
            raise _empty_int_sequence_type_error()
        if isinstance(item, int):
            arrays.append(array([item], dtype='int64'))
        elif isinstance(item, ndarray):
            arrays.append(item)
            all_scalar = False
        else:
            arrays.append(array(item, dtype='int64'))
            all_scalar = False
    return tuple(arrays), all_scalar, False


def unravel_index(indices, shape, order='C'):
    # Handle 0-d shape
    shape = tuple(int(s) for s in shape)
    if len(shape) == 0:
        if isinstance(indices, int):
            if indices == 0:
                return ()
            raise ValueError("index {} is out of bounds for array with size 1".format(indices))
        _idx_seq = list(indices) if not isinstance(indices, ndarray) else indices.tolist()
        if not isinstance(_idx_seq, list):
            _idx_seq = [_idx_seq]
        for _v in _idx_seq:
            if int(_v) != 0:
                raise ValueError(
                    "index {} is out of bounds for array with size 1".format(int(_v)))
        raise ValueError("multiple indices for 0d array")
    indices, scalar_input, _was_ndarray = _normalize_integral_index_array(indices)
    if indices.size == 0:
        return tuple([array([], dtype='int64') for _ in shape])
    if order == 'C':
        result = _native.unravel_index(indices, shape)
        orig_shape = indices.shape
        if len(orig_shape) > 1:
            result = tuple(r.reshape(list(orig_shape)) for r in result)
        if scalar_input:
            return tuple(int(r.tolist()[0]) if hasattr(r, 'tolist') else int(r) for r in result)
        return result
    # Validate bounds
    total_size = 1
    for s in shape:
        total_size *= s
    idx_list = _flat_int_index_values(indices)
    for idx in idx_list:
        idx_val = idx
        if idx_val < 0 or idx_val >= total_size:
            raise ValueError(
                "index {} is out of bounds for array with size {}".format(
                    idx_val, total_size))
    # Compute unravel in Python to support order='F'
    flat = idx_list
    ndim = len(shape)
    if order == 'F':
        # F-order: first axis changes fastest
        # idx[k] = (n // prod(shape[:k])) % shape[k]
        result_cols = []
        for k in range(ndim):
            stride = 1
            for j in range(k):
                stride *= shape[j]
            result_cols.append(array([(v // stride) % shape[k] for v in flat], dtype='int64'))
    else:
        # C-order: last axis changes fastest
        result_cols = []
        stride = 1
        for k in range(ndim - 1, -1, -1):
            stride_k = stride
            result_cols.insert(0, array([(v // stride_k) % shape[k] for v in flat], dtype='int64'))
            stride *= shape[k]
    # Restore original shape if indices was multi-dimensional
    orig_shape = indices.shape
    if len(orig_shape) > 1:
        result_cols = [r.reshape(list(orig_shape)) for r in result_cols]
    result = tuple(result_cols)
    # When scalar input, return tuple of int scalars (not arrays)
    if scalar_input:
        return tuple(int(r.tolist()[0]) if hasattr(r, 'tolist') else int(r) for r in result)
    return result


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    arrays, all_scalar_inputs, was_2d_array = _normalize_multi_index_sequence(multi_index)
    dims = tuple(int(d) for d in dims)
    if len(arrays) != len(dims):
        raise ValueError("parameter multi_index must be a sequence of length {}".format(len(dims)))
    # Validate dimensions - check for 0 in dims with non-empty indices
    for d, arr in zip(dims, arrays):
        if d == 0:
            if hasattr(arr, 'size') and arr.size > 0:
                raise ValueError("invalid entry in coordinates array")
    # Apply modes to indices
    processed = []
    for arr, d in zip(arrays, dims):
        vals = _flat_int_index_values(arr)
        new_vals = []
        for v in vals:
            if mode == 'raise':
                if v < 0 or v >= d:
                    raise ValueError(
                        "index {} is out of bounds for axis with size {}".format(v, d))
                new_vals.append(v)
            elif mode == 'wrap':
                new_vals.append(v % d if d > 0 else 0)
            elif mode == 'clip':
                new_vals.append(max(0, min(v, d - 1)))
            elif isinstance(mode, (list, tuple)):
                # Per-axis mode handled below
                new_vals.append(v)
            else:
                new_vals.append(v)
        processed.append(new_vals)
    # Handle tuple of per-axis modes
    if isinstance(mode, (list, tuple)):
        if len(mode) != len(dims):
            raise ValueError("mode must have same length as multi_index")
        for axis_idx, (vals, d, m) in enumerate(zip(processed, dims, mode)):
            new_vals = []
            for v in vals:
                if m == 'raise':
                    if v < 0 or v >= d:
                        raise ValueError(
                            "index {} is out of bounds for axis with size {}".format(v, d))
                    new_vals.append(v)
                elif m == 'wrap':
                    new_vals.append(v % d if d > 0 else 0)
                elif m == 'clip':
                    new_vals.append(max(0, min(v, d - 1)))
                else:
                    new_vals.append(v)
            processed[axis_idx] = new_vals
    # Compute flat index
    n = len(processed[0]) if processed else 0
    if n == 0:
        # Check if original arrays were ndarrays of int (not empty float seqs)
        for arr in arrays:
            if hasattr(arr, 'size') and arr.size == 0:
                _dt = str(arr.dtype) if hasattr(arr, 'dtype') else ''
                if _dt.startswith('float'):
                    raise TypeError("only int indices permitted")
        return array([], dtype='int64')
    if order == 'C' and mode == 'raise':
        result = _native.ravel_multi_index(arrays, dims)
        if not was_2d_array and all_scalar_inputs:
            if hasattr(result, 'tolist'):
                vals = result.tolist()
                return int(vals[0] if isinstance(vals, list) else vals)
            return int(result)
        return result
    if order == 'F':
        # F-order: result = idx[0] + d[0]*idx[1] + d[0]*d[1]*idx[2] + ...
        result_vals = [0] * n
        stride = 1
        for k, (vals, d) in enumerate(zip(processed, dims)):
            for i in range(n):
                result_vals[i] += stride * vals[i]
            stride *= d
    else:
        # C-order: result = idx[-1] + d[-1]*idx[-2] + ...
        result_vals = [0] * n
        stride = 1
        for k in range(len(dims) - 1, -1, -1):
            vals = processed[k]
            d = dims[k]
            for i in range(n):
                result_vals[i] += stride * vals[i]
            stride *= d
    result = array(result_vals, dtype='int64')
    # When all scalar inputs, return int scalar
    if not was_2d_array and all_scalar_inputs:
        return int(result_vals[0])
    return result


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
    if dtype is None:
        dtype = 'float64'
    row_idx = arange(N).reshape((N, 1))
    col_idx = arange(M).reshape((1, M))
    return (col_idx <= (row_idx + k)).astype(dtype)


def tril(m, k=0):
    """Lower triangle of an array. Return a copy with elements above the k-th diagonal zeroed."""
    m = asarray(m)
    out_dtype = str(m.dtype)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    rows, cols = m.shape[-2], m.shape[-1]
    mask = tri(rows, cols, k=k, dtype='bool')
    z = zeros(m.shape, dtype=out_dtype)
    import numpy as _np
    result = _np.where(mask, m, z)
    if str(result.dtype) != out_dtype:
        result = result.astype(out_dtype)
    return result


def triu(m, k=0):
    """Upper triangle of an array. Return a copy with elements below the k-th diagonal zeroed."""
    m = asarray(m)
    out_dtype = str(m.dtype)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    rows, cols = m.shape[-2], m.shape[-1]
    # triu = NOT tril(k-1)
    mask = tri(rows, cols, k=k - 1, dtype='bool')
    z = zeros(m.shape, dtype=out_dtype)
    import numpy as _np
    # triu: keep where mask is False (above k-1 diagonal = at or above k diagonal)
    result = _np.where(mask, z, m)
    if str(result.dtype) != out_dtype:
        result = result.astype(out_dtype)
    return result


def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array of at least 2-d in place."""
    if not isinstance(a, ndarray):
        raise ValueError("Input must be an ndarray")
    if a.ndim < 2:
        raise ValueError("Input must be at least 2-d.")
    if a.ndim > 2:
        # All dimensions must be equal length
        s0 = a.shape[0]
        for d in a.shape[1:]:
            if d != s0:
                raise ValueError(
                    "All dimensions of input must be of equal length")
        # N-d: step through all equal-indexed elements
        for k in _builtin_range(s0):
            idx = tuple([k] * a.ndim)
            a[idx] = val
        return

    n = a.shape[0]
    m = a.shape[1]
    if wrap:
        # wrap mode: after reaching end of row, wrap around to continue
        # filling diagonal elements for tall matrices
        step = m + 1
        total = n * m
        k = 0
        while k < total:
            i = k // m
            j = k % m
            if isinstance(val, (list, tuple, ndarray)):
                vl = val if not isinstance(val, ndarray) else val.tolist()
                if isinstance(vl, list):
                    a[i, j] = vl[(k // step) % len(vl)]
                else:
                    a[i, j] = vl
            else:
                a[i, j] = val
            k += step
    else:
        diag_len = _builtin_min(n, m)
        for k in _builtin_range(diag_len):
            if isinstance(val, (list, tuple, ndarray)):
                vl = val if not isinstance(val, ndarray) else val.tolist()
                if isinstance(vl, list):
                    a[k, k] = vl[k % len(vl)]
                else:
                    a[k, k] = vl
            else:
                a[k, k] = val


def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array."""
    idx = arange(0, n)
    return tuple([idx] * ndim)


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array."""
    arr = asarray(arr)
    if arr.ndim < 2:
        raise ValueError("input array must be at least 2-d")
    # All dimensions must be equal length
    s0 = arr.shape[0]
    for d in arr.shape[1:]:
        if d != s0:
            raise ValueError(
                "All dimensions of input must be of equal length")
    return diag_indices(s0, arr.ndim)


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


def advanced_fancy_index(arr, indices):
    """Handle multi-axis fancy indexing: arr[[0,1], [2,3]] -> [arr[0,2], arr[1,3]]."""
    arr = asarray(arr)
    # Normalise each index array to a flat Python list of ints
    idx_arrays = [_flat_int_index_values(asarray(idx)) for idx in indices]
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


# --- mgrid / ogrid / ix_ ----------------------------------------------------

def _normalize_grid_key(key):
    if not isinstance(key, tuple):
        key = (key,)
    return key


def _coerce_grid_item(item):
    if isinstance(item, slice):
        grid, _ = _grid_slice(item)
        return grid
    return array([float(item)])


def _coerce_ix_arg(arg):
    """Normalize one ix_ input to a 1-D ndarray with index semantics."""
    is_int_input = isinstance(arg, range) or (
        isinstance(arg, (list, tuple)) and len(arg) > 0 and
        all(isinstance(x, int) and not isinstance(x, bool) for x in arg)
    )
    if isinstance(arg, ndarray):
        arr = arg
    elif isinstance(arg, range):
        arr = array(list(arg)).astype('int64')
    elif is_int_input:
        arr = array(arg).astype('int64')
    else:
        arr = asarray(arg)
    if arr.ndim != 1:
        raise ValueError("Cross index must be 1 dimensional")
    if str(arr.dtype) == 'bool':
        arr = array([j for j in _builtin_range(arr.size) if arr[j]]).astype('int64')
    if arr.size == 0 and str(arr.dtype) == 'float64':
        arr = arr.astype('int64')
    return arr


def _grid_slice(s):
    """Process a single slice for mgrid/ogrid. Returns (array, is_complex_step)."""
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else 0
    step = s.step if s.step is not None else 1
    # Complex step means "use linspace with this many points"
    _step_is_complex = isinstance(step, complex)
    if not _step_is_complex and hasattr(step, 'imag'):
        try:
            _step_is_complex = (step.imag != 0)
        except Exception:
            _step_is_complex = False
    if _step_is_complex:
        num = int(abs(step))
        grid = linspace(float(start), float(stop), num=num, endpoint=True)
        return grid, True
    # Use arange for numeric steps
    grid = arange(start, stop, step)
    return grid, False


class _MGrid:
    """Return dense multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        key = _normalize_grid_key(key)
        ndim = len(key)
        arrays = [_coerce_grid_item(item) for item in key]

        if ndim == 1:
            return arrays[0]

        # Create dense meshgrid, return as stacked ndarray
        shapes = [a.size for a in arrays]
        grids = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = shapes[i]
            reshaped = arr.reshape(shape)
            reps = list(shapes)
            reps[i] = 1
            grids.append(tile(reshaped, reps))
        return stack(grids)

mgrid = _MGrid()


class _OGrid:
    """Return open (sparse) multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        key = _normalize_grid_key(key)
        ndim = len(key)
        arrays = [_coerce_grid_item(item) for item in key]

        if ndim == 1:
            return arrays[0]

        # Sparse: each array reshaped to broadcast along its own axis
        result = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = arr.size
            result.append(arr.reshape(shape))
        return result

ogrid = _OGrid()


def ix_(*args):
    """Construct an open mesh from multiple sequences for cross-indexing."""
    ndim = len(args)
    result = []
    for i, arg in enumerate(args):
        arr = _coerce_ix_arg(arg)
        shape = [1] * ndim
        shape[i] = arr.size
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
                # Complex step: linspace
                _step_is_complex = isinstance(step, complex)
                if not _step_is_complex and hasattr(step, 'imag'):
                    try:
                        _step_is_complex = (step.imag != 0)
                    except Exception:
                        _step_is_complex = False
                if _step_is_complex:
                    num = int(abs(step))
                    pieces.append(linspace(float(start), float(stop), num=num, endpoint=True))
                else:
                    pieces.append(arange(start, stop, step))
            elif isinstance(item, (int, float)):
                pieces.append(array([item]))
            else:
                arr = asarray(item)
                # Ensure at least 1-d
                if arr.ndim == 0:
                    arr = arr.reshape((1,))
                pieces.append(arr)
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
                if arr.ndim == 0:
                    arr = arr.reshape((1,))
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
        for s in shape:
            if int(s) < 0:
                raise ValueError(
                    "negative dimensions are not allowed")
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
