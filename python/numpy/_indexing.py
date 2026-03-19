"""Index generation, iteration, histograms."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import AxisError, _builtin_range, _builtin_min, _builtin_max
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


def meshgrid(*xi, indexing='xy'):
    arrays = [a if isinstance(a, ndarray) else array(a) for a in xi]
    return _native.meshgrid(arrays, indexing)


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


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Compute the bin edges for a histogram without computing the histogram itself."""
    a = asarray(a)
    if isinstance(bins, str):
        return _histogram_bin_edges_from_method(a, bins, range=range)
    if isinstance(bins, int):
        flat = a.flatten().tolist()
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            if flat:
                lo, hi = _builtin_min(flat), _builtin_max(flat)
            else:
                lo, hi = 0.0, 1.0
        if lo == hi:
            lo = lo - 0.5
            hi = hi + 0.5
        edges = linspace(lo, hi, bins + 1)
        return edges
    else:
        return asarray(bins)


def _histogram_bin_edges_from_method(a, method, range=None):
    """Compute bin edges using an automatic bin-width method (string name)."""
    flat = a.flatten()
    n = flat.size
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
        # Filter data to range
        vals = [float(flat[i]) for i in _builtin_range(n) if lo <= float(flat[i]) <= hi]
        n_eff = len(vals)
    else:
        vals = [float(flat[i]) for i in _builtin_range(n)]
        n_eff = n
        if n_eff == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = _builtin_min(vals)
            hi = _builtin_max(vals)
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
    if not isinstance(a, ndarray):
        a = array(a)
    # Validate weights
    if weights is not None:
        w = asarray(weights)
        if w.shape != a.shape:
            raise ValueError("weights should have the same shape as a")
    # Handle string bin methods
    if isinstance(bins, str):
        edges = _histogram_bin_edges_from_method(a, bins, range=range)
        return histogram(a, bins=edges, range=range, density=density, weights=weights)
    if isinstance(bins, (list, tuple, ndarray)):
        # Custom bin edges
        edges = asarray(bins).flatten()
        if edges.ndim != 1:
            raise ValueError("bins must be 1d")
        edge_list = edges.tolist()
        n_bins = len(edge_list) - 1
        if n_bins < 1:
            raise ValueError("bins must have at least 2 edges")
        # Validate finite range
        for e in edge_list:
            if _math.isinf(e) or _math.isnan(e):
                raise ValueError("bins must be finite")
        flat = a.flatten().tolist()
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
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
        if not (_math.isfinite(lo) and _math.isfinite(hi)):
            raise ValueError("supplied range of [{}, {}] is not finite".format(lo, hi))
        if lo > hi:
            raise ValueError("max must be larger than min in range parameter")
        if lo == hi:
            lo = lo - 0.5
            hi = hi + 0.5
    if weights is not None or range is not None:
        # Python fallback for weights/range with integer bins
        flat = a.flatten().tolist()
        if range is None:
            if len(flat) == 0:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = _builtin_min(flat), _builtin_max(flat)
            if lo == hi:
                lo = lo - 0.5
                hi = hi + 0.5
        edges = linspace(lo, hi, num=bins + 1, endpoint=True)
        edge_list = edges.tolist()
        counts = [0.0] * bins
        w_list = None
        if weights is not None:
            w_list = asarray(weights).flatten().tolist()
        for idx_val, val in enumerate(flat):
            if val < lo or val > hi:
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
        if len(bins) == 1:
            raise ValueError("bins must be a sequence of 1 or 2 elements")
        if len(bins) == 2:
            xbins, ybins = bins[0], bins[1]
        else:
            raise ValueError("bins must be a sequence of 1 or 2 elements")
        # Each bin spec can be int or array-like
        dd_bins = [xbins, ybins]
    else:
        dd_bins = [bins, bins]

    # Stack x, y into (N, 2) sample
    sample = _np.column_stack([x.flatten(), y.flatten()])

    hist, edges = _np.histogramdd(sample, bins=dd_bins, range=range,
                                  density=density, weights=weights)
    return hist, edges[0], edges[1]


def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """Compute the multidimensional histogram of some data."""
    sample = asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    n_samples, n_dims = sample.shape[0], sample.shape[1]
    sample_list = sample.tolist()

    _range = range

    # Determine bins per dimension - each can be int or array-like edge list
    if isinstance(bins, int):
        bins_spec = [bins] * n_dims
    elif isinstance(bins, (list, tuple)):
        if len(bins) == n_dims:
            bins_spec = list(bins)
        else:
            # Try to interpret as a single int/array for all dims
            bins_spec = [bins] * n_dims
    else:
        bins_spec = [int(bins)] * n_dims

    # Build edges for each dimension
    edges = []
    bins_per_dim = []
    for d in _builtin_range(n_dims):
        b = bins_spec[d]
        if isinstance(b, (int, float)):
            nb = int(b)
            vals = [row[d] for row in sample_list] if n_samples > 0 else []
            if _range is not None and _range[d] is not None:
                lo, hi = float(_range[d][0]), float(_range[d][1])
            elif len(vals) > 0:
                lo, hi = _builtin_min(vals), _builtin_max(vals)
            else:
                lo, hi = 0.0, 1.0
            edge = linspace(lo, hi, num=nb + 1, endpoint=True).tolist()
            edges.append(edge)
            bins_per_dim.append(nb)
        elif isinstance(b, (list, tuple)):
            # b is edge array
            edge = [float(v) for v in b]
            edges.append(edge)
            bins_per_dim.append(len(edge) - 1)
        elif isinstance(b, ndarray):
            edge = b.flatten().tolist()
            edges.append(edge)
            bins_per_dim.append(len(edge) - 1)
        else:
            nb = int(b)
            vals = [row[d] for row in sample_list] if n_samples > 0 else []
            if _range is not None and _range[d] is not None:
                lo, hi = float(_range[d][0]), float(_range[d][1])
            elif len(vals) > 0:
                lo, hi = _builtin_min(vals), _builtin_max(vals)
            else:
                lo, hi = 0.0, 1.0
            edge = linspace(lo, hi, num=nb + 1, endpoint=True).tolist()
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


def unravel_index(indices, shape, order='C'):
    if isinstance(indices, float):
        raise TypeError("Expected type 'int' but 'float' found.")
    # Handle 0-d shape
    shape = tuple(shape) if not isinstance(shape, tuple) else shape
    if len(shape) == 0:
        if isinstance(indices, int):
            if indices == 0:
                return ()
            raise ValueError("index {} is out of bounds for array with size 1".format(indices))
        # Check if it's a sequence with a single element that's out of bounds
        _idx_seq = list(indices) if not isinstance(indices, ndarray) else indices.tolist()
        if not isinstance(_idx_seq, list):
            _idx_seq = [_idx_seq]
        for _v in _idx_seq:
            if int(_v) != 0:
                raise ValueError(
                    "index {} is out of bounds for array with size 1".format(int(_v)))
        # Array/sequence with 0-d shape
        raise ValueError("multiple indices for 0d array")
    # Validate: reject empty sequences
    if isinstance(indices, (list, tuple)) and len(indices) == 0:
        raise TypeError(
            "indices must be integral: the provided empty sequence was"
            " inferred as float. Wrap it with np.intp for clarity.")
    _was_ndarray = isinstance(indices, ndarray)
    if not _was_ndarray:
        if isinstance(indices, int):
            indices = array([indices])
        else:
            indices = array(indices)
    # Reject empty float-typed arrays (ambiguous - could be int or float)
    if _was_ndarray and indices.size == 0:
        _dt = str(indices.dtype)
        if _dt.startswith('float'):
            raise TypeError("only int indices permitted")
    if indices.size == 0:
        # Return empty arrays per dimension
        return tuple([array([], dtype='int64') for _ in shape])
    # Validate bounds
    total_size = 1
    for s in shape:
        total_size *= s
    idx_list = indices.flatten().tolist()
    for idx in idx_list:
        idx_val = int(idx)
        if idx_val < 0 or idx_val >= total_size:
            raise ValueError(
                "index {} is out of bounds for array with size {}".format(
                    idx_val, total_size))
    return _native.unravel_index(indices, shape)


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    # Validate types - reject floats
    for a in multi_index:
        if isinstance(a, float):
            raise TypeError("Expected type 'int' but 'float' found.")
    arrays = tuple(array([a]) if isinstance(a, (int, float)) else (a if isinstance(a, ndarray) else array(a)) for a in multi_index)
    # Validate dimensions - check for 0 in dims with non-empty indices
    dims = tuple(int(d) for d in dims)
    for d in dims:
        if d == 0:
            # Check if any indices are non-empty
            for arr in arrays:
                if arr.size > 0:
                    raise ValueError(
                        "invalid entry in coordinates array")
    # Validate bounds in 'raise' mode
    if mode == 'raise':
        for i, (arr, d) in enumerate(zip(arrays, dims)):
            vals = arr.flatten().tolist()
            for v in vals:
                v_int = int(v)
                if v_int < 0 or v_int >= d:
                    raise ValueError(
                        "index {} is out of bounds for array "
                        "with size {}".format(v_int, d))
    # Check for overflow: total product of dims
    total = 1
    for d in dims:
        new_total = total * d
        if d != 0 and new_total // d != total:
            raise ValueError(
                "invalid dims: product of dims would overflow")
        total = new_total
    import sys
    if total > sys.maxsize:
        raise ValueError(
            "invalid dims: product of dims too large")
    return _native.ravel_multi_index(arrays, dims)


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
    rows = []
    for i in range(N):
        row = []
        for j in range(M):
            row.append(1 if j <= i + k else 0)
        rows.append(row)
    return array(rows, dtype=dtype)


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


# --- mgrid / ogrid / ix_ ----------------------------------------------------

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
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                grid, _ = _grid_slice(s)
                arrays.append(grid)
            else:
                arrays.append(array([float(s)]))

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
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                grid, _ = _grid_slice(s)
                arrays.append(grid)
            else:
                arrays.append(array([float(s)]))

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
        arr = asarray(arg)
        if arr.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        # Boolean arrays: convert to integer index via nonzero/where
        if str(arr.dtype) == 'bool':
            arr = array([j for j in _builtin_range(arr.size) if arr[j]])
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
