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


def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """Compute the multidimensional histogram of some data."""
    sample = asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    n_samples, n_dims = sample.shape[0], sample.shape[1]
    sample_list = sample.tolist()

    # Determine bins per dimension
    if isinstance(bins, int):
        bins_per_dim = [bins] * n_dims
    elif isinstance(bins, (list, tuple)):
        bins_per_dim = [int(b) for b in bins]
    else:
        bins_per_dim = [int(bins)] * n_dims

    _range = range

    # Build edges
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
    if not isinstance(indices, ndarray):
        if isinstance(indices, int):
            indices = array([indices])
        else:
            indices = array(indices)
    return _native.unravel_index(indices, shape)


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    arrays = tuple(array([a]) if isinstance(a, (int, float)) else (a if isinstance(a, ndarray) else array(a)) for a in multi_index)
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
