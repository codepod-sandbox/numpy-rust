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

__all__ = [
    # Shape
    'reshape', 'ravel', 'flatten', 'expand_dims', 'squeeze', 'transpose',
    'moveaxis', 'swapaxes', 'resize',
    # At-least
    'atleast_1d', 'atleast_2d', 'atleast_3d',
    # Stacking
    'stack', 'hstack', 'vstack', 'dstack', 'column_stack', 'row_stack',
    # Splitting
    'split', 'array_split', 'hsplit', 'vsplit', 'dsplit',
    # Repetition / manipulation
    'repeat', 'tile', 'append', 'insert', 'delete',
    # Sorting
    'sort', 'argsort', 'lexsort', 'partition', 'argpartition', 'unique',
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
    'size', 'block',
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


def reshape(a, newshape=None, order="C", *, shape=None, copy=None, **kwargs):
    # Handle the case where newshape is passed as keyword 'newshape'
    if 'newshape' in kwargs:
        if newshape is not None and shape is not None:
            raise TypeError("You cannot specify 'newshape' and 'shape' arguments at the same time.")
        import warnings as _w
        _w.warn("keyword argument 'newshape' is deprecated, use 'shape' instead", DeprecationWarning, stacklevel=2)
        newshape = kwargs.pop('newshape')
    if newshape is not None and shape is not None:
        raise TypeError("You cannot specify 'newshape' and 'shape' arguments at the same time.")
    if shape is not None:
        newshape = shape
    if newshape is None:
        raise TypeError("reshape() missing 1 required positional argument: 'shape'")
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Delegate copy= and order= to the instance method (which handles memory tagging)
    kw = {}
    if order is not None:
        kw['order'] = order
    if copy is not None:
        kw['copy'] = copy
    return a.reshape(newshape, **kw)


def _transpose_with_axes(a, axes):
    """Transpose ndarray with an arbitrary axis permutation (pure Python).

    Parameters
    ----------
    a : ndarray
    axes : list/tuple of int - the desired permutation of axes.

    Returns an ndarray with axes reordered according to *axes*.
    """
    if not isinstance(a, ndarray):
        a = asarray(a)
    shape = a.shape
    ndim_a = len(shape)
    if axes is None:
        return a.T
    axes = list(axes)
    if len(axes) != ndim_a:
        raise ValueError("axes don't match array")
    # Fast-paths
    if axes == list(range(ndim_a)):
        return a.copy() if hasattr(a, 'copy') else array(a.tolist())
    if ndim_a == 2 and axes == [1, 0]:
        return a.T
    # Generic: walk every index of the output and pick from source
    new_shape = tuple(shape[ax] for ax in axes)
    size = 1
    for s in new_shape:
        size *= s
    flat_data = a.flatten()
    # Build strides of original array (row-major)
    src_strides = [0] * ndim_a
    s = 1
    for i in range(ndim_a - 1, -1, -1):
        src_strides[i] = s
        s *= shape[i]
    result = [0.0] * size
    # Iterate over every multi-index of the *output*
    out_idx = [0] * ndim_a
    for flat_i in range(size):
        # Map output index -> source index
        src_flat = 0
        for d in range(ndim_a):
            src_flat += out_idx[d] * src_strides[axes[d]]
        result[flat_i] = float(flat_data[src_flat])
        # Increment out_idx (rightmost first)
        for d in range(ndim_a - 1, -1, -1):
            out_idx[d] += 1
            if out_idx[d] < new_shape[d]:
                break
            out_idx[d] = 0
    return array(result).reshape(list(new_shape))


def transpose(a, axes=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axes is not None:
        return _transpose_with_axes(a, axes)
    return a.T


def flatten(a, order="C"):
    return a.flatten()


def ravel(a, order="C"):
    if isinstance(a, ndarray):
        return a.ravel()
    return array(a).ravel()


def squeeze(a, axis=None):
    a = asarray(a)
    if axis is not None and isinstance(axis, (tuple, list)):
        result = a
        for ax in sorted(axis, reverse=True):
            result = squeeze(result, axis=ax)
        return result
    if isinstance(a, ndarray):
        if axis is not None and axis < 0:
            axis += a.ndim
        return a.squeeze(axis)
    return a


def expand_dims(a, axis):
    a = asarray(a)
    if isinstance(axis, (tuple, list)):
        ndim_out = a.ndim + len(axis)
        # Normalize negative axes
        normalized = []
        for ax in axis:
            if ax < 0:
                ax = ndim_out + ax
            normalized.append(ax)
        normalized.sort()
        result = a
        for ax in normalized:
            result = expand_dims(result, ax)
        return result
    if isinstance(a, ndarray):
        return a.expand_dims(axis)
    return a


def append(arr, values, axis=None):
    """Append values to the end of an array."""
    if not isinstance(arr, ndarray):
        arr = array(arr)
    if not isinstance(values, ndarray):
        values = array(values)
    if axis is None:
        return concatenate([arr.flatten(), values.flatten()])
    return concatenate([arr, values], axis=axis)


def atleast_1d(*arys):
    """Convert inputs to arrays with at least one dimension."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results


def atleast_2d(*arys):
    """Convert inputs to arrays with at least two dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1, 1])
        elif a.ndim == 1:
            a = a.reshape([1, len(a)])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results


def atleast_3d(*arys):
    """Convert inputs to arrays with at least three dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1, 1, 1])
        elif a.ndim == 1:
            a = a.reshape([1, len(a), 1])
        elif a.ndim == 2:
            a = a.reshape(list(a.shape) + [1])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results


def stack(arrays, axis=0, out=None):
    return _native.stack_native(list(arrays), axis)


def vstack(tup):
    if not isinstance(tup, (list, tuple)):
        raise TypeError("arrays to stack must be passed as a list or tuple, not " + type(tup).__name__)
    arrs = [atleast_2d(asarray(a)) for a in tup]
    if len(arrs) == 0:
        raise ValueError("need at least one array to concatenate")
    return concatenate(arrs, 0)


def hstack(tup):
    if not isinstance(tup, (list, tuple)):
        raise TypeError("arrays to stack must be passed as a list or tuple, not " + type(tup).__name__)
    arrs = [atleast_1d(asarray(a)) for a in tup]
    if len(arrs) == 0:
        raise ValueError("need at least one array to concatenate")
    if arrs[0].ndim > 1:
        return concatenate(arrs, 1)
    return concatenate(arrs, 0)


def column_stack(tup):
    """Stack 1-D arrays as columns into a 2-D array."""
    arrays = []
    for a in tup:
        a = asarray(a)
        if a.ndim == 1:
            a = a.reshape((a.size, 1))  # Make column vector
        arrays.append(a)
    return concatenate(arrays, 1)


def dstack(tup):
    return _native.dstack(list(tup))


row_stack = vstack


def split(a, indices_or_sections, axis=0):
    return _native.split(a, indices_or_sections, axis)


def vsplit(a, indices_or_sections):
    return split(a, indices_or_sections, 0)


def hsplit(a, indices_or_sections):
    if a.ndim == 1:
        return split(a, indices_or_sections, 0)
    return split(a, indices_or_sections, 1)


def array_split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays (allows unequal division)."""
    ary = asarray(ary)
    n = ary.shape[axis]
    if isinstance(indices_or_sections, (int, float)):
        nsections = int(indices_or_sections)
        neach, extras = _builtin_divmod(n, nsections)
        boundaries = [0]
        for i in range(nsections):
            boundaries.append(boundaries[-1] + neach + (1 if i < extras else 0))
        indices = boundaries[1:-1]
    else:
        indices = list(indices_or_sections)
    return split(ary, indices, axis)


def dsplit(ary, indices_or_sections):
    """Split array into multiple sub-arrays along the 3rd axis (depth)."""
    ary = asarray(ary)
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return array_split(ary, indices_or_sections, axis=2)


def block(arrays):
    """Assemble an nd-array from nested lists of blocks.

    Follows NumPy's algorithm: determine list_ndim (nesting depth) and
    result_ndim = max(list_ndim, max leaf ndim), then promote all leaves
    to result_ndim dimensions and concatenate bottom-up.
    """
    if isinstance(arrays, ndarray):
        return arrays
    if isinstance(arrays, tuple):
        raise TypeError("tuple is not allowed, use lists instead")
    if not isinstance(arrays, list):
        return asarray(arrays)
    if len(arrays) == 0:
        raise ValueError("block requires at least one element but got empty list")

    def _check_tuples(lst):
        """Check for tuples (not allowed in block)."""
        for item in lst:
            if isinstance(item, tuple):
                raise TypeError("tuple is not allowed, use lists instead")
            if isinstance(item, list):
                _check_tuples(item)
    _check_tuples(arrays)

    def _block_depth(lst):
        if not isinstance(lst, list):
            return 0
        if len(lst) == 0:
            return 1
        return 1 + _block_depth(lst[0])

    def _check_depths(lst, parent_depth=0):
        if not isinstance(lst, list):
            return
        depths = []
        for item in lst:
            depths.append(_block_depth(item))
        if len(set(depths)) > 1:
            raise ValueError(
                "List depths are mismatched. First element was at depth "
                "{}, but there is an element at depth {}".format(
                    depths[0] + parent_depth,
                    [d for d in depths if d != depths[0]][0] + parent_depth))
        for item in lst:
            if isinstance(item, list):
                _check_depths(item, parent_depth + 1)

    _check_depths(arrays)

    def _check_empty(lst):
        """Check for empty sub-lists."""
        for item in lst:
            if isinstance(item, list):
                if len(item) == 0:
                    raise ValueError("block requires at least one element but got empty list")
                _check_empty(item)
    _check_empty(arrays)

    # Collect all leaf arrays and find max ndim
    list_ndim = _block_depth(arrays)

    def _collect_leaves(lst):
        if not isinstance(lst, list):
            return [atleast_1d(asarray(lst))]
        result = []
        for item in lst:
            result.extend(_collect_leaves(item))
        return result

    leaves = _collect_leaves(arrays)
    max_leaf_ndim = _builtin_max(a.ndim for a in leaves) if leaves else 0
    result_ndim = _builtin_max(list_ndim, max_leaf_ndim)

    def _block_rec(lst, depth):
        """Recursively assemble blocks."""
        if not isinstance(lst, list):
            a = atleast_1d(asarray(lst))
            while a.ndim < result_ndim:
                a = expand_dims(a, 0)
            return a
        # Recurse into sub-elements
        sub_results = [_block_rec(item, depth + 1) for item in lst]
        # Concatenate along axis = depth
        # But we need to account for the fact that result_ndim may be
        # larger than list_ndim. The axis mapping is:
        # nesting depth d -> axis (result_ndim - list_ndim + d)
        # This is because the list dimensions correspond to the LAST
        # list_ndim axes of result_ndim... wait, no.
        # In NumPy: nesting depth d maps to axis d. But arrays are
        # promoted to result_ndim by prepending 1s. So a scalar at
        # depth 2 in a depth-2 nesting becomes shape (1,1,...,1).
        # Concatenation at depth 0 is axis 0, depth 1 is axis 1, etc.
        axis = result_ndim - list_ndim + depth
        return concatenate(sub_results, axis=axis)

    return _block_rec(arrays, 0)


def copyto(dst, src, casting='same_kind', where=True):
    """Copy values from one array to another, broadcasting as necessary."""
    src = asarray(src)
    dst = asarray(dst)
    src_b = broadcast_to(src, dst.shape)
    if where is True:
        return src_b
    mask = asarray(where)
    flat_m = mask.flatten()
    flat_s = src_b.flatten()
    flat_d = dst.flatten()
    n = flat_d.size
    result_vals = []
    for i in range(n):
        if flat_m[i]:
            result_vals.append(float(flat_s[i]))
        else:
            result_vals.append(float(flat_d[i]))
    result = array(result_vals)
    if dst.ndim > 1:
        result = result.reshape(dst.shape)
    return result


def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values."""
    arr = asarray(arr)
    mask = asarray(mask)
    vals_arr = asarray(vals).flatten()
    flat_a = arr.flatten()
    flat_m = mask.flatten()
    n = flat_a.size
    nv = vals_arr.size
    result = []
    vi = 0
    for i in range(n):
        if flat_m[i]:
            result.append(float(vals_arr[vi % nv]))
            vi += 1
        else:
            result.append(float(flat_a[i]))
    r = array(result)
    if arr.ndim > 1:
        r = r.reshape(arr.shape)
    return r


class _BroadcastIter:
    """Iterator wrapper that exposes .base pointing to the original array."""
    def __init__(self, broadcasted, base_array):
        self._flat = broadcasted.flat
        self.base = base_array
    def __iter__(self):
        return self._flat.__iter__()
    def __next__(self):
        return self._flat.__next__()


class broadcast:
    """Broadcast shape computation for multiple arrays."""
    def __init__(self, *args, **kwargs):
        if kwargs:
            raise ValueError("broadcast() does not accept keyword arguments")
        if len(args) > 64:
            raise ValueError("Need at most 64 array objects.")
        # Flatten any nested broadcast objects
        flat_args = []
        for a in args:
            if isinstance(a, broadcast):
                flat_args.extend(a._arrays)
            else:
                flat_args.append(a)
        arrays = [asarray(a) for a in flat_args]
        shapes = [a.shape for a in arrays]
        if len(shapes) == 0:
            self.shape = ()
        else:
            try:
                self.shape = broadcast_shapes(*shapes)
            except ValueError:
                # Generate numpy-compatible error message with arg indices
                ndim = _builtin_max(len(s) for s in shapes)
                for dim in range(ndim):
                    sizes = {}
                    for idx, s in enumerate(shapes):
                        offset = ndim - len(s)
                        d = dim - offset
                        if d >= 0:
                            sz = s[d]
                            if sz != 1:
                                sizes.setdefault(sz, []).append(idx)
                    if len(sizes) > 1:
                        items = sorted(sizes.items(), key=lambda x: x[1][0])
                        first_idx = items[0][1][0]
                        last_idx = items[-1][1][0]
                        raise ValueError(
                            f"arg {first_idx} with shape {shapes[first_idx]} and "
                            f"arg {last_idx} with shape {shapes[last_idx]}"
                        )
                raise
        self.nd = len(self.shape)
        self.ndim = self.nd
        self.size = 1
        for s in self.shape:
            self.size *= s
        self.numiter = len(arrays)
        self._arrays = arrays
        self.iters = tuple(_BroadcastIter(broadcast_to(a, self.shape), a) for a in arrays)
        self.index = 0

    def __iter__(self):
        # Broadcast each array to the common shape and iterate
        broadcasted = [broadcast_to(a, self.shape).flatten() for a in self._arrays]
        for i in range(self.size):
            yield tuple(float(b[i]) for b in broadcasted)

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration
        broadcasted = [broadcast_to(a, self.shape).flatten() for a in self._arrays]
        result = tuple(float(b[self.index]) for b in broadcasted)
        self.index += 1
        return result

    def reset(self):
        self.index = 0


def repeat(a, repeats, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.repeat(a, repeats, axis)


def tile(a, reps):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.tile(a, reps)


def _native_resize(col, total):
    """Tile a 1D ndarray to length total."""
    n = len(col)
    flat = col.flatten().tolist()
    result_vals = [flat[i % n] for i in _builtin_range(total)]
    return asarray(result_vals).astype(str(col.dtype))


def _resize_structured(a, new_shape):
    """Resize a StructuredArray by tiling its fields to fill new_shape."""
    import json as _json
    import _numpy_native as _native_mod
    from numpy import StructuredArray
    from _numpy_native import ndarray

    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    else:
        new_shape = tuple(new_shape)

    total = 1
    for s in new_shape:
        total *= s

    dt = a.dtype
    native = object.__getattribute__(a, '_native_arr')

    if total == 0:
        return zeros(new_shape, dtype=dt)

    # Build dtype_json for the new native array
    dtype_json = _json.dumps([[nm, str(dt.fields[nm][0])] for nm in dt.names])

    # Tile each field independently
    new_fields = []
    for name in dt.names:
        col = native[name]  # PyNdArray, 1D
        if len(col) == 0:
            tiled = zeros(total, dtype=str(col.dtype))
            if not isinstance(tiled, ndarray):
                tiled = asarray([0] * total).astype(str(col.dtype))
        else:
            tiled = _native_resize(col, total)
        new_fields.append((name, tiled))

    native_fields = [(name, col) for name, col in new_fields]
    new_native = _native_mod.StructuredArray(native_fields, [total], dtype_json)
    flat = StructuredArray(new_native)

    if len(new_shape) == 1:
        return flat
    return flat.reshape(new_shape)


def resize(a, new_shape):
    from numpy import StructuredArray
    if isinstance(a, StructuredArray):
        return _resize_structured(a, new_shape)
    a = asarray(a)
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    total = 1
    for s in new_shape:
        total *= s
    dt = a.dtype
    if total == 0:
        return zeros(new_shape, dtype=dt)
    flat = a.flatten().tolist()
    n = len(flat)
    if n == 0:
        return zeros(new_shape, dtype=dt)
    result = []
    for i in range(total):
        result.append(flat[i % n])
    return array(result, dtype=dt).reshape(new_shape)


def choose(a, choices, out=None, mode="raise"):
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Convert choices: complex scalars become float (real part) to match Rust clip behavior
    def _to_choice_array(c):
        if isinstance(c, complex):
            return asarray(c.real)
        if isinstance(c, ndarray):
            return c
        return asarray(c)
    choice_arrays = [_to_choice_array(c) for c in choices]
    result = _native.choose(a, choice_arrays)
    if out is not None and isinstance(out, ndarray):
        flat = result.flatten().tolist()
        for i in range(len(flat)):
            out.flat[i] = flat[i]
        return out
    return result


def compress(condition, a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    cond = condition if isinstance(condition, ndarray) else asarray(condition)
    return _native.compress(cond, a, axis)


def roll(a, shift, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        # Flatten, roll, reshape back
        flat = a.flatten()
        n = flat.size
        if n == 0:
            return a.copy()
        s = int(shift) % n if n > 0 else 0
        if s == 0:
            return a.copy()
        # Roll via concatenation
        parts = [flat[n - s:], flat[:n - s]]
        return concatenate(parts).reshape(a.shape)
    if isinstance(axis, (tuple, list)):
        # Multiple axes: apply roll sequentially
        result = a
        shifts = shift if isinstance(shift, (tuple, list)) else [shift] * len(axis)
        for sh, ax in zip(shifts, axis):
            result = roll(result, sh, ax)
        return result
    # Single axis roll
    ax = int(axis)
    if ax < 0:
        ax += a.ndim
    n = a.shape[ax]
    if n == 0:
        return a.copy()
    s = int(shift) % n if n > 0 else 0
    if s == 0:
        return a.copy()
    return _native.roll(a, s, ax)


def moveaxis(a, source, destination):
    """Move axes of an array to new positions.

    Other axes remain in their original order.
    """
    from . import ma as _ma_mod
    _is_masked = isinstance(a, _ma_mod.MaskedArray)
    if _is_masked:
        a = a.data
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    # Normalise to lists
    if isinstance(source, int):
        source = [source]
    if isinstance(destination, int):
        destination = [destination]
    # Validate lengths
    if len(source) != len(destination):
        raise ValueError("`source` and `destination` arguments must have the same number of elements")
    # Normalize and validate bounds
    src_norm = []
    for s in source:
        s_n = s if s >= 0 else s + ndim_a
        if s_n < 0 or s_n >= ndim_a:
            raise AxisError("source {} is out of bounds for array of dimension {}".format(s, ndim_a))
        src_norm.append(s_n)
    dst_norm = []
    for d in destination:
        d_n = d if d >= 0 else d + ndim_a
        if d_n < 0 or d_n >= ndim_a:
            raise AxisError("destination {} is out of bounds for array of dimension {}".format(d, ndim_a))
        dst_norm.append(d_n)
    # Check for repeated axes
    if len(set(src_norm)) != len(src_norm):
        raise ValueError("repeated axis in `source`")
    if len(set(dst_norm)) != len(dst_norm):
        raise ValueError("repeated axis in `destination`")
    source = src_norm
    destination = dst_norm
    # Build permutation: start with axes not in source, in order
    order = [i for i in range(ndim_a) if i not in source]
    # Insert source axes at destination positions (must insert in sorted dest order)
    pairs = sorted(zip(destination, source))
    for dst, src in pairs:
        order.insert(dst, src)
    result = _transpose_with_axes(a, order)
    if _is_masked:
        result = _ma_mod.MaskedArray(result)
    return result


def rollaxis(a, axis, start=0):
    """Roll the specified axis backwards, until it lies in position *start*."""
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    if axis < -ndim_a or axis >= ndim_a:
        raise AxisError("axis {} is out of bounds for array of dimension {}".format(axis, ndim_a))
    if axis < 0:
        axis += ndim_a
    if start < -ndim_a or start > ndim_a:
        raise AxisError("start {} is out of bounds for array of dimension {}".format(start, ndim_a))
    if start < 0:
        start += ndim_a
    if start > axis:
        start -= 1
    return moveaxis(a, axis, start)


def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array."""
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    if axis1 < 0:
        axis1 += ndim_a
    if axis2 < 0:
        axis2 += ndim_a
    if axis1 == axis2:
        return a
    order = list(range(ndim_a))
    order[axis1], order[axis2] = order[axis2], order[axis1]
    return _transpose_with_axes(a, order)


def sort(a, axis=-1, kind=None, order=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    original_dtype = str(a.dtype)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    result = a.sort(axis)
    # Preserve original dtype (Rust sort may convert to float64)
    if original_dtype in ('int32', 'int64') and str(result.dtype) != original_dtype:
        result = result.astype(original_dtype)
    return result


def argsort(a, axis=-1, kind=None, order=None):
    if isinstance(a, (tuple, list)):
        if len(a) == 0:
            return _native.zeros((0,), 'int64')
        a = _native.array([float(x) for x in a])
    elif isinstance(a, _ObjectArray):
        vals = [float(x) if isinstance(x, (int, float)) else 0. for x in (a._data or [])]
        inds = sorted(range(len(vals)), key=lambda i: vals[i])
        return _native.array([float(i) for i in inds]) if inds else _native.zeros((0,), 'int64')
    else:
        a = asarray(a)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    return a.argsort(axis)


def size(a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        return a.size
    if isinstance(axis, tuple):
        if len(axis) == 0:
            return 1
        result = 1
        for ax in axis:
            result *= a.shape[ax]
        return result
    return a.shape[axis]


def take(a, indices, axis=None, out=None, mode="raise"):
    if isinstance(a, ndarray):
        return _native.take(a, indices, axis)
    flat = array(a).flatten()
    if isinstance(indices, ndarray):
        idx = [int(float(indices.flatten()[i])) for i in range(indices.size)]
    else:
        idx = list(indices) if hasattr(indices, '__iter__') else [indices]
    return array([float(flat[i]) for i in idx])


def flip(a, axis=None):
    """Reverse the order of elements along the given axis."""
    if isinstance(a, ndarray):
        return _native.flip(a, axis)
    return array(list(reversed(a))) if axis is None else a


def flipud(a):
    """Flip array upside down (reverse along axis 0)."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.flipud(a)


def fliplr(a):
    """Flip array left-right (reverse along axis 1)."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return _native.fliplr(a)


def rot90(a, k=1, axes=(0, 1)):
    """Rotate array 90 degrees in the plane of the first two axes."""
    if isinstance(a, ndarray):
        return _native.rot90(a, k)
    return a


def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """Return sorted unique elements of an array.

    Parameters
    ----------
    a : array_like
    return_index : bool
        If True, return indices of first occurrences.
    return_inverse : bool
        If True, return indices to reconstruct original from unique.
    return_counts : bool
        If True, return count of each unique value.
    axis : int, optional
        The axis along which to find unique slices. If None, flatten first.

    Returns
    -------
    unique : ndarray
    unique_indices : ndarray (optional)
    unique_inverse : ndarray (optional)
    unique_counts : ndarray (optional)
    """
    from ._helpers import _ObjectArray
    if not isinstance(a, ndarray):
        a = asarray(a)

    def _get_slice(arr, ax, i):
        if ax == 0:
            return arr[i]
        tmp = swapaxes(arr, 0, ax)
        return tmp[i]

    def _slice_key(sl):
        if isinstance(sl, ndarray):
            return tuple(sl.flatten().tolist())
        if isinstance(sl, _ObjectArray):
            return tuple(sl._data)
        return (sl,)

    if axis is not None:
        # For axis=0: find unique rows (or slices along axis 0)
        if axis < 0:
            axis = a.ndim + axis
        n_slices = a.shape[axis]
        # Extract each slice as a tuple for hashing
        seen = {}
        unique_indices_list = []
        for i in range(n_slices):
            sl = _get_slice(a, axis, i)
            key = _slice_key(sl)
            if key not in seen:
                seen[key] = i
                unique_indices_list.append(i)
        # Build result array from unique indices
        if axis == 0:
            rows = [a[i] for i in unique_indices_list]
        else:
            tmp = swapaxes(a, 0, axis)
            rows = [tmp[i] for i in unique_indices_list]
        # Sort by the first element of each row for consistent ordering
        pairs = list(zip(unique_indices_list, rows))
        pairs.sort(key=lambda p: _slice_key(p[1]))
        unique_indices_list = [p[0] for p in pairs]
        rows = [p[1] for p in pairs]
        # For 1D arrays, rows are scalars — use array() directly
        if len(rows) > 0 and not isinstance(rows[0], (ndarray, _ObjectArray)):
            result = array(rows, dtype=a.dtype)
        else:
            result = stack(rows, axis=0)
            if axis != 0:
                result = swapaxes(result, 0, axis)
        extras = return_index or return_inverse or return_counts
        if not extras:
            return result
        ret = (result,)
        if return_index:
            ret = ret + (array(unique_indices_list, dtype='int64'),)
        if return_inverse:
            # Map each original slice index to its position in unique result
            key_to_pos = {}
            for pos, idx in enumerate(unique_indices_list):
                sl = _get_slice(a, axis, idx)
                key = _slice_key(sl)
                key_to_pos[key] = pos
            inv = []
            for i in range(n_slices):
                sl = _get_slice(a, axis, i)
                key = _slice_key(sl)
                inv.append(key_to_pos[key])
            ret = ret + (array(inv, dtype='int64'),)
        if return_counts:
            # Count occurrences of each unique
            count_map = {}
            for i in range(n_slices):
                sl = _get_slice(a, axis, i)
                key = _slice_key(sl)
                if key not in count_map:
                    count_map[key] = 0
                count_map[key] += 1
            counts_list = []
            for idx in unique_indices_list:
                sl = _get_slice(a, axis, idx)
                key = _slice_key(sl)
                counts_list.append(count_map[key])
            ret = ret + (array(counts_list, dtype='int64'),)
        return ret
    flat = a.flatten()
    n = flat.shape[0]
    vals = flat.tolist()

    # If dtype is complex, convert (re, im) tuples from tolist() to Python complex
    _dt = str(a.dtype)
    if 'complex' in _dt:
        new_vals = []
        for v in vals:
            if isinstance(v, tuple) and len(v) == 2:
                new_vals.append(complex(v[0], v[1]))
            elif isinstance(v, complex):
                new_vals.append(v)
            else:
                new_vals.append(complex(v))
        vals = new_vals

    # Sort key that handles complex, tuples, and mixed types
    def _sort_key(t):
        v = t[1]
        if isinstance(v, complex):
            return (v.real, v.imag)
        if isinstance(v, tuple):
            return v
        try:
            return (v,)
        except TypeError:
            return (str(v),)

    # Build sorted unique with tracking info
    try:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
    except TypeError:
        indexed = sorted(enumerate(vals), key=_sort_key)
    unique_vals = []
    first_indices = []
    counts = []
    _sentinel = object()
    prev = _sentinel
    for orig_idx, v in indexed:
        if prev is _sentinel or v != prev:
            unique_vals.append(v)
            first_indices.append(orig_idx)
            counts.append(1)
            prev = v
        else:
            counts[-1] += 1
            # Keep the smallest original index
            if orig_idx < first_indices[-1]:
                first_indices[-1] = orig_idx

    result_unique = array(unique_vals, dtype=a.dtype)
    extras = return_index or return_inverse or return_counts
    if not extras:
        return result_unique
    ret = (result_unique,)
    if return_index:
        ret = ret + (array(first_indices, dtype='int64'),)
    if return_inverse:
        # For each element in the original flat array, find its position in unique_vals
        val_to_pos = {}
        for i, v in enumerate(unique_vals):
            val_to_pos[v] = i
        inverse = [val_to_pos[v] for v in vals]
        ret = ret + (array(inverse, dtype='int64'),)
    if return_counts:
        ret = ret + (array(counts, dtype='int64'),)
    return ret


def broadcast_to(arr, shape):
    """Broadcast an array to a new shape using reshape + tile."""
    arr = asarray(arr)
    arr_shape = arr.shape
    if arr_shape == tuple(shape):
        return arr
    ndim = len(shape)
    arr_ndim = len(arr_shape)
    # Prepend 1s to make same ndim
    if arr_ndim < ndim:
        new_shape = [1] * (ndim - arr_ndim) + list(arr_shape)
        arr = arr.reshape(new_shape)
        arr_shape = tuple(new_shape)
    # Check compatibility and compute reps
    reps = []
    for i in range(ndim):
        if arr_shape[i] == shape[i]:
            reps.append(1)
        elif arr_shape[i] == 1:
            reps.append(shape[i])
        else:
            raise ValueError(f"cannot broadcast shape {arr_shape} to {tuple(shape)}")
    return tile(arr, reps)


def extract(condition, arr):
    """Return elements of arr where condition is True."""
    condition = asarray(condition).flatten()
    arr = asarray(arr).flatten()
    result = []
    for i in range(len(arr)):
        if float(condition[i]) != 0.0:
            result.append(float(arr[i]))
    if not result:
        return array([])
    return array(result)


def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted."""
    arr = asarray(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    n = arr.shape[axis]
    # Normalize obj to a list of indices
    if isinstance(obj, int):
        indices_to_del = [obj if obj >= 0 else n + obj]
    elif isinstance(obj, (list, tuple)):
        indices_to_del = [i if i >= 0 else n + i for i in obj]
    elif isinstance(obj, ndarray):
        indices_to_del = [int(obj[i]) for i in range(len(obj))]
        indices_to_del = [i if i >= 0 else n + i for i in indices_to_del]
    else:
        indices_to_del = [int(obj)]
    del_set = set(indices_to_del)
    keep = [i for i in range(n) if i not in del_set]
    if not keep:
        # All deleted - return empty
        new_shape = list(arr.shape)
        new_shape[axis] = 0
        return array([])
    # Build result by selecting kept indices along axis
    if arr.ndim == 1:
        result = [float(arr[i]) for i in keep]
        return array(result)
    else:
        if axis == 0:
            # For multi-dimensional, concatenate slices along axis 0
            slices = []
            for i in keep:
                slices.append(arr[i])
            if slices:
                rows = []
                for s in slices:
                    if s.ndim == 0:
                        rows.append([float(s)])
                    else:
                        rows.append([float(s[j]) for j in range(len(s))])
                return array(rows)
            return array([])
        else:
            # For non-zero axis: transpose so target axis is first,
            # delete along axis 0, then transpose back
            # Build axis permutation: move target axis to front
            ndim = arr.ndim
            perm = [axis] + [i for i in range(ndim) if i != axis]
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            t = transpose(arr, perm)
            t_del = delete(t, obj, axis=0)
            return transpose(t_del, inv_perm)


def insert(arr, obj, values, axis=None):
    """Insert values along the given axis before the given indices."""
    arr = asarray(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    n = arr.shape[axis]
    idx = obj if isinstance(obj, int) else int(obj)
    if idx < 0:
        idx = n + idx
    if arr.ndim == 1:
        result = []
        if isinstance(values, (int, float)):
            values = [values]
        elif isinstance(values, ndarray):
            values = [float(values[i]) for i in range(len(values))]
        for i in range(n):
            if i == idx:
                for v in values:
                    result.append(float(v))
            result.append(float(arr[i]))
        if idx >= n:
            for v in values:
                result.append(float(v))
        return array(result)
    # Multi-dimensional: transpose target axis to front, insert along axis 0, transpose back
    ndims = arr.ndim
    if axis < 0:
        axis = ndims + axis
    # Build permutation to move target axis to position 0
    perm = [axis] + [i for i in range(ndims) if i != axis]
    inv_perm = [0] * ndims
    for i, p in enumerate(perm):
        inv_perm[p] = i
    t = transpose(arr, perm)
    # t now has target axis as axis 0; shape is (n, ...)
    values = asarray(values)
    # Build slices for before, inserted, and after
    before_slices = []
    for i in range(idx):
        before_slices.append(t[i])
    after_slices = []
    for i in range(idx, t.shape[0]):
        after_slices.append(t[i])
    # Reshape values to match the sub-array shape if needed
    sub_shape = t.shape[1:] if ndims > 1 else ()
    if values.ndim == 0 or (values.ndim == 1 and len(sub_shape) == 1 and values.shape[0] == sub_shape[0]):
        # values is a single row to insert
        vals_row = values.reshape(list(sub_shape)) if sub_shape else values
        inserted = [vals_row]
    else:
        inserted = [values[i] for i in range(values.shape[0])]
    all_rows = before_slices + inserted + after_slices
    # Stack rows manually: build flat list
    row_size = 1
    for s in sub_shape:
        row_size = row_size * s
    flat = []
    for row in all_rows:
        r = asarray(row).flatten()
        for j in range(row_size):
            flat.append(float(r[j]))
    new_shape = list(t.shape)
    new_shape[0] = len(all_rows)
    result = array(flat).reshape(new_shape)
    return transpose(result, inv_perm)


def select(condlist, choicelist, default=0):
    """Return array drawn from elements in choicelist, depending on conditions."""
    if len(condlist) != len(choicelist):
        raise ValueError("condlist and choicelist must be the same length")
    condlist = [asarray(c) for c in condlist]
    choicelist = [asarray(c) for c in choicelist]
    # Determine output shape from first array
    shape = condlist[0].shape
    n = condlist[0].size
    result = [float(default)] * n
    # Process in reverse order so first matching condition wins
    for i in range(len(condlist) - 1, -1, -1):
        cond = condlist[i].flatten()
        choice = choicelist[i].flatten()
        for j in range(n):
            if float(cond[j]) != 0.0:
                result[j] = float(choice[j])
    result_arr = array(result)
    if len(shape) > 1:
        result_arr = result_arr.reshape(list(shape))
    return result_arr


def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function."""
    x = asarray(x)
    n = x.size
    flat_x = x.flatten()
    result = [0.0] * n

    # funclist should have len(condlist) or len(condlist)+1 entries
    # If len(condlist)+1, the last entry is the "otherwise" value
    has_otherwise = len(funclist) == len(condlist) + 1

    for i in range(n):
        val = float(flat_x[i])
        matched = False
        for j, cond in enumerate(condlist):
            c = asarray(cond).flatten()
            if float(c[i]) != 0.0:
                if callable(funclist[j]):
                    result[i] = float(funclist[j](val))
                else:
                    result[i] = float(funclist[j])
                matched = True
                break
        if not matched and has_otherwise:
            if callable(funclist[-1]):
                result[i] = float(funclist[-1](val))
            else:
                result[i] = float(funclist[-1])

    result_arr = array(result)
    if len(x.shape) > 1:
        result_arr = result_arr.reshape(list(x.shape))
    return result_arr


def lexsort(keys):
    """Perform an indirect stable sort using a sequence of keys.
    Last key is used as the primary sort key."""
    keys_list = [asarray(k) for k in keys]
    n = keys_list[0].size
    indices = list(range(n))
    # Flatten all keys
    flat_keys = []
    for k in keys_list:
        fk = k.flatten()
        flat_keys.append([fk[i] for i in range(n)])
    # Sort indices using keys (last key = primary, first key = least significant)
    # Python sort is stable, so we sort from least significant to most significant
    for ki in range(len(flat_keys)):
        vals = flat_keys[ki]
        indices.sort(key=lambda idx, v=vals: v[idx])
    return array(indices)


def partition(a, kth, axis=-1):
    """Return a partitioned copy of an array.
    Creates a copy where the element at kth position is where it would be in a sorted array.
    Elements before kth are <= element at kth, elements after are >=."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        flat = a.flatten()
        n = flat.size
        vals = [flat[i] for i in range(n)]
        vals.sort()
        return array(vals)
    # For multi-dim, sort along axis (full sort, which satisfies partition contract)
    return sort(a, axis=axis)


def argpartition(a, kth, axis=-1):
    """Return indices that would partition the array.
    Same as argsort but only guarantees kth element is correct."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        return argsort(a)
    return argsort(a, axis=axis)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function to 1-D slices of an array along the given axis."""
    arr = asarray(arr)
    if arr.ndim == 1:
        return asarray(func1d(arr, *args, **kwargs))
    nd = arr.ndim
    if axis < 0:
        axis = nd + axis
    # General nD implementation
    # Build the shape of the non-target axes (the "outer" iteration shape)
    shape = arr.shape
    out_shape = tuple(shape[i] for i in range(nd) if i != axis)
    # Compute total number of outer iterations
    n_outer = 1
    for s in out_shape:
        n_outer *= s
    # For each combination of indices in the non-target axes, extract the 1D slice
    results = []
    for flat_idx in range(n_outer):
        # Convert flat_idx to multi-index in out_shape
        idx = []
        rem = flat_idx
        for s in reversed(out_shape):
            idx.append(rem % s)
            rem = rem // s
        idx.reverse()
        # Build the full index with a slice at the target axis position
        # Extract the 1D slice along the target axis
        # We need to index arr with idx inserted around the target axis
        outer_idx_pos = 0
        slice_vals = []
        for k in range(shape[axis]):
            # Build index tuple for element [idx[0], ..., k, ..., idx[-1]]
            full_idx = []
            oi = 0
            for dim in range(nd):
                if dim == axis:
                    full_idx.append(k)
                else:
                    full_idx.append(idx[oi])
                    oi += 1
            # Navigate to element
            elem = arr
            for fi in full_idx:
                elem = elem[fi]
            slice_vals.append(float(elem))
        slice_arr = array(slice_vals)
        result = func1d(slice_arr, *args, **kwargs)
        results.append(result)
    # Reshape results back to out_shape
    # If func1d returns scalar, result shape is out_shape
    first_res = results[0]
    if isinstance(first_res, ndarray):
        # func returns array - more complex, but handle scalar reduction case
        result_arr = asarray(results)
        return result_arr.reshape(out_shape)
    else:
        return array([float(r) for r in results]).reshape(out_shape)


class vectorize:
    """Generalized function class.

    Takes a nested sequence of objects or numpy arrays as inputs and returns
    a single numpy array or a tuple of numpy arrays by applying the function
    element-by-element.
    """
    def __init__(self, pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = otypes
        if doc is not None:
            self.__doc__ = doc
        elif pyfunc.__doc__:
            self.__doc__ = pyfunc.__doc__

    def __call__(self, *args, **kwargs):
        # Convert all args to arrays
        arr_args = [asarray(a) for a in args]
        if len(arr_args) == 0:
            return array([])
        # Get the size from first arg
        first = arr_args[0].flatten()
        n = first.size
        results = []
        for i in range(n):
            elem_args = []
            for a in arr_args:
                flat = a.flatten()
                if flat.size == 1:
                    elem_args.append(flat[0])
                else:
                    elem_args.append(flat[i])
            results.append(self.pyfunc(*elem_args, **kwargs))
        result = array(results)
        # Try to reshape to the shape of first arg
        if arr_args[0].ndim > 1:
            result = result.reshape(arr_args[0].shape)
        return result


def put(a, ind, v, mode='raise'):
    """Replaces specified elements of an array with given values.

    Modifies 'a' in-place and returns None.
    """
    from ._helpers import _ObjectArray
    if not isinstance(a, (ndarray, _ObjectArray)):
        raise TypeError("argument 1 must be numpy.ndarray")
    ind_arr = asarray(ind).flatten()
    v_arr = asarray(v).flatten()
    n = a.size
    ni = ind_arr.size
    nv = v_arr.size
    if nv == 0:
        # Empty values: no-op (numpy behavior)
        return None
    for idx in range(ni):
        i = int(ind_arr[idx])
        if mode == 'wrap':
            if n == 0:
                return None
            i = i % n
        elif mode == 'clip':
            if n == 0:
                return None
            if i < 0:
                i = 0
            elif i >= n:
                i = n - 1
        elif mode == 'raise':
            if i < 0:
                i = n + i
            if i < 0 or i >= n:
                raise IndexError("index {} is out of bounds for axis 0 with size {}".format(
                    int(ind_arr[idx]), n))
        if isinstance(a, _ObjectArray):
            a._data[i] = v_arr[idx % nv] if isinstance(v_arr, _ObjectArray) else float(v_arr[idx % nv])
        else:
            a.flat[i] = float(v_arr[idx % nv])
    return None


def putmask(a, mask, values):
    """Changes elements of an array based on conditional and input values.

    Modifies 'a' in-place and returns None.
    """
    mask = asarray(mask)
    values = asarray(values)
    flat_m = mask.flatten()
    flat_v = values.flatten()
    n = a.size
    nv = flat_v.size
    if nv == 0:
        return None
    vi = 0
    for i in range(n):
        if flat_m[i]:
            a.flat[i] = float(flat_v[vi % nv])
            vi += 1
    return None


def broadcast_arrays(*args):
    """Broadcast any number of arrays against each other."""
    arrays = [asarray(a) for a in args]
    if len(arrays) == 0:
        return []
    # Find common shape
    shape = list(arrays[0].shape)
    for a in arrays[1:]:
        ashape = list(a.shape)
        # Pad shorter shape with 1s on left
        while len(shape) < len(ashape):
            shape.insert(0, 1)
        while len(ashape) < len(shape):
            ashape.insert(0, 1)
        new_shape = []
        for s1, s2 in zip(shape, ashape):
            if s1 == s2:
                new_shape.append(s1)
            elif s1 == 1:
                new_shape.append(s2)
            elif s2 == 1:
                new_shape.append(s1)
            else:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        shape = new_shape
    return [broadcast_to(a, tuple(shape)) for a in arrays]


def apply_over_axes(func, a, axes):
    """Apply a function repeatedly over multiple axes."""
    a = asarray(a)
    if isinstance(axes, int):
        axes = [axes]
    for ax in axes:
        result = func(a, axis=ax)
        if isinstance(result, ndarray):
            a = result
        else:
            a = asarray(result)
    return a


def broadcast_shapes(*shapes):
    """Compute the broadcast result shape from multiple shapes."""
    if not shapes:
        return ()
    ndim = _builtin_max(len(s) for s in shapes)
    result = [1] * ndim
    for shape in shapes:
        offset = ndim - len(shape)
        for i, dim in enumerate(shape):
            j = i + offset
            if result[j] == 1:
                result[j] = dim
            elif dim != 1 and dim != result[j]:
                raise ValueError(
                    f"shape mismatch: objects cannot be broadcast to a single shape. Mismatch at dimension {j}"
                )
    return tuple(result)


def trim_zeros(filt, trim='fb'):
    """Trim leading and/or trailing zeros from a 1-D array."""
    filt = asarray(filt)
    data = filt.tolist()
    first = 0
    last = len(data)
    if 'f' in trim or 'F' in trim:
        for i in _builtin_range(len(data)):
            if data[i] != 0:
                first = i
                break
        else:
            return array([])
    if 'b' in trim or 'B' in trim:
        for i in _builtin_range(len(data) - 1, -1, -1):
            if data[i] != 0:
                last = i + 1
                break
        else:
            return array([])
    return array(data[first:last])


# ---------------------------------------------------------------------------
# pad
# ---------------------------------------------------------------------------

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

    # Handle callable mode (legacy vector functionality)
    if callable(mode):
        return _pad_callable(a, pad_width, mode, kwargs)

    # Validate pad_width contains only integral types
    _validate_pad_width_type(pad_width)

    # Normalise pad_width to array of shape (ndim, 2)
    pw = _normalize_pad_width(pad_width, a.ndim)

    # Check for empty axes that need padding with non-constant/empty modes
    if mode not in ('constant', 'empty'):
        for ax in range(a.ndim):
            if a.shape[ax] == 0 and (pw[ax][0] > 0 or pw[ax][1] > 0):
                raise ValueError(
                    "can't extend empty axis %d using modes other than "
                    "'constant' or 'empty'" % ax
                )

    # Shortcut: no padding needed
    total_pad = sum(pw[ax][0] + pw[ax][1] for ax in range(a.ndim))
    if total_pad == 0:
        return a.copy() if hasattr(a, 'copy') else array(a)

    if mode == 'constant':
        return _pad_constant(a, pw, kwargs.get('constant_values', 0))
    elif mode == 'edge':
        return _pad_edge(a, pw)
    elif mode == 'linear_ramp':
        return _pad_linear_ramp(a, pw, kwargs.get('end_values', 0))
    elif mode == 'reflect':
        return _pad_reflect(a, pw, kwargs.get('reflect_type', 'even'))
    elif mode == 'symmetric':
        return _pad_symmetric(a, pw, kwargs.get('reflect_type', 'even'))
    elif mode == 'wrap':
        return _pad_wrap(a, pw)
    elif mode in ('maximum', 'minimum', 'mean', 'median'):
        return _pad_stat(a, pw, mode, kwargs.get('stat_length', None))
    elif mode == 'empty':
        return _pad_empty(a, pw)
    else:
        raise ValueError("Unknown padding mode: %s" % mode)


def _normalize_pad_width(pad_width, ndim):
    """Normalize pad_width to list of (before, after) tuples, one per axis."""
    import math

    def _to_int(x):
        return int(math.floor(float(x) + 0.5)) if isinstance(x, float) else int(x)

    if isinstance(pad_width, (int, float)):
        v = _to_int(pad_width)
        return [(v, v)] * ndim

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
                        if len(p) == 1:
                            v = _to_int(p[0])
                            result.append((v, v))
                        elif len(p) == 2:
                            result.append((_to_int(p[0]), _to_int(p[1])))
                        else:
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
            before_arr = ones(tuple(before_shape), dtype=a.dtype) * asarray(before_val).astype(a.dtype)
            result = concatenate([before_arr, result], axis=ax)

        if after > 0:
            after_shape = list(result.shape)
            after_shape[ax] = after
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
            parts.append(tile(edge, reps))

        parts.append(result)

        if after > 0:
            slices = [slice(None)] * result.ndim
            slices[ax] = slice(n - 1, n)
            edge = result[tuple(slices)]
            reps = [1] * result.ndim
            reps[ax] = after
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
                # odd reflect: 2*edge - reflected value
                slices = [slice(None)] * result.ndim
                slices[ax] = slice(0, 1)
                edge = result[tuple(slices)]
                reps = [1] * result.ndim
                reps[ax] = before
                edge_broadcast = tile(edge, reps)
                before_arr = edge_broadcast * 2 - before_arr
            parts.append(before_arr)

        parts.append(result)

        if after > 0:
            indices = []
            for i in range(1, after + 1):
                idx = n - 1 - _reflect_index(i, n)
                indices.append(idx)
            after_arr = _take_along_axis(result, indices, ax)
            if reflect_type == 'odd':
                slices = [slice(None)] * result.ndim
                slices[ax] = slice(n - 1, n)
                edge = result[tuple(slices)]
                reps = [1] * result.ndim
                reps[ax] = after
                edge_broadcast = tile(edge, reps)
                after_arr = edge_broadcast * 2 - after_arr
            parts.append(after_arr)

        result = concatenate(parts, axis=ax)

    return result


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
                # symmetric: index 1 -> element 0, index 2 -> element 1, etc.
                # with wrapping via period 2*n
                idx = (i - 1) % (2 * n)
                if idx >= n:
                    idx = 2 * n - 1 - idx
                indices.append(idx)
            before_arr = _take_along_axis(result, indices, ax)
            if reflect_type == 'odd':
                slices = [slice(None)] * result.ndim
                slices[ax] = slice(0, 1)
                edge = result[tuple(slices)]
                reps = [1] * result.ndim
                reps[ax] = before
                edge_broadcast = tile(edge, reps)
                before_arr = edge_broadcast * 2 - before_arr
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
                slices = [slice(None)] * result.ndim
                slices[ax] = slice(n - 1, n)
                edge = result[tuple(slices)]
                reps = [1] * result.ndim
                reps[ax] = after
                edge_broadcast = tile(edge, reps)
                after_arr = edge_broadcast * 2 - after_arr
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
    # Guard against empty chunks
    if chunk.shape[axis] == 0:
        shape = list(chunk.shape)
        shape[axis] = 1
        return np.full(tuple(shape), float('nan'))
    if mode == 'mean':
        return np.mean(chunk, axis=axis, keepdims=True)
    elif mode == 'median':
        return np.median(chunk, axis=axis, keepdims=True)
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
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(xp, ndarray):
        xp = array(xp)
    if not isinstance(fp, ndarray):
        fp = array(fp)
    result = _nat.interp(x, xp, fp)
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


# ---------------------------------------------------------------------------
# bincount
# ---------------------------------------------------------------------------

def bincount(x, weights=None, minlength=0):
    import _numpy_native as _nat
    if not isinstance(x, ndarray):
        x = array(x)
    if weights is not None and not isinstance(weights, ndarray):
        weights = array(weights)
    return _nat.bincount(x, weights, minlength)
