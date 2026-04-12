"""Joining and splitting arrays: stack, vstack, hstack, split, append, block, etc."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._core_types import dtype, _ScalarType, _normalize_dtype
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate, linspace

_builtin_divmod = divmod

__all__ = [
    'stack', 'vstack', 'hstack', 'column_stack', 'dstack', 'row_stack', 'unstack',
    'split', 'vsplit', 'hsplit', 'array_split', 'dsplit',
    'append', 'block',
]


def _require_array_sequence(arrays):
    """Require a concrete list/tuple of array-like values."""
    if hasattr(arrays, '__next__'):
        raise TypeError("arrays to stack must be passed as a list or tuple, not a generator")
    if not isinstance(arrays, (list, tuple)):
        raise TypeError("arrays to stack must be passed as a list or tuple, not " + type(arrays).__name__)
    return arrays


def _coerce_array_sequence(arrays, transform=None):
    """Coerce a stack-like input sequence through one shared array boundary."""
    seq = _require_array_sequence(arrays)
    coerced = []
    for arr in seq:
        value = asarray(arr)
        if transform is not None:
            value = transform(value)
        coerced.append(value)
    return coerced


def stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind'):
    from ._helpers import _ObjectArray
    from ._core_types import can_cast
    from ._shape import expand_dims
    # Cannot specify both out and dtype
    if out is not None and dtype is not None:
        raise TypeError("stack() does not support 'out' and 'dtype' together")
    arrays = _coerce_array_sequence(arrays)
    if not arrays:
        raise ValueError("need at least one array to stack")
    # Validate all arrays have the same shape
    ref_shape = arrays[0].shape
    for i, a in enumerate(arrays[1:], 1):
        if a.shape != ref_shape:
            raise ValueError(
                "all input arrays must have the same shape, but the array at "
                "index 0 has shape {} and the array at index {} has shape {}".format(
                    ref_shape, i, a.shape))
    # After stacking, ndim will be input ndim + 1
    ndim = arrays[0].ndim + 1
    orig_axis = axis
    # Normalize negative axis
    if axis < 0:
        axis = axis + ndim
        if axis < 0:
            raise AxisError(orig_axis, ndim)
    # Validate positive axis is in bounds (valid range: 0 <= axis <= ndim-1 = input.ndim)
    if axis >= ndim:
        raise AxisError(orig_axis, ndim)
    # Validate casting against target dtype
    target_dtype = None
    if out is not None:
        out_arr = asarray(out) if not isinstance(out, ndarray) else out
        target_dtype = str(out_arr.dtype)
    elif dtype is not None:
        target_dtype = str(dtype)
    if target_dtype is not None:
        for a in arrays:
            if not can_cast(a, target_dtype, casting=casting):
                raise TypeError(
                    "Cannot cast array data from dtype('{}') to dtype('{}') "
                    "according to the rule '{}'".format(a.dtype, target_dtype, casting))
    # If any array is _ObjectArray, use Python-level implementation
    if any(isinstance(a, _ObjectArray) for a in arrays):
        # Expand dims for each array, then concatenate
        expanded = [expand_dims(a, axis=axis) for a in arrays]
        result = concatenate(expanded, axis=axis)
    else:
        result = _native.stack_native(arrays, axis)
    if target_dtype is not None:
        result = result.astype(target_dtype)
    if out is not None:
        out_arr[:] = result
        return out_arr
    return result


def vstack(tup, *, dtype=None, casting='same_kind'):
    from ._shape import atleast_2d
    arrs = _coerce_array_sequence(tup, atleast_2d)
    if len(arrs) == 0:
        raise ValueError("need at least one array to concatenate")
    return concatenate(arrs, 0, dtype=dtype, casting=casting)


def hstack(tup, *, dtype=None, casting='same_kind'):
    from ._shape import atleast_1d
    arrs = _coerce_array_sequence(tup, atleast_1d)
    if len(arrs) == 0:
        raise ValueError("need at least one array to concatenate")
    if arrs[0].ndim > 1:
        return concatenate(arrs, 1, dtype=dtype, casting=casting)
    return concatenate(arrs, 0, dtype=dtype, casting=casting)


def column_stack(tup):
    """Stack 1-D arrays as columns into a 2-D array."""
    arrays = []
    for a in _coerce_array_sequence(tup):
        if a.ndim == 1:
            a = a.reshape((a.size, 1))  # Make column vector
        arrays.append(a)
    return concatenate(arrays, 1)


def dstack(tup):
    return _native.dstack(_coerce_array_sequence(tup))


row_stack = vstack


def unstack(x, /, *, axis=0):
    """Split an array into a tuple of arrays along an axis."""
    x = asarray(x)
    if x.ndim == 0:
        raise ValueError("unstack requires array to have at least 1 dimension")
    ndim = x.ndim
    orig_axis = axis
    if axis < 0:
        axis = axis + ndim
    if axis < 0 or axis >= ndim:
        raise ValueError("axis {} is out of bounds for array of dimension {}".format(
            orig_axis, ndim))
    n = x.shape[axis]
    return tuple(x[(slice(None),) * axis + (i,)] for i in range(n))


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


def append(arr, values, axis=None):
    """Append values to the end of an array."""
    if not isinstance(arr, ndarray):
        arr = array(arr)
    if not isinstance(values, ndarray):
        values = array(values)
    if axis is None:
        return concatenate([arr.flatten(), values.flatten()])
    return concatenate([arr, values], axis=axis)


def block(arrays):
    """Assemble an nd-array from nested lists of blocks.

    Follows NumPy's algorithm: determine list_ndim (nesting depth) and
    result_ndim = max(list_ndim, max leaf ndim), then promote all leaves
    to result_ndim dimensions and concatenate bottom-up.
    """
    from ._shape import atleast_1d, expand_dims
    if isinstance(arrays, ndarray):
        return arrays.copy()
    if isinstance(arrays, tuple):
        raise TypeError("only lists are allowed, not tuple")
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
