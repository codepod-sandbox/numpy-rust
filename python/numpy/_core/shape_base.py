"""numpy._core.shape_base - array shape manipulation."""
import numpy

stack = numpy.stack
vstack = numpy.vstack
hstack = numpy.hstack
concatenate = numpy.concatenate


def _block_concatenate(arrays, list_ndim, result_ndim):
    """Internal helper for np.block using recursive concatenation.

    *arrays* is a (possibly nested) list whose leaves are ndarrays.
    *list_ndim* is the nesting depth of the original block specification.
    *result_ndim* is the target ndim (max of leaf ndims and list_ndim).
    """
    if list_ndim == 0:
        # Leaf: a single array – promote to result_ndim (and copy)
        arr = numpy.array(arrays, copy=True)  # always copy
        while arr.ndim < result_ndim:
            arr = numpy.expand_dims(arr, 0)
        return arr

    if len(arrays) == 0:
        raise ValueError("List of blocks for block() is empty")

    # Recurse into sub-lists and concatenate along the appropriate axis
    arrs = [_block_concatenate(a, list_ndim - 1, result_ndim) for a in arrays]
    # The concatenation axis: the outermost nesting level maps to axis
    # result_ndim - list_ndim.  E.g. for list_ndim=2, result_ndim=2
    # the outer level concatenates along axis 0.
    axis = result_ndim - list_ndim
    return numpy.concatenate(arrs, axis=axis)


def _block_dispatcher(arrays):
    """Dispatcher for np.block — yields leaf items from nested lists."""
    if isinstance(arrays, list):
        for item in arrays:
            yield from _block_dispatcher(item)
    else:
        yield arrays


def _block_setup(arrays):
    """Setup function for np.block.

    Returns (nested_arrays, list_ndim, result_ndim, total_size) where
    *nested_arrays* mirrors the input structure but with every leaf
    replaced by an ndarray.
    """
    if isinstance(arrays, tuple):
        raise TypeError("only lists are allowed, not tuple")

    max_ndim = [0]
    total_size = [0]

    def _depth(lst):
        """Return the nesting depth (number of list layers)."""
        if isinstance(lst, list):
            if len(lst) == 0:
                return 1
            return 1 + _depth(lst[0])
        return 0

    def _check_depth(lst, expected_depth, parent_depth=0):
        """Validate all elements at same nesting depth."""
        if not isinstance(lst, list):
            return
        depths = [_depth(item) for item in lst]
        if len(set(depths)) > 1:
            raise ValueError(
                "List depths are mismatched. First element was at depth {}, "
                "but there is an element at depth {}".format(
                    depths[0] + parent_depth,
                    [d for d in depths if d != depths[0]][0] + parent_depth))
        for item in lst:
            if isinstance(item, list):
                if len(item) == 0:
                    raise ValueError("List of blocks for block() is empty")
                _check_depth(item, expected_depth - 1, parent_depth + 1)

    def _check_tuples(lst):
        for item in lst:
            if isinstance(item, tuple):
                raise TypeError("only lists are allowed, not tuple")
            if isinstance(item, list):
                _check_tuples(item)

    if isinstance(arrays, list):
        if len(arrays) == 0:
            raise ValueError("List of blocks for block() is empty")
        _check_tuples(arrays)
        _check_depth(arrays, _depth(arrays))

    def _convert(lst):
        if isinstance(lst, list):
            return [_convert(item) for item in lst]
        arr = numpy.array(lst, copy=True)  # always copy
        if arr.ndim > max_ndim[0]:
            max_ndim[0] = arr.ndim
        total_size[0] += arr.size
        return arr

    list_ndim = _depth(arrays)
    nested = _convert(arrays)
    result_ndim = max(max_ndim[0], list_ndim)
    return nested, list_ndim, result_ndim, total_size[0]


def _block_slicing(arrays, list_ndim, result_ndim):
    """Internal helper for np.block using slicing.

    Uses the same recursive algorithm as _block_concatenate (the
    performance difference only matters for very large arrays in
    CPython/NumPy; here we just delegate).
    """
    return _block_concatenate(arrays, list_ndim, result_ndim)


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
