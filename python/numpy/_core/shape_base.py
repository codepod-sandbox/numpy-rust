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
        # Leaf: a single array – promote to result_ndim
        arr = numpy.asarray(arrays)
        while arr.ndim < result_ndim:
            arr = numpy.expand_dims(arr, 0)
        return arr

    # Recurse into sub-lists and concatenate along the appropriate axis
    arrs = [_block_concatenate(a, list_ndim - 1, result_ndim) for a in arrays]
    # The concatenation axis: the outermost nesting level maps to axis
    # result_ndim - list_ndim.  E.g. for list_ndim=2, result_ndim=2
    # the outer level concatenates along axis 0.
    axis = result_ndim - list_ndim
    return numpy.concatenate(arrs, axis=axis)


def _block_dispatcher(arrays, depth=0):
    """Dispatcher for np.block (stub)."""
    return arrays


def _block_setup(arrays):
    """Setup function for np.block.

    Returns (nested_arrays, list_ndim, result_ndim, total_size) where
    *nested_arrays* mirrors the input structure but with every leaf
    replaced by an ndarray.
    """
    max_ndim = [0]
    total_size = [0]

    def _depth(lst):
        """Return the nesting depth (number of list layers)."""
        if isinstance(lst, list):
            if len(lst) == 0:
                return 1
            return 1 + _depth(lst[0])
        return 0

    def _convert(lst):
        if isinstance(lst, list):
            return [_convert(item) for item in lst]
        arr = numpy.asarray(lst)
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
