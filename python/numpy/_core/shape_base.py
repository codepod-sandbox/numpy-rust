"""numpy._core.shape_base - array shape manipulation."""
import numpy

stack = numpy.stack
vstack = numpy.vstack
hstack = numpy.hstack
concatenate = numpy.concatenate


def _block_concatenate(arrays, list_ndim, result_ndim):
    """Internal helper for np.block (stub)."""
    raise NotImplementedError("_block_concatenate is not implemented")


def _block_dispatcher(arrays, depth=0):
    """Dispatcher for np.block (stub)."""
    return arrays


def _block_setup(arrays):
    """Setup function for np.block - returns (list_of_arrays, list_ndim, result_ndim)."""
    def _flatten(lst):
        flat = []
        depth = 0
        def _recurse(l, d):
            nonlocal depth
            if isinstance(l, list):
                if d > depth:
                    depth = d
                for item in l:
                    _recurse(item, d + 1)
            else:
                flat.append(l)
        _recurse(lst, 1)
        return flat, depth

    flat, list_ndim = _flatten(arrays)
    result = []
    max_ndim = 0
    for a in flat:
        arr = numpy.asarray(a)
        result.append(arr)
        if arr.ndim > max_ndim:
            max_ndim = arr.ndim
    result_ndim = max(max_ndim, list_ndim)
    return result, list_ndim, result_ndim


def _block_slicing(arrays, list_ndim, result_ndim):
    """Internal helper for np.block using slicing (stub)."""
    raise NotImplementedError("_block_slicing is not implemented")


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
