"""numpy.lib._arraypad_impl - array padding helpers."""


def _as_pairs(x, ndim, as_index=False):
    """Broadcast x to shape (ndim, 2)."""
    if isinstance(x, (int, float)):
        return [(int(x), int(x))] * ndim
    if hasattr(x, '__len__'):
        if len(x) == 2 and not hasattr(x[0], '__len__'):
            return [(int(x[0]), int(x[1]))] * ndim
        result = []
        for item in x:
            if hasattr(item, '__len__'):
                result.append((int(item[0]), int(item[1])))
            else:
                result.append((int(item), int(item)))
        return result
    return [(int(x), int(x))] * ndim


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._arraypad_impl' has no attribute {name!r}")
