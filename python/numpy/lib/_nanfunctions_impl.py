"""numpy.lib._nanfunctions_impl - nan helper functions."""
import numpy as np


def _nan_mask(a):
    """Return a boolean mask where True indicates NaN values."""
    a = np.asarray(a)
    try:
        return np.isnan(a)
    except (TypeError, ValueError):
        return np.zeros(a.shape, dtype='bool') if hasattr(a, 'shape') else False


def _replace_nan(a, val):
    """Replace NaN values with val, return (result, mask)."""
    a = np.asarray(a)
    mask = _nan_mask(a)
    # Return a copy with NaN replaced
    result = a.copy()
    flat = result.flatten()
    mask_flat = mask.flatten() if hasattr(mask, 'flatten') else [mask]
    out_list = []
    for i in range(flat.size):
        v = float(flat[i])
        if i < len(mask_flat) and (mask_flat[i] if hasattr(mask_flat, '__getitem__') else mask_flat):
            out_list.append(float(val))
        elif v != v:  # NaN check
            out_list.append(float(val))
        else:
            out_list.append(v)
    return np.array(out_list).reshape(a.shape), mask


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._nanfunctions_impl' has no attribute {name!r}")
