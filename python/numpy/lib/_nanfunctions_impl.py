"""numpy.lib._nanfunctions_impl - nan helper functions."""
import numpy as np

__all__ = [
    'nanmin', 'nanmax', 'nanargmin', 'nanargmax',
    'nansum', 'nanprod', 'nancumsum', 'nancumprod',
    'nanmean', 'nanmedian', 'nanpercentile', 'nanquantile',
    'nanvar', 'nanstd',
]


def _nan_mask(a, out=None):
    """Return a boolean mask where True indicates non-NaN values.

    Returns True (scalar) for types that cannot contain NaN (int, bool).
    Returns an ndarray of bool for float/complex types.
    If out is provided (and the result is an array), writes the result into out.
    """
    a = np.asarray(a)
    # Integer and bool types cannot contain NaN — return scalar True
    if not np.issubdtype(a.dtype, np.inexact):
        return True
    try:
        mask = ~np.isnan(a)
    except (TypeError, ValueError):
        mask = np.ones(a.shape, dtype=bool)
    if out is not None:
        out[...] = mask
        return out
    return mask


def _replace_nan(a, val):
    """Replace NaN values with val, return (result, mask).

    For integer and bool dtypes (which cannot contain NaN), returns
    (a, None) with no copy — the original array is returned unchanged.
    For float/complex dtypes, always returns (copy, mask_array) where
    mask_array is a bool array of NaN positions (True where NaN).
    """
    a = np.asarray(a)
    # Types that cannot contain NaN: return original, no mask
    if not np.issubdtype(a.dtype, np.inexact):
        return a, None
    mask = np.isnan(a)
    result = a.copy()
    if mask.any():
        result[mask] = val
    return result, mask


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._nanfunctions_impl' has no attribute {name!r}")
