"""numpy.polynomial.polyutils - utility functions."""
import numpy as np
from numpy.exceptions import RankWarning


def trimseq(seq):
    """Remove trailing zeros from a sequence."""
    if len(seq) == 0:
        return seq
    seq = list(seq)
    while len(seq) > 1 and seq[-1] == 0:
        seq.pop()
    return seq


def trimcoef(c, tol=0):
    """Remove trailing coefficients smaller than tol."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    c = np.asarray(c)
    c_list = list(c.flatten().tolist())
    while len(c_list) > 1 and abs(c_list[-1]) <= tol:
        c_list.pop()
    # If only one element left and it's also <= tol, replace with 0
    if len(c_list) == 1 and abs(c_list[0]) <= tol and tol > 0:
        c_list = [0]
    return np.array(c_list)


def as_series(alist, trim=True):
    """Convert list of array-likes to list of 1-D arrays."""
    arrays = []
    for a in alist:
        a = np.asarray(a)
        if a.ndim == 0:
            raise ValueError("Coefficient array is not 1-d")
        if a.ndim > 1:
            raise ValueError("Coefficient array is not 1-d")
        if a.size == 0:
            raise ValueError("Coefficient array is not 1-d")
        arrays.append(a)
    # Check for mixed types that can't be promoted
    dtypes = [a.dtype for a in arrays]
    # Try to find common dtype
    result = []
    for a in arrays:
        a = a.flatten()
        if trim:
            a_list = trimseq(list(a.tolist()))
            a = np.array(a_list) if a_list else np.array([0.0])
        result.append(a)
    return result


def getdomain(x):
    """Return a domain suitable for the given data points."""
    x = np.asarray(x)
    flat = x.flatten()
    vals = flat.tolist()
    if vals and isinstance(vals[0], complex):
        mn = complex(min(v.real for v in vals), min(v.imag for v in vals))
        mx = complex(max(v.real for v in vals), max(v.imag for v in vals))
        return np.array([mn, mx])
    vals = [float(v) for v in vals]
    return np.array([min(vals), max(vals)])


def mapdomain(x, old, new):
    """Apply linear map to input points x from domain old to domain new."""
    x = np.asarray(x)
    old = list(np.asarray(old).flatten().tolist())
    new = list(np.asarray(new).flatten().tolist())
    off, scl = mapparms(old, new)
    return x * scl + off


def mapparms(old, new):
    """Return offset and scale for linear map between domains."""
    old = list(np.asarray(old).flatten().tolist())
    new = list(np.asarray(new).flatten().tolist())
    old_len = old[1] - old[0]
    new_len = new[1] - new[0]
    if old_len == 0:
        scl = 0.0
    else:
        scl = new_len / old_len
    off = new[0] - old[0] * scl
    return off, scl
