"""Bitwise and logical operations, delegating to Rust native implementations."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._creation import asarray, array
from ._helpers import _copy_into

__all__ = [
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not', 'invert',
    'bitwise_count',
    'left_shift', 'right_shift',
    'logical_and', 'logical_or', 'logical_xor', 'logical_not',
]

# --- Bitwise operations ------------------------------------------------------

def _to_int(x):
    """Ensure array is int64 for bitwise operations."""
    a = asarray(x) if not isinstance(x, ndarray) else x
    return a.astype("int64")

def bitwise_and(x1, x2, out=None, **kwargs):
    """Element-wise bitwise AND of integer arrays."""
    r = _native.bitwise_and(_to_int(x1), _to_int(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def bitwise_or(x1, x2, out=None, **kwargs):
    """Element-wise bitwise OR of integer arrays."""
    r = _native.bitwise_or(_to_int(x1), _to_int(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def bitwise_xor(x1, x2, out=None, **kwargs):
    """Element-wise bitwise XOR of integer arrays."""
    r = _native.bitwise_xor(_to_int(x1), _to_int(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def bitwise_not(x, out=None, **kwargs):
    """Element-wise bitwise NOT (invert) of integer array."""
    r = _native.bitwise_not(_to_int(x))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

invert = bitwise_not

def bitwise_count(x, out=None, **kwargs):
    """Element-wise count of 1-bits (population count / popcount)."""
    a = asarray(x) if not isinstance(x, ndarray) else x
    # int(v) & mask handles negative integers (count bits in unsigned repr)
    flat = [bin(int(v) & 0xFFFFFFFFFFFFFFFF).count('1')
            for v in a.flatten().tolist()]
    r = array(flat, dtype='uint8').reshape(a.shape)
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def left_shift(x1, x2, out=None, **kwargs):
    """Element-wise left bit shift."""
    r = _native.left_shift(_to_int(x1), _to_int(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def right_shift(x1, x2, out=None, **kwargs):
    """Element-wise right bit shift."""
    r = _native.right_shift(_to_int(x1), _to_int(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

# --- Logical operations ------------------------------------------------------

def logical_and(x1, x2, out=None, **kwargs):
    a = asarray(x1) if not isinstance(x1, ndarray) else x1
    b = asarray(x2) if not isinstance(x2, ndarray) else x2
    r = _native.logical_and(a, b)
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_or(x1, x2, out=None, **kwargs):
    a = asarray(x1) if not isinstance(x1, ndarray) else x1
    b = asarray(x2) if not isinstance(x2, ndarray) else x2
    r = _native.logical_or(a, b)
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_xor(x1, x2, out=None, **kwargs):
    a = asarray(x1) if not isinstance(x1, ndarray) else x1
    b = asarray(x2) if not isinstance(x2, ndarray) else x2
    r = _native.logical_xor(a, b)
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_not(x, out=None, **kwargs):
    if isinstance(x, ndarray):
        r = _native.logical_not(x)
    else:
        r = asarray(x)
        r = _native.logical_not(r)
    if out is not None:
        _copy_into(out, r)
        return out
    return r
