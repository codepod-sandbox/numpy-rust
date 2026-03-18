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
    a = asarray(x) if not isinstance(x, ndarray) else x
    # For boolean arrays, invert means logical not
    if str(a.dtype) == 'bool':
        r = _native.logical_not(a)
    else:
        r = _native.bitwise_not(_to_int(a))
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

def _to_bool(x):
    """Convert to bool array via element-wise Python truthiness."""
    a = asarray(x) if not isinstance(x, ndarray) else x
    try:
        return a.astype('bool')
    except Exception:
        flat = [bool(v) for v in a.flatten().tolist()]
        return array(flat, dtype='bool').reshape(a.shape)

def logical_and(x1, x2, out=None, **kwargs):
    r = _native.logical_and(_to_bool(x1), _to_bool(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_or(x1, x2, out=None, **kwargs):
    r = _native.logical_or(_to_bool(x1), _to_bool(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_xor(x1, x2, out=None, **kwargs):
    r = _native.logical_xor(_to_bool(x1), _to_bool(x2))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_not(x, out=None, **kwargs):
    r = _native.logical_not(_to_bool(x))
    if out is not None:
        _copy_into(out, r)
        return out
    return r
