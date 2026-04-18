"""Bit manipulation functions for numpy-rust."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _flat_arraylike_data
from ._creation import array, asarray

__all__ = [
    'packbits', 'unpackbits',
    'binary_repr', 'base_repr',
]

def _reverse_unpackbits_blocks(bits, count=None):
    vals = _flat_arraylike_data(bits)
    out = []
    for i in range(0, len(vals), 8):
        out.extend(reversed(vals[i:i + 8]))
    if count is not None:
        if count >= 0:
            out = out[:count]
        else:
            out = out[:len(out) + count]
    return array(out)


# ---------------------------------------------------------------------------
# Bit manipulation
# ---------------------------------------------------------------------------

def packbits(a, axis=None, bitorder='big'):
    """Pack a binary-valued array into uint8."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'little' or 'big'")
    if axis is not None:
        axis = int(axis)
        if axis < 0:
            axis += a.ndim
        if axis < 0 or axis >= a.ndim:
            raise ValueError("axis out of range")
    if axis is None:
        flat = a.flatten()
        if bitorder == 'big':
            return _native.packbits(flat, 0)
        return _native.packbits(flat, 0, bitorder)
    if bitorder == 'big':
        return _native.packbits(a, axis)
    return _native.packbits(a, axis, bitorder)


def unpackbits(a, axis=None, count=None, bitorder='big'):
    """Unpack elements of a uint8 array into a binary-valued output array."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'little' or 'big'")
    if axis is not None:
        axis = int(axis)
        if axis < 0:
            axis += a.ndim
        if axis < 0 or axis >= a.ndim:
            raise ValueError("axis out of range")
    count = None if count is None else int(count)
    if axis is None:
        flat = a.flatten()
        if count is None and bitorder == 'big':
            return _native.unpackbits(flat, 0)
        if count is None:
            return _reverse_unpackbits_blocks(_native.unpackbits(flat, 0))
        if bitorder == 'big':
            return _native.unpackbits(flat, 0, count)
        return _reverse_unpackbits_blocks(_native.unpackbits(flat, 0), count=count)
    if count is None and bitorder == 'big':
        return _native.unpackbits(a, axis)
    if count is None:
        return _native.unpackbits(a, axis, None, bitorder)
    if bitorder == 'big':
        return _native.unpackbits(a, axis, count)
    return _native.unpackbits(a, axis, count, bitorder)


# ---------------------------------------------------------------------------
# Misc numeric
# ---------------------------------------------------------------------------

def binary_repr(num, width=None):
    num = int(num)  # Convert numpy scalars to plain Python int to avoid overflow
    if num >= 0:
        s = bin(num)[2:]
        if width is not None:
            s = s.zfill(width)
        return s
    else:
        if width is not None:
            s = bin(2**width + num)[2:]
            return s.zfill(width)
        else:
            return '-' + bin(-num)[2:]


def base_repr(number, base=2, padding=0):
    if base < 2 or base > 36:
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    if number == 0:
        return "0" * (padding + 1)
    digits = []
    n = __import__("builtins").abs(number)
    while n:
        digits.append(str(n % base) if n % base < 10 else chr(ord('A') + n % base - 10))
        n //= base
    s = "".join(reversed(digits))
    s = "0" * padding + s
    if number < 0:
        s = "-" + s
    return s
