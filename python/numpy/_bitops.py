"""Bit manipulation functions for numpy-rust."""
from _numpy_native import ndarray
from ._helpers import _flat_arraylike_data
from ._creation import array, asarray

__all__ = [
    'packbits', 'unpackbits',
    'binary_repr', 'base_repr',
]


# ---------------------------------------------------------------------------
# Bit manipulation
# ---------------------------------------------------------------------------

def packbits(a, axis=None, bitorder='big'):
    """Pack a binary-valued array into uint8."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        vals = _flat_arraylike_data(a.flatten())
        if bitorder == 'little':
            result = []
            for i in range(0, len(vals), 8):
                chunk = vals[i:i+8]
                byte = 0
                for j in range(len(chunk)):
                    if int(chunk[j]):
                        byte |= (1 << j)
                result.append(byte)
        else:
            result = []
            for i in range(0, len(vals), 8):
                chunk = vals[i:i+8]
                byte = 0
                for j in range(len(chunk)):
                    if int(chunk[j]):
                        byte |= (1 << (7 - j))
                result.append(byte)
        return array(result, dtype='uint8')
    else:
        # axis-wise packbits: apply along the given axis
        import numpy as _np
        # Move target axis to last position for easy iteration
        a2 = _np.moveaxis(a, axis, -1)
        orig_shape = a2.shape
        flat = a2.reshape(-1, orig_shape[-1])
        packed_rows = []
        for i in range(flat.shape[0]):
            row = _flat_arraylike_data(flat[i])
            if bitorder == 'little':
                out = []
                for i in range(0, len(row), 8):
                    chunk = row[i:i+8]
                    byte = 0
                    for j in range(len(chunk)):
                        if int(chunk[j]):
                            byte |= (1 << j)
                    out.append(byte)
            else:
                out = []
                for i in range(0, len(row), 8):
                    chunk = row[i:i+8]
                    byte = 0
                    for j in range(len(chunk)):
                        if int(chunk[j]):
                            byte |= (1 << (7 - j))
                    out.append(byte)
            packed_rows.append(out)
        packed_len = len(packed_rows[0]) if packed_rows else 0
        result2 = array(packed_rows, dtype='uint8').reshape(orig_shape[:-1] + (packed_len,))
        return _np.moveaxis(result2, -1, axis)


def unpackbits(a, axis=None, count=None, bitorder='big'):
    """Unpack elements of a uint8 array into a binary-valued output array."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        vals = _flat_arraylike_data(a.flatten())
        result = []
        for v in vals:
            byte = int(v)
            if bitorder == 'little':
                for j in range(8):
                    result.append((byte >> j) & 1)
            else:
                for j in range(7, -1, -1):
                    result.append((byte >> j) & 1)
        if count is not None:
            count = int(count)
            if count >= 0:
                result = result[:count]
            else:
                result = result[:len(result) + count]
        return array(result, dtype='uint8')
    else:
        import numpy as _np
        a2 = _np.moveaxis(a, axis, -1)
        orig_shape = a2.shape
        flat = a2.reshape(-1, orig_shape[-1])
        unpacked_rows = []
        for i in range(flat.shape[0]):
            row = _flat_arraylike_data(flat[i])
            bits = []
            for v in row:
                byte = int(v)
                if bitorder == 'little':
                    for j in range(8):
                        bits.append((byte >> j) & 1)
                else:
                    for j in range(7, -1, -1):
                        bits.append((byte >> j) & 1)
            unpacked_rows.append(bits)
        unpacked_len = len(unpacked_rows[0]) if unpacked_rows else 0
        if count is not None:
            count = int(count)
            if count >= 0:
                unpacked_rows = [r[:count] for r in unpacked_rows]
                unpacked_len = count
            else:
                unpacked_rows = [r[:unpacked_len + count] for r in unpacked_rows]
                unpacked_len = max(0, unpacked_len + count)
        result2 = array(unpacked_rows, dtype='uint8').reshape(orig_shape[:-1] + (unpacked_len,))
        return _np.moveaxis(result2, -1, axis)


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
