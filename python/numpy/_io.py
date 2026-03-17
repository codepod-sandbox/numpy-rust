"""File I/O functions."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._creation import array, asarray
from ._manipulation import reshape
import struct as _struct
import zipfile as _zipfile
import ast as _ast
import io as _io_module

_MAGIC = b'\x93NUMPY'

# Maps dtype name -> (npy_descr, struct_char, itemsize)
_DTYPE_INFO = {
    'float16':    ('<f2', 'e', 2),
    'float32':    ('<f4', 'f', 4),
    'float64':    ('<f8', 'd', 8),
    'int8':       ('|i1', 'b', 1),
    'int16':      ('<i2', 'h', 2),
    'int32':      ('<i4', 'i', 4),
    'int64':      ('<i8', 'q', 8),
    'uint8':      ('|u1', 'B', 1),
    'uint16':     ('<u2', 'H', 2),
    'uint32':     ('<u4', 'I', 4),
    'uint64':     ('<u8', 'Q', 8),
    'bool':       ('|b1', '?', 1),
    'complex128': ('<c16', 'd', 16),  # 2 doubles per element
}

# Reverse: .npy descriptor -> dtype name (handles both endians)
_DESCR_TO_DTYPE = {
    '<f2': 'float16',   '>f2': 'float16',
    '<f4': 'float32',   '>f4': 'float32',
    '<f8': 'float64',   '>f8': 'float64',   '|f8': 'float64',
    '<i1': 'int8',      '>i1': 'int8',      '|i1': 'int8',
    '<i2': 'int16',     '>i2': 'int16',
    '<i4': 'int32',     '>i4': 'int32',
    '<i8': 'int64',     '>i8': 'int64',
    '<u1': 'uint8',     '>u1': 'uint8',     '|u1': 'uint8',
    '<u2': 'uint16',    '>u2': 'uint16',
    '<u4': 'uint32',    '>u4': 'uint32',
    '<u8': 'uint64',    '>u8': 'uint64',
    '|b1': 'bool',      '<b1': 'bool',
    '<c8':  'complex128',  '>c8':  'complex128',
    '<c16': 'complex128',  '>c16': 'complex128',   '|c16': 'complex128',
}


def _dtype_to_descr(dtype_str):
    """Return (npy_descr, struct_char, itemsize) for dtype name."""
    info = _DTYPE_INFO.get(dtype_str)
    if info is None:
        raise ValueError(
            f"cannot save {dtype_str!r} arrays to .npy (unsupported dtype)"
        )
    return info


def _descr_to_dtype(descr):
    """Map .npy descriptor string to our dtype name."""
    dt = _DESCR_TO_DTYPE.get(descr)
    if dt is None:
        raise ValueError(f"unsupported .npy dtype descriptor: {descr!r}")
    return dt


def _array_to_npy_bytes(arr):
    """Encode an ndarray as a .npy binary blob (version 1.0)."""
    dtype_str = str(arr.dtype)
    descr, struct_char, _ = _dtype_to_descr(dtype_str)
    shape = arr.shape

    # Build header dict string and pad to 64-byte alignment.
    # Total prefix = magic(6) + version(2) + header_len_field(2) = 10 bytes.
    # Requirement: (10 + header_len) % 64 == 0
    header_dict = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape!r}, }}"
    base_len = len(header_dict.encode('latin-1')) + 1  # +1 for '\n'
    pad = (64 - ((10 + base_len) % 64)) % 64
    header = header_dict + ' ' * pad + '\n'
    header_bytes = header.encode('latin-1')

    flat = arr.flatten().tolist()
    n = len(flat)

    if dtype_str == 'complex128':
        pairs = []
        for v in flat:
            if isinstance(v, complex):
                pairs.extend([v.real, v.imag])
            elif isinstance(v, (tuple, list)) and len(v) == 2:
                pairs.extend([float(v[0]), float(v[1])])
            else:
                pairs.extend([float(v), 0.0])
        data = _struct.pack('<' + 'd' * len(pairs), *pairs)
    elif dtype_str == 'bool':
        data = _struct.pack('?' * n, *flat)
    else:
        data = _struct.pack('<' + struct_char * n, *flat)

    header_len = len(header_bytes)
    return _MAGIC + b'\x01\x00' + _struct.pack('<H', header_len) + header_bytes + data


__all__ = ['loadtxt', 'savetxt', 'genfromtxt', 'save', 'load', 'savez', 'savez_compressed']


def loadtxt(fname, dtype=None, comments='#', delimiter=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, **kwargs):
    """Load data from a text file. Each row must have the same number of values."""
    if isinstance(fname, str):
        f = open(fname, 'r')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        rows = []
        lines_read = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip comment lines
            if comments and line.startswith(comments):
                continue
            if lines_read < skiprows:
                lines_read += 1
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            # Split by delimiter
            if delimiter is None:
                parts = line.split()
            else:
                parts = line.split(delimiter)
            # Select columns
            if usecols is not None:
                parts = [parts[i] for i in usecols]
            row = [float(x.strip()) for x in parts]
            rows.append(row)
        if not rows:
            return array([])
        if len(rows) == 1 and ndmin < 2:
            result = array(rows[0])
        else:
            result = array(rows)
        if unpack:
            return result.T
        return result
    finally:
        if close_file:
            f.close()


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None):
    """Save an array to a text file."""
    if not isinstance(X, ndarray):
        X = array(X)
    if X.ndim == 1:
        X = X.reshape([X.size, 1])

    if isinstance(fname, str):
        f = open(fname, 'w')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        if header:
            for hline in header.split('\n'):
                f.write(comments + hline + newline)

        rows = X.shape[0]
        cols = X.shape[1]
        for i in range(rows):
            vals = []
            for j in range(cols):
                vals.append(fmt % float(X[i][j]))
            f.write(delimiter.join(vals) + newline)

        if footer:
            for fline in footer.split('\n'):
                f.write(comments + fline + newline)
    finally:
        if close_file:
            f.close()


def genfromtxt(fname, dtype=None, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, missing_values=None, filling_values=None,
               usecols=None, names=None, excludelist=None, deletechars=None,
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt='f%i', unpack=False, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding='bytes', **kwargs):
    """Load data from text file, handling missing values."""
    if filling_values is None:
        filling_values = float('nan')

    if isinstance(fname, str):
        with open(fname, 'r') as f:
            lines = f.readlines()
    else:
        lines = fname.readlines()

    # Skip header/footer
    lines = lines[skip_header:]
    if skip_footer > 0:
        lines = lines[:-skip_footer]
    if max_rows is not None:
        lines = lines[:max_rows]

    rows = []
    for line in lines:
        line = line.strip()
        if not line or (comments and line.startswith(comments)):
            continue
        if delimiter is None:
            parts = line.split()
        else:
            parts = line.split(delimiter)

        if usecols is not None:
            parts = [parts[i] for i in usecols]

        row = []
        for p in parts:
            p = p.strip()
            if missing_values and p in (missing_values if isinstance(missing_values, (list, tuple, set)) else [missing_values]):
                row.append(filling_values)
            else:
                try:
                    row.append(float(p))
                except (ValueError, TypeError):
                    row.append(filling_values)
        rows.append(row)

    if not rows:
        return array([])
    result = array(rows)
    if unpack:
        return result.T
    return result


def save(file, arr, **kwargs):
    """Save an array to a .npy file (text-based format for compatibility)."""
    arr = asarray(arr)
    with open(file, 'w') as f:
        f.write(f"# shape: {list(arr.shape)}\n")
        flat = arr.flatten()
        vals = [str(flat[i]) for i in range(flat.size)]
        f.write(','.join(vals) + '\n')


def load(file, mmap_mode=None, fix_imports=True, encoding='ASCII', **kwargs):
    """Load array from a .npy file (text-based format)."""
    with open(file, 'r') as f:
        lines = f.readlines()
    # Parse shape from first line
    shape_line = lines[0].strip()
    if shape_line.startswith('# shape:'):
        import json
        shape = tuple(json.loads(shape_line.split(':')[1].strip()))
        data_line = lines[1].strip()
    else:
        # Fallback: treat as flat data
        data_line = lines[0].strip()
        shape = None
    vals = [float(v) for v in data_line.split(',')]
    result = array(vals)
    if shape is not None and len(shape) > 1:
        result = result.reshape(shape)
    return result


def savez(file, *args, **kwds):
    """Save several arrays into a single file in text format.
    Since we can't use actual npz (zip) format, save as multi-section text."""
    arrays = {}
    for i, arr in enumerate(args):
        arrays[f'arr_{i}'] = asarray(arr)
    for name, arr in kwds.items():
        arrays[name] = asarray(arr)
    with open(file, 'w') as f:
        for name, arr in arrays.items():
            f.write(f"# {name} shape: {list(arr.shape)}\n")
            flat = arr.flatten()
            vals = [str(flat[i]) for i in range(flat.size)]
            f.write(','.join(vals) + '\n')


savez_compressed = savez  # alias, same behavior in our sandbox
