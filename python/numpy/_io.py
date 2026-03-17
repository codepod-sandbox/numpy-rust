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
    'complex64':  ('<c8',  'f', 8),   # 2 floats per element
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
    '<c8':  'complex64',   '>c8':  'complex64',
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

    if dtype_str in ('complex128', 'complex64'):
        float_char = 'd' if dtype_str == 'complex128' else 'f'
        pairs = []
        for v in flat:
            if isinstance(v, complex):
                pairs.extend([v.real, v.imag])
            elif isinstance(v, (tuple, list)) and len(v) == 2:
                pairs.extend([float(v[0]), float(v[1])])
            else:
                pairs.extend([float(v), 0.0])
        data = _struct.pack('<' + float_char * len(pairs), *pairs)
    elif dtype_str == 'bool':
        data = _struct.pack('?' * n, *flat)
    else:
        data = _struct.pack('<' + struct_char * n, *flat)

    header_len = len(header_bytes)
    return _MAGIC + b'\x01\x00' + _struct.pack('<H', header_len) + header_bytes + data



def _npy_bytes_to_array(data):
    """Parse a .npy binary blob and return an ndarray."""
    if len(data) < 10 or data[:6] != _MAGIC:
        raise ValueError("not a .npy file: bad magic bytes")

    major = data[6]

    if major == 1:
        header_len = _struct.unpack_from('<H', data, 8)[0]
        header_start = 10
    elif major == 2:
        header_len = _struct.unpack_from('<I', data, 8)[0]
        header_start = 12
    else:
        raise ValueError(f"unsupported .npy version: {major}")

    header_bytes = data[header_start:header_start + header_len]
    header_str = header_bytes.decode('latin-1').strip()
    hdr = _ast.literal_eval(header_str)

    descr = hdr['descr']
    fortran_order = hdr['fortran_order']
    shape = tuple(hdr['shape'])

    dtype_str = _descr_to_dtype(descr)
    data_start = header_start + header_len
    raw = data[data_start:]

    n = 1
    for s in shape:
        n *= s

    endian = '>'  if descr[0] == '>' else '<'

    if dtype_str in ('complex128', 'complex64'):
        # complex64 uses 2 floats, complex128 uses 2 doubles
        float_char = 'f' if dtype_str == 'complex64' else 'd'
        vals = _struct.unpack_from(endian + float_char * (n * 2), raw)
        flat = [complex(vals[i * 2], vals[i * 2 + 1]) for i in range(n)]
    elif dtype_str == 'bool':
        flat = list(_struct.unpack_from('?' * n, raw))
    else:
        _, struct_char, _ = _DTYPE_INFO[dtype_str]
        flat = list(_struct.unpack_from(endian + struct_char * n, raw))

    if fortran_order and len(shape) > 1:
        result = array(flat, dtype=dtype_str).reshape(list(shape[::-1]))
        axes = list(range(len(shape) - 1, -1, -1))
        result = result.transpose(axes)
    elif shape == ():
        result = array(flat[0] if flat else 0, dtype=dtype_str)
    elif n == 0:
        result = array([], dtype=dtype_str).reshape(list(shape))
    else:
        result = array(flat, dtype=dtype_str)
        if shape != (n,):
            result = result.reshape(list(shape))

    return result

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
    """Save an array to a .npy file in binary format."""
    arr = asarray(arr)
    data = _array_to_npy_bytes(arr)
    if isinstance(file, str):
        with open(file, 'wb') as f:
            f.write(data)
    else:
        file.write(data)


def load(file, mmap_mode=None, **kwargs):
    """Load array(s) from a .npy or .npz file."""
    if isinstance(file, str):
        # Detect format by magic bytes first, then fall back to extension
        with open(file, 'rb') as f:
            magic = f.read(4)
        if magic[:2] == b'PK':
            # ZIP magic: .npz file
            return NpzFile(file)
        elif magic[:1] == b'\x93':
            # NumPy .npy magic
            with open(file, 'rb') as f:
                return _npy_bytes_to_array(f.read())
        elif file.lower().endswith('.npz'):
            return NpzFile(file)
        elif magic[:1] == b'#':
            # Legacy text format from old stub
            with open(file, 'r') as tf:
                lines = tf.readlines()
            shape_line = lines[0].strip()
            if shape_line.startswith('# shape:'):
                import json
                shape = tuple(json.loads(shape_line.split(':')[1].strip()))
                data_line = lines[1].strip()
            else:
                data_line = lines[0].strip()
                shape = None
            vals = [float(v) for v in data_line.split(',')]
            result = array(vals)
            if shape is not None and len(shape) > 1:
                result = result.reshape(list(shape))
            return result
        else:
            raise ValueError(f"unknown file format: {file!r}")
    else:
        # File-like object — detect format from magic bytes
        header = file.read(4)
        if len(header) >= 2 and header[:2] == b'PK':
            # ZIP magic: this is a .npz file
            file.seek(0)
            return NpzFile(file)
        elif len(header) >= 1 and header[:1] == b'\x93':
            # NumPy magic: read the rest as .npy
            rest = file.read()
            return _npy_bytes_to_array(header + rest)
        else:
            # Fallback: try as .npy
            rest = file.read()
            return _npy_bytes_to_array(header + rest)


class NpzFile:
    """Dict-like wrapper for .npz archives."""

    def __init__(self, file):
        # Track whether we own a file descriptor (for cleanup by callers)
        if isinstance(file, str):
            self.fid = open(file, 'rb')
            self._zip = _zipfile.ZipFile(self.fid, 'r')
        else:
            self.fid = None
            self._zip = _zipfile.ZipFile(file, 'r')
        self.files = [
            name[:-4] for name in self._zip.namelist()
            if name.endswith('.npy')
        ]

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        if key not in self.files:
            raise KeyError(key)
        raw = self._zip.read(key + '.npy')
        return _npy_bytes_to_array(raw)

    def __iter__(self):
        return iter(self.files)

    def keys(self):
        return iter(self.files)

    def __contains__(self, key):
        return key in self.files

    def __len__(self):
        return len(self.files)

    _MAX_REPR_ARRAY_COUNT = 5

    def __repr__(self):
        # Match NumPy's format: "NpzFile 'fname' with keys: k1, k2..."
        if self.fid is not None:
            fname = repr(self.fid.name)
        else:
            fname = repr('<memory>')
        keys = sorted(self.files)
        if len(keys) > self._MAX_REPR_ARRAY_COUNT:
            shown = ', '.join(keys[:self._MAX_REPR_ARRAY_COUNT]) + '...'
        else:
            shown = ', '.join(keys)
        return f"NpzFile {fname} with keys: {shown}"

    def close(self):
        self._zip.close()
        if self.fid is not None:
            self.fid.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _savez_impl(file, args, kwds, compress):
    arrays = {}
    for i, arr in enumerate(args):
        arrays[f'arr_{i}'] = asarray(arr)
    for name, arr in kwds.items():
        arrays[name] = asarray(arr)
    if isinstance(file, str) and not file.lower().endswith('.npz'):
        file = file + '.npz'
    try:
        compression = _zipfile.ZIP_DEFLATED if compress else _zipfile.ZIP_STORED
    except AttributeError:
        compression = _zipfile.ZIP_STORED
    with _zipfile.ZipFile(file, 'w', compression=compression) as zf:
        for name, arr in arrays.items():
            zf.writestr(name + '.npy', _array_to_npy_bytes(arr))


def savez(file, *args, **kwds):
    """Save arrays into an uncompressed .npz archive."""
    _savez_impl(file, args, kwds, compress=False)


def savez_compressed(file, *args, **kwds):
    """Save arrays into a compressed .npz archive."""
    _savez_impl(file, args, kwds, compress=True)
