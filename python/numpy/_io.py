"""File I/O functions."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._creation import array, asarray
from ._manipulation import reshape
import struct as _struct
import zipfile as _zipfile
import ast as _ast
import io as _io_module
import re as _re

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

# Type mapping for structured dtype fields: bare type string -> (struct_char, itemsize, npy_prefix)
_STRUCTURED_FIELD_MAP = {
    'bool':  ('?', 1),  'b1': ('?', 1),
    'i1':    ('b', 1),  'int8': ('b', 1),
    'i2':    ('h', 2),  'int16': ('h', 2),
    'i4':    ('i', 4),  'int32': ('i', 4),
    'i8':    ('q', 8),  'int64': ('q', 8),
    'u1':    ('B', 1),  'uint8': ('B', 1),
    'u2':    ('H', 2),  'uint16': ('H', 2),
    'u4':    ('I', 4),  'uint32': ('I', 4),
    'u8':    ('Q', 8),  'uint64': ('Q', 8),
    'f2':    ('e', 2),  'float16': ('e', 2),
    'f4':    ('f', 4),  'float32': ('f', 4),
    'f8':    ('d', 8),  'float64': ('d', 8),
    'c8':    ('ff', 8), 'complex64': ('ff', 8),
    'c16':   ('dd', 16), 'complex128': ('dd', 16),
}

# Map from bare type key (stripped of endian prefix) to npy descr with endian
_STRUCTURED_FIELD_NPY_DESCR = {
    'bool': '|b1',  'b1': '|b1',
    'i1': '|i1',  'int8': '|i1',
    'i2': '<i2',  'int16': '<i2',
    'i4': '<i4',  'int32': '<i4',
    'i8': '<i8',  'int64': '<i8',
    'u1': '|u1',  'uint8': '|u1',
    'u2': '<u2',  'uint16': '<u2',
    'u4': '<u4',  'uint32': '<u4',
    'u8': '<u8',  'uint64': '<u8',
    'f2': '<f2',  'float16': '<f2',
    'f4': '<f4',  'float32': '<f4',
    'f8': '<f8',  'float64': '<f8',
    'c8': '<c8',  'complex64': '<c8',
    'c16': '<c16', 'complex128': '<c16',
}


def _strip_endian(type_str):
    """Strip leading endian prefix from a type string like '<i4' -> 'i4'."""
    if type_str and type_str[0] in '<>|=':
        return type_str[1:]
    return type_str


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


def _make_npy_bytes(magic_ver, header_bytes, data_bytes):
    """Assemble a .npy blob with the given version byte (1 or 2), header, and data."""
    header_len = len(header_bytes)
    if magic_ver == 1:
        return _MAGIC + b'\x01\x00' + _struct.pack('<H', header_len) + header_bytes + data_bytes
    else:
        return _MAGIC + b'\x02\x00' + _struct.pack('<I', header_len) + header_bytes + data_bytes


def _build_header(descr_str, shape, fortran_order=False):
    """Build a padded .npy header, returning (header_bytes, version).

    Chooses version 1.0 unless the header would exceed 65535 bytes.
    """
    shape_repr = repr(tuple(shape))
    if isinstance(descr_str, str) and descr_str.startswith('['):
        # Structured dtype: descr already looks like a Python list literal
        header_dict = f"{{'descr': {descr_str}, 'fortran_order': {fortran_order}, 'shape': {shape_repr}, }}"
    else:
        header_dict = f"{{'descr': '{descr_str}', 'fortran_order': {fortran_order}, 'shape': {shape_repr}, }}"

    raw = header_dict.encode('latin-1')
    # Try version 1.0 first (prefix = 10 bytes)
    base_len = len(raw) + 1  # +1 for trailing '\n'
    pad1 = (64 - ((10 + base_len) % 64)) % 64
    if 10 + base_len + pad1 <= 10 + 65535:
        header = header_dict + ' ' * pad1 + '\n'
        return header.encode('latin-1'), 1
    # Fall back to version 2.0 (prefix = 12 bytes)
    pad2 = (64 - ((12 + base_len) % 64)) % 64
    header = header_dict + ' ' * pad2 + '\n'
    return header.encode('latin-1'), 2


def _array_to_npy_bytes(arr):
    """Encode an ndarray (or _ObjectArray / StructuredArray) as a .npy binary blob."""
    from ._helpers import _ObjectArray
    from numpy import StructuredArray

    dtype_str = str(arr.dtype)
    shape = arr.shape

    # ------------------------------------------------------------------ object
    if dtype_str in ('object', "<class 'object'>"):
        import pickle
        flat = list(arr.flatten().tolist()) if hasattr(arr, 'flatten') else list(arr._data)
        data_bytes = pickle.dumps(flat)
        header_bytes, ver = _build_header('|O', shape)
        return _make_npy_bytes(ver, header_bytes, data_bytes)

    # ------------------------------------------------------------------- str
    if dtype_str == 'str':
        flat = arr.flatten().tolist() if hasattr(arr, 'flatten') else list(arr._data)
        if flat:
            max_len = max(len(s) for s in flat)
        else:
            max_len = 1
        if max_len == 0:
            max_len = 1
        descr = f'<U{max_len}'
        chunks = []
        for s in flat:
            encoded = s.encode('utf-32-le')
            # Pad or truncate to max_len chars * 4 bytes
            target = max_len * 4
            if len(encoded) < target:
                encoded = encoded + b'\x00' * (target - len(encoded))
            else:
                encoded = encoded[:target]
            chunks.append(encoded)
        data_bytes = b''.join(chunks)
        header_bytes, ver = _build_header(descr, shape)
        return _make_npy_bytes(ver, header_bytes, data_bytes)

    # -------------------------------------------------------------- structured
    if dtype_str.startswith('[') or dtype_str.startswith('dtype(') or isinstance(arr, StructuredArray):
        # Get the field list: list of (name, bare_type_str)
        # For StructuredArray: use arr.dtype.descr which gives [(name, dtype_name), ...]
        # For _ObjectArray with structured dtype: parse str(arr.dtype)
        dt_obj = arr.dtype
        if hasattr(dt_obj, 'descr') and hasattr(dt_obj, 'names') and dt_obj.names is not None:
            # StructuredDtype / dtype wrapping StructuredDtype
            fields_raw = [(name, str(dt_obj.fields[name][0])) for name in dt_obj.names]
        elif dtype_str.startswith('['):
            fields_raw = _ast.literal_eval(dtype_str)
        else:
            # Fallback: try descr from dtype object
            descr_val = getattr(dt_obj, 'descr', None)
            if descr_val:
                fields_raw = [(item[0], item[1]) for item in descr_val if item[0]]
            else:
                raise ValueError(f"cannot determine structured dtype fields from {dtype_str!r}")

        # Build npy-compatible field list and struct format
        npy_fields = []
        row_fmt = ''
        for name, type_str in fields_raw:
            bare = _strip_endian(type_str)
            info = _STRUCTURED_FIELD_MAP.get(bare)
            if info is None:
                raise ValueError(f"unsupported structured field type: {type_str!r}")
            struct_char, sz = info
            npy_desc = _STRUCTURED_FIELD_NPY_DESCR.get(bare, f'<{bare}')
            npy_fields.append((name, npy_desc))
            row_fmt += struct_char

        # Build descr string for npy header (Python list-of-tuples literal)
        descr_parts = ", ".join(f"('{n}', '{d}')" for n, d in npy_fields)
        descr_str_npy = f'[{descr_parts}]'

        # Get row data — tolist() returns list of record tuples
        rows = arr.tolist()
        nrows = len(rows)
        # Structured arrays are always 1D (nrows records) in .npy format
        npy_shape = (nrows,)

        chunks = []
        for row in rows:
            # row is a tuple of field values
            values = []
            for val, (name, type_str) in zip(row, fields_raw):
                bare = _strip_endian(type_str)
                struct_char, _ = _STRUCTURED_FIELD_MAP[bare]
                if struct_char in ('ff', 'dd'):
                    if isinstance(val, complex):
                        values.extend([float(val.real), float(val.imag)])
                    else:
                        values.extend([float(val), 0.0])
                elif struct_char == '?':
                    values.append(bool(val))
                else:
                    values.append(val)
            chunks.append(_struct.pack('<' + row_fmt, *values))

        data_bytes = b''.join(chunks)
        header_bytes, ver = _build_header(descr_str_npy, npy_shape)
        return _make_npy_bytes(ver, header_bytes, data_bytes)

    # --------------------------------------------------------- standard numeric
    descr, struct_char, _ = _dtype_to_descr(dtype_str)

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
        data_bytes = _struct.pack('<' + float_char * len(pairs), *pairs)
    elif dtype_str == 'bool':
        data_bytes = _struct.pack('?' * n, *flat)
    else:
        data_bytes = _struct.pack('<' + struct_char * n, *flat)

    header_bytes, ver = _build_header(descr, shape)
    return _make_npy_bytes(ver, header_bytes, data_bytes)


def _parse_structured_descr(descr):
    """Parse a structured dtype descriptor (list or string starting with '[').

    Returns list of (name, bare_type) tuples.
    """
    if isinstance(descr, list):
        fields = descr
    else:
        fields = _ast.literal_eval(descr)
    # Normalize: strip endian from type strings
    result = []
    for item in fields:
        name, type_str = item[0], item[1]
        bare = _strip_endian(type_str)
        result.append((name, bare, type_str))
    return result


def _build_complex_array(flat, shape, dtype_str):
    reals = [v.real if isinstance(v, complex) else float(v) for v in flat]
    imags = [v.imag if isinstance(v, complex) else 0.0 for v in flat]
    result_shape = list(shape) if shape != () else [1]
    re_arr = _native.array(reals).reshape(result_shape).astype("complex128")
    im_arr = _native.array(imags).reshape(result_shape).astype("complex128")
    j_arr = _native.zeros([1], "complex128")
    j_arr[0] = (0.0, 1.0)
    result = re_arr + im_arr * j_arr.reshape([])
    if dtype_str != "complex128":
        result = result.astype(dtype_str)
    if shape == ():
        return result.reshape([])
    return result


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

    data_start = header_start + header_len
    raw = data[data_start:]

    n = 1
    for s in shape:
        n *= s

    # ------------------------------------------------------------------ object
    if descr == '|O':
        import pickle
        flat = pickle.loads(raw)
        if not flat and n == 0:
            result = array([], dtype=object)
            if shape != (0,):
                result = result.reshape(list(shape))
            return result
        result = array(flat, dtype=object)
        if shape not in ((), (n,)):
            result = result.reshape(list(shape))
        return result

    # ------------------------------------------------------------------- str
    if isinstance(descr, str) and _re.match(r'^[<>|]?U\d+$', descr):
        # Extract char count
        m = _re.search(r'U(\d+)$', descr)
        char_count = int(m.group(1))
        bytes_per_elem = char_count * 4
        flat = []
        for i in range(n):
            chunk = raw[i * bytes_per_elem:(i + 1) * bytes_per_elem]
            s = chunk.decode('utf-32-le').rstrip('\x00')
            flat.append(s)
        if shape == ():
            return array(flat[0] if flat else '', dtype='str')
        result = array(flat, dtype='str')
        if shape != (n,):
            result = result.reshape(list(shape))
        return result

    # -------------------------------------------------------------- structured
    if isinstance(descr, (list, str)) and (
        isinstance(descr, list) or
        (isinstance(descr, str) and descr.strip().startswith('['))
    ):
        fields = _parse_structured_descr(descr)
        # Build struct format
        fmt = '<'
        field_sizes = []
        for name, bare, orig in fields:
            info = _STRUCTURED_FIELD_MAP.get(bare)
            if info is None:
                raise ValueError(f"unsupported structured field type: {orig!r}")
            struct_char, sz = info
            fmt += struct_char
            field_sizes.append((name, bare, struct_char, sz))

        # Calculate itemsize
        itemsize = sum(sz for _, _, _, sz in field_sizes)

        # Unpack all rows
        rows = []
        for i in range(n):
            row_bytes = raw[i * itemsize:(i + 1) * itemsize]
            unpacked = _struct.unpack(fmt, row_bytes)
            # Rebuild row: complex types need re-pairing
            row_vals = []
            idx = 0
            for name, bare, struct_char, sz in field_sizes:
                if struct_char in ('ff', 'dd'):
                    row_vals.append(complex(unpacked[idx], unpacked[idx + 1]))
                    idx += 2
                else:
                    row_vals.append(unpacked[idx])
                    idx += 1
            rows.append(tuple(row_vals))

        # Reconstruct dtype spec
        dtype_spec = [(name, orig) for name, bare, orig in fields]
        result = array(rows, dtype=dtype_spec)
        if shape not in ((), (n,)):
            result = result.reshape(list(shape))
        return result

    # --------------------------------------------------------- standard numeric
    dtype_str = _descr_to_dtype(descr)
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

    if dtype_str in ('complex128', 'complex64'):
        result = _build_complex_array(flat, shape, dtype_str)
        if fortran_order and len(shape) > 1 and hasattr(result, '_mark_fortran'):
            result._mark_fortran()
        return result
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


class BagObj:
    """Attribute-style accessor for NpzFile (npz.f.arr_0 == npz['arr_0'])."""

    def __init__(self, npz):
        self._npz = npz

    def __getattr__(self, key):
        try:
            return self._npz[key]
        except KeyError:
            raise AttributeError(key)

    def __dir__(self):
        return self._npz.files


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
        self.f = BagObj(self)

    @property
    def zip(self):
        """Alias for _zip (used by some tests as data.zip.fp)."""
        return self._zip

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        names = self._zip.namelist()
        # Exact match (e.g. "metadata", "test1.npy", or "test2" holding .npy bytes)
        if key in names:
            raw = self._zip.read(key)
            if raw[:6] == _MAGIC:
                return _npy_bytes_to_array(raw)
            return raw
        # Strip-suffix lookup: "test1" → finds "test1.npy"
        if key + '.npy' in names:
            return _npy_bytes_to_array(self._zip.read(key + '.npy'))
        raise KeyError(key)

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
