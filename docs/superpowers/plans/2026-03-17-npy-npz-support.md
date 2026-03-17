# .npy / .npz Binary Format Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace text-based `save`/`load`/`savez` stubs with a full binary .npy/.npz implementation interoperable with CPython NumPy.

**Architecture:** All changes in `python/numpy/_io.py` (helpers + rewrites) and a new test file `tests/python/test_io_npy.py`. Uses `struct` for byte packing, `zipfile` for .npz archives, `ast.literal_eval` for safe header parsing (parses only Python literals, not arbitrary code) — all confirmed available in RustPython. No Rust changes needed.

**Tech Stack:** Python stdlib only: `struct`, `zipfile`, `ast`, `io.BytesIO`

---

## File Structure

| File | Change |
|------|--------|
| `python/numpy/_io.py` | Add `struct`/`zipfile`/`ast`/`io` imports; add `_MAGIC`, `_DTYPE_INFO`, `_DESCR_TO_DTYPE`, `_dtype_to_descr()`, `_descr_to_dtype()`, `_array_to_npy_bytes()`, `_npy_bytes_to_array()`, `NpzFile`; replace `save()`, `load()`, `savez()`, `savez_compressed()` |
| `tests/python/test_io_npy.py` | New test file — 14 tests with standard runner at bottom |
| `README.md` | Remove "Binary .npy/.npz not implemented" limitation entry |

---

## Chunk 1: Core helpers and write path

### Task 1: Dtype helpers

**Files:**
- Modify: `python/numpy/_io.py:1-8`
- Create: `tests/python/test_io_npy.py`

- [ ] **Step 1: Create test file with dtype round-trip test**

Create `tests/python/test_io_npy.py`:

```python
"""Tests for binary .npy / .npz file format support."""
import numpy as np
import os

_TMP = '/tmp/_test_npy_'  # prefix for temp files

def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def test_dtype_roundtrip_each_dtype():
    """All 13 supported dtypes: save then load, value and dtype preserved."""
    dtypes = [
        'float16', 'float32', 'float64',
        'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64',
        'bool', 'complex128',
    ]
    for dt in dtypes:
        if dt == 'complex128':
            a = np.array([1+2j, 3+4j], dtype=dt)
        elif dt == 'bool':
            a = np.array([True, False, True], dtype=dt)
        else:
            a = np.array([1, 2, 3], dtype=dt)
        path = _TMP + dt + '.npy'
        np.save(path, a)
        b = np.load(path)
        _cleanup(path)
        assert b.dtype == a.dtype, f"{dt}: dtype {b.dtype} != {a.dtype}"
        assert b.shape == a.shape, f"{dt}: shape {b.shape} != {a.shape}"
        al, bl = a.tolist(), b.tolist()
        for i in range(len(al)):
            av, bv = al[i], bl[i]
            if dt == 'complex128':
                assert abs(av.real - bv.real) < 1e-5 and abs(av.imag - bv.imag) < 1e-5, \
                    f"{dt}[{i}]: {av} != {bv}"
            elif dt in ('float16', 'float32', 'float64'):
                assert abs(float(av) - float(bv)) < 1e-5, f"{dt}[{i}]: {av} != {bv}"
            else:
                assert av == bv, f"{dt}[{i}]: {av} != {bv}"


# Runner — added to at the end of each task
tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
passed = 0
failed = 0
for t in tests:
    try:
        t()
        passed += 1
    except Exception as e:
        import traceback
        print(f'FAIL {t.__name__}: {e}')
        traceback.print_exc()
        failed += 1
print(f'test_io_npy: {passed} passed, {failed} failed')
if failed:
    raise SystemExit(1)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: `FAIL test_dtype_roundtrip_each_dtype` — load returns wrong dtype/values since `save` is still text-based.

- [ ] **Step 3: Add imports and dtype maps to `_io.py`**

After the existing 5 import lines at the top of `python/numpy/_io.py`, add:

```python
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
    '<c8':  'complex128',  '>c8':  'complex128',   # complex64 -> upcast
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
```

- [ ] **Step 4: Verify no import errors**

```bash
./target/release/numpy-python -c "from numpy._io import _dtype_to_descr, _descr_to_dtype; print(_dtype_to_descr('float64'))"
```

Expected: `('<f8', 'd', 8)`

---

### Task 2: `_array_to_npy_bytes` — write path

**Files:**
- Modify: `python/numpy/_io.py` (add after `_descr_to_dtype`)
- Modify: `tests/python/test_io_npy.py`

- [ ] **Step 1: Add write-path test**

Add before the runner block in `tests/python/test_io_npy.py`:

```python
def test_npy_bytes_magic_and_header():
    """_array_to_npy_bytes produces valid magic, version, and header."""
    from numpy._io import _array_to_npy_bytes, _MAGIC
    import ast, struct
    a = np.array([1.0, 2.0, 3.0])  # float64
    data = _array_to_npy_bytes(a)
    assert data[:6] == _MAGIC, "bad magic"
    assert data[6] == 1 and data[7] == 0, "expected version 1.0"
    header_len = struct.unpack_from('<H', data, 8)[0]
    assert (10 + header_len) % 64 == 0, "header not 64-byte aligned"
    header_str = data[10:10 + header_len].decode('latin-1').strip()
    hdr = ast.literal_eval(header_str)
    assert hdr['descr'] == '<f8'
    assert hdr['fortran_order'] == False
    assert hdr['shape'] == (3,)
    vals = struct.unpack_from('<3d', data, 10 + header_len)
    assert abs(vals[0] - 1.0) < 1e-10
    assert abs(vals[1] - 2.0) < 1e-10
    assert abs(vals[2] - 3.0) < 1e-10
```

- [ ] **Step 2: Run — confirm this test fails**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: `FAIL test_npy_bytes_magic_and_header`

- [ ] **Step 3: Add `_array_to_npy_bytes` to `_io.py`** after `_descr_to_dtype`

```python
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
```

- [ ] **Step 4: Run tests**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: `test_npy_bytes_magic_and_header` passes. `test_dtype_roundtrip_each_dtype` still fails (load not yet updated).

- [ ] **Step 5: Commit**

```bash
git add python/numpy/_io.py tests/python/test_io_npy.py
git commit -m "feat(io): add dtype helpers and _array_to_npy_bytes"
```

---

### Task 3: `_npy_bytes_to_array` — read path

**Files:**
- Modify: `python/numpy/_io.py` (add after `_array_to_npy_bytes`)
- Modify: `tests/python/test_io_npy.py`

- [ ] **Step 1: Add read-path tests**

Add before the runner in `tests/python/test_io_npy.py`:

```python
def test_npy_roundtrip_internal():
    """_npy_bytes_to_array(_array_to_npy_bytes(arr)) == arr for key dtypes."""
    from numpy._io import _array_to_npy_bytes, _npy_bytes_to_array
    cases = [
        ('float64', [1.5, 2.5, 3.5]),
        ('int32',   [1, -2, 3]),
        ('uint8',   [0, 128, 255]),
        ('bool',    [True, False, True]),
        ('complex128', [1+2j, 3+4j]),
    ]
    for dt, vals in cases:
        a = np.array(vals, dtype=dt)
        b = _npy_bytes_to_array(_array_to_npy_bytes(a))
        assert str(b.dtype) == dt, f"{dt}: got dtype {b.dtype}"
        assert b.shape == a.shape, f"{dt}: shape mismatch"
        al, bl = a.tolist(), b.tolist()
        for i, (av, bv) in enumerate(zip(al, bl)):
            if dt == 'complex128':
                assert abs(av.real - bv.real) < 1e-10
                assert abs(av.imag - bv.imag) < 1e-10
            elif 'float' in dt:
                assert abs(float(av) - float(bv)) < 1e-5
            else:
                assert av == bv, f"{dt}[{i}]: {av} != {bv}"


def test_npy_roundtrip_2d():
    """2D arrays preserve shape."""
    from numpy._io import _array_to_npy_bytes, _npy_bytes_to_array
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype='int32')
    b = _npy_bytes_to_array(_array_to_npy_bytes(a))
    assert b.shape == (2, 3)
    assert b.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_npy_roundtrip_0d():
    """0-d arrays (scalar) preserve shape ()."""
    from numpy._io import _array_to_npy_bytes, _npy_bytes_to_array
    a = np.array(3.14)
    b = _npy_bytes_to_array(_array_to_npy_bytes(a))
    assert b.shape == (), f"expected shape () got {b.shape}"
    assert abs(float(b) - 3.14) < 1e-6


def test_npy_roundtrip_empty():
    """Empty arrays (shape (0,)) round-trip."""
    from numpy._io import _array_to_npy_bytes, _npy_bytes_to_array
    a = np.array([], dtype='float64')
    b = _npy_bytes_to_array(_array_to_npy_bytes(a))
    assert b.shape == (0,), f"expected (0,), got {b.shape}"


def test_npy_bad_magic():
    """ValueError on bad magic bytes."""
    from numpy._io import _npy_bytes_to_array
    try:
        _npy_bytes_to_array(b'not a npy file at all')
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert 'magic' in str(e).lower() or 'npy' in str(e).lower()


def test_interop_read():
    """Read .npy bytes matching real CPython NumPy output for [1.0, 2.0, 3.0]."""
    from numpy._io import _npy_bytes_to_array, _MAGIC
    import struct
    # Construct bytes that exactly match what CPython np.save produces:
    # header_len must satisfy (10 + header_len) % 64 == 0 → header_len = 118
    hdr_dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }"
    hdr_len = 118
    hdr = hdr_dict + ' ' * (hdr_len - len(hdr_dict) - 1) + '\n'
    assert len(hdr) == hdr_len
    blob = _MAGIC + b'\x01\x00' + struct.pack('<H', hdr_len)
    blob += hdr.encode('latin-1')
    blob += struct.pack('<3d', 1.0, 2.0, 3.0)
    a = _npy_bytes_to_array(blob)
    assert a.shape == (3,)
    assert str(a.dtype) == 'float64'
    vals = a.tolist()
    assert abs(vals[0] - 1.0) < 1e-10
    assert abs(vals[1] - 2.0) < 1e-10
    assert abs(vals[2] - 3.0) < 1e-10
```

- [ ] **Step 2: Run — confirm new tests fail**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: failures on the 6 new tests.

- [ ] **Step 3: Add `_npy_bytes_to_array` to `_io.py`** after `_array_to_npy_bytes`

```python
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
    # ast.literal_eval safely parses Python dict/tuple literals (no arbitrary code)
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

    # Byte order prefix for multi-byte struct formats
    endian = '>' if descr[0] == '>' else '<'

    if dtype_str == 'complex128':
        # complex64 in file -> float32 pairs; complex128 -> float64 pairs
        float_char = 'f' if descr in ('<c8', '>c8') else 'd'
        vals = _struct.unpack_from(endian + float_char * (n * 2), raw)
        flat = [complex(vals[i * 2], vals[i * 2 + 1]) for i in range(n)]
    elif dtype_str == 'bool':
        flat = list(_struct.unpack_from('?' * n, raw))
    else:
        _, struct_char, _ = _DTYPE_INFO[dtype_str]
        flat = list(_struct.unpack_from(endian + struct_char * n, raw))

    if fortran_order and len(shape) > 1:
        # Fortran-order: values are column-major; reshape transposed then transpose
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
```

- [ ] **Step 4: Run tests**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: 7 tests pass (`test_npy_*` all pass). `test_dtype_roundtrip_each_dtype` still fails.

- [ ] **Step 5: Commit**

```bash
git add python/numpy/_io.py tests/python/test_io_npy.py
git commit -m "feat(io): add _npy_bytes_to_array (read path)"
```

---

## Chunk 2: Public API and .npz

### Task 4: `save()` / `load()` public API

**Files:**
- Modify: `python/numpy/_io.py:147-175`
- Modify: `tests/python/test_io_npy.py`

- [ ] **Step 1: Add save/load tests**

Add before the runner in `tests/python/test_io_npy.py`:

```python
def test_save_load_file_path():
    """save/load round-trip via file path."""
    path = _TMP + 'basic.npy'
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.save(path, a)
    b = np.load(path)
    _cleanup(path)
    assert b.shape == (2, 2)
    assert b.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_save_load_fileobj():
    """save/load round-trip via BytesIO file-like object."""
    import io
    a = np.array([10, 20, 30], dtype='int32')
    buf = io.BytesIO()
    np.save(buf, a)
    buf.seek(0)
    b = np.load(buf)
    assert b.shape == (3,)
    assert b.tolist() == [10, 20, 30]


def test_load_legacy_text():
    """Files written by the old text-based save() still load."""
    path = _TMP + 'legacy.npy'
    with open(path, 'w') as f:
        f.write('# shape: [3]\n')
        f.write('1.0,2.0,3.0\n')
    b = np.load(path)
    _cleanup(path)
    assert b.shape == (3,), f"expected (3,) got {b.shape}"
    assert abs(b.tolist()[0] - 1.0) < 1e-6


def test_save_object_raises():
    """Saving an object array raises ValueError."""
    path = _TMP + 'obj.npy'
    a = np.array(['hello', 'world'])
    try:
        np.save(path, a)
        _cleanup(path)
        assert False, "should have raised ValueError"
    except (ValueError, TypeError):
        pass
    _cleanup(path)
```

- [ ] **Step 2: Run — confirm failures**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: `test_save_load_file_path`, `test_save_load_fileobj` fail (load returns wrong values).

- [ ] **Step 3: Replace `save()` and `load()` in `_io.py` (lines 147–175)**

```python
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
        if file.lower().endswith('.npz'):
            return NpzFile(file)
        with open(file, 'rb') as f:
            first = f.read(1)
            f.seek(0)
            if first == b'\x93':
                return _npy_bytes_to_array(f.read())
            elif first == b'#':
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
        # File-like object — assume binary .npy
        raw = file.read()
        return _npy_bytes_to_array(raw)
```

- [ ] **Step 4: Run tests**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: `test_dtype_roundtrip_each_dtype`, `test_save_load_file_path`, `test_save_load_fileobj`, `test_load_legacy_text`, `test_save_object_raises` all pass.

- [ ] **Step 5: Commit**

```bash
git add python/numpy/_io.py tests/python/test_io_npy.py
git commit -m "feat(io): replace save()/load() with binary .npy implementation"
```

---

### Task 5: `savez()` / `savez_compressed()` / `NpzFile`

**Files:**
- Modify: `python/numpy/_io.py:178-194`
- Modify: `tests/python/test_io_npy.py`

- [ ] **Step 1: Add savez/NpzFile tests**

Add before the runner in `tests/python/test_io_npy.py`:

```python
def test_savez_positional():
    """savez positional args become arr_0, arr_1, ..."""
    path = _TMP + 'multi.npz'
    a = np.array([1, 2, 3], dtype='int32')
    b = np.array([4.0, 5.0], dtype='float64')
    np.savez(path, a, b)
    npz = np.load(path)
    _cleanup(path)
    assert isinstance(npz.files, list), f"files should be list, got {type(npz.files)}"
    assert 'arr_0' in npz.files, f"files: {npz.files}"
    assert 'arr_1' in npz.files
    assert npz['arr_0'].tolist() == [1, 2, 3]
    assert npz['arr_1'].tolist() == [4.0, 5.0]
    npz.close()


def test_savez_named():
    """savez keyword args preserve names."""
    path = _TMP + 'named.npz'
    x = np.array([10, 20], dtype='int64')
    y = np.array([1.5, 2.5], dtype='float32')
    np.savez(path, x=x, y=y)
    with np.load(path) as npz:
        assert 'x' in npz.files and 'y' in npz.files
        assert npz['x'].tolist() == [10, 20]
        assert abs(npz['y'].tolist()[0] - 1.5) < 1e-5
    _cleanup(path)


def test_savez_compressed():
    """savez_compressed produces valid .npz with correct data."""
    path = _TMP + 'compressed.npz'
    a = np.array(list(range(100)), dtype='float64')
    np.savez_compressed(path, data=a)
    with np.load(path) as npz:
        assert 'data' in npz.files
        vals = npz['data'].tolist()
        assert len(vals) == 100
        assert abs(vals[0] - 0.0) < 1e-10
        assert abs(vals[99] - 99.0) < 1e-10
    _cleanup(path)


def test_npzfile_interface():
    """NpzFile: files is list, __contains__, __iter__, context manager, KeyError."""
    path = _TMP + 'iface.npz'
    np.savez(path, a=np.array([1]), b=np.array([2]))
    npz = np.load(path)
    assert isinstance(npz.files, list)
    assert 'a' in npz
    assert 'b' in npz
    names = list(npz)
    assert set(names) == {'a', 'b'}
    try:
        _ = npz['missing']
        assert False, "should have raised KeyError"
    except KeyError:
        pass
    npz.close()
    _cleanup(path)
```

- [ ] **Step 2: Run — confirm failures**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: `test_savez_*` and `test_npzfile_interface` fail.

- [ ] **Step 3: Replace `savez`/`savez_compressed` with `NpzFile` + helpers in `_io.py` (lines 178–194)**

```python
class NpzFile:
    """Dict-like wrapper for .npz archives."""

    def __init__(self, file):
        self._zip = _zipfile.ZipFile(file, 'r')
        self.files = [
            name[:-4] for name in self._zip.namelist()
            if name.endswith('.npy')
        ]

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError(key)
        raw = self._zip.read(key + '.npy')
        return _npy_bytes_to_array(raw)

    def __iter__(self):
        return iter(self.files)

    def __contains__(self, key):
        return key in self.files

    def close(self):
        self._zip.close()

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
    compression = _zipfile.ZIP_DEFLATED if compress else _zipfile.ZIP_STORED
    with _zipfile.ZipFile(file, 'w', compression=compression) as zf:
        for name, arr in arrays.items():
            zf.writestr(name + '.npy', _array_to_npy_bytes(arr))


def savez(file, *args, **kwds):
    """Save arrays into an uncompressed .npz archive."""
    _savez_impl(file, args, kwds, compress=False)


def savez_compressed(file, *args, **kwds):
    """Save arrays into a compressed .npz archive."""
    _savez_impl(file, args, kwds, compress=True)
```

- [ ] **Step 4: Run tests**

```bash
./target/release/numpy-python tests/python/test_io_npy.py
```

Expected: all tests pass. Should see `test_io_npy: 14 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add python/numpy/_io.py tests/python/test_io_npy.py
git commit -m "feat(io): add NpzFile and binary savez/savez_compressed"
```

---

### Task 6: Full suite and README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full vendored test suite**

```bash
bash tests/python/run_tests.sh target/release/numpy-python
```

Expected: all pass. If any existing test uses `np.save`/`np.load` expecting text format and now fails, fix that test to use the new binary format.

- [ ] **Step 2: Remove the .npy/.npz limitation from README.md**

In `README.md`, find and remove this line from the Architecture limits section:

```
- **Binary `.npy`/`.npz`.** `np.save`/`np.load` use text. Binary format not implemented.
```

- [ ] **Step 3: Verify the line is gone**

```bash
grep -n "Binary\|text.*npy\|npy.*text" README.md
```

Expected: no matches.

- [ ] **Step 4: Run compat tests to confirm no regressions**

```bash
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci
```

Expected: 1211 passed, 0 unexpected failures.

- [ ] **Step 5: Commit README**

```bash
git add README.md
git commit -m "docs: remove npy/npz limitation — binary format now implemented"
```
