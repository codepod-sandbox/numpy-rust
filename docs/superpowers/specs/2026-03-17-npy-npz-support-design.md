# .npy / .npz Binary Format Support Design

## Goal

Replace the current text-based `save`/`load`/`savez` stubs with a proper binary implementation of the NumPy .npy and .npz formats, enabling full interoperability with files created by CPython NumPy.

## Architecture

All changes are confined to `python/numpy/_io.py`. No Rust changes are needed. `struct`, `zipfile`, `ast`, and `io.BytesIO` are all confirmed available in RustPython. Serialization uses `arr.flatten().tolist()` + `struct.pack` — no `tobytes()` call required.

## .npy Format Spec

```
[magic: 6 bytes b'\x93NUMPY']
[major: 1 byte] [minor: 1 byte]
[header_len: 2 bytes LE for v1.x, 4 bytes LE for v2.x]
[header: ASCII dict string, space-padded so that (10 + header_len) % 64 == 0, ends with '\n']
[data: raw bytes, C-contiguous, little-endian]
```

Header example: `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }\n`

We **write** version 1.0 always. We **read** version 1.x and 2.x.

## Dtype Coverage

Confirmed working in RustPython (dtype preserved, `tolist()` returns correct Python values):

| Our dtype    | .npy descr | struct char |
|--------------|------------|-------------|
| `float16`    | `<f2`      | `e`         |
| `float32`    | `<f4`      | `f`         |
| `float64`    | `<f8`      | `d`         |
| `int8`       | `\|i1`     | `b`         |
| `int16`      | `<i2`      | `h`         |
| `int32`      | `<i4`      | `i`         |
| `int64`      | `<i8`      | `q`         |
| `uint8`      | `\|u1`     | `B`         |
| `uint16`     | `<u2`      | `H`         |
| `uint32`     | `<u4`      | `I`         |
| `uint64`     | `<u8`      | `Q`         |
| `bool`       | `\|b1`     | `?`         |
| `complex128` | `<c16`     | `dd` (re, im pairs) |

**Not supported:** complex64 (our backend promotes it to complex128), object, string, datetime arrays.
Loading a `<c8` file from real NumPy returns a `complex128` array with a warning.
Object/string/datetime arrays raise `ValueError: cannot save object arrays to .npy`.

## Components

### `_dtype_to_descr(dtype_str) -> (descr, struct_char, itemsize)`
Maps dtype name to (.npy descriptor string, struct format char, bytes per element).
Raises `ValueError` for unsupported types.

### `_descr_to_dtype(descr) -> str`
Reverse map from .npy descriptor to our dtype name. Handles both endian prefixes (`<`, `>`, `|`, `=`).
Raises `ValueError` for unrecognised descriptors.

### `_array_to_npy_bytes(arr) -> bytes`
Builds complete .npy binary:
1. Map `arr.dtype` to descr and struct char
2. Build header string, pad with spaces so `(10 + len(header_bytes)) % 64 == 0`, end with `\n`
3. Build: magic + `\x01\x00` + 2-byte LE header_len + header bytes
4. Flatten array and pack with `struct.pack('<' + struct_char * n, *flat)`
5. For complex128: interleave re/im — `struct.pack('<dd', v.real, v.imag)` per element
6. For 0-d arrays: shape is `()`, single value is packed

### `_npy_bytes_to_array(data: bytes) -> ndarray`
Parses .npy binary:
1. Verify first 6 bytes == `b'\x93NUMPY'`; raise `ValueError` otherwise
2. Read major/minor version bytes
3. If major == 1: header_len = 2-byte LE; if major == 2: header_len = 4-byte LE
4. Read header bytes, decode as latin-1, parse with `ast.literal_eval`
5. Extract `descr`, `fortran_order`, `shape` from parsed dict
6. Map descr to dtype and struct char
7. Compute element count = product of shape (1 for 0-d)
8. Unpack bytes with `struct.unpack`
9. If `fortran_order`: unpack values are in column-major order; reshape accordingly
10. Reshape to `shape` (empty tuple for 0-d → return 0-d array)

### `save(file, arr, **kwargs)`
- `arr = asarray(arr)`
- Write `_array_to_npy_bytes(arr)` in binary mode
- `file` may be a path string or a writable binary file-like object

### `load(file, mmap_mode=None, **kwargs)`
- If `file` is a **string** ending in `.npz` (case-insensitive): return `NpzFile(file)`
- If `file` is a **file-like object**: read first byte to detect format; can't detect npz by extension so assume .npy
- Open path in `'rb'` mode; read first byte:
  - `0x93` → binary .npy → `_npy_bytes_to_array(entire_contents)`
  - `b'#'` → legacy text format → re-open in `'r'` mode and use existing text parser
  - Anything else → raise `ValueError: unknown file format`
- `mmap_mode` is accepted and ignored

### `savez(file, *args, **kwds)`
- Build name→array dict: positional args → `arr_0`, `arr_1`, …; keyword args keep names
- If `file` is a string without `.npz` suffix, append it
- Open `zipfile.ZipFile(file, 'w', compression=ZIP_STORED)`
- Write each array as `{name}.npy` entry using `_array_to_npy_bytes`

### `savez_compressed(file, *args, **kwds)`
Same as `savez` but uses `ZIP_DEFLATED` compression.

### `NpzFile` class

```python
class NpzFile:
    def __init__(self, file):
        self._zip = zipfile.ZipFile(file, 'r')
        self.files = [n[:-4] for n in self._zip.namelist() if n.endswith('.npy')]

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError(key)
        data = self._zip.read(key + '.npy')
        return _npy_bytes_to_array(data)

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
```

`files` is a plain list (not a property) set at construction time, matching real NumPy's interface.
Arrays are loaded on demand — no caching.

## Backward Compatibility

`load()` detects the first byte: `0x93` = binary .npy, `b'#'` = old text format (re-read in text mode).

## Error Handling

| Situation | Error |
|-----------|-------|
| Bad magic bytes | `ValueError: not a .npy file` |
| Unsupported dtype | `ValueError: cannot save <dtype> arrays to .npy` |
| Key missing from NpzFile | `KeyError` |
| `<c8` file loaded | `UserWarning`, returns complex128 array |

## Testing

New file: `tests/python/test_io_npy.py`

| Test | What it checks |
|------|----------------|
| `test_save_load_float64` | Basic round-trip |
| `test_save_load_all_dtypes` | All 13 supported dtypes: value and dtype preserved |
| `test_save_load_2d` | 2D shape preserved |
| `test_save_load_0d` | Shape `()` round-trips correctly |
| `test_save_load_empty` | Shape `(0,)` array |
| `test_save_load_to_fileobj` | File-like object (BytesIO) as target |
| `test_savez_positional` | `arr_0`, `arr_1` naming |
| `test_savez_named` | Keyword arg names preserved |
| `test_savez_compressed` | Compression flag; data identical to uncompressed |
| `test_npzfile_files_is_list` | `isinstance(npz.files, list)`, `'arr_0' in npz.files` |
| `test_npzfile_iter_context` | `__iter__`, context manager closes zip |
| `test_load_legacy_text` | File written by old text `save()` still loads |
| `test_save_object_raises` | Object array raises ValueError |
| `test_interop_read` | Read .npy bytes created by real NumPy (embedded bytes literal) |
