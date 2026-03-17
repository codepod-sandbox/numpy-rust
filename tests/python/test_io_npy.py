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


def test_npy_bytes_magic_and_header():
    """_array_to_npy_bytes produces valid magic, version, and header."""
    from numpy._io import _array_to_npy_bytes, _MAGIC
    import ast, struct
    a = np.array([1.0, 2.0, 3.0])  # float64
    data = _array_to_npy_bytes(a)
    assert data[:6] == _MAGIC, "bad magic"
    assert data[6] == 1 and data[7] == 0, "expected version 1.0"
    header_len = struct.unpack_from("<H", data, 8)[0]
    assert (10 + header_len) % 64 == 0, "header not 64-byte aligned"
    header_str = data[10:10 + header_len].decode("latin-1").strip()
    hdr = ast.literal_eval(header_str)
    assert hdr["descr"] == "<f8"
    assert hdr["fortran_order"] == False
    assert hdr["shape"] == (3,)
    vals = struct.unpack_from("<3d", data, 10 + header_len)
    assert abs(vals[0] - 1.0) < 1e-10
    assert abs(vals[1] - 2.0) < 1e-10
    assert abs(vals[2] - 3.0) < 1e-10


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
    # header_len must satisfy (10 + header_len) % 64 == 0 -> header_len = 118
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


# Runner
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
