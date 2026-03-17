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
