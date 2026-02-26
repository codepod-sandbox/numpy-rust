"""Tests for indexing, slicing, and shape manipulation — standard NumPy code."""
import numpy as np


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a!r} == {b!r}. {msg}")


def assert_close(a, b, tol=1e-7, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"Expected {a!r} close to {b!r} (tol={tol}). {msg}")


# --- Integer indexing ---

def test_getitem_1d():
    a = np.array([10.0, 20.0, 30.0])
    assert_close(a[0], 10.0)
    assert_close(a[1], 20.0)
    assert_close(a[2], 30.0)

def test_getitem_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_close(a[(0, 0)], 1.0)
    assert_close(a[(0, 1)], 2.0)
    assert_close(a[(1, 0)], 3.0)
    assert_close(a[(1, 1)], 4.0)

def test_getitem_negative():
    a = np.array([10.0, 20.0, 30.0])
    assert_close(a[-1], 30.0)


# --- Reshape ---

def test_reshape():
    a = np.arange(0.0, 12.0)
    b = a.reshape((3, 4))
    assert_eq(b.shape, (3, 4))
    assert_eq(b.ndim, 2)

def test_reshape_flat():
    a = np.ones((3, 4))
    b = a.reshape((12,))
    assert_eq(b.shape, (12,))


# --- Flatten / Ravel ---

def test_flatten():
    a = np.ones((2, 3))
    b = a.flatten()
    assert_eq(b.shape, (6,))
    assert_eq(b.ndim, 1)

def test_ravel():
    a = np.ones((2, 3))
    b = a.ravel()
    assert_eq(b.shape, (6,))


# --- Transpose ---

def test_transpose():
    a = np.ones((2, 3))
    b = a.T
    assert_eq(b.shape, (3, 2))

def test_transpose_3d():
    a = np.ones((2, 3, 4))
    b = a.T
    assert_eq(b.shape, (4, 3, 2))


# --- Copy ---

def test_copy():
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    assert_eq(b.shape, a.shape)
    assert_eq(b.dtype, a.dtype)


# --- astype ---

def test_astype_float32():
    a = np.array([1.0, 2.0, 3.0])
    b = a.astype("float32")
    assert_eq(b.dtype, "float32")

def test_astype_int64():
    a = np.array([1.0, 2.0, 3.0])
    b = a.astype("int64")
    assert_eq(b.dtype, "int64")


# --- len ---

def test_len():
    a = np.ones((5, 3))
    assert_eq(len(a), 5)

def test_len_1d():
    a = np.array([1.0, 2.0, 3.0])
    assert_eq(len(a), 3)


# --- Slice indexing (1D) ---

def test_slice_1d_range():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = a[1:4]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 20.0)
    assert_close(b[1], 30.0)
    assert_close(b[2], 40.0)

def test_slice_1d_from_start():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = a[:3]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 10.0)
    assert_close(b[2], 30.0)

def test_slice_1d_to_end():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = a[2:]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 30.0)
    assert_close(b[2], 50.0)

def test_slice_1d_step():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    b = a[::2]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 10.0)
    assert_close(b[1], 30.0)
    assert_close(b[2], 50.0)

def test_slice_1d_negative():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = a[-3:]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 30.0)
    assert_close(b[2], 50.0)

def test_slice_1d_reverse():
    a = np.array([10.0, 20.0, 30.0])
    b = a[::-1]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 30.0)
    assert_close(b[1], 20.0)
    assert_close(b[2], 10.0)

def test_slice_1d_full():
    a = np.array([10.0, 20.0, 30.0])
    b = a[:]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 10.0)
    assert_close(b[2], 30.0)

# --- Slice indexing (2D) ---

def test_slice_2d_rows():
    a = np.arange(0.0, 12.0).reshape((4, 3))
    b = a[1:3]
    assert_eq(b.shape, (2, 3))

def test_slice_2d_mixed():
    """Integer + slice in a tuple: a[0, 1:3]"""
    a = np.arange(0.0, 12.0).reshape((3, 4))
    b = a[(0, slice(1, 3))]
    assert_eq(b.shape, (2,))
    assert_close(b[0], 1.0)
    assert_close(b[1], 2.0)

def test_slice_2d_both():
    """Slice both axes: a[0:2, 1:3]"""
    a = np.arange(0.0, 12.0).reshape((3, 4))
    b = a[(slice(0, 2), slice(1, 3))]
    assert_eq(b.shape, (2, 2))

def test_slice_2d_step():
    """Step on both axes: a[::2, ::2]"""
    a = np.arange(0.0, 16.0).reshape((4, 4))
    b = a[(slice(None, None, 2), slice(None, None, 2))]
    assert_eq(b.shape, (2, 2))
    assert_close(b[(0, 0)], 0.0)
    assert_close(b[(0, 1)], 2.0)
    assert_close(b[(1, 0)], 8.0)
    assert_close(b[(1, 1)], 10.0)


# --- repr/str ---

def test_repr():
    a = np.zeros((2,))
    r = repr(a)
    assert_eq(type(r).__name__, "str")


# Run all tests
tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
passed = 0
failed = 0
for t in tests:
    try:
        t()
        passed += 1
    except Exception as e:
        print(f"FAIL {t.__name__}: {e}")
        failed += 1

print(f"test_indexing: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
