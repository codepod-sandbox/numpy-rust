"""Tests for array creation functions — standard NumPy code."""
import numpy as np


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a!r} == {b!r}. {msg}")


def assert_close(a, b, tol=1e-7, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"Expected {a!r} close to {b!r} (tol={tol}). {msg}")


# --- np.zeros ---

def test_zeros_1d():
    a = np.zeros((5,))
    assert_eq(a.shape, (5,))
    assert_eq(a.dtype, "float64")
    assert_eq(a.ndim, 1)
    assert_eq(a.size, 5)

def test_zeros_2d():
    a = np.zeros((3, 4))
    assert_eq(a.shape, (3, 4))
    assert_eq(a.size, 12)

def test_zeros_3d():
    a = np.zeros((2, 3, 4))
    assert_eq(a.ndim, 3)
    assert_eq(a.size, 24)


# --- np.ones ---

def test_ones_1d():
    a = np.ones((3,))
    assert_eq(a.shape, (3,))
    assert_close(a.sum(), 3.0)

def test_ones_2d():
    a = np.ones((2, 3))
    assert_eq(a.shape, (2, 3))
    assert_eq(a.size, 6)


# --- np.array ---

def test_array_1d():
    a = np.array([1.0, 2.0, 3.0])
    assert_eq(a.shape, (3,))
    assert_eq(a.ndim, 1)

def test_array_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_eq(a.shape, (2, 2))
    assert_eq(a.ndim, 2)

def test_array_values():
    a = np.array([10.0, 20.0, 30.0])
    assert_close(a[0], 10.0)
    assert_close(a[1], 20.0)
    assert_close(a[2], 30.0)


# --- np.arange ---

def test_arange_basic():
    a = np.arange(0.0, 5.0)
    assert_eq(a.shape, (5,))
    assert_close(a[0], 0.0)
    assert_close(a[4], 4.0)

def test_arange_step():
    a = np.arange(0.0, 10.0, 2.0)
    assert_eq(a.shape, (5,))


# --- np.linspace ---

def test_linspace():
    a = np.linspace(0.0, 1.0, 11)
    assert_eq(a.shape, (11,))
    assert_close(a[0], 0.0)
    assert_close(a[10], 1.0)

def test_linspace_3():
    a = np.linspace(0.0, 10.0, 3)
    assert_eq(a.shape, (3,))
    assert_close(a[1], 5.0)


# --- np.eye ---

def test_eye():
    a = np.eye(3)
    assert_eq(a.shape, (3, 3))
    assert_close(a[(0, 0)], 1.0)
    assert_close(a[(0, 1)], 0.0)


# --- np.concatenate ---

def test_concatenate_1d():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    c = np.concatenate([a, b])
    assert_eq(c.shape, (4,))
    assert_close(c[0], 1.0)
    assert_close(c[3], 4.0)


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

print(f"test_array_creation: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
