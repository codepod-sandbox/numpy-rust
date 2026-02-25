"""Tests for arithmetic, reductions, and unary math — standard NumPy code."""
import numpy as np


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a!r} == {b!r}. {msg}")


def assert_close(a, b, tol=1e-7, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"Expected {a!r} close to {b!r} (tol={tol}). {msg}")


# --- Arithmetic operators ---

def test_add():
    a = np.ones((3, 3))
    b = np.ones((3, 3))
    c = a + b
    assert_eq(c.shape, (3, 3))
    assert_close(c.sum(), 18.0)

def test_sub():
    a = np.ones((3,))
    b = np.zeros((3,))
    c = a - b
    assert_close(c.sum(), 3.0)

def test_mul():
    a = np.array([2.0, 3.0])
    b = np.array([4.0, 5.0])
    c = a * b
    assert_close(c[0], 8.0)
    assert_close(c[1], 15.0)

def test_div():
    a = np.array([10.0, 20.0])
    b = np.array([2.0, 5.0])
    c = a / b
    assert_close(c[0], 5.0)
    assert_close(c[1], 4.0)

def test_neg():
    a = np.ones((3,))
    b = -a
    assert_close(b.sum(), -3.0)


# --- Broadcasting ---

def test_broadcast_add():
    a = np.ones((3, 3))
    b = np.ones((3,))
    c = a + b
    assert_eq(c.shape, (3, 3))
    assert_close(c.sum(), 18.0)


# --- Comparisons ---

def test_eq():
    a = np.ones((3,))
    b = np.ones((3,))
    c = a.__eq__(b)
    assert_eq(c.all(), True)

def test_lt():
    a = np.zeros((3,))
    b = np.ones((3,))
    c = a.__lt__(b)
    assert_eq(c.all(), True)

def test_gt():
    a = np.ones((3,))
    b = np.zeros((3,))
    c = a.__gt__(b)
    assert_eq(c.all(), True)


# --- Reductions (no axis -> scalar) ---

def test_sum_scalar():
    a = np.ones((2, 3))
    s = a.sum()
    assert_close(s, 6.0)

def test_sum_axis0():
    a = np.ones((2, 3))
    s = a.sum(0)
    assert_eq(s.shape, (3,))

def test_sum_axis1():
    a = np.ones((2, 3))
    s = a.sum(1)
    assert_eq(s.shape, (2,))

def test_mean_scalar():
    a = np.array([2.0, 4.0, 6.0])
    assert_close(a.mean(), 4.0)

def test_min_scalar():
    a = np.array([3.0, 1.0, 2.0])
    assert_close(a.min(), 1.0)

def test_max_scalar():
    a = np.array([3.0, 1.0, 2.0])
    assert_close(a.max(), 3.0)

def test_argmin():
    a = np.array([3.0, 1.0, 2.0])
    assert_eq(a.argmin(), 1)

def test_argmax():
    a = np.array([3.0, 1.0, 2.0])
    assert_eq(a.argmax(), 0)

def test_std_scalar():
    a = np.array([1.0, 1.0, 1.0])
    assert_close(a.std(), 0.0)

def test_var_scalar():
    a = np.array([1.0, 1.0, 1.0])
    assert_close(a.var(), 0.0)


# --- all / any ---

def test_all_true():
    a = np.ones((3,))
    assert_eq(a.all(), True)

def test_all_false():
    a = np.zeros((3,))
    assert_eq(a.all(), False)

def test_any_true():
    a = np.array([0.0, 1.0, 0.0])
    assert_eq(a.any(), True)

def test_any_false():
    a = np.zeros((3,))
    assert_eq(a.any(), False)


# --- Unary math ---

def test_abs():
    a = np.array([-1.0, -2.0, 3.0])
    b = a.abs()
    assert_close(b.sum(), 6.0)

def test_sqrt():
    a = np.array([4.0, 9.0, 16.0])
    b = a.sqrt()
    assert_close(b[0], 2.0)
    assert_close(b[1], 3.0)
    assert_close(b[2], 4.0)


# --- float() / int() conversion ---

def test_float_conversion():
    a = np.array([3.14])
    assert_close(float(a), 3.14)

def test_int_conversion():
    a = np.array([42.0])
    assert_eq(int(a), 42)


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

print(f"test_numeric: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
