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


# --- Iteration ---

def test_iter_1d():
    a = np.array([1.0, 2.0, 3.0])
    result = list(a)
    assert_eq(len(result), 3)
    assert_close(result[0], 1.0)
    assert_close(result[1], 2.0)
    assert_close(result[2], 3.0)

def test_iter_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    rows = list(a)
    assert_eq(len(rows), 2)
    assert_eq(rows[0].shape, (2,))
    assert_close(rows[0][0], 1.0)
    assert_close(rows[1][1], 4.0)

def test_iter_in_sum():
    a = np.array([1.0, 2.0, 3.0])
    assert_close(sum(a), 6.0)


# --- Boolean operators ---

def test_bitwise_and():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = (a > 1.0) & (a < 5.0)
    assert_eq(mask.dtype, "bool")
    # mask should be [False, True, True, True, False]
    assert_eq(mask[0], False)
    assert_eq(mask[1], True)
    assert_eq(mask[4], False)

def test_bitwise_or():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = (a < 2.0) | (a > 4.0)
    assert_eq(mask.dtype, "bool")
    # mask should be [True, False, False, False, True]
    assert_eq(mask[0], True)
    assert_eq(mask[1], False)
    assert_eq(mask[4], True)

def test_bitwise_not():
    a = np.array([1.0, 2.0, 3.0])
    mask = a > 1.0
    inv = ~mask
    assert_eq(inv.dtype, "bool")
    assert_eq(inv[0], True)
    assert_eq(inv[1], False)

def test_compound_filter():
    a = np.array([1.0, 5.0, 10.0, 15.0, 3.0])
    result = a[(a > 2.0) & (a < 12.0)]
    assert_eq(result.shape, (3,))
    assert_close(result[0], 5.0)
    assert_close(result[1], 10.0)
    assert_close(result[2], 3.0)


# --- Module-level math ---

def test_np_abs():
    a = np.array([-1.0, -2.0, 3.0])
    b = np.abs(a)
    assert_close(b[0], 1.0)
    assert_close(b[1], 2.0)
    assert_close(b[2], 3.0)

def test_np_sqrt():
    a = np.array([4.0, 9.0, 16.0])
    b = np.sqrt(a)
    assert_close(b[0], 2.0)
    assert_close(b[1], 3.0)

def test_np_exp():
    a = np.array([0.0, 1.0])
    b = np.exp(a)
    assert_close(b[0], 1.0)
    assert_close(b[1], 2.718281828, tol=1e-5)

def test_np_log():
    a = np.array([1.0, 2.718281828])
    b = np.log(a)
    assert_close(b[0], 0.0)
    assert_close(b[1], 1.0, tol=1e-5)

def test_np_sin():
    a = np.array([0.0])
    b = np.sin(a)
    assert_close(b[0], 0.0)

def test_np_cos():
    a = np.array([0.0])
    b = np.cos(a)
    assert_close(b[0], 1.0)

def test_np_sum():
    a = np.array([1.0, 2.0, 3.0])
    assert_close(np.sum(a), 6.0)

def test_np_mean():
    a = np.array([2.0, 4.0, 6.0])
    assert_close(np.mean(a), 4.0)


# --- Matmul operator ---

def test_matmul_operator():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    # C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_close(C[0, 0], 19.0)
    assert_close(C[0, 1], 22.0)
    assert_close(C[1, 0], 43.0)
    assert_close(C[1, 1], 50.0)

def test_matmul_matvec():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.array([1.0, 1.0])
    result = A @ v
    # [1+2, 3+4] = [3, 7]
    assert_close(result[0], 3.0)
    assert_close(result[1], 7.0)


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
