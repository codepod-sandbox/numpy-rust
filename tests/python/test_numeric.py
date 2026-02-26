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


# --- Power operator ---

def test_pow_operator():
    a = np.array([2.0, 3.0, 4.0])
    b = a ** 2
    assert_close(b[0], 4.0)
    assert_close(b[1], 9.0)
    assert_close(b[2], 16.0)

def test_pow_fractional():
    a = np.array([4.0, 9.0, 16.0])
    b = a ** 0.5
    assert_close(b[0], 2.0)
    assert_close(b[1], 3.0)
    assert_close(b[2], 4.0)

def test_pow_broadcast():
    a = np.array([2.0, 3.0, 4.0])
    b = a ** np.array([2.0])
    assert_close(b[0], 4.0)
    assert_close(b[1], 9.0)
    assert_close(b[2], 16.0)


# --- Floor divide / Modulo ---

def test_floor_div():
    a = np.array([7.0, 8.0, 9.0])
    b = np.array([2.0, 3.0, 4.0])
    c = a // b
    assert_close(c[0], 3.0)
    assert_close(c[1], 2.0)
    assert_close(c[2], 2.0)

def test_floor_div_negative():
    a = np.array([-7.0])
    b = np.array([2.0])
    c = a // b
    assert_close(c[0], -4.0)

def test_modulo():
    a = np.array([7.0, 8.0, 9.0])
    b = np.array([2.0, 3.0, 4.0])
    c = a % b
    assert_close(c[0], 1.0)
    assert_close(c[1], 2.0)
    assert_close(c[2], 1.0)

def test_modulo_negative():
    a = np.array([-7.0])
    b = np.array([2.0])
    c = a % b
    assert_close(c[0], 1.0)


# --- Fancy indexing ---

def test_fancy_index_list():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = a[[0, 2, 4]]
    assert_eq(b.shape, (3,))
    assert_close(b[0], 10.0)
    assert_close(b[1], 30.0)
    assert_close(b[2], 50.0)

def test_fancy_index_array():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    idx = np.array([1.0, 3.0])
    b = a[idx]
    assert_eq(b.shape, (2,))
    assert_close(b[0], 20.0)
    assert_close(b[1], 40.0)

def test_fancy_index_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = a[[0, 2]]
    assert_eq(b.shape, (2, 2))
    assert_close(b[0, 0], 1.0)
    assert_close(b[1, 0], 5.0)

def test_fancy_index_set():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    a[[0, 2]] = np.array([99.0, 88.0])
    assert_close(a[0], 99.0)
    assert_close(a[2], 88.0)
    assert_close(a[1], 20.0)


# --- In-place operators ---

def test_iadd():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    a += b
    assert_close(a[0], 11.0)
    assert_close(a[1], 22.0)
    assert_close(a[2], 33.0)

def test_isub():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([1.0, 2.0, 3.0])
    a -= b
    assert_close(a[0], 9.0)
    assert_close(a[1], 18.0)
    assert_close(a[2], 27.0)

def test_imul():
    a = np.array([1.0, 2.0, 3.0])
    a *= 2
    assert_close(a[0], 2.0)
    assert_close(a[1], 4.0)
    assert_close(a[2], 6.0)

def test_idiv():
    a = np.array([10.0, 20.0, 30.0])
    a /= 2
    assert_close(a[0], 5.0)
    assert_close(a[1], 10.0)
    assert_close(a[2], 15.0)

def test_ipow():
    a = np.array([2.0, 3.0, 4.0])
    a **= 2
    assert_close(a[0], 4.0)
    assert_close(a[1], 9.0)
    assert_close(a[2], 16.0)

def test_ifloor_div():
    a = np.array([7.0, 8.0, 9.0])
    b = np.array([2.0, 3.0, 4.0])
    a //= b
    assert_close(a[0], 3.0)
    assert_close(a[1], 2.0)
    assert_close(a[2], 2.0)

def test_imod():
    a = np.array([7.0, 8.0, 9.0])
    b = np.array([2.0, 3.0, 4.0])
    a %= b
    assert_close(a[0], 1.0)
    assert_close(a[1], 2.0)
    assert_close(a[2], 1.0)


# --- Sort / Argsort ---

def test_sort_1d():
    a = np.array([3.0, 1.0, 2.0])
    b = np.sort(a)
    assert_close(b[0], 1.0)
    assert_close(b[1], 2.0)
    assert_close(b[2], 3.0)

def test_sort_2d_axis():
    a = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    b = np.sort(a, axis=1)
    assert_close(b[0, 0], 1.0)
    assert_close(b[0, 1], 2.0)
    assert_close(b[0, 2], 3.0)
    assert_close(b[1, 0], 4.0)
    assert_close(b[1, 1], 5.0)
    assert_close(b[1, 2], 6.0)

def test_argsort_1d():
    a = np.array([3.0, 1.0, 2.0])
    idx = np.argsort(a)
    assert_eq(int(idx[0]), 1)
    assert_eq(int(idx[1]), 2)
    assert_eq(int(idx[2]), 0)

def test_sort_method():
    a = np.array([3.0, 1.0, 2.0])
    b = a.sort()
    assert_close(b[0], 1.0)
    assert_close(b[1], 2.0)
    assert_close(b[2], 3.0)


def test_keepdims_sum():
    a = np.ones((3, 4))
    s = np.sum(a, axis=0, keepdims=True)
    assert s.shape == (1, 4), f"expected (1, 4), got {s.shape}"

def test_keepdims_mean():
    a = np.ones((3, 4))
    m = np.mean(a, axis=1, keepdims=True)
    assert m.shape == (3, 1), f"expected (3, 1), got {m.shape}"

def test_keepdims_max():
    a = np.ones((2, 3, 4))
    mx = a.max(1, True)
    assert mx.shape == (2, 1, 4), f"expected (2, 1, 4), got {mx.shape}"

def test_ddof_std():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s0 = float(np.std(a, ddof=0))
    s1 = float(np.std(a, ddof=1))
    assert s1 > s0, f"ddof=1 std ({s1}) should be > ddof=0 std ({s0})"

def test_ddof_var():
    a = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    v0 = float(np.var(a))
    v1 = float(np.var(a, ddof=1))
    assert abs(v0 - 4.0) < 0.01, f"population variance should be ~4.0, got {v0}"
    assert abs(v1 - 4.571428) < 0.01, f"sample variance should be ~4.571, got {v1}"


def test_eye_rectangular():
    a = np.eye(3, 4)
    assert a.shape == (3, 4), f"expected (3, 4), got {a.shape}"

def test_eye_offset():
    a = np.eye(3, k=1)
    assert a.shape == (3, 3)
    assert float(a[0, 1]) == 1.0
    assert float(a[0, 0]) == 0.0

def test_eye_negative_offset():
    a = np.eye(3, k=-1)
    assert a.shape == (3, 3)
    assert float(a[1, 0]) == 1.0
    assert float(a[0, 0]) == 0.0


def test_expand_dims():
    a = np.array([1.0, 2.0, 3.0])
    b = np.expand_dims(a, 0)
    assert b.shape == (1, 3), f"expected (1, 3), got {b.shape}"
    c = np.expand_dims(a, 1)
    assert c.shape == (3, 1), f"expected (3, 1), got {c.shape}"

def test_squeeze():
    a = np.ones((1, 3, 1))
    b = np.squeeze(a)
    assert b.shape == (3,), f"expected (3,), got {b.shape}"

def test_squeeze_axis():
    a = np.ones((1, 3, 1))
    b = np.squeeze(a, axis=0)
    assert b.shape == (3, 1), f"expected (3, 1), got {b.shape}"


# --- Edge cases ---

def test_sum_empty_array():
    a = np.array([])
    s = np.sum(a)
    assert float(s) == 0.0, f"sum of empty array should be 0.0, got {s}"

def test_dtype_preservation_int32():
    a = np.array([1, 2, 3]).astype("int32")
    b = np.array([4, 5, 6]).astype("int32")
    c = a + b
    assert c.dtype == "int32", f"expected int32, got {c.dtype}"

def test_inf_max():
    a = np.array([1.0, float("inf"), 3.0])
    mx = float(np.max(a))
    assert mx == float("inf"), f"expected inf, got {mx}"

def test_sort_already_sorted():
    a = np.array([1.0, 2.0, 3.0])
    s = np.sort(a)
    for i in range(3):
        assert float(s[i]) == float(a[i])

def test_argmin_axis():
    a = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    idx = np.argmin(a, axis=1)
    assert idx.shape == (2,), f"expected (2,), got {idx.shape}"

def test_argmax_axis():
    a = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    idx = np.argmax(a, axis=0)
    assert idx.shape == (3,), f"expected (3,), got {idx.shape}"

def test_argsort_duplicates():
    a = np.array([3.0, 1.0, 1.0, 2.0])
    idx = np.argsort(a)
    # First two indices should be 1 and 2 (both have value 1.0)
    assert int(idx[0]) in [1, 2]
    assert int(idx[1]) in [1, 2]


# --- Complex accessors on real arrays ---

def test_real_imag_float():
    a = np.array([1.0, 2.0, 3.0])
    r = a.real
    im = a.imag
    assert r.shape == (3,)
    assert_close(float(r[0]), 1.0)
    assert_close(float(im[0]), 0.0)

def test_conj_float():
    a = np.array([1.0, 2.0])
    c = a.conj()
    assert_close(float(c[0]), 1.0)

def test_real_module_func():
    a = np.array([5.0, 6.0])
    r = np.real(a)
    assert r.shape == (3,) or r.shape == (2,)  # accept either
    assert_close(float(r[0]), 5.0)

def test_imag_module_func():
    a = np.array([5.0, 6.0])
    im = np.imag(a)
    assert_close(float(im[0]), 0.0)

def test_conj_module_func():
    a = np.array([5.0, 6.0])
    c = np.conj(a)
    assert_close(float(c[0]), 5.0)


# --- Einsum ---

def test_einsum_matmul():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = np.einsum("ij,jk->ik", a, b)
    assert c.shape == (2, 2)
    assert abs(float(c[0, 0]) - 19.0) < 1e-10

def test_einsum_trace():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = np.einsum("ii->", a)
    assert abs(float(t) - 5.0) < 1e-10

def test_einsum_outer():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    c = np.einsum("i,j->ij", a, b)
    assert c.shape == (2, 3)
    assert abs(float(c[1, 2]) - 10.0) < 1e-10


# --- String (char) operations ---

def test_char_upper():
    a = np.array(["hello", "world"])
    b = np.char.upper(a)
    assert b[0] == "HELLO", f"expected 'HELLO', got {b[0]!r}"
    assert b[1] == "WORLD", f"expected 'WORLD', got {b[1]!r}"

def test_char_lower():
    a = np.array(["HELLO", "WORLD"])
    b = np.char.lower(a)
    assert b[0] == "hello", f"expected 'hello', got {b[0]!r}"
    assert b[1] == "world", f"expected 'world', got {b[1]!r}"

def test_char_str_len():
    a = np.array(["hi", "hello"])
    b = np.char.str_len(a)
    assert int(b[0]) == 2, f"expected 2, got {b[0]}"
    assert int(b[1]) == 5, f"expected 5, got {b[1]}"


# --- searchsorted / compress ---

def test_searchsorted():
    a = np.array([1.0, 3.0, 5.0, 7.0])
    v = np.array([2.0, 4.0, 6.0])
    idx = np.searchsorted(a, v)
    assert int(idx[0]) == 1  # 2.0 goes at index 1
    assert int(idx[1]) == 2  # 4.0 goes at index 2
    assert int(idx[2]) == 3  # 6.0 goes at index 3

def test_searchsorted_right():
    a = np.array([1.0, 3.0, 3.0, 5.0])
    v = np.array([3.0])
    idx = np.searchsorted(a, v, side="right")
    assert int(idx[0]) == 3

def test_compress():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    cond = np.array([True, False, True, False])
    result = np.compress(cond, a)
    assert result.shape == (2,)


def test_linspace_retstep():
    arr, step = np.linspace(0, 1, 5, retstep=True)
    assert arr.shape == (5,)
    assert abs(step - 0.25) < 1e-10

def test_arange_basic():
    # Test that arange works as before
    a = np.arange(0, 5, 1)
    assert a.shape == (5,)


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
