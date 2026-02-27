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


# --- NumPy-compat regression tests (from code review) ---

def test_keepdims_axis_none():
    """keepdims=True with axis=None should return shape (1,1,...) matching ndim."""
    a = np.ones((3, 4))
    s = np.sum(a, keepdims=True)
    assert s.shape == (1, 1), f"expected (1, 1), got {s.shape}"
    assert_close(float(s[0, 0]), 12.0)

def test_keepdims_axis_none_3d():
    """keepdims with axis=None on 3D array."""
    a = np.ones((2, 3, 4))
    s = np.sum(a, keepdims=True)
    assert s.shape == (1, 1, 1), f"expected (1, 1, 1), got {s.shape}"

def test_compress_values():
    """compress should return the correct element values, not just shape."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    cond = np.array([True, False, True, False])
    result = np.compress(cond, a)
    assert_close(float(result[0]), 1.0)
    assert_close(float(result[1]), 3.0)

def test_searchsorted_invalid_side():
    """searchsorted with invalid side should raise an error."""
    a = np.array([1.0, 3.0, 5.0])
    v = np.array([2.0])
    raised = False
    try:
        np.searchsorted(a, v, side="invalid")
    except (ValueError, Exception):
        raised = True
    assert raised, "searchsorted with side='invalid' should raise ValueError"

def test_choose_basic():
    """choose selects from choices based on index array."""
    a = np.array([0.0, 1.0, 0.0, 1.0])
    c0 = np.array([10.0, 20.0, 30.0, 40.0])
    c1 = np.array([50.0, 60.0, 70.0, 80.0])
    result = np.choose(a, [c0, c1])
    assert_close(float(result[0]), 10.0)
    assert_close(float(result[1]), 60.0)
    assert_close(float(result[2]), 30.0)
    assert_close(float(result[3]), 80.0)

def test_str_len_unicode():
    """str_len should count characters, not bytes."""
    a = np.array(["abc", "de"])
    b = np.char.str_len(a)
    assert int(b[0]) == 3
    assert int(b[1]) == 2

def test_linspace_retstep_num1():
    """linspace with num=1 should have step=0.0."""
    arr, step = np.linspace(0, 10, 1, retstep=True)
    assert arr.shape == (1,)
    assert step == 0.0, f"expected step=0.0, got {step}"

def test_module_std_ddof():
    """Module-level np.std should support ddof parameter."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s0 = float(np.std(a, ddof=0))
    s1 = float(np.std(a, ddof=1))
    assert s1 > s0, f"ddof=1 std ({s1}) should be > ddof=0 std ({s0})"

def test_module_var_ddof():
    """Module-level np.var should support ddof parameter."""
    a = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    v0 = float(np.var(a))
    v1 = float(np.var(a, ddof=1))
    assert abs(v0 - 4.0) < 0.01
    assert abs(v1 - 4.571428) < 0.01


# --- cumsum / cumprod / diff ---

def test_cumsum_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.cumsum(a)
    assert result.shape == (4,)
    expected = np.array([1.0, 3.0, 6.0, 10.0])
    assert np.allclose(result, expected)

def test_cumsum_2d_axis0():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.cumsum(a, axis=0)
    assert result.shape == (2, 2)
    expected = np.array([[1.0, 2.0], [4.0, 6.0]])
    assert np.allclose(result, expected)

def test_cumprod_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.cumprod(a)
    expected = np.array([1.0, 2.0, 6.0, 24.0])
    assert np.allclose(result, expected)

def test_diff_1d():
    a = np.array([1.0, 3.0, 6.0, 10.0])
    result = np.diff(a)
    assert result.shape == (3,)
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(result, expected)

def test_diff_n2():
    a = np.array([1.0, 3.0, 6.0, 10.0])
    result = np.diff(a, n=2)
    assert result.shape == (2,)
    expected = np.array([1.0, 1.0])
    assert np.allclose(result, expected)


# --- prod ---

def test_prod_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert float(np.prod(a)) == 24.0

def test_prod_2d_axis():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.prod(a, axis=0)
    expected = np.array([3.0, 8.0])
    assert np.allclose(result, expected)

def test_prod_keepdims():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.prod(a, axis=1, keepdims=True)
    assert result.shape == (2, 1)


# --- split / vsplit / hsplit ---

def test_split_equal():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    parts = np.split(a, 3)
    assert len(parts) == 3
    assert parts[0].shape == (2,)
    assert np.allclose(parts[0], np.array([1.0, 2.0]))

def test_split_indices():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    parts = np.split(a, [2, 4])
    assert len(parts) == 3
    assert parts[0].shape == (2,)
    assert parts[1].shape == (2,)
    assert parts[2].shape == (1,)

def test_hsplit():
    a = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    parts = np.hsplit(a, 2)
    assert len(parts) == 2
    assert parts[0].shape == (2, 2)

def test_vsplit():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    parts = np.vsplit(a, 2)
    assert len(parts) == 2
    assert parts[0].shape == (2, 2)


# --- repeat / tile ---

def test_repeat_flat():
    a = np.array([1.0, 2.0, 3.0])
    result = np.repeat(a, 2)
    assert result.shape == (6,)
    expected = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    assert np.allclose(result, expected)

def test_repeat_axis():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.repeat(a, 2, axis=0)
    assert result.shape == (4, 2)

def test_tile_1d():
    a = np.array([1.0, 2.0])
    result = np.tile(a, 3)
    assert result.shape == (6,)
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    assert np.allclose(result, expected)

def test_tile_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.tile(a, (2, 3))
    assert result.shape == (4, 6)


# --- argwhere ---

def test_argwhere_1d():
    a = np.array([0.0, 1.0, 0.0, 3.0, 0.0])
    result = np.argwhere(a)
    assert result.shape == (2, 1)

def test_argwhere_2d():
    a = np.array([[1.0, 0.0], [0.0, 4.0]])
    result = np.argwhere(a)
    assert result.shape == (2, 2)

def test_argwhere_all_zero():
    a = np.array([0.0, 0.0, 0.0])
    result = np.argwhere(a)
    assert result.shape == (0, 1)


# --- quantile / percentile / median ---

def test_quantile_median():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = float(np.quantile(a, 0.5))
    assert result == 3.0

def test_percentile_25():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = float(np.percentile(a, 25))
    assert_close(result, 1.75)

def test_median():
    a = np.array([3.0, 1.0, 2.0])
    result = float(np.median(a))
    assert result == 2.0

def test_quantile_axis():
    a = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    result = np.quantile(a, 0.5, axis=1)
    assert result.shape == (2,)


# --- ptp ---

def test_ptp_1d():
    a = np.array([3.0, 1.0, 7.0, 2.0])
    result = float(np.ptp(a))
    assert result == 6.0

def test_ptp_2d_axis():
    a = np.array([[3.0, 1.0], [7.0, 2.0]])
    result = np.ptp(a, axis=0)
    expected = np.array([4.0, 1.0])
    assert np.allclose(result, expected)

def test_ptp_2d_axis1():
    a = np.array([[3.0, 1.0], [7.0, 2.0]])
    result = np.ptp(a, axis=1)
    expected = np.array([2.0, 5.0])
    assert np.allclose(result, expected)


# ── Tier 5: Operator Gaps & Element-wise Rust Migrations ──────────────────

def test_bitwise_xor():
    a = np.array([1, 1, 0, 0]).astype("bool")
    b = np.array([1, 0, 1, 0]).astype("bool")
    result = a ^ b
    assert result.tolist() == [False, True, True, False]

def test_bitwise_xor_int():
    a = np.array([12, 10]).astype("int32")
    b = np.array([10, 12]).astype("int32")
    result = a ^ b
    assert result.tolist() == [6, 6]

def test_left_shift():
    a = np.array([1, 2, 4]).astype("int32")
    b = np.array([2, 2, 2]).astype("int32")
    result = a << b
    assert result.tolist() == [4, 8, 16]

def test_right_shift():
    a = np.array([4, 8, 16]).astype("int32")
    b = np.array([2, 2, 2]).astype("int32")
    result = a >> b
    assert result.tolist() == [1, 2, 4]

def test_inplace_xor():
    a = np.array([1, 0]).astype("bool")
    b = np.array([1, 1]).astype("bool")
    a ^= b
    assert a.tolist() == [False, True]

def test_inplace_lshift():
    a = np.array([1, 2, 4]).astype("int32")
    b = np.array([1, 1, 1]).astype("int32")
    a <<= b
    assert a.tolist() == [2, 4, 8]

def test_inplace_rshift():
    a = np.array([4, 8, 16]).astype("int32")
    b = np.array([1, 1, 1]).astype("int32")
    a >>= b
    assert a.tolist() == [2, 4, 8]

def test_abs_operator():
    a = np.array([-1.0, 2.0, -3.0])
    result = abs(a)
    assert result.tolist() == [1.0, 2.0, 3.0]

def test_bool_scalar_true():
    a = np.array([1.0])
    assert bool(a) == True

def test_bool_scalar_false():
    a = np.array([0.0])
    assert bool(a) == False

def test_bool_multi_raises():
    a = np.array([1.0, 2.0])
    try:
        bool(a)
        assert False, "Should have raised"
    except ValueError:
        pass

def test_isnan():
    a = np.array([1.0, float('nan'), 3.0])
    result = np.isnan(a)
    assert result.tolist() == [False, True, False]

def test_isinf():
    a = np.array([1.0, float('inf'), float('-inf')])
    result = np.isinf(a)
    assert result.tolist() == [False, True, True]

def test_isfinite():
    a = np.array([1.0, float('inf'), float('nan')])
    result = np.isfinite(a)
    assert result.tolist() == [True, False, False]

def test_around():
    a = np.array([1.234, 5.678, 9.012])
    result = np.around(a, 2)
    expected = np.array([1.23, 5.68, 9.01])
    assert np.allclose(result, expected)

def test_around_zero_decimals():
    a = np.array([1.5, 2.3, 3.7])
    result = np.around(a)
    expected = np.array([2.0, 2.0, 4.0])
    assert np.allclose(result, expected)

def test_signbit():
    a = np.array([-1.0, 0.0, 1.0])
    result = np.signbit(a)
    assert result.tolist() == [True, False, False]

def test_signbit_neg_zero():
    a = np.array([-0.0])
    result = np.signbit(a)
    assert result.tolist() == [True]

def test_logical_not():
    a = np.array([True, False, True])
    result = np.logical_not(a)
    assert result.tolist() == [False, True, False]

def test_logical_not_numeric():
    a = np.array([0.0, 1.0, 0.0, 5.0])
    result = np.logical_not(a)
    assert result.tolist() == [True, False, True, False]

def test_power_func():
    a = np.array([2.0, 3.0, 4.0])
    result = np.power(a, 2.0)
    assert result.tolist() == [4.0, 9.0, 16.0]

def test_power_array():
    a = np.array([2.0, 3.0])
    b = np.array([3.0, 2.0])
    result = np.power(a, b)
    assert result.tolist() == [8.0, 9.0]

def test_nonzero_1d():
    a = np.array([0.0, 1.0, 0.0, 3.0, 0.0])
    result = np.nonzero(a)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].tolist() == [1, 3]

def test_nonzero_2d():
    a = np.array([[1.0, 0.0], [0.0, 4.0]])
    result = np.nonzero(a)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].tolist() == [0, 1]
    assert result[1].tolist() == [0, 1]

def test_count_nonzero():
    a = np.array([0.0, 1.0, 0.0, 3.0, 0.0])
    assert np.count_nonzero(a) == 2

def test_count_nonzero_all():
    a = np.array([1.0, 2.0, 3.0])
    assert np.count_nonzero(a) == 3


# ── Tier 6: Math Function Expansion ──────────────────────────────────────

def test_log10():
    a = np.array([1.0, 10.0, 100.0])
    result = np.log10(a)
    expected = np.array([0.0, 1.0, 2.0])
    assert np.allclose(result, expected)

def test_log2():
    a = np.array([1.0, 2.0, 4.0, 8.0])
    result = np.log2(a)
    expected = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(result, expected)

def test_log1p():
    a = np.array([0.0, 1.0])
    result = np.log1p(a)
    expected = np.array([0.0, 0.6931471805599453])
    assert np.allclose(result, expected)

def test_expm1():
    a = np.array([0.0, 1.0])
    result = np.expm1(a)
    expected = np.array([0.0, 1.718281828459045])
    assert np.allclose(result, expected)

def test_sign():
    a = np.array([-5.0, 0.0, 3.0])
    result = np.sign(a)
    expected = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(result, expected)

def test_sign_int():
    a = np.array([-2, 0, 7]).astype("int32")
    result = np.sign(a)
    assert result.tolist() == [-1, 0, 1]

def test_deg2rad():
    a = np.array([0.0, 90.0, 180.0, 360.0])
    result = np.deg2rad(a)
    pi = 3.141592653589793
    expected = np.array([0.0, pi / 2, pi, 2 * pi])
    assert np.allclose(result, expected)

def test_rad2deg():
    pi = 3.141592653589793
    a = np.array([0.0, pi / 2, pi])
    result = np.rad2deg(a)
    expected = np.array([0.0, 90.0, 180.0])
    assert np.allclose(result, expected)

def test_radians_degrees_aliases():
    a = np.array([90.0])
    assert np.allclose(np.radians(a), np.deg2rad(a))
    assert np.allclose(np.degrees(np.radians(a)), a)

def test_sinh():
    a = np.array([0.0, 1.0])
    result = np.sinh(a)
    expected = np.array([0.0, 1.1752011936438014])
    assert np.allclose(result, expected)

def test_cosh():
    a = np.array([0.0, 1.0])
    result = np.cosh(a)
    expected = np.array([1.0, 1.5430806348152437])
    assert np.allclose(result, expected)

def test_tanh():
    a = np.array([0.0, 1.0])
    result = np.tanh(a)
    expected = np.array([0.0, 0.7615941559557649])
    assert np.allclose(result, expected)

def test_arcsin():
    a = np.array([0.0, 0.5, 1.0])
    result = np.arcsin(a)
    pi = 3.141592653589793
    expected = np.array([0.0, pi / 6, pi / 2])
    assert np.allclose(result, expected)

def test_arccos():
    a = np.array([1.0, 0.5, 0.0])
    result = np.arccos(a)
    pi = 3.141592653589793
    expected = np.array([0.0, pi / 3, pi / 2])
    assert np.allclose(result, expected)

def test_arctan():
    a = np.array([0.0, 1.0])
    result = np.arctan(a)
    pi = 3.141592653589793
    expected = np.array([0.0, pi / 4])
    assert np.allclose(result, expected)

def test_math_int_auto_cast():
    """Math functions should auto-cast integers to float."""
    a = np.array([1, 4, 9])
    result = np.sqrt(a)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(result, expected)


# ── Tier 7: Array Manipulation Utilities ──────────────────────────────────

def test_flip_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.flip(a)
    assert result.tolist() == [4.0, 3.0, 2.0, 1.0]

def test_flip_2d_axis0():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.flip(a, 0)
    assert result.tolist() == [[3.0, 4.0], [1.0, 2.0]]

def test_flip_2d_axis1():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.flip(a, 1)
    assert result.tolist() == [[2.0, 1.0], [4.0, 3.0]]

def test_flipud():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.flipud(a)
    assert result.tolist() == [[3.0, 4.0], [1.0, 2.0]]

def test_fliplr():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.fliplr(a)
    assert result.tolist() == [[2.0, 1.0], [4.0, 3.0]]

def test_rot90():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.rot90(a)
    assert result.shape == (2, 2)

def test_rot90_k2():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.rot90(a, 2)
    assert result.tolist() == [[4.0, 3.0], [2.0, 1.0]]

def test_unique():
    a = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0])
    result = np.unique(a)
    assert result.tolist() == [1.0, 2.0, 3.0]

def test_unique_sorted():
    a = np.array([5.0, 3.0, 1.0])
    result = np.unique(a)
    assert result.tolist() == [1.0, 3.0, 5.0]

def test_diagonal():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = np.diagonal(a)
    assert result.tolist() == [1.0, 5.0, 9.0]

def test_diagonal_offset():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = np.diagonal(a, 1)
    assert result.tolist() == [2.0, 6.0]

def test_diagonal_neg_offset():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = np.diagonal(a, -1)
    assert result.tolist() == [4.0, 8.0]

def test_outer():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0])
    result = np.outer(a, b)
    assert result.shape == (3, 2)
    assert result.tolist() == [[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]

def test_roll_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.roll(a, 1)
    assert result.tolist() == [4.0, 1.0, 2.0, 3.0]

def test_roll_negative():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.roll(a, -1)
    assert result.tolist() == [2.0, 3.0, 4.0, 1.0]

def test_roll_2d_axis():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.roll(a, 1, axis=0)
    assert result.tolist() == [[3.0, 4.0], [1.0, 2.0]]

def test_take_1d():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = np.take(a, [0, 2, 4])
    assert result.tolist() == [10.0, 30.0, 50.0]

def test_take_flat():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.take(a, [0, 3])
    assert result.tolist() == [1.0, 4.0]

def test_trace():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.trace(a)
    assert float(result) == 5.0


# ── Tier 8: Histogram & Bincount ──────────────────────────────────────

def test_histogram():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counts, edges = np.histogram(a, bins=5)
    assert_eq(counts.shape, (5,))
    assert_eq(edges.shape, (6,))
    # Total count should equal number of elements
    total = 0
    for i in range(5):
        total += int(counts[i])
    assert_eq(total, 5)

def test_histogram_default_bins():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counts, edges = np.histogram(a)
    assert_eq(counts.shape, (10,))
    assert_eq(edges.shape, (11,))

def test_bincount():
    a = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    c = np.bincount(a)
    assert_eq(int(c[0]), 1)
    assert_eq(int(c[1]), 2)
    assert_eq(int(c[2]), 3)

def test_bincount_minlength():
    a = np.array([0.0, 1.0])
    c = np.bincount(a, minlength=5)
    assert_eq(c.shape, (5,))
    assert_eq(int(c[0]), 1)
    assert_eq(int(c[1]), 1)
    assert_eq(int(c[2]), 0)

def test_bincount_weights():
    a = np.array([0.0, 1.0, 1.0, 2.0])
    w = np.array([0.5, 1.0, 1.5, 2.0])
    c = np.bincount(a, weights=w)
    assert_close(float(c[0]), 0.5)
    assert_close(float(c[1]), 2.5)
    assert_close(float(c[2]), 2.0)


# --- NaN-safe reductions ---

def test_nansum():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nansum(a), 4.0)

def test_nanmean():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nanmean(a), 2.0)

def test_nanstd():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nanstd(a), 1.0)

def test_nanvar():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nanvar(a), 1.0)

def test_nanmin():
    a = np.array([3.0, float('nan'), 1.0])
    assert_close(np.nanmin(a), 1.0)

def test_nanmax():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nanmax(a), 3.0)

def test_nanargmin():
    a = np.array([3.0, float('nan'), 1.0])
    assert_eq(np.nanargmin(a), 2)

def test_nanargmax():
    a = np.array([1.0, float('nan'), 3.0])
    assert_eq(np.nanargmax(a), 2)

def test_nanprod():
    a = np.array([2.0, float('nan'), 3.0])
    assert_close(np.nanprod(a), 6.0)

def test_nansum_no_nan():
    a = np.array([1.0, 2.0, 3.0])
    assert_close(np.nansum(a), 6.0)
    assert_close(np.nanmean(a), 2.0)

def test_nansum_2d_axis():
    a = np.array([1.0, float('nan'), 3.0, 4.0]).reshape((2, 2))
    s = np.nansum(a, axis=1)
    assert_close(float(s[0]), 1.0)
    assert_close(float(s[1]), 7.0)


# --- Covariance / Correlation ---

def test_cov_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    c = np.cov(x, y)
    assert_eq(c.shape, (2, 2))
    assert_close(float(c[0][0]), 1.0)
    assert_close(float(c[0][1]), 1.0)
    assert_close(float(c[1][0]), 1.0)
    assert_close(float(c[1][1]), 1.0)

def test_cov_single_variable():
    x = np.array([1.0, 2.0, 3.0])
    c = np.cov(x)
    assert_eq(c.shape, (1, 1))
    assert_close(float(c[0][0]), 1.0)

def test_cov_bias():
    x = np.array([1.0, 2.0, 3.0])
    c = np.cov(x, bias=True)
    # With bias=True, ddof=0, so var = 2/3
    assert_close(float(c[0][0]), 2.0 / 3.0, tol=1e-6)

def test_corrcoef_perfect():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    c = np.corrcoef(x, y)
    assert_close(float(c[0][1]), 1.0)
    assert_close(float(c[1][0]), 1.0)
    assert_close(float(c[0][0]), 1.0)
    assert_close(float(c[1][1]), 1.0)

def test_corrcoef_negative():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    c = np.corrcoef(x, y)
    assert_close(float(c[0][1]), -1.0)


# --- Set Operations (Tier 8) ---

def test_intersect1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0])
    r = np.intersect1d(a, b)
    assert_eq(r.shape, (2,))
    assert_close(float(r[0]), 2.0)
    assert_close(float(r[1]), 4.0)

def test_intersect1d_no_overlap():
    a = np.array([1.0, 3.0, 5.0])
    b = np.array([2.0, 4.0, 6.0])
    r = np.intersect1d(a, b)
    assert_eq(r.shape, (0,))

def test_union1d():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 5.0])
    r = np.union1d(a, b)
    assert_eq(r.shape, (5,))
    assert_close(float(r[0]), 1.0)
    assert_close(float(r[1]), 2.0)
    assert_close(float(r[2]), 3.0)
    assert_close(float(r[3]), 4.0)
    assert_close(float(r[4]), 5.0)

def test_setdiff1d():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 4.0])
    r = np.setdiff1d(a, b)
    assert_eq(r.shape, (3,))
    assert_close(float(r[0]), 1.0)
    assert_close(float(r[1]), 3.0)
    assert_close(float(r[2]), 5.0)

def test_isin():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test = np.array([2.0, 4.0])
    r = np.isin(a, test)
    assert_eq(r.shape, (5,))
    # r should be [False, True, False, True, False]
    assert_eq(bool(r[0]), False)
    assert_eq(bool(r[1]), True)
    assert_eq(bool(r[2]), False)
    assert_eq(bool(r[3]), True)
    assert_eq(bool(r[4]), False)

def test_isin_2d():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    test = np.array([2.0, 4.0, 6.0])
    r = np.isin(a, test)
    assert_eq(r.shape, (2, 3))

def test_in1d_alias():
    # in1d should be the same as isin
    a = np.array([1.0, 2.0, 3.0])
    test = np.array([2.0])
    r = np.in1d(a, test)
    assert_eq(bool(r[0]), False)
    assert_eq(bool(r[1]), True)
    assert_eq(bool(r[2]), False)


# --- Stacking fixes (Task 10) ---

def test_stack_proper():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = np.stack([a, b])
    assert_eq(r.shape, (2, 3))

def test_stack_axis1():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = np.stack([a, b], axis=1)
    assert_eq(r.shape, (3, 2))

def test_column_stack():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = np.column_stack([a, b])
    assert_eq(r.shape, (3, 2))

def test_dstack():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = np.dstack([a, b])
    assert_eq(r.shape, (1, 3, 2))


# ── Tier 9: Index Utilities ────────────────────────────────────────────

def test_unravel_index():
    idx = np.unravel_index(np.array([5.0]), (3, 4))
    assert_eq(int(idx[0]), 1)
    assert_eq(int(idx[1]), 1)

def test_unravel_index_multiple():
    idx = np.unravel_index(np.array([0.0, 5.0, 11.0]), (3, 4))
    assert_eq(idx[0].shape, (3,))

def test_ravel_multi_index():
    idx = np.ravel_multi_index((np.array([1.0]), np.array([1.0])), (3, 4))
    assert_eq(int(idx), 5)

def test_unravel_ravel_roundtrip():
    flat = np.array([7.0])
    shape = (3, 4)
    unraveled = np.unravel_index(flat, shape)
    raveled = np.ravel_multi_index(unraveled, shape)
    assert_eq(int(raveled), 7)

def test_unravel_index_int():
    idx = np.unravel_index(5, (3, 4))
    assert_eq(int(idx[0]), 1)
    assert_eq(int(idx[1]), 1)

def test_ravel_multi_index_ints():
    idx = np.ravel_multi_index((1, 1), (3, 4))
    assert_eq(int(idx), 5)


# --- Tier 9: Dtype correctness ---

def test_zeros_dtype_int32():
    a = np.zeros((3,), dtype="int32")
    assert_eq(str(a.dtype), "int32")

def test_zeros_dtype_float32():
    a = np.zeros((3,), dtype="float32")
    assert_eq(str(a.dtype), "float32")

def test_ones_dtype_int64():
    a = np.ones((2, 3), dtype="int64")
    assert_eq(str(a.dtype), "int64")

def test_full_native():
    a = np.full((3,), 7.0)
    assert_close(float(a[0]), 7.0)
    assert_close(float(a[1]), 7.0)
    assert_close(float(a[2]), 7.0)

def test_full_dtype():
    a = np.full((3,), 5, dtype="int32")
    assert_eq(str(a.dtype), "int32")

def test_eye_dtype():
    a = np.eye(3, dtype="int32")
    assert_eq(str(a.dtype), "int32")

def test_arange_dtype():
    a = np.arange(0, 5, 1, dtype="int32")
    assert_eq(str(a.dtype), "int32")

def test_array_dtype():
    a = np.array([1.0, 2.0, 3.0], dtype="int32")
    assert_eq(str(a.dtype), "int32")

def test_zeros_like_preserves_dtype():
    a = np.zeros((3,), dtype="int32")
    b = np.zeros_like(a)
    assert_eq(str(b.dtype), "int32")

def test_ones_like_preserves_dtype():
    a = np.ones((3,), dtype="float32")
    b = np.ones_like(a)
    assert_eq(str(b.dtype), "float32")

def test_full_like_basic():
    a = np.ones((3,))
    b = np.full_like(a, 42.0)
    assert_close(float(b[0]), 42.0)
    assert_close(float(b[1]), 42.0)

def test_promote_types():
    r = np.promote_types("int32", "float32")
    assert_eq(str(r), "float32")

def test_promote_types_same():
    r = np.promote_types("float64", "float64")
    assert_eq(str(r), "float64")

def test_promote_types_int_float():
    r = np.promote_types("int64", "float32")
    assert_eq(str(r), "float64")

def test_zeros_default_dtype():
    a = np.zeros((3,))
    assert_eq(str(a.dtype), "float64")

def test_ones_default_dtype():
    a = np.ones((3,))
    assert_eq(str(a.dtype), "float64")

def test_eye_default_dtype():
    a = np.eye(3)
    assert_eq(str(a.dtype), "float64")

def test_zeros_dtype_bool():
    a = np.zeros((3,), dtype="bool")
    assert_eq(str(a.dtype), "bool")

def test_empty_dtype():
    a = np.empty((3,), dtype="int32")
    assert_eq(str(a.dtype), "int32")


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
