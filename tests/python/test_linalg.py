"""Tests for linalg, fft, random, and dot — standard NumPy code."""
import numpy as np


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a!r} == {b!r}. {msg}")


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"Expected {a!r} close to {b!r} (tol={tol}). {msg}")


# --- np.dot ---

def test_dot_1d():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.dot(a, b)
    assert_close(c, 32.0)

def test_dot_2d():
    a = np.eye(3)
    b = np.ones((3, 3))
    c = np.dot(a, b)
    assert_eq(c.shape, (3, 3))


# --- np.linalg ---

def test_linalg_inv():
    a = np.eye(3)
    b = np.linalg.inv(a)
    assert_eq(b.shape, (3, 3))

def test_linalg_det():
    a = np.eye(3)
    d = np.linalg.det(a)
    assert_close(d, 1.0)

def test_linalg_solve():
    a = np.eye(3)
    b = np.array([[1.0], [1.0], [1.0]])
    x = np.linalg.solve(a, b)
    assert_eq(x.shape, (3, 1))
    assert_close(x[(0, 0)], 1.0)

def test_linalg_norm():
    a = np.array([3.0, 4.0])
    n = np.linalg.norm(a)
    assert_close(n, 5.0)

def test_linalg_eig():
    a = np.eye(2)
    vals, vecs = np.linalg.eig(a)
    assert_eq(vals.shape, (2,))
    assert_eq(vecs.shape, (2, 2))

def test_linalg_svd():
    a = np.eye(2)
    u, s, vt = np.linalg.svd(a)
    assert_eq(u.shape, (2, 2))
    assert_eq(s.shape, (2,))
    assert_eq(vt.shape, (2, 2))

def test_linalg_qr():
    a = np.eye(3)
    q, r = np.linalg.qr(a)
    assert_eq(q.shape, (3, 3))
    assert_eq(r.shape, (3, 3))

def test_linalg_cholesky():
    a = np.eye(3)
    l = np.linalg.cholesky(a)
    assert_eq(l.shape, (3, 3))


# --- np.fft ---

def test_fft():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    f = np.fft.fft(a)
    assert_eq(f.shape, (4, 2))

def test_ifft_roundtrip():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    f = np.fft.fft(a)
    b = np.fft.ifft(f)
    assert_eq(b.shape, (4, 2))
    assert_close(b[(0, 0)], 1.0, tol=1e-6)

def test_fftfreq():
    f = np.fft.fftfreq(4, 1.0)
    assert_eq(f.shape, (4,))


# --- np.random ---

def test_random_seed():
    np.random.seed(42)

def test_random_rand():
    np.random.seed(0)
    a = np.random.rand((10,))
    assert_eq(a.shape, (10,))

def test_random_randn():
    np.random.seed(0)
    a = np.random.randn((5,))
    assert_eq(a.shape, (5,))

def test_random_randint():
    np.random.seed(0)
    a = np.random.randint(0, 10, (5,))
    assert_eq(a.shape, (5,))

def test_random_normal():
    np.random.seed(0)
    a = np.random.normal(0.0, 1.0, (100,))
    assert_eq(a.shape, (100,))

def test_random_uniform():
    np.random.seed(0)
    a = np.random.uniform(0.0, 1.0, (100,))
    assert_eq(a.shape, (100,))


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

print(f"test_linalg: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
