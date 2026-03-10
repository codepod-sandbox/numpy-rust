import numpy as np


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a!r} == {b!r}. {msg}")


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"Expected {a!r} close to {b!r} (tol={tol}). {msg}")


def test_ufunc_attributes():
    u = np.add
    assert_eq(u.__name__, "add")
    assert_eq(u.nin, 2)
    assert_eq(u.nout, 1)
    assert_eq(u.identity, 0)
    assert "add" in repr(u)

    m = np.multiply
    assert_eq(m.__name__, "multiply")
    assert_eq(m.identity, 1)


def test_ufunc_constructor():
    u = np.ufunc("add", 2, 1)
    assert_eq(u.__name__, "add")
    assert_eq(u.nin, 2)
    assert_eq(u.nout, 1)


def test_ufunc_reduce():
    a = np.array([1.0, 2.0, 3.0])
    assert_close(float(np.add.reduce(a)), 6.0)
    assert_close(float(np.multiply.reduce(a)), 6.0)
    assert_close(float(np.maximum.reduce(a)), 3.0)
    assert_close(float(np.minimum.reduce(a)), 1.0)
    assert_close(float(np.subtract.reduce(a)), -4.0)


def test_ufunc_accumulate():
    a = np.array([1.0, 2.0, 3.0])
    r = np.add.accumulate(a)
    assert_eq(r.tolist(), [1.0, 3.0, 6.0])


def test_ufunc_outer():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    r = np.add.outer(a, b)
    assert_eq(r.tolist(), [[4.0, 5.0], [5.0, 6.0]])
    assert_eq(r.shape, (2, 2))


def test_ufunc_reduceat():
    a = np.arange(6)
    r = np.add.reduceat(a, [0, 2, 4])
    assert_eq(r.tolist(), [1.0, 5.0, 9.0])


def test_ufunc_invalid_kwargs():
    try:
        np.add(1, 2, casting="nope")
        raise AssertionError("Expected TypeError for invalid kwarg")
    except TypeError:
        pass


if __name__ == "__main__":
    # simple runner
    passed = 0
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    print(f"test_ufunc: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
