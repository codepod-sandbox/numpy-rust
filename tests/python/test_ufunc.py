"""Tests for the numpy ufunc protocol."""
import numpy as np
from numpy import array, ndarray


# --- isinstance and attributes ---

def test_ufunc_isinstance():
    """ufunc objects are instances of np.ufunc."""
    assert isinstance(np.add, np.ufunc)
    assert isinstance(np.sin, np.ufunc)
    assert isinstance(np.multiply, np.ufunc)

def test_ufunc_attributes():
    """ufunc objects have nin, nout, nargs, identity."""
    assert np.add.nin == 2
    assert np.add.nout == 1
    assert np.add.nargs == 3
    assert np.add.identity == 0
    assert np.multiply.identity == 1
    assert np.subtract.identity is None
    assert np.sin.nin == 1

def test_ufunc_repr():
    assert repr(np.add) == "<ufunc 'add'>"
    assert repr(np.sin) == "<ufunc 'sin'>"

def test_ufunc_name():
    assert np.add.__name__ == 'add'
    assert np.maximum.__name__ == 'maximum'


# --- __call__ preserves existing behavior ---

def test_call_binary():
    r = np.add(array([1, 2, 3]), array([4, 5, 6]))
    assert list(r) == [5, 7, 9]

def test_call_unary():
    r = np.negative(array([1, -2, 3]))
    assert list(r) == [-1, 2, -3]

def test_call_scalar():
    r = np.add(2, 3)
    assert float(r) == 5.0


# --- out= support ---

def test_call_out():
    a = array([1.0, 2.0, 3.0])
    b = array([4.0, 5.0, 6.0])
    out = array([0.0, 0.0, 0.0])
    result = np.add(a, b, out=out)
    assert list(out) == [5.0, 7.0, 9.0]
    # result should be the out array
    assert result is out

def test_call_out_tuple():
    """NumPy accepts out as a tuple of one array."""
    a = array([1.0, 2.0])
    out = array([0.0, 0.0])
    result = np.add(a, array([10.0, 20.0]), out=(out,))
    assert list(out) == [11.0, 22.0]
    assert result is out


# --- dtype= support ---

def test_call_dtype():
    """dtype= on __call__ casts result."""
    r = np.add(array([1, 2]), array([3, 4]), dtype='float64')
    assert str(r.dtype) == 'float64'


# --- reduce ---

def test_reduce_basic():
    assert float(np.add.reduce(array([1, 2, 3, 4]))) == 10.0

def test_reduce_axis():
    a = array([[1, 2], [3, 4]])
    r = np.add.reduce(a, axis=0)
    assert list(r) == [4, 6]

def test_reduce_axis1():
    a = array([[1, 2], [3, 4]])
    r = np.add.reduce(a, axis=1)
    assert list(r) == [3, 7]

def test_reduce_keepdims():
    a = array([[1, 2], [3, 4]])
    r = np.add.reduce(a, axis=0, keepdims=True)
    assert r.shape == (1, 2)

def test_reduce_multiply():
    assert float(np.multiply.reduce(array([1, 2, 3, 4]))) == 24.0

def test_reduce_maximum():
    assert float(np.maximum.reduce(array([3, 1, 4, 1, 5]))) == 5.0

def test_reduce_minimum():
    assert float(np.minimum.reduce(array([3, 1, 4, 1, 5]))) == 1.0

def test_reduce_subtract():
    """subtract.reduce([10, 3, 2]) = 10 - 3 - 2 = 5."""
    assert float(np.subtract.reduce(array([10, 3, 2]))) == 5.0

def test_reduce_logical_and():
    r = np.logical_and.reduce(array([True, True, False]))
    assert bool(r) == False

def test_reduce_logical_or():
    r = np.logical_or.reduce(array([False, False, True]))
    assert bool(r) == True

def test_reduce_dtype():
    """dtype= on reduce casts input before folding."""
    r = np.add.reduce(array([1, 2, 3]), dtype='float64')
    assert str(np.asarray(r).dtype) == 'float64'

def test_reduce_out():
    out = array([0.0])
    r = np.add.reduce(array([1, 2, 3]), out=out)
    assert float(out[0]) == 6.0
    assert r is out

def test_reduce_initial():
    """initial= provides a starting value."""
    r = np.add.reduce(array([1, 2, 3]), initial=10)
    assert float(r) == 16.0

def test_reduce_unary_raises():
    """Unary ufuncs cannot reduce."""
    try:
        np.sin.reduce(array([1, 2, 3]))
        assert False, "should have raised"
    except (ValueError, TypeError):
        pass

def test_reduce_generic_fallback():
    """Binary ufuncs without fast path still work via generic fold."""
    r = np.power.reduce(array([2, 3, 2]))
    # 2 ** 3 = 8, 8 ** 2 = 64
    assert float(r) == 64.0


# --- accumulate ---

def test_accumulate_add():
    r = np.add.accumulate(array([1, 2, 3, 4]))
    assert list(r) == [1, 3, 6, 10]

def test_accumulate_multiply():
    r = np.multiply.accumulate(array([1, 2, 3, 4]))
    assert list(r) == [1, 2, 6, 24]

def test_accumulate_maximum():
    r = np.maximum.accumulate(array([3, 1, 4, 1, 5]))
    assert list(r) == [3, 3, 4, 4, 5]

def test_accumulate_generic():
    """subtract.accumulate uses generic scan."""
    r = np.subtract.accumulate(array([10, 3, 2]))
    assert list(r) == [10, 7, 5]

def test_accumulate_axis():
    a = array([[1, 2], [3, 4], [5, 6]])
    r = np.add.accumulate(a, axis=0)
    assert list(r[0]) == [1, 2]
    assert list(r[1]) == [4, 6]
    assert list(r[2]) == [9, 12]

def test_accumulate_dtype():
    r = np.add.accumulate(array([1, 2, 3]), dtype='float64')
    assert str(r.dtype) == 'float64'

def test_accumulate_out():
    out = array([0.0, 0.0, 0.0])
    r = np.add.accumulate(array([1, 2, 3]), out=out)
    assert list(out) == [1.0, 3.0, 6.0]
    assert r is out

def test_accumulate_unary_raises():
    try:
        np.sin.accumulate(array([1, 2, 3]))
        assert False, "should have raised"
    except (ValueError, TypeError):
        pass


# --- outer ---

def test_outer_add():
    r = np.add.outer(array([1, 2, 3]), array([10, 20]))
    assert r.shape == (3, 2)
    assert list(r[0]) == [11, 21]
    assert list(r[1]) == [12, 22]
    assert list(r[2]) == [13, 23]

def test_outer_multiply():
    r = np.multiply.outer(array([1, 2, 3]), array([4, 5]))
    assert r.shape == (3, 2)
    assert list(r[0]) == [4, 5]
    assert list(r[1]) == [8, 10]
    assert list(r[2]) == [12, 15]

def test_outer_subtract():
    r = np.subtract.outer(array([10, 20]), array([1, 2, 3]))
    assert r.shape == (2, 3)
    assert list(r[0]) == [9, 8, 7]

def test_outer_unary_raises():
    try:
        np.sin.outer(array([1, 2]), array([3, 4]))
        assert False, "should have raised"
    except (ValueError, TypeError):
        pass


# --- reduceat ---

def test_reduceat_add():
    r = np.add.reduceat(array([0, 1, 2, 3, 4, 5, 6, 7]), [0, 4, 1, 5, 2, 6, 3, 7])
    # segments: [0:4]=6, [4]=4, [1:5]=10, [5]=5, [2:6]=14, [6]=6, [3:7]=18, [7]=7
    assert float(r[0]) == 6.0
    assert float(r[1]) == 4.0


# --- at ---

def test_at_add():
    a = array([1.0, 2.0, 3.0, 4.0])
    np.add.at(a, [0, 2], array([10.0, 20.0]))
    assert float(a[0]) == 11.0
    assert float(a[2]) == 23.0
    assert float(a[1]) == 2.0  # unchanged

def test_at_requires_b():
    """at() raises when b is not provided for a binary ufunc."""
    a = array([1.0, 2.0, 3.0])
    try:
        np.add.at(a, [0])
        assert False, "should have raised"
    except ValueError:
        pass


# --- aliases ---

def test_aliases_are_ufuncs():
    """Aliases like invert, abs, mod should also be ufunc instances."""
    assert isinstance(np.invert, np.ufunc)
    assert isinstance(np.absolute, np.ufunc)
    assert isinstance(np.mod, np.ufunc)
    assert np.abs is np.absolute
    assert np.invert is np.bitwise_not


# --- all binary ops are ufuncs ---

def test_all_binary_are_ufuncs():
    """All standard binary operations should be ufunc instances."""
    binary_names = [
        'add', 'subtract', 'multiply', 'divide', 'true_divide',
        'floor_divide', 'power', 'remainder',
        'maximum', 'minimum', 'fmax', 'fmin',
        'logical_and', 'logical_or', 'logical_xor',
        'bitwise_and', 'bitwise_or', 'bitwise_xor',
        'left_shift', 'right_shift',
        'greater', 'less', 'equal', 'not_equal',
        'greater_equal', 'less_equal',
        'arctan2', 'hypot', 'copysign', 'ldexp', 'heaviside',
        'nextafter', 'fmod',
    ]
    for name in binary_names:
        obj = getattr(np, name)
        assert isinstance(obj, np.ufunc), f"np.{name} is not a ufunc"
        assert obj.nin == 2, f"np.{name}.nin != 2"

def test_all_unary_are_ufuncs():
    """All standard unary operations should be ufunc instances."""
    unary_names = [
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh',
        'exp', 'exp2', 'log', 'log2', 'log10',
        'sqrt', 'cbrt', 'square', 'reciprocal',
        'negative', 'positive', 'absolute', 'sign',
        'floor', 'ceil', 'rint', 'trunc',
        'deg2rad', 'rad2deg', 'signbit',
        'logical_not', 'bitwise_not',
        'isnan', 'isinf', 'isfinite',
    ]
    for name in unary_names:
        obj = getattr(np, name)
        assert isinstance(obj, np.ufunc), f"np.{name} is not a ufunc"
        assert obj.nin == 1, f"np.{name}.nin != 1"


# --- Self-running test runner (RustPython has no pytest) ---
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

print(f"test_ufunc: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
