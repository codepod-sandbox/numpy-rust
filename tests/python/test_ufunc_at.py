import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import numpy as np
from numpy.testing import assert_array_equal

def test_at_binary_basic():
    a = np.arange(10)
    np.add.at(a, [2, 5, 2], 1)
    assert list(a) == [0, 1, 4, 3, 4, 6, 6, 7, 8, 9]

def test_at_unary():
    a = np.arange(10)
    np.negative.at(a, [2, 5, 3])
    assert list(a) == [0, 1, -2, -3, 4, -5, 6, 7, 8, 9]

def test_at_negative_indices_float():
    a = np.arange(10, dtype=np.float64)
    np.add.at(a, np.array([-1, 1, -1, 2], dtype=np.intp),
              np.array([1., 5., 2., 10.]))
    assert a[9] == 12.0   # 9 + 1 + 2
    assert a[1] == 6.0    # 1 + 5
    assert a[2] == 12.0   # 2 + 10

def test_at_negative_indices_int():
    a = np.arange(10, dtype=np.int32)
    np.add.at(a, np.array([-1, 1, -1, 2], dtype=np.intp),
              np.array([1, 5, 2, 10], dtype=np.int32))
    assert a[9] == 12
    assert a[1] == 6
    assert a[2] == 12

def test_at_preserves_int_dtype():
    a = np.array([1, 2, 3], dtype=np.int32)
    np.add.at(a, [0, 1], np.array([10, 20], dtype=np.int32))
    assert a.dtype == np.int32
    assert a[0] == 11 and a[1] == 22

def test_at_binary_no_b_raises():
    a = np.arange(5)
    try:
        np.add.at(a, [0, 1])
        assert False, "should have raised"
    except (ValueError, TypeError):
        pass

def test_at_unary_with_b_raises():
    a = np.arange(5)
    try:
        np.negative.at(a, [0, 1], [1, 2])
        assert False, "should have raised"
    except (ValueError, TypeError):
        pass

def test_at_nout_gt1_raises():
    a = np.arange(10, dtype=float)
    try:
        np.modf.at(a, [1])
        assert False, "should have raised"
    except (ValueError, TypeError):
        pass

def test_at_array_b():
    a = np.arange(10)
    np.add.at(a, [2, 5, 2], np.array([100, 100, 100]))
    assert a[2] == 202 and a[5] == 105

def test_at_broadcast_failure():
    a = np.arange(5)
    try:
        np.add.at(a, [0, 1], [1, 2, 3])
        assert False, "should have raised ValueError"
    except ValueError:
        pass

def test_at_output_casting():
    arr = np.array([-1])
    np.equal.at(arr, [0], [0])
    assert arr[0] == 0  # equal(-1,0)=False cast to int = 0

def test_at_slice_index():
    arr = np.zeros(5)
    np.add.at(arr, slice(None), np.ones(5))
    assert list(arr) == [1., 1., 1., 1., 1.]

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
