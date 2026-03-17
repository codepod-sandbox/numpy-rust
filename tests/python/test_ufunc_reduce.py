import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import numpy as np
from numpy.testing import assert_array_equal

def test_reduce_basic():
    a = np.ones((5, 2), dtype=int)
    r = np.add.reduce(a)
    assert list(r) == [5, 5]

def test_reduce_out_wrong_shape_raises():
    a = np.arange(12.).reshape(4, 3)
    out = np.empty((1, 1), a.dtype)
    try:
        np.add.reduce(a, axis=0, out=out)
        assert False, "should have raised ValueError"
    except ValueError:
        pass

def test_reduce_out_correct_shape_returned():
    a = np.arange(12.).reshape(4, 3)
    out = np.empty((3,), a.dtype)
    r = np.add.reduce(a, axis=0, out=out)
    assert r is out
    assert list(r) == [18., 22., 26.]

def test_reduce_keepdims_out():
    a = np.arange(12.).reshape(4, 3)
    out = np.empty((1, 3), a.dtype)
    r = np.add.reduce(a, axis=0, out=out, keepdims=True)
    assert r is out
    assert r.shape == (1, 3)

def test_reduce_where():
    a = np.arange(9.).reshape(3, 3)
    where = np.array([[True, False, True],
                      [True, False, True],
                      [True, False, True]])
    r = np.add.reduce(a, axis=0, where=where, initial=0.)
    assert r[0] == 9.0   # 0+3+6
    assert r[2] == 15.0  # 2+5+8

def test_reduce_empty_axis():
    a = np.arange(6.).reshape(2, 3)
    r = np.add.reduce(a, axis=())
    assert r.shape == a.shape
    assert_array_equal(r, a)

def test_reduce_invalid_axis_raises():
    try:
        np.add.reduce(np.ones((5, 2)), axis="invalid")
        assert False
    except TypeError:
        pass

def test_accumulate_out_wrong_shape_raises():
    a = np.arange(5)
    out = np.arange(3)
    try:
        np.add.accumulate(a, out=out)
        assert False
    except ValueError:
        pass

def test_reduceat_out_wrong_shape_raises():
    a = np.arange(5)
    out = np.arange(3)
    try:
        np.add.reduceat(a, [0, 3], out=out)
        assert False
    except ValueError:
        pass

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        try:
            t()
            print("PASS {}".format(t.__name__))
        except Exception as e:
            print("FAIL {}: {}".format(t.__name__, e))
