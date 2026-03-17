import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import numpy as np
import warnings

def test_where_param_with_out():
    a = np.arange(7)
    b = np.ones(7)
    c = np.zeros(7)
    np.add(a, b, out=c, where=(a % 2 == 1))
    assert list(c) == [0, 2, 0, 4, 0, 6, 0], "got {}".format(list(c))

def test_where_warns_without_out():
    a = np.arange(7)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = np.add(a, a, where=(a % 2 == 0))
    assert len(w) >= 1 and issubclass(w[0].category, UserWarning)

def test_serialize_roundtrip():
    r = np.sin.__reduce__()
    assert r[0] is not None
    assert r[1] == ('numpy', 'sin')

def test_invalid_args_typeerror():
    try:
        np.sqrt(None)
        assert False
    except TypeError as e:
        assert 'loop of ufunc' in str(e) or 'NoneType' in str(e)

def test_logical_any_dtype():
    a = np.array([1., 0., 1.])
    b = np.array([1., 1., 0.])
    r = np.logical_and(a, b)
    assert list(r) == [True, False, False]

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        try:
            t()
            print("PASS {}".format(t.__name__))
        except Exception as e:
            print("FAIL {}: {}".format(t.__name__, e))
