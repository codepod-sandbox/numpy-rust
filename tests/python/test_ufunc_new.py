import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import numpy as np

def test_gcd_is_ufunc():
    assert isinstance(np.gcd, np.ufunc), "type={}".format(type(np.gcd))

def test_gcd_basic():
    r = np.gcd(np.array([12, 15, 0]), np.array([8, 10, 5]))
    assert list(r) == [4, 5, 5], "got {}".format(list(r))

def test_lcm_is_ufunc():
    assert isinstance(np.lcm, np.ufunc)

def test_lcm_basic():
    r = np.lcm(np.array([4, 6]), np.array([6, 4]))
    assert list(r) == [12, 12]

def test_divmod_is_ufunc():
    assert isinstance(np.divmod, np.ufunc)
    assert np.divmod.nout == 2

def test_divmod_basic():
    q, r = np.divmod(np.array([10, 11, 12]), np.array([3, 3, 3]))
    assert list(q) == [3, 3, 4]
    assert list(r) == [1, 2, 0]

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        try:
            t()
            print("PASS {}".format(t.__name__))
        except Exception as e:
            print("FAIL {}: {}".format(t.__name__, e))
