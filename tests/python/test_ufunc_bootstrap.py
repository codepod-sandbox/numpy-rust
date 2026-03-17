import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import numpy as np

def test_bitwise_count_scalar():
    a = np.array([0, 1, 2, 3, 255], dtype='int32')
    r = np.bitwise_count(a)
    assert list(r) == [0, 1, 1, 2, 8], f"got {list(r)}"

def test_bitwise_count_is_ufunc():
    assert isinstance(np.bitwise_count, np.ufunc)

def test_bitwise_count_has_O_type():
    assert 'O->O' in np.bitwise_count.types

def test_ufunc_public_constructor_func():
    def myfunc(a, b): return a + b
    u = np.ufunc(myfunc, name='myfunc', nin=2, nout=1)
    assert isinstance(u, np.ufunc)
    assert u.__name__ == 'myfunc'
    assert u.nin == 2

def test_ufunc_public_constructor_str():
    u = np.ufunc("dummy", 1, 1, types=["O->O"])
    assert isinstance(u, np.ufunc)
    assert u.__name__ == "dummy"
    assert u.types == ["O->O"]
    assert u.ntypes == 1

def test_ufunc_suite_remove_works():
    # Simulate what test_ufunc.py:37 does
    UNARY_UFUNCS = [obj for obj in np.__dict__.values()
                    if isinstance(obj, np.ufunc)]
    UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]
    UNARY_OBJECT_UFUNCS.remove(np.bitwise_count)  # must NOT raise ValueError
    assert np.bitwise_count not in UNARY_OBJECT_UFUNCS

def test_types_populated():
    assert 'ff->f' in np.add.types, f"np.add.types = {np.add.types}"
    assert 'dd->d' in np.add.types
    assert np.add.ntypes == len(np.add.types)
    assert len(np.sin.types) >= 2
    # comparison ufuncs
    assert 'ff->?' in np.greater.types or 'dd->?' in np.greater.types, f"np.greater.types = {np.greater.types}"

if __name__ == '__main__':
    test_bitwise_count_scalar()
    test_bitwise_count_is_ufunc()
    test_bitwise_count_has_O_type()
    test_ufunc_public_constructor_func()
    test_ufunc_public_constructor_str()
    test_ufunc_suite_remove_works()
    test_types_populated()
    print("All bootstrap tests passed")
