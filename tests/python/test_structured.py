"""Tests for structured array support."""
import numpy as np

passed = 0
failed = 0

def check(name, got, expected):
    global passed, failed
    if got == expected:
        passed += 1
    else:
        print(f"FAIL {name}: got {repr(got)}, expected {repr(expected)}")
        failed += 1

# Task 4: dtype.descr
dt = np.dtype([('x', 'float64'), ('y', 'int32')])
check("dtype.names", dt.names, ('x', 'y'))
check("dtype.fields keys", set(dt.fields.keys()), {'x', 'y'})
check("dtype.descr", dt.descr, [('x', 'float64'), ('y', 'int32')])

# Task 5: void scalar
dt2 = np.dtype([('a', 'int64'), ('b', 'float64')])
row = np.void({'a': 1, 'b': 2.5}, dt2)
check("void getitem", row['a'], 1)
check("void getattr", row.b, 2.5)
check("void repr", row.__repr__(), repr((1, 2.5)))

print(f"passed: {passed}, failed: {failed}")
if failed:
    raise SystemExit(1)
