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

print(f"passed: {passed}, failed: {failed}")
if failed:
    raise SystemExit(1)
