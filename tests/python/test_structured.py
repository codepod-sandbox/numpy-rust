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

# Task 6: creation
sdt = np.dtype([('x', 'float64'), ('y', 'int32')])
data = [(1.0, 2), (3.0, 4), (5.0, 6)]
arr = np.array(data, dtype=sdt)
check("sa type", type(arr).__name__, 'StructuredArray')
check("sa shape", arr.shape, (3,))
check("sa ndim", arr.ndim, 1)
x_col = arr['x']
check("x_col[0]", float(x_col[0]), 1.0)
check("x_col[2]", float(x_col[2]), 5.0)
y_col = arr['y']
check("y_col[1]", int(y_col[1]), 4)

# zeros with structured dtype
zarr = np.zeros(3, dtype=sdt)
check("zeros sa type", type(zarr).__name__, 'StructuredArray')
zx = zarr['x']
check("zeros x[0]", float(zx[0]), 0.0)

print(f"passed: {passed}, failed: {failed}")
if failed:
    raise SystemExit(1)
