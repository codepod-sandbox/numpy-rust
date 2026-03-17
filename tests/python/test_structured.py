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

# Task 7: row access, field assignment, iteration
sdt3 = np.dtype([('name_x', 'float64'), ('name_y', 'int32')])
arr3 = np.array([(10.0, 1), (20.0, 2), (30.0, 3)], dtype=sdt3)

# Row access → void
row0 = arr3[0]
check("row0 type", type(row0).__name__, 'void')
check("row0['name_x']", float(row0['name_x']), 10.0)
check("row0.name_y", int(row0.name_y), 1)

# Negative indexing
row_last = arr3[-1]
check("row_last['name_x']", float(row_last['name_x']), 30.0)

# Field assignment
arr3['name_x'] = np.array([100.0, 200.0, 300.0])
check("after assign arr3['name_x'][0]", float(arr3['name_x'][0]), 100.0)

# Multi-field subset
sub = arr3[['name_x', 'name_y']]
check("sub type", type(sub).__name__, 'StructuredArray')
check("sub.dtype.names", sub.dtype.names, ('name_x', 'name_y'))

# Iteration
rows = list(arr3)
check("iter len", len(rows), 3)
check("iter row0 type", type(rows[0]).__name__, 'void')
check("iter row2 name_y", int(rows[2]['name_y']), 3)

# Empty array (n=0)
empty_sa = np.zeros(0, dtype=sdt3)
check("empty shape", empty_sa.shape, (0,))
check("empty len", len(empty_sa), 0)

print(f"passed: {passed}, failed: {failed}")
if failed:
    raise SystemExit(1)
