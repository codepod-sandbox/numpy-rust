# 2D Structured Arrays Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `StructuredArray` to support multi-D shapes (primarily 2D) so that `np.resize`/`np.zeros` work with structured dtypes and `TestResize.test_reshape_from_zero` passes.

**Architecture:** Rust stores data as 1D flat columns (unchanged). Python `StructuredArray` tracks a `_py_shape` overlay — `None` means use Rust shape (existing 1D behavior), a tuple means the array is logically multi-D. Rust gains `slice_rows(start, end)` so Python can extract contiguous row ranges for row-access on 2D arrays.

**Tech Stack:** Rust (`ndarray`, `rustpython_vm`), Python (RustPython), existing test harness (`./target/release/numpy-python`, `bash tests/python/run_tests.sh`, `./target/release/numpy-python tests/numpy_compat/run_compat.py --ci`)

---

## File Map

| File | Change |
|------|--------|
| `crates/numpy-rust-core/src/struct_array.rs` | Add `slice_rows(start, end)` method |
| `crates/numpy-rust-python/src/py_array.rs` | Make `py_slice_to_slice_arg` `pub(crate)` |
| `crates/numpy-rust-python/src/py_struct_array.rs` | Add `PySlice` branch in `__getitem__` |
| `python/numpy/__init__.py` | `StructuredArray`: add `_py_shape`, `reshape`, multi-D `__getitem__`/`__len__`/`__iter__` |
| `python/numpy/_creation.py` | `zeros`/`empty`/`full`: remove 1D restriction, create flat + `_py_shape` for multi-D |
| `python/numpy/_manipulation.py` | `resize`: add `StructuredArray` branch |
| `tests/python/test_structured.py` | Add 2D structured array tests |
| `tests/numpy_compat/xfail.txt` | Remove `TestResize.test_reshape_from_zero` |

---

## Chunk 1: Rust slice_rows + PySlice support

### Task 1: `slice_rows` in `struct_array.rs`

**Files:**
- Modify: `crates/numpy-rust-core/src/struct_array.rs`

Background: `StructArrayData` stores named field columns as `ArrayData`. `NdArray::from_data(col)` wraps it; `.slice(&[SliceArg::Range{start,stop,step}])` extracts a sub-range; `.data()` gets the `ArrayData` back. All defined in the same crate (`numpy-rust-core`).

- [ ] **Step 1: Write the failing Rust unit test**

Add at the bottom of the `#[cfg(test)]` block in `struct_array.rs`:

```rust
#[test]
fn test_slice_rows() {
    let sa = StructArrayData::new(
        vec![
            FieldSpec {
                name: "x".into(),
                data: make_int_col(vec![10, 20, 30, 40]),
            },
            FieldSpec {
                name: "y".into(),
                data: make_float_col(vec![1.1, 2.2, 3.3, 4.4]),
            },
        ],
        vec![4],
    );
    let sliced = sa.slice_rows(1, 3).unwrap();
    assert_eq!(sliced.len(), 2);
    assert_eq!(sliced.shape, vec![2]);
    let row0 = sliced.get_row(0).unwrap();
    assert!(matches!(row0[0], Scalar::Int64(20)));
    let row1 = sliced.get_row(1).unwrap();
    assert!(matches!(row1[0], Scalar::Int64(30)));
    // empty slice
    let empty = sa.slice_rows(2, 2).unwrap();
    assert_eq!(empty.len(), 0);
    // out of bounds
    assert!(sa.slice_rows(0, 5).is_err());
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p numpy-rust-core test_slice_rows 2>&1 | tail -5
```
Expected: error `no method named \`slice_rows\``

- [ ] **Step 3: Implement `slice_rows`**

Add after `set_field` in `struct_array.rs`:

```rust
/// Extract a contiguous range of rows [start, end) as a new StructArrayData.
pub fn slice_rows(&self, start: usize, end: usize) -> Result<Self> {
    use crate::indexing::SliceArg;
    let n = self.len();
    if end > n {
        return Err(NumpyError::ValueError(format!(
            "slice_rows: end {} out of bounds for array with {} rows",
            end, n
        )));
    }
    if start > end {
        return Err(NumpyError::ValueError(format!(
            "slice_rows: start {} > end {}",
            start, end
        )));
    }
    let sliced_fields: Vec<FieldSpec> = self
        .fields
        .iter()
        .map(|f| {
            let nd = NdArray::from_data(f.data.clone());
            let sliced = nd.slice(&[SliceArg::Range {
                start: Some(start as isize),
                stop: Some(end as isize),
                step: 1,
            }])?;
            Ok(FieldSpec {
                name: f.name.clone(),
                data: sliced.data().clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StructArrayData::new(sliced_fields, vec![end - start]))
}
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
cargo test -p numpy-rust-core test_slice_rows 2>&1 | tail -5
```
Expected: `test test_slice_rows ... ok`

- [ ] **Step 5: Commit**

```bash
git add crates/numpy-rust-core/src/struct_array.rs
git commit -m "feat: add StructArrayData::slice_rows for 2D row extraction"
```

---

### Task 2: `PySlice` support in `py_struct_array.rs`

**Files:**
- Modify: `crates/numpy-rust-python/src/py_array.rs` (make helper pub(crate))
- Modify: `crates/numpy-rust-python/src/py_struct_array.rs`

Background: `py_slice_to_slice_arg` in `py_array.rs` converts a `PySlice` to a `SliceArg`. We need it in `py_struct_array.rs` to resolve slice start/stop. The easiest path: make it `pub(crate)`.

- [ ] **Step 1: Write the Python-level slice test**

Add to `tests/python/test_structured.py` after the existing tests:

```python
# Task 2: slice indexing on 1D structured array
sdt = np.dtype([('x', 'float64'), ('y', 'int32')])
arr = np.array([(1.0, 2), (3.0, 4), (5.0, 6), (7.0, 8)], dtype=sdt)
sliced = arr[1:3]
check("slice type", type(sliced).__name__, 'StructuredArray')
check("slice shape", sliced.shape, (2,))
check("slice x[0]", float(sliced['x'][0]), 3.0)
check("slice x[1]", float(sliced['x'][1]), 5.0)
# empty slice
empty_sl = arr[2:2]
check("empty slice shape", empty_sl.shape, (0,))
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
bash tests/python/run_tests.sh 2>&1 | grep -A2 "slice type\|FAIL"
```
Expected: `FAIL slice type` (StructuredArray indices must be ... not slice)

- [ ] **Step 3: Make `py_slice_to_slice_arg` `pub(crate)` in `py_array.rs`**

Find line (near 3166):
```rust
fn py_slice_to_slice_arg(slice: &PySlice, vm: &VirtualMachine) -> PyResult<SliceArg> {
```
Change to:
```rust
pub(crate) fn py_slice_to_slice_arg(slice: &PySlice, vm: &VirtualMachine) -> PyResult<SliceArg> {
```

- [ ] **Step 4: Add `PySlice` + `PyTuple` imports and slice branch in `py_struct_array.rs`**

At the top of `py_struct_array.rs`, add to imports:
```rust
use vm::builtins::{PySlice, PyStr};
```
(replace the existing `use vm::builtins::PyStr;`)

Also add at top:
```rust
use crate::py_array::py_slice_to_slice_arg;
use numpy_rust_core::indexing::SliceArg;
```

In `__getitem__`, insert a new branch **before** the integer-key branch (after the list-of-strings branch, around line 215):

```rust
// Slice key -> extract contiguous row range
if let Some(slice) = key.downcast_ref::<PySlice>() {
    let n = self.inner.read().unwrap().len();
    let slice_arg = py_slice_to_slice_arg(slice, vm)?;
    let (start, end) = match slice_arg {
        SliceArg::Range { start, stop, step } => {
            if step != 1 {
                return Err(vm.new_value_error(
                    "StructuredArray slice step must be 1".to_owned(),
                ));
            }
            let s = start.unwrap_or(0);
            let e = stop.unwrap_or(n as isize);
            let s = if s < 0 { (n as isize + s).max(0) as usize } else { (s as usize).min(n) };
            let e = if e < 0 { (n as isize + e).max(0) as usize } else { (e as usize).min(n) };
            (s, e.max(s))
        }
        SliceArg::Full => (0, n),
        SliceArg::Index(_) => unreachable!("py_slice_to_slice_arg never returns Index"),
    };
    let sliced = {
        let inner = self.inner.read().unwrap();
        inner.slice_rows(start, end).map_err(|e| numpy_err(e, vm))?
    };
    return Ok(Self::new_from_data(sliced, self.dtype_json.clone()).into_pyobject(vm));
}
```

- [ ] **Step 5: Build**

```bash
cargo build --release 2>&1 | grep -E "^error" | head -10
```
Expected: no errors

- [ ] **Step 6: Run test to confirm it passes**

```bash
bash tests/python/run_tests.sh 2>&1 | grep -E "FAIL|passed|failed" | tail -5
```
Expected: new slice tests pass, same total pass count + 3

- [ ] **Step 7: Commit**

```bash
git add crates/numpy-rust-python/src/py_array.rs \
        crates/numpy-rust-python/src/py_struct_array.rs \
        tests/python/test_structured.py
git commit -m "feat: add PySlice support to PyStructuredArray.__getitem__"
```

---

## Chunk 2: Python 2D shape tracking

### Task 3: `StructuredArray` 2D shape tracking and indexing

**Files:**
- Modify: `python/numpy/__init__.py` (StructuredArray class, lines ~120–175)

Background on `StructuredArray`:
- `_native_arr` is a Rust `PyStructuredArray` — always 1D flat
- `_native_arr[i]` returns a Python list of scalars (row)
- `_native_arr['field']` returns a `PyNdArray` column
- `_native_arr[i:j]` now returns a 1D `PyStructuredArray` (after Task 2)
- `_native_arr.shape` returns a Python tuple, e.g. `(6,)`
- `_native_arr.dtype` returns JSON string

New attribute `_py_shape`: `None` → use Rust shape; tuple → logical multi-D shape. Must satisfy `product(_py_shape) == len(_native_arr)`.

- [ ] **Step 1: Write tests for 2D StructuredArray**

Add to `tests/python/test_structured.py`:

```python
# Task 3: 2D StructuredArray shape tracking
sdt2 = np.dtype([('a', 'float32')])
flat = np.array([(1.0,), (2.0,), (3.0,), (4.0,), (5.0,), (6.0,)], dtype=sdt2)

# reshape
arr2d = flat.reshape((2, 3))
check("2d shape", arr2d.shape, (2, 3))
check("2d ndim", arr2d.ndim, 2)
check("2d len", len(arr2d), 2)  # first dimension

# field access returns 2D ndarray
col = arr2d['a']
check("2d field shape", col.shape, (2, 3))

# integer row access on 2D returns 1D StructuredArray
row0 = arr2d[0]
check("2d row type", type(row0).__name__, 'StructuredArray')
check("2d row shape", row0.shape, (3,))
check("2d row field[0]", float(row0['a'][0]), 1.0)
check("2d row field[2]", float(row0['a'][2]), 3.0)

row1 = arr2d[1]
check("2d row1 field[0]", float(row1['a'][0]), 4.0)

# tuple indexing returns void scalar
elem = arr2d[0, 2]
check("2d elem type", type(elem).__name__, 'void')
check("2d elem value", float(elem['a']), 3.0)

# reshape back to 1D
flat2 = arr2d.reshape((6,))
check("reshape back shape", flat2.shape, (6,))
check("reshape back ndim", flat2.ndim, 1)
```

- [ ] **Step 2: Run to confirm failures**

```bash
bash tests/python/run_tests.sh 2>&1 | grep "FAIL" | head -10
```
Expected: multiple FAILs for 2d shape/field/row/elem tests (reshape probably errors out)

- [ ] **Step 3: Update `StructuredArray.__init__` to accept `_py_shape`**

In `python/numpy/__init__.py`, find the `StructuredArray` class. Replace `__init__`:

```python
def __init__(self, native_arr, py_shape=None):
    object.__setattr__(self, '_native_arr', native_arr)
    dt = _parse_dtype_json(native_arr.dtype)
    object.__setattr__(self, 'dtype', dt)
    # _py_shape: None = use Rust 1D shape; tuple = Python-tracked multi-D shape
    object.__setattr__(self, '_py_shape', tuple(py_shape) if py_shape is not None else None)
```

- [ ] **Step 4: Update `shape`, `ndim`, `__len__` properties**

Replace the `shape` property:
```python
@property
def shape(self):
    py_shape = object.__getattribute__(self, '_py_shape')
    if py_shape is not None:
        return py_shape
    return tuple(object.__getattribute__(self, '_native_arr').shape)
```

Replace `ndim`:
```python
@property
def ndim(self):
    return len(self.shape)
```

Replace `__len__`:
```python
def __len__(self):
    return self.shape[0]
```

- [ ] **Step 5: Add `reshape` method**

Add after `__len__`:
```python
def reshape(self, *new_shape):
    if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
        new_shape = tuple(new_shape[0])
    else:
        new_shape = tuple(new_shape)
    total = 1
    for s in new_shape:
        total *= s
    native = object.__getattribute__(self, '_native_arr')
    if total != len(native):
        raise ValueError(
            "cannot reshape structured array of size {} into shape {}".format(
                len(native), new_shape
            )
        )
    # If result is 1D, clear _py_shape so Rust shape is used
    if len(new_shape) == 1:
        return StructuredArray(native)
    return StructuredArray(native, py_shape=new_shape)
```

- [ ] **Step 6: Update `__getitem__` for multi-D**

Replace the existing `__getitem__` with:

```python
def __getitem__(self, key):
    native = object.__getattribute__(self, '_native_arr')
    dt = object.__getattribute__(self, 'dtype')
    shape = self.shape

    # Tuple key: (i, j) for 2D element access
    if isinstance(key, tuple) and len(shape) > 1:
        if len(key) != len(shape):
            raise IndexError(
                "too many indices for array: array is {}-dimensional, "
                "but {} were indexed".format(len(shape), len(key))
            )
        flat = 0
        stride = 1
        for k in range(len(shape) - 1, -1, -1):
            flat += key[k] * stride
            stride *= shape[k]
        result = native[flat]
        if isinstance(result, list):
            return void({n: v for n, v in zip(dt.names, result)}, dt)
        return result

    # Integer key on multi-D: return a row (1D StructuredArray)
    if isinstance(key, int) and len(shape) > 1:
        cols = shape[1]  # number of elements per row
        row_native = native[key * cols : (key + 1) * cols]
        if hasattr(row_native, 'field_names'):
            return StructuredArray(row_native)
        # fallback: shouldn't happen
        return row_native

    # All other cases: delegate to existing behavior
    result = native[key]
    dt = object.__getattribute__(self, 'dtype')

    # String key: field access — reshape if multi-D
    if isinstance(key, str):
        if len(shape) > 1:
            return result.reshape(shape)
        return result

    # Integer key (1D) → void scalar
    if isinstance(result, list):
        return void({n: v for n, v in zip(dt.names, result)}, dt)

    # Slice or list-of-strings → StructuredArray
    if hasattr(result, 'field_names'):
        return StructuredArray(result)

    return result
```

- [ ] **Step 7: Update `__iter__` for multi-D**

Replace `__iter__`:
```python
def __iter__(self):
    shape = self.shape
    if len(shape) > 1:
        for i in _builtin_range(shape[0]):
            yield self[i]
        return
    dt = object.__getattribute__(self, 'dtype')
    native = object.__getattribute__(self, '_native_arr')
    for i in _builtin_range(len(native)):
        row_list = native[i]
        yield void({n: v for n, v in zip(dt.names, row_list)}, dt)
```

Note: `_builtin_range` is exported by `_helpers` and available in `__init__.py` via `from ._helpers import *`. Use it here (it's the real builtin `range`, safe from any shadowing).

- [ ] **Step 8: Build and run tests**

```bash
cargo build --release 2>&1 | grep "^error" | head -5
bash tests/python/run_tests.sh 2>&1 | grep -E "FAIL|passed|failed" | tail -5
```
Expected: all 2D structured array tests pass

- [ ] **Step 9: Commit**

```bash
git add python/numpy/__init__.py tests/python/test_structured.py
git commit -m "feat: add 2D shape tracking to StructuredArray (reshape, row access, field reshape)"
```

---

## Chunk 3: Creation + resize + compat test

### Task 4: Multi-D `zeros`/`empty` for structured dtypes in `_creation.py`

**Files:**
- Modify: `python/numpy/_creation.py`

Background: `zeros` currently raises `ValueError` for multi-D structured shapes. The fix: remove that guard, compute `nrows = product(shape)`, create a flat `StructuredArray`, wrap with `_py_shape`.

Note: `empty` routes through `zeros` or has its own similar path at line ~517. Both need the same fix.

- [ ] **Step 1: Write test**

Add to `tests/python/test_structured.py`:

```python
# Task 4: zeros multi-D structured
sdt3 = np.dtype([('v', 'float64')])
z2d = np.zeros((2, 3), dtype=sdt3)
check("zeros 2d shape", z2d.shape, (2, 3))
check("zeros 2d ndim", z2d.ndim, 2)
check("zeros 2d len", len(z2d), 2)
check("zeros 2d v[0][0]", float(z2d[0]['v'][0]), 0.0)
check("zeros 2d v[1][2]", float(z2d[1, 2]['v']), 0.0)

# empty multi-D
e2d = np.empty((3, 2), dtype=sdt3)
check("empty 2d shape", e2d.shape, (3, 2))
```

- [ ] **Step 2: Run to confirm failures**

```bash
bash tests/python/run_tests.sh 2>&1 | grep "FAIL\|Error" | head -5
```
Expected: FAIL or ValueError for zeros 2d shape

- [ ] **Step 3: Fix `zeros` in `_creation.py`**

Find the `zeros` function (around line 331). Replace the structured array block:

```python
# OLD:
if _is_structured_dtype(parsed):
    if isinstance(shape, (list, tuple)) and len(shape) > 1:
        raise ValueError(
            "structured arrays only support 1D shape in this implementation; "
            "got shape {}".format(tuple(shape))
        )
    nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
    return _create_empty_structured(nrows, parsed, fill_value=0)

# NEW:
if _is_structured_dtype(parsed):
    if isinstance(shape, (list, tuple)) and len(shape) > 1:
        nrows = 1
        for s in shape:
            nrows *= s
        arr = _create_empty_structured(nrows, parsed, fill_value=0)
        return arr.reshape(tuple(shape))
    nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
    return _create_empty_structured(nrows, parsed, fill_value=0)
```

- [ ] **Step 4: Fix `empty` in `_creation.py`**

Find the `empty` function (around line 517). Same pattern:

```python
# OLD:
if _is_structured_dtype(parsed):
    if isinstance(shape, (list, tuple)) and len(shape) > 1:
        raise ValueError(
            "structured arrays only support 1D shape in this implementation; "
            "got shape {}".format(tuple(shape))
        )
    nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
    return _create_empty_structured(nrows, parsed, fill_value=fill_value)

# NEW:
if _is_structured_dtype(parsed):
    if isinstance(shape, (list, tuple)) and len(shape) > 1:
        nrows = 1
        for s in shape:
            nrows *= s
        arr = _create_empty_structured(nrows, parsed, fill_value=fill_value)
        return arr.reshape(tuple(shape))
    nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
    return _create_empty_structured(nrows, parsed, fill_value=fill_value)
```

- [ ] **Step 5: Fix `full` in `_creation.py`**

Find the `full` function (around line 514). Same pattern as `zeros`/`empty`:

```python
# OLD:
if _is_structured_dtype(parsed):
    if isinstance(shape, (list, tuple)) and len(shape) > 1:
        raise ValueError(
            "structured arrays only support 1D shape in this implementation; "
            "got shape {}".format(tuple(shape))
        )
    nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
    return _create_empty_structured(nrows, parsed, fill_value=fill_value)

# NEW:
if _is_structured_dtype(parsed):
    if isinstance(shape, (list, tuple)) and len(shape) > 1:
        nrows = 1
        for s in shape:
            nrows *= s
        arr = _create_empty_structured(nrows, parsed, fill_value=fill_value)
        return arr.reshape(tuple(shape))
    nrows = shape[0] if isinstance(shape, (list, tuple)) else shape
    return _create_empty_structured(nrows, parsed, fill_value=fill_value)
```

This also fixes `ones((2,3), dtype=structured_dt)` since `ones` delegates to `full`.

- [ ] **Step 6: Add `ones` test**

Add to `tests/python/test_structured.py`:

```python
# Task 4b: ones and full multi-D structured
o2d = np.ones((2, 2), dtype=np.dtype([('v', 'int32')]))
check("ones 2d shape", o2d.shape, (2, 2))
check("ones 2d val", int(o2d[0, 0]['v']), 1)
```

- [ ] **Step 7: Run tests**

```bash
bash tests/python/run_tests.sh 2>&1 | grep -E "FAIL|passed|failed" | tail -5
```
Expected: new zeros/empty/ones/full 2D tests pass

- [ ] **Step 8: Commit**

```bash
git add python/numpy/_creation.py tests/python/test_structured.py
git commit -m "feat: zeros/empty/full/ones support multi-D structured dtype shapes"
```

---

### Task 5: `resize` for `StructuredArray` + remove xfail

**Files:**
- Modify: `python/numpy/_manipulation.py`
- Modify: `tests/numpy_compat/xfail.txt`
- Modify: `tests/python/test_structured.py`

Background: The current `resize` calls `asarray(a)` which passes `StructuredArray` through unchanged, then calls `a.flatten().tolist()` — `StructuredArray` has no `flatten`. Fix: detect `StructuredArray` before the existing path, tile each field with `ndarray` resize, build new flat structured array, wrap with `_py_shape`.

The compat test:
```python
A = np.zeros(0, dtype=[('a', np.float32)])  # shape (0,)
Ar = np.resize(A, (2, 1))                   # shape (2, 1)
assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))
assert_equal(A.dtype, Ar.dtype)
```

- [ ] **Step 1: Write unit tests**

Add to `tests/python/test_structured.py`:

```python
# Task 5: resize structured array
sdt4 = np.dtype([('a', 'float32')])
A_empty = np.zeros(0, dtype=sdt4)
Ar = np.resize(A_empty, (2, 1))
check("resize empty shape", Ar.shape, (2, 1))
check("resize empty dtype", Ar.dtype, A_empty.dtype)
check("resize empty val", float(Ar[0, 0]['a']), 0.0)

# resize non-empty: tiling
A1 = np.array([(1.0,), (2.0,)], dtype=sdt4)
Ar2 = np.resize(A1, (2, 3))
check("resize tile shape", Ar2.shape, (2, 3))
check("resize tile [0,0]", float(Ar2[0, 0]['a']), 1.0)
check("resize tile [0,1]", float(Ar2[0, 1]['a']), 2.0)
check("resize tile [0,2]", float(Ar2[0, 2]['a']), 1.0)  # wraps around
check("resize tile [1,0]", float(Ar2[1, 0]['a']), 2.0)

# resize 1D
Ar3 = np.resize(A1, (4,))
check("resize 1d shape", Ar3.shape, (4,))
check("resize 1d [2]", float(Ar3['a'][2]), 1.0)
```

- [ ] **Step 2: Run to confirm failures**

```bash
bash tests/python/run_tests.sh 2>&1 | grep "FAIL\|Error" | head -5
```
Expected: FAILs or AttributeError on resize tests

- [ ] **Step 3: Add `StructuredArray` branch in `resize`**

In `_manipulation.py`, find `resize` (around line 481). Add a `StructuredArray` branch at the top, before `a = asarray(a)`:

```python
def resize(a, new_shape):
    from numpy import StructuredArray
    if isinstance(a, StructuredArray):
        return _resize_structured(a, new_shape)
    a = asarray(a)
    # ... existing code unchanged ...
```

Add the helper function just above `resize`:

```python
def _resize_structured(a, new_shape):
    """Resize a StructuredArray by tiling its fields to fill new_shape."""
    import json as _json
    from numpy import StructuredArray
    from _numpy_native import ndarray

    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    else:
        new_shape = tuple(new_shape)

    total = 1
    for s in new_shape:
        total *= s

    dt = a.dtype
    native = object.__getattribute__(a, '_native_arr')

    if total == 0:
        # zeros-filled empty structured array with target shape
        return zeros(new_shape, dtype=dt)

    # Build dtype_json for the new native array
    dtype_json = _json.dumps([[nm, str(dt.fields[nm][0])] for nm in dt.names])

    # Tile each field independently using ndarray resize
    new_fields = []
    for name in dt.names:
        col = native[name]  # PyNdArray, 1D
        if len(col) == 0:
            # source is empty: fill with zeros
            tiled = zeros(total, dtype=str(col.dtype))
            if not isinstance(tiled, ndarray):
                tiled = asarray([0] * total).astype(str(col.dtype))
        else:
            tiled = _native_resize(col, total)
        new_fields.append((name, tiled))

    # Build native StructuredArray
    import _numpy_native as _native_mod
    native_fields = [(name, col) for name, col in new_fields]
    new_native = _native_mod.StructuredArray(native_fields, [total], dtype_json)
    flat = StructuredArray(new_native)

    if len(new_shape) == 1:
        return flat
    return flat.reshape(new_shape)
```

Note: `_native_resize` is the existing `resize` function applied to a 1D `ndarray`. Add a small helper at the top of `_manipulation.py`:

```python
def _native_resize(col, total):
    """Tile a 1D ndarray to length total."""
    n = len(col)
    flat = col.flatten().tolist()
    result_vals = [flat[i % n] for i in _builtin_range(total)]
    return asarray(result_vals).astype(str(col.dtype))
```

`_manipulation.py` already has `import _numpy_native as _native` at the top — use `_native.StructuredArray(...)` directly.

- [ ] **Step 4: Run unit tests**

```bash
bash tests/python/run_tests.sh 2>&1 | grep -E "FAIL|passed|failed" | tail -5
```
Expected: all resize structured tests pass

- [ ] **Step 5: Run compat test in isolation**

```bash
./target/release/numpy-python -c "
import numpy as np
A = np.zeros(0, dtype=[('a', np.float32)])
Ar = np.resize(A, (2, 1))
print('shape:', Ar.shape)
print('dtype:', Ar.dtype)
print('val:', Ar[0,0])
" 2>&1
```
Expected: `shape: (2, 1)`, no error

- [ ] **Step 6: Remove from xfail.txt**

Edit `tests/numpy_compat/xfail.txt`:
```
# Remove this line:
TestResize.test_reshape_from_zero
```

- [ ] **Step 7: Run full compat suite**

```bash
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -5
```
Expected: `N passed, 0 unexpected failures, 3 expected failures (xfail)` — the test now passes, so there should be 3 xfails (not 4).

- [ ] **Step 8: Commit**

```bash
git add python/numpy/_manipulation.py \
        tests/python/test_structured.py \
        tests/numpy_compat/xfail.txt
git commit -m "feat: resize/zeros support 2D structured arrays; fix test_reshape_from_zero xfail"
```

---

## Final verification

- [ ] **Run all test suites**

```bash
cargo test --release 2>&1 | tail -5
bash tests/python/run_tests.sh 2>&1 | tail -5
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -5
```

Expected:
- Rust: all tests pass (including `test_slice_rows`)
- Python: all tests pass
- Compat: N passed, 0 unexpected failures, **3** expected failures (xfail)
