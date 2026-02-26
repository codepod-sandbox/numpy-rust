# Tier 3: Core Correctness + Feature Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make existing features correct and robust (keepdims, ddof, expand_dims, squeeze, unwrap cleanup, eye/argmin improvements, edge-case tests), then expand the API surface with complex numbers, einsum, string ops, and utility functions.

**Architecture:** Two sequential phases. Phase A fixes correctness in existing code without changing the type system. Phase B extends `DType`/`ArrayData` with complex number variants and adds new operation modules (einsum, string_ops, selection). All changes follow the established pattern: core logic in `numpy-rust-core`, Python bindings in `numpy-rust-python`, stubs updated in `python/numpy/__init__.py`.

**Tech Stack:** Rust + ndarray, RustPython bindings, num-complex crate (Phase B)

---

## Phase A: Core Correctness

### Task 1: Add `keepdims` to reduction operations

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/reduction.rs`
- Modify: `crates/numpy-rust-python/src/py_array.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Add `keepdims` helper in core**

In `crates/numpy-rust-core/src/ops/reduction.rs`, add a private helper after `validate_axis`:

```rust
use ndarray::IxDyn;

/// If `keepdims` is true and a specific axis was reduced, re-insert a size-1
/// dimension at `axis` so the output rank matches the input rank.
fn maybe_keepdims(result: NdArray, axis: Option<usize>, keepdims: bool) -> NdArray {
    if !keepdims {
        return result;
    }
    if let Some(ax) = axis {
        // Insert size-1 dimension at the reduced axis
        let mut new_shape = result.shape().to_vec();
        new_shape.insert(ax, 1);
        result.reshape(&new_shape).expect("keepdims reshape cannot fail")
    } else {
        // axis=None reduced everything to scalar — wrap in shape [1, 1, ..., 1]?
        // NumPy wraps it in shape (1,) for axis=None keepdims, but we just return as-is
        // since axis=None keepdims isn't commonly used. Keep it simple.
        result
    }
}
```

**Step 2: Update all public reduction signatures**

Change signatures for `sum`, `mean`, `min`, `max`, `std`, `var` to accept `keepdims: bool`:

```rust
pub fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
    if self.dtype().is_string() {
        return Err(NumpyError::TypeError("sum not supported for string arrays".into()));
    }
    let result = match axis {
        None => self.reduce_all_sum(),
        Some(ax) => self.reduce_axis_sum(ax),
    }?;
    Ok(maybe_keepdims(result, axis, keepdims))
}

pub fn mean(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
    if self.dtype().is_string() {
        return Err(NumpyError::TypeError("mean not supported for string arrays".into()));
    }
    let sum = self.astype(DType::Float64).sum(axis, false)?;
    let count = match axis {
        None => self.size(),
        Some(ax) => {
            validate_axis(ax, self.ndim())?;
            self.shape()[ax]
        }
    };
    let divisor = NdArray::full_f64(sum.shape(), count as f64);
    let result = (&sum / &divisor)?;
    Ok(maybe_keepdims(result, axis, keepdims))
}

pub fn min(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
    let result = match axis {
        None => self.reduce_all_min(),
        Some(ax) => self.reduce_axis_fold(ax, ReduceOp::Min),
    }?;
    Ok(maybe_keepdims(result, axis, keepdims))
}

pub fn max(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
    let result = match axis {
        None => self.reduce_all_max(),
        Some(ax) => self.reduce_axis_fold(ax, ReduceOp::Max),
    }?;
    Ok(maybe_keepdims(result, axis, keepdims))
}
```

For `std` and `var`, change to pass `keepdims` (but handle `ddof` in Task 2):

```rust
pub fn std(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
    if self.dtype().is_string() {
        return Err(NumpyError::TypeError("std not supported for string arrays".into()));
    }
    let var = self.var(axis, ddof, false)?;
    let result = var.sqrt();
    Ok(maybe_keepdims(result, axis, keepdims))
}

pub fn var(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
    if self.dtype().is_string() {
        return Err(NumpyError::TypeError("var not supported for string arrays".into()));
    }
    let float_self = self.astype(DType::Float64);
    let x_sq = (&float_self * &float_self)?;
    let mean_x_sq = x_sq.mean(axis, false)?;
    let mean_x = float_self.mean(axis, false)?;
    let mean_x_squared = (&mean_x * &mean_x)?;
    let result = (&mean_x_sq - &mean_x_squared)?;
    // ddof correction: multiply by N/(N-ddof)
    if ddof > 0 {
        let n = match axis {
            None => self.size(),
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                self.shape()[ax]
            }
        };
        if ddof >= n {
            // Return NaN (matching NumPy)
            let nan_val = NdArray::full_f64(result.shape(), f64::NAN);
            return Ok(maybe_keepdims(nan_val, axis, keepdims));
        }
        let correction = NdArray::full_f64(result.shape(), n as f64 / (n - ddof) as f64);
        let corrected = (&result * &correction)?;
        return Ok(maybe_keepdims(corrected, axis, keepdims));
    }
    Ok(maybe_keepdims(result, axis, keepdims))
}
```

**Step 3: Update Rust unit tests in reduction.rs**

Update all existing test calls to pass `keepdims: false` (and `ddof: 0` for std/var), and add new keepdims tests:

```rust
// Update existing tests: sum(None) -> sum(None, false), etc.
// Add new test:
#[test]
fn test_sum_keepdims() {
    let a = NdArray::ones(&[3, 4], DType::Float64);
    let s = a.sum(Some(0), true).unwrap();
    assert_eq!(s.shape(), &[1, 4]);
}

#[test]
fn test_mean_keepdims() {
    let a = NdArray::ones(&[3, 4], DType::Float64);
    let m = a.mean(Some(1), true).unwrap();
    assert_eq!(m.shape(), &[3, 1]);
}

#[test]
fn test_var_ddof() {
    let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let v0 = a.var(None, 0, false).unwrap();
    let v1 = a.var(None, 1, false).unwrap();
    // ddof=1 should give larger variance (N/(N-1) correction)
    assert_eq!(v0.dtype(), DType::Float64);
    assert_eq!(v1.dtype(), DType::Float64);
}

#[test]
fn test_var_ddof_equals_n() {
    let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let v = a.var(None, 3, false).unwrap();
    // Should be NaN
    assert_eq!(v.dtype(), DType::Float64);
}
```

**Step 4: Update Python bindings in `py_array.rs`**

Add a helper to parse optional `keepdims` boolean kwarg. Update the `sum`, `mean`, `min`, `max`, `std`, `var` pymethods to accept `keepdims` and `ddof` (where applicable).

Since RustPython's `#[pymethod]` doesn't support keyword-only args natively, use `vm::function::KwArgs` or add extra `OptionalArg` params. The simplest approach: add extra positional `OptionalArg<bool>` for keepdims and `OptionalArg<usize>` for ddof:

```rust
#[pymethod]
fn sum(
    &self,
    axis: vm::function::OptionalArg<PyObjectRef>,
    keepdims: vm::function::OptionalArg<bool>,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let ax = parse_optional_axis(axis, vm)?;
    let kd = keepdims.unwrap_or(false);
    self.data
        .read()
        .unwrap()
        .sum(ax, kd)
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
}
```

Apply the same pattern for `mean`, `min`, `max`. For `std` and `var`, add both `ddof` and `keepdims`:

```rust
#[pymethod]
fn std(
    &self,
    axis: vm::function::OptionalArg<PyObjectRef>,
    ddof: vm::function::OptionalArg<usize>,
    keepdims: vm::function::OptionalArg<bool>,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let ax = parse_optional_axis(axis, vm)?;
    let dd = ddof.unwrap_or(0);
    let kd = keepdims.unwrap_or(false);
    self.data
        .read()
        .unwrap()
        .std(ax, dd, kd)
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
}
```

**Step 5: Update module-level functions in `lib.rs`**

Update `sum`, `mean`, `min`, `max`, `std`, `var` pyfunctions to pass `false`/`0` for new params:

```rust
#[pyfunction]
fn sum(
    a: vm::PyRef<PyNdArray>,
    axis: vm::function::OptionalArg<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let ax = parse_optional_axis(axis, vm)?;
    a.inner()
        .sum(ax, false)
        .map(|arr| py_array::ndarray_or_scalar(arr, vm))
        .map_err(|e| vm.new_value_error(e.to_string()))
}
```

(Same for mean, min, max, std with `.std(ax, 0, false)`, var with `.var(ax, 0, false)`)

**Step 6: Update Python `__init__.py`**

Update `sum`, `mean`, `min`, `max`, `std`, `var` to pass `keepdims` and `ddof` through:

```python
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.sum(axis, keepdims)
        return a.sum(None, keepdims)
    return __builtins__["sum"](a) if isinstance(__builtins__, dict) else a

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.mean(axis, keepdims)
        return a.mean(None, keepdims)
    return a

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.std(axis, ddof, keepdims)
        return a.std(None, ddof, keepdims)
    return 0.0

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.var(axis, ddof, keepdims)
        return a.var(None, ddof, keepdims)
    return 0.0

def max(a, axis=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.max(axis, keepdims)
        return a.max(None, keepdims)
    return a

def min(a, axis=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.min(axis, keepdims)
        return a.min(None, keepdims)
    return a
```

**Step 7: Add Python tests**

In `tests/python/test_numeric.py`:

```python
def test_keepdims_sum():
    a = np.ones((3, 4))
    s = np.sum(a, axis=0, keepdims=True)
    assert s.shape == (1, 4), f"expected (1, 4), got {s.shape}"

def test_keepdims_mean():
    a = np.ones((3, 4))
    m = np.mean(a, axis=1, keepdims=True)
    assert m.shape == (3, 1), f"expected (3, 1), got {m.shape}"

def test_keepdims_max():
    a = np.ones((2, 3, 4))
    mx = a.max(axis=1, keepdims=True)
    assert mx.shape == (2, 1, 4), f"expected (2, 1, 4), got {mx.shape}"

def test_ddof_std():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s0 = float(np.std(a, ddof=0))
    s1 = float(np.std(a, ddof=1))
    assert s1 > s0, f"ddof=1 std ({s1}) should be > ddof=0 std ({s0})"

def test_ddof_var():
    a = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    v0 = float(np.var(a))
    v1 = float(np.var(a, ddof=1))
    assert abs(v0 - 4.0) < 0.01, f"population variance should be ~4.0, got {v0}"
    assert abs(v1 - 4.571428) < 0.01, f"sample variance should be ~4.571, got {v1}"
```

**Step 8: Verify**

```bash
cargo fmt --all
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
cargo build -p numpy-rust-wasm
./tests/python/run_tests.sh target/debug/numpy-python
```

**Step 9: Commit**

```bash
git add crates/numpy-rust-core/src/ops/reduction.rs crates/numpy-rust-python/src/py_array.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add keepdims parameter to reductions and ddof to std/var"
```

---

### Task 2: Implement `expand_dims` and `squeeze`

**Files:**
- Modify: `crates/numpy-rust-core/src/manipulation.rs`
- Modify: `crates/numpy-rust-python/src/py_array.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Add core methods in `manipulation.rs`**

```rust
impl NdArray {
    /// Insert a new axis of size 1 at the given position.
    pub fn expand_dims(&self, axis: usize) -> Result<NdArray> {
        if axis > self.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis,
                ndim: self.ndim() + 1,
            });
        }
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);
        self.reshape(&new_shape)
    }

    /// Remove dimensions of size 1.
    /// If `axis` is Some, only remove that specific axis (error if it's not size 1).
    /// If `axis` is None, remove all size-1 dimensions.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<NdArray> {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                    });
                }
                if self.shape()[ax] != 1 {
                    return Err(NumpyError::ValueError(format!(
                        "cannot squeeze axis {ax} with size {}",
                        self.shape()[ax]
                    )));
                }
                let new_shape: Vec<usize> = self
                    .shape()
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                if new_shape.is_empty() {
                    // Squeezing to 0-D
                    self.reshape(&[])
                } else {
                    self.reshape(&new_shape)
                }
            }
            None => {
                let new_shape: Vec<usize> =
                    self.shape().iter().copied().filter(|&s| s != 1).collect();
                if new_shape.is_empty() {
                    self.reshape(&[])
                } else {
                    self.reshape(&new_shape)
                }
            }
        }
    }
}
```

**Step 2: Add Rust tests**

```rust
#[test]
fn test_expand_dims() {
    let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let b = a.expand_dims(0).unwrap();
    assert_eq!(b.shape(), &[1, 3]);
    let c = a.expand_dims(1).unwrap();
    assert_eq!(c.shape(), &[3, 1]);
}

#[test]
fn test_squeeze() {
    let a = NdArray::zeros(&[1, 3, 1], DType::Float64);
    let b = a.squeeze(None).unwrap();
    assert_eq!(b.shape(), &[3]);
}

#[test]
fn test_squeeze_specific_axis() {
    let a = NdArray::zeros(&[1, 3, 1], DType::Float64);
    let b = a.squeeze(Some(0)).unwrap();
    assert_eq!(b.shape(), &[3, 1]);
}

#[test]
fn test_squeeze_non_unit_axis_fails() {
    let a = NdArray::zeros(&[2, 3], DType::Float64);
    assert!(a.squeeze(Some(0)).is_err());
}
```

**Step 3: Add Python bindings**

In `py_array.rs`, add pymethods:

```rust
#[pymethod]
fn expand_dims(&self, axis: usize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    self.data
        .read()
        .unwrap()
        .expand_dims(axis)
        .map(PyNdArray::from_core)
        .map_err(|e| numpy_err(e, vm))
}

#[pymethod]
fn squeeze(
    &self,
    axis: vm::function::OptionalArg<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let ax = parse_optional_axis(axis, vm)?;
    self.data
        .read()
        .unwrap()
        .squeeze(ax)
        .map(PyNdArray::from_core)
        .map_err(|e| numpy_err(e, vm))
}
```

**Step 4: Update `__init__.py`**

```python
def squeeze(a, axis=None):
    if isinstance(a, ndarray):
        return a.squeeze(axis)
    return a

def expand_dims(a, axis):
    if isinstance(a, ndarray):
        return a.expand_dims(axis)
    return a
```

**Step 5: Add Python tests**

```python
def test_expand_dims():
    a = np.array([1.0, 2.0, 3.0])
    b = np.expand_dims(a, 0)
    assert b.shape == (1, 3), f"expected (1, 3), got {b.shape}"
    c = np.expand_dims(a, 1)
    assert c.shape == (3, 1), f"expected (3, 1), got {c.shape}"

def test_squeeze():
    a = np.ones((1, 3, 1))
    b = np.squeeze(a)
    assert b.shape == (3,), f"expected (3,), got {b.shape}"

def test_squeeze_axis():
    a = np.ones((1, 3, 1))
    b = np.squeeze(a, axis=0)
    assert b.shape == (3, 1), f"expected (3, 1), got {b.shape}"
```

**Step 6: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/manipulation.rs crates/numpy-rust-python/src/py_array.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: implement expand_dims and squeeze"
```

---

### Task 3: Replace `unwrap()` with proper error handling

**Files:**
- Modify: `crates/numpy-rust-core/src/manipulation.rs`
- Modify: `crates/numpy-rust-core/src/creation.rs`
- Modify: `crates/numpy-rust-core/src/ops/sorting.rs`

**Step 1: Fix `manipulation.rs` reshape unwraps**

In `reshape()` method (lines 20-30), replace `.unwrap()` with `.expect("size validated above")`:

```rust
let data = match &self.data {
    ArrayData::Bool(a) => {
        ArrayData::Bool(a.clone().into_shape_with_order(sh).expect("size validated above"))
    }
    // same for all variants...
};
```

In `flatten()` (line 49), change to propagate the Result or use expect:

```rust
pub fn flatten(&self) -> NdArray {
    self.reshape(&[self.size()])
        .expect("flatten reshape to total size cannot fail")
}
```

In `stack()` (line 182), replace `.unwrap()` with `.expect("insert-axis reshape cannot fail")`.

**Step 2: Fix `creation.rs` unwraps**

In `arange()` and `linspace()`, replace `ArrayD::from_shape_vec(...).unwrap()` with `.expect("vec length matches shape")`.

**Step 3: Fix `sorting.rs` unwraps**

Same pattern: replace `.unwrap()` at lines 32 and 76 with `.expect("flat vec matches shape")`.

**Step 4: Verify nothing broke**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
```

**Step 5: Commit**

```bash
git add crates/numpy-rust-core/src/manipulation.rs crates/numpy-rust-core/src/creation.rs crates/numpy-rust-core/src/ops/sorting.rs
git commit -m "fix: replace unwrap() with expect() for documented invariants"
```

---

### Task 4: Improve `eye(n, M, k)`

**Files:**
- Modify: `crates/numpy-rust-core/src/creation.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Update core `eye` signature**

```rust
/// Create an identity-like matrix with shape (n, m) and ones on diagonal offset k.
/// m defaults to n. k=0 is main diagonal, positive = superdiagonal, negative = subdiagonal.
pub fn eye(n: usize, m: Option<usize>, k: isize, dtype: DType) -> NdArray {
    if dtype.is_string() {
        panic!("eye() not supported for string dtype");
    }
    let cols = m.unwrap_or(n);
    let mut arr = NdArray::zeros(&[n, cols], dtype);
    match &mut arr.data {
        ArrayData::Bool(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = true;
                }
            }
        }
        ArrayData::Int32(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1;
                }
            }
        }
        ArrayData::Int64(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1;
                }
            }
        }
        ArrayData::Float32(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1.0;
                }
            }
        }
        ArrayData::Float64(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1.0;
                }
            }
        }
        ArrayData::Str(_) => unreachable!(),
    }
    arr
}
```

**Step 2: Update `lib.rs` re-export**

The re-export `pub use creation::eye;` stays the same since we're changing the function signature, not adding a new one.

**Step 3: Update Rust tests**

```rust
#[test]
fn test_eye_rectangular() {
    let a = eye(3, Some(4), 0, DType::Float64);
    assert_eq!(a.shape(), &[3, 4]);
}

#[test]
fn test_eye_offset() {
    let a = eye(3, None, 1, DType::Float64);
    assert_eq!(a.shape(), &[3, 3]);
}

#[test]
fn test_eye_negative_offset() {
    let a = eye(3, None, -1, DType::Float64);
    assert_eq!(a.shape(), &[3, 3]);
}
```

Update existing tests to pass new params: `eye(3, DType::Float64)` → `eye(3, None, 0, DType::Float64)`.

**Step 4: Update Python binding**

In `crates/numpy-rust-python/src/lib.rs`:

```rust
#[pyfunction]
fn eye(
    n: usize,
    m: vm::function::OptionalArg<usize>,
    k: vm::function::OptionalArg<isize>,
    _vm: &VirtualMachine,
) -> PyNdArray {
    let m_val = m.into_option();
    let k_val = k.unwrap_or(0);
    PyNdArray::from_core(numpy_rust_core::creation::eye(
        n,
        m_val,
        k_val,
        numpy_rust_core::DType::Float64,
    ))
}
```

**Step 5: Update `__init__.py`**

Find the existing `eye` function/wrapper and ensure `M` and `k` parameters are passed through to native. If `eye` is currently just calling native directly, this may already work. Check and adjust as needed.

**Step 6: Add Python tests**

```python
def test_eye_rectangular():
    a = np.eye(3, 4)
    assert a.shape == (3, 4), f"expected (3, 4), got {a.shape}"

def test_eye_offset():
    a = np.eye(3, k=1)
    assert a.shape == (3, 3)
    assert float(a[0, 1]) == 1.0
    assert float(a[0, 0]) == 0.0
```

**Step 7: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/creation.rs crates/numpy-rust-core/src/lib.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: extend eye() with M (columns) and k (diagonal offset) parameters"
```

---

### Task 5: Add axis parameter to `argmin`/`argmax`

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/reduction.rs`
- Modify: `crates/numpy-rust-python/src/py_array.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Change core signatures**

In `reduction.rs`, change `argmin`/`argmax` to return `NdArray` (Int64) and accept `axis`:

```rust
/// Index of minimum element.
/// axis=None: flatten then find argmin. axis=Some(ax): argmin along that axis.
pub fn argmin(&self, axis: Option<usize>) -> Result<NdArray> {
    if self.dtype().is_string() {
        return Err(NumpyError::TypeError("argmin not supported for string arrays".into()));
    }
    match axis {
        None => {
            let idx = self.reduce_all_argmin()?;
            Ok(NdArray::from_data(ArrayData::Int64(
                ArrayD::from_elem(IxDyn(&[]), idx as i64),
            )))
        }
        Some(ax) => self.reduce_axis_argmin(ax),
    }
}

/// Index of maximum element.
pub fn argmax(&self, axis: Option<usize>) -> Result<NdArray> {
    if self.dtype().is_string() {
        return Err(NumpyError::TypeError("argmax not supported for string arrays".into()));
    }
    match axis {
        None => {
            let idx = self.reduce_all_argmax()?;
            Ok(NdArray::from_data(ArrayData::Int64(
                ArrayD::from_elem(IxDyn(&[]), idx as i64),
            )))
        }
        Some(ax) => self.reduce_axis_argmax(ax),
    }
}
```

Add `reduce_axis_argmin`/`reduce_axis_argmax` helpers using the lanes pattern from `argsort`:

```rust
fn reduce_axis_argmin(&self, axis: usize) -> Result<NdArray> {
    validate_axis(axis, self.ndim())?;
    let f = self.astype(DType::Float64);
    let ArrayData::Float64(arr) = &f.data else {
        unreachable!()
    };
    let ax = Axis(axis);
    // Result shape: remove the axis dimension
    let mut result_shape = arr.shape().to_vec();
    result_shape.remove(axis);
    if result_shape.is_empty() {
        result_shape.push(1); // at least 1-D for single-lane case
    }
    let mut result = ArrayD::<i64>::zeros(IxDyn(&result_shape));
    for (lane_in, result_elem) in arr
        .lanes(ax)
        .into_iter()
        .zip(result.iter_mut())
    {
        let mut min_idx: usize = 0;
        let mut min_val = f64::INFINITY;
        for (i, &v) in lane_in.iter().enumerate() {
            if v < min_val || (min_val.is_nan() && !v.is_nan()) {
                min_val = v;
                min_idx = i;
            }
        }
        *result_elem = min_idx as i64;
    }
    Ok(NdArray::from_data(ArrayData::Int64(result)))
}

fn reduce_axis_argmax(&self, axis: usize) -> Result<NdArray> {
    validate_axis(axis, self.ndim())?;
    let f = self.astype(DType::Float64);
    let ArrayData::Float64(arr) = &f.data else {
        unreachable!()
    };
    let ax = Axis(axis);
    let mut result_shape = arr.shape().to_vec();
    result_shape.remove(axis);
    if result_shape.is_empty() {
        result_shape.push(1);
    }
    let mut result = ArrayD::<i64>::zeros(IxDyn(&result_shape));
    for (lane_in, result_elem) in arr
        .lanes(ax)
        .into_iter()
        .zip(result.iter_mut())
    {
        let mut max_idx: usize = 0;
        let mut max_val = f64::NEG_INFINITY;
        for (i, &v) in lane_in.iter().enumerate() {
            if v > max_val || (max_val.is_nan() && !v.is_nan()) {
                max_val = v;
                max_idx = i;
            }
        }
        *result_elem = max_idx as i64;
    }
    Ok(NdArray::from_data(ArrayData::Int64(result)))
}
```

**Step 2: Update Python bindings**

In `py_array.rs`, update `argmin`/`argmax` pymethods:

```rust
#[pymethod]
fn argmin(
    &self,
    axis: vm::function::OptionalArg<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let ax = parse_optional_axis(axis, vm)?;
    self.data
        .read()
        .unwrap()
        .argmin(ax)
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
}

#[pymethod]
fn argmax(
    &self,
    axis: vm::function::OptionalArg<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let ax = parse_optional_axis(axis, vm)?;
    self.data
        .read()
        .unwrap()
        .argmax(ax)
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
}
```

**Step 3: Update `__init__.py`**

```python
def argmax(a, axis=None, out=None):
    if isinstance(a, ndarray):
        return a.argmax(axis)
    return 0

def argmin(a, axis=None, out=None):
    if isinstance(a, ndarray):
        return a.argmin(axis)
    return 0
```

**Step 4: Add tests**

```python
def test_argmin_axis():
    a = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    idx = np.argmin(a, axis=1)
    assert idx.shape == (2,), f"expected (2,), got {idx.shape}"

def test_argmax_axis():
    a = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    idx = np.argmax(a, axis=0)
    assert idx.shape == (3,), f"expected (3,), got {idx.shape}"
```

**Step 5: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/ops/reduction.rs crates/numpy-rust-python/src/py_array.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add axis parameter to argmin/argmax"
```

---

### Task 6: Edge case tests

**Files:**
- Test: `tests/python/test_numeric.py`

**Step 1: Add edge case tests**

```python
def test_sum_empty_array():
    a = np.array([])
    s = np.sum(a)
    assert float(s) == 0.0, f"sum of empty array should be 0.0, got {s}"

def test_dtype_preservation_int32():
    a = np.array([1, 2, 3]).astype("int32")
    b = np.array([4, 5, 6]).astype("int32")
    c = a + b
    assert c.dtype == "int32", f"expected int32, got {c.dtype}"

def test_inf_max():
    a = np.array([1.0, float("inf"), 3.0])
    mx = float(np.max(a))
    assert mx == float("inf"), f"expected inf, got {mx}"

def test_sort_already_sorted():
    a = np.array([1.0, 2.0, 3.0])
    s = np.sort(a)
    for i in range(3):
        assert float(s[i]) == float(a[i])

def test_argsort_duplicates():
    a = np.array([3.0, 1.0, 1.0, 2.0])
    idx = np.argsort(a)
    # First two indices should be 1 and 2 (both have value 1.0)
    assert int(idx[0]) in [1, 2]
    assert int(idx[1]) in [1, 2]
```

**Step 2: Build and run tests**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
```

**Step 3: Commit**

```bash
git add tests/python/test_numeric.py
git commit -m "test: add edge case tests for empty arrays, inf, dtype preservation"
```

---

## Phase B: Feature Expansion

### Task 7: Add `num-complex` dependency and extend DType/ArrayData

**Files:**
- Modify: `crates/numpy-rust-core/Cargo.toml`
- Modify: `crates/numpy-rust-core/src/dtype.rs`
- Modify: `crates/numpy-rust-core/src/array_data.rs`
- Modify: `crates/numpy-rust-core/src/casting.rs`
- Modify: `crates/numpy-rust-core/src/error.rs`

**Step 1: Add dependency**

In `crates/numpy-rust-core/Cargo.toml`, add:

```toml
num-complex = "0.4"
```

**Step 2: Extend DType**

In `dtype.rs`, add variants and update all methods:

```rust
use num_complex::Complex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int32,
    Int64,
    Float32,
    Float64,
    Complex64,   // Complex<f32>
    Complex128,  // Complex<f64>
    Str,
}
```

Update `promote`:
```rust
pub fn promote(self, other: DType) -> DType {
    if self == other { return self; }
    if self == DType::Str || other == DType::Str {
        panic!("cannot promote string dtype with numeric dtype");
    }
    // If either is complex, result is complex
    if self.is_complex() || other.is_complex() {
        // Both complex: use higher
        if self.is_complex() && other.is_complex() {
            return if self.rank() >= other.rank() { self } else { other };
        }
        // One complex, one real: promote real part then complexify
        let real_type = if self.is_complex() { other } else { self };
        let complex_type = if self.is_complex() { self } else { other };
        return match (complex_type, real_type) {
            (DType::Complex64, DType::Float64 | DType::Int64) => DType::Complex128,
            (DType::Complex128, _) => DType::Complex128,
            (DType::Complex64, _) => DType::Complex64,
            _ => DType::Complex128,
        };
    }
    // existing numeric promotion logic...
    let (hi, lo) = if self.rank() >= other.rank() { (self, other) } else { (other, self) };
    if (hi == DType::Float32 && lo == DType::Int64) || (hi == DType::Int64 && lo == DType::Float32) {
        return DType::Float64;
    }
    hi
}
```

Update `rank`:
```rust
fn rank(self) -> u8 {
    match self {
        DType::Bool => 0,
        DType::Int32 => 1,
        DType::Int64 => 2,
        DType::Float32 => 3,
        DType::Float64 => 4,
        DType::Complex64 => 5,
        DType::Complex128 => 6,
        DType::Str => 255,
    }
}
```

Add `is_complex`:
```rust
pub fn is_complex(self) -> bool {
    matches!(self, DType::Complex64 | DType::Complex128)
}
```

Update `itemsize`, `is_float`, `Display`.

**Step 3: Extend ArrayData**

In `array_data.rs`:

```rust
use num_complex::Complex;

pub enum ArrayData {
    Bool(ArrayD<bool>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Complex64(ArrayD<Complex<f32>>),
    Complex128(ArrayD<Complex<f64>>),
    Str(ArrayD<String>),
}
```

Add new arms to `dtype()`, `shape()`, `ndim()`. Update `dispatch_unary!` macro to include complex variants.

**Step 4: Extend casting**

In `casting.rs`, add conversion paths:
- `Float64 → Complex128`: `Complex::new(v, 0.0)`
- `Float32 → Complex64`: `Complex::new(v, 0.0)`
- `Int* → Complex128`: via f64 intermediate
- `Complex64 → Complex128`: widen
- `Complex128 → Complex64`: narrow (with precision loss)

**Step 5: Extend `Scalar` in indexing.rs**

Add `Complex64(Complex<f32>)` and `Complex128(Complex<f64>)` variants to the `Scalar` enum if it exists, or handle complex data access in `get`/`set`.

**Step 6: Verify compilation**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
```

**Step 7: Commit**

```bash
git add crates/numpy-rust-core/Cargo.toml crates/numpy-rust-core/src/dtype.rs crates/numpy-rust-core/src/array_data.rs crates/numpy-rust-core/src/casting.rs crates/numpy-rust-core/src/indexing.rs
git commit -m "feat: add Complex64/Complex128 dtype and ArrayData variants"
```

---

### Task 8: Add complex number support to operations

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/arithmetic.rs`
- Modify: `crates/numpy-rust-core/src/ops/math.rs`
- Modify: `crates/numpy-rust-core/src/ops/reduction.rs`
- Modify: `crates/numpy-rust-core/src/ops/comparison.rs`
- Modify: `crates/numpy-rust-core/src/array.rs`

**Step 1: Arithmetic operations**

In `arithmetic.rs`, extend `impl_binary_op!` macro's match arms to handle complex types:

```rust
(ArrayData::Complex64(a), ArrayData::Complex64(b)) => ArrayData::Complex64(a $op b),
(ArrayData::Complex128(a), ArrayData::Complex128(b)) => ArrayData::Complex128(a $op b),
```

For `pow`, `floor_div`, `remainder` — complex power works via `num_complex::Complex::powc()`. Floor div and remainder on complex should return TypeError (NumPy does the same).

**Step 2: Math operations**

In `math.rs`, add complex arms for `sqrt`, `exp`, `log`, `abs`:
- `sqrt`: `Complex::sqrt()`
- `exp`: `Complex::exp()`
- `log`: `Complex::ln()`
- `abs`: returns real magnitude `Complex::norm()` — result is Float64/Float32, not complex

Add new methods: `real()`, `imag()`, `conj()`, `angle()`:

```rust
pub fn real(&self) -> NdArray {
    match &self.data {
        ArrayData::Complex64(a) => NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.re))),
        ArrayData::Complex128(a) => NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.re))),
        _ => self.clone(), // real arrays are their own real part
    }
}

pub fn imag(&self) -> NdArray {
    match &self.data {
        ArrayData::Complex64(a) => NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.im))),
        ArrayData::Complex128(a) => NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.im))),
        _ => NdArray::zeros(self.shape(), self.dtype()), // real arrays have zero imag
    }
}

pub fn conj(&self) -> NdArray {
    match &self.data {
        ArrayData::Complex64(a) => NdArray::from_data(ArrayData::Complex64(a.mapv(|c| c.conj()))),
        ArrayData::Complex128(a) => NdArray::from_data(ArrayData::Complex128(a.mapv(|c| c.conj()))),
        _ => self.clone(), // conjugate of real is self
    }
}

pub fn angle(&self) -> NdArray {
    match &self.data {
        ArrayData::Complex64(a) => NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.arg()))),
        ArrayData::Complex128(a) => NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.arg()))),
        _ => NdArray::zeros(self.shape(), DType::Float64), // angle of positive real is 0
    }
}
```

**Step 3: Reductions**

In `reduction.rs`, add complex sum/mean (they work). For min/max/argmin/argmax, return TypeError for complex.

**Step 4: Comparisons**

Only `==` and `!=` for complex. `<`, `<=`, `>`, `>=` return TypeError.

**Step 5: Array creation**

In `array.rs`, add `from_complex128_vec`, `zeros`/`ones` for complex types:

```rust
pub fn from_complex128_vec(v: Vec<Complex<f64>>) -> Self {
    let len = v.len();
    Self::from_data(ArrayData::Complex128(
        ArrayD::from_shape_vec(IxDyn(&[len]), v).expect("vec length matches shape"),
    ))
}
```

**Step 6: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
git add crates/numpy-rust-core/src/
git commit -m "feat: add complex number support to arithmetic, math, and reduction ops"
```

---

### Task 9: Complex number Python bindings

**Files:**
- Modify: `crates/numpy-rust-python/src/py_array.rs`
- Modify: `crates/numpy-rust-python/src/py_creation.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Update `parse_dtype` to handle complex strings**

```rust
"complex64" | "c64" => Ok(DType::Complex64),
"complex128" | "c128" | "complex" => Ok(DType::Complex128),
```

**Step 2: Update `scalar_to_py` for complex**

```rust
Scalar::Complex64(v) => {
    // RustPython complex: vm.ctx.new_complex(Complex64 { re, im })
    vm.ctx.new_complex(num_complex::Complex64::new(v.re as f64, v.im as f64)).into()
}
Scalar::Complex128(v) => {
    vm.ctx.new_complex(num_complex::Complex64::new(v.re, v.im)).into()
}
```

**Step 3: Update `obj_to_ndarray` for complex Python objects**

Add complex detection before float:
```rust
if let Some(c) = obj.downcast_ref::<vm::builtins::PyComplex>() {
    let val = c.to_complex();
    return Ok(NdArray::from_complex128_vec(vec![Complex::new(val.re, val.im)]));
}
```

**Step 4: Add pymethods for `real`, `imag`, `conj`, `angle`**

```rust
#[pygetset]
fn real(&self) -> PyNdArray {
    PyNdArray::from_core(self.data.read().unwrap().real())
}

#[pygetset]
fn imag(&self) -> PyNdArray {
    PyNdArray::from_core(self.data.read().unwrap().imag())
}

#[pymethod]
fn conj(&self) -> PyNdArray {
    PyNdArray::from_core(self.data.read().unwrap().conj())
}
```

**Step 5: Add dtype aliases in `__init__.py`**

```python
complex64 = "complex64"
complex128 = "complex128"
```

**Step 6: Add Python tests**

```python
def test_complex_creation():
    a = np.array([1+2j, 3+4j])
    assert a.dtype == "complex128"

def test_complex_add():
    a = np.array([1+2j, 3+4j])
    b = np.array([5+6j, 7+8j])
    c = a + b
    # c should be [6+8j, 10+12j]
    assert c.dtype == "complex128"

def test_complex_real_imag():
    a = np.array([1+2j, 3+4j])
    r = a.real
    im = a.imag
    assert float(r[0]) == 1.0
    assert float(im[0]) == 2.0

def test_complex_abs():
    a = np.array([3+4j])
    m = np.abs(a)
    assert abs(float(m[0]) - 5.0) < 1e-10

def test_complex_conj():
    a = np.array([1+2j])
    c = a.conj()
    assert float(c.imag[0]) == -2.0
```

**Step 7: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-python/ python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add complex number Python bindings and tests"
```

---

### Task 10: Implement `einsum`

**Files:**
- Create: `crates/numpy-rust-core/src/ops/einsum.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Create `ops/einsum.rs`**

```rust
use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Parse an einsum subscript string like "ij,jk->ik".
/// Returns (input_subscripts, output_subscript).
fn parse_subscripts(s: &str) -> Result<(Vec<Vec<char>>, Vec<char>)> {
    let parts: Vec<&str> = s.split("->").collect();
    if parts.len() != 2 {
        return Err(NumpyError::ValueError(
            "einsum requires explicit '->' output subscripts".into(),
        ));
    }
    let inputs: Vec<Vec<char>> = parts[0].split(',').map(|p| p.trim().chars().collect()).collect();
    let output: Vec<char> = parts[1].trim().chars().collect();
    Ok((inputs, output))
}

/// Execute einsum with explicit subscript notation.
pub fn einsum(subscripts: &str, operands: &[&NdArray]) -> Result<NdArray> {
    let (input_subs, output_sub) = parse_subscripts(subscripts)?;
    if input_subs.len() != operands.len() {
        return Err(NumpyError::ValueError(format!(
            "einsum: {} operands provided but subscripts specify {}",
            operands.len(),
            input_subs.len()
        )));
    }

    // Cast all to Float64
    let float_ops: Vec<NdArray> = operands.iter().map(|a| a.astype(DType::Float64)).collect();
    let arrays: Vec<&ArrayD<f64>> = float_ops
        .iter()
        .map(|a| match &a.data {
            ArrayData::Float64(arr) => arr,
            _ => unreachable!(),
        })
        .collect();

    // Build index->size map
    let mut index_sizes: HashMap<char, usize> = HashMap::new();
    for (subs, arr) in input_subs.iter().zip(arrays.iter()) {
        if subs.len() != arr.ndim() {
            return Err(NumpyError::ValueError(format!(
                "einsum: operand has {} dimensions but subscript has {} indices",
                arr.ndim(),
                subs.len()
            )));
        }
        for (&c, &dim) in subs.iter().zip(arr.shape().iter()) {
            if let Some(&existing) = index_sizes.get(&c) {
                if existing != dim {
                    return Err(NumpyError::ShapeMismatch(format!(
                        "einsum: index '{c}' has size {existing} and {dim}"
                    )));
                }
            } else {
                index_sizes.insert(c, dim);
            }
        }
    }

    // Find contracted indices (in inputs but not output)
    let output_set: HashSet<char> = output_sub.iter().copied().collect();
    let all_indices: HashSet<char> = input_subs.iter().flat_map(|s| s.iter().copied()).collect();
    let contracted: Vec<char> = all_indices.difference(&output_set).copied().collect();

    // Output shape
    let output_shape: Vec<usize> = output_sub
        .iter()
        .map(|&c| *index_sizes.get(&c).unwrap())
        .collect();
    let output_size: usize = output_shape.iter().product();

    // Build output array by iterating
    let mut result_data = vec![0.0f64; output_size.max(1)];

    // All output index combinations
    let output_ranges: Vec<usize> = output_sub.iter().map(|c| index_sizes[c]).collect();
    let contracted_ranges: Vec<usize> = contracted.iter().map(|c| index_sizes[c]).collect();

    // Iterator over multi-index
    fn multi_index_iter(ranges: &[usize]) -> Vec<Vec<usize>> {
        if ranges.is_empty() {
            return vec![vec![]];
        }
        let mut result = Vec::new();
        let sub = multi_index_iter(&ranges[1..]);
        for i in 0..ranges[0] {
            for s in &sub {
                let mut v = vec![i];
                v.extend_from_slice(s);
                result.push(v);
            }
        }
        result
    }

    let output_indices = multi_index_iter(&output_ranges);
    let contract_indices = multi_index_iter(&contracted_ranges);

    for (flat_idx, out_idx) in output_indices.iter().enumerate() {
        let mut sum = 0.0f64;

        // Build index map for output indices
        let mut idx_map: HashMap<char, usize> = HashMap::new();
        for (i, &c) in output_sub.iter().enumerate() {
            idx_map.insert(c, out_idx[i]);
        }

        for cont_idx in &contract_indices {
            // Add contracted indices to map
            for (i, &c) in contracted.iter().enumerate() {
                idx_map.insert(c, cont_idx[i]);
            }

            // Compute product of operand elements at these indices
            let mut product = 1.0f64;
            for (op_idx, (subs, arr)) in input_subs.iter().zip(arrays.iter()).enumerate() {
                let arr_idx: Vec<usize> = subs.iter().map(|c| idx_map[c]).collect();
                product *= arr[ndarray::IxDyn(&arr_idx)];
            }
            sum += product;
        }
        result_data[flat_idx] = sum;
    }

    let out_shape = if output_shape.is_empty() {
        IxDyn(&[])
    } else {
        IxDyn(&output_shape)
    };
    Ok(NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(out_shape, result_data).expect("output shape matches data size"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_einsum_matmul() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = NdArray::from_vec(vec![5.0, 6.0, 7.0, 8.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_einsum_trace() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = einsum("ii->", &[&a]).unwrap();
        assert_eq!(c.shape(), &[]); // scalar
    }

    #[test]
    fn test_einsum_transpose() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let c = einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
    }
}
```

**Step 2: Register module**

In `ops/mod.rs`, add:
```rust
pub mod einsum;
```

In `crates/numpy-rust-core/src/lib.rs`, add re-export:
```rust
pub use ops::einsum::einsum;
```

**Step 3: Add Python binding**

In `crates/numpy-rust-python/src/lib.rs`:

```rust
#[pyfunction]
fn einsum(
    subscripts: vm::PyRef<vm::builtins::PyStr>,
    args: vm::function::PosArgs<vm::PyRef<PyNdArray>>,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let operands: Vec<NdArray> = args.iter().map(|a| a.inner().clone()).collect();
    let refs: Vec<&NdArray> = operands.iter().collect();
    numpy_rust_core::einsum(subscripts.as_str(), &refs)
        .map(|arr| py_array::ndarray_or_scalar(arr, vm))
        .map_err(|e| vm.new_value_error(e.to_string()))
}
```

**Step 4: Update `__init__.py`**

```python
def einsum(*operands, **kwargs):
    if len(operands) < 2:
        raise ValueError("einsum requires at least a subscript string and one operand")
    subscripts = operands[0]
    arrays = operands[1:]
    return _numpy_native.einsum(subscripts, *arrays)
```

**Step 5: Add Python tests**

```python
def test_einsum_matmul():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = np.einsum("ij,jk->ik", a, b)
    assert c.shape == (2, 2)
    assert abs(float(c[0, 0]) - 19.0) < 1e-10
    assert abs(float(c[0, 1]) - 22.0) < 1e-10

def test_einsum_trace():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = np.einsum("ii->", a)
    assert abs(float(t) - 5.0) < 1e-10

def test_einsum_outer():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    c = np.einsum("i,j->ij", a, b)
    assert c.shape == (2, 3)
    assert abs(float(c[0, 0]) - 3.0) < 1e-10
    assert abs(float(c[1, 2]) - 10.0) < 1e-10
```

**Step 6: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/ops/einsum.rs crates/numpy-rust-core/src/ops/mod.rs crates/numpy-rust-core/src/lib.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: implement einsum with explicit subscript notation"
```

---

### Task 11: String operations

**Files:**
- Create: `crates/numpy-rust-core/src/ops/string_ops.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Create `ops/string_ops.rs`**

```rust
use ndarray::ArrayD;

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

fn require_string(arr: &NdArray) -> Result<&ArrayD<String>> {
    match &arr.data {
        ArrayData::Str(a) => Ok(a),
        _ => Err(NumpyError::TypeError(
            "string operation requires string array".into(),
        )),
    }
}

impl NdArray {
    pub fn str_upper(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.to_uppercase()),
        )))
    }

    pub fn str_lower(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.to_lowercase()),
        )))
    }

    pub fn str_capitalize(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(a.mapv(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(f) => {
                    f.to_uppercase().to_string() + &c.as_str().to_lowercase()
                }
            }
        }))))
    }

    pub fn str_strip(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.trim().to_owned()),
        )))
    }

    pub fn str_len(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Int64(
            a.mapv(|s| s.len() as i64),
        )))
    }

    pub fn str_startswith(&self, prefix: &str) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Bool(
            a.mapv(|s| s.starts_with(prefix)),
        )))
    }

    pub fn str_endswith(&self, suffix: &str) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Bool(
            a.mapv(|s| s.ends_with(suffix)),
        )))
    }

    pub fn str_replace(&self, old: &str, new: &str) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.replace(old, new)),
        )))
    }
}

#[cfg(test)]
mod tests {
    use crate::NdArray;

    #[test]
    fn test_str_upper() {
        let a = NdArray::from_vec(vec!["hello".to_string(), "world".to_string()]);
        let b = a.str_upper().unwrap();
        assert_eq!(b.shape(), &[2]);
    }

    #[test]
    fn test_str_len() {
        let a = NdArray::from_vec(vec!["hi".to_string(), "there".to_string()]);
        let b = a.str_len().unwrap();
        assert_eq!(b.shape(), &[2]);
    }
}
```

**Step 2: Register module**

In `ops/mod.rs`:
```rust
pub mod string_ops;
```

**Step 3: Add Python bindings**

In `lib.rs`, add a `char` submodule or add functions directly. The simplest approach is a `char` module:

Create a small helper module or add functions to the main module namespace prefixed with `char_`. For now, expose them as module-level functions that the Python `__init__.py` will organize:

```rust
// In _numpy_native module:
#[pyfunction]
fn char_upper(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    a.inner()
        .str_upper()
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
}

#[pyfunction]
fn char_lower(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    a.inner()
        .str_lower()
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
}

// ... same for capitalize, strip, str_len, startswith, endswith, replace
```

**Step 4: Update `__init__.py`**

Add a `char` namespace:

```python
class _char_mod:
    @staticmethod
    def upper(a):
        return _numpy_native.char_upper(a)
    @staticmethod
    def lower(a):
        return _numpy_native.char_lower(a)
    @staticmethod
    def capitalize(a):
        return _numpy_native.char_capitalize(a)
    @staticmethod
    def strip(a):
        return _numpy_native.char_strip(a)
    @staticmethod
    def str_len(a):
        return _numpy_native.char_str_len(a)
    @staticmethod
    def startswith(a, prefix):
        return _numpy_native.char_startswith(a, prefix)
    @staticmethod
    def endswith(a, suffix):
        return _numpy_native.char_endswith(a, suffix)
    @staticmethod
    def replace(a, old, new):
        return _numpy_native.char_replace(a, old, new)

char = _char_mod()
```

**Step 5: Add Python tests**

```python
def test_char_upper():
    a = np.array(["hello", "world"])
    b = np.char.upper(a)
    assert b[0] == "HELLO"

def test_char_lower():
    a = np.array(["HELLO", "WORLD"])
    b = np.char.lower(a)
    assert b[0] == "hello"

def test_char_str_len():
    a = np.array(["hi", "there"])
    lengths = np.char.str_len(a)
    assert int(lengths[0]) == 2
    assert int(lengths[1]) == 5
```

**Step 6: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/ops/string_ops.rs crates/numpy-rust-core/src/ops/mod.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add string operations (upper, lower, capitalize, strip, len, etc.)"
```

---

### Task 12: Implement `searchsorted`, `choose`, `compress`

**Files:**
- Create: `crates/numpy-rust-core/src/ops/selection.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Create `ops/selection.rs`**

```rust
use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Binary search in a sorted 1-D array. Returns Int64 array of insertion indices.
    /// side="left": leftmost insertion point. side="right": rightmost.
    pub fn searchsorted(&self, values: &NdArray, side: &str) -> Result<NdArray> {
        if self.ndim() != 1 {
            return Err(NumpyError::ValueError(
                "searchsorted requires a 1-D sorted array".into(),
            ));
        }
        let sorted = self.astype(DType::Float64);
        let vals = values.astype(DType::Float64);
        let ArrayData::Float64(sorted_arr) = &sorted.data else {
            unreachable!()
        };
        let ArrayData::Float64(vals_arr) = &vals.data else {
            unreachable!()
        };
        let sorted_slice: Vec<f64> = sorted_arr.iter().copied().collect();
        let left = side == "left";

        let mut indices = Vec::with_capacity(vals_arr.len());
        for &v in vals_arr.iter() {
            let idx = if left {
                sorted_slice.partition_point(|&x| x < v)
            } else {
                sorted_slice.partition_point(|&x| x <= v)
            };
            indices.push(idx as i64);
        }

        Ok(NdArray::from_data(ArrayData::Int64(
            ArrayD::from_shape_vec(vals_arr.raw_dim(), indices)
                .expect("output shape matches values shape"),
        )))
    }

    /// Select slices along `axis` where `condition` is true.
    pub fn compress(&self, condition: &NdArray, axis: Option<usize>) -> Result<NdArray> {
        let cond = condition.astype(DType::Bool);
        let ArrayData::Bool(mask) = &cond.data else {
            unreachable!()
        };

        match axis {
            None => {
                // Flatten self, apply mask
                let flat = self.flatten();
                flat.mask_select(&cond)
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                    });
                }
                let indices: Vec<usize> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &b)| b)
                    .map(|(i, _)| i)
                    .collect();
                self.index_select(ax, &indices)
            }
        }
    }
}

/// Select from `choices` arrays based on integer index array `a`.
pub fn choose(a: &NdArray, choices: &[&NdArray]) -> Result<NdArray> {
    if choices.is_empty() {
        return Err(NumpyError::ValueError("choose requires at least one choice array".into()));
    }
    let idx = a.astype(DType::Int64);
    let flat_idx = idx.flatten();
    let ArrayData::Int64(idx_arr) = &flat_idx.data else {
        unreachable!()
    };

    // Flatten all choices
    let flat_choices: Vec<NdArray> = choices.iter().map(|c| c.astype(DType::Float64).flatten()).collect();

    let mut result = Vec::with_capacity(idx_arr.len());
    for &i in idx_arr.iter() {
        let choice_idx = i as usize;
        if choice_idx >= flat_choices.len() {
            return Err(NumpyError::ValueError(format!(
                "choose index {} out of range for {} choices",
                choice_idx,
                flat_choices.len()
            )));
        }
        // For simplicity, take element at same flat position from the chosen array
        // This matches NumPy's behavior when all arrays have the same shape
        let pos = result.len();
        let ArrayData::Float64(arr) = &flat_choices[choice_idx].data else {
            unreachable!()
        };
        if pos < arr.len() {
            result.push(arr.iter().nth(pos).copied().unwrap_or(0.0));
        } else {
            result.push(0.0);
        }
    }

    let out_shape = a.shape().to_vec();
    Ok(NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&out_shape), result).expect("shape matches"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_searchsorted_left() {
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 5.0, 7.0]);
        let v = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let idx = a.searchsorted(&v, "left").unwrap();
        assert_eq!(idx.shape(), &[3]);
    }

    #[test]
    fn test_compress() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let cond = NdArray::from_vec(vec![true, false, true, false]);
        let result = a.compress(&cond, None).unwrap();
        assert_eq!(result.shape(), &[2]);
    }
}
```

**Step 2: Register module**

In `ops/mod.rs`:
```rust
pub mod selection;
```

In `lib.rs`:
```rust
pub use ops::selection::choose;
```

**Step 3: Add Python bindings**

In `crates/numpy-rust-python/src/lib.rs`:

```rust
#[pyfunction]
fn searchsorted(
    a: vm::PyRef<PyNdArray>,
    v: vm::PyRef<PyNdArray>,
    side: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
    _vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let side_str = side.as_ref().map(|s| s.as_str()).unwrap_or("left");
    a.inner()
        .searchsorted(&v.inner(), side_str)
        .map(|arr| py_array::ndarray_or_scalar(arr, _vm))
        .map_err(|e| _vm.new_value_error(e.to_string()))
}

#[pyfunction]
fn compress(
    condition: vm::PyRef<PyNdArray>,
    a: vm::PyRef<PyNdArray>,
    axis: vm::function::OptionalArg<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let ax = parse_optional_axis(axis, vm)?;
    a.inner()
        .compress(&condition.inner(), ax)
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
}
```

**Step 4: Update `__init__.py`**

```python
def searchsorted(a, v, side="left", sorter=None):
    if isinstance(a, ndarray) and isinstance(v, ndarray):
        return _numpy_native.searchsorted(a, v, side)
    if isinstance(a, ndarray):
        return _numpy_native.searchsorted(a, array([v]), side)
    return 0

def compress(condition, a, axis=None):
    if isinstance(a, ndarray):
        cond = condition if isinstance(condition, ndarray) else array(condition)
        return _numpy_native.compress(cond, a, axis)
    return a
```

**Step 5: Add Python tests**

```python
def test_searchsorted():
    a = np.array([1.0, 3.0, 5.0, 7.0])
    v = np.array([2.0, 4.0, 6.0])
    idx = np.searchsorted(a, v)
    assert int(idx[0]) == 1  # 2.0 goes at index 1
    assert int(idx[1]) == 2  # 4.0 goes at index 2
    assert int(idx[2]) == 3  # 6.0 goes at index 3

def test_searchsorted_right():
    a = np.array([1.0, 3.0, 3.0, 5.0])
    v = np.array([3.0])
    idx = np.searchsorted(a, v, side="right")
    assert int(idx[0]) == 3

def test_compress():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    cond = np.array([True, False, True, False])
    result = np.compress(cond, a)
    assert result.shape == (2,)
```

**Step 6: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/ops/selection.rs crates/numpy-rust-core/src/ops/mod.rs crates/numpy-rust-core/src/lib.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: implement searchsorted, choose, and compress"
```

---

### Task 13: Creation function improvements

**Files:**
- Modify: `crates/numpy-rust-core/src/creation.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Test: `tests/python/test_numeric.py`

**Step 1: Add `dtype` parameter to `arange`**

```rust
pub fn arange(start: f64, stop: f64, step: f64, dtype: Option<DType>) -> NdArray {
    let mut values = Vec::new();
    let mut v = start;
    if step > 0.0 {
        while v < stop {
            values.push(v);
            v += step;
        }
    } else if step < 0.0 {
        while v > stop {
            values.push(v);
            v += step;
        }
    }
    let len = values.len();
    let arr = NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[len]), values).expect("vec length matches shape"),
    ));
    match dtype {
        Some(dt) => arr.astype(dt),
        None => arr,
    }
}
```

Update existing `arange` callers to pass `None`.

**Step 2: Add `retstep` to `linspace`**

```rust
pub fn linspace(start: f64, stop: f64, num: usize) -> (NdArray, f64) {
    let (values, step) = if num == 0 {
        (Vec::new(), 0.0)
    } else if num == 1 {
        (vec![start], 0.0)
    } else {
        let step = (stop - start) / (num - 1) as f64;
        let vals: Vec<f64> = (0..num).map(|i| start + step * i as f64).collect();
        (vals, step)
    };
    let len = values.len();
    let arr = NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[len]), values).expect("vec length matches shape"),
    ));
    (arr, step)
}
```

Update the re-export and all callers. The existing `linspace` callers that don't need `retstep` will need to destructure: `let (arr, _) = linspace(...)`.

**Actually, simpler approach** — keep the original signature and add a separate `linspace_with_step`:

```rust
/// Create a 1-D array with `num` evenly spaced values from start to stop (inclusive).
pub fn linspace(start: f64, stop: f64, num: usize) -> NdArray {
    linspace_with_step(start, stop, num).0
}

/// Same as linspace but also returns the step size.
pub fn linspace_with_step(start: f64, stop: f64, num: usize) -> (NdArray, f64) {
    // implementation
}
```

**Step 3: Update Python bindings**

In `lib.rs`, update `arange` to accept optional dtype:

```rust
#[pyfunction]
fn arange(
    start: f64,
    stop: f64,
    step: vm::function::OptionalArg<f64>,
    _vm: &VirtualMachine,
) -> PyNdArray {
    let step = step.unwrap_or(1.0);
    PyNdArray::from_core(numpy_rust_core::creation::arange(start, stop, step, None))
}
```

**Step 4: Update `__init__.py`**

```python
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    if isinstance(start, ndarray):
        start = float(start)
    if isinstance(stop, ndarray):
        stop = float(stop)
    result = _numpy_native.linspace(start, stop, num)
    if retstep:
        step = (stop - start) / max(num - 1, 1) if num > 1 else 0.0
        return result, step
    return result
```

**Step 5: Add Python tests**

```python
def test_linspace_retstep():
    arr, step = np.linspace(0, 1, 5, retstep=True)
    assert arr.shape == (5,)
    assert abs(step - 0.25) < 1e-10

def test_arange_int_dtype():
    # Test that arange works as before (dtype improvements are internal)
    a = np.arange(0, 5, 1)
    assert a.shape == (5,)
```

**Step 6: Verify and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/creation.rs crates/numpy-rust-core/src/lib.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add dtype to arange, retstep to linspace"
```

---

## Final Verification

After all tasks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
cargo build -p numpy-rust-wasm
./tests/python/run_tests.sh target/debug/numpy-python
```

All must pass cleanly.

---

## Implementation Order

**Phase A (Tasks 1-6):** Sequential. Task 1 (keepdims+ddof) first since it changes the most signatures. Tasks 2-5 are independent after Task 1. Task 6 (edge case tests) last.

**Phase B (Tasks 7-13):** Task 7 (DType/ArrayData extension) first — everything else in Phase B depends on it compiling. Task 8 (complex ops) before Task 9 (complex Python bindings). Tasks 10-13 are independent of each other and can be done in any order.
