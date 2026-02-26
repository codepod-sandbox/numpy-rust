# Tier 4: Most-Used Missing Functions — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the most commonly used missing NumPy functions: cumsum, cumprod, diff, split/vsplit/hsplit, argwhere, repeat, tile, prod, percentile/quantile, and ptp — all with proper axis support.

**Architecture:** Core implementations in `numpy-rust-core` (new `ops/cumulative.rs` and `ops/statistics.rs`, additions to `reduction.rs` and `manipulation.rs`), Python bindings in `numpy-rust-python`, thin wrappers in `python/numpy/__init__.py`. Every operation follows the existing lane-iteration pattern for axis support.

**Tech Stack:** Rust, ndarray crate (ArrayD, Axis, lanes/lanes_mut), RustPython bindings (#[pymethod], #[pyfunction])

---

### Task 1: `cumsum` and `cumprod` — Core

**Files:**
- Create: `crates/numpy-rust-core/src/ops/cumulative.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Step 1: Write the Rust unit tests**

Add to the bottom of the new `cumulative.rs`:

```rust
#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_cumsum_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let cs = a.cumsum(None).unwrap();
        assert_eq!(cs.shape(), &[4]);
        // Values: [1, 3, 6, 10]
        let expected = NdArray::from_vec(vec![1.0, 3.0, 6.0, 10.0]);
        let diff = (&cs - &expected).unwrap();
        assert!(diff.abs().max(None, false).unwrap().get(&[]).unwrap() == numpy_rust_core::indexing::Scalar::Float64(0.0));
    }

    #[test]
    fn test_cumsum_2d_axis0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3]).unwrap();
        let cs = a.cumsum(Some(0)).unwrap();
        assert_eq!(cs.shape(), &[2, 3]);
        // Row 0: [1,2,3], Row 1: [1+4, 2+5, 3+6] = [5,7,9]
    }

    #[test]
    fn test_cumsum_2d_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3]).unwrap();
        let cs = a.cumsum(Some(1)).unwrap();
        assert_eq!(cs.shape(), &[2, 3]);
    }

    #[test]
    fn test_cumprod_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let cp = a.cumprod(None).unwrap();
        assert_eq!(cp.shape(), &[4]);
        // Values: [1, 2, 6, 24]
    }

    #[test]
    fn test_cumprod_2d_axis0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3]).unwrap();
        let cp = a.cumprod(Some(0)).unwrap();
        assert_eq!(cp.shape(), &[2, 3]);
    }
}
```

**Step 2: Implement `cumsum` and `cumprod` in core**

Create `crates/numpy-rust-core/src/ops/cumulative.rs`:

```rust
use ndarray::{ArrayD, Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Cumulative sum along an axis. axis=None flattens first.
    pub fn cumsum(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError("cumsum not supported for string arrays".into()));
        }
        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        match axis {
            None => {
                let flat: Vec<f64> = arr.iter().copied().collect();
                let mut result = Vec::with_capacity(flat.len());
                let mut acc = 0.0;
                for v in &flat {
                    acc += v;
                    result.push(acc);
                }
                Ok(NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[result.len()]), result)
                        .expect("flat vec matches shape"),
                )))
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis { axis: ax, ndim: self.ndim() });
                }
                let mut out = arr.clone();
                for mut lane in out.lanes_mut(Axis(ax)) {
                    let mut acc = 0.0;
                    for v in lane.iter_mut() {
                        acc += *v;
                        *v = acc;
                    }
                }
                Ok(NdArray::from_data(ArrayData::Float64(out)))
            }
        }
    }

    /// Cumulative product along an axis. axis=None flattens first.
    pub fn cumprod(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError("cumprod not supported for string arrays".into()));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError("cumprod not supported for complex arrays".into()));
        }
        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        match axis {
            None => {
                let flat: Vec<f64> = arr.iter().copied().collect();
                let mut result = Vec::with_capacity(flat.len());
                let mut acc = 1.0;
                for v in &flat {
                    acc *= v;
                    result.push(acc);
                }
                Ok(NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[result.len()]), result)
                        .expect("flat vec matches shape"),
                )))
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis { axis: ax, ndim: self.ndim() });
                }
                let mut out = arr.clone();
                for mut lane in out.lanes_mut(Axis(ax)) {
                    let mut acc = 1.0;
                    for v in lane.iter_mut() {
                        acc *= *v;
                        *v = acc;
                    }
                }
                Ok(NdArray::from_data(ArrayData::Float64(out)))
            }
        }
    }
}
```

Register module in `crates/numpy-rust-core/src/ops/mod.rs` — add `pub mod cumulative;` line.

No re-export needed in `lib.rs` (methods on NdArray are available automatically).

**Step 3: Run tests**

```bash
cargo test --workspace --all-features
```

Expected: All existing 229+ tests pass, plus 5 new cumulative tests.

**Step 4: Commit**

```bash
git add crates/numpy-rust-core/src/ops/cumulative.rs crates/numpy-rust-core/src/ops/mod.rs
git commit -m "feat: add cumsum and cumprod core operations"
```

---

### Task 2: `diff` — Core

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/cumulative.rs`

**Step 1: Write the Rust unit tests**

Add to tests in `cumulative.rs`:

```rust
    #[test]
    fn test_diff_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 6.0, 10.0]);
        let d = a.diff(1, None).unwrap();
        assert_eq!(d.shape(), &[3]);
        // Values: [2, 3, 4]
    }

    #[test]
    fn test_diff_n2() {
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 6.0, 10.0]);
        let d = a.diff(2, None).unwrap();
        assert_eq!(d.shape(), &[2]);
        // Values: [1, 1]
    }

    #[test]
    fn test_diff_2d_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 4.0, 7.0, 3.0, 5.0, 9.0, 15.0])
            .reshape(&[2, 4]).unwrap();
        let d = a.diff(1, Some(1)).unwrap();
        assert_eq!(d.shape(), &[2, 3]);
    }

    #[test]
    fn test_diff_2d_axis0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3]).unwrap();
        let d = a.diff(1, Some(0)).unwrap();
        assert_eq!(d.shape(), &[1, 3]);
    }
```

**Step 2: Implement `diff`**

Add to `cumulative.rs`:

```rust
impl NdArray {
    /// N-th discrete difference along axis. axis=None flattens first.
    /// Result shape has `shape[axis] - n` along the diff axis.
    pub fn diff(&self, n: usize, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError("diff not supported for string arrays".into()));
        }
        if n == 0 {
            return Ok(self.clone());
        }

        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        match axis {
            None => {
                let flat: Vec<f64> = arr.iter().copied().collect();
                let mut current = flat;
                for _ in 0..n {
                    if current.len() <= 1 {
                        return Ok(NdArray::from_data(ArrayData::Float64(
                            ArrayD::from_shape_vec(IxDyn(&[0]), vec![])
                                .expect("empty shape valid"),
                        )));
                    }
                    current = current.windows(2).map(|w| w[1] - w[0]).collect();
                }
                Ok(NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[current.len()]), current)
                        .expect("flat vec matches shape"),
                )))
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis { axis: ax, ndim: self.ndim() });
                }
                let mut current = arr.clone();
                for _ in 0..n {
                    let axis_len = current.shape()[ax];
                    if axis_len <= 1 {
                        let mut new_shape = current.shape().to_vec();
                        new_shape[ax] = 0;
                        return Ok(NdArray::from_data(ArrayData::Float64(
                            ArrayD::zeros(IxDyn(&new_shape)),
                        )));
                    }
                    // Slice [1:] - [:-1] along axis
                    use ndarray::SliceInfoElem;
                    let mut slice_hi: Vec<SliceInfoElem> = (0..current.ndim())
                        .map(|_| SliceInfoElem::Slice { start: 0, end: None, step: 1 })
                        .collect();
                    let mut slice_lo = slice_hi.clone();
                    slice_hi[ax] = SliceInfoElem::Slice { start: 1, end: None, step: 1 };
                    slice_lo[ax] = SliceInfoElem::Slice { start: 0, end: Some((axis_len - 1) as isize), step: 1 };
                    let hi = current.slice(slice_hi.as_slice()).to_owned();
                    let lo = current.slice(slice_lo.as_slice()).to_owned();
                    current = &hi - &lo;
                }
                Ok(NdArray::from_data(ArrayData::Float64(current)))
            }
        }
    }
}
```

**Step 3: Run tests**

```bash
cargo test --workspace --all-features
```

**Step 4: Commit**

```bash
git add crates/numpy-rust-core/src/ops/cumulative.rs
git commit -m "feat: add diff (discrete difference) core operation"
```

---

### Task 3: `cumsum`, `cumprod`, `diff` — Python bindings

**Files:**
- Modify: `crates/numpy-rust-python/src/py_array.rs` (add pymethods)
- Modify: `crates/numpy-rust-python/src/lib.rs` (add pyfunctions)
- Modify: `python/numpy/__init__.py` (add wrappers)
- Modify: `tests/python/test_numeric.py` (add tests)

**Step 1: Add pymethods to `py_array.rs`**

Add alongside existing `sum`/`mean` methods (around line 408):

```rust
    #[pymethod]
    fn cumsum(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .cumsum(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pymethod]
    fn cumprod(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .cumprod(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

**Step 2: Add pyfunctions to `lib.rs`**

```rust
    #[pyfunction]
    fn cumsum(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .cumsum(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn cumprod(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .cumprod(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn diff(
        a: vm::PyRef<PyNdArray>,
        n: vm::function::OptionalArg<usize>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let n = n.unwrap_or(1);
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .diff(n, axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

**Step 3: Add Python wrappers to `__init__.py`**

```python
def cumsum(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        return a.cumsum(axis)
    return array(a).cumsum(axis)

def cumprod(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        return a.cumprod(axis)
    return array(a).cumprod(axis)

def diff(a, n=1, axis=-1, prepend=None, append=None):
    return _native.diff(a, n, axis)
```

**Step 4: Add Python integration tests to `test_numeric.py`**

```python
def test_cumsum_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.cumsum(a)
    assert result.shape == (4,)
    expected = np.array([1.0, 3.0, 6.0, 10.0])
    assert np.allclose(result, expected)

def test_cumsum_2d_axis0():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.cumsum(a, axis=0)
    assert result.shape == (2, 2)
    expected = np.array([[1.0, 2.0], [4.0, 6.0]])
    assert np.allclose(result, expected)

def test_cumprod_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.cumprod(a)
    expected = np.array([1.0, 2.0, 6.0, 24.0])
    assert np.allclose(result, expected)

def test_diff_1d():
    a = np.array([1.0, 3.0, 6.0, 10.0])
    result = np.diff(a)
    assert result.shape == (3,)
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(result, expected)

def test_diff_n2():
    a = np.array([1.0, 3.0, 6.0, 10.0])
    result = np.diff(a, n=2)
    assert result.shape == (2,)
    expected = np.array([1.0, 1.0])
    assert np.allclose(result, expected)
```

**Step 5: Build and test**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
```

**Step 6: Commit**

```bash
git add crates/numpy-rust-python/src/py_array.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add cumsum, cumprod, diff Python bindings and tests"
```

---

### Task 4: `prod` with axis/keepdims — Core + Python

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/reduction.rs`
- Modify: `crates/numpy-rust-python/src/py_array.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Rust unit tests to `reduction.rs`**

```rust
    #[test]
    fn test_prod_all() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let p = a.prod(None, false).unwrap();
        assert_eq!(p.shape(), &[]);
    }

    #[test]
    fn test_prod_axis() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let p = a.prod(Some(0), false).unwrap();
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_prod_keepdims() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let p = a.prod(Some(1), true).unwrap();
        assert_eq!(p.shape(), &[2, 1]);
    }
```

**Step 2: Implement `prod` in `reduction.rs`**

Add the method alongside `sum`:

```rust
    /// Product of array elements over a given axis, or all elements if axis is None.
    pub fn prod(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError("prod not supported for string arrays".into()));
        }
        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        let result = match axis {
            None => {
                let p: f64 = arr.iter().product();
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), p)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let prod_arr = arr.fold_axis(Axis(ax), 1.0, |&acc, &x| acc * x);
                NdArray::from_data(ArrayData::Float64(prod_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }
```

**Step 3: Add pymethod and pyfunction**

`py_array.rs` — add `prod` method (same pattern as `sum`):

```rust
    #[pymethod]
    fn prod(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .prod(axis, keepdims)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

`lib.rs` — add `prod` pyfunction:

```rust
    #[pyfunction]
    fn prod(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .prod(axis, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

**Step 4: Replace `prod` stub in `__init__.py`**

Replace the existing broken `prod` function:

```python
def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        return a.prod(axis, keepdims)
    return _native.prod(array(a), axis, keepdims)
```

**Step 5: Add Python tests**

```python
def test_prod_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert float(np.prod(a)) == 24.0

def test_prod_2d_axis():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.prod(a, axis=0)
    expected = np.array([3.0, 8.0])
    assert np.allclose(result, expected)

def test_prod_keepdims():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.prod(a, axis=1, keepdims=True)
    assert result.shape == (2, 1)
```

**Step 6: Build, test, commit**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/ops/reduction.rs crates/numpy-rust-python/src/py_array.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add prod reduction with axis and keepdims support"
```

---

### Task 5: `split`, `vsplit`, `hsplit` — Core + Python

**Files:**
- Modify: `crates/numpy-rust-core/src/manipulation.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs` (re-export)
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Rust unit tests to `manipulation.rs`**

```rust
    #[test]
    fn test_split_equal() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let parts = split(&a, &SplitSpec::NSections(3), 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[2].shape(), &[2]);
    }

    #[test]
    fn test_split_indices() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let parts = split(&a, &SplitSpec::Indices(vec![2, 4]), 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[2].shape(), &[1]);
    }

    #[test]
    fn test_split_2d_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .reshape(&[2, 4]).unwrap();
        let parts = split(&a, &SplitSpec::NSections(2), 1).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[1].shape(), &[2, 2]);
    }
```

**Step 2: Implement `split`, `vsplit`, `hsplit`**

Add to `manipulation.rs`:

```rust
/// Specification for how to split an array.
pub enum SplitSpec {
    /// Split into N equal sections.
    NSections(usize),
    /// Split at the given indices.
    Indices(Vec<usize>),
}

/// Split array along an axis.
pub fn split(a: &NdArray, spec: &SplitSpec, axis: usize) -> Result<Vec<NdArray>> {
    if axis >= a.ndim() {
        return Err(NumpyError::InvalidAxis { axis, ndim: a.ndim() });
    }
    let axis_len = a.shape()[axis];

    let indices = match spec {
        SplitSpec::NSections(n) => {
            if *n == 0 {
                return Err(NumpyError::ValueError("number of sections must be > 0".into()));
            }
            if axis_len % n != 0 {
                return Err(NumpyError::ValueError(format!(
                    "array split does not result in an equal division: {} into {}",
                    axis_len, n
                )));
            }
            let section_size = axis_len / n;
            (1..*n).map(|i| i * section_size).collect::<Vec<_>>()
        }
        SplitSpec::Indices(idx) => idx.clone(),
    };

    // Build split points: [0, idx[0], idx[1], ..., axis_len]
    let mut boundaries = Vec::with_capacity(indices.len() + 2);
    boundaries.push(0usize);
    boundaries.extend_from_slice(&indices);
    boundaries.push(axis_len);

    let mut result = Vec::with_capacity(boundaries.len() - 1);
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        // Build slice args: Full for all axes except `axis` which gets start:end
        let mut args = Vec::with_capacity(a.ndim());
        for i in 0..a.ndim() {
            if i == axis {
                args.push(crate::indexing::SliceArg::Range {
                    start: Some(start as isize),
                    stop: Some(end as isize),
                    step: 1,
                });
            } else {
                args.push(crate::indexing::SliceArg::Full);
            }
        }
        result.push(a.slice(&args)?);
    }

    Ok(result)
}

/// Split along axis 0.
pub fn vsplit(a: &NdArray, spec: &SplitSpec) -> Result<Vec<NdArray>> {
    split(a, spec, 0)
}

/// Split along axis 1 (or axis 0 for 1-D).
pub fn hsplit(a: &NdArray, spec: &SplitSpec) -> Result<Vec<NdArray>> {
    if a.ndim() == 1 {
        split(a, spec, 0)
    } else {
        split(a, spec, 1)
    }
}
```

Add re-exports to `lib.rs`:

```rust
pub use manipulation::{concatenate, hstack, split, hsplit, vsplit, stack, vstack, SplitSpec};
```

**Step 3: Add Python bindings**

`lib.rs` — add `split` pyfunction. The tricky part is parsing `indices_or_sections` which can be an int or a list:

```rust
    #[pyfunction]
    fn split(
        a: vm::PyRef<PyNdArray>,
        indices_or_sections: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = axis.unwrap_or(0);
        let spec = if let Ok(n) = indices_or_sections.try_to_value::<usize>(vm) {
            numpy_rust_core::SplitSpec::NSections(n)
        } else if let Some(list) = indices_or_sections.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            let indices: Vec<usize> = items
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?;
            numpy_rust_core::SplitSpec::Indices(indices)
        } else {
            return Err(vm.new_type_error("indices_or_sections must be int or list".into()));
        };
        let parts = numpy_rust_core::split(&a.inner(), &spec, axis)
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_parts: Vec<PyObjectRef> = parts
            .into_iter()
            .map(|p| PyNdArray::from_core(p).into_pyobject(vm))
            .collect();
        Ok(vm.ctx.new_list(py_parts).into())
    }
```

**Step 4: Add `__init__.py` wrappers**

```python
def split(a, indices_or_sections, axis=0):
    return _native.split(a, indices_or_sections, axis)

def vsplit(a, indices_or_sections):
    return split(a, indices_or_sections, 0)

def hsplit(a, indices_or_sections):
    if a.ndim == 1:
        return split(a, indices_or_sections, 0)
    return split(a, indices_or_sections, 1)
```

**Step 5: Add Python tests**

```python
def test_split_equal():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    parts = np.split(a, 3)
    assert len(parts) == 3
    assert parts[0].shape == (2,)
    assert np.allclose(parts[0], np.array([1.0, 2.0]))

def test_split_indices():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    parts = np.split(a, [2, 4])
    assert len(parts) == 3
    assert parts[0].shape == (2,)
    assert parts[1].shape == (2,)
    assert parts[2].shape == (1,)

def test_hsplit():
    a = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    parts = np.hsplit(a, 2)
    assert len(parts) == 2
    assert parts[0].shape == (2, 2)

def test_vsplit():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    parts = np.vsplit(a, 2)
    assert len(parts) == 2
    assert parts[0].shape == (2, 2)
```

**Step 6: Build, test, commit**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/manipulation.rs crates/numpy-rust-core/src/lib.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add split, vsplit, hsplit array splitting operations"
```

---

### Task 6: `argwhere` — Core + Python

**Files:**
- Modify: `crates/numpy-rust-core/src/utility.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs` (re-export)
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Rust unit tests to `utility.rs`**

```rust
    #[test]
    fn test_argwhere_1d() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0, 0.0, 3.0, 0.0]);
        let result = argwhere(&a);
        assert_eq!(result.shape(), &[2, 1]); // 2 nonzero elements, 1D → (N, 1)
    }

    #[test]
    fn test_argwhere_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 0.0, 0.0, 4.0])
            .reshape(&[2, 2]).unwrap();
        let result = argwhere(&a);
        assert_eq!(result.shape(), &[2, 2]); // 2 nonzero elements, 2D → (N, 2)
    }

    #[test]
    fn test_argwhere_all_zero() {
        let a = NdArray::from_vec(vec![0.0_f64, 0.0, 0.0]);
        let result = argwhere(&a);
        assert_eq!(result.shape(), &[0, 1]); // 0 nonzero, 1D → (0, 1)
    }
```

**Step 2: Implement `argwhere`**

Add to `utility.rs`:

```rust
/// Return the indices of non-zero elements as an (N, ndim) Int64 array.
/// Like `numpy.argwhere(a)`.
pub fn argwhere(a: &NdArray) -> NdArray {
    let shape = a.shape().to_vec();
    let ndim = a.ndim();
    let flat = a.astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else { unreachable!() };

    // Collect indices of non-zero elements
    let mut coords: Vec<i64> = Vec::new();
    let mut count = 0usize;

    for (linear_idx, &val) in arr.iter().enumerate() {
        if val != 0.0 {
            // Unravel linear index to multi-dimensional coordinates
            let mut remaining = linear_idx;
            let mut coord = vec![0i64; ndim];
            for d in (0..ndim).rev() {
                coord[d] = (remaining % shape[d]) as i64;
                remaining /= shape[d];
            }
            coords.extend_from_slice(&coord);
            count += 1;
        }
    }

    let result_shape = if ndim == 0 {
        vec![count, 0]
    } else {
        vec![count, ndim]
    };
    NdArray::from_data(ArrayData::Int64(
        ArrayD::from_shape_vec(IxDyn(&result_shape), coords)
            .expect("coords match result shape"),
    ))
}
```

Add re-export in `lib.rs`:

```rust
pub use utility::{argwhere, dot, where_cond};
```

**Step 3: Add Python bindings**

`lib.rs`:

```rust
    #[pyfunction]
    fn argwhere(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::argwhere(&a.inner()))
    }
```

`__init__.py`:

```python
def argwhere(a):
    return _native.argwhere(a)
```

**Step 4: Python tests**

```python
def test_argwhere_1d():
    a = np.array([0.0, 1.0, 0.0, 3.0, 0.0])
    result = np.argwhere(a)
    assert result.shape == (2, 1)

def test_argwhere_2d():
    a = np.array([[1.0, 0.0], [0.0, 4.0]])
    result = np.argwhere(a)
    assert result.shape == (2, 2)

def test_argwhere_all_zero():
    a = np.array([0.0, 0.0, 0.0])
    result = np.argwhere(a)
    assert result.shape == (0, 1)
```

**Step 5: Build, test, commit**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/utility.rs crates/numpy-rust-core/src/lib.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add argwhere (indices of non-zero elements)"
```

---

### Task 7: `repeat` and `tile` — Core + Python

**Files:**
- Modify: `crates/numpy-rust-core/src/manipulation.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Rust unit tests to `manipulation.rs`**

```rust
    #[test]
    fn test_repeat_flat() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let r = repeat(&a, 2, None).unwrap();
        assert_eq!(r.shape(), &[6]); // [1,1,2,2,3,3]
    }

    #[test]
    fn test_repeat_axis0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2]).unwrap();
        let r = repeat(&a, 3, Some(0)).unwrap();
        assert_eq!(r.shape(), &[6, 2]); // each row repeated 3 times
    }

    #[test]
    fn test_tile_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let t = tile(&a, &[3]).unwrap();
        assert_eq!(t.shape(), &[6]); // [1,2,1,2,1,2]
    }

    #[test]
    fn test_tile_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2]).unwrap();
        let t = tile(&a, &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[4, 6]); // 2x repeat rows, 3x repeat cols
    }
```

**Step 2: Implement `repeat` and `tile`**

Add to `manipulation.rs`:

```rust
/// Repeat elements of an array.
/// axis=None: flatten then repeat. axis=Some: repeat along that axis.
pub fn repeat(a: &NdArray, repeats: usize, axis: Option<usize>) -> Result<NdArray> {
    match axis {
        None => {
            let flat = a.flatten();
            repeat_along_axis(&flat, repeats, 0)
        }
        Some(ax) => {
            if ax >= a.ndim() {
                return Err(NumpyError::InvalidAxis { axis: ax, ndim: a.ndim() });
            }
            repeat_along_axis(a, repeats, ax)
        }
    }
}

fn repeat_along_axis(a: &NdArray, repeats: usize, axis: usize) -> Result<NdArray> {
    // For each slice along axis, duplicate it `repeats` times
    let axis_len = a.shape()[axis];
    let mut parts = Vec::with_capacity(axis_len * repeats);
    for i in 0..axis_len {
        let slice = a.index_select(axis, &[i])?;
        for _ in 0..repeats {
            parts.push(slice.clone());
        }
    }
    let refs: Vec<&NdArray> = parts.iter().collect();
    concatenate(&refs, axis)
}

/// Tile an array by repeating it along each axis.
pub fn tile(a: &NdArray, reps: &[usize]) -> Result<NdArray> {
    if reps.is_empty() {
        return Ok(a.clone());
    }
    // Pad reps or array dims so they match
    let ndim = a.ndim().max(reps.len());
    let mut arr = a.clone();

    // If reps has more dims than array, prepend size-1 dims to array
    while arr.ndim() < ndim {
        arr = arr.expand_dims(0)?;
    }

    // If array has more dims than reps, prepend 1s to reps
    let mut full_reps = vec![1usize; ndim];
    let offset = ndim - reps.len();
    for (i, &r) in reps.iter().enumerate() {
        full_reps[offset + i] = r;
    }

    // Concatenate along each axis
    for (ax, &r) in full_reps.iter().enumerate() {
        if r > 1 {
            let copies: Vec<NdArray> = (0..r).map(|_| arr.clone()).collect();
            let refs: Vec<&NdArray> = copies.iter().collect();
            arr = concatenate(&refs, ax)?;
        }
    }

    Ok(arr)
}
```

**Step 3: Add Python bindings**

`lib.rs`:

```rust
    #[pyfunction]
    fn repeat(
        a: vm::PyRef<PyNdArray>,
        repeats: usize,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        numpy_rust_core::manipulation::repeat(&a.inner(), repeats, axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn tile(
        a: vm::PyRef<PyNdArray>,
        reps: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        // reps can be an int or a tuple/list
        let reps_vec = if let Ok(n) = reps.try_to_value::<usize>(vm) {
            vec![n]
        } else if let Some(tuple) = reps.downcast_ref::<vm::builtins::PyTuple>() {
            tuple
                .as_slice()
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        } else if let Some(list) = reps.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            return Err(vm.new_type_error("reps must be int, tuple, or list".into()));
        };
        numpy_rust_core::manipulation::tile(&a.inner(), &reps_vec)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

**Step 4: Replace `repeat` and `tile` stubs in `__init__.py`**

```python
def repeat(a, repeats, axis=None):
    return _native.repeat(a, repeats, axis)

def tile(a, reps):
    return _native.tile(a, reps)
```

**Step 5: Python tests**

```python
def test_repeat_flat():
    a = np.array([1.0, 2.0, 3.0])
    result = np.repeat(a, 2)
    assert result.shape == (6,)
    expected = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    assert np.allclose(result, expected)

def test_repeat_axis():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.repeat(a, 2, axis=0)
    assert result.shape == (4, 2)

def test_tile_1d():
    a = np.array([1.0, 2.0])
    result = np.tile(a, 3)
    assert result.shape == (6,)
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    assert np.allclose(result, expected)

def test_tile_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.tile(a, (2, 3))
    assert result.shape == (4, 6)
```

**Step 6: Build, test, commit**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/manipulation.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add repeat and tile with proper axis support"
```

---

### Task 8: `percentile` and `quantile` — Core + Python

**Files:**
- Create: `crates/numpy-rust-core/src/ops/statistics.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Rust unit tests**

```rust
#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};
    use crate::indexing::Scalar;

    #[test]
    fn test_quantile_median() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let q = a.quantile(0.5, None).unwrap();
        assert_eq!(q.shape(), &[]);
        assert_eq!(q.get(&[]).unwrap(), Scalar::Float64(3.0));
    }

    #[test]
    fn test_quantile_min_max() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let q0 = a.quantile(0.0, None).unwrap();
        let q1 = a.quantile(1.0, None).unwrap();
        assert_eq!(q0.get(&[]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(q1.get(&[]).unwrap(), Scalar::Float64(3.0));
    }

    #[test]
    fn test_quantile_interpolated() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let q = a.quantile(0.25, None).unwrap();
        // Linear interpolation: index = 0.25 * 3 = 0.75 → 1.0 + 0.75*(2.0-1.0) = 1.75
        assert_eq!(q.get(&[]).unwrap(), Scalar::Float64(1.75));
    }

    #[test]
    fn test_quantile_axis() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0, 6.0, 4.0, 5.0])
            .reshape(&[2, 3]).unwrap();
        let q = a.quantile(0.5, Some(1)).unwrap();
        assert_eq!(q.shape(), &[2]); // median of each row
    }
}
```

**Step 2: Implement `quantile` and `percentile`**

Create `crates/numpy-rust-core/src/ops/statistics.rs`:

```rust
use ndarray::{ArrayD, Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Compute the q-th quantile (0.0 to 1.0) using linear interpolation.
    /// axis=None: compute over flattened array.
    pub fn quantile(&self, q: f64, axis: Option<usize>) -> Result<NdArray> {
        if q < 0.0 || q > 1.0 {
            return Err(NumpyError::ValueError(format!(
                "quantile q must be between 0 and 1, got {}", q
            )));
        }
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError("quantile not supported for string arrays".into()));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError("quantile not supported for complex arrays".into()));
        }

        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        match axis {
            None => {
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                if flat.is_empty() {
                    return Err(NumpyError::ValueError("empty array".into()));
                }
                flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let val = interpolate_quantile(&flat, q);
                Ok(NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_elem(IxDyn(&[]), val),
                )))
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis { axis: ax, ndim: self.ndim() });
                }
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() {
                    IxDyn(&[])
                } else {
                    IxDyn(&result_shape)
                };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, result_elem) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    let mut v: Vec<f64> = lane.iter().copied().collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    *result_elem = interpolate_quantile(&v, q);
                }
                Ok(NdArray::from_data(ArrayData::Float64(result)))
            }
        }
    }

    /// Compute the q-th percentile (0 to 100).
    pub fn percentile(&self, q: f64, axis: Option<usize>) -> Result<NdArray> {
        self.quantile(q / 100.0, axis)
    }
}

/// Linear interpolation for quantile on a sorted slice.
fn interpolate_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= n {
        sorted[n - 1]
    } else {
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}
```

Register: add `pub mod statistics;` to `ops/mod.rs`.

**Step 3: Add Python bindings**

`lib.rs`:

```rust
    #[pyfunction]
    fn quantile(
        a: vm::PyRef<PyNdArray>,
        q: f64,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .quantile(q, axis)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn percentile(
        a: vm::PyRef<PyNdArray>,
        q: f64,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .percentile(q, axis)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

`__init__.py`:

```python
def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    return _native.quantile(a, float(q), axis)

def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    return _native.percentile(a, float(q), axis)

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    return _native.quantile(a, 0.5, axis)
```

**Step 4: Python tests**

```python
def test_quantile_median():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = float(np.quantile(a, 0.5))
    assert result == 3.0

def test_percentile_25():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = float(np.percentile(a, 25))
    assert abs(result - 1.75) < 1e-10

def test_median():
    a = np.array([3.0, 1.0, 2.0])
    result = float(np.median(a))
    assert result == 2.0

def test_quantile_axis():
    a = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    result = np.quantile(a, 0.5, axis=1)
    assert result.shape == (2,)
```

**Step 5: Build, test, commit**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add crates/numpy-rust-core/src/ops/statistics.rs crates/numpy-rust-core/src/ops/mod.rs crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add quantile, percentile, and median"
```

---

### Task 9: `ptp` with axis — Python fix

**Files:**
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

This is a thin wrapper — no core Rust changes needed since `max(axis) - min(axis)` already works.

**Step 1: Fix `ptp` in `__init__.py`**

Replace the existing broken stub:

```python
def ptp(a, axis=None):
    if isinstance(a, ndarray):
        return a.max(axis) - a.min(axis)
    return 0
```

**Step 2: Python tests**

```python
def test_ptp_1d():
    a = np.array([3.0, 1.0, 7.0, 2.0])
    result = float(np.ptp(a))
    assert result == 6.0

def test_ptp_2d_axis():
    a = np.array([[3.0, 1.0], [7.0, 2.0]])
    result = np.ptp(a, axis=0)
    expected = np.array([4.0, 1.0])
    assert np.allclose(result, expected)

def test_ptp_2d_axis1():
    a = np.array([[3.0, 1.0], [7.0, 2.0]])
    result = np.ptp(a, axis=1)
    expected = np.array([2.0, 5.0])
    assert np.allclose(result, expected)
```

**Step 3: Build, test, commit**

```bash
cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python
git add python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "fix: ptp now supports axis parameter"
```

---

## Verification

After all tasks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
./tests/python/run_tests.sh target/debug/numpy-python
```

## Summary

| Task | Feature | New files | Modified files |
|------|---------|-----------|----------------|
| 1 | cumsum, cumprod core | `ops/cumulative.rs` | `ops/mod.rs` |
| 2 | diff core | | `ops/cumulative.rs` |
| 3 | cumsum/cumprod/diff Python | | `py_array.rs`, `lib.rs`, `__init__.py`, tests |
| 4 | prod with axis | | `reduction.rs`, `py_array.rs`, `lib.rs`, `__init__.py`, tests |
| 5 | split/vsplit/hsplit | | `manipulation.rs`, `lib.rs`(x2), `__init__.py`, tests |
| 6 | argwhere | | `utility.rs`, `lib.rs`(x2), `__init__.py`, tests |
| 7 | repeat, tile | | `manipulation.rs`, `lib.rs`, `__init__.py`, tests |
| 8 | percentile/quantile/median | `ops/statistics.rs` | `ops/mod.rs`, `lib.rs`, `__init__.py`, tests |
| 9 | ptp with axis | | `__init__.py`, tests |
