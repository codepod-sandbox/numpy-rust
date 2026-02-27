# Tier 8: NaN-safe Functions + Statistics — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NaN-safe reductions, correlation/covariance, histogram/bincount, set operations, stacking fixes, and index utilities to maximize numpy compatibility for LLM-generated data analysis code.

**Architecture:** New Rust modules `ops/nan_reduction.rs` and `ops/correlation.rs`, extensions to `ops/statistics.rs`, `ops/selection.rs`, `manipulation.rs`, and `indexing.rs`. Python bindings in `lib.rs`, wrappers in `python/numpy/__init__.py`. All follow existing patterns.

**Tech Stack:** Rust (ndarray, num-complex), RustPython bindings, Python wrapper layer.

---

### Task 1: NaN-safe Reductions — Core Rust

**Files:**
- Create: `crates/numpy-rust-core/src/ops/nan_reduction.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs` (add `pub mod nan_reduction;`)

**Step 1: Create `nan_reduction.rs` with all 9 nan-safe functions**

```rust
use ndarray::{ArrayD, Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Helper: filter NaN values from a float slice, returning non-NaN values.
fn filter_nan(vals: &[f64]) -> Vec<f64> {
    vals.iter().copied().filter(|v| !v.is_nan()).collect()
}

/// Helper: get the float64 array data, casting if needed.
fn as_f64(arr: &NdArray) -> NdArray {
    arr.astype(DType::Float64)
}

/// Validate axis and return the axis size.
fn validate_axis(axis: usize, ndim: usize) -> Result<()> {
    if axis >= ndim {
        return Err(NumpyError::InvalidAxis { axis, ndim });
    }
    Ok(())
}

/// If `keepdims` is true, re-insert a size-1 dimension at `axis`.
fn maybe_keepdims(
    result: NdArray,
    axis: Option<usize>,
    keepdims: bool,
    original_ndim: usize,
) -> NdArray {
    if !keepdims {
        return result;
    }
    if let Some(ax) = axis {
        let mut new_shape = result.shape().to_vec();
        new_shape.insert(ax, 1);
        result.reshape(&new_shape).expect("keepdims reshape cannot fail")
    } else {
        let new_shape = vec![1; original_ndim];
        result.reshape(&new_shape).expect("keepdims reshape cannot fail")
    }
}

impl NdArray {
    /// Sum of array elements, ignoring NaN values.
    pub fn nansum(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let s: f64 = arr.iter().filter(|v| !v.is_nan()).sum();
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), s)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    *out = lane.iter().filter(|v| !v.is_nan()).sum();
                }
                NdArray::from_data(ArrayData::Float64(result))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Mean of array elements, ignoring NaN values.
    pub fn nanmean(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let valid: Vec<f64> = filter_nan(&arr.iter().copied().collect::<Vec<_>>());
                let mean = if valid.is_empty() { f64::NAN } else { valid.iter().sum::<f64>() / valid.len() as f64 };
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), mean)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    let valid: Vec<f64> = lane.iter().copied().filter(|v| !v.is_nan()).collect();
                    *out = if valid.is_empty() { f64::NAN } else { valid.iter().sum::<f64>() / valid.len() as f64 };
                }
                NdArray::from_data(ArrayData::Float64(result))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Variance of array elements, ignoring NaN values.
    pub fn nanvar(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let valid: Vec<f64> = filter_nan(&arr.iter().copied().collect::<Vec<_>>());
                let val = compute_var(&valid, ddof);
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), val)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    let valid: Vec<f64> = lane.iter().copied().filter(|v| !v.is_nan()).collect();
                    *out = compute_var(&valid, ddof);
                }
                NdArray::from_data(ArrayData::Float64(result))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Standard deviation, ignoring NaN values.
    pub fn nanstd(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        let var = self.nanvar(axis, ddof, false)?;
        let result = var.sqrt();
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Minimum, ignoring NaN values. Raises error if all values are NaN.
    pub fn nanmin(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let val = arr.iter().copied().filter(|v| !v.is_nan())
                    .reduce(f64::min)
                    .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), val)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    *out = lane.iter().copied().filter(|v| !v.is_nan())
                        .reduce(f64::min)
                        .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                }
                NdArray::from_data(ArrayData::Float64(result))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Maximum, ignoring NaN values. Raises error if all values are NaN.
    pub fn nanmax(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let val = arr.iter().copied().filter(|v| !v.is_nan())
                    .reduce(f64::max)
                    .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), val)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    *out = lane.iter().copied().filter(|v| !v.is_nan())
                        .reduce(f64::max)
                        .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                }
                NdArray::from_data(ArrayData::Float64(result))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Index of minimum, ignoring NaN values.
    pub fn nanargmin(&self, axis: Option<usize>) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        match axis {
            None => {
                let (idx, _) = arr.iter().enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                Ok(NdArray::from_data(ArrayData::Int64(ArrayD::from_elem(IxDyn(&[]), idx as i64))))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<i64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    let (idx, _) = lane.iter().enumerate()
                        .filter(|(_, v)| !v.is_nan())
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                    *out = idx as i64;
                }
                Ok(NdArray::from_data(ArrayData::Int64(result)))
            }
        }
    }

    /// Index of maximum, ignoring NaN values.
    pub fn nanargmax(&self, axis: Option<usize>) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        match axis {
            None => {
                let (idx, _) = arr.iter().enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                Ok(NdArray::from_data(ArrayData::Int64(ArrayD::from_elem(IxDyn(&[]), idx as i64))))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<i64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    let (idx, _) = lane.iter().enumerate()
                        .filter(|(_, v)| !v.is_nan())
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .ok_or_else(|| NumpyError::ValueError("All-NaN slice encountered".into()))?;
                    *out = idx as i64;
                }
                Ok(NdArray::from_data(ArrayData::Int64(result)))
            }
        }
    }

    /// Product of array elements, ignoring NaN values (NaN treated as 1).
    pub fn nanprod(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = as_f64(self);
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let p: f64 = arr.iter().filter(|v| !v.is_nan()).product();
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), p)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() { IxDyn(&[]) } else { IxDyn(&result_shape) };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, out) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    *out = lane.iter().filter(|v| !v.is_nan()).product();
                }
                NdArray::from_data(ArrayData::Float64(result))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }
}

/// Compute variance of a slice (NaN-free values already filtered).
fn compute_var(values: &[f64], ddof: usize) -> f64 {
    let n = values.len();
    if n == 0 || n <= ddof {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
    sum_sq / (n - ddof) as f64
}

#[cfg(test)]
mod tests {
    use crate::NdArray;
    use crate::array_data::ArrayData;

    #[test]
    fn test_nansum_basic() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0]);
        let s = a.nansum(None, false).unwrap();
        assert_eq!(s.shape(), &[]);
        let ArrayData::Float64(arr) = s.data() else { panic!() };
        assert!((arr[[]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanmean_basic() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0]);
        let m = a.nanmean(None, false).unwrap();
        let ArrayData::Float64(arr) = m.data() else { panic!() };
        assert!((arr[[]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanmean_all_nan() {
        let a = NdArray::from_vec(vec![f64::NAN, f64::NAN]);
        let m = a.nanmean(None, false).unwrap();
        let ArrayData::Float64(arr) = m.data() else { panic!() };
        assert!(arr[[]].is_nan());
    }

    #[test]
    fn test_nanvar_basic() {
        // [1, NaN, 3] -> valid=[1,3], mean=2, var=((1-2)^2+(3-2)^2)/2 = 1.0
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0]);
        let v = a.nanvar(None, 0, false).unwrap();
        let ArrayData::Float64(arr) = v.data() else { panic!() };
        assert!((arr[[]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanstd_basic() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0]);
        let s = a.nanstd(None, 0, false).unwrap();
        let ArrayData::Float64(arr) = s.data() else { panic!() };
        assert!((arr[[]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanmin_basic() {
        let a = NdArray::from_vec(vec![3.0_f64, f64::NAN, 1.0]);
        let m = a.nanmin(None, false).unwrap();
        let ArrayData::Float64(arr) = m.data() else { panic!() };
        assert!((arr[[]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanmax_basic() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0]);
        let m = a.nanmax(None, false).unwrap();
        let ArrayData::Float64(arr) = m.data() else { panic!() };
        assert!((arr[[]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanmin_all_nan() {
        let a = NdArray::from_vec(vec![f64::NAN, f64::NAN]);
        assert!(a.nanmin(None, false).is_err());
    }

    #[test]
    fn test_nanargmin_basic() {
        let a = NdArray::from_vec(vec![3.0_f64, f64::NAN, 1.0]);
        let idx = a.nanargmin(None).unwrap();
        let ArrayData::Int64(arr) = idx.data() else { panic!() };
        assert_eq!(arr[[]], 2);
    }

    #[test]
    fn test_nanargmax_basic() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0]);
        let idx = a.nanargmax(None).unwrap();
        let ArrayData::Int64(arr) = idx.data() else { panic!() };
        assert_eq!(arr[[]], 2);
    }

    #[test]
    fn test_nanprod_basic() {
        let a = NdArray::from_vec(vec![2.0_f64, f64::NAN, 3.0]);
        let p = a.nanprod(None, false).unwrap();
        let ArrayData::Float64(arr) = p.data() else { panic!() };
        assert!((arr[[]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_nansum_axis() {
        // [[1, NaN], [3, 4]] -> axis=1 -> [1, 7]
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0, 4.0])
            .reshape(&[2, 2]).unwrap();
        let s = a.nansum(Some(1), false).unwrap();
        assert_eq!(s.shape(), &[2]);
        let ArrayData::Float64(arr) = s.data() else { panic!() };
        assert!((arr[[0]] - 1.0).abs() < 1e-10);
        assert!((arr[[1]] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_nansum_keepdims() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0, 4.0])
            .reshape(&[2, 2]).unwrap();
        let s = a.nansum(Some(1), true).unwrap();
        assert_eq!(s.shape(), &[2, 1]);
    }
}
```

**Step 2: Register the module**

Add `pub mod nan_reduction;` to `crates/numpy-rust-core/src/ops/mod.rs`.

**Step 3: Run tests**

Run: `cargo test -p numpy-rust-core --all-features -- nan_reduction`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/numpy-rust-core/src/ops/nan_reduction.rs crates/numpy-rust-core/src/ops/mod.rs
git commit -m "feat: add NaN-safe reduction functions (nansum/nanmean/nanstd/nanvar/nanmin/nanmax/nanargmin/nanargmax/nanprod)"
```

---

### Task 2: NaN-safe Reductions — Python Bindings

**Files:**
- Modify: `crates/numpy-rust-python/src/lib.rs` (add 9 `#[pyfunction]` bindings)
- Modify: `python/numpy/__init__.py` (add 9 wrapper functions)

**Step 1: Add Python bindings in `lib.rs`**

Add these inside the `_numpy_native` module, near the existing reduction functions:

```rust
    // --- NaN-safe reductions ---

    #[pyfunction]
    fn nansum(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nansum(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanmean(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanmean(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanstd(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let ddof = ddof.unwrap_or(0);
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanstd(ax, ddof, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanvar(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let ddof = ddof.unwrap_or(0);
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanvar(ax, ddof, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanmin(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanmin(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanmax(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanmax(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanargmin(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .nanargmin(ax)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanargmax(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .nanargmax(ax)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanprod(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanprod(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

**Step 2: Add Python wrappers in `__init__.py`**

Add near the existing `sum`/`mean` functions:

```python
def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nansum(a, axis, keepdims)

def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanmean(a, axis, keepdims)

def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanstd(a, axis, ddof, keepdims)

def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanvar(a, axis, ddof, keepdims)

def nanmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanmin(a, axis, keepdims)

def nanmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanmax(a, axis, keepdims)

def nanargmin(a, axis=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanargmin(a, axis)

def nanargmax(a, axis=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanargmax(a, axis)

def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanprod(a, axis, keepdims)
```

**Step 3: Verify compilation**

Run: `cargo build --workspace --all-features`
Expected: Success.

**Step 4: Commit**

```bash
git add crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py
git commit -m "feat: add Python bindings for NaN-safe reductions"
```

---

### Task 3: NaN-safe Reductions — Python Integration Tests

**Files:**
- Modify: `tests/python/test_numeric.py` (add test section)

**Step 1: Add tests at the end of `test_numeric.py`**

```python
# --- NaN-safe reductions ---

def test_nansum():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nansum(a), 4.0)

def test_nanmean():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nanmean(a), 2.0)

def test_nanstd():
    a = np.array([1.0, float('nan'), 3.0])
    # valid=[1,3], mean=2, var=1, std=1
    assert_close(np.nanstd(a), 1.0)

def test_nanvar():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nanvar(a), 1.0)

def test_nanmin():
    a = np.array([3.0, float('nan'), 1.0])
    assert_close(np.nanmin(a), 1.0)

def test_nanmax():
    a = np.array([1.0, float('nan'), 3.0])
    assert_close(np.nanmax(a), 3.0)

def test_nanargmin():
    a = np.array([3.0, float('nan'), 1.0])
    assert_eq(np.nanargmin(a), 2)

def test_nanargmax():
    a = np.array([1.0, float('nan'), 3.0])
    assert_eq(np.nanargmax(a), 2)

def test_nanprod():
    a = np.array([2.0, float('nan'), 3.0])
    assert_close(np.nanprod(a), 6.0)

def test_nansum_no_nan():
    """NaN-safe functions should work identically when no NaN present."""
    a = np.array([1.0, 2.0, 3.0])
    assert_close(np.nansum(a), 6.0)
    assert_close(np.nanmean(a), 2.0)

def test_nansum_2d_axis():
    a = np.array([1.0, float('nan'), 3.0, 4.0]).reshape((2, 2))
    s = np.nansum(a, axis=1)
    assert_close(float(s[0]), 1.0)
    assert_close(float(s[1]), 7.0)
```

**Step 2: Build and run Python tests**

Run: `cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/python/test_numeric.py
git commit -m "test: add NaN-safe reduction Python integration tests"
```

---

### Task 4: Correlation & Covariance — Core Rust

**Files:**
- Create: `crates/numpy-rust-core/src/ops/correlation.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs` (add `pub mod correlation;`)

**Step 1: Create `correlation.rs`**

```rust
use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Estimate a covariance matrix. Rows = variables, columns = observations (when rowvar=true).
    /// ddof defaults to 1 for cov.
    pub fn cov(&self, rowvar: bool, ddof: usize) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let data = if !rowvar { f.transpose() } else { f };

        // Ensure 2-D: if 1-D, treat as single variable (1 x N)
        let mat = if data.ndim() == 1 {
            data.reshape(&[1, data.size()])?
        } else if data.ndim() == 2 {
            data
        } else {
            return Err(NumpyError::ValueError(
                "cov requires 1-D or 2-D array".into(),
            ));
        };

        let ArrayData::Float64(arr) = &mat.data else { unreachable!() };
        let n_vars = arr.shape()[0];
        let n_obs = arr.shape()[1];

        if n_obs < 1 {
            return Err(NumpyError::ValueError("cov requires at least 1 observation".into()));
        }

        // Compute means for each variable (row)
        let means: Vec<f64> = (0..n_vars)
            .map(|i| {
                let row = arr.row(i);
                row.iter().sum::<f64>() / n_obs as f64
            })
            .collect();

        // Build centered data
        let mut centered = arr.clone();
        for i in 0..n_vars {
            for j in 0..n_obs {
                centered[[i, j]] -= means[i];
            }
        }

        // C = centered @ centered.T / (N - ddof)
        let denom = if n_obs <= ddof { 0.0 } else { (n_obs - ddof) as f64 };
        let mut cov_matrix = ArrayD::<f64>::zeros(IxDyn(&[n_vars, n_vars]));
        for i in 0..n_vars {
            for j in 0..n_vars {
                let mut sum = 0.0;
                for k in 0..n_obs {
                    sum += centered[[i, k]] * centered[[j, k]];
                }
                cov_matrix[[i, j]] = if denom > 0.0 { sum / denom } else { f64::NAN };
            }
        }

        Ok(NdArray::from_data(ArrayData::Float64(cov_matrix)))
    }

    /// Pearson correlation coefficient matrix.
    pub fn corrcoef(&self, rowvar: bool) -> Result<NdArray> {
        let c = self.cov(rowvar, 0)?;
        let ArrayData::Float64(cov_arr) = &c.data else { unreachable!() };
        let n = cov_arr.shape()[0];

        let mut corr = ArrayD::<f64>::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            for j in 0..n {
                let denom = (cov_arr[[i, i]] * cov_arr[[j, j]]).sqrt();
                corr[[i, j]] = if denom > 0.0 {
                    cov_arr[[i, j]] / denom
                } else {
                    f64::NAN
                };
            }
        }

        Ok(NdArray::from_data(ArrayData::Float64(corr)))
    }
}

/// Covariance of two 1-D arrays: stack them and compute cov.
pub fn cov_xy(x: &NdArray, y: &NdArray, ddof: usize) -> Result<NdArray> {
    if x.ndim() != 1 || y.ndim() != 1 {
        return Err(NumpyError::ValueError("cov with two args requires 1-D arrays".into()));
    }
    if x.size() != y.size() {
        return Err(NumpyError::ValueError("cov: x and y must have same length".into()));
    }
    let stacked = crate::concatenate(&[x, y], 0)?;
    let mat = stacked.reshape(&[2, x.size()])?;
    mat.cov(true, ddof)
}

/// Correlation coefficient of two 1-D arrays.
pub fn corrcoef_xy(x: &NdArray, y: &NdArray) -> Result<NdArray> {
    if x.ndim() != 1 || y.ndim() != 1 {
        return Err(NumpyError::ValueError("corrcoef with two args requires 1-D arrays".into()));
    }
    if x.size() != y.size() {
        return Err(NumpyError::ValueError("corrcoef: x and y must have same length".into()));
    }
    let stacked = crate::concatenate(&[x, y], 0)?;
    let mat = stacked.reshape(&[2, x.size()])?;
    mat.corrcoef(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_cov_basic() {
        // Two variables, 3 observations each
        // x = [1, 2, 3], y = [4, 5, 6]
        let data = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3]).unwrap();
        let c = data.cov(true, 1).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let ArrayData::Float64(arr) = c.data() else { panic!() };
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-10); // var(x) = 1
        assert!((arr[[1, 1]] - 1.0).abs() < 1e-10); // var(y) = 1
        assert!((arr[[0, 1]] - 1.0).abs() < 1e-10); // cov(x,y) = 1
    }

    #[test]
    fn test_corrcoef_perfect() {
        // Perfectly correlated
        let data = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 2.0, 4.0, 6.0])
            .reshape(&[2, 3]).unwrap();
        let c = data.corrcoef(true).unwrap();
        let ArrayData::Float64(arr) = c.data() else { panic!() };
        assert!((arr[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cov_xy() {
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = cov_xy(&x, &y, 1).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_corrcoef_xy() {
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let c = corrcoef_xy(&x, &y).unwrap();
        let ArrayData::Float64(arr) = c.data() else { panic!() };
        assert!((arr[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cov_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = a.cov(true, 1).unwrap();
        assert_eq!(c.shape(), &[1, 1]);
        let ArrayData::Float64(arr) = c.data() else { panic!() };
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-10);
    }
}
```

**Step 2: Register the module**

Add `pub mod correlation;` to `crates/numpy-rust-core/src/ops/mod.rs`.

**Step 3: Run tests**

Run: `cargo test -p numpy-rust-core --all-features -- correlation`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/numpy-rust-core/src/ops/correlation.rs crates/numpy-rust-core/src/ops/mod.rs
git commit -m "feat: add cov/corrcoef correlation functions"
```

---

### Task 5: Correlation & Covariance — Python Bindings + Tests

**Files:**
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Python bindings in `lib.rs`**

```rust
    #[pyfunction]
    fn cov(
        m: vm::PyRef<PyNdArray>,
        y: vm::function::OptionalArg<vm::PyRef<PyNdArray>>,
        rowvar: vm::function::OptionalArg<bool>,
        ddof: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let rowvar = rowvar.unwrap_or(true);
        let ddof = ddof.unwrap_or(1);
        match y.into_option() {
            Some(y_arr) => numpy_rust_core::ops::correlation::cov_xy(&m.inner(), &y_arr.inner(), ddof)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string())),
            None => m.inner().cov(rowvar, ddof)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string())),
        }
    }

    #[pyfunction]
    fn corrcoef(
        x: vm::PyRef<PyNdArray>,
        y: vm::function::OptionalArg<vm::PyRef<PyNdArray>>,
        rowvar: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let rowvar = rowvar.unwrap_or(true);
        match y.into_option() {
            Some(y_arr) => numpy_rust_core::ops::correlation::corrcoef_xy(&x.inner(), &y_arr.inner())
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string())),
            None => x.inner().corrcoef(rowvar)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string())),
        }
    }
```

**Step 2: Add Python wrappers in `__init__.py`**

```python
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    if not isinstance(m, ndarray):
        m = array(m)
    _ddof = ddof if ddof is not None else (0 if bias else 1)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.cov(m, y, rowvar, _ddof)
    return _native.cov(m, None, rowvar, _ddof)

def corrcoef(x, y=None, rowvar=True):
    if not isinstance(x, ndarray):
        x = array(x)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.corrcoef(x, y, rowvar)
    return _native.corrcoef(x, None, rowvar)
```

**Step 3: Add integration tests**

```python
def test_cov_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    c = np.cov(x, y)
    assert_eq(c.shape, (2, 2))
    assert_close(float(c[0][0]), 1.0)

def test_corrcoef_perfect():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    c = np.corrcoef(x, y)
    assert_close(float(c[0][1]), 1.0)
```

**Step 4: Build, test, commit**

Run: `cargo build --workspace --all-features && ./tests/python/run_tests.sh target/debug/numpy-python`

```bash
git add crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add Python bindings for cov/corrcoef + integration tests"
```

---

### Task 6: Histogram & Bincount — Core Rust

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/statistics.rs` (add functions)

**Step 1: Add histogram and bincount**

Append to `statistics.rs` before `#[cfg(test)]`:

```rust
impl NdArray {
    /// Compute histogram of 1-D data with uniform bins.
    /// Returns (counts: Int64, bin_edges: Float64).
    pub fn histogram(&self, bins: usize, range: Option<(f64, f64)>) -> Result<(NdArray, NdArray)> {
        if bins == 0 {
            return Err(NumpyError::ValueError("bins must be > 0".into()));
        }
        let f = self.astype(DType::Float64).flatten();
        let ArrayData::Float64(arr) = &f.data else { unreachable!() };

        let (lo, hi) = match range {
            Some((l, h)) => (l, h),
            None => {
                let min_v = arr.iter().copied().reduce(f64::min)
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                let max_v = arr.iter().copied().reduce(f64::max)
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                (min_v, max_v)
            }
        };

        let bin_width = if hi > lo { (hi - lo) / bins as f64 } else { 1.0 };
        let mut counts = vec![0_i64; bins];
        for &v in arr.iter() {
            if v.is_nan() {
                continue;
            }
            let idx = if bin_width > 0.0 {
                ((v - lo) / bin_width).floor() as isize
            } else {
                0
            };
            // Clamp to [0, bins-1]; right edge of last bin is inclusive
            let idx = idx.max(0).min(bins as isize - 1) as usize;
            counts[idx] += 1;
        }

        let edges: Vec<f64> = (0..=bins).map(|i| lo + i as f64 * bin_width).collect();

        let counts_arr = NdArray::from_vec(counts);
        let edges_arr = NdArray::from_vec(edges);
        Ok((counts_arr, edges_arr))
    }

    /// Count occurrences of each non-negative integer value.
    /// If weights is provided, sum weights instead of counting.
    pub fn bincount(&self, weights: Option<&NdArray>, minlength: usize) -> Result<NdArray> {
        let idx = self.astype(DType::Int64).flatten();
        let ArrayData::Int64(idx_arr) = &idx.data else { unreachable!() };

        // Validate non-negative
        for &v in idx_arr.iter() {
            if v < 0 {
                return Err(NumpyError::ValueError(
                    "bincount: input must be non-negative".into(),
                ));
            }
        }

        let max_val = idx_arr.iter().copied().max().unwrap_or(-1);
        let out_len = (max_val as usize + 1).max(minlength);

        match weights {
            Some(w) => {
                let w_f = w.astype(DType::Float64).flatten();
                let ArrayData::Float64(w_arr) = &w_f.data else { unreachable!() };
                if idx_arr.len() != w_arr.len() {
                    return Err(NumpyError::ValueError(
                        "bincount: weights must have same length as input".into(),
                    ));
                }
                let mut result = vec![0.0_f64; out_len];
                for (&i, &w) in idx_arr.iter().zip(w_arr.iter()) {
                    result[i as usize] += w;
                }
                Ok(NdArray::from_vec(result))
            }
            None => {
                let mut result = vec![0_i64; out_len];
                for &i in idx_arr.iter() {
                    result[i as usize] += 1;
                }
                Ok(NdArray::from_vec(result))
            }
        }
    }
}
```

And add tests:

```rust
    #[test]
    fn test_histogram_basic() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let (counts, edges) = a.histogram(5, None).unwrap();
        assert_eq!(counts.shape(), &[5]);
        assert_eq!(edges.shape(), &[6]);
    }

    #[test]
    fn test_histogram_range() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let (counts, _) = a.histogram(2, Some((0.0, 6.0))).unwrap();
        assert_eq!(counts.shape(), &[2]);
    }

    #[test]
    fn test_bincount_basic() {
        let a = NdArray::from_vec(vec![0_i64, 1, 1, 2, 2, 2]);
        let c = a.bincount(None, 0).unwrap();
        let ArrayData::Int64(arr) = c.data() else { panic!() };
        assert_eq!(arr[[0]], 1); // 0 appears once
        assert_eq!(arr[[1]], 2); // 1 appears twice
        assert_eq!(arr[[2]], 3); // 2 appears thrice
    }

    #[test]
    fn test_bincount_minlength() {
        let a = NdArray::from_vec(vec![0_i64, 1]);
        let c = a.bincount(None, 5).unwrap();
        assert_eq!(c.shape(), &[5]);
    }

    #[test]
    fn test_bincount_weights() {
        let a = NdArray::from_vec(vec![0_i64, 1, 1]);
        let w = NdArray::from_vec(vec![0.5_f64, 1.0, 1.5]);
        let c = a.bincount(Some(&w), 0).unwrap();
        let ArrayData::Float64(arr) = c.data() else { panic!() };
        assert!((arr[[0]] - 0.5).abs() < 1e-10);
        assert!((arr[[1]] - 2.5).abs() < 1e-10);
    }
```

**Step 2: Run tests**

Run: `cargo test -p numpy-rust-core --all-features -- histogram bincount`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add crates/numpy-rust-core/src/ops/statistics.rs
git commit -m "feat: add histogram and bincount functions"
```

---

### Task 7: Histogram & Bincount — Python Bindings + Tests

**Files:**
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Python bindings**

```rust
    #[pyfunction]
    fn histogram(
        a: vm::PyRef<PyNdArray>,
        bins: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let bins = bins.unwrap_or(10);
        let (counts, edges) = a.inner()
            .histogram(bins, None)
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_counts = PyNdArray::from_core(counts).into_pyobject(vm);
        let py_edges = PyNdArray::from_core(edges).into_pyobject(vm);
        Ok(vm.ctx.new_tuple(vec![py_counts, py_edges]).into())
    }

    #[pyfunction]
    fn bincount(
        x: vm::PyRef<PyNdArray>,
        weights: vm::function::OptionalArg<vm::PyRef<PyNdArray>>,
        minlength: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let w = weights.as_ref().map(|w| w.inner());
        let w_ref = w.as_deref();
        let minlength = minlength.unwrap_or(0);
        x.inner()
            .bincount(w_ref, minlength)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

**Step 2: Add Python wrappers**

```python
def histogram(a, bins=10, range=None, density=None, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.histogram(a, bins)

def bincount(x, weights=None, minlength=0):
    if not isinstance(x, ndarray):
        x = array(x)
    if weights is not None and not isinstance(weights, ndarray):
        weights = array(weights)
    return _native.bincount(x, weights, minlength)
```

**Step 3: Add integration tests**

```python
def test_histogram():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counts, edges = np.histogram(a, bins=5)
    assert_eq(counts.shape, (5,))
    assert_eq(edges.shape, (6,))

def test_bincount():
    a = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    c = np.bincount(a)
    assert_eq(int(c[0]), 1)
    assert_eq(int(c[1]), 2)
    assert_eq(int(c[2]), 3)
```

**Step 4: Build, test, commit**

```bash
git add crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add Python bindings for histogram/bincount + integration tests"
```

---

### Task 8: Set Operations — Core Rust

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/selection.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs` (add exports)

**Step 1: Add set operations to `selection.rs`**

```rust
use std::collections::HashSet;

/// Return sorted unique values present in both arrays.
pub fn intersect1d(a: &NdArray, b: &NdArray) -> NdArray {
    let a_f = a.astype(DType::Float64).flatten();
    let b_f = b.astype(DType::Float64).flatten();
    let ArrayData::Float64(a_arr) = &a_f.data else { unreachable!() };
    let ArrayData::Float64(b_arr) = &b_f.data else { unreachable!() };

    let set_b: HashSet<u64> = b_arr.iter().map(|v| v.to_bits()).collect();
    let mut result: Vec<f64> = a_arr.iter()
        .copied()
        .filter(|v| set_b.contains(&v.to_bits()))
        .collect();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup();
    NdArray::from_vec(result)
}

/// Return sorted union of unique values from both arrays.
pub fn union1d(a: &NdArray, b: &NdArray) -> NdArray {
    let a_f = a.astype(DType::Float64).flatten();
    let b_f = b.astype(DType::Float64).flatten();
    let ArrayData::Float64(a_arr) = &a_f.data else { unreachable!() };
    let ArrayData::Float64(b_arr) = &b_f.data else { unreachable!() };

    let mut all: Vec<f64> = a_arr.iter().chain(b_arr.iter()).copied().collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all.dedup();
    NdArray::from_vec(all)
}

/// Return sorted values in `a` that are not in `b`.
pub fn setdiff1d(a: &NdArray, b: &NdArray) -> NdArray {
    let a_f = a.astype(DType::Float64).flatten();
    let b_f = b.astype(DType::Float64).flatten();
    let ArrayData::Float64(a_arr) = &a_f.data else { unreachable!() };
    let ArrayData::Float64(b_arr) = &b_f.data else { unreachable!() };

    let set_b: HashSet<u64> = b_arr.iter().map(|v| v.to_bits()).collect();
    let mut result: Vec<f64> = a_arr.iter()
        .copied()
        .filter(|v| !set_b.contains(&v.to_bits()))
        .collect();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup();
    NdArray::from_vec(result)
}

/// Test whether each element of `element` is in `test_elements`.
/// Returns boolean array with same shape as `element`.
pub fn isin(element: &NdArray, test_elements: &NdArray) -> NdArray {
    let elem_f = element.astype(DType::Float64);
    let test_f = test_elements.astype(DType::Float64).flatten();
    let ArrayData::Float64(test_arr) = &test_f.data else { unreachable!() };
    let ArrayData::Float64(elem_arr) = &elem_f.data else { unreachable!() };

    let set: HashSet<u64> = test_arr.iter().map(|v| v.to_bits()).collect();
    let result: Vec<bool> = elem_arr.iter()
        .map(|v| set.contains(&v.to_bits()))
        .collect();

    let shape = element.shape().to_vec();
    NdArray::from_data(ArrayData::Bool(
        ArrayD::from_shape_vec(IxDyn(&shape), result).expect("shape matches"),
    ))
}
```

Add tests:

```rust
    #[test]
    fn test_intersect1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let r = intersect1d(&a, &b);
        assert_eq!(r.shape(), &[2]);
        let ArrayData::Float64(arr) = r.data() else { panic!() };
        assert_eq!(arr[[0]], 2.0);
        assert_eq!(arr[[1]], 4.0);
    }

    #[test]
    fn test_union1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 3.0]);
        let r = union1d(&a, &b);
        assert_eq!(r.shape(), &[3]);
    }

    #[test]
    fn test_setdiff1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![2.0_f64]);
        let r = setdiff1d(&a, &b);
        assert_eq!(r.shape(), &[2]);
    }

    #[test]
    fn test_isin() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let test = NdArray::from_vec(vec![2.0_f64, 4.0]);
        let r = isin(&a, &test);
        assert_eq!(r.shape(), &[4]);
        let ArrayData::Bool(arr) = r.data() else { panic!() };
        assert!(!arr[[0]]);
        assert!(arr[[1]]);
        assert!(!arr[[2]]);
        assert!(arr[[3]]);
    }
```

**Step 2: Export from `lib.rs`**

Add to the existing `pub use ops::selection::choose;` line in `crates/numpy-rust-core/src/lib.rs`:

```rust
pub use ops::selection::{choose, intersect1d, isin, setdiff1d, union1d};
```

**Step 3: Run tests**

Run: `cargo test -p numpy-rust-core --all-features -- intersect1d union1d setdiff1d isin`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/numpy-rust-core/src/ops/selection.rs crates/numpy-rust-core/src/lib.rs
git commit -m "feat: add set operations (intersect1d/union1d/setdiff1d/isin)"
```

---

### Task 9: Set Operations — Python Bindings + Tests

**Files:**
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/__init__.py`
- Modify: `tests/python/test_numeric.py`

**Step 1: Add Python bindings**

```rust
    #[pyfunction]
    fn intersect1d(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::intersect1d(&a.inner(), &b.inner()))
    }

    #[pyfunction]
    fn union1d(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::union1d(&a.inner(), &b.inner()))
    }

    #[pyfunction]
    fn setdiff1d(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::setdiff1d(&a.inner(), &b.inner()))
    }

    #[pyfunction]
    fn isin(
        element: vm::PyRef<PyNdArray>,
        test_elements: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::isin(&element.inner(), &test_elements.inner()))
    }
```

**Step 2: Add Python wrappers**

```python
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.intersect1d(ar1, ar2)

def union1d(ar1, ar2):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.union1d(ar1, ar2)

def setdiff1d(ar1, ar2, assume_unique=False):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.setdiff1d(ar1, ar2)

def isin(element, test_elements, assume_unique=False, invert=False):
    if not isinstance(element, ndarray):
        element = array(element)
    if not isinstance(test_elements, ndarray):
        test_elements = array(test_elements)
    result = _native.isin(element, test_elements)
    if invert:
        return logical_not(result)
    return result

# Alias
in1d = isin
```

**Step 3: Add integration tests**

```python
def test_intersect1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0])
    r = np.intersect1d(a, b)
    assert_eq(r.shape, (2,))
    assert_close(float(r[0]), 2.0)
    assert_close(float(r[1]), 4.0)

def test_union1d():
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 3.0])
    r = np.union1d(a, b)
    assert_eq(r.shape, (3,))

def test_setdiff1d():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0])
    r = np.setdiff1d(a, b)
    assert_eq(r.shape, (2,))

def test_isin():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    test = np.array([2.0, 4.0])
    r = np.isin(a, test)
    assert_eq(r.shape, (4,))
```

**Step 4: Build, test, commit**

```bash
git add crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add Python bindings for set operations + integration tests"
```

---

### Task 10: Stacking Fixes — Fix `stack` + Add `column_stack`, `dstack`

**Files:**
- Modify: `crates/numpy-rust-core/src/manipulation.rs` (add `column_stack`, `dstack`)
- Modify: `crates/numpy-rust-core/src/lib.rs` (export new functions)
- Modify: `python/numpy/__init__.py` (fix `stack`, add `column_stack`, `dstack`)
- Modify: `crates/numpy-rust-python/src/lib.rs` (add bindings)

**Step 1: Add `column_stack` and `dstack` to `manipulation.rs`**

Add after the existing `hstack` function:

```rust
/// Stack 1-D arrays as columns into a 2-D array.
/// If arrays are 2-D, concatenate along axis 1.
pub fn column_stack(arrays: &[&NdArray]) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to column_stack".into(),
        ));
    }
    // If all 1-D, reshape each to (N, 1) and concatenate along axis 1
    if arrays.iter().all(|a| a.ndim() == 1) {
        let expanded: Vec<NdArray> = arrays
            .iter()
            .map(|a| a.reshape(&[a.size(), 1]).expect("reshape to column"))
            .collect();
        let refs: Vec<&NdArray> = expanded.iter().collect();
        concatenate(&refs, 1)
    } else {
        concatenate(arrays, 1)
    }
}

/// Stack arrays along the third axis (depth).
pub fn dstack(arrays: &[&NdArray]) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to dstack".into(),
        ));
    }
    // Expand each array to at least 3-D
    let expanded: Vec<NdArray> = arrays
        .iter()
        .map(|a| {
            if a.ndim() == 1 {
                a.reshape(&[1, a.size(), 1]).expect("reshape for dstack")
            } else if a.ndim() == 2 {
                let s = a.shape();
                a.reshape(&[s[0], s[1], 1]).expect("reshape for dstack")
            } else {
                (*a).clone()
            }
        })
        .collect();
    let refs: Vec<&NdArray> = expanded.iter().collect();
    concatenate(&refs, 2)
}
```

**Step 2: Export from `lib.rs`**

Update the `pub use manipulation::` line:

```rust
pub use manipulation::{
    column_stack, concatenate, dstack, hsplit, hstack, split, stack, unique, vsplit, vstack,
    SplitSpec,
};
```

**Step 3: Add Python bindings in `lib.rs`**

```rust
    #[pyfunction]
    fn column_stack(
        arrays: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let list = arrays
            .downcast_ref::<vm::builtins::PyList>()
            .or_else(|| arrays.downcast_ref::<vm::builtins::PyTuple>().map(|_| todo!()))
            .ok_or_else(|| vm.new_type_error("column_stack requires list or tuple".to_owned()));
        // Re-use the concatenate helper pattern
        py_creation::py_column_stack(arrays, vm)
    }

    #[pyfunction]
    fn dstack(
        arrays: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        py_creation::py_dstack(arrays, vm)
    }
```

Actually, the simpler approach is to add helper functions in `py_creation.rs` that follow the `py_concatenate` pattern. But since the core functions exist, the simplest is to parse the list in `lib.rs` directly. Let me simplify:

```rust
    #[pyfunction]
    fn column_stack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr_list = extract_array_list(&arrays, vm)?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            arr_list.iter().map(|a| a.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        numpy_rust_core::column_stack(&refs)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn dstack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr_list = extract_array_list(&arrays, vm)?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            arr_list.iter().map(|a| a.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        numpy_rust_core::dstack(&refs)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn stack_native(arrays: PyObjectRef, axis: vm::function::OptionalArg<usize>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let ax = axis.unwrap_or(0);
        let arr_list = extract_array_list(&arrays, vm)?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            arr_list.iter().map(|a| a.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        numpy_rust_core::stack(&refs, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

You'll need a helper to extract arrays from a list/tuple. Add this as a free function in `lib.rs`:

```rust
    fn extract_array_list(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<vm::PyRef<PyNdArray>>> {
        if let Some(list) = obj.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            items.iter()
                .map(|item| item.clone().try_into_value::<vm::PyRef<PyNdArray>>(vm))
                .collect::<PyResult<Vec<_>>>()
        } else if let Some(tuple) = obj.downcast_ref::<vm::builtins::PyTuple>() {
            tuple.as_slice().iter()
                .map(|item| item.clone().try_into_value::<vm::PyRef<PyNdArray>>(vm))
                .collect::<PyResult<Vec<_>>>()
        } else {
            Err(vm.new_type_error("expected list or tuple of arrays".to_owned()))
        }
    }
```

**Step 4: Fix Python wrappers in `__init__.py`**

Replace the existing `stack` and add `column_stack`, `dstack`:

```python
def stack(arrays, axis=0, out=None):
    return _native.stack_native(arrays, axis)

def column_stack(tup):
    return _native.column_stack(tup)

def dstack(tup):
    return _native.dstack(tup)
```

**Step 5: Add tests and run**

```python
def test_stack_proper():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = np.stack([a, b])
    assert_eq(r.shape, (2, 3))

def test_column_stack():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = np.column_stack([a, b])
    assert_eq(r.shape, (3, 2))

def test_dstack():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = np.dstack([a, b])
    assert_eq(r.shape, (1, 3, 2))
```

**Step 6: Build, test, commit**

```bash
git add crates/numpy-rust-core/src/manipulation.rs crates/numpy-rust-core/src/lib.rs \
       crates/numpy-rust-python/src/lib.rs python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: fix stack + add column_stack/dstack"
```

---

### Task 11: Index Utilities — `unravel_index` and `ravel_multi_index`

**Files:**
- Modify: `crates/numpy-rust-core/src/indexing.rs` (add functions)
- Modify: `crates/numpy-rust-python/src/lib.rs` (add bindings)
- Modify: `python/numpy/__init__.py` (add wrappers)
- Modify: `tests/python/test_numeric.py` (add tests)

**Step 1: Add core implementations to `indexing.rs`**

```rust
/// Convert flat index to multi-dimensional index (C-order).
/// Returns a Vec of arrays (one per dimension).
pub fn unravel_index(indices: &NdArray, shape: &[usize]) -> Result<Vec<NdArray>> {
    let idx = indices.astype(DType::Int64).flatten();
    let ArrayData::Int64(idx_arr) = &idx.data else { unreachable!() };

    let ndim = shape.len();
    let total_size: usize = shape.iter().product();

    // Compute strides (C-order)
    let mut strides = vec![1_usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut result_vecs: Vec<Vec<i64>> = vec![Vec::with_capacity(idx_arr.len()); ndim];

    for &flat in idx_arr.iter() {
        if flat < 0 || flat as usize >= total_size {
            return Err(NumpyError::ValueError(format!(
                "index {} is out of bounds for array with size {}",
                flat, total_size
            )));
        }
        let mut remaining = flat as usize;
        for d in 0..ndim {
            result_vecs[d].push((remaining / strides[d]) as i64);
            remaining %= strides[d];
        }
    }

    Ok(result_vecs.into_iter().map(NdArray::from_vec).collect())
}

/// Convert multi-dimensional index to flat index (C-order).
pub fn ravel_multi_index(multi_index: &[&NdArray], dims: &[usize]) -> Result<NdArray> {
    let ndim = dims.len();
    if multi_index.len() != ndim {
        return Err(NumpyError::ValueError(format!(
            "ravel_multi_index: {} index arrays for {} dimensions",
            multi_index.len(), ndim
        )));
    }

    let mut strides = vec![1_usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    let size = multi_index[0].size();
    let arrays: Vec<NdArray> = multi_index.iter().map(|a| a.astype(DType::Int64).flatten()).collect();
    let int_arrs: Vec<&ArrayData> = arrays.iter().map(|a| a.data()).collect();

    let mut result = vec![0_i64; size];
    for d in 0..ndim {
        let ArrayData::Int64(arr) = int_arrs[d] else { unreachable!() };
        for (i, &v) in arr.iter().enumerate() {
            result[i] += v * strides[d] as i64;
        }
    }

    Ok(NdArray::from_vec(result))
}
```

Add tests:

```rust
    #[test]
    fn test_unravel_index() {
        // flat index 5 in shape (3, 4) -> (1, 1)
        let idx = NdArray::from_vec(vec![5_i64]);
        let result = unravel_index(&idx, &[3, 4]).unwrap();
        assert_eq!(result.len(), 2);
        let ArrayData::Int64(r0) = result[0].data() else { panic!() };
        let ArrayData::Int64(r1) = result[1].data() else { panic!() };
        assert_eq!(r0[[0]], 1);
        assert_eq!(r1[[0]], 1);
    }

    #[test]
    fn test_ravel_multi_index() {
        let i0 = NdArray::from_vec(vec![1_i64]);
        let i1 = NdArray::from_vec(vec![1_i64]);
        let result = ravel_multi_index(&[&i0, &i1], &[3, 4]).unwrap();
        let ArrayData::Int64(arr) = result.data() else { panic!() };
        assert_eq!(arr[[0]], 5);
    }
```

**Step 2: Add Python bindings and wrappers**

Bindings in `lib.rs`:

```rust
    #[pyfunction]
    fn unravel_index(
        indices: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let idx_arr = obj_to_ndarray(&indices, vm)?;
        let shape_vec = py_creation::parse_shape(&shape, vm)?;
        let result = numpy_rust_core::indexing::unravel_index(&idx_arr, &shape_vec)
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_arrays: Vec<PyObjectRef> = result
            .into_iter()
            .map(|a| py_array::ndarray_or_scalar(a, vm))
            .collect();
        Ok(vm.ctx.new_tuple(py_arrays).into())
    }

    #[pyfunction]
    fn ravel_multi_index(
        multi_index: PyObjectRef,
        dims: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let list = multi_index.downcast_ref::<vm::builtins::PyTuple>()
            .ok_or_else(|| vm.new_type_error("multi_index must be a tuple".to_owned()))?;
        let idx_arrays: Vec<vm::PyRef<PyNdArray>> = list.as_slice().iter()
            .map(|item| obj_to_ndarray(item, vm).map(|a| {
                let py = PyNdArray::from_core(a);
                vm::PyRef::new_ref(py, PyNdArray::make_class(&vm.ctx), None)
            }))
            .collect::<PyResult<Vec<_>>>()?;
        // This is complex — simpler approach: parse to NdArrays directly
        let arrs: Vec<numpy_rust_core::NdArray> = list.as_slice().iter()
            .map(|item| obj_to_ndarray(item, vm))
            .collect::<PyResult<Vec<_>>>()?;
        let refs: Vec<&numpy_rust_core::NdArray> = arrs.iter().collect();
        let dims_vec = py_creation::parse_shape(&dims, vm)?;
        numpy_rust_core::indexing::ravel_multi_index(&refs, &dims_vec)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

Python wrappers:

```python
def unravel_index(indices, shape, order='C'):
    if not isinstance(indices, ndarray):
        if isinstance(indices, int):
            indices = array([indices])
        else:
            indices = array(indices)
    return _native.unravel_index(indices, shape)

def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    arrays = tuple(array(a) if not isinstance(a, ndarray) else a for a in multi_index)
    return _native.ravel_multi_index(arrays, dims)
```

**Step 3: Add tests**

```python
def test_unravel_index():
    idx = np.unravel_index(5, (3, 4))
    assert_eq(int(idx[0]), 1)
    assert_eq(int(idx[1]), 1)

def test_ravel_multi_index():
    idx = np.ravel_multi_index((np.array([1.0]), np.array([1.0])), (3, 4))
    assert_eq(int(idx), 5)
```

**Step 4: Build, test, commit**

```bash
git add crates/numpy-rust-core/src/indexing.rs crates/numpy-rust-python/src/lib.rs \
       python/numpy/__init__.py tests/python/test_numeric.py
git commit -m "feat: add unravel_index/ravel_multi_index index utilities"
```

---

### Task 12: Final Verification

**Step 1: Run full Rust test suite**

Run: `cargo test --workspace --all-features`
Expected: All tests pass.

**Step 2: Run lints**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-features -- -D warnings`
Expected: Clean.

**Step 3: Run full Python integration tests**

Run: `cargo build -p numpy-rust-wasm && ./tests/python/run_tests.sh target/debug/numpy-python`
Expected: All tests pass.

**Step 4: Final commit (if any fixups needed)**

```bash
git add -A && git commit -m "fix: tier 8 cleanup and fixups"
```
