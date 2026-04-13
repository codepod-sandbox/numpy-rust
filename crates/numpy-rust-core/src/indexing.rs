use ndarray::{IxDyn, SliceInfoElem};
use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::error::{NumpyError, Result};
use crate::resolver::resolve_assignment_cast;
use crate::storage::ArrayStorage;
use crate::storage::{array_data_to_scalar, normalize_string_assignment, scalar_to_array_data};
use crate::DType;
use crate::NdArray;

/// Describes how to slice one axis.
#[derive(Debug, Clone)]
pub enum SliceArg {
    /// Select a single index (removes the dimension).
    Index(isize),
    /// Slice with start, stop, step (all optional, like Python's a[start:stop:step]).
    Range {
        start: Option<isize>,
        stop: Option<isize>,
        step: isize,
    },
    /// Select all elements along this axis (equivalent to `:`).
    Full,
}

/// A scalar value extracted from an array.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Complex64(Complex<f32>),
    Complex128(Complex<f64>),
    Str(String),
}

/// A scalar expressed in the array's logical dtype rather than its storage dtype.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalScalar {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Complex(f64, f64),
    Str(String),
}

impl Scalar {
    fn dtype(&self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int32(_) => DType::Int32,
            Scalar::Int64(_) => DType::Int64,
            Scalar::Float32(_) => DType::Float32,
            Scalar::Float64(_) => DType::Float64,
            Scalar::Complex64(_) => DType::Complex64,
            Scalar::Complex128(_) => DType::Complex128,
            Scalar::Str(_) => DType::Str,
        }
    }
}

fn logical_scalar_from_storage(value: Scalar, dtype: DType) -> LogicalScalar {
    match dtype {
        DType::Bool => match value {
            Scalar::Bool(v) => LogicalScalar::Bool(v),
            other => unreachable!("bool dtype must yield bool scalar, got {other:?}"),
        },
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => match value {
            Scalar::Int32(v) => LogicalScalar::Int(v as i64),
            Scalar::Int64(v) => LogicalScalar::Int(v),
            other => unreachable!("signed integer dtype must yield int scalar, got {other:?}"),
        },
        DType::UInt8 | DType::UInt16 => match value {
            Scalar::Int32(v) => LogicalScalar::UInt(v as u32 as u64),
            other => unreachable!("small unsigned dtype must yield Int32 storage, got {other:?}"),
        },
        DType::UInt32 | DType::UInt64 => match value {
            Scalar::Int64(v) => LogicalScalar::UInt(v as u64),
            other => unreachable!("wide unsigned dtype must yield Int64 storage, got {other:?}"),
        },
        DType::Float16 | DType::Float32 => match value {
            Scalar::Float32(v) => LogicalScalar::Float(v as f64),
            other => unreachable!("float32-backed dtype must yield Float32 storage, got {other:?}"),
        },
        DType::Float64 => match value {
            Scalar::Float64(v) => LogicalScalar::Float(v),
            other => unreachable!("float64 dtype must yield Float64 storage, got {other:?}"),
        },
        DType::Complex64 => match value {
            Scalar::Complex64(v) => LogicalScalar::Complex(v.re as f64, v.im as f64),
            other => unreachable!("complex64 dtype must yield Complex64 storage, got {other:?}"),
        },
        DType::Complex128 => match value {
            Scalar::Complex128(v) => LogicalScalar::Complex(v.re, v.im),
            other => {
                unreachable!("complex128 dtype must yield Complex128 storage, got {other:?}")
            }
        },
        DType::Str => match value {
            Scalar::Str(v) => LogicalScalar::Str(v),
            other => unreachable!("string dtype must yield string scalar, got {other:?}"),
        },
        DType::Object | DType::Datetime64 | DType::Timedelta64 => {
            unreachable!("boxed dtypes use boxed scalar access, not numeric logical scalars")
        }
    }
}

fn coerce_scalar_for_assignment(
    value: Scalar,
    target: &ArrayStorage,
    target_dtype: DType,
) -> Result<Scalar> {
    let plan = resolve_assignment_cast(value.dtype(), target_dtype)?;
    let mut data = crate::casting::cast_array_data(
        scalar_to_array_data(&value),
        plan.execution_storage_dtype(),
    );
    if plan.requires_narrowing() {
        data = crate::casting::narrow_truncate(data, target_dtype);
    }
    let data = normalize_string_assignment(target, data);
    Ok(array_data_to_scalar(&data))
}

fn coerce_array_for_assignment(
    values: &NdArray,
    target: &ArrayStorage,
    target_dtype: DType,
) -> Result<ArrayData> {
    resolve_assignment_cast(values.dtype(), target_dtype)?;
    let data = if values.dtype() == target_dtype {
        values.data().clone()
    } else {
        values.astype(target_dtype).data().clone()
    };
    Ok(normalize_string_assignment(target, data))
}

impl NdArray {
    /// Get a single element by multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Result<Scalar> {
        let idx = IxDyn(index);
        match self.data() {
            ArrayData::Bool(a) => a
                .get(idx)
                .map(|&v| Scalar::Bool(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Int32(a) => a
                .get(idx)
                .map(|&v| Scalar::Int32(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Int64(a) => a
                .get(idx)
                .map(|&v| Scalar::Int64(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Float32(a) => a
                .get(idx)
                .map(|&v| Scalar::Float32(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Float64(a) => a
                .get(idx)
                .map(|&v| Scalar::Float64(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Complex64(a) => a
                .get(idx)
                .map(|&v| Scalar::Complex64(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Complex128(a) => a
                .get(idx)
                .map(|&v| Scalar::Complex128(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Str(a) => a
                .get(idx)
                .map(|v| Scalar::Str(v.clone()))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
        }
    }

    pub fn get_logical(&self, index: &[usize]) -> Result<LogicalScalar> {
        self.get(index)
            .map(|value| logical_scalar_from_storage(value, self.dtype()))
    }

    /// Slice the array using SliceArg descriptors for each axis.
    /// Axes not specified are treated as Full.
    pub fn slice(&self, args: &[SliceArg]) -> Result<NdArray> {
        let ndim = self.ndim();
        let shape = self.shape();

        // Validate that we don't have more slice args than dimensions
        if args.len() > ndim {
            return Err(NumpyError::ValueError(format!(
                "too many indices for array: array is {}-dimensional, but {} were indexed",
                ndim,
                args.len()
            )));
        }

        // Build ndarray SliceInfoElem for each axis
        let mut slice_elems: Vec<SliceInfoElem> = Vec::with_capacity(ndim);
        let mut remove_axes: Vec<usize> = Vec::new();

        for i in 0..ndim {
            let arg = if i < args.len() {
                &args[i]
            } else {
                &SliceArg::Full
            };
            match arg {
                SliceArg::Index(idx) => {
                    let resolved = resolve_index(*idx, shape[i])?;
                    slice_elems.push(SliceInfoElem::Index(resolved as isize));
                    remove_axes.push(i);
                }
                SliceArg::Range { start, stop, step } => {
                    if *step == 0 {
                        return Err(NumpyError::ValueError("slice step cannot be zero".into()));
                    }
                    let n = shape[i] as isize;
                    // Clamp start and stop to valid range following Python slice semantics
                    let clamp = |v: isize, default: isize| -> isize {
                        let v = if v < 0 { (v + n).max(0) } else { v.min(n) };
                        let _ = default; // only used for None case
                        v
                    };
                    let s = start.map(|v| clamp(v, 0)).unwrap_or(0);
                    let e = stop.map(|v| clamp(v, n)).unwrap_or(n);
                    slice_elems.push(SliceInfoElem::Slice {
                        start: s,
                        end: Some(e),
                        step: *step,
                    });
                }
                SliceArg::Full => {
                    slice_elems.push(SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    });
                }
            }
        }

        let info: Vec<SliceInfoElem> = slice_elems;
        let storage = self.storage().slice_view(info.as_slice())?;
        Ok(NdArray::from_parts(storage, self.descriptor()))
    }

    /// Select elements along an axis by integer indices.
    pub fn index_select(&self, axis: usize, indices: &[usize]) -> Result<NdArray> {
        if axis >= self.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        // Validate indices before using them
        let axis_len = self.shape()[axis];
        for &i in indices {
            if i >= axis_len {
                return Err(NumpyError::ValueError(format!(
                    "index {} is out of bounds for axis {} with size {}",
                    i, axis, axis_len
                )));
            }
        }

        let data = self.data().index_select(axis, indices)?;
        Ok(NdArray::from_parts(
            crate::storage::ArrayStorage::from_array_data_with_string_width(
                data,
                self.storage().string_width(),
            ),
            self.descriptor(),
        ))
    }

    /// Set elements along an axis by integer indices from a values array.
    /// Like `a[[0, 2]] = values` in NumPy.
    pub fn index_set(&mut self, axis: usize, indices: &[usize], values: &NdArray) -> Result<()> {
        if axis >= self.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let cast_values = coerce_array_for_assignment(values, self.storage(), self.dtype())?;
        self.storage_mut()
            .assign_indexed_values(axis, indices, &cast_values)?;
        self.refresh_runtime_state();
        Ok(())
    }

    /// Set a single element by multi-dimensional index.
    pub fn set(&mut self, index: &[usize], value: Scalar) -> Result<()> {
        let value = coerce_scalar_for_assignment(value, self.storage(), self.dtype())?;
        self.storage_mut().assign_element(index, value)?;
        self.refresh_runtime_state();
        Ok(())
    }

    /// Set a slice of the array from values in another array.
    pub fn set_slice(&mut self, args: &[SliceArg], values: &NdArray) -> Result<()> {
        let ndim = self.ndim();
        let shape = self.shape().to_vec();

        // Build ndarray SliceInfoElem for each axis
        let mut slice_elems: Vec<SliceInfoElem> = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let arg = if i < args.len() {
                &args[i]
            } else {
                &SliceArg::Full
            };
            match arg {
                SliceArg::Index(idx) => {
                    let resolved = resolve_index(*idx, shape[i])?;
                    slice_elems.push(SliceInfoElem::Index(resolved as isize));
                }
                SliceArg::Range { start, stop, step } => {
                    let s = start.unwrap_or(0);
                    let e = stop.unwrap_or(shape[i] as isize);
                    slice_elems.push(SliceInfoElem::Slice {
                        start: s,
                        end: Some(e),
                        step: *step,
                    });
                }
                SliceArg::Full => {
                    slice_elems.push(SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    });
                }
            }
        }

        let info: Vec<SliceInfoElem> = slice_elems;
        let cast_values = coerce_array_for_assignment(values, self.storage(), self.dtype())?;
        self.storage_mut()
            .assign_slice_values(info.as_slice(), &cast_values)?;
        self.refresh_runtime_state();
        Ok(())
    }

    /// Set elements where mask is true to the corresponding values.
    pub fn mask_set(&mut self, mask: &NdArray, values: &NdArray) -> Result<()> {
        let bool_mask = match mask.data() {
            ArrayData::Bool(m) => m,
            _ => return Err(NumpyError::TypeError("mask must be a boolean array".into())),
        };

        if bool_mask.len() != self.size() {
            return Err(NumpyError::ShapeMismatch(format!(
                "mask size {} does not match array size {}",
                bool_mask.len(),
                self.size()
            )));
        }

        let flat_mask: Vec<bool> = bool_mask.iter().copied().collect();
        let true_count: usize = flat_mask.iter().filter(|&&m| m).count();

        if values.size() != true_count && values.size() != 1 {
            return Err(NumpyError::ShapeMismatch(format!(
                "values size {} does not match number of True entries {}",
                values.size(),
                true_count
            )));
        }

        let cast_values = coerce_array_for_assignment(values, self.storage(), self.dtype())?;
        self.storage_mut()
            .assign_masked_values(&flat_mask, &cast_values)?;
        self.refresh_runtime_state();
        Ok(())
    }

    /// Select elements where mask is true, returning a 1-D array.
    /// Like `a[mask]` in NumPy where mask is a boolean array.
    pub fn mask_select(&self, mask: &NdArray) -> Result<NdArray> {
        // Mask must be Bool dtype
        let bool_mask = match mask.data() {
            ArrayData::Bool(m) => m,
            _ => return Err(NumpyError::TypeError("mask must be a boolean array".into())),
        };

        // Flatten both to 1-D for element-wise selection
        if bool_mask.len() != self.size() {
            return Err(NumpyError::ShapeMismatch(format!(
                "mask size {} does not match array size {}",
                bool_mask.len(),
                self.size()
            )));
        }

        let flat_mask: Vec<bool> = bool_mask.iter().copied().collect();

        let data = self.data().select_masked(&flat_mask);
        Ok(NdArray::from_parts(
            crate::storage::ArrayStorage::from_array_data_with_string_width(
                data,
                self.storage().string_width(),
            ),
            self.descriptor(),
        ))
    }
}

/// Convert flat indices to multi-dimensional indices (C-order).
///
/// Given an array of flat indices and a shape, return a Vec of NdArray (one Int64 array
/// per dimension) representing the multi-dimensional indices.
pub fn unravel_index(indices: &NdArray, shape: &[usize]) -> Result<Vec<NdArray>> {
    // Cast to Int64 and flatten
    let idx_arr = indices.astype(crate::DType::Int64).flatten();
    let flat_indices = match idx_arr.data() {
        ArrayData::Int64(a) => a.as_slice().unwrap().to_vec(),
        _ => unreachable!("astype(Int64) must produce Int64"),
    };

    // Compute total size
    let total_size: usize = shape.iter().product();
    if total_size == 0 {
        return Err(NumpyError::ValueError(
            "unravel_index: cannot unravel into zero-size shape".into(),
        ));
    }

    // Compute C-order strides: strides[i] = product(shape[i+1:])
    let ndim = shape.len();
    let mut strides = vec![1_usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let n = flat_indices.len();
    // Prepare result vectors, one per dimension
    let mut result_vecs: Vec<Vec<i64>> = vec![Vec::with_capacity(n); ndim];

    for &flat_idx in &flat_indices {
        if flat_idx < 0 || (flat_idx as usize) >= total_size {
            return Err(NumpyError::ValueError(format!(
                "index {} is out of bounds for array with size {}",
                flat_idx, total_size
            )));
        }
        let mut remainder = flat_idx as usize;
        for d in 0..ndim {
            result_vecs[d].push((remainder / strides[d]) as i64);
            remainder %= strides[d];
        }
    }

    // Convert to NdArray
    let result: Vec<NdArray> = result_vecs.into_iter().map(NdArray::from_vec).collect();

    Ok(result)
}

/// Convert multi-dimensional indices to flat indices (C-order).
///
/// Given a slice of index arrays (one per dimension) and the dimensions,
/// compute the flat index for each set of multi-dimensional indices.
pub fn ravel_multi_index(multi_index: &[&NdArray], dims: &[usize]) -> Result<NdArray> {
    if multi_index.len() != dims.len() {
        return Err(NumpyError::ValueError(format!(
            "ravel_multi_index: number of index arrays ({}) must match number of dims ({})",
            multi_index.len(),
            dims.len()
        )));
    }

    let ndim = dims.len();

    // Cast each to Int64 and flatten
    let index_vecs: Vec<Vec<i64>> = multi_index
        .iter()
        .map(|arr| {
            let cast = arr.astype(crate::DType::Int64).flatten();
            match cast.data() {
                ArrayData::Int64(a) => a.as_slice().unwrap().to_vec(),
                _ => unreachable!("astype(Int64) must produce Int64"),
            }
        })
        .collect();

    // All must have the same length
    let n = index_vecs[0].len();
    for (d, v) in index_vecs.iter().enumerate() {
        if v.len() != n {
            return Err(NumpyError::ValueError(format!(
                "ravel_multi_index: all index arrays must have the same length (dim {} has length {}, expected {})",
                d, v.len(), n
            )));
        }
    }

    // Compute C-order strides
    let mut strides = vec![1_usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Compute flat indices
    let flat: Vec<i64> = (0..n)
        .map(|i| {
            strides
                .iter()
                .enumerate()
                .map(|(d, stride)| index_vecs[d][i] * *stride as i64)
                .sum()
        })
        .collect();

    Ok(NdArray::from_vec(flat))
}

/// Resolve a possibly-negative index to a positive one.
pub(crate) fn resolve_index(idx: isize, dim_size: usize) -> Result<usize> {
    let resolved = if idx < 0 {
        dim_size as isize + idx
    } else {
        idx
    };
    if resolved < 0 || resolved as usize >= dim_size {
        return Err(NumpyError::ValueError(format!(
            "index {idx} is out of bounds for axis with size {dim_size}"
        )));
    }
    Ok(resolved as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_get_1d() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0]);
        assert_eq!(a.get(&[0]).unwrap(), Scalar::Float64(10.0));
        assert_eq!(a.get(&[2]).unwrap(), Scalar::Float64(30.0));
    }

    #[test]
    fn test_get_2d() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3, 4, 5, 6])
            .reshape(&[2, 3])
            .unwrap();
        assert_eq!(a.get(&[0, 0]).unwrap(), Scalar::Int32(1));
        assert_eq!(a.get(&[1, 2]).unwrap(), Scalar::Int32(6));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.get(&[5]).is_err());
    }

    #[test]
    fn test_slice_full() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let b = a.slice(&[SliceArg::Full, SliceArg::Full]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_slice_range() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let b = a
            .slice(&[SliceArg::Range {
                start: Some(1),
                stop: Some(4),
                step: 1,
            }])
            .unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_slice_step() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = a
            .slice(&[SliceArg::Range {
                start: Some(0),
                stop: Some(6),
                step: 2,
            }])
            .unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_slice_index_reduces_dim() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = a.slice(&[SliceArg::Index(0), SliceArg::Full]).unwrap();
        assert_eq!(b.shape(), &[4]);
    }

    #[test]
    fn test_slice_negative_index() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = a.slice(&[SliceArg::Index(-1), SliceArg::Full]).unwrap();
        assert_eq!(b.shape(), &[4]);
    }

    #[test]
    fn test_index_select() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
        let b = a.index_select(0, &[0, 2, 4]).unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_index_select_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[3, 2])
            .unwrap();
        let b = a.index_select(0, &[0, 2]).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_index_select_invalid_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.index_select(5, &[0]).is_err());
    }

    #[test]
    fn test_mask_select() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
        let mask = NdArray::from_vec(vec![true, false, true, false, true]);
        let b = a.mask_select(&mask).unwrap();
        assert_eq!(b.shape(), &[3]);
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_mask_select_all_false() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let mask = NdArray::from_vec(vec![false, false, false]);
        let b = a.mask_select(&mask).unwrap();
        assert_eq!(b.shape(), &[0]);
    }

    #[test]
    fn test_mask_select_all_true() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let mask = NdArray::from_vec(vec![true, true, true]);
        let b = a.mask_select(&mask).unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_mask_select_size_mismatch() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let mask = NdArray::from_vec(vec![true, false]);
        assert!(a.mask_select(&mask).is_err());
    }

    #[test]
    fn test_mask_select_non_bool_mask() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let mask = NdArray::from_vec(vec![1_i32, 0]);
        assert!(a.mask_select(&mask).is_err());
    }

    #[test]
    fn test_set_1d() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        a.set(&[0], Scalar::Float64(99.0)).unwrap();
        assert_eq!(a.get(&[0]).unwrap(), Scalar::Float64(99.0));
        assert_eq!(a.get(&[1]).unwrap(), Scalar::Float64(2.0));
    }

    #[test]
    fn test_set_2d() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        a.set(&[0, 1], Scalar::Float64(99.0)).unwrap();
        assert_eq!(a.get(&[0, 1]).unwrap(), Scalar::Float64(99.0));
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.set(&[5], Scalar::Float64(99.0)).is_err());
    }

    #[test]
    fn test_set_slice_1d() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let vals = NdArray::from_vec(vec![88.0_f64, 77.0]);
        a.set_slice(
            &[SliceArg::Range {
                start: Some(1),
                stop: Some(3),
                step: 1,
            }],
            &vals,
        )
        .unwrap();
        assert_eq!(a.get(&[1]).unwrap(), Scalar::Float64(88.0));
        assert_eq!(a.get(&[2]).unwrap(), Scalar::Float64(77.0));
        assert_eq!(a.get(&[0]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_view_slice_write_updates_base() {
        let base = NdArray::from_vec(vec![0_i64, 1, 2, 3, 4, 5]);
        let mut view = base
            .slice(&[SliceArg::Range {
                start: Some(1),
                stop: Some(4),
                step: 1,
            }])
            .unwrap();
        let vals = NdArray::from_vec(vec![9_i64, 8, 7]);
        view.set_slice(&[SliceArg::Full], &vals).unwrap();
        let ArrayData::Int64(arr) = base.data() else {
            panic!("expected Int64");
        };
        assert_eq!(arr.as_slice().unwrap(), &[0, 9, 8, 7, 4, 5]);
    }

    #[test]
    fn test_get_logical_reinterprets_uint64_storage() {
        let a = NdArray::from_vec(vec![-1_i64]).with_declared_dtype(DType::UInt64);
        assert_eq!(a.get_logical(&[0]).unwrap(), LogicalScalar::UInt(u64::MAX));
    }

    #[test]
    fn test_get_logical_reinterprets_float16_storage() {
        let a = NdArray::from_vec(vec![1.5_f32]).with_declared_dtype(DType::Float16);
        assert_eq!(a.get_logical(&[0]).unwrap(), LogicalScalar::Float(1.5));
    }

    #[test]
    fn test_unravel_index_basic() {
        // flat index 5 in shape (3, 4) -> row=1, col=1
        let idx = NdArray::from_vec(vec![5_i64]);
        let result = unravel_index(&idx, &[3, 4]).unwrap();
        assert_eq!(result.len(), 2);
        let ArrayData::Int64(r0) = result[0].data() else {
            panic!()
        };
        let ArrayData::Int64(r1) = result[1].data() else {
            panic!()
        };
        assert_eq!(r0[[0]], 1);
        assert_eq!(r1[[0]], 1);
    }

    #[test]
    fn test_unravel_index_multiple() {
        // indices [0, 5, 11] in shape (3, 4)
        let idx = NdArray::from_vec(vec![0_i64, 5, 11]);
        let result = unravel_index(&idx, &[3, 4]).unwrap();
        let ArrayData::Int64(r0) = result[0].data() else {
            panic!()
        };
        let ArrayData::Int64(r1) = result[1].data() else {
            panic!()
        };
        // 0 -> (0, 0), 5 -> (1, 1), 11 -> (2, 3)
        assert_eq!(r0[[0]], 0);
        assert_eq!(r1[[0]], 0);
        assert_eq!(r0[[1]], 1);
        assert_eq!(r1[[1]], 1);
        assert_eq!(r0[[2]], 2);
        assert_eq!(r1[[2]], 3);
    }

    #[test]
    fn test_ravel_multi_index_basic() {
        let i0 = NdArray::from_vec(vec![1_i64]);
        let i1 = NdArray::from_vec(vec![1_i64]);
        let result = ravel_multi_index(&[&i0, &i1], &[3, 4]).unwrap();
        let ArrayData::Int64(arr) = result.data() else {
            panic!()
        };
        assert_eq!(arr[[0]], 5);
    }

    #[test]
    fn test_unravel_ravel_roundtrip() {
        let idx = NdArray::from_vec(vec![7_i64]);
        let shape = [3, 4];
        let unraveled = unravel_index(&idx, &shape).unwrap();
        let refs: Vec<&NdArray> = unraveled.iter().collect();
        let raveled = ravel_multi_index(&refs, &shape).unwrap();
        let ArrayData::Int64(arr) = raveled.data() else {
            panic!()
        };
        assert_eq!(arr[[0]], 7);
    }
}
