use ndarray::{Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::casting::cast_array_data;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Return a new array with the given shape. Total size must match.
    pub fn reshape(&self, shape: &[usize]) -> Result<NdArray> {
        let new_size: usize = shape.iter().product();
        if new_size != self.size() {
            return Err(NumpyError::ReshapeError {
                from: self.size(),
                to: shape.to_vec(),
            });
        }
        let sh = IxDyn(shape);
        let data = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Int32(a) => ArrayData::Int32(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Int64(a) => ArrayData::Int64(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Float32(a) => ArrayData::Float32(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Float64(a) => ArrayData::Float64(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Complex64(a) => ArrayData::Complex64(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Complex128(a) => ArrayData::Complex128(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Str(a) => ArrayData::Str(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
        };
        Ok(NdArray::from_data(data))
    }

    /// Transpose the array (reverse axes).
    pub fn transpose(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(a.t().to_owned()),
            ArrayData::Int32(a) => ArrayData::Int32(a.t().to_owned()),
            ArrayData::Int64(a) => ArrayData::Int64(a.t().to_owned()),
            ArrayData::Float32(a) => ArrayData::Float32(a.t().to_owned()),
            ArrayData::Float64(a) => ArrayData::Float64(a.t().to_owned()),
            ArrayData::Complex64(a) => ArrayData::Complex64(a.t().to_owned()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.t().to_owned()),
            ArrayData::Str(a) => ArrayData::Str(a.t().to_owned()),
        };
        NdArray::from_data(data)
    }

    /// Return a 1-D copy of the array.
    pub fn flatten(&self) -> NdArray {
        self.reshape(&[self.size()])
            .expect("flatten reshape cannot fail")
    }

    /// Return a contiguous flattened array (same as flatten for our purposes).
    pub fn ravel(&self) -> NdArray {
        self.flatten()
    }

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

/// Concatenate arrays along an axis. All arrays must have the same shape
/// except in the concatenation dimension.
pub fn concatenate(arrays: &[&NdArray], axis: usize) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to concatenate".into(),
        ));
    }

    let ndim = arrays[0].ndim();
    if axis >= ndim {
        return Err(NumpyError::InvalidAxis { axis, ndim });
    }

    // Promote all to common dtype
    let common_dtype = arrays
        .iter()
        .skip(1)
        .fold(arrays[0].dtype(), |acc, a| acc.promote(a.dtype()));

    let promoted: Vec<_> = arrays
        .iter()
        .map(|a| cast_array_data(&a.data, common_dtype))
        .collect();

    let ax = Axis(axis);

    macro_rules! concat_variant {
        ($variant:ident) => {{
            let views: Vec<_> = promoted
                .iter()
                .map(|d| match d {
                    ArrayData::$variant(a) => a.view(),
                    _ => unreachable!(),
                })
                .collect();
            ArrayData::$variant(
                ndarray::concatenate(ax, &views)
                    .map_err(|e| NumpyError::ShapeMismatch(e.to_string()))?,
            )
        }};
    }

    let data = match common_dtype {
        crate::DType::Bool => concat_variant!(Bool),
        crate::DType::Int32 => concat_variant!(Int32),
        crate::DType::Int64 => concat_variant!(Int64),
        crate::DType::Float32 => concat_variant!(Float32),
        crate::DType::Float64 => concat_variant!(Float64),
        crate::DType::Complex64 => concat_variant!(Complex64),
        crate::DType::Complex128 => concat_variant!(Complex128),
        crate::DType::Str => concat_variant!(Str),
    };

    Ok(NdArray::from_data(data))
}

/// Stack arrays along a new axis.
pub fn stack(arrays: &[&NdArray], axis: usize) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to stack".into(),
        ));
    }

    // Each array gets a new axis inserted at `axis`, then concatenate
    let expanded: Vec<NdArray> = arrays
        .iter()
        .map(|a| {
            let mut new_shape = a.shape().to_vec();
            new_shape.insert(axis, 1);
            a.reshape(&new_shape)
                .expect("insert-axis reshape cannot fail")
        })
        .collect();

    let refs: Vec<&NdArray> = expanded.iter().collect();
    concatenate(&refs, axis)
}

/// Vertical stack -- concatenate along axis 0.
pub fn vstack(arrays: &[&NdArray]) -> Result<NdArray> {
    concatenate(arrays, 0)
}

/// Horizontal stack -- concatenate along axis 1 (or axis 0 for 1-D arrays).
pub fn hstack(arrays: &[&NdArray]) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to hstack".into(),
        ));
    }
    if arrays[0].ndim() == 1 {
        concatenate(arrays, 0)
    } else {
        concatenate(arrays, 1)
    }
}

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
        return Err(NumpyError::InvalidAxis {
            axis,
            ndim: a.ndim(),
        });
    }
    let axis_len = a.shape()[axis];

    let indices = match spec {
        SplitSpec::NSections(n) => {
            if *n == 0 {
                return Err(NumpyError::ValueError(
                    "number of sections must be > 0".into(),
                ));
            }
            if !axis_len.is_multiple_of(*n) {
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

    let mut boundaries = Vec::with_capacity(indices.len() + 2);
    boundaries.push(0usize);
    boundaries.extend_from_slice(&indices);
    boundaries.push(axis_len);

    let mut result = Vec::with_capacity(boundaries.len() - 1);
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
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

/// Split along axis 1 (or axis 0 for 1-D arrays).
pub fn hsplit(a: &NdArray, spec: &SplitSpec) -> Result<Vec<NdArray>> {
    if a.ndim() == 1 {
        split(a, spec, 0)
    } else {
        split(a, spec, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_reshape() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.reshape(&[2, 3]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_invalid() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(a.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn test_transpose() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = a.transpose();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose_3d() {
        let a = NdArray::zeros(&[2, 3, 4], DType::Float64);
        let b = a.transpose();
        assert_eq!(b.shape(), &[4, 3, 2]);
    }

    #[test]
    fn test_flatten() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = a.flatten();
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_ravel() {
        let a = NdArray::zeros(&[2, 3, 4], DType::Int32);
        let b = a.ravel();
        assert_eq!(b.shape(), &[24]);
    }

    #[test]
    fn test_concatenate_axis0() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 3], DType::Float64);
        let c = concatenate(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[4, 3]);
    }

    #[test]
    fn test_concatenate_axis1() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 4], DType::Float64);
        let c = concatenate(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 7]);
    }

    #[test]
    fn test_concatenate_type_promotion() {
        let a = NdArray::from_vec(vec![1_i32, 2]);
        let b = NdArray::from_vec(vec![3.0_f64, 4.0]);
        let c = concatenate(&[&a, &b], 0).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
        assert_eq!(c.shape(), &[4]);
    }

    #[test]
    fn test_stack() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_stack_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = stack(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
    }

    #[test]
    fn test_vstack() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[1, 3], DType::Float64);
        let c = vstack(&[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[3, 3]);
    }

    #[test]
    fn test_hstack_2d() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 1], DType::Float64);
        let c = hstack(&[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_hstack_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = NdArray::from_vec(vec![3.0_f64, 4.0, 5.0]);
        let c = hstack(&[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[5]);
    }

    #[test]
    fn test_expand_dims() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = a.expand_dims(0).unwrap();
        assert_eq!(b.shape(), &[1, 3]);
        let c = a.expand_dims(1).unwrap();
        assert_eq!(c.shape(), &[3, 1]);
    }

    #[test]
    fn test_expand_dims_invalid() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(a.expand_dims(3).is_err()); // ndim=1, max valid axis=1
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

    #[test]
    fn test_split_equal() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let parts = super::split(&a, &super::SplitSpec::NSections(3), 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[2].shape(), &[2]);
    }

    #[test]
    fn test_split_indices() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let parts = super::split(&a, &super::SplitSpec::Indices(vec![2, 4]), 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[2].shape(), &[1]);
    }

    #[test]
    fn test_split_2d_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .reshape(&[2, 4])
            .unwrap();
        let parts = super::split(&a, &super::SplitSpec::NSections(2), 1).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[1].shape(), &[2, 2]);
    }
}
