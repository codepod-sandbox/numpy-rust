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
            ArrayData::Bool(a) => {
                ArrayData::Bool(a.clone().into_shape_with_order(sh).unwrap())
            }
            ArrayData::Int32(a) => {
                ArrayData::Int32(a.clone().into_shape_with_order(sh).unwrap())
            }
            ArrayData::Int64(a) => {
                ArrayData::Int64(a.clone().into_shape_with_order(sh).unwrap())
            }
            ArrayData::Float32(a) => {
                ArrayData::Float32(a.clone().into_shape_with_order(sh).unwrap())
            }
            ArrayData::Float64(a) => {
                ArrayData::Float64(a.clone().into_shape_with_order(sh).unwrap())
            }
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
        };
        NdArray::from_data(data)
    }

    /// Return a 1-D copy of the array.
    pub fn flatten(&self) -> NdArray {
        self.reshape(&[self.size()]).unwrap()
    }

    /// Return a contiguous flattened array (same as flatten for our purposes).
    pub fn ravel(&self) -> NdArray {
        self.flatten()
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
    let data = match common_dtype {
        crate::DType::Bool => {
            let views: Vec<_> = promoted
                .iter()
                .map(|d| match d {
                    ArrayData::Bool(a) => a.view(),
                    _ => unreachable!(),
                })
                .collect();
            ArrayData::Bool(ndarray::concatenate(ax, &views).map_err(|e| {
                NumpyError::ShapeMismatch(e.to_string())
            })?)
        }
        crate::DType::Int32 => {
            let views: Vec<_> = promoted
                .iter()
                .map(|d| match d {
                    ArrayData::Int32(a) => a.view(),
                    _ => unreachable!(),
                })
                .collect();
            ArrayData::Int32(ndarray::concatenate(ax, &views).map_err(|e| {
                NumpyError::ShapeMismatch(e.to_string())
            })?)
        }
        crate::DType::Int64 => {
            let views: Vec<_> = promoted
                .iter()
                .map(|d| match d {
                    ArrayData::Int64(a) => a.view(),
                    _ => unreachable!(),
                })
                .collect();
            ArrayData::Int64(ndarray::concatenate(ax, &views).map_err(|e| {
                NumpyError::ShapeMismatch(e.to_string())
            })?)
        }
        crate::DType::Float32 => {
            let views: Vec<_> = promoted
                .iter()
                .map(|d| match d {
                    ArrayData::Float32(a) => a.view(),
                    _ => unreachable!(),
                })
                .collect();
            ArrayData::Float32(ndarray::concatenate(ax, &views).map_err(|e| {
                NumpyError::ShapeMismatch(e.to_string())
            })?)
        }
        crate::DType::Float64 => {
            let views: Vec<_> = promoted
                .iter()
                .map(|d| match d {
                    ArrayData::Float64(a) => a.view(),
                    _ => unreachable!(),
                })
                .collect();
            ArrayData::Float64(ndarray::concatenate(ax, &views).map_err(|e| {
                NumpyError::ShapeMismatch(e.to_string())
            })?)
        }
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
            a.reshape(&new_shape).unwrap()
        })
        .collect();

    let refs: Vec<&NdArray> = expanded.iter().collect();
    concatenate(&refs, axis)
}

/// Vertical stack — concatenate along axis 0.
pub fn vstack(arrays: &[&NdArray]) -> Result<NdArray> {
    concatenate(arrays, 0)
}

/// Horizontal stack — concatenate along axis 1 (or axis 0 for 1-D arrays).
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
}
