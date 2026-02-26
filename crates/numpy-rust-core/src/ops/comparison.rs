use ndarray::ArrayD;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Prepare two NdArrays for comparison: promote types and broadcast shapes.
fn prepare_cmp(lhs: &NdArray, rhs: &NdArray) -> Result<(ArrayData, ArrayData)> {
    // String vs numeric comparison is not supported
    if lhs.dtype().is_string() != rhs.dtype().is_string() {
        return Err(crate::error::NumpyError::TypeError(
            "comparison between string and numeric arrays not supported".into(),
        ));
    }
    // String+String: skip promotion (both already Str)
    if lhs.dtype().is_string() {
        let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;
        let a = broadcast_array_data(&lhs.data, &out_shape);
        let b = broadcast_array_data(&rhs.data, &out_shape);
        return Ok((a, b));
    }
    let common_dtype = lhs.dtype().promote(rhs.dtype());
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let a = cast_array_data(&lhs.data, common_dtype);
    let b = cast_array_data(&rhs.data, common_dtype);

    let a = broadcast_array_data(&a, &out_shape);
    let b = broadcast_array_data(&b, &out_shape);

    Ok((a, b))
}

/// Implement equality / inequality for complex (they support == and !=).
macro_rules! impl_eq_cmp {
    ($name:ident, $op:tt) => {
        impl NdArray {
            pub fn $name(&self, other: &NdArray) -> Result<NdArray> {
                let (a, b) = prepare_cmp(self, other)?;
                let result: ArrayD<bool> = match (a, b) {
                    (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Int32(a), ArrayData::Int32(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Int64(a), ArrayData::Int64(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Float32(a), ArrayData::Float32(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Float64(a), ArrayData::Float64(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Complex64(a), ArrayData::Complex64(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Complex128(a), ArrayData::Complex128(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Str(a), ArrayData::Str(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|x, y| x $op y)
                    }
                    _ => unreachable!("promotion ensures matching types"),
                };
                Ok(NdArray::from_data(ArrayData::Bool(result)))
            }
        }
    };
}

impl_eq_cmp!(eq, ==);
impl_eq_cmp!(ne, !=);

/// Implement ordering comparisons that do NOT work on complex.
macro_rules! impl_ord_cmp {
    ($name:ident, $op:tt) => {
        impl NdArray {
            pub fn $name(&self, other: &NdArray) -> Result<NdArray> {
                if self.dtype().is_complex() || other.dtype().is_complex() {
                    return Err(NumpyError::TypeError(
                        concat!(stringify!($name), " not supported for complex arrays").into(),
                    ));
                }
                let (a, b) = prepare_cmp(self, other)?;
                let result: ArrayD<bool> = match (a, b) {
                    (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Int32(a), ArrayData::Int32(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Int64(a), ArrayData::Int64(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Float32(a), ArrayData::Float32(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Float64(a), ArrayData::Float64(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x $op y)
                    }
                    (ArrayData::Str(a), ArrayData::Str(b)) => {
                        ndarray::Zip::from(&a).and(&b).map_collect(|x, y| x $op y)
                    }
                    _ => unreachable!("promotion ensures matching types (complex rejected above)"),
                };
                Ok(NdArray::from_data(ArrayData::Bool(result)))
            }
        }
    };
}

impl_ord_cmp!(lt, <);
impl_ord_cmp!(gt, >);
impl_ord_cmp!(le, <=);
impl_ord_cmp!(ge, >=);

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};
    use num_complex::Complex;

    #[test]
    fn test_eq_same_shape() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 9.0, 3.0]);
        let c = a.eq(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_ne() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 9.0]);
        let c = a.ne(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_lt() {
        let a = NdArray::from_vec(vec![1.0_f64, 5.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 3.0]);
        let c = a.lt(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_gt_broadcast() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::ones(&[4], DType::Float64);
        let c = a.gt(&b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_le_cross_dtype() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![1.5_f64, 2.5, 2.5]);
        let c = a.le(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_ge() {
        let a = NdArray::from_vec(vec![3_i32, 1, 2]);
        let b = NdArray::from_vec(vec![2_i32, 2, 2]);
        let c = a.ge(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_cmp_incompatible_fails() {
        let a = NdArray::zeros(&[3], DType::Float64);
        let b = NdArray::zeros(&[4], DType::Float64);
        assert!(a.eq(&b).is_err());
    }

    #[test]
    fn test_eq_complex() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0)]);
        let b = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0), Complex::new(0.0, 0.0)]);
        let c = a.eq(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_lt_complex_fails() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        let b = NdArray::from_vec(vec![Complex::new(3.0f64, 4.0)]);
        assert!(a.lt(&b).is_err());
    }

    #[test]
    fn test_gt_complex_fails() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        let b = NdArray::from_vec(vec![Complex::new(3.0f64, 4.0)]);
        assert!(a.gt(&b).is_err());
    }
}
