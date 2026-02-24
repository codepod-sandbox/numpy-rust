use ndarray::ArrayD;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::error::Result;
use crate::NdArray;

/// Prepare two NdArrays for comparison: promote types and broadcast shapes.
fn prepare_cmp(lhs: &NdArray, rhs: &NdArray) -> Result<(ArrayData, ArrayData)> {
    let common_dtype = lhs.dtype().promote(rhs.dtype());
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let a = cast_array_data(&lhs.data, common_dtype);
    let b = cast_array_data(&rhs.data, common_dtype);

    let a = broadcast_array_data(&a, &out_shape);
    let b = broadcast_array_data(&b, &out_shape);

    Ok((a, b))
}

macro_rules! impl_cmp {
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
                    _ => unreachable!("promotion ensures matching types"),
                };
                Ok(NdArray::from_data(ArrayData::Bool(result)))
            }
        }
    };
}

impl_cmp!(eq, ==);
impl_cmp!(ne, !=);
impl_cmp!(lt, <);
impl_cmp!(gt, >);
impl_cmp!(le, <=);
impl_cmp!(ge, >=);

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

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
}
