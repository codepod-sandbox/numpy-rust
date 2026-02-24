use std::ops;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::error::Result;
use crate::NdArray;

/// Prepare two NdArrays for a binary operation: promote types and broadcast shapes.
fn prepare_binary(lhs: &NdArray, rhs: &NdArray) -> Result<(ArrayData, ArrayData)> {
    let common_dtype = lhs.dtype().promote(rhs.dtype());
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let a = cast_array_data(&lhs.data, common_dtype);
    let b = cast_array_data(&rhs.data, common_dtype);

    let a = broadcast_array_data(&a, &out_shape);
    let b = broadcast_array_data(&b, &out_shape);

    Ok((a, b))
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl ops::$trait<&NdArray> for &NdArray {
            type Output = Result<NdArray>;

            fn $method(self, rhs: &NdArray) -> Result<NdArray> {
                let (a, b) = prepare_binary(self, rhs)?;
                let data = match (a, b) {
                    (ArrayData::Float64(a), ArrayData::Float64(b)) => ArrayData::Float64(a $op b),
                    (ArrayData::Float32(a), ArrayData::Float32(b)) => ArrayData::Float32(a $op b),
                    (ArrayData::Int64(a), ArrayData::Int64(b)) => ArrayData::Int64(a $op b),
                    (ArrayData::Int32(a), ArrayData::Int32(b)) => ArrayData::Int32(a $op b),
                    _ => unreachable!("promotion ensures matching types"),
                };
                Ok(NdArray::from_data(data))
            }
        }
    };
}

impl_binary_op!(Add, add, +);
impl_binary_op!(Sub, sub, -);
impl_binary_op!(Mul, mul, *);
impl_binary_op!(Div, div, /);

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_add_same_shape() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_add_type_promotion() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_add_broadcast() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::ones(&[4], DType::Float64);
        let c = (&a + &b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_sub() {
        let a = NdArray::from_vec(vec![5.0_f64, 3.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 1.0]);
        let c = (&a - &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_mul() {
        let a = NdArray::from_vec(vec![2.0_f64, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0]);
        let _c = (&a * &b).unwrap();
    }

    #[test]
    fn test_div() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 5.0]);
        let _c = (&a / &b).unwrap();
    }

    #[test]
    fn test_broadcast_incompatible_fails() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::zeros(&[5], DType::Float64);
        assert!((&a + &b).is_err());
    }

    #[test]
    fn test_add_i32() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![10_i32, 20, 30]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.dtype(), DType::Int32);
    }

    #[test]
    fn test_mul_broadcast_scalar() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::full_f64(&[], 2.0); // scalar
        let c = (&a * &b).unwrap();
        assert_eq!(c.shape(), &[3]);
    }
}
