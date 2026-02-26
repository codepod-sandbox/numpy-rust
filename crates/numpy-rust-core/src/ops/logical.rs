use ndarray::ArrayD;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Prepare two NdArrays for bitwise ops: promote types and broadcast shapes.
/// Only Bool and integer types are supported.
fn prepare_bitwise(lhs: &NdArray, rhs: &NdArray) -> Result<(ArrayData, ArrayData)> {
    if lhs.dtype().is_string() || rhs.dtype().is_string() {
        return Err(NumpyError::TypeError(
            "bitwise operations not supported for string arrays".into(),
        ));
    }
    if lhs.dtype().is_float() || rhs.dtype().is_float() {
        return Err(NumpyError::TypeError(
            "bitwise operations not supported for float arrays".into(),
        ));
    }
    if lhs.dtype().is_complex() || rhs.dtype().is_complex() {
        return Err(NumpyError::TypeError(
            "bitwise operations not supported for complex arrays".into(),
        ));
    }

    let common_dtype = lhs.dtype().promote(rhs.dtype());
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let a = cast_array_data(&lhs.data, common_dtype);
    let b = cast_array_data(&rhs.data, common_dtype);

    let a = broadcast_array_data(&a, &out_shape);
    let b = broadcast_array_data(&b, &out_shape);

    Ok((a, b))
}

impl NdArray {
    /// Element-wise bitwise AND. For Bool arrays: logical AND. For integers: bitwise &.
    pub fn bitwise_and(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b) = prepare_bitwise(self, other)?;
        let result = match (a, b) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                let r: ArrayD<bool> = ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x && y);
                ArrayData::Bool(r)
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => {
                ArrayData::Int32(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x & y))
            }
            (ArrayData::Int64(a), ArrayData::Int64(b)) => {
                ArrayData::Int64(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x & y))
            }
            _ => unreachable!("promotion ensures matching types"),
        };
        Ok(NdArray::from_data(result))
    }

    /// Element-wise bitwise OR. For Bool arrays: logical OR. For integers: bitwise |.
    pub fn bitwise_or(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b) = prepare_bitwise(self, other)?;
        let result = match (a, b) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                let r: ArrayD<bool> = ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x || y);
                ArrayData::Bool(r)
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => {
                ArrayData::Int32(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x | y))
            }
            (ArrayData::Int64(a), ArrayData::Int64(b)) => {
                ArrayData::Int64(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x | y))
            }
            _ => unreachable!("promotion ensures matching types"),
        };
        Ok(NdArray::from_data(result))
    }

    /// Element-wise bitwise NOT. For Bool arrays: logical NOT. For integers: bitwise !.
    pub fn bitwise_not(&self) -> Result<NdArray> {
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "bitwise NOT not supported for complex arrays".into(),
            ));
        }
        let result = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(a.mapv(|x| !x)),
            ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| !x)),
            ArrayData::Int64(a) => ArrayData::Int64(a.mapv(|x| !x)),
            ArrayData::Float32(_) | ArrayData::Float64(_) => {
                return Err(NumpyError::TypeError(
                    "bitwise NOT not supported for float arrays".into(),
                ));
            }
            ArrayData::Complex64(_) | ArrayData::Complex128(_) => {
                return Err(NumpyError::TypeError(
                    "bitwise NOT not supported for complex arrays".into(),
                ));
            }
            ArrayData::Str(_) => {
                return Err(NumpyError::TypeError(
                    "bitwise NOT not supported for string arrays".into(),
                ));
            }
        };
        Ok(NdArray::from_data(result))
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_bitwise_and_bool() {
        let a = NdArray::from_vec(vec![true, true, false, false]);
        let b = NdArray::from_vec(vec![true, false, true, false]);
        let c = a.bitwise_and(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.shape(), &[4]);
    }

    #[test]
    fn test_bitwise_or_bool() {
        let a = NdArray::from_vec(vec![true, true, false, false]);
        let b = NdArray::from_vec(vec![true, false, true, false]);
        let c = a.bitwise_or(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_bitwise_not_bool() {
        let a = NdArray::from_vec(vec![true, false]);
        let b = a.bitwise_not().unwrap();
        assert_eq!(b.dtype(), DType::Bool);
    }

    #[test]
    fn test_bitwise_and_int() {
        let a = NdArray::from_vec(vec![0b1100_i32, 0b1010]);
        let b = NdArray::from_vec(vec![0b1010_i32, 0b1100]);
        let c = a.bitwise_and(&b).unwrap();
        assert_eq!(c.dtype(), DType::Int32);
    }

    #[test]
    fn test_bitwise_float_fails() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.bitwise_and(&b).is_err());
    }

    #[test]
    fn test_bitwise_complex_fails() {
        let a = NdArray::zeros(&[2], DType::Complex128);
        let b = NdArray::zeros(&[2], DType::Complex128);
        assert!(a.bitwise_and(&b).is_err());
        assert!(a.bitwise_or(&b).is_err());
        assert!(a.bitwise_not().is_err());
    }
}
