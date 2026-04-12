use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::kernel::TruthKernelOp;
use crate::NdArray;

/// Prepare two NdArrays for bitwise ops: promote types and broadcast shapes.
/// Only Bool and integer types are supported.
/// Returns `(lhs_data, rhs_data, logical_result_dtype)`.
fn prepare_bitwise(lhs: &NdArray, rhs: &NdArray) -> Result<(ArrayData, ArrayData, DType)> {
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

    let logical_dtype = lhs.dtype().promote(rhs.dtype());
    let storage_dtype = logical_dtype.storage_dtype();
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let a = cast_array_data(lhs.data(), storage_dtype);
    let b = cast_array_data(rhs.data(), storage_dtype);

    let a = broadcast_array_data(&a, &out_shape);
    let b = broadcast_array_data(&b, &out_shape);

    Ok((a, b, logical_dtype))
}

/// Prepare two NdArrays for logical ops: broadcast shapes. All dtypes allowed.
fn prepare_logical(lhs: &NdArray, rhs: &NdArray) -> Result<(ArrayData, ArrayData)> {
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;
    let a = broadcast_array_data(lhs.data(), &out_shape);
    let b = broadcast_array_data(rhs.data(), &out_shape);
    Ok((a, b))
}

fn truth_array(data: ArrayData) -> ArrayData {
    let descriptor = descriptor_for_dtype(data.dtype());
    let kernel = descriptor
        .truth_kernel(TruthKernelOp::ToBool)
        .unwrap_or_else(|| panic!("truth kernel not registered for {}", data.dtype()));
    kernel(data).expect("truth kernel dtype mismatch")
}

impl NdArray {
    /// Element-wise bitwise AND. For Bool arrays: logical AND. For integers: bitwise &.
    pub fn bitwise_and(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b, logical_dtype) = prepare_bitwise(self, other)?;
        let result = match (a, b) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                let r = ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x && y)
                    .into_shared();
                ArrayData::Bool(r)
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => ArrayData::Int32(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x & y)
                    .into_shared(),
            ),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => ArrayData::Int64(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x & y)
                    .into_shared(),
            ),
            _ => unreachable!("promotion ensures matching types"),
        };
        let result = if logical_dtype.is_narrow() {
            crate::casting::narrow_truncate(result, logical_dtype)
        } else {
            result
        };
        let mut r = NdArray::from_data(result);
        if logical_dtype.is_narrow() {
            r.set_declared_dtype(logical_dtype);
        }
        Ok(r)
    }

    /// Element-wise bitwise OR. For Bool arrays: logical OR. For integers: bitwise |.
    pub fn bitwise_or(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b, logical_dtype) = prepare_bitwise(self, other)?;
        let result = match (a, b) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                let r = ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x || y)
                    .into_shared();
                ArrayData::Bool(r)
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => ArrayData::Int32(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x | y)
                    .into_shared(),
            ),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => ArrayData::Int64(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x | y)
                    .into_shared(),
            ),
            _ => unreachable!("promotion ensures matching types"),
        };
        let result = if logical_dtype.is_narrow() {
            crate::casting::narrow_truncate(result, logical_dtype)
        } else {
            result
        };
        let mut r = NdArray::from_data(result);
        if logical_dtype.is_narrow() {
            r.set_declared_dtype(logical_dtype);
        }
        Ok(r)
    }

    /// Element-wise bitwise XOR. For Bool arrays: logical XOR. For integers: bitwise ^.
    pub fn bitwise_xor(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b, logical_dtype) = prepare_bitwise(self, other)?;
        let result = match (a, b) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                let r = ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x ^ y)
                    .into_shared();
                ArrayData::Bool(r)
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => ArrayData::Int32(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x ^ y)
                    .into_shared(),
            ),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => ArrayData::Int64(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| x ^ y)
                    .into_shared(),
            ),
            _ => unreachable!("promotion ensures matching types"),
        };
        let result = if logical_dtype.is_narrow() {
            crate::casting::narrow_truncate(result, logical_dtype)
        } else {
            result
        };
        let mut r = NdArray::from_data(result);
        if logical_dtype.is_narrow() {
            r.set_declared_dtype(logical_dtype);
        }
        Ok(r)
    }

    /// Element-wise left shift. Bool arrays are cast to Int64 first.
    /// Shift amounts are masked to avoid overflow panics.
    pub fn left_shift(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b, logical_dtype) = prepare_bitwise(self, other)?;
        let result = match (a, b) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                let a = a.mapv(|x| x as i64);
                let b = b.mapv(|x| x as i64);
                ArrayData::Int64(
                    ndarray::Zip::from(&a)
                        .and(&b)
                        .map_collect(|&x, &y| if !(0..64).contains(&y) { 0 } else { x << y })
                        .into_shared(),
                )
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => ArrayData::Int32(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| if !(0..32).contains(&y) { 0 } else { x << y })
                    .into_shared(),
            ),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => ArrayData::Int64(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| if !(0..64).contains(&y) { 0 } else { x << y })
                    .into_shared(),
            ),
            _ => unreachable!("promotion ensures matching types"),
        };
        let result = if logical_dtype.is_narrow() {
            crate::casting::narrow_truncate(result, logical_dtype)
        } else {
            result
        };
        let mut r = NdArray::from_data(result);
        if logical_dtype.is_narrow() {
            r.set_declared_dtype(logical_dtype);
        }
        Ok(r)
    }

    /// Element-wise right shift. Bool arrays are cast to Int64 first.
    /// Shift amounts are masked to avoid overflow panics.
    pub fn right_shift(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b, logical_dtype) = prepare_bitwise(self, other)?;
        let result = match (a, b) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => {
                let a = a.mapv(|x| x as i64);
                let b = b.mapv(|x| x as i64);
                ArrayData::Int64(
                    ndarray::Zip::from(&a)
                        .and(&b)
                        .map_collect(|&x, &y| {
                            if !(0..64).contains(&y) {
                                x >> 63
                            } else {
                                x >> y
                            }
                        })
                        .into_shared(),
                )
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => ArrayData::Int32(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| {
                        if !(0..32).contains(&y) {
                            x >> 31
                        } else {
                            x >> y
                        }
                    })
                    .into_shared(),
            ),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => ArrayData::Int64(
                ndarray::Zip::from(&a)
                    .and(&b)
                    .map_collect(|&x, &y| {
                        if !(0..64).contains(&y) {
                            x >> 63
                        } else {
                            x >> y
                        }
                    })
                    .into_shared(),
            ),
            _ => unreachable!("promotion ensures matching types"),
        };
        let result = if logical_dtype.is_narrow() {
            crate::casting::narrow_truncate(result, logical_dtype)
        } else {
            result
        };
        let mut r = NdArray::from_data(result);
        if logical_dtype.is_narrow() {
            r.set_declared_dtype(logical_dtype);
        }
        Ok(r)
    }

    /// Element-wise logical NOT. Returns Bool array (true where element is falsy).
    pub fn logical_not(&self) -> NdArray {
        let ArrayData::Bool(data) = truth_array(self.data().clone()) else {
            unreachable!("truth kernel must produce bool arrays")
        };
        NdArray::from_data(ArrayData::Bool(data.mapv(|x| !x).into_shared()))
    }

    /// Element-wise logical AND. Returns Bool array. Works on all dtypes (truthy check).
    pub fn logical_and(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b) = prepare_logical(self, other)?;
        let ArrayData::Bool(ba) = truth_array(a) else {
            unreachable!("truth kernel must produce bool arrays")
        };
        let ArrayData::Bool(bb) = truth_array(b) else {
            unreachable!("truth kernel must produce bool arrays")
        };
        let result = ndarray::Zip::from(&ba)
            .and(&bb)
            .map_collect(|&x, &y| x && y)
            .into_shared();
        Ok(NdArray::from_data(ArrayData::Bool(result)))
    }

    /// Element-wise logical OR. Returns Bool array. Works on all dtypes (truthy check).
    pub fn logical_or(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b) = prepare_logical(self, other)?;
        let ArrayData::Bool(ba) = truth_array(a) else {
            unreachable!("truth kernel must produce bool arrays")
        };
        let ArrayData::Bool(bb) = truth_array(b) else {
            unreachable!("truth kernel must produce bool arrays")
        };
        let result = ndarray::Zip::from(&ba)
            .and(&bb)
            .map_collect(|&x, &y| x || y)
            .into_shared();
        Ok(NdArray::from_data(ArrayData::Bool(result)))
    }

    /// Element-wise logical XOR. Returns Bool array. Works on all dtypes (truthy check).
    pub fn logical_xor(&self, other: &NdArray) -> Result<NdArray> {
        let (a, b) = prepare_logical(self, other)?;
        let ArrayData::Bool(ba) = truth_array(a) else {
            unreachable!("truth kernel must produce bool arrays")
        };
        let ArrayData::Bool(bb) = truth_array(b) else {
            unreachable!("truth kernel must produce bool arrays")
        };
        let result = ndarray::Zip::from(&ba)
            .and(&bb)
            .map_collect(|&x, &y| x ^ y)
            .into_shared();
        Ok(NdArray::from_data(ArrayData::Bool(result)))
    }

    /// Element-wise bitwise NOT. For Bool arrays: logical NOT. For integers: bitwise !.
    pub fn bitwise_not(&self) -> Result<NdArray> {
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "bitwise NOT not supported for complex arrays".into(),
            ));
        }
        let result = match self.data() {
            ArrayData::Bool(a) => ArrayData::Bool(a.mapv(|x| !x).into_shared()),
            ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| !x).into_shared()),
            ArrayData::Int64(a) => ArrayData::Int64(a.mapv(|x| !x).into_shared()),
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
        let mut r = NdArray::from_data(result);
        r.preserve_descriptor_from(self);
        Ok(r)
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

    #[test]
    fn test_bitwise_xor_bool() {
        let a = NdArray::from_vec(vec![true, true, false, false]);
        let b = NdArray::from_vec(vec![true, false, true, false]);
        let c = a.bitwise_xor(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.shape(), &[4]);
    }

    #[test]
    fn test_bitwise_xor_int() {
        let a = NdArray::from_vec(vec![0b1100_i32, 0b1010]);
        let b = NdArray::from_vec(vec![0b1010_i32, 0b1100]);
        let c = a.bitwise_xor(&b).unwrap();
        assert_eq!(c.dtype(), DType::Int32);
        assert_eq!(c.shape(), &[2]);
    }

    #[test]
    fn test_left_shift_int() {
        let a = NdArray::from_vec(vec![1_i64, 2, 4]);
        let b = NdArray::from_vec(vec![3_i64, 2, 1]);
        let c = a.left_shift(&b).unwrap();
        assert_eq!(c.dtype(), DType::Int64);
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_right_shift_int() {
        let a = NdArray::from_vec(vec![8_i64, 16, 32]);
        let b = NdArray::from_vec(vec![2_i64, 3, 4]);
        let c = a.right_shift(&b).unwrap();
        assert_eq!(c.dtype(), DType::Int64);
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_shift_float_fails() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.left_shift(&b).is_err());
        assert!(a.right_shift(&b).is_err());
    }

    #[test]
    fn test_logical_not() {
        let a = NdArray::from_vec(vec![true, false, true]);
        let b = a.logical_not();
        assert_eq!(b.dtype(), DType::Bool);
        assert_eq!(b.shape(), &[3]);

        let c = NdArray::from_vec(vec![0_i32, 1, 2]);
        let d = c.logical_not();
        assert_eq!(d.dtype(), DType::Bool);
        assert_eq!(d.shape(), &[3]);

        let e = NdArray::from_vec(vec![0.0_f64, 1.0, 0.0]);
        let f = e.logical_not();
        assert_eq!(f.dtype(), DType::Bool);
        assert_eq!(f.shape(), &[3]);
    }

    #[test]
    fn test_logical_and() {
        use crate::array_data::ArrayData;
        let a = NdArray::from_vec(vec![true, true, false, false]);
        let b = NdArray::from_vec(vec![true, false, true, false]);
        let c = a.logical_and(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.shape(), &[4]);
        if let ArrayData::Bool(arr) = c.data() {
            assert_eq!(arr.as_slice().unwrap(), &[true, false, false, false]);
        } else {
            panic!("expected Bool");
        }
    }

    #[test]
    fn test_logical_or() {
        use crate::array_data::ArrayData;
        let a = NdArray::from_vec(vec![true, true, false, false]);
        let b = NdArray::from_vec(vec![true, false, true, false]);
        let c = a.logical_or(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.shape(), &[4]);
        if let ArrayData::Bool(arr) = c.data() {
            assert_eq!(arr.as_slice().unwrap(), &[true, true, true, false]);
        } else {
            panic!("expected Bool");
        }
    }

    #[test]
    fn test_logical_xor() {
        use crate::array_data::ArrayData;
        let a = NdArray::from_vec(vec![true, true, false, false]);
        let b = NdArray::from_vec(vec![true, false, true, false]);
        let c = a.logical_xor(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.shape(), &[4]);
        if let ArrayData::Bool(arr) = c.data() {
            assert_eq!(arr.as_slice().unwrap(), &[false, true, true, false]);
        } else {
            panic!("expected Bool");
        }
    }

    #[test]
    fn test_logical_and_int() {
        use crate::array_data::ArrayData;
        let a = NdArray::from_vec(vec![1_i32, 0, 2]);
        let b = NdArray::from_vec(vec![1_i32, 1, 0]);
        let c = a.logical_and(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        if let ArrayData::Bool(arr) = c.data() {
            assert_eq!(arr.as_slice().unwrap(), &[true, false, false]);
        } else {
            panic!("expected Bool");
        }
    }
}
