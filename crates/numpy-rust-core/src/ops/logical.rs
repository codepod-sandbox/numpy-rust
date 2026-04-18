use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::kernel::{BitwiseBinaryKernelOp, BitwiseUnaryKernelOp, TruthKernelOp};
use crate::NdArray;

fn pack_byte_chunk(chunk: &[bool], little: bool) -> i32 {
    let mut byte = 0_i32;
    if little {
        for (j, bit) in chunk.iter().enumerate() {
            if *bit {
                byte |= 1 << j;
            }
        }
    } else {
        for (j, bit) in chunk.iter().enumerate() {
            if *bit {
                byte |= 1 << (7 - j);
            }
        }
    }
    byte
}

fn unpack_byte(byte: i64, little: bool, out: &mut Vec<i32>) {
    let byte = byte & 0xFF;
    if little {
        for j in 0..8 {
            out.push(((byte >> j) & 1) as i32);
        }
    } else {
        for j in (0..8).rev() {
            out.push(((byte >> j) & 1) as i32);
        }
    }
}

fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; order.len()];
    for (i, &ax) in order.iter().enumerate() {
        inverse[ax] = i;
    }
    inverse
}

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

fn finalize_bitwise_result(result: ArrayData, logical_dtype: DType) -> NdArray {
    let result = if logical_dtype.is_narrow() {
        crate::casting::narrow_truncate(result, logical_dtype)
    } else {
        result
    };
    let mut array = NdArray::from_data(result);
    if logical_dtype.is_narrow() {
        array.set_declared_dtype(logical_dtype);
    }
    array
}

fn execute_bitwise_binary(
    lhs: &NdArray,
    rhs: &NdArray,
    op: BitwiseBinaryKernelOp,
) -> Result<NdArray> {
    let (a, b, logical_dtype) = prepare_bitwise(lhs, rhs)?;
    let descriptor = descriptor_for_dtype(a.dtype());
    let kernel = descriptor.bitwise_binary_kernel(op).ok_or_else(|| {
        NumpyError::TypeError("unsupported operand types for bitwise operation".into())
    })?;
    let result = kernel(a, b)?;
    Ok(finalize_bitwise_result(result, logical_dtype))
}

impl NdArray {
    pub fn bitwise_count(&self) -> Result<NdArray> {
        if self.dtype().is_float() || self.dtype().is_complex() || self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "Expected an input array of integer or boolean data type".into(),
            ));
        }

        let cast = cast_array_data(self.data(), DType::Int64);
        let ArrayData::Int64(values) = cast else {
            unreachable!("int64 cast must produce int64 storage")
        };
        let shape = self.shape().to_vec();
        let counts: Vec<i32> = values
            .iter()
            .map(|&v| ((v as u64).count_ones()) as i32)
            .collect();
        let result = NdArray::from_vec(counts)
            .reshape(&shape)?
            .with_declared_dtype(DType::UInt8);
        Ok(result)
    }

    pub fn packbits(&self, axis: Option<usize>, little: bool) -> Result<NdArray> {
        if self.dtype().is_float() || self.dtype().is_complex() || self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "Expected an input array of integer or boolean data type".into(),
            ));
        }
        let ArrayData::Bool(bools) = truth_array(self.data().clone()) else {
            unreachable!("truth kernel must produce bool arrays")
        };

        if let Some(axis) = axis {
            if axis >= self.ndim() {
                return Err(NumpyError::InvalidAxis {
                    axis,
                    ndim: self.ndim(),
                });
            }
            let mut order: Vec<usize> = (0..self.ndim()).collect();
            order.remove(axis);
            order.push(axis);
            let permuted = NdArray::from_data(ArrayData::Bool(bools)).transpose_axes(&order)?;
            let ArrayData::Bool(pb) = permuted.data().clone() else {
                unreachable!()
            };
            let axis_len = permuted.shape()[permuted.ndim() - 1];
            let packed_len = axis_len.div_ceil(8);
            let outer = if axis_len == 0 {
                permuted.size()
            } else {
                permuted.size() / axis_len
            };
            let flat: Vec<bool> = pb.iter().copied().collect();
            let mut packed = Vec::with_capacity(outer * packed_len);
            for row in 0..outer {
                let start = row * axis_len;
                let row_slice = &flat[start..start + axis_len];
                for chunk in row_slice.chunks(8) {
                    packed.push(pack_byte_chunk(chunk, little));
                }
            }
            let mut out_shape = permuted.shape().to_vec();
            out_shape[permuted.ndim() - 1] = packed_len;
            let packed_arr = NdArray::from_vec(packed)
                .reshape(&out_shape)?
                .with_declared_dtype(DType::UInt8);
            let inverse = inverse_permutation(&order);
            return packed_arr.transpose_axes(&inverse);
        }

        let flat: Vec<bool> = bools.iter().copied().collect();
        let mut packed = Vec::with_capacity(flat.len().div_ceil(8));
        for chunk in flat.chunks(8) {
            packed.push(pack_byte_chunk(chunk, little));
        }
        Ok(NdArray::from_vec(packed).with_declared_dtype(DType::UInt8))
    }

    pub fn unpackbits(
        &self,
        axis: Option<usize>,
        count: Option<isize>,
        little: bool,
    ) -> Result<NdArray> {
        if !(self.dtype().is_integer() || self.dtype().is_bool()) {
            return Err(NumpyError::TypeError(
                "Expected an input array of unsigned byte data type".into(),
            ));
        }
        let cast = cast_array_data(self.data(), DType::Int64);
        let ArrayData::Int64(bytes) = cast else {
            unreachable!("int64 cast must produce int64 storage")
        };

        let adjust_len = |bits_len: usize| -> usize {
            match count {
                Some(c) if c >= 0 => usize::min(bits_len, c as usize),
                Some(c) => bits_len.saturating_sub((-c) as usize),
                None => bits_len,
            }
        };

        if let Some(axis) = axis {
            if axis >= self.ndim() {
                return Err(NumpyError::InvalidAxis {
                    axis,
                    ndim: self.ndim(),
                });
            }
            let mut order: Vec<usize> = (0..self.ndim()).collect();
            order.remove(axis);
            order.push(axis);
            let permuted = NdArray::from_data(ArrayData::Int64(bytes)).transpose_axes(&order)?;
            let ArrayData::Int64(pb) = permuted.data().clone() else {
                unreachable!()
            };
            let axis_len = permuted.shape()[permuted.ndim() - 1];
            let bits_len = axis_len * 8;
            let out_bits_len = adjust_len(bits_len);
            let outer = if axis_len == 0 {
                permuted.size()
            } else {
                permuted.size() / axis_len
            };
            let flat: Vec<i64> = pb.iter().copied().collect();
            let mut unpacked = Vec::with_capacity(outer * out_bits_len);
            for row in 0..outer {
                let start = row * axis_len;
                let row_slice = &flat[start..start + axis_len];
                let mut row_bits = Vec::with_capacity(bits_len);
                for &byte in row_slice {
                    unpack_byte(byte, little, &mut row_bits);
                }
                unpacked.extend_from_slice(&row_bits[..out_bits_len]);
            }
            let mut out_shape = permuted.shape().to_vec();
            out_shape[permuted.ndim() - 1] = out_bits_len;
            let unpacked_arr = NdArray::from_vec(unpacked)
                .reshape(&out_shape)?
                .with_declared_dtype(DType::UInt8);
            let inverse = inverse_permutation(&order);
            return unpacked_arr.transpose_axes(&inverse);
        }

        let flat: Vec<i64> = bytes.iter().copied().collect();
        let bits_len = flat.len() * 8;
        let out_bits_len = adjust_len(bits_len);
        let mut unpacked = Vec::with_capacity(bits_len);
        for byte in flat {
            unpack_byte(byte, little, &mut unpacked);
        }
        unpacked.truncate(out_bits_len);
        Ok(NdArray::from_vec(unpacked).with_declared_dtype(DType::UInt8))
    }

    /// Element-wise bitwise AND. For Bool arrays: logical AND. For integers: bitwise &.
    pub fn bitwise_and(&self, other: &NdArray) -> Result<NdArray> {
        execute_bitwise_binary(self, other, BitwiseBinaryKernelOp::And)
    }

    /// Element-wise bitwise OR. For Bool arrays: logical OR. For integers: bitwise |.
    pub fn bitwise_or(&self, other: &NdArray) -> Result<NdArray> {
        execute_bitwise_binary(self, other, BitwiseBinaryKernelOp::Or)
    }

    /// Element-wise bitwise XOR. For Bool arrays: logical XOR. For integers: bitwise ^.
    pub fn bitwise_xor(&self, other: &NdArray) -> Result<NdArray> {
        execute_bitwise_binary(self, other, BitwiseBinaryKernelOp::Xor)
    }

    /// Element-wise left shift. Bool arrays are cast to Int64 first.
    /// Shift amounts are masked to avoid overflow panics.
    pub fn left_shift(&self, other: &NdArray) -> Result<NdArray> {
        execute_bitwise_binary(self, other, BitwiseBinaryKernelOp::LeftShift)
    }

    /// Element-wise right shift. Bool arrays are cast to Int64 first.
    /// Shift amounts are masked to avoid overflow panics.
    pub fn right_shift(&self, other: &NdArray) -> Result<NdArray> {
        execute_bitwise_binary(self, other, BitwiseBinaryKernelOp::RightShift)
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
        if self.dtype().is_float() {
            return Err(NumpyError::TypeError(
                "bitwise NOT not supported for float arrays".into(),
            ));
        }
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "bitwise NOT not supported for string arrays".into(),
            ));
        }
        let descriptor = descriptor_for_dtype(self.dtype());
        let kernel = descriptor
            .bitwise_unary_kernel(BitwiseUnaryKernelOp::Not)
            .ok_or_else(|| {
                NumpyError::TypeError("unsupported operand type for bitwise NOT".into())
            })?;
        let mut result = NdArray::from_data(kernel(self.data().clone())?);
        result.preserve_descriptor_from(self);
        Ok(result)
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
