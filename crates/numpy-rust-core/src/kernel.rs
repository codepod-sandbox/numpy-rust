use ndarray::{Axis, IxDyn};
use num_complex::Complex;

use crate::array_data::{ArrayD, ArrayData};
use crate::dtype::DType;
use crate::error::{NumpyError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Unary,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticKernelOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Remainder,
}

pub type BinaryArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
pub type ReduceAllArrayKernel = fn(ArrayData) -> Result<ArrayData>;
pub type ReduceAxisArrayKernel = fn(ArrayData, usize) -> Result<ArrayData>;

pub fn binary_kernel_for_dtype(dtype: DType, op: ArithmeticKernelOp) -> Option<BinaryArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Int32, ArithmeticKernelOp::Add) => Some(add_int32),
        (DType::Int64, ArithmeticKernelOp::Add) => Some(add_int64),
        (DType::Float32, ArithmeticKernelOp::Add) => Some(add_float32),
        (DType::Float64, ArithmeticKernelOp::Add) => Some(add_float64),
        (DType::Complex64, ArithmeticKernelOp::Add) => Some(add_complex64),
        (DType::Complex128, ArithmeticKernelOp::Add) => Some(add_complex128),
        _ => None,
    }
}

pub fn sum_all_kernel_for_dtype(dtype: DType) -> Option<ReduceAllArrayKernel> {
    match dtype.storage_dtype() {
        DType::Int32 => Some(sum_all_int32),
        DType::Int64 => Some(sum_all_int64),
        DType::Float32 => Some(sum_all_float32),
        DType::Float64 => Some(sum_all_float64),
        DType::Complex64 => Some(sum_all_complex64),
        DType::Complex128 => Some(sum_all_complex128),
        _ => None,
    }
}

pub fn sum_axis_kernel_for_dtype(dtype: DType) -> Option<ReduceAxisArrayKernel> {
    match dtype.storage_dtype() {
        DType::Int32 => Some(sum_axis_int32),
        DType::Int64 => Some(sum_axis_int64),
        DType::Float32 => Some(sum_axis_float32),
        DType::Float64 => Some(sum_axis_float64),
        DType::Complex64 => Some(sum_axis_complex64),
        DType::Complex128 => Some(sum_axis_complex128),
        _ => None,
    }
}

macro_rules! simple_binary_kernel {
    ($name:ident, $variant:ident, $op:tt) => {
        fn $name(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
            match (lhs, rhs) {
                (ArrayData::$variant(a), ArrayData::$variant(b)) => Ok(ArrayData::$variant(a $op b)),
                _ => Err(NumpyError::TypeError(
                    "unsupported operand types for binary operation".into(),
                )),
            }
        }
    };
}

simple_binary_kernel!(add_int32, Int32, +);
simple_binary_kernel!(add_int64, Int64, +);
simple_binary_kernel!(add_float32, Float32, +);
simple_binary_kernel!(add_float64, Float64, +);
simple_binary_kernel!(add_complex64, Complex64, +);
simple_binary_kernel!(add_complex128, Complex128, +);

macro_rules! sum_all_kernel {
    ($name:ident, $variant:ident, $ty:ty) => {
        fn $name(data: ArrayData) -> Result<ArrayData> {
            match data {
                ArrayData::$variant(a) => {
                    let s: $ty = a.iter().copied().sum();
                    Ok(ArrayData::$variant(
                        ArrayD::from_elem(IxDyn(&[]), s).into_shared(),
                    ))
                }
                _ => Err(NumpyError::TypeError("sum kernel dtype mismatch".into())),
            }
        }
    };
}

sum_all_kernel!(sum_all_int32, Int32, i32);
sum_all_kernel!(sum_all_int64, Int64, i64);
sum_all_kernel!(sum_all_float32, Float32, f32);
sum_all_kernel!(sum_all_float64, Float64, f64);
sum_all_kernel!(sum_all_complex64, Complex64, Complex<f32>);
sum_all_kernel!(sum_all_complex128, Complex128, Complex<f64>);

macro_rules! sum_axis_kernel {
    ($name:ident, $variant:ident) => {
        fn $name(data: ArrayData, axis: usize) -> Result<ArrayData> {
            match data {
                ArrayData::$variant(a) => {
                    Ok(ArrayData::$variant(a.sum_axis(Axis(axis)).into_shared()))
                }
                _ => Err(NumpyError::TypeError("sum kernel dtype mismatch".into())),
            }
        }
    };
}

sum_axis_kernel!(sum_axis_int32, Int32);
sum_axis_kernel!(sum_axis_int64, Int64);
sum_axis_kernel!(sum_axis_float32, Float32);
sum_axis_kernel!(sum_axis_float64, Float64);
sum_axis_kernel!(sum_axis_complex64, Complex64);
sum_axis_kernel!(sum_axis_complex128, Complex128);
