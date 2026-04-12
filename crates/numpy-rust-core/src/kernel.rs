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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonKernelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

pub type BinaryArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
pub type ComparisonArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
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

pub fn comparison_kernel_for_dtype(
    dtype: DType,
    op: ComparisonKernelOp,
) -> Option<ComparisonArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Bool, ComparisonKernelOp::Eq) => Some(cmp_eq_bool),
        (DType::Bool, ComparisonKernelOp::Ne) => Some(cmp_ne_bool),
        (DType::Bool, ComparisonKernelOp::Lt) => Some(cmp_lt_bool),
        (DType::Bool, ComparisonKernelOp::Le) => Some(cmp_le_bool),
        (DType::Bool, ComparisonKernelOp::Gt) => Some(cmp_gt_bool),
        (DType::Bool, ComparisonKernelOp::Ge) => Some(cmp_ge_bool),
        (DType::Int32, ComparisonKernelOp::Eq) => Some(cmp_eq_int32),
        (DType::Int32, ComparisonKernelOp::Ne) => Some(cmp_ne_int32),
        (DType::Int32, ComparisonKernelOp::Lt) => Some(cmp_lt_int32),
        (DType::Int32, ComparisonKernelOp::Le) => Some(cmp_le_int32),
        (DType::Int32, ComparisonKernelOp::Gt) => Some(cmp_gt_int32),
        (DType::Int32, ComparisonKernelOp::Ge) => Some(cmp_ge_int32),
        (DType::Int64, ComparisonKernelOp::Eq) => Some(cmp_eq_int64),
        (DType::Int64, ComparisonKernelOp::Ne) => Some(cmp_ne_int64),
        (DType::Int64, ComparisonKernelOp::Lt) => Some(cmp_lt_int64),
        (DType::Int64, ComparisonKernelOp::Le) => Some(cmp_le_int64),
        (DType::Int64, ComparisonKernelOp::Gt) => Some(cmp_gt_int64),
        (DType::Int64, ComparisonKernelOp::Ge) => Some(cmp_ge_int64),
        (DType::Float32, ComparisonKernelOp::Eq) => Some(cmp_eq_float32),
        (DType::Float32, ComparisonKernelOp::Ne) => Some(cmp_ne_float32),
        (DType::Float32, ComparisonKernelOp::Lt) => Some(cmp_lt_float32),
        (DType::Float32, ComparisonKernelOp::Le) => Some(cmp_le_float32),
        (DType::Float32, ComparisonKernelOp::Gt) => Some(cmp_gt_float32),
        (DType::Float32, ComparisonKernelOp::Ge) => Some(cmp_ge_float32),
        (DType::Float64, ComparisonKernelOp::Eq) => Some(cmp_eq_float64),
        (DType::Float64, ComparisonKernelOp::Ne) => Some(cmp_ne_float64),
        (DType::Float64, ComparisonKernelOp::Lt) => Some(cmp_lt_float64),
        (DType::Float64, ComparisonKernelOp::Le) => Some(cmp_le_float64),
        (DType::Float64, ComparisonKernelOp::Gt) => Some(cmp_gt_float64),
        (DType::Float64, ComparisonKernelOp::Ge) => Some(cmp_ge_float64),
        (DType::Complex64, ComparisonKernelOp::Eq) => Some(cmp_eq_complex64),
        (DType::Complex64, ComparisonKernelOp::Ne) => Some(cmp_ne_complex64),
        (DType::Complex64, ComparisonKernelOp::Lt) => Some(cmp_lt_complex64),
        (DType::Complex64, ComparisonKernelOp::Le) => Some(cmp_le_complex64),
        (DType::Complex64, ComparisonKernelOp::Gt) => Some(cmp_gt_complex64),
        (DType::Complex64, ComparisonKernelOp::Ge) => Some(cmp_ge_complex64),
        (DType::Complex128, ComparisonKernelOp::Eq) => Some(cmp_eq_complex128),
        (DType::Complex128, ComparisonKernelOp::Ne) => Some(cmp_ne_complex128),
        (DType::Complex128, ComparisonKernelOp::Lt) => Some(cmp_lt_complex128),
        (DType::Complex128, ComparisonKernelOp::Le) => Some(cmp_le_complex128),
        (DType::Complex128, ComparisonKernelOp::Gt) => Some(cmp_gt_complex128),
        (DType::Complex128, ComparisonKernelOp::Ge) => Some(cmp_ge_complex128),
        (DType::Str, ComparisonKernelOp::Eq) => Some(cmp_eq_str),
        (DType::Str, ComparisonKernelOp::Ne) => Some(cmp_ne_str),
        (DType::Str, ComparisonKernelOp::Lt) => Some(cmp_lt_str),
        (DType::Str, ComparisonKernelOp::Le) => Some(cmp_le_str),
        (DType::Str, ComparisonKernelOp::Gt) => Some(cmp_gt_str),
        (DType::Str, ComparisonKernelOp::Ge) => Some(cmp_ge_str),
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

pub(crate) fn complex_cmp<T: num_traits::Float>(
    a: &Complex<T>,
    b: &Complex<T>,
) -> std::cmp::Ordering {
    match a.re.partial_cmp(&b.re) {
        Some(std::cmp::Ordering::Equal) => {
            a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal)
        }
        Some(ord) => ord,
        None => std::cmp::Ordering::Equal,
    }
}

macro_rules! primitive_comparison_kernel {
    ($name:ident, $variant:ident, |$x:ident, $y:ident| $body:expr) => {
        fn $name(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
            match (lhs, rhs) {
                (ArrayData::$variant(a), ArrayData::$variant(b)) => Ok(ArrayData::Bool(
                    ndarray::Zip::from(&a)
                        .and(&b)
                        .map_collect(|&$x, &$y| $body)
                        .into_shared(),
                )),
                _ => Err(NumpyError::TypeError(
                    "comparison kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

macro_rules! ref_comparison_kernel {
    ($name:ident, $variant:ident, |$x:ident, $y:ident| $body:expr) => {
        fn $name(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
            match (lhs, rhs) {
                (ArrayData::$variant(a), ArrayData::$variant(b)) => Ok(ArrayData::Bool(
                    ndarray::Zip::from(&a)
                        .and(&b)
                        .map_collect(|$x, $y| $body)
                        .into_shared(),
                )),
                _ => Err(NumpyError::TypeError(
                    "comparison kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

primitive_comparison_kernel!(cmp_eq_bool, Bool, |x, y| x == y);
primitive_comparison_kernel!(cmp_ne_bool, Bool, |x, y| x != y);
primitive_comparison_kernel!(cmp_lt_bool, Bool, |x, y| !x & y);
primitive_comparison_kernel!(cmp_le_bool, Bool, |x, y| x <= y);
primitive_comparison_kernel!(cmp_gt_bool, Bool, |x, y| x & !y);
primitive_comparison_kernel!(cmp_ge_bool, Bool, |x, y| x >= y);
primitive_comparison_kernel!(cmp_eq_int32, Int32, |x, y| x == y);
primitive_comparison_kernel!(cmp_ne_int32, Int32, |x, y| x != y);
primitive_comparison_kernel!(cmp_lt_int32, Int32, |x, y| x < y);
primitive_comparison_kernel!(cmp_le_int32, Int32, |x, y| x <= y);
primitive_comparison_kernel!(cmp_gt_int32, Int32, |x, y| x > y);
primitive_comparison_kernel!(cmp_ge_int32, Int32, |x, y| x >= y);
primitive_comparison_kernel!(cmp_eq_int64, Int64, |x, y| x == y);
primitive_comparison_kernel!(cmp_ne_int64, Int64, |x, y| x != y);
primitive_comparison_kernel!(cmp_lt_int64, Int64, |x, y| x < y);
primitive_comparison_kernel!(cmp_le_int64, Int64, |x, y| x <= y);
primitive_comparison_kernel!(cmp_gt_int64, Int64, |x, y| x > y);
primitive_comparison_kernel!(cmp_ge_int64, Int64, |x, y| x >= y);
primitive_comparison_kernel!(cmp_eq_float32, Float32, |x, y| x == y);
primitive_comparison_kernel!(cmp_ne_float32, Float32, |x, y| x != y);
primitive_comparison_kernel!(cmp_lt_float32, Float32, |x, y| x < y);
primitive_comparison_kernel!(cmp_le_float32, Float32, |x, y| x <= y);
primitive_comparison_kernel!(cmp_gt_float32, Float32, |x, y| x > y);
primitive_comparison_kernel!(cmp_ge_float32, Float32, |x, y| x >= y);
primitive_comparison_kernel!(cmp_eq_float64, Float64, |x, y| x == y);
primitive_comparison_kernel!(cmp_ne_float64, Float64, |x, y| x != y);
primitive_comparison_kernel!(cmp_lt_float64, Float64, |x, y| x < y);
primitive_comparison_kernel!(cmp_le_float64, Float64, |x, y| x <= y);
primitive_comparison_kernel!(cmp_gt_float64, Float64, |x, y| x > y);
primitive_comparison_kernel!(cmp_ge_float64, Float64, |x, y| x >= y);
primitive_comparison_kernel!(cmp_eq_complex64, Complex64, |x, y| x == y);
primitive_comparison_kernel!(cmp_ne_complex64, Complex64, |x, y| x != y);
primitive_comparison_kernel!(cmp_lt_complex64, Complex64, |x, y| {
    matches!(complex_cmp(&x, &y), std::cmp::Ordering::Less)
});
primitive_comparison_kernel!(cmp_le_complex64, Complex64, |x, y| {
    matches!(
        complex_cmp(&x, &y),
        std::cmp::Ordering::Less | std::cmp::Ordering::Equal
    )
});
primitive_comparison_kernel!(cmp_gt_complex64, Complex64, |x, y| {
    matches!(complex_cmp(&x, &y), std::cmp::Ordering::Greater)
});
primitive_comparison_kernel!(cmp_ge_complex64, Complex64, |x, y| {
    matches!(
        complex_cmp(&x, &y),
        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
    )
});
primitive_comparison_kernel!(cmp_eq_complex128, Complex128, |x, y| x == y);
primitive_comparison_kernel!(cmp_ne_complex128, Complex128, |x, y| x != y);
primitive_comparison_kernel!(cmp_lt_complex128, Complex128, |x, y| {
    matches!(complex_cmp(&x, &y), std::cmp::Ordering::Less)
});
primitive_comparison_kernel!(cmp_le_complex128, Complex128, |x, y| {
    matches!(
        complex_cmp(&x, &y),
        std::cmp::Ordering::Less | std::cmp::Ordering::Equal
    )
});
primitive_comparison_kernel!(cmp_gt_complex128, Complex128, |x, y| {
    matches!(complex_cmp(&x, &y), std::cmp::Ordering::Greater)
});
primitive_comparison_kernel!(cmp_ge_complex128, Complex128, |x, y| {
    matches!(
        complex_cmp(&x, &y),
        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
    )
});
ref_comparison_kernel!(cmp_eq_str, Str, |x, y| x == y);
ref_comparison_kernel!(cmp_ne_str, Str, |x, y| x != y);
ref_comparison_kernel!(cmp_lt_str, Str, |x, y| x < y);
ref_comparison_kernel!(cmp_le_str, Str, |x, y| x <= y);
ref_comparison_kernel!(cmp_gt_str, Str, |x, y| x > y);
ref_comparison_kernel!(cmp_ge_str, Str, |x, y| x >= y);

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
