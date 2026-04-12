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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotKernelOp {
    Dot1d1d,
    MatMul2d2d,
    MatMul2d1d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhereKernelOp {
    Select,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateKernelOp {
    IsNaN,
    IsFinite,
    IsInf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKernelOp {
    Sum,
    Prod,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgReductionKernelOp {
    ArgMin,
    ArgMax,
}

pub type BinaryArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
pub type ComparisonArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
pub type DotArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
pub type WhereArrayKernel = fn(ArrayData, ArrayData, ArrayData) -> Result<ArrayData>;
pub type PredicateArrayKernel = fn(ArrayData) -> Result<ArrayData>;
pub type ReduceAllArrayKernel = fn(ArrayData) -> Result<ArrayData>;
pub type ReduceAxisArrayKernel = fn(ArrayData, usize) -> Result<ArrayData>;
pub type ArgReduceAllKernel = fn(ArrayData) -> Result<usize>;
pub type ArgReduceAxisKernel = fn(ArrayData, usize) -> Result<ArrayData>;

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

pub fn dot_kernel_for_dtype(dtype: DType, op: DotKernelOp) -> Option<DotArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Float32, DotKernelOp::Dot1d1d) => Some(dot_1d_1d_float32),
        (DType::Float64, DotKernelOp::Dot1d1d) => Some(dot_1d_1d_float64),
        (DType::Int32, DotKernelOp::Dot1d1d) => Some(dot_1d_1d_int32),
        (DType::Int64, DotKernelOp::Dot1d1d) => Some(dot_1d_1d_int64),
        (DType::Complex64, DotKernelOp::Dot1d1d) => Some(dot_1d_1d_complex64),
        (DType::Complex128, DotKernelOp::Dot1d1d) => Some(dot_1d_1d_complex128),
        (DType::Float32, DotKernelOp::MatMul2d2d) => Some(matmul_2d_2d_float32),
        (DType::Float64, DotKernelOp::MatMul2d2d) => Some(matmul_2d_2d_float64),
        (DType::Int32, DotKernelOp::MatMul2d2d) => Some(matmul_2d_2d_int32),
        (DType::Int64, DotKernelOp::MatMul2d2d) => Some(matmul_2d_2d_int64),
        (DType::Complex64, DotKernelOp::MatMul2d2d) => Some(matmul_2d_2d_complex64),
        (DType::Complex128, DotKernelOp::MatMul2d2d) => Some(matmul_2d_2d_complex128),
        (DType::Float32, DotKernelOp::MatMul2d1d) => Some(matmul_2d_1d_float32),
        (DType::Float64, DotKernelOp::MatMul2d1d) => Some(matmul_2d_1d_float64),
        (DType::Int32, DotKernelOp::MatMul2d1d) => Some(matmul_2d_1d_int32),
        (DType::Int64, DotKernelOp::MatMul2d1d) => Some(matmul_2d_1d_int64),
        (DType::Complex64, DotKernelOp::MatMul2d1d) => Some(matmul_2d_1d_complex64),
        (DType::Complex128, DotKernelOp::MatMul2d1d) => Some(matmul_2d_1d_complex128),
        _ => None,
    }
}

pub fn where_kernel_for_dtype(dtype: DType, op: WhereKernelOp) -> Option<WhereArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Bool, WhereKernelOp::Select) => Some(where_select_bool),
        (DType::Int32, WhereKernelOp::Select) => Some(where_select_int32),
        (DType::Int64, WhereKernelOp::Select) => Some(where_select_int64),
        (DType::Float32, WhereKernelOp::Select) => Some(where_select_float32),
        (DType::Float64, WhereKernelOp::Select) => Some(where_select_float64),
        (DType::Complex64, WhereKernelOp::Select) => Some(where_select_complex64),
        (DType::Complex128, WhereKernelOp::Select) => Some(where_select_complex128),
        (DType::Str, WhereKernelOp::Select) => Some(where_select_str),
        _ => None,
    }
}

pub fn predicate_kernel_for_dtype(
    dtype: DType,
    op: PredicateKernelOp,
) -> Option<PredicateArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Bool, PredicateKernelOp::IsNaN) => Some(isnan_bool),
        (DType::Bool, PredicateKernelOp::IsFinite) => Some(isfinite_bool),
        (DType::Bool, PredicateKernelOp::IsInf) => Some(isinf_bool),
        (DType::Int32, PredicateKernelOp::IsNaN) => Some(isnan_int32),
        (DType::Int32, PredicateKernelOp::IsFinite) => Some(isfinite_int32),
        (DType::Int32, PredicateKernelOp::IsInf) => Some(isinf_int32),
        (DType::Int64, PredicateKernelOp::IsNaN) => Some(isnan_int64),
        (DType::Int64, PredicateKernelOp::IsFinite) => Some(isfinite_int64),
        (DType::Int64, PredicateKernelOp::IsInf) => Some(isinf_int64),
        (DType::Float32, PredicateKernelOp::IsNaN) => Some(isnan_float32),
        (DType::Float32, PredicateKernelOp::IsFinite) => Some(isfinite_float32),
        (DType::Float32, PredicateKernelOp::IsInf) => Some(isinf_float32),
        (DType::Float64, PredicateKernelOp::IsNaN) => Some(isnan_float64),
        (DType::Float64, PredicateKernelOp::IsFinite) => Some(isfinite_float64),
        (DType::Float64, PredicateKernelOp::IsInf) => Some(isinf_float64),
        (DType::Complex64, PredicateKernelOp::IsNaN) => Some(isnan_complex64),
        (DType::Complex64, PredicateKernelOp::IsFinite) => Some(isfinite_complex64),
        (DType::Complex64, PredicateKernelOp::IsInf) => Some(isinf_complex64),
        (DType::Complex128, PredicateKernelOp::IsNaN) => Some(isnan_complex128),
        (DType::Complex128, PredicateKernelOp::IsFinite) => Some(isfinite_complex128),
        (DType::Complex128, PredicateKernelOp::IsInf) => Some(isinf_complex128),
        (DType::Str, PredicateKernelOp::IsNaN) => Some(isnan_str),
        (DType::Str, PredicateKernelOp::IsFinite) => Some(isfinite_str),
        (DType::Str, PredicateKernelOp::IsInf) => Some(isinf_str),
        _ => None,
    }
}

pub fn reduction_all_kernel_for_dtype(
    dtype: DType,
    op: ReductionKernelOp,
) -> Option<ReduceAllArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Int32, ReductionKernelOp::Sum) => Some(sum_all_int32),
        (DType::Int64, ReductionKernelOp::Sum) => Some(sum_all_int64),
        (DType::Float32, ReductionKernelOp::Sum) => Some(sum_all_float32),
        (DType::Float64, ReductionKernelOp::Sum) => Some(sum_all_float64),
        (DType::Complex64, ReductionKernelOp::Sum) => Some(sum_all_complex64),
        (DType::Complex128, ReductionKernelOp::Sum) => Some(sum_all_complex128),
        (DType::Float64, ReductionKernelOp::Prod) => Some(prod_all_float64),
        (DType::Bool, ReductionKernelOp::Min) => Some(min_all_bool),
        (DType::Bool, ReductionKernelOp::Max) => Some(max_all_bool),
        (DType::Int32, ReductionKernelOp::Min) => Some(min_all_int32),
        (DType::Int32, ReductionKernelOp::Max) => Some(max_all_int32),
        (DType::Int64, ReductionKernelOp::Min) => Some(min_all_int64),
        (DType::Int64, ReductionKernelOp::Max) => Some(max_all_int64),
        (DType::Float32, ReductionKernelOp::Min) => Some(min_all_float32),
        (DType::Float32, ReductionKernelOp::Max) => Some(max_all_float32),
        (DType::Float64, ReductionKernelOp::Min) => Some(min_all_float64),
        (DType::Float64, ReductionKernelOp::Max) => Some(max_all_float64),
        (DType::Complex64, ReductionKernelOp::Min) => Some(min_all_complex64),
        (DType::Complex64, ReductionKernelOp::Max) => Some(max_all_complex64),
        (DType::Complex128, ReductionKernelOp::Min) => Some(min_all_complex128),
        (DType::Complex128, ReductionKernelOp::Max) => Some(max_all_complex128),
        (DType::Str, ReductionKernelOp::Min) => Some(min_all_str),
        (DType::Str, ReductionKernelOp::Max) => Some(max_all_str),
        _ => None,
    }
}

pub fn reduction_axis_kernel_for_dtype(
    dtype: DType,
    op: ReductionKernelOp,
) -> Option<ReduceAxisArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Int32, ReductionKernelOp::Sum) => Some(sum_axis_int32),
        (DType::Int64, ReductionKernelOp::Sum) => Some(sum_axis_int64),
        (DType::Float32, ReductionKernelOp::Sum) => Some(sum_axis_float32),
        (DType::Float64, ReductionKernelOp::Sum) => Some(sum_axis_float64),
        (DType::Complex64, ReductionKernelOp::Sum) => Some(sum_axis_complex64),
        (DType::Complex128, ReductionKernelOp::Sum) => Some(sum_axis_complex128),
        (DType::Float64, ReductionKernelOp::Prod) => Some(prod_axis_float64),
        (DType::Bool, ReductionKernelOp::Min) => Some(min_axis_bool),
        (DType::Bool, ReductionKernelOp::Max) => Some(max_axis_bool),
        (DType::Int32, ReductionKernelOp::Min) => Some(min_axis_int32),
        (DType::Int32, ReductionKernelOp::Max) => Some(max_axis_int32),
        (DType::Int64, ReductionKernelOp::Min) => Some(min_axis_int64),
        (DType::Int64, ReductionKernelOp::Max) => Some(max_axis_int64),
        (DType::Float32, ReductionKernelOp::Min) => Some(min_axis_float32),
        (DType::Float32, ReductionKernelOp::Max) => Some(max_axis_float32),
        (DType::Float64, ReductionKernelOp::Min) => Some(min_axis_float64),
        (DType::Float64, ReductionKernelOp::Max) => Some(max_axis_float64),
        (DType::Complex64, ReductionKernelOp::Min) => Some(min_axis_complex64),
        (DType::Complex64, ReductionKernelOp::Max) => Some(max_axis_complex64),
        (DType::Complex128, ReductionKernelOp::Min) => Some(min_axis_complex128),
        (DType::Complex128, ReductionKernelOp::Max) => Some(max_axis_complex128),
        (DType::Str, ReductionKernelOp::Min) => Some(min_axis_str),
        (DType::Str, ReductionKernelOp::Max) => Some(max_axis_str),
        _ => None,
    }
}

pub fn arg_reduction_all_kernel_for_dtype(
    dtype: DType,
    op: ArgReductionKernelOp,
) -> Option<ArgReduceAllKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Float64, ArgReductionKernelOp::ArgMin) => Some(argmin_all_float64),
        (DType::Float64, ArgReductionKernelOp::ArgMax) => Some(argmax_all_float64),
        _ => None,
    }
}

pub fn arg_reduction_axis_kernel_for_dtype(
    dtype: DType,
    op: ArgReductionKernelOp,
) -> Option<ArgReduceAxisKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Float64, ArgReductionKernelOp::ArgMin) => Some(argmin_axis_float64),
        (DType::Float64, ArgReductionKernelOp::ArgMax) => Some(argmax_axis_float64),
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

macro_rules! dot_1d_1d_kernel {
    ($name:ident, $variant:ident) => {
        fn $name(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
            match (lhs, rhs) {
                (ArrayData::$variant(a), ArrayData::$variant(b)) => {
                    let s = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
                    Ok(ArrayData::$variant(
                        ArrayD::from_elem(IxDyn(&[]), s).into_shared(),
                    ))
                }
                _ => Err(NumpyError::TypeError("dot kernel dtype mismatch".into())),
            }
        }
    };
}

macro_rules! matmul_2d_2d_kernel {
    ($name:ident, $variant:ident) => {
        fn $name(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
            match (lhs, rhs) {
                (ArrayData::$variant(a), ArrayData::$variant(b)) => {
                    let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                    let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                    Ok(ArrayData::$variant(a2.dot(&b2).into_dyn().into_shared()))
                }
                _ => Err(NumpyError::TypeError("matmul kernel dtype mismatch".into())),
            }
        }
    };
}

macro_rules! matmul_2d_1d_kernel {
    ($name:ident, $variant:ident) => {
        fn $name(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
            match (lhs, rhs) {
                (ArrayData::$variant(a), ArrayData::$variant(b)) => {
                    let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                    let b1 = b.view().into_dimensionality::<ndarray::Ix1>().unwrap();
                    Ok(ArrayData::$variant(a2.dot(&b1).into_dyn().into_shared()))
                }
                _ => Err(NumpyError::TypeError("matvec kernel dtype mismatch".into())),
            }
        }
    };
}

dot_1d_1d_kernel!(dot_1d_1d_int32, Int32);
dot_1d_1d_kernel!(dot_1d_1d_int64, Int64);
dot_1d_1d_kernel!(dot_1d_1d_float32, Float32);
dot_1d_1d_kernel!(dot_1d_1d_float64, Float64);
dot_1d_1d_kernel!(dot_1d_1d_complex64, Complex64);
dot_1d_1d_kernel!(dot_1d_1d_complex128, Complex128);

matmul_2d_2d_kernel!(matmul_2d_2d_int32, Int32);
matmul_2d_2d_kernel!(matmul_2d_2d_int64, Int64);
matmul_2d_2d_kernel!(matmul_2d_2d_float32, Float32);
matmul_2d_2d_kernel!(matmul_2d_2d_float64, Float64);
matmul_2d_2d_kernel!(matmul_2d_2d_complex64, Complex64);
matmul_2d_2d_kernel!(matmul_2d_2d_complex128, Complex128);

matmul_2d_1d_kernel!(matmul_2d_1d_int32, Int32);
matmul_2d_1d_kernel!(matmul_2d_1d_int64, Int64);
matmul_2d_1d_kernel!(matmul_2d_1d_float32, Float32);
matmul_2d_1d_kernel!(matmul_2d_1d_float64, Float64);
matmul_2d_1d_kernel!(matmul_2d_1d_complex64, Complex64);
matmul_2d_1d_kernel!(matmul_2d_1d_complex128, Complex128);

macro_rules! primitive_where_kernel {
    ($name:ident, $variant:ident) => {
        fn $name(cond: ArrayData, x: ArrayData, y: ArrayData) -> Result<ArrayData> {
            match (cond, x, y) {
                (ArrayData::Bool(cond), ArrayData::$variant(x), ArrayData::$variant(y)) => {
                    Ok(ArrayData::$variant(
                        ndarray::Zip::from(&cond)
                            .and(&x)
                            .and(&y)
                            .map_collect(|&c, &xv, &yv| if c { xv } else { yv })
                            .into_shared(),
                    ))
                }
                _ => Err(NumpyError::TypeError("where kernel dtype mismatch".into())),
            }
        }
    };
}

fn where_select_str(cond: ArrayData, x: ArrayData, y: ArrayData) -> Result<ArrayData> {
    match (cond, x, y) {
        (ArrayData::Bool(cond), ArrayData::Str(x), ArrayData::Str(y)) => Ok(ArrayData::Str(
            ndarray::Zip::from(&cond)
                .and(&x)
                .and(&y)
                .map_collect(|&c, xv, yv| if c { xv.clone() } else { yv.clone() })
                .into_shared(),
        )),
        _ => Err(NumpyError::TypeError("where kernel dtype mismatch".into())),
    }
}

primitive_where_kernel!(where_select_bool, Bool);
primitive_where_kernel!(where_select_int32, Int32);
primitive_where_kernel!(where_select_int64, Int64);
primitive_where_kernel!(where_select_float32, Float32);
primitive_where_kernel!(where_select_float64, Float64);
primitive_where_kernel!(where_select_complex64, Complex64);
primitive_where_kernel!(where_select_complex128, Complex128);

macro_rules! simple_predicate_kernel {
    ($name:ident, $variant:ident, $predicate:expr) => {
        fn $name(input: ArrayData) -> Result<ArrayData> {
            match input {
                ArrayData::$variant(data) => {
                    Ok(ArrayData::Bool(data.mapv($predicate).into_shared()))
                }
                _ => Err(NumpyError::TypeError(
                    "predicate kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

macro_rules! constant_predicate_kernel {
    ($name:ident, $variant:ident, $value:expr) => {
        fn $name(input: ArrayData) -> Result<ArrayData> {
            match input {
                ArrayData::$variant(data) => Ok(ArrayData::Bool(
                    ArrayD::from_elem(IxDyn(data.shape()), $value).into_shared(),
                )),
                _ => Err(NumpyError::TypeError(
                    "predicate kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

constant_predicate_kernel!(isnan_bool, Bool, false);
constant_predicate_kernel!(isfinite_bool, Bool, true);
constant_predicate_kernel!(isinf_bool, Bool, false);
constant_predicate_kernel!(isnan_int32, Int32, false);
constant_predicate_kernel!(isfinite_int32, Int32, true);
constant_predicate_kernel!(isinf_int32, Int32, false);
constant_predicate_kernel!(isnan_int64, Int64, false);
constant_predicate_kernel!(isfinite_int64, Int64, true);
constant_predicate_kernel!(isinf_int64, Int64, false);
simple_predicate_kernel!(isnan_float32, Float32, |x: f32| x.is_nan());
simple_predicate_kernel!(isfinite_float32, Float32, |x: f32| x.is_finite());
simple_predicate_kernel!(isinf_float32, Float32, |x: f32| x.is_infinite());
simple_predicate_kernel!(isnan_float64, Float64, |x: f64| x.is_nan());
simple_predicate_kernel!(isfinite_float64, Float64, |x: f64| x.is_finite());
simple_predicate_kernel!(isinf_float64, Float64, |x: f64| x.is_infinite());
simple_predicate_kernel!(isnan_complex64, Complex64, |x: Complex<f32>| x.re.is_nan()
    || x.im.is_nan());
simple_predicate_kernel!(isfinite_complex64, Complex64, |x: Complex<f32>| x
    .re
    .is_finite()
    && x.im.is_finite());
simple_predicate_kernel!(isinf_complex64, Complex64, |x: Complex<f32>| x
    .re
    .is_infinite()
    || x.im.is_infinite());
simple_predicate_kernel!(isnan_complex128, Complex128, |x: Complex<f64>| x
    .re
    .is_nan()
    || x.im.is_nan());
simple_predicate_kernel!(isfinite_complex128, Complex128, |x: Complex<f64>| x
    .re
    .is_finite()
    && x.im.is_finite());
simple_predicate_kernel!(isinf_complex128, Complex128, |x: Complex<f64>| x
    .re
    .is_infinite()
    || x.im.is_infinite());
constant_predicate_kernel!(isnan_str, Str, false);
constant_predicate_kernel!(isfinite_str, Str, true);
constant_predicate_kernel!(isinf_str, Str, false);

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

fn prod_all_float64(data: ArrayData) -> Result<ArrayData> {
    match data {
        ArrayData::Float64(a) => Ok(ArrayData::Float64(
            ArrayD::from_elem(IxDyn(&[]), a.iter().product()).into_shared(),
        )),
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

fn prod_axis_float64(data: ArrayData, axis: usize) -> Result<ArrayData> {
    match data {
        ArrayData::Float64(a) => Ok(ArrayData::Float64(
            a.fold_axis(Axis(axis), 1.0, |&acc, &x| acc * x)
                .into_shared(),
        )),
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

macro_rules! scalar_all_ord_kernel {
    ($name:ident, $variant:ident, $expr:expr) => {
        fn $name(data: ArrayData) -> Result<ArrayData> {
            match data {
                ArrayData::$variant(a) => {
                    let v = $expr(a).ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                    Ok(ArrayData::$variant(
                        ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
                    ))
                }
                _ => Err(NumpyError::TypeError(
                    "reduction kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

scalar_all_ord_kernel!(min_all_bool, Bool, |a: ndarray::ArrayBase<
    ndarray::OwnedArcRepr<bool>,
    IxDyn,
>| a.iter().min().copied());
scalar_all_ord_kernel!(max_all_bool, Bool, |a: ndarray::ArrayBase<
    ndarray::OwnedArcRepr<bool>,
    IxDyn,
>| a.iter().max().copied());
scalar_all_ord_kernel!(min_all_int32, Int32, |a: ndarray::ArrayBase<
    ndarray::OwnedArcRepr<i32>,
    IxDyn,
>| a.iter().min().copied());
scalar_all_ord_kernel!(max_all_int32, Int32, |a: ndarray::ArrayBase<
    ndarray::OwnedArcRepr<i32>,
    IxDyn,
>| a.iter().max().copied());
scalar_all_ord_kernel!(min_all_int64, Int64, |a: ndarray::ArrayBase<
    ndarray::OwnedArcRepr<i64>,
    IxDyn,
>| a.iter().min().copied());
scalar_all_ord_kernel!(max_all_int64, Int64, |a: ndarray::ArrayBase<
    ndarray::OwnedArcRepr<i64>,
    IxDyn,
>| a.iter().max().copied());

fn min_all_float32(data: ArrayData) -> Result<ArrayData> {
    match data {
        ArrayData::Float32(a) => {
            let v = a
                .iter()
                .copied()
                .reduce(f32::min)
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
            Ok(ArrayData::Float32(
                ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
            ))
        }
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

fn max_all_float32(data: ArrayData) -> Result<ArrayData> {
    match data {
        ArrayData::Float32(a) => {
            let v = a
                .iter()
                .copied()
                .reduce(f32::max)
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
            Ok(ArrayData::Float32(
                ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
            ))
        }
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

fn min_all_float64(data: ArrayData) -> Result<ArrayData> {
    match data {
        ArrayData::Float64(a) => {
            let v = a
                .iter()
                .copied()
                .reduce(f64::min)
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
            Ok(ArrayData::Float64(
                ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
            ))
        }
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

fn max_all_float64(data: ArrayData) -> Result<ArrayData> {
    match data {
        ArrayData::Float64(a) => {
            let v = a
                .iter()
                .copied()
                .reduce(f64::max)
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
            Ok(ArrayData::Float64(
                ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
            ))
        }
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

macro_rules! complex_all_kernel {
    ($name:ident, $variant:ident, $keep_acc_when:pat) => {
        fn $name(data: ArrayData) -> Result<ArrayData> {
            match data {
                ArrayData::$variant(a) => {
                    let v = a
                        .iter()
                        .copied()
                        .reduce(|acc, x| {
                            if matches!(complex_cmp(&acc, &x), $keep_acc_when) {
                                acc
                            } else {
                                x
                            }
                        })
                        .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                    Ok(ArrayData::$variant(
                        ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
                    ))
                }
                _ => Err(NumpyError::TypeError(
                    "reduction kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

complex_all_kernel!(
    min_all_complex64,
    Complex64,
    std::cmp::Ordering::Less | std::cmp::Ordering::Equal
);
complex_all_kernel!(
    max_all_complex64,
    Complex64,
    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
);
complex_all_kernel!(
    min_all_complex128,
    Complex128,
    std::cmp::Ordering::Less | std::cmp::Ordering::Equal
);
complex_all_kernel!(
    max_all_complex128,
    Complex128,
    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
);

fn min_all_str(data: ArrayData) -> Result<ArrayData> {
    match data {
        ArrayData::Str(a) => {
            let v = a
                .iter()
                .min()
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?
                .clone();
            Ok(ArrayData::Str(
                ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
            ))
        }
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

fn max_all_str(data: ArrayData) -> Result<ArrayData> {
    match data {
        ArrayData::Str(a) => {
            let v = a
                .iter()
                .max()
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?
                .clone();
            Ok(ArrayData::Str(
                ArrayD::from_elem(IxDyn(&[]), v).into_shared(),
            ))
        }
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

macro_rules! fold_axis_ord_kernel {
    ($name:ident, $variant:ident, $init:expr, $fold:expr) => {
        fn $name(data: ArrayData, axis: usize) -> Result<ArrayData> {
            match data {
                ArrayData::$variant(a) => Ok(ArrayData::$variant(
                    a.fold_axis(Axis(axis), $init, $fold).into_shared(),
                )),
                _ => Err(NumpyError::TypeError(
                    "reduction kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

fold_axis_ord_kernel!(min_axis_bool, Bool, true, |&acc, &x| acc && x);
fold_axis_ord_kernel!(max_axis_bool, Bool, false, |&acc, &x| acc || x);
fold_axis_ord_kernel!(min_axis_int32, Int32, i32::MAX, |&acc, &x| acc.min(x));
fold_axis_ord_kernel!(max_axis_int32, Int32, i32::MIN, |&acc, &x| acc.max(x));
fold_axis_ord_kernel!(min_axis_int64, Int64, i64::MAX, |&acc, &x| acc.min(x));
fold_axis_ord_kernel!(max_axis_int64, Int64, i64::MIN, |&acc, &x| acc.max(x));
fold_axis_ord_kernel!(min_axis_float32, Float32, f32::INFINITY, |&acc, &x| acc
    .min(x));
fold_axis_ord_kernel!(max_axis_float32, Float32, f32::NEG_INFINITY, |&acc, &x| acc
    .max(x));
fold_axis_ord_kernel!(min_axis_float64, Float64, f64::INFINITY, |&acc, &x| acc
    .min(x));
fold_axis_ord_kernel!(max_axis_float64, Float64, f64::NEG_INFINITY, |&acc, &x| acc
    .max(x));

fn min_axis_str(data: ArrayData, axis: usize) -> Result<ArrayData> {
    match data {
        ArrayData::Str(a) => Ok(ArrayData::Str(
            a.fold_axis(Axis(axis), String::from("\u{10FFFF}"), |acc, x| {
                if x < acc {
                    x.clone()
                } else {
                    acc.clone()
                }
            })
            .into_shared(),
        )),
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

fn max_axis_str(data: ArrayData, axis: usize) -> Result<ArrayData> {
    match data {
        ArrayData::Str(a) => Ok(ArrayData::Str(
            a.fold_axis(Axis(axis), String::new(), |acc, x| {
                if x > acc {
                    x.clone()
                } else {
                    acc.clone()
                }
            })
            .into_shared(),
        )),
        _ => Err(NumpyError::TypeError(
            "reduction kernel dtype mismatch".into(),
        )),
    }
}

macro_rules! complex_axis_kernel {
    ($name:ident, $variant:ident, $init:expr, $keep_acc_when:pat) => {
        fn $name(data: ArrayData, axis: usize) -> Result<ArrayData> {
            match data {
                ArrayData::$variant(a) => Ok(ArrayData::$variant(
                    a.fold_axis(Axis(axis), $init, |&acc, &x| {
                        if matches!(complex_cmp(&acc, &x), $keep_acc_when) {
                            acc
                        } else {
                            x
                        }
                    })
                    .into_shared(),
                )),
                _ => Err(NumpyError::TypeError(
                    "reduction kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

complex_axis_kernel!(
    min_axis_complex64,
    Complex64,
    Complex::new(f32::INFINITY, 0.0),
    std::cmp::Ordering::Less | std::cmp::Ordering::Equal
);
complex_axis_kernel!(
    max_axis_complex64,
    Complex64,
    Complex::new(f32::NEG_INFINITY, 0.0),
    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
);
complex_axis_kernel!(
    min_axis_complex128,
    Complex128,
    Complex::new(f64::INFINITY, 0.0),
    std::cmp::Ordering::Less | std::cmp::Ordering::Equal
);
complex_axis_kernel!(
    max_axis_complex128,
    Complex128,
    Complex::new(f64::NEG_INFINITY, 0.0),
    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
);

fn argmin_all_float64(data: ArrayData) -> Result<usize> {
    match data {
        ArrayData::Float64(a) => a
            .iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| NumpyError::ValueError("empty array".into())),
        _ => Err(NumpyError::TypeError(
            "arg reduction kernel dtype mismatch".into(),
        )),
    }
}

fn argmax_all_float64(data: ArrayData) -> Result<usize> {
    match data {
        ArrayData::Float64(a) => a
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| NumpyError::ValueError("empty array".into())),
        _ => Err(NumpyError::TypeError(
            "arg reduction kernel dtype mismatch".into(),
        )),
    }
}

macro_rules! arg_axis_kernel {
    ($name:ident, $cmp:tt, $init:expr) => {
        fn $name(data: ArrayData, axis: usize) -> Result<ArrayData> {
            match data {
                ArrayData::Float64(arr) => {
                    let ax = Axis(axis);
                    let mut result_shape = arr.shape().to_vec();
                    result_shape.remove(axis);
                    let result_dim = if result_shape.is_empty() {
                        IxDyn(&[])
                    } else {
                        IxDyn(&result_shape)
                    };
                    let mut result = ArrayD::<i64>::zeros(result_dim);
                    for (lane_in, result_elem) in arr.lanes(ax).into_iter().zip(result.iter_mut()) {
                        let mut idx: usize = 0;
                        let mut val = $init;
                        for (i, &v) in lane_in.iter().enumerate() {
                            if v $cmp val {
                                val = v;
                                idx = i;
                            }
                        }
                        *result_elem = idx as i64;
                    }
                    Ok(ArrayData::Int64(result.into_shared()))
                }
                _ => Err(NumpyError::TypeError("arg reduction kernel dtype mismatch".into())),
            }
        }
    };
}

arg_axis_kernel!(argmin_axis_float64, <, f64::INFINITY);
arg_axis_kernel!(argmax_axis_float64, >, f64::NEG_INFINITY);
