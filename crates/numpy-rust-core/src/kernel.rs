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
pub enum PredicatePresenceOp {
    HasNaN,
    HasInf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueUnaryKernelOp {
    SignBit,
    Sign,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealUnaryKernelOp {
    Spacing,
    I0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealBinaryKernelOp {
    ArcTan2,
    LDExp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathUnaryKernelOp {
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Log10,
    Log2,
    Sinh,
    Cosh,
    Tanh,
    ArcSin,
    ArcCos,
    ArcTan,
    ArcSinh,
    ArcCosh,
    ArcTanh,
    Floor,
    Ceil,
    Round,
    Log1p,
    Expm1,
    Deg2Rad,
    Rad2Deg,
    Trunc,
    Cbrt,
    Gamma,
    LGamma,
    Erf,
    Erfc,
    J0,
    J1,
    Y0,
    Y1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBinaryKernelOp {
    CopySign,
    Hypot,
    FMod,
    NextAfter,
    LogAddExp,
    LogAddExp2,
    FMax,
    FMin,
    Maximum,
    Minimum,
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
pub type PredicatePresenceKernel = fn(&ArrayData) -> Result<bool>;
pub type UnaryArrayKernel = fn(ArrayData) -> Result<ArrayData>;
pub type BinaryMathArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
pub type RealBinaryArrayKernel = fn(ArrayData, ArrayData) -> Result<ArrayData>;
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

pub fn predicate_presence_kernel_for_dtype(
    dtype: DType,
    op: PredicatePresenceOp,
) -> Option<PredicatePresenceKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Bool, PredicatePresenceOp::HasNaN) => Some(has_nan_bool),
        (DType::Bool, PredicatePresenceOp::HasInf) => Some(has_inf_bool),
        (DType::Int32, PredicatePresenceOp::HasNaN) => Some(has_nan_int32),
        (DType::Int32, PredicatePresenceOp::HasInf) => Some(has_inf_int32),
        (DType::Int64, PredicatePresenceOp::HasNaN) => Some(has_nan_int64),
        (DType::Int64, PredicatePresenceOp::HasInf) => Some(has_inf_int64),
        (DType::Float32, PredicatePresenceOp::HasNaN) => Some(has_nan_float32),
        (DType::Float32, PredicatePresenceOp::HasInf) => Some(has_inf_float32),
        (DType::Float64, PredicatePresenceOp::HasNaN) => Some(has_nan_float64),
        (DType::Float64, PredicatePresenceOp::HasInf) => Some(has_inf_float64),
        (DType::Complex64, PredicatePresenceOp::HasNaN) => Some(has_nan_complex64),
        (DType::Complex64, PredicatePresenceOp::HasInf) => Some(has_inf_complex64),
        (DType::Complex128, PredicatePresenceOp::HasNaN) => Some(has_nan_complex128),
        (DType::Complex128, PredicatePresenceOp::HasInf) => Some(has_inf_complex128),
        (DType::Str, PredicatePresenceOp::HasNaN) => Some(has_nan_str),
        (DType::Str, PredicatePresenceOp::HasInf) => Some(has_inf_str),
        _ => None,
    }
}

pub fn math_unary_kernel_for_dtype(
    dtype: DType,
    op: MathUnaryKernelOp,
) -> Option<UnaryArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Float32, MathUnaryKernelOp::Sqrt) => Some(unary_sqrt_float32),
        (DType::Float64, MathUnaryKernelOp::Sqrt) => Some(unary_sqrt_float64),
        (DType::Complex64, MathUnaryKernelOp::Sqrt) => Some(unary_sqrt_complex64),
        (DType::Complex128, MathUnaryKernelOp::Sqrt) => Some(unary_sqrt_complex128),
        (DType::Float32, MathUnaryKernelOp::Exp) => Some(unary_exp_float32),
        (DType::Float64, MathUnaryKernelOp::Exp) => Some(unary_exp_float64),
        (DType::Complex64, MathUnaryKernelOp::Exp) => Some(unary_exp_complex64),
        (DType::Complex128, MathUnaryKernelOp::Exp) => Some(unary_exp_complex128),
        (DType::Float32, MathUnaryKernelOp::Log) => Some(unary_log_float32),
        (DType::Float64, MathUnaryKernelOp::Log) => Some(unary_log_float64),
        (DType::Complex64, MathUnaryKernelOp::Log) => Some(unary_log_complex64),
        (DType::Complex128, MathUnaryKernelOp::Log) => Some(unary_log_complex128),
        (DType::Float32, MathUnaryKernelOp::Sin) => Some(unary_sin_float32),
        (DType::Float64, MathUnaryKernelOp::Sin) => Some(unary_sin_float64),
        (DType::Complex64, MathUnaryKernelOp::Sin) => Some(unary_sin_complex64),
        (DType::Complex128, MathUnaryKernelOp::Sin) => Some(unary_sin_complex128),
        (DType::Float32, MathUnaryKernelOp::Cos) => Some(unary_cos_float32),
        (DType::Float64, MathUnaryKernelOp::Cos) => Some(unary_cos_float64),
        (DType::Complex64, MathUnaryKernelOp::Cos) => Some(unary_cos_complex64),
        (DType::Complex128, MathUnaryKernelOp::Cos) => Some(unary_cos_complex128),
        (DType::Float32, MathUnaryKernelOp::Tan) => Some(unary_tan_float32),
        (DType::Float64, MathUnaryKernelOp::Tan) => Some(unary_tan_float64),
        (DType::Complex64, MathUnaryKernelOp::Tan) => Some(unary_tan_complex64),
        (DType::Complex128, MathUnaryKernelOp::Tan) => Some(unary_tan_complex128),
        (DType::Float32, MathUnaryKernelOp::Log10) => Some(unary_log10_float32),
        (DType::Float64, MathUnaryKernelOp::Log10) => Some(unary_log10_float64),
        (DType::Complex64, MathUnaryKernelOp::Log10) => Some(unary_log10_complex64),
        (DType::Complex128, MathUnaryKernelOp::Log10) => Some(unary_log10_complex128),
        (DType::Float32, MathUnaryKernelOp::Log2) => Some(unary_log2_float32),
        (DType::Float64, MathUnaryKernelOp::Log2) => Some(unary_log2_float64),
        (DType::Complex64, MathUnaryKernelOp::Log2) => Some(unary_log2_complex64),
        (DType::Complex128, MathUnaryKernelOp::Log2) => Some(unary_log2_complex128),
        (DType::Float32, MathUnaryKernelOp::Sinh) => Some(unary_sinh_float32),
        (DType::Float64, MathUnaryKernelOp::Sinh) => Some(unary_sinh_float64),
        (DType::Complex64, MathUnaryKernelOp::Sinh) => Some(unary_sinh_complex64),
        (DType::Complex128, MathUnaryKernelOp::Sinh) => Some(unary_sinh_complex128),
        (DType::Float32, MathUnaryKernelOp::Cosh) => Some(unary_cosh_float32),
        (DType::Float64, MathUnaryKernelOp::Cosh) => Some(unary_cosh_float64),
        (DType::Complex64, MathUnaryKernelOp::Cosh) => Some(unary_cosh_complex64),
        (DType::Complex128, MathUnaryKernelOp::Cosh) => Some(unary_cosh_complex128),
        (DType::Float32, MathUnaryKernelOp::Tanh) => Some(unary_tanh_float32),
        (DType::Float64, MathUnaryKernelOp::Tanh) => Some(unary_tanh_float64),
        (DType::Complex64, MathUnaryKernelOp::Tanh) => Some(unary_tanh_complex64),
        (DType::Complex128, MathUnaryKernelOp::Tanh) => Some(unary_tanh_complex128),
        (DType::Float32, MathUnaryKernelOp::ArcSin) => Some(unary_arcsin_float32),
        (DType::Float64, MathUnaryKernelOp::ArcSin) => Some(unary_arcsin_float64),
        (DType::Complex64, MathUnaryKernelOp::ArcSin) => Some(unary_arcsin_complex64),
        (DType::Complex128, MathUnaryKernelOp::ArcSin) => Some(unary_arcsin_complex128),
        (DType::Float32, MathUnaryKernelOp::ArcCos) => Some(unary_arccos_float32),
        (DType::Float64, MathUnaryKernelOp::ArcCos) => Some(unary_arccos_float64),
        (DType::Complex64, MathUnaryKernelOp::ArcCos) => Some(unary_arccos_complex64),
        (DType::Complex128, MathUnaryKernelOp::ArcCos) => Some(unary_arccos_complex128),
        (DType::Float32, MathUnaryKernelOp::ArcTan) => Some(unary_arctan_float32),
        (DType::Float64, MathUnaryKernelOp::ArcTan) => Some(unary_arctan_float64),
        (DType::Complex64, MathUnaryKernelOp::ArcTan) => Some(unary_arctan_complex64),
        (DType::Complex128, MathUnaryKernelOp::ArcTan) => Some(unary_arctan_complex128),
        (DType::Float32, MathUnaryKernelOp::ArcSinh) => Some(unary_arcsinh_float32),
        (DType::Float64, MathUnaryKernelOp::ArcSinh) => Some(unary_arcsinh_float64),
        (DType::Complex64, MathUnaryKernelOp::ArcSinh) => Some(unary_arcsinh_complex64),
        (DType::Complex128, MathUnaryKernelOp::ArcSinh) => Some(unary_arcsinh_complex128),
        (DType::Float32, MathUnaryKernelOp::ArcCosh) => Some(unary_arccosh_float32),
        (DType::Float64, MathUnaryKernelOp::ArcCosh) => Some(unary_arccosh_float64),
        (DType::Complex64, MathUnaryKernelOp::ArcCosh) => Some(unary_arccosh_complex64),
        (DType::Complex128, MathUnaryKernelOp::ArcCosh) => Some(unary_arccosh_complex128),
        (DType::Float32, MathUnaryKernelOp::ArcTanh) => Some(unary_arctanh_float32),
        (DType::Float64, MathUnaryKernelOp::ArcTanh) => Some(unary_arctanh_float64),
        (DType::Complex64, MathUnaryKernelOp::ArcTanh) => Some(unary_arctanh_complex64),
        (DType::Complex128, MathUnaryKernelOp::ArcTanh) => Some(unary_arctanh_complex128),
        (DType::Float32, MathUnaryKernelOp::Floor) => Some(unary_floor_float32),
        (DType::Float64, MathUnaryKernelOp::Floor) => Some(unary_floor_float64),
        (DType::Float32, MathUnaryKernelOp::Ceil) => Some(unary_ceil_float32),
        (DType::Float64, MathUnaryKernelOp::Ceil) => Some(unary_ceil_float64),
        (DType::Float32, MathUnaryKernelOp::Round) => Some(unary_round_float32),
        (DType::Float64, MathUnaryKernelOp::Round) => Some(unary_round_float64),
        (DType::Float32, MathUnaryKernelOp::Log1p) => Some(unary_log1p_float32),
        (DType::Float64, MathUnaryKernelOp::Log1p) => Some(unary_log1p_float64),
        (DType::Float32, MathUnaryKernelOp::Expm1) => Some(unary_expm1_float32),
        (DType::Float64, MathUnaryKernelOp::Expm1) => Some(unary_expm1_float64),
        (DType::Float32, MathUnaryKernelOp::Deg2Rad) => Some(unary_deg2rad_float32),
        (DType::Float64, MathUnaryKernelOp::Deg2Rad) => Some(unary_deg2rad_float64),
        (DType::Float32, MathUnaryKernelOp::Rad2Deg) => Some(unary_rad2deg_float32),
        (DType::Float64, MathUnaryKernelOp::Rad2Deg) => Some(unary_rad2deg_float64),
        (DType::Float32, MathUnaryKernelOp::Trunc) => Some(unary_trunc_float32),
        (DType::Float64, MathUnaryKernelOp::Trunc) => Some(unary_trunc_float64),
        (DType::Float32, MathUnaryKernelOp::Cbrt) => Some(unary_cbrt_float32),
        (DType::Float64, MathUnaryKernelOp::Cbrt) => Some(unary_cbrt_float64),
        (DType::Float32, MathUnaryKernelOp::Gamma) => Some(unary_gamma_float32),
        (DType::Float64, MathUnaryKernelOp::Gamma) => Some(unary_gamma_float64),
        (DType::Float32, MathUnaryKernelOp::LGamma) => Some(unary_lgamma_float32),
        (DType::Float64, MathUnaryKernelOp::LGamma) => Some(unary_lgamma_float64),
        (DType::Float32, MathUnaryKernelOp::Erf) => Some(unary_erf_float32),
        (DType::Float64, MathUnaryKernelOp::Erf) => Some(unary_erf_float64),
        (DType::Float32, MathUnaryKernelOp::Erfc) => Some(unary_erfc_float32),
        (DType::Float64, MathUnaryKernelOp::Erfc) => Some(unary_erfc_float64),
        (DType::Float32, MathUnaryKernelOp::J0) => Some(unary_j0_float32),
        (DType::Float64, MathUnaryKernelOp::J0) => Some(unary_j0_float64),
        (DType::Float32, MathUnaryKernelOp::J1) => Some(unary_j1_float32),
        (DType::Float64, MathUnaryKernelOp::J1) => Some(unary_j1_float64),
        (DType::Float32, MathUnaryKernelOp::Y0) => Some(unary_y0_float32),
        (DType::Float64, MathUnaryKernelOp::Y0) => Some(unary_y0_float64),
        (DType::Float32, MathUnaryKernelOp::Y1) => Some(unary_y1_float32),
        (DType::Float64, MathUnaryKernelOp::Y1) => Some(unary_y1_float64),
        _ => None,
    }
}

pub fn value_unary_kernel_for_dtype(
    dtype: DType,
    op: ValueUnaryKernelOp,
) -> Option<UnaryArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Bool, ValueUnaryKernelOp::SignBit) => Some(signbit_bool),
        (DType::Int32, ValueUnaryKernelOp::SignBit) => Some(signbit_int32),
        (DType::Int64, ValueUnaryKernelOp::SignBit) => Some(signbit_int64),
        (DType::Float32, ValueUnaryKernelOp::SignBit) => Some(signbit_float32),
        (DType::Float64, ValueUnaryKernelOp::SignBit) => Some(signbit_float64),
        (DType::Complex64, ValueUnaryKernelOp::SignBit) => Some(signbit_complex64),
        (DType::Complex128, ValueUnaryKernelOp::SignBit) => Some(signbit_complex128),
        (DType::Str, ValueUnaryKernelOp::SignBit) => Some(signbit_str),
        (DType::Bool, ValueUnaryKernelOp::Sign) => Some(sign_bool),
        (DType::Int32, ValueUnaryKernelOp::Sign) => Some(sign_int32),
        (DType::Int64, ValueUnaryKernelOp::Sign) => Some(sign_int64),
        (DType::Float32, ValueUnaryKernelOp::Sign) => Some(sign_float32),
        (DType::Float64, ValueUnaryKernelOp::Sign) => Some(sign_float64),
        (DType::Complex64, ValueUnaryKernelOp::Sign) => Some(sign_complex64),
        (DType::Complex128, ValueUnaryKernelOp::Sign) => Some(sign_complex128),
        (DType::Bool, ValueUnaryKernelOp::Neg) => Some(neg_bool),
        (DType::Int32, ValueUnaryKernelOp::Neg) => Some(neg_int32),
        (DType::Int64, ValueUnaryKernelOp::Neg) => Some(neg_int64),
        (DType::Float32, ValueUnaryKernelOp::Neg) => Some(neg_float32),
        (DType::Float64, ValueUnaryKernelOp::Neg) => Some(neg_float64),
        (DType::Complex64, ValueUnaryKernelOp::Neg) => Some(neg_complex64),
        (DType::Complex128, ValueUnaryKernelOp::Neg) => Some(neg_complex128),
        _ => None,
    }
}

pub fn real_unary_kernel_for_dtype(
    dtype: DType,
    op: RealUnaryKernelOp,
) -> Option<UnaryArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Float32, RealUnaryKernelOp::Spacing) => Some(real_spacing_float32),
        (DType::Float64, RealUnaryKernelOp::Spacing) => Some(real_spacing_float64),
        (DType::Float32, RealUnaryKernelOp::I0) => Some(real_i0_float32),
        (DType::Float64, RealUnaryKernelOp::I0) => Some(real_i0_float64),
        _ => None,
    }
}

pub fn real_binary_kernel_for_dtype(
    dtype: DType,
    op: RealBinaryKernelOp,
) -> Option<RealBinaryArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Float32, RealBinaryKernelOp::ArcTan2) => Some(real_arctan2_float32),
        (DType::Float64, RealBinaryKernelOp::ArcTan2) => Some(real_arctan2_float64),
        (DType::Float32, RealBinaryKernelOp::LDExp) => Some(real_ldexp_float32),
        (DType::Float64, RealBinaryKernelOp::LDExp) => Some(real_ldexp_float64),
        _ => None,
    }
}

pub fn math_binary_kernel_for_dtype(
    dtype: DType,
    op: MathBinaryKernelOp,
) -> Option<BinaryMathArrayKernel> {
    match (dtype.storage_dtype(), op) {
        (DType::Float32, MathBinaryKernelOp::CopySign) => Some(binary_copysign_float32),
        (DType::Float64, MathBinaryKernelOp::CopySign) => Some(binary_copysign_float64),
        (DType::Float32, MathBinaryKernelOp::Hypot) => Some(binary_hypot_float32),
        (DType::Float64, MathBinaryKernelOp::Hypot) => Some(binary_hypot_float64),
        (DType::Float32, MathBinaryKernelOp::FMod) => Some(binary_fmod_float32),
        (DType::Float64, MathBinaryKernelOp::FMod) => Some(binary_fmod_float64),
        (DType::Float32, MathBinaryKernelOp::NextAfter) => Some(binary_nextafter_float32),
        (DType::Float64, MathBinaryKernelOp::NextAfter) => Some(binary_nextafter_float64),
        (DType::Float32, MathBinaryKernelOp::LogAddExp) => Some(binary_logaddexp_float32),
        (DType::Float64, MathBinaryKernelOp::LogAddExp) => Some(binary_logaddexp_float64),
        (DType::Float32, MathBinaryKernelOp::LogAddExp2) => Some(binary_logaddexp2_float32),
        (DType::Float64, MathBinaryKernelOp::LogAddExp2) => Some(binary_logaddexp2_float64),
        (DType::Float32, MathBinaryKernelOp::FMax) => Some(binary_fmax_float32),
        (DType::Float64, MathBinaryKernelOp::FMax) => Some(binary_fmax_float64),
        (DType::Float32, MathBinaryKernelOp::FMin) => Some(binary_fmin_float32),
        (DType::Float64, MathBinaryKernelOp::FMin) => Some(binary_fmin_float64),
        (DType::Float32, MathBinaryKernelOp::Maximum) => Some(binary_maximum_float32),
        (DType::Float64, MathBinaryKernelOp::Maximum) => Some(binary_maximum_float64),
        (DType::Float32, MathBinaryKernelOp::Minimum) => Some(binary_minimum_float32),
        (DType::Float64, MathBinaryKernelOp::Minimum) => Some(binary_minimum_float64),
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

macro_rules! constant_presence_kernel {
    ($name:ident, $variant:ident, $value:expr) => {
        fn $name(input: &ArrayData) -> Result<bool> {
            match input {
                ArrayData::$variant(_) => Ok($value),
                _ => Err(NumpyError::TypeError(
                    "predicate presence kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

macro_rules! iter_presence_kernel {
    ($name:ident, $variant:ident, |$x:ident| $body:expr) => {
        fn $name(input: &ArrayData) -> Result<bool> {
            match input {
                ArrayData::$variant(data) => Ok(data.iter().any(|$x| $body)),
                _ => Err(NumpyError::TypeError(
                    "predicate presence kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

constant_presence_kernel!(has_nan_bool, Bool, false);
constant_presence_kernel!(has_inf_bool, Bool, false);
constant_presence_kernel!(has_nan_int32, Int32, false);
constant_presence_kernel!(has_inf_int32, Int32, false);
constant_presence_kernel!(has_nan_int64, Int64, false);
constant_presence_kernel!(has_inf_int64, Int64, false);
iter_presence_kernel!(has_nan_float32, Float32, |x| x.is_nan());
iter_presence_kernel!(has_inf_float32, Float32, |x| x.is_infinite());
iter_presence_kernel!(has_nan_float64, Float64, |x| x.is_nan());
iter_presence_kernel!(has_inf_float64, Float64, |x| x.is_infinite());
iter_presence_kernel!(has_nan_complex64, Complex64, |x| x.re.is_nan()
    || x.im.is_nan());
iter_presence_kernel!(has_inf_complex64, Complex64, |x| x.re.is_infinite()
    || x.im.is_infinite());
iter_presence_kernel!(has_nan_complex128, Complex128, |x| x.re.is_nan()
    || x.im.is_nan());
iter_presence_kernel!(has_inf_complex128, Complex128, |x| x.re.is_infinite()
    || x.im.is_infinite());
constant_presence_kernel!(has_nan_str, Str, false);
constant_presence_kernel!(has_inf_str, Str, false);

macro_rules! constant_bool_unary_kernel {
    ($name:ident, $variant:ident, $value:expr) => {
        fn $name(input: ArrayData) -> Result<ArrayData> {
            match input {
                ArrayData::$variant(data) => Ok(ArrayData::Bool(
                    ArrayD::from_elem(IxDyn(data.shape()), $value).into_shared(),
                )),
                _ => Err(NumpyError::TypeError(
                    "value unary kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

macro_rules! map_unary_kernel {
    ($name:ident, $variant:ident, $out_variant:ident, $op:expr) => {
        fn $name(input: ArrayData) -> Result<ArrayData> {
            match input {
                ArrayData::$variant(data) => {
                    Ok(ArrayData::$out_variant(data.mapv($op).into_shared()))
                }
                _ => Err(NumpyError::TypeError(
                    "value unary kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

constant_bool_unary_kernel!(signbit_bool, Bool, false);
map_unary_kernel!(signbit_int32, Int32, Bool, |x: i32| x < 0);
map_unary_kernel!(signbit_int64, Int64, Bool, |x: i64| x < 0);
map_unary_kernel!(signbit_float32, Float32, Bool, |x: f32| x
    .is_sign_negative());
map_unary_kernel!(signbit_float64, Float64, Bool, |x: f64| x
    .is_sign_negative());
constant_bool_unary_kernel!(signbit_complex64, Complex64, false);
constant_bool_unary_kernel!(signbit_complex128, Complex128, false);
constant_bool_unary_kernel!(signbit_str, Str, false);

map_unary_kernel!(sign_bool, Bool, Int32, |x: bool| if x { 1 } else { 0 });
map_unary_kernel!(sign_int32, Int32, Int32, |x: i32| x.signum());
map_unary_kernel!(sign_int64, Int64, Int64, |x: i64| x.signum());
map_unary_kernel!(sign_float32, Float32, Float32, |x: f32| {
    if x.is_nan() {
        x
    } else if x == 0.0 {
        0.0
    } else {
        x.signum()
    }
});
map_unary_kernel!(sign_float64, Float64, Float64, |x: f64| {
    if x.is_nan() {
        x
    } else if x == 0.0 {
        0.0
    } else {
        x.signum()
    }
});
map_unary_kernel!(sign_complex64, Complex64, Complex64, |x: Complex<f32>| x);
map_unary_kernel!(sign_complex128, Complex128, Complex128, |x: Complex<
    f64,
>| x);

map_unary_kernel!(neg_bool, Bool, Int32, |x: bool| if x { -1 } else { 0 });
map_unary_kernel!(neg_int32, Int32, Int32, |x: i32| -x);
map_unary_kernel!(neg_int64, Int64, Int64, |x: i64| -x);
map_unary_kernel!(neg_float32, Float32, Float32, |x: f32| -x);
map_unary_kernel!(neg_float64, Float64, Float64, |x: f64| -x);
map_unary_kernel!(neg_complex64, Complex64, Complex64, |x: Complex<f32>| -x);
map_unary_kernel!(neg_complex128, Complex128, Complex128, |x: Complex<f64>| -x);

macro_rules! real_unary_kernel {
    ($name:ident, $variant:ident, $op:expr) => {
        fn $name(input: ArrayData) -> Result<ArrayData> {
            match input {
                ArrayData::$variant(data) => Ok(ArrayData::$variant(data.mapv($op).into_shared())),
                _ => Err(NumpyError::TypeError(
                    "real unary kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

real_unary_kernel!(real_spacing_float32, Float32, |x: f32| {
    let ax = x.abs();
    libm::nextafterf(ax, f32::INFINITY) - ax
});
real_unary_kernel!(real_spacing_float64, Float64, |x: f64| {
    let ax = x.abs();
    libm::nextafter(ax, f64::INFINITY) - ax
});
real_unary_kernel!(real_i0_float64, Float64, |x: f64| {
    let mut val = 1.0_f64;
    let mut term = 1.0_f64;
    let h = x * 0.5;
    for k in 1_u32..30 {
        term *= (h * h) / (k * k) as f64;
        val += term;
        if term.abs() < 1e-15 * val.abs() {
            break;
        }
    }
    val
});
real_unary_kernel!(real_i0_float32, Float32, |x: f32| {
    let mut val = 1.0_f32;
    let mut term = 1.0_f32;
    let h = x * 0.5;
    for k in 1_u32..25 {
        term *= (h * h) / (k * k) as f32;
        val += term;
        if term.abs() < 1e-7_f32 * val.abs() {
            break;
        }
    }
    val
});

fn real_arctan2_float32(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
    match (lhs, rhs) {
        (ArrayData::Float32(y), ArrayData::Float32(x)) => {
            let mut out = y.clone();
            ndarray::Zip::from(&mut out)
                .and(&x)
                .for_each(|o, &xi| *o = o.atan2(xi));
            Ok(ArrayData::Float32(out))
        }
        _ => Err(NumpyError::TypeError(
            "real binary kernel dtype mismatch".into(),
        )),
    }
}

fn real_arctan2_float64(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
    match (lhs, rhs) {
        (ArrayData::Float64(y), ArrayData::Float64(x)) => {
            let mut out = y.clone();
            ndarray::Zip::from(&mut out)
                .and(&x)
                .for_each(|o, &xi| *o = o.atan2(xi));
            Ok(ArrayData::Float64(out))
        }
        _ => Err(NumpyError::TypeError(
            "real binary kernel dtype mismatch".into(),
        )),
    }
}

fn real_ldexp_float32(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
    match (lhs, rhs) {
        (ArrayData::Float32(a), ArrayData::Int32(e)) => Ok(ArrayData::Float32(
            ndarray::Zip::from(&a)
                .and(&e)
                .map_collect(|&x, &n| libm::ldexpf(x, n))
                .into_shared(),
        )),
        _ => Err(NumpyError::TypeError(
            "real binary kernel dtype mismatch".into(),
        )),
    }
}

fn real_ldexp_float64(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
    match (lhs, rhs) {
        (ArrayData::Float64(a), ArrayData::Int32(e)) => Ok(ArrayData::Float64(
            ndarray::Zip::from(&a)
                .and(&e)
                .map_collect(|&x, &n| libm::ldexp(x, n))
                .into_shared(),
        )),
        _ => Err(NumpyError::TypeError(
            "real binary kernel dtype mismatch".into(),
        )),
    }
}

macro_rules! unary_math_kernel {
    ($name:ident, $variant:ident, $op:expr) => {
        fn $name(input: ArrayData) -> Result<ArrayData> {
            match input {
                ArrayData::$variant(data) => Ok(ArrayData::$variant(data.mapv($op).into_shared())),
                _ => Err(NumpyError::TypeError(
                    "unary math kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

unary_math_kernel!(unary_sqrt_float32, Float32, |x: f32| x.sqrt());
unary_math_kernel!(unary_sqrt_float64, Float64, |x: f64| x.sqrt());
unary_math_kernel!(unary_sqrt_complex64, Complex64, |x: Complex<f32>| x.sqrt());
unary_math_kernel!(unary_sqrt_complex128, Complex128, |x: Complex<f64>| x
    .sqrt());
unary_math_kernel!(unary_exp_float32, Float32, |x: f32| x.exp());
unary_math_kernel!(unary_exp_float64, Float64, |x: f64| x.exp());
unary_math_kernel!(unary_exp_complex64, Complex64, |x: Complex<f32>| x.exp());
unary_math_kernel!(unary_exp_complex128, Complex128, |x: Complex<f64>| x.exp());
unary_math_kernel!(unary_log_float32, Float32, |x: f32| x.ln());
unary_math_kernel!(unary_log_float64, Float64, |x: f64| x.ln());
unary_math_kernel!(unary_log_complex64, Complex64, |x: Complex<f32>| x.ln());
unary_math_kernel!(unary_log_complex128, Complex128, |x: Complex<f64>| x.ln());
unary_math_kernel!(unary_sin_float32, Float32, |x: f32| x.sin());
unary_math_kernel!(unary_sin_float64, Float64, |x: f64| x.sin());
unary_math_kernel!(unary_sin_complex64, Complex64, |x: Complex<f32>| x.sin());
unary_math_kernel!(unary_sin_complex128, Complex128, |x: Complex<f64>| x.sin());
unary_math_kernel!(unary_cos_float32, Float32, |x: f32| x.cos());
unary_math_kernel!(unary_cos_float64, Float64, |x: f64| x.cos());
unary_math_kernel!(unary_cos_complex64, Complex64, |x: Complex<f32>| x.cos());
unary_math_kernel!(unary_cos_complex128, Complex128, |x: Complex<f64>| x.cos());
unary_math_kernel!(unary_tan_float32, Float32, |x: f32| x.tan());
unary_math_kernel!(unary_tan_float64, Float64, |x: f64| x.tan());
unary_math_kernel!(unary_tan_complex64, Complex64, |x: Complex<f32>| x.tan());
unary_math_kernel!(unary_tan_complex128, Complex128, |x: Complex<f64>| x.tan());
unary_math_kernel!(unary_log10_float32, Float32, |x: f32| x.log10());
unary_math_kernel!(unary_log10_float64, Float64, |x: f64| x.log10());
unary_math_kernel!(unary_log10_complex64, Complex64, |x: Complex<f32>| x.ln()
    / Complex::new(std::f32::consts::LN_10, 0.0));
unary_math_kernel!(unary_log10_complex128, Complex128, |x: Complex<f64>| x.ln()
    / Complex::new(std::f64::consts::LN_10, 0.0));
unary_math_kernel!(unary_log2_float32, Float32, |x: f32| x.log2());
unary_math_kernel!(unary_log2_float64, Float64, |x: f64| x.log2());
unary_math_kernel!(unary_log2_complex64, Complex64, |x: Complex<f32>| x.ln()
    / Complex::new(std::f32::consts::LN_2, 0.0));
unary_math_kernel!(unary_log2_complex128, Complex128, |x: Complex<f64>| x.ln()
    / Complex::new(std::f64::consts::LN_2, 0.0));
unary_math_kernel!(unary_sinh_float32, Float32, |x: f32| x.sinh());
unary_math_kernel!(unary_sinh_float64, Float64, |x: f64| x.sinh());
unary_math_kernel!(unary_sinh_complex64, Complex64, |x: Complex<f32>| x.sinh());
unary_math_kernel!(unary_sinh_complex128, Complex128, |x: Complex<f64>| x
    .sinh());
unary_math_kernel!(unary_cosh_float32, Float32, |x: f32| x.cosh());
unary_math_kernel!(unary_cosh_float64, Float64, |x: f64| x.cosh());
unary_math_kernel!(unary_cosh_complex64, Complex64, |x: Complex<f32>| x.cosh());
unary_math_kernel!(unary_cosh_complex128, Complex128, |x: Complex<f64>| x
    .cosh());
unary_math_kernel!(unary_tanh_float32, Float32, |x: f32| x.tanh());
unary_math_kernel!(unary_tanh_float64, Float64, |x: f64| x.tanh());
unary_math_kernel!(unary_tanh_complex64, Complex64, |x: Complex<f32>| x.tanh());
unary_math_kernel!(unary_tanh_complex128, Complex128, |x: Complex<f64>| x
    .tanh());
unary_math_kernel!(unary_arcsin_float32, Float32, |x: f32| x.asin());
unary_math_kernel!(unary_arcsin_float64, Float64, |x: f64| x.asin());
unary_math_kernel!(unary_arcsin_complex64, Complex64, |x: Complex<f32>| x
    .asin());
unary_math_kernel!(unary_arcsin_complex128, Complex128, |x: Complex<f64>| x
    .asin());
unary_math_kernel!(unary_arccos_float32, Float32, |x: f32| x.acos());
unary_math_kernel!(unary_arccos_float64, Float64, |x: f64| x.acos());
unary_math_kernel!(unary_arccos_complex64, Complex64, |x: Complex<f32>| x
    .acos());
unary_math_kernel!(unary_arccos_complex128, Complex128, |x: Complex<f64>| x
    .acos());
unary_math_kernel!(unary_arctan_float32, Float32, |x: f32| x.atan());
unary_math_kernel!(unary_arctan_float64, Float64, |x: f64| x.atan());
unary_math_kernel!(unary_arctan_complex64, Complex64, |x: Complex<f32>| x
    .atan());
unary_math_kernel!(unary_arctan_complex128, Complex128, |x: Complex<f64>| x
    .atan());
unary_math_kernel!(unary_arcsinh_float32, Float32, |x: f32| x.asinh());
unary_math_kernel!(unary_arcsinh_float64, Float64, |x: f64| x.asinh());
unary_math_kernel!(unary_arcsinh_complex64, Complex64, |x: Complex<f32>| x
    .asinh());
unary_math_kernel!(unary_arcsinh_complex128, Complex128, |x: Complex<f64>| x
    .asinh());
unary_math_kernel!(unary_arccosh_float32, Float32, |x: f32| x.acosh());
unary_math_kernel!(unary_arccosh_float64, Float64, |x: f64| x.acosh());
unary_math_kernel!(unary_arccosh_complex64, Complex64, |x: Complex<f32>| x
    .acosh());
unary_math_kernel!(unary_arccosh_complex128, Complex128, |x: Complex<f64>| x
    .acosh());
unary_math_kernel!(unary_arctanh_float32, Float32, |x: f32| x.atanh());
unary_math_kernel!(unary_arctanh_float64, Float64, |x: f64| x.atanh());
unary_math_kernel!(unary_arctanh_complex64, Complex64, |x: Complex<f32>| x
    .atanh());
unary_math_kernel!(unary_arctanh_complex128, Complex128, |x: Complex<f64>| x
    .atanh());
unary_math_kernel!(unary_floor_float32, Float32, |x: f32| x.floor());
unary_math_kernel!(unary_floor_float64, Float64, |x: f64| x.floor());
unary_math_kernel!(unary_ceil_float32, Float32, |x: f32| x.ceil());
unary_math_kernel!(unary_ceil_float64, Float64, |x: f64| x.ceil());
unary_math_kernel!(unary_round_float32, Float32, |x: f32| x.round());
unary_math_kernel!(unary_round_float64, Float64, |x: f64| x.round());
unary_math_kernel!(unary_log1p_float32, Float32, |x: f32| x.ln_1p());
unary_math_kernel!(unary_log1p_float64, Float64, |x: f64| x.ln_1p());
unary_math_kernel!(unary_expm1_float32, Float32, |x: f32| x.exp_m1());
unary_math_kernel!(unary_expm1_float64, Float64, |x: f64| x.exp_m1());
unary_math_kernel!(unary_deg2rad_float32, Float32, |x: f32| x.to_radians());
unary_math_kernel!(unary_deg2rad_float64, Float64, |x: f64| x.to_radians());
unary_math_kernel!(unary_rad2deg_float32, Float32, |x: f32| x.to_degrees());
unary_math_kernel!(unary_rad2deg_float64, Float64, |x: f64| x.to_degrees());
unary_math_kernel!(unary_trunc_float32, Float32, |x: f32| x.trunc());
unary_math_kernel!(unary_trunc_float64, Float64, |x: f64| x.trunc());
unary_math_kernel!(unary_cbrt_float32, Float32, libm::cbrtf);
unary_math_kernel!(unary_cbrt_float64, Float64, libm::cbrt);
unary_math_kernel!(unary_gamma_float32, Float32, libm::tgammaf);
unary_math_kernel!(unary_gamma_float64, Float64, libm::tgamma);
unary_math_kernel!(unary_lgamma_float32, Float32, libm::lgammaf);
unary_math_kernel!(unary_lgamma_float64, Float64, libm::lgamma);
unary_math_kernel!(unary_erf_float32, Float32, libm::erff);
unary_math_kernel!(unary_erf_float64, Float64, libm::erf);
unary_math_kernel!(unary_erfc_float32, Float32, libm::erfcf);
unary_math_kernel!(unary_erfc_float64, Float64, libm::erfc);
unary_math_kernel!(unary_j0_float32, Float32, libm::j0f);
unary_math_kernel!(unary_j0_float64, Float64, libm::j0);
unary_math_kernel!(unary_j1_float32, Float32, libm::j1f);
unary_math_kernel!(unary_j1_float64, Float64, libm::j1);
unary_math_kernel!(unary_y0_float32, Float32, libm::y0f);
unary_math_kernel!(unary_y0_float64, Float64, libm::y0);
unary_math_kernel!(unary_y1_float32, Float32, libm::y1f);
unary_math_kernel!(unary_y1_float64, Float64, libm::y1);

macro_rules! binary_math_kernel {
    ($name:ident, $variant:ident, $op:expr) => {
        fn $name(lhs: ArrayData, rhs: ArrayData) -> Result<ArrayData> {
            match (lhs, rhs) {
                (ArrayData::$variant(a), ArrayData::$variant(b)) => Ok(ArrayData::$variant(
                    ndarray::Zip::from(&a)
                        .and(&b)
                        .map_collect(|&x, &y| $op(x, y))
                        .into_shared(),
                )),
                _ => Err(NumpyError::TypeError(
                    "binary math kernel dtype mismatch".into(),
                )),
            }
        }
    };
}

binary_math_kernel!(binary_copysign_float32, Float32, libm::copysignf);
binary_math_kernel!(binary_copysign_float64, Float64, libm::copysign);
binary_math_kernel!(binary_hypot_float32, Float32, libm::hypotf);
binary_math_kernel!(binary_hypot_float64, Float64, libm::hypot);
binary_math_kernel!(binary_fmod_float32, Float32, libm::fmodf);
binary_math_kernel!(binary_fmod_float64, Float64, libm::fmod);
binary_math_kernel!(binary_nextafter_float32, Float32, libm::nextafterf);
binary_math_kernel!(binary_nextafter_float64, Float64, libm::nextafter);
binary_math_kernel!(binary_logaddexp_float32, Float32, |a: f32, b: f32| {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        let mx = a.max(b);
        mx + (1.0_f32 + (-(a - b).abs()).exp()).ln()
    }
});
binary_math_kernel!(binary_logaddexp_float64, Float64, |a: f64, b: f64| {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        let mx = a.max(b);
        mx + (1.0_f64 + (-(a - b).abs()).exp()).ln()
    }
});
binary_math_kernel!(binary_logaddexp2_float32, Float32, |a: f32, b: f32| {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        let mx = a.max(b);
        mx + (1.0_f32 + (-(a - b).abs()).exp2()).log2()
    }
});
binary_math_kernel!(binary_logaddexp2_float64, Float64, |a: f64, b: f64| {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        let mx = a.max(b);
        mx + (1.0_f64 + (-(a - b).abs()).exp2()).log2()
    }
});
binary_math_kernel!(binary_fmax_float32, Float32, f32::max);
binary_math_kernel!(binary_fmax_float64, Float64, f64::max);
binary_math_kernel!(binary_fmin_float32, Float32, f32::min);
binary_math_kernel!(binary_fmin_float64, Float64, f64::min);
binary_math_kernel!(binary_maximum_float32, Float32, |a: f32, b: f32| {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
});
binary_math_kernel!(binary_maximum_float64, Float64, |a: f64, b: f64| {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
});
binary_math_kernel!(binary_minimum_float32, Float32, |a: f32, b: f32| {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
});
binary_math_kernel!(binary_minimum_float64, Float64, |a: f64, b: f64| {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
});

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
