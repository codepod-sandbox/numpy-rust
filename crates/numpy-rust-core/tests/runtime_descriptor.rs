use numpy_rust_core::kernel::{
    ArithmeticKernelOp, BitwiseBinaryKernelOp, BitwiseUnaryKernelOp, DecomposeUnaryKernelOp,
    MathBinaryKernelOp, MathUnaryKernelOp, RealBinaryKernelOp, RealUnaryKernelOp, TruthKernelOp,
    TruthReduceKernelOp, ValueUnaryKernelOp,
};
use numpy_rust_core::kernel::{DotKernelOp, PredicateKernelOp, PredicatePresenceOp, WhereKernelOp};
use numpy_rust_core::{descriptor_for_dtype, resolve_binary_op, BinaryOp, DType};

#[test]
fn descriptor_lookup_returns_expected_metadata() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert_eq!(desc.name(), "float64");
    assert_eq!(desc.itemsize(), 8);
}

#[test]
fn binary_add_resolution_promotes_int32_and_float64_to_float64() {
    let plan = resolve_binary_op(BinaryOp::Add, DType::Int32, DType::Float64).unwrap();
    assert_eq!(plan.output_dtype(), DType::Float64);
}

#[test]
fn binary_mul_resolution_promotes_bool_and_bool_to_int8() {
    let plan = resolve_binary_op(BinaryOp::Mul, DType::Bool, DType::Bool).unwrap();
    assert_eq!(plan.output_dtype(), DType::Int8);
}

#[test]
fn binary_remainder_resolution_promotes_bool_and_bool_to_int8() {
    let plan = resolve_binary_op(BinaryOp::Remainder, DType::Bool, DType::Bool).unwrap();
    assert_eq!(plan.output_dtype(), DType::Int8);
}

#[test]
fn float64_descriptor_registers_dot_kernel() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.dot_kernel(DotKernelOp::Dot1d1d).is_some());
    assert!(desc.dot_kernel(DotKernelOp::MatMul2d2d).is_some());
    assert!(desc.dot_kernel(DotKernelOp::MatMul2d1d).is_some());
}

#[test]
fn float64_descriptor_registers_where_kernel() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.where_kernel(WhereKernelOp::Select).is_some());
}

#[test]
fn float64_descriptor_registers_predicate_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.predicate_kernel(PredicateKernelOp::IsNaN).is_some());
    assert!(desc.predicate_kernel(PredicateKernelOp::IsFinite).is_some());
    assert!(desc.predicate_kernel(PredicateKernelOp::IsInf).is_some());
}

#[test]
fn float64_descriptor_registers_predicate_presence_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc
        .predicate_presence_kernel(PredicatePresenceOp::HasNaN)
        .is_some());
    assert!(desc
        .predicate_presence_kernel(PredicatePresenceOp::HasInf)
        .is_some());
}

#[test]
fn float64_descriptor_registers_truth_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.truth_kernel(TruthKernelOp::ToBool).is_some());
    assert!(desc
        .truth_reduce_kernel(TruthReduceKernelOp::AllTruthy)
        .is_some());
    assert!(desc
        .truth_reduce_kernel(TruthReduceKernelOp::AnyTruthy)
        .is_some());
}

#[test]
fn int64_descriptor_registers_bitwise_kernels() {
    let desc = descriptor_for_dtype(DType::Int64);
    assert!(desc
        .bitwise_unary_kernel(BitwiseUnaryKernelOp::Not)
        .is_some());
    assert!(desc
        .bitwise_binary_kernel(BitwiseBinaryKernelOp::And)
        .is_some());
    assert!(desc
        .bitwise_binary_kernel(BitwiseBinaryKernelOp::LeftShift)
        .is_some());
}

#[test]
fn float64_descriptor_registers_math_unary_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.math_unary_kernel(MathUnaryKernelOp::Sqrt).is_some());
    assert!(desc.math_unary_kernel(MathUnaryKernelOp::Log1p).is_some());
}

#[test]
fn float64_descriptor_registers_math_binary_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.math_binary_kernel(MathBinaryKernelOp::Hypot).is_some());
    assert!(desc
        .math_binary_kernel(MathBinaryKernelOp::Maximum)
        .is_some());
}

#[test]
fn float64_descriptor_registers_basic_binary_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.binary_kernel(ArithmeticKernelOp::Add).is_some());
    assert!(desc.binary_kernel(ArithmeticKernelOp::Sub).is_some());
    assert!(desc.binary_kernel(ArithmeticKernelOp::Mul).is_some());
    assert!(desc.binary_kernel(ArithmeticKernelOp::Div).is_some());
    assert!(desc.binary_kernel(ArithmeticKernelOp::FloorDiv).is_some());
    assert!(desc.binary_kernel(ArithmeticKernelOp::Remainder).is_some());
}

#[test]
fn float64_descriptor_registers_value_unary_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc
        .value_unary_kernel(ValueUnaryKernelOp::SignBit)
        .is_some());
    assert!(desc.value_unary_kernel(ValueUnaryKernelOp::Neg).is_some());
}

#[test]
fn float64_descriptor_registers_real_special_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc.real_unary_kernel(RealUnaryKernelOp::Spacing).is_some());
    assert!(desc.real_unary_kernel(RealUnaryKernelOp::I0).is_some());
    assert!(desc
        .real_binary_kernel(RealBinaryKernelOp::ArcTan2)
        .is_some());
    assert!(desc.real_binary_kernel(RealBinaryKernelOp::LDExp).is_some());
}

#[test]
fn float64_descriptor_registers_decompose_kernels() {
    let desc = descriptor_for_dtype(DType::Float64);
    assert!(desc
        .decompose_unary_kernel(DecomposeUnaryKernelOp::Frexp)
        .is_some());
    assert!(desc
        .decompose_unary_kernel(DecomposeUnaryKernelOp::Modf)
        .is_some());
}
