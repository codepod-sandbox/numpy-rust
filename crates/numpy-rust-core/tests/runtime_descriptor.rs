use numpy_rust_core::kernel::{DotKernelOp, PredicateKernelOp, WhereKernelOp};
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
