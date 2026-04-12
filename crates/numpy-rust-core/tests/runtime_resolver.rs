use numpy_rust_core::resolver::{resolve_cast, resolve_reduction_op, CastingRule, ReductionOp};
use numpy_rust_core::{resolve_binary_op, BinaryOp, DType, NumpyError};

#[test]
fn same_kind_cast_rejects_float64_to_int32() {
    let error = resolve_cast(DType::Float64, DType::Int32, CastingRule::SameKind).unwrap_err();

    assert!(matches!(error, NumpyError::TypeError(_)));
    assert_eq!(
        error.to_string(),
        "type error: cannot cast float64 to int32 under same_kind"
    );
}

#[test]
fn binary_add_resolution_promotes_int32_and_float64_to_float64() {
    let plan = resolve_binary_op(BinaryOp::Add, DType::Int32, DType::Float64).unwrap();

    assert_eq!(plan.output_dtype(), DType::Float64);
    assert_eq!(plan.result_storage_dtype(), DType::Float64);
    assert!(!plan.requires_output_narrowing());
    assert_eq!(plan.result_cast().source_dtype(), DType::Float64);
    assert_eq!(plan.result_cast().target_dtype(), DType::Float64);
    assert_eq!(plan.lhs_cast().target_dtype(), DType::Float64);
    assert_eq!(plan.lhs_cast().execution_storage_dtype(), DType::Float64);
    assert_eq!(plan.rhs_cast().target_dtype(), DType::Float64);
    assert_eq!(plan.rhs_cast().execution_storage_dtype(), DType::Float64);
}

#[test]
fn binary_add_resolution_promotes_bool_and_bool_to_int8() {
    let plan = resolve_binary_op(BinaryOp::Add, DType::Bool, DType::Bool).unwrap();

    assert_eq!(plan.output_dtype(), DType::Int8);
    assert_eq!(plan.result_storage_dtype(), DType::Int32);
    assert!(plan.requires_output_narrowing());
    assert_eq!(plan.result_cast().source_dtype(), DType::Int32);
    assert_eq!(plan.result_cast().target_dtype(), DType::Int8);
    assert_eq!(plan.lhs_cast().target_dtype(), DType::Int8);
    assert_eq!(plan.lhs_cast().execution_storage_dtype(), DType::Int32);
    assert_eq!(plan.rhs_cast().target_dtype(), DType::Int8);
    assert_eq!(plan.rhs_cast().execution_storage_dtype(), DType::Int32);
}

#[test]
fn binary_add_resolution_rejects_string_operands() {
    let error = resolve_binary_op(BinaryOp::Add, DType::Str, DType::Str).unwrap_err();

    assert!(matches!(error, NumpyError::TypeError(_)));
    assert_eq!(
        error.to_string(),
        "type error: arithmetic not supported for string arrays"
    );
}

#[test]
fn prod_reduction_resolution_uses_float64_execution_and_result() {
    let plan = resolve_reduction_op(ReductionOp::Prod, DType::Int32).unwrap();

    assert_eq!(plan.input_cast().source_dtype(), DType::Int32);
    assert_eq!(plan.input_cast().target_dtype(), DType::Float64);
    assert_eq!(plan.input_cast().execution_storage_dtype(), DType::Float64);
    assert_eq!(plan.result_dtype(), DType::Float64);
    assert_eq!(plan.result_storage_dtype(), DType::Float64);
}
