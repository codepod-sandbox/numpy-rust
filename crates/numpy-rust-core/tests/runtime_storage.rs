use numpy_rust_core::{DType, NdArray};

#[test]
fn zeros_uses_descriptor_backed_dtype_identity() {
    let arr = NdArray::zeros(&[2, 3], DType::Float64);
    assert_eq!(arr.dtype(), DType::Float64);
    assert_eq!(arr.shape(), &[2, 3]);
}

#[test]
fn slice_preserves_storage_overlap() {
    let arr = NdArray::ones(&[4], DType::Float64);
    let view = arr.slice_axis_for_test(1, 3);
    assert!(arr.shares_memory_with(&view));
}
