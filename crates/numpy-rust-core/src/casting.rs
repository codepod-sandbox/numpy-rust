use crate::array_data::ArrayData;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::resolver::{resolve_cast, CastPlan, CastingRule};

/// Truncate values in an ArrayData to fit the bit-width of a narrow dtype.
/// For example, Int32 storage with UInt8 dtype: 400 → 400 as u8 as i32 = 144.
/// This must be called after any operation that produces a narrow-typed result.
pub fn narrow_truncate(data: ArrayData, dtype: DType) -> ArrayData {
    if let Some(kernel) = descriptor_for_dtype(dtype).narrow_finalize_kernel() {
        kernel(data)
    } else {
        data
    }
}

/// Cast an ArrayData to a different dtype.
/// If already the target dtype, returns a clone.
/// For narrow dtypes, casts to the storage type instead.
pub fn cast_array_data(data: impl AsRef<ArrayData>, target: DType) -> ArrayData {
    let data = data.as_ref();
    let plan = resolve_cast(data.dtype(), target, CastingRule::Unsafe)
        .expect("resolver must approve unsafe backend casts");

    if !plan.requires_storage_cast() {
        return data.clone();
    }

    execute_cast_plan(data, plan)
}

fn execute_cast_plan(data: &ArrayData, plan: CastPlan) -> ArrayData {
    let descriptor = descriptor_for_dtype(plan.target_storage_dtype());
    let kernel = descriptor
        .cast_kernel()
        .expect("storage dtype descriptor must register cast execution");
    kernel(data)
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_cast_i32_to_f64() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = a.astype(DType::Float64);
        assert_eq!(b.dtype(), DType::Float64);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn test_cast_f64_to_i32_truncates() {
        let a = NdArray::from_vec(vec![1.7_f64, 2.3, 3.9]);
        let b = a.astype(DType::Int32);
        assert_eq!(b.dtype(), DType::Int32);
    }

    #[test]
    fn test_cast_same_type_is_clone() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = a.astype(DType::Float64);
        assert_eq!(b.dtype(), DType::Float64);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn test_cast_bool_to_int() {
        let a = NdArray::from_vec(vec![true, false, true]);
        let b = a.astype(DType::Int32);
        assert_eq!(b.dtype(), DType::Int32);
    }

    #[test]
    fn test_cast_f64_to_bool() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0, -3.5]);
        let b = a.astype(DType::Bool);
        assert_eq!(b.dtype(), DType::Bool);
    }

    #[test]
    fn test_cast_preserves_shape() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = a.astype(DType::Int32);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_cast_f64_to_complex128() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = a.astype(DType::Complex128);
        assert_eq!(b.dtype(), DType::Complex128);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn test_cast_f32_to_complex64() {
        let a = NdArray::from_vec(vec![1.0_f32, 2.0, 3.0]);
        let b = a.astype(DType::Complex64);
        assert_eq!(b.dtype(), DType::Complex64);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn test_cast_complex128_to_f64() {
        let a = NdArray::zeros(&[3], DType::Complex128);
        let b = a.astype(DType::Float64);
        assert_eq!(b.dtype(), DType::Float64);
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_cast_complex64_to_complex128() {
        let a = NdArray::zeros(&[3], DType::Complex64);
        let b = a.astype(DType::Complex128);
        assert_eq!(b.dtype(), DType::Complex128);
    }

    #[test]
    fn test_cast_float64_to_object() {
        let a = NdArray::from_vec(vec![1.5_f64, 2.5]);
        let b = a.astype(DType::Object);
        assert_eq!(b.dtype(), DType::Object);
        let storage = b.storage().boxed_storage().unwrap();
        assert_eq!(
            storage.elements().unwrap(),
            vec![
                crate::BoxedScalar::Object(crate::BoxedObjectScalar::Float(1.5)),
                crate::BoxedScalar::Object(crate::BoxedObjectScalar::Float(2.5)),
            ]
        );
    }
}
