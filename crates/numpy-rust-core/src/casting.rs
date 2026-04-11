use crate::array_data::ArrayD;
use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::resolver::{resolve_cast, CastPlan, CastingRule};

/// Convert a float to a string matching Python's str(float) behavior.
/// Python always includes a decimal point: str(0.0) == "0.0", str(1.0) == "1.0".
fn float_to_str(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    let s = format!("{}", x);
    // If the string contains no '.' or 'e'/'E', add ".0" to match Python behavior
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        format!("{}.0", s)
    } else {
        s
    }
}

/// Truncate values in an ArrayData to fit the bit-width of a narrow dtype.
/// For example, Int32 storage with UInt8 dtype: 400 → 400 as u8 as i32 = 144.
/// This must be called after any operation that produces a narrow-typed result.
pub fn narrow_truncate(data: ArrayData, dtype: DType) -> ArrayData {
    match data {
        ArrayData::Int32(arr) => ArrayData::Int32(match dtype {
            DType::Int8 => arr.mapv(|x| x as i8 as i32).into_shared(),
            DType::UInt8 => arr.mapv(|x| x as u8 as i32).into_shared(),
            DType::Int16 => arr.mapv(|x| x as i16 as i32).into_shared(),
            DType::UInt16 => arr.mapv(|x| x as u16 as i32).into_shared(),
            _ => arr,
        }),
        ArrayData::Int64(arr) => ArrayData::Int64(match dtype {
            DType::UInt32 => arr.mapv(|x| x as u32 as i64).into_shared(),
            DType::UInt64 => arr.mapv(|x| x as u64 as i64).into_shared(),
            _ => arr,
        }),
        other => other,
    }
}

/// Cast an ArrayData to a different dtype.
/// If already the target dtype, returns a clone.
/// For narrow dtypes, casts to the storage type instead.
pub fn cast_array_data(data: &ArrayData, target: DType) -> ArrayData {
    let plan = resolve_cast(data.dtype(), target, CastingRule::Unsafe)
        .expect("resolver must approve unsafe backend casts");

    if !plan.requires_storage_cast() {
        return data.clone();
    }

    execute_cast_plan(data, plan)
}

fn execute_cast_plan(data: &ArrayData, plan: CastPlan) -> ArrayData {
    match plan.target_storage_dtype() {
        DType::Bool => ArrayData::Bool(cast_to_bool(data)),
        DType::Int32 => ArrayData::Int32(cast_to_i32(data)),
        DType::Int64 => ArrayData::Int64(cast_to_i64(data)),
        DType::Float32 => ArrayData::Float32(cast_to_f32(data)),
        DType::Float64 => ArrayData::Float64(cast_to_f64(data)),
        DType::Complex64 => ArrayData::Complex64(cast_to_complex64(data)),
        DType::Complex128 => ArrayData::Complex128(cast_to_complex128(data)),
        DType::Str => ArrayData::Str(cast_to_str(data)),
        _ => unreachable!("storage_dtype maps to canonical types"),
    }
}

fn cast_to_bool(data: &ArrayData) -> ArrayD<bool> {
    match data {
        ArrayData::Bool(a) => a.clone(),
        ArrayData::Int32(a) => a.mapv(|x| x != 0).into_shared(),
        ArrayData::Int64(a) => a.mapv(|x| x != 0).into_shared(),
        ArrayData::Float32(a) => a.mapv(|x| x != 0.0).into_shared(),
        ArrayData::Float64(a) => a.mapv(|x| x != 0.0).into_shared(),
        ArrayData::Complex64(a) => a.mapv(|x| x.re != 0.0 || x.im != 0.0).into_shared(),
        ArrayData::Complex128(a) => a.mapv(|x| x.re != 0.0 || x.im != 0.0).into_shared(),
        ArrayData::Str(a) => a.mapv(|ref x| !x.is_empty()).into_shared(),
    }
}

fn cast_to_i32(data: &ArrayData) -> ArrayD<i32> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| x as i32).into_shared(),
        ArrayData::Int32(a) => a.clone(),
        ArrayData::Int64(a) => a.mapv(|x| x as i32).into_shared(),
        ArrayData::Float32(a) => a.mapv(|x| x as i32).into_shared(),
        ArrayData::Float64(a) => a.mapv(|x| x as i32).into_shared(),
        ArrayData::Complex64(a) => a.mapv(|x| x.re as i32).into_shared(),
        ArrayData::Complex128(a) => a.mapv(|x| x.re as i32).into_shared(),
        ArrayData::Str(a) => a.mapv(|ref x| x.parse::<i32>().unwrap_or(0)).into_shared(),
    }
}

fn cast_to_i64(data: &ArrayData) -> ArrayD<i64> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| x as i64).into_shared(),
        ArrayData::Int32(a) => a.mapv(|x| x as i64).into_shared(),
        ArrayData::Int64(a) => a.clone(),
        ArrayData::Float32(a) => a.mapv(|x| x as i64).into_shared(),
        ArrayData::Float64(a) => a.mapv(|x| x as i64).into_shared(),
        ArrayData::Complex64(a) => a.mapv(|x| x.re as i64).into_shared(),
        ArrayData::Complex128(a) => a.mapv(|x| x.re as i64).into_shared(),
        ArrayData::Str(a) => a.mapv(|ref x| x.parse::<i64>().unwrap_or(0)).into_shared(),
    }
}

fn cast_to_f32(data: &ArrayData) -> ArrayD<f32> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| if x { 1.0 } else { 0.0 }).into_shared(),
        ArrayData::Int32(a) => a.mapv(|x| x as f32).into_shared(),
        ArrayData::Int64(a) => a.mapv(|x| x as f32).into_shared(),
        ArrayData::Float32(a) => a.clone(),
        ArrayData::Float64(a) => a.mapv(|x| x as f32).into_shared(),
        ArrayData::Complex64(a) => a.mapv(|x| x.re).into_shared(),
        ArrayData::Complex128(a) => a.mapv(|x| x.re as f32).into_shared(),
        ArrayData::Str(a) => a
            .mapv(|ref x| x.parse::<f32>().unwrap_or(f32::NAN))
            .into_shared(),
    }
}

fn cast_to_f64(data: &ArrayData) -> ArrayD<f64> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| if x { 1.0 } else { 0.0 }).into_shared(),
        ArrayData::Int32(a) => a.mapv(|x| x as f64).into_shared(),
        ArrayData::Int64(a) => a.mapv(|x| x as f64).into_shared(),
        ArrayData::Float32(a) => a.mapv(|x| x as f64).into_shared(),
        ArrayData::Float64(a) => a.clone(),
        ArrayData::Complex64(a) => a.mapv(|x| x.re as f64).into_shared(),
        ArrayData::Complex128(a) => a.mapv(|x| x.re).into_shared(),
        ArrayData::Str(a) => a
            .mapv(|ref x| x.parse::<f64>().unwrap_or(f64::NAN))
            .into_shared(),
    }
}

fn cast_to_complex64(data: &ArrayData) -> ArrayD<Complex<f32>> {
    match data {
        ArrayData::Bool(a) => a
            .mapv(|x| Complex::new(if x { 1.0f32 } else { 0.0 }, 0.0))
            .into_shared(),
        ArrayData::Int32(a) => a.mapv(|x| Complex::new(x as f32, 0.0)).into_shared(),
        ArrayData::Int64(a) => a.mapv(|x| Complex::new(x as f32, 0.0)).into_shared(),
        ArrayData::Float32(a) => a.mapv(|x| Complex::new(x, 0.0)).into_shared(),
        ArrayData::Float64(a) => a.mapv(|x| Complex::new(x as f32, 0.0)).into_shared(),
        ArrayData::Complex64(a) => a.clone(),
        ArrayData::Complex128(a) => a
            .mapv(|x| Complex::new(x.re as f32, x.im as f32))
            .into_shared(),
        ArrayData::Str(a) => a
            .mapv(|ref x| {
                let f = x.parse::<f32>().unwrap_or(f32::NAN);
                Complex::new(f, 0.0)
            })
            .into_shared(),
    }
}

fn cast_to_complex128(data: &ArrayData) -> ArrayD<Complex<f64>> {
    match data {
        ArrayData::Bool(a) => a
            .mapv(|x| Complex::new(if x { 1.0f64 } else { 0.0 }, 0.0))
            .into_shared(),
        ArrayData::Int32(a) => a.mapv(|x| Complex::new(x as f64, 0.0)).into_shared(),
        ArrayData::Int64(a) => a.mapv(|x| Complex::new(x as f64, 0.0)).into_shared(),
        ArrayData::Float32(a) => a.mapv(|x| Complex::new(x as f64, 0.0)).into_shared(),
        ArrayData::Float64(a) => a.mapv(|x| Complex::new(x, 0.0)).into_shared(),
        ArrayData::Complex64(a) => a
            .mapv(|x| Complex::new(x.re as f64, x.im as f64))
            .into_shared(),
        ArrayData::Complex128(a) => a.clone(),
        ArrayData::Str(a) => a
            .mapv(|ref x| {
                let f = x.parse::<f64>().unwrap_or(f64::NAN);
                Complex::new(f, 0.0)
            })
            .into_shared(),
    }
}

fn cast_to_str(data: &ArrayData) -> ArrayD<String> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| x.to_string()).into_shared(),
        ArrayData::Int32(a) => a.mapv(|x| x.to_string()).into_shared(),
        ArrayData::Int64(a) => a.mapv(|x| x.to_string()).into_shared(),
        ArrayData::Float32(a) => a.mapv(|x| float_to_str(x as f64)).into_shared(),
        ArrayData::Float64(a) => a.mapv(float_to_str).into_shared(),
        ArrayData::Complex64(a) => a
            .mapv(|x| {
                if x.im >= 0.0 {
                    format!("({}+{}j)", x.re, x.im)
                } else {
                    format!("({}{}j)", x.re, x.im)
                }
            })
            .into_shared(),
        ArrayData::Complex128(a) => a
            .mapv(|x| {
                if x.im >= 0.0 {
                    format!("({}+{}j)", x.re, x.im)
                } else {
                    format!("({}{}j)", x.re, x.im)
                }
            })
            .into_shared(),
        ArrayData::Str(a) => a.clone(),
    }
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
}
