use crate::array_data::ArrayD;
use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::resolver::{resolve_cast, CastPlan, CastingRule};

trait CastScalar<T> {
    fn cast_scalar(self) -> T;
}

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

fn complex_to_str<T>(value: Complex<T>) -> String
where
    T: std::fmt::Display + PartialOrd + From<f32> + Copy,
{
    if value.im >= T::from(0.0) {
        format!("({}+{}j)", value.re, value.im)
    } else {
        format!("({}{}j)", value.re, value.im)
    }
}

macro_rules! impl_cast_scalar {
    ($src:ty => $dst:ty, |$value:ident| $body:expr) => {
        impl CastScalar<$dst> for $src {
            fn cast_scalar(self) -> $dst {
                let $value = self;
                $body
            }
        }
    };
}

impl_cast_scalar!(bool => bool, |value| value);
impl_cast_scalar!(i32 => bool, |value| value != 0);
impl_cast_scalar!(i64 => bool, |value| value != 0);
impl_cast_scalar!(f32 => bool, |value| value != 0.0);
impl_cast_scalar!(f64 => bool, |value| value != 0.0);
impl_cast_scalar!(Complex<f32> => bool, |value| value.re != 0.0 || value.im != 0.0);
impl_cast_scalar!(Complex<f64> => bool, |value| value.re != 0.0 || value.im != 0.0);
impl_cast_scalar!(String => bool, |value| !value.is_empty());

impl_cast_scalar!(bool => i32, |value| value as i32);
impl_cast_scalar!(i32 => i32, |value| value);
impl_cast_scalar!(i64 => i32, |value| value as i32);
impl_cast_scalar!(f32 => i32, |value| value as i32);
impl_cast_scalar!(f64 => i32, |value| value as i32);
impl_cast_scalar!(Complex<f32> => i32, |value| value.re as i32);
impl_cast_scalar!(Complex<f64> => i32, |value| value.re as i32);
impl_cast_scalar!(String => i32, |value| value.parse::<i32>().unwrap_or(0));

impl_cast_scalar!(bool => i64, |value| value as i64);
impl_cast_scalar!(i32 => i64, |value| value as i64);
impl_cast_scalar!(i64 => i64, |value| value);
impl_cast_scalar!(f32 => i64, |value| value as i64);
impl_cast_scalar!(f64 => i64, |value| value as i64);
impl_cast_scalar!(Complex<f32> => i64, |value| value.re as i64);
impl_cast_scalar!(Complex<f64> => i64, |value| value.re as i64);
impl_cast_scalar!(String => i64, |value| value.parse::<i64>().unwrap_or(0));

impl_cast_scalar!(bool => f32, |value| if value { 1.0 } else { 0.0 });
impl_cast_scalar!(i32 => f32, |value| value as f32);
impl_cast_scalar!(i64 => f32, |value| value as f32);
impl_cast_scalar!(f32 => f32, |value| value);
impl_cast_scalar!(f64 => f32, |value| value as f32);
impl_cast_scalar!(Complex<f32> => f32, |value| value.re);
impl_cast_scalar!(Complex<f64> => f32, |value| value.re as f32);
impl_cast_scalar!(String => f32, |value| value.parse::<f32>().unwrap_or(f32::NAN));

impl_cast_scalar!(bool => f64, |value| if value { 1.0 } else { 0.0 });
impl_cast_scalar!(i32 => f64, |value| value as f64);
impl_cast_scalar!(i64 => f64, |value| value as f64);
impl_cast_scalar!(f32 => f64, |value| value as f64);
impl_cast_scalar!(f64 => f64, |value| value);
impl_cast_scalar!(Complex<f32> => f64, |value| value.re as f64);
impl_cast_scalar!(Complex<f64> => f64, |value| value.re);
impl_cast_scalar!(String => f64, |value| value.parse::<f64>().unwrap_or(f64::NAN));

impl_cast_scalar!(bool => Complex<f32>, |value| Complex::new(if value { 1.0 } else { 0.0 }, 0.0));
impl_cast_scalar!(i32 => Complex<f32>, |value| Complex::new(value as f32, 0.0));
impl_cast_scalar!(i64 => Complex<f32>, |value| Complex::new(value as f32, 0.0));
impl_cast_scalar!(f32 => Complex<f32>, |value| Complex::new(value, 0.0));
impl_cast_scalar!(f64 => Complex<f32>, |value| Complex::new(value as f32, 0.0));
impl_cast_scalar!(Complex<f32> => Complex<f32>, |value| value);
impl_cast_scalar!(Complex<f64> => Complex<f32>, |value| Complex::new(value.re as f32, value.im as f32));
impl_cast_scalar!(String => Complex<f32>, |value| Complex::new(value.parse::<f32>().unwrap_or(f32::NAN), 0.0));

impl_cast_scalar!(bool => Complex<f64>, |value| Complex::new(if value { 1.0 } else { 0.0 }, 0.0));
impl_cast_scalar!(i32 => Complex<f64>, |value| Complex::new(value as f64, 0.0));
impl_cast_scalar!(i64 => Complex<f64>, |value| Complex::new(value as f64, 0.0));
impl_cast_scalar!(f32 => Complex<f64>, |value| Complex::new(value as f64, 0.0));
impl_cast_scalar!(f64 => Complex<f64>, |value| Complex::new(value, 0.0));
impl_cast_scalar!(Complex<f32> => Complex<f64>, |value| Complex::new(value.re as f64, value.im as f64));
impl_cast_scalar!(Complex<f64> => Complex<f64>, |value| value);
impl_cast_scalar!(String => Complex<f64>, |value| Complex::new(value.parse::<f64>().unwrap_or(f64::NAN), 0.0));

impl_cast_scalar!(bool => String, |value| value.to_string());
impl_cast_scalar!(i32 => String, |value| value.to_string());
impl_cast_scalar!(i64 => String, |value| value.to_string());
impl_cast_scalar!(f32 => String, |value| float_to_str(value as f64));
impl_cast_scalar!(f64 => String, |value| float_to_str(value));
impl_cast_scalar!(Complex<f32> => String, |value| complex_to_str(value));
impl_cast_scalar!(Complex<f64> => String, |value| complex_to_str(value));
impl_cast_scalar!(String => String, |value| value);

fn cast_array_storage<T>(data: &ArrayData) -> ArrayD<T>
where
    bool: CastScalar<T>,
    i32: CastScalar<T>,
    i64: CastScalar<T>,
    f32: CastScalar<T>,
    f64: CastScalar<T>,
    Complex<f32>: CastScalar<T>,
    Complex<f64>: CastScalar<T>,
    String: CastScalar<T>,
    T: Clone,
{
    match data {
        ArrayData::Bool(a) => a.mapv(<bool as CastScalar<T>>::cast_scalar).into_shared(),
        ArrayData::Int32(a) => a.mapv(<i32 as CastScalar<T>>::cast_scalar).into_shared(),
        ArrayData::Int64(a) => a.mapv(<i64 as CastScalar<T>>::cast_scalar).into_shared(),
        ArrayData::Float32(a) => a.mapv(<f32 as CastScalar<T>>::cast_scalar).into_shared(),
        ArrayData::Float64(a) => a.mapv(<f64 as CastScalar<T>>::cast_scalar).into_shared(),
        ArrayData::Complex64(a) => a
            .mapv(<Complex<f32> as CastScalar<T>>::cast_scalar)
            .into_shared(),
        ArrayData::Complex128(a) => a
            .mapv(<Complex<f64> as CastScalar<T>>::cast_scalar)
            .into_shared(),
        ArrayData::Str(a) => a.mapv(<String as CastScalar<T>>::cast_scalar).into_shared(),
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
        DType::Bool => ArrayData::Bool(cast_array_storage::<bool>(data)),
        DType::Int32 => ArrayData::Int32(cast_array_storage::<i32>(data)),
        DType::Int64 => ArrayData::Int64(cast_array_storage::<i64>(data)),
        DType::Float32 => ArrayData::Float32(cast_array_storage::<f32>(data)),
        DType::Float64 => ArrayData::Float64(cast_array_storage::<f64>(data)),
        DType::Complex64 => ArrayData::Complex64(cast_array_storage::<Complex<f32>>(data)),
        DType::Complex128 => ArrayData::Complex128(cast_array_storage::<Complex<f64>>(data)),
        DType::Str => ArrayData::Str(cast_array_storage::<String>(data)),
        _ => unreachable!("storage_dtype maps to canonical types"),
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
