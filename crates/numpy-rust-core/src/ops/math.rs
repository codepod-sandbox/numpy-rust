use crate::array_data::ArrayData;
use crate::casting::cast_array_data;
use crate::dtype::DType;
use crate::NdArray;

/// Helper: ensure array is floating-point (cast int/bool to f64, matching NumPy behavior).
fn ensure_float(data: &ArrayData) -> ArrayData {
    if data.dtype().is_string() {
        panic!("math functions not supported for string arrays");
    }
    match data.dtype() {
        DType::Float32 | DType::Float64 => data.clone(),
        _ => cast_array_data(data, DType::Float64),
    }
}

/// Apply a float unary op, returning the result in the same float dtype.
macro_rules! float_unary {
    ($name:ident, $f32_op:expr, $f64_op:expr) => {
        impl NdArray {
            pub fn $name(&self) -> NdArray {
                let data = ensure_float(&self.data);
                let result = match data {
                    ArrayData::Float32(a) => ArrayData::Float32(a.mapv($f32_op)),
                    ArrayData::Float64(a) => ArrayData::Float64(a.mapv($f64_op)),
                    _ => unreachable!(),
                };
                NdArray::from_data(result)
            }
        }
    };
}

float_unary!(sqrt, |x: f32| x.sqrt(), |x: f64| x.sqrt());
float_unary!(exp, |x: f32| x.exp(), |x: f64| x.exp());
float_unary!(log, |x: f32| x.ln(), |x: f64| x.ln());
float_unary!(sin, |x: f32| x.sin(), |x: f64| x.sin());
float_unary!(cos, |x: f32| x.cos(), |x: f64| x.cos());
float_unary!(tan, |x: f32| x.tan(), |x: f64| x.tan());
float_unary!(floor, |x: f32| x.floor(), |x: f64| x.floor());
float_unary!(ceil, |x: f32| x.ceil(), |x: f64| x.ceil());
float_unary!(round, |x: f32| x.round(), |x: f64| x.round());

impl NdArray {
    /// Element-wise absolute value. Works on int and float types.
    pub fn abs(&self) -> NdArray {
        let result = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(a.clone()),
            ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| x.abs())),
            ArrayData::Int64(a) => ArrayData::Int64(a.mapv(|x| x.abs())),
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.abs())),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.abs())),
            ArrayData::Str(_) => panic!("abs not supported for string arrays"),
        };
        NdArray::from_data(result)
    }

    /// Element-wise negation. Works on int and float types.
    pub fn neg(&self) -> NdArray {
        let result = match &self.data {
            ArrayData::Bool(_) => {
                // NumPy: -True == -1, -False == 0 → cast to i32 then negate
                let cast = cast_array_data(&self.data, DType::Int32);
                match cast {
                    ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| -x)),
                    _ => unreachable!(),
                }
            }
            ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| -x)),
            ArrayData::Int64(a) => ArrayData::Int64(a.mapv(|x| -x)),
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| -x)),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| -x)),
            ArrayData::Str(_) => panic!("negation not supported for string arrays"),
        };
        NdArray::from_data(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_sqrt() {
        let a = NdArray::from_vec(vec![4.0_f64, 9.0, 16.0]);
        let b = a.sqrt();
        assert_eq!(b.dtype(), DType::Float64);
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_sqrt_int_casts_to_float() {
        let a = NdArray::from_vec(vec![4_i32, 9, 16]);
        let b = a.sqrt();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_exp() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let b = a.exp();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_log() {
        let a = NdArray::from_vec(vec![1.0_f64, std::f64::consts::E]);
        let b = a.log();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_sin_cos_tan() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let _ = a.sin();
        let _ = a.cos();
        let _ = a.tan();
    }

    #[test]
    fn test_floor_ceil_round() {
        let a = NdArray::from_vec(vec![1.3_f64, 2.7, -0.5]);
        let _ = a.floor();
        let _ = a.ceil();
        let _ = a.round();
    }

    #[test]
    fn test_abs_float() {
        let a = NdArray::from_vec(vec![-1.0_f64, 2.0, -3.0]);
        let b = a.abs();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_abs_int() {
        let a = NdArray::from_vec(vec![-1_i32, 2, -3]);
        let b = a.abs();
        assert_eq!(b.dtype(), DType::Int32);
    }

    #[test]
    fn test_neg() {
        let a = NdArray::from_vec(vec![1.0_f64, -2.0, 3.0]);
        let b = a.neg();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_neg_int() {
        let a = NdArray::from_vec(vec![1_i32, -2, 3]);
        let b = a.neg();
        assert_eq!(b.dtype(), DType::Int32);
    }

    #[test]
    fn test_neg_bool() {
        let a = NdArray::from_vec(vec![true, false]);
        let b = a.neg();
        assert_eq!(b.dtype(), DType::Int32);
    }

    #[test]
    fn test_sqrt_f32_stays_f32() {
        let a = NdArray::from_vec(vec![4.0_f32, 9.0]);
        let b = a.sqrt();
        assert_eq!(b.dtype(), DType::Float32);
    }
}
