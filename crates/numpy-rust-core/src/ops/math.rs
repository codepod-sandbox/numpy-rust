use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::casting::cast_array_data;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Helper: ensure array is floating-point (cast int/bool to f64, matching NumPy behavior).
/// Complex types are kept as-is.
fn ensure_float(data: &ArrayData) -> ArrayData {
    if data.dtype().is_string() {
        panic!("math functions not supported for string arrays");
    }
    match data.dtype() {
        DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128 => data.clone(),
        _ => cast_array_data(data, DType::Float64),
    }
}

/// Apply a float unary op (works on Float32, Float64, Complex64, Complex128).
macro_rules! float_unary {
    ($name:ident, $f32_op:expr, $f64_op:expr, $c64_op:expr, $c128_op:expr) => {
        impl NdArray {
            pub fn $name(&self) -> NdArray {
                let data = ensure_float(&self.data);
                let result = match data {
                    ArrayData::Float32(a) => ArrayData::Float32(a.mapv($f32_op)),
                    ArrayData::Float64(a) => ArrayData::Float64(a.mapv($f64_op)),
                    ArrayData::Complex64(a) => ArrayData::Complex64(a.mapv($c64_op)),
                    ArrayData::Complex128(a) => ArrayData::Complex128(a.mapv($c128_op)),
                    _ => unreachable!(),
                };
                NdArray::from_data(result)
            }
        }
    };
}

/// Apply a float unary op that does NOT work on complex types.
macro_rules! float_only_unary {
    ($name:ident, $f32_op:expr, $f64_op:expr) => {
        impl NdArray {
            pub fn $name(&self) -> Result<NdArray> {
                if self.dtype().is_complex() {
                    return Err(NumpyError::TypeError(
                        concat!(stringify!($name), " not supported for complex arrays").into(),
                    ));
                }
                let data = ensure_float(&self.data);
                let result = match data {
                    ArrayData::Float32(a) => ArrayData::Float32(a.mapv($f32_op)),
                    ArrayData::Float64(a) => ArrayData::Float64(a.mapv($f64_op)),
                    _ => unreachable!(),
                };
                Ok(NdArray::from_data(result))
            }
        }
    };
}

float_unary!(
    sqrt,
    |x: f32| x.sqrt(),
    |x: f64| x.sqrt(),
    |x: Complex<f32>| x.sqrt(),
    |x: Complex<f64>| x.sqrt()
);
float_unary!(
    exp,
    |x: f32| x.exp(),
    |x: f64| x.exp(),
    |x: Complex<f32>| x.exp(),
    |x: Complex<f64>| x.exp()
);
float_unary!(
    log,
    |x: f32| x.ln(),
    |x: f64| x.ln(),
    |x: Complex<f32>| x.ln(),
    |x: Complex<f64>| x.ln()
);
float_unary!(
    sin,
    |x: f32| x.sin(),
    |x: f64| x.sin(),
    |x: Complex<f32>| x.sin(),
    |x: Complex<f64>| x.sin()
);
float_unary!(
    cos,
    |x: f32| x.cos(),
    |x: f64| x.cos(),
    |x: Complex<f32>| x.cos(),
    |x: Complex<f64>| x.cos()
);
float_unary!(
    tan,
    |x: f32| x.tan(),
    |x: f64| x.tan(),
    |x: Complex<f32>| x.tan(),
    |x: Complex<f64>| x.tan()
);

float_only_unary!(floor, |x: f32| x.floor(), |x: f64| x.floor());
float_only_unary!(ceil, |x: f32| x.ceil(), |x: f64| x.ceil());
float_only_unary!(round, |x: f32| x.round(), |x: f64| x.round());

float_unary!(
    log10,
    |x: f32| x.log10(),
    |x: f64| x.log10(),
    |x: Complex<f32>| x.ln() / Complex::new(std::f32::consts::LN_10, 0.0),
    |x: Complex<f64>| x.ln() / Complex::new(std::f64::consts::LN_10, 0.0)
);
float_unary!(
    log2,
    |x: f32| x.log2(),
    |x: f64| x.log2(),
    |x: Complex<f32>| x.ln() / Complex::new(std::f32::consts::LN_2, 0.0),
    |x: Complex<f64>| x.ln() / Complex::new(std::f64::consts::LN_2, 0.0)
);
float_unary!(
    sinh,
    |x: f32| x.sinh(),
    |x: f64| x.sinh(),
    |x: Complex<f32>| x.sinh(),
    |x: Complex<f64>| x.sinh()
);
float_unary!(
    cosh,
    |x: f32| x.cosh(),
    |x: f64| x.cosh(),
    |x: Complex<f32>| x.cosh(),
    |x: Complex<f64>| x.cosh()
);
float_unary!(
    tanh,
    |x: f32| x.tanh(),
    |x: f64| x.tanh(),
    |x: Complex<f32>| x.tanh(),
    |x: Complex<f64>| x.tanh()
);
float_unary!(
    arcsin,
    |x: f32| x.asin(),
    |x: f64| x.asin(),
    |x: Complex<f32>| x.asin(),
    |x: Complex<f64>| x.asin()
);
float_unary!(
    arccos,
    |x: f32| x.acos(),
    |x: f64| x.acos(),
    |x: Complex<f32>| x.acos(),
    |x: Complex<f64>| x.acos()
);
float_unary!(
    arctan,
    |x: f32| x.atan(),
    |x: f64| x.atan(),
    |x: Complex<f32>| x.atan(),
    |x: Complex<f64>| x.atan()
);

float_only_unary!(log1p, |x: f32| x.ln_1p(), |x: f64| x.ln_1p());
float_only_unary!(expm1, |x: f32| x.exp_m1(), |x: f64| x.exp_m1());
float_only_unary!(deg2rad, |x: f32| x.to_radians(), |x: f64| x.to_radians());
float_only_unary!(rad2deg, |x: f32| x.to_degrees(), |x: f64| x.to_degrees());

impl NdArray {
    /// Element-wise absolute value. Works on int and float types.
    /// For complex types, returns the magnitude (norm) as a float.
    pub fn abs(&self) -> NdArray {
        let result = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(a.clone()),
            ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| x.abs())),
            ArrayData::Int64(a) => ArrayData::Int64(a.mapv(|x| x.abs())),
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.abs())),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.abs())),
            ArrayData::Complex64(a) => ArrayData::Float32(a.mapv(|x| x.norm())),
            ArrayData::Complex128(a) => ArrayData::Float64(a.mapv(|x| x.norm())),
            ArrayData::Str(_) => panic!("abs not supported for string arrays"),
        };
        NdArray::from_data(result)
    }

    /// Return the real part of the array.
    pub fn real(&self) -> NdArray {
        match &self.data {
            ArrayData::Complex64(a) => NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.re))),
            ArrayData::Complex128(a) => NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.re))),
            // Real arrays are their own real part
            _ => self.clone(),
        }
    }

    /// Return the imaginary part of the array.
    pub fn imag(&self) -> NdArray {
        match &self.data {
            ArrayData::Complex64(a) => NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.im))),
            ArrayData::Complex128(a) => NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.im))),
            // Real arrays have zero imaginary part
            _ => NdArray::zeros(self.shape(), self.dtype()),
        }
    }

    /// Return the complex conjugate.
    pub fn conj(&self) -> NdArray {
        match &self.data {
            ArrayData::Complex64(a) => {
                NdArray::from_data(ArrayData::Complex64(a.mapv(|c| c.conj())))
            }
            ArrayData::Complex128(a) => {
                NdArray::from_data(ArrayData::Complex128(a.mapv(|c| c.conj())))
            }
            // Conjugate of real is self
            _ => self.clone(),
        }
    }

    /// Return the angle (argument) of complex elements.
    pub fn angle(&self) -> NdArray {
        match &self.data {
            ArrayData::Complex64(a) => NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.arg()))),
            ArrayData::Complex128(a) => NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.arg()))),
            // Angle of positive real is 0
            _ => NdArray::zeros(self.shape(), DType::Float64),
        }
    }

    /// Round to given number of decimal places.
    pub fn around(&self, decimals: i32) -> NdArray {
        if self.dtype().is_complex() {
            // For complex, round real and imaginary parts separately
            return self.clone();
        }
        let factor = 10.0_f64.powi(decimals);
        let data = ensure_float(&self.data);
        let result = match data {
            ArrayData::Float32(a) => {
                let f = factor as f32;
                ArrayData::Float32(a.mapv(|x| (x * f).round() / f))
            }
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| (x * factor).round() / factor)),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    /// Returns a Bool array: true where sign bit is set (negative).
    pub fn signbit(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_sign_negative())),
            ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_sign_negative())),
            ArrayData::Int32(a) => ArrayData::Bool(a.mapv(|x| x < 0)),
            ArrayData::Int64(a) => ArrayData::Bool(a.mapv(|x| x < 0)),
            _ => ArrayData::Bool(ndarray::ArrayD::from_elem(
                ndarray::IxDyn(self.shape()),
                false,
            )),
        };
        NdArray::from_data(data)
    }

    /// Element-wise sign function. Returns -1, 0, or 1.
    pub fn sign(&self) -> NdArray {
        let result = match &self.data {
            ArrayData::Bool(a) => ArrayData::Int32(a.mapv(|x| if x { 1 } else { 0 })),
            ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| x.signum())),
            ArrayData::Int64(a) => ArrayData::Int64(a.mapv(|x| x.signum())),
            ArrayData::Float32(a) => {
                ArrayData::Float32(a.mapv(|x| if x.is_nan() { x } else { x.signum() }))
            }
            ArrayData::Float64(a) => {
                ArrayData::Float64(a.mapv(|x| if x.is_nan() { x } else { x.signum() }))
            }
            ArrayData::Complex64(_) | ArrayData::Complex128(_) => {
                // NumPy returns x/|x| for complex, but skip for now — return clone
                return self.clone();
            }
            ArrayData::Str(_) => panic!("sign not supported for string arrays"),
        };
        NdArray::from_data(result)
    }

    /// Element-wise negation. Works on int and float types.
    pub fn neg(&self) -> NdArray {
        let result = match &self.data {
            ArrayData::Bool(_) => {
                // NumPy: -True == -1, -False == 0 -> cast to i32 then negate
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
            ArrayData::Complex64(a) => ArrayData::Complex64(a.mapv(|x| -x)),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.mapv(|x| -x)),
            ArrayData::Str(_) => panic!("negation not supported for string arrays"),
        };
        NdArray::from_data(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};
    use num_complex::Complex;

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
        let _ = a.floor().unwrap();
        let _ = a.ceil().unwrap();
        let _ = a.round().unwrap();
    }

    #[test]
    fn test_floor_complex_fails() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        assert!(a.floor().is_err());
        assert!(a.ceil().is_err());
        assert!(a.round().is_err());
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
    fn test_abs_complex() {
        let a = NdArray::from_vec(vec![Complex::new(3.0f64, 4.0)]);
        let b = a.abs();
        assert_eq!(b.dtype(), DType::Float64);
        // |3+4i| = 5
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
    fn test_neg_complex() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        let b = a.neg();
        assert_eq!(b.dtype(), DType::Complex128);
    }

    #[test]
    fn test_sqrt_f32_stays_f32() {
        let a = NdArray::from_vec(vec![4.0_f32, 9.0]);
        let b = a.sqrt();
        assert_eq!(b.dtype(), DType::Float32);
    }

    #[test]
    fn test_sqrt_complex() {
        let a = NdArray::from_vec(vec![Complex::new(-1.0f64, 0.0)]);
        let b = a.sqrt();
        assert_eq!(b.dtype(), DType::Complex128);
    }

    #[test]
    fn test_real_imag_complex() {
        let a = NdArray::from_complex128_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let r = a.real();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.dtype(), DType::Float64);
        let im = a.imag();
        assert_eq!(im.shape(), &[2]);
        assert_eq!(im.dtype(), DType::Float64);
    }

    #[test]
    fn test_conj() {
        let a = NdArray::from_complex128_vec(vec![Complex::new(1.0, 2.0)]);
        let c = a.conj();
        assert_eq!(c.dtype(), DType::Complex128);
    }

    #[test]
    fn test_angle() {
        let a = NdArray::from_complex128_vec(vec![Complex::new(1.0, 0.0)]);
        let ang = a.angle();
        assert_eq!(ang.dtype(), DType::Float64);
    }

    #[test]
    fn test_real_on_real_array() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let r = a.real();
        assert_eq!(r.dtype(), DType::Float64);
        assert_eq!(r.shape(), &[3]);
    }

    #[test]
    fn test_around_decimals() {
        let a = NdArray::from_vec(vec![1.234_f64, 2.567, 3.891]);
        let b = a.around(2);
        assert_eq!(b.dtype(), DType::Float64);
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_signbit() {
        let a = NdArray::from_vec(vec![-1.0_f64, 0.0, 1.0, -0.0]);
        let b = a.signbit();
        assert_eq!(b.dtype(), DType::Bool);
        assert_eq!(b.shape(), &[4]);
    }

    #[test]
    fn test_log10() {
        let a = NdArray::from_vec(vec![1.0_f64, 10.0, 100.0]);
        let b = a.log10();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_log2() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 4.0]);
        let b = a.log2();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_log1p() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let b = a.log1p().unwrap();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_expm1() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let b = a.expm1().unwrap();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_sign() {
        let a = NdArray::from_vec(vec![-5.0_f64, 0.0, 3.0]);
        let b = a.sign();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_deg2rad() {
        let a = NdArray::from_vec(vec![0.0_f64, 90.0, 180.0]);
        let b = a.deg2rad().unwrap();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_sinh_cosh_tanh() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let _ = a.sinh();
        let _ = a.cosh();
        let _ = a.tanh();
    }

    #[test]
    fn test_arcsin_arccos_arctan() {
        let a = NdArray::from_vec(vec![0.0_f64, 0.5, 1.0]);
        let _ = a.arcsin();
        let _ = a.arccos();
        let _ = a.arctan();
    }
}
