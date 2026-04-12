use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::kernel::{
    DecomposeUnaryKernelOp, MathBinaryKernelOp, MathUnaryKernelOp, RealBinaryKernelOp,
    RealUnaryKernelOp, ValueUnaryKernelOp,
};
use crate::NdArray;

fn map_complex_data<R>(
    data: &ArrayData,
    on64: impl FnOnce(&ndarray::ArcArray<Complex<f32>, ndarray::IxDyn>) -> R,
    on128: impl FnOnce(&ndarray::ArcArray<Complex<f64>, ndarray::IxDyn>) -> R,
) -> Option<R> {
    match data {
        ArrayData::Complex64(a) => Some(on64(a)),
        ArrayData::Complex128(a) => Some(on128(a)),
        _ => None,
    }
}

fn update_complex32_component(
    current: &ndarray::ArcArray<Complex<f32>, ndarray::IxDyn>,
    values: &NdArray,
    update: impl Fn(Complex<f32>, f32) -> Complex<f32>,
) -> Option<ArrayData> {
    let values = values.astype(DType::Float32);
    let ArrayData::Float32(values) = values.data() else {
        return None;
    };

    let replacement_values: Vec<f32> = values.iter().copied().collect();
    let new_data: Vec<_> = if replacement_values.len() == 1 {
        current
            .iter()
            .map(|&c| update(c, replacement_values[0]))
            .collect()
    } else {
        current
            .iter()
            .zip(replacement_values.iter())
            .map(|(&c, &v)| update(c, v))
            .collect()
    };

    ndarray::Array::from_shape_vec(ndarray::IxDyn(current.shape()), new_data)
        .ok()
        .map(|arr| ArrayData::Complex64(arr.into_shared()))
}

fn update_complex128_component(
    current: &ndarray::ArcArray<Complex<f64>, ndarray::IxDyn>,
    values: &NdArray,
    update: impl Fn(Complex<f64>, f64) -> Complex<f64>,
) -> Option<ArrayData> {
    let values = values.astype(DType::Float64);
    let ArrayData::Float64(values) = values.data() else {
        return None;
    };

    let replacement_values: Vec<f64> = values.iter().copied().collect();
    let new_data: Vec<_> = if replacement_values.len() == 1 {
        current
            .iter()
            .map(|&c| update(c, replacement_values[0]))
            .collect()
    } else {
        current
            .iter()
            .zip(replacement_values.iter())
            .map(|(&c, &v)| update(c, v))
            .collect()
    };

    ndarray::Array::from_shape_vec(ndarray::IxDyn(current.shape()), new_data)
        .ok()
        .map(|arr| ArrayData::Complex128(arr.into_shared()))
}

/// Helper: ensure array is floating-point (cast int/bool to f64, matching NumPy behavior).
/// Complex types are kept as-is.
fn ensure_float(data: &ArrayData) -> ArrayData {
    if data.dtype().is_string() {
        // Cast string to float64 — non-numeric strings become NaN
        return cast_array_data(data, DType::Float64);
    }
    match data.dtype() {
        DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128 => data.clone(),
        _ => cast_array_data(data, DType::Float64),
    }
}

fn execute_math_unary_on_data(data: ArrayData, op: MathUnaryKernelOp) -> NdArray {
    let descriptor = descriptor_for_dtype(data.dtype());
    let kernel = descriptor
        .math_unary_kernel(op)
        .unwrap_or_else(|| panic!("math unary kernel not registered for {}", data.dtype()));
    NdArray::from_data(kernel(data).expect("math unary kernel dtype mismatch"))
}

fn execute_math_unary(input: &NdArray, op: MathUnaryKernelOp) -> NdArray {
    execute_math_unary_on_data(ensure_float(input.data()), op)
}

fn execute_value_unary(
    input: &NdArray,
    op: ValueUnaryKernelOp,
    preserve_descriptor: bool,
) -> NdArray {
    let descriptor = input.descriptor();
    let kernel = descriptor
        .value_unary_kernel(op)
        .unwrap_or_else(|| panic!("value unary kernel not registered for {}", input.dtype()));
    let data = kernel(input.data().clone()).expect("value unary kernel dtype mismatch");
    let mut result = NdArray::from_data(data);
    if preserve_descriptor && input.dtype().storage_dtype() == result.data().dtype() {
        result.preserve_descriptor_from(input);
    }
    result
}

fn execute_real_math_unary(
    input: &NdArray,
    op: MathUnaryKernelOp,
    op_name: &'static str,
) -> Result<NdArray> {
    if input.dtype().is_complex() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for complex arrays"
        )));
    }
    Ok(execute_math_unary(input, op))
}

fn execute_real_math_binary(
    lhs: &NdArray,
    rhs: &NdArray,
    op: MathBinaryKernelOp,
    op_name: &'static str,
) -> Result<NdArray> {
    if lhs.dtype().is_complex() || rhs.dtype().is_complex() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for complex arrays"
        )));
    }

    let lhs_data = ensure_float(lhs.data());
    let rhs_data = ensure_float(rhs.data());
    let execution_dtype = match (&lhs_data, &rhs_data) {
        (ArrayData::Float32(_), ArrayData::Float32(_)) => DType::Float32,
        _ => DType::Float64,
    };
    let lhs_data = cast_array_data(&lhs_data, execution_dtype);
    let rhs_data = cast_array_data(&rhs_data, execution_dtype);
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;
    let lhs_data = broadcast_array_data(&lhs_data, &out_shape);
    let rhs_data = broadcast_array_data(&rhs_data, &out_shape);

    let descriptor = descriptor_for_dtype(execution_dtype);
    let kernel = descriptor
        .math_binary_kernel(op)
        .unwrap_or_else(|| panic!("math binary kernel not registered for {execution_dtype}"));
    Ok(NdArray::from_data(
        kernel(lhs_data, rhs_data).expect("math binary kernel dtype mismatch"),
    ))
}

fn execute_real_unary(
    input: &NdArray,
    op: RealUnaryKernelOp,
    op_name: &'static str,
) -> Result<NdArray> {
    if input.dtype().is_complex() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for complex arrays"
        )));
    }
    let data = ensure_float(input.data());
    let descriptor = descriptor_for_dtype(data.dtype());
    let kernel = descriptor
        .real_unary_kernel(op)
        .unwrap_or_else(|| panic!("real unary kernel not registered for {}", data.dtype()));
    Ok(NdArray::from_data(
        kernel(data).expect("real unary kernel dtype mismatch"),
    ))
}

fn execute_real_binary(
    lhs: &NdArray,
    rhs: &NdArray,
    op: RealBinaryKernelOp,
    op_name: &'static str,
) -> Result<NdArray> {
    if lhs.dtype().is_complex() || rhs.dtype().is_complex() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for complex arrays"
        )));
    }

    let lhs_data = ensure_float(lhs.data());
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;
    let execution_dtype = match op {
        RealBinaryKernelOp::ArcTan2 => match (&lhs_data, &ensure_float(rhs.data())) {
            (ArrayData::Float32(_), ArrayData::Float32(_)) => DType::Float32,
            _ => DType::Float64,
        },
        RealBinaryKernelOp::LDExp => match lhs_data {
            ArrayData::Float32(_) => DType::Float32,
            _ => DType::Float64,
        },
    };

    let lhs_data = cast_array_data(&lhs_data, execution_dtype);
    let lhs_data = broadcast_array_data(&lhs_data, &out_shape);
    let rhs_data = match op {
        RealBinaryKernelOp::ArcTan2 => {
            let rhs_data = ensure_float(rhs.data());
            let rhs_data = cast_array_data(&rhs_data, execution_dtype);
            broadcast_array_data(&rhs_data, &out_shape)
        }
        RealBinaryKernelOp::LDExp => {
            let rhs_data = cast_array_data(rhs.data(), DType::Int32);
            broadcast_array_data(&rhs_data, &out_shape)
        }
    };

    let descriptor = descriptor_for_dtype(execution_dtype);
    let kernel = descriptor
        .real_binary_kernel(op)
        .unwrap_or_else(|| panic!("real binary kernel not registered for {execution_dtype}"));
    Ok(NdArray::from_data(
        kernel(lhs_data, rhs_data).expect("real binary kernel dtype mismatch"),
    ))
}

fn execute_real_decompose(
    input: &NdArray,
    op: DecomposeUnaryKernelOp,
    op_name: &'static str,
) -> Result<(NdArray, NdArray)> {
    if input.dtype().is_complex() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for complex arrays"
        )));
    }
    let data = ensure_float(input.data());
    let descriptor = descriptor_for_dtype(data.dtype());
    let kernel = descriptor
        .decompose_unary_kernel(op)
        .unwrap_or_else(|| panic!("decompose unary kernel not registered for {}", data.dtype()));
    let (first, second) = kernel(data).expect("decompose unary kernel dtype mismatch");
    Ok((NdArray::from_data(first), NdArray::from_data(second)))
}

/// Apply a float unary op (works on Float32, Float64, Complex64, Complex128).
macro_rules! float_unary {
    ($name:ident, $op:expr) => {
        impl NdArray {
            pub fn $name(&self) -> NdArray {
                execute_math_unary(self, $op)
            }
        }
    };
}

/// Apply a float unary op that does NOT work on complex types.
macro_rules! float_only_unary {
    ($name:ident, $op:expr) => {
        impl NdArray {
            pub fn $name(&self) -> Result<NdArray> {
                execute_real_math_unary(self, $op, stringify!($name))
            }
        }
    };
}

float_unary!(sqrt, MathUnaryKernelOp::Sqrt);
float_unary!(exp, MathUnaryKernelOp::Exp);
float_unary!(log, MathUnaryKernelOp::Log);
float_unary!(sin, MathUnaryKernelOp::Sin);
float_unary!(cos, MathUnaryKernelOp::Cos);
float_unary!(tan, MathUnaryKernelOp::Tan);

float_only_unary!(floor, MathUnaryKernelOp::Floor);
float_only_unary!(ceil, MathUnaryKernelOp::Ceil);
float_only_unary!(round, MathUnaryKernelOp::Round);

float_unary!(log10, MathUnaryKernelOp::Log10);
float_unary!(log2, MathUnaryKernelOp::Log2);
float_unary!(sinh, MathUnaryKernelOp::Sinh);
float_unary!(cosh, MathUnaryKernelOp::Cosh);
float_unary!(tanh, MathUnaryKernelOp::Tanh);
float_unary!(arcsin, MathUnaryKernelOp::ArcSin);
float_unary!(arccos, MathUnaryKernelOp::ArcCos);
float_unary!(arctan, MathUnaryKernelOp::ArcTan);

float_unary!(arcsinh, MathUnaryKernelOp::ArcSinh);
float_unary!(arccosh, MathUnaryKernelOp::ArcCosh);
float_unary!(arctanh, MathUnaryKernelOp::ArcTanh);

float_only_unary!(log1p, MathUnaryKernelOp::Log1p);
float_only_unary!(expm1, MathUnaryKernelOp::Expm1);
float_only_unary!(deg2rad, MathUnaryKernelOp::Deg2Rad);
float_only_unary!(rad2deg, MathUnaryKernelOp::Rad2Deg);
float_only_unary!(trunc, MathUnaryKernelOp::Trunc);

// --- libm-backed unary functions ---
// These use float_only_unary! (no complex support).
// libm functions return NaN for out-of-domain inputs; no panics.
float_only_unary!(cbrt, MathUnaryKernelOp::Cbrt);
float_only_unary!(gamma, MathUnaryKernelOp::Gamma);
float_only_unary!(lgamma, MathUnaryKernelOp::LGamma);
float_only_unary!(erf, MathUnaryKernelOp::Erf);
float_only_unary!(erfc, MathUnaryKernelOp::Erfc);
float_only_unary!(j0, MathUnaryKernelOp::J0);
float_only_unary!(j1, MathUnaryKernelOp::J1);
float_only_unary!(y0, MathUnaryKernelOp::Y0);
float_only_unary!(y1, MathUnaryKernelOp::Y1);

// --- Binary float math macro ---
macro_rules! math_binary {
    ($name:ident, $op:expr) => {
        impl NdArray {
            pub fn $name(&self, other: &NdArray) -> Result<NdArray> {
                execute_real_math_binary(self, other, $op, stringify!($name))
            }
        }
    };
}

math_binary!(copysign, MathBinaryKernelOp::CopySign);
math_binary!(hypot, MathBinaryKernelOp::Hypot);
math_binary!(fmod, MathBinaryKernelOp::FMod);
math_binary!(nextafter, MathBinaryKernelOp::NextAfter);
math_binary!(logaddexp, MathBinaryKernelOp::LogAddExp);
math_binary!(logaddexp2, MathBinaryKernelOp::LogAddExp2);
math_binary!(fmax, MathBinaryKernelOp::FMax);
math_binary!(fmin, MathBinaryKernelOp::FMin);
math_binary!(maximum, MathBinaryKernelOp::Maximum);
math_binary!(minimum, MathBinaryKernelOp::Minimum);

impl NdArray {
    /// ldexp(x, n) = x * 2^n, element-wise. n is cast to Int32.
    pub fn ldexp(&self, exp_arr: &NdArray) -> Result<NdArray> {
        execute_real_binary(self, exp_arr, RealBinaryKernelOp::LDExp, "ldexp")
    }
}

impl NdArray {
    /// Element-wise absolute value. Works on int and float types.
    /// For complex types, returns the magnitude (norm) as a float.
    pub fn abs(&self) -> NdArray {
        let result = match self.data() {
            ArrayData::Bool(a) => ArrayData::Bool(a.clone()),
            ArrayData::Int32(a) => ArrayData::Int32(a.mapv(|x| x.abs()).into_shared()),
            ArrayData::Int64(a) => ArrayData::Int64(a.mapv(|x| x.abs()).into_shared()),
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.abs()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.abs()).into_shared()),
            ArrayData::Complex64(a) => ArrayData::Float32(a.mapv(|x| x.norm()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Float64(a.mapv(|x| x.norm()).into_shared()),
            ArrayData::Str(_) => panic!("abs not supported for string arrays"),
        };
        let mut r = NdArray::from_data(result);
        // Preserve narrow dtype for non-complex (complex abs returns float)
        if !self.dtype().is_complex() {
            r.preserve_descriptor_from(self);
        }
        r
    }

    /// Return the real part of the array.
    pub fn real(&self) -> NdArray {
        map_complex_data(
            self.data(),
            |a| NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.re).into_shared())),
            |a| NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.re).into_shared())),
        )
        .unwrap_or_else(|| self.clone())
    }

    /// Return the imaginary part of the array.
    pub fn imag(&self) -> NdArray {
        map_complex_data(
            self.data(),
            |a| NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.im).into_shared())),
            |a| NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.im).into_shared())),
        )
        .unwrap_or_else(|| NdArray::zeros(self.shape(), self.dtype()))
    }

    /// Set the real part of the array from `real_values` (in-place replacement).
    pub fn set_real(&mut self, real_values: &NdArray) {
        match self.data() {
            ArrayData::Complex64(a) => {
                if let Some(new_data) =
                    update_complex32_component(a, real_values, |c, re| Complex::new(re, c.im))
                {
                    self.replace_data_with_dtype(new_data, DType::Complex64);
                }
            }
            ArrayData::Complex128(a) => {
                if let Some(new_data) =
                    update_complex128_component(a, real_values, |c, re| Complex::new(re, c.im))
                {
                    self.replace_data_with_dtype(new_data, DType::Complex128);
                }
            }
            _ => {
                // For real arrays: set_real copies values in
                *self = real_values.astype(self.dtype());
            }
        }
    }

    /// Set the imaginary part of the array from `imag_values` (in-place replacement).
    pub fn set_imag(&mut self, imag_values: &NdArray) {
        match self.data() {
            ArrayData::Complex64(a) => {
                if let Some(new_data) =
                    update_complex32_component(a, imag_values, |c, im| Complex::new(c.re, im))
                {
                    self.replace_data_with_dtype(new_data, DType::Complex64);
                }
            }
            ArrayData::Complex128(a) => {
                if let Some(new_data) =
                    update_complex128_component(a, imag_values, |c, im| Complex::new(c.re, im))
                {
                    self.replace_data_with_dtype(new_data, DType::Complex128);
                }
            }
            _ => {
                // For real arrays: setting imag is a no-op (values are all zero)
            }
        }
    }

    /// Return the complex conjugate.
    pub fn conj(&self) -> NdArray {
        map_complex_data(
            self.data(),
            |a| NdArray::from_data(ArrayData::Complex64(a.mapv(|c| c.conj()).into_shared())),
            |a| NdArray::from_data(ArrayData::Complex128(a.mapv(|c| c.conj()).into_shared())),
        )
        .unwrap_or_else(|| self.clone())
    }

    /// Return the angle (argument) of complex elements.
    pub fn angle(&self) -> NdArray {
        map_complex_data(
            self.data(),
            |a| NdArray::from_data(ArrayData::Float32(a.mapv(|c| c.arg()).into_shared())),
            |a| NdArray::from_data(ArrayData::Float64(a.mapv(|c| c.arg()).into_shared())),
        )
        .unwrap_or_else(|| NdArray::zeros(self.shape(), DType::Float64))
    }

    /// Round to given number of decimal places.
    pub fn around(&self, decimals: i32) -> NdArray {
        if self.dtype().is_complex() {
            // For complex, round real and imaginary parts separately
            return self.clone();
        }
        let factor = 10.0_f64.powi(decimals);
        let data = ensure_float(self.data());
        let result = match data {
            ArrayData::Float32(a) => {
                let f = factor as f32;
                ArrayData::Float32(a.mapv(|x| (x * f).round() / f).into_shared())
            }
            ArrayData::Float64(a) => {
                ArrayData::Float64(a.mapv(|x| (x * factor).round() / factor).into_shared())
            }
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    /// Returns a Bool array: true where sign bit is set (negative).
    pub fn signbit(&self) -> NdArray {
        execute_value_unary(self, ValueUnaryKernelOp::SignBit, false)
    }

    /// Element-wise sign function. Returns -1, 0, or 1.
    pub fn sign(&self) -> NdArray {
        execute_value_unary(self, ValueUnaryKernelOp::Sign, false)
    }

    /// Element-wise negation. Works on int and float types.
    pub fn neg(&self) -> NdArray {
        execute_value_unary(self, ValueUnaryKernelOp::Neg, true)
    }
}

impl NdArray {
    /// Element-wise arctan2(self, other) — the angle of (other, self) from the positive x-axis.
    /// self=y, other=x. Result is in [-pi, pi].
    /// Not supported for complex arrays.
    pub fn arctan2(&self, other: &NdArray) -> Result<NdArray> {
        execute_real_binary(self, other, RealBinaryKernelOp::ArcTan2, "arctan2")
    }

    /// Clip (limit) array values to [a_min, a_max].
    pub fn clip(&self, a_min: Option<f64>, a_max: Option<f64>) -> NdArray {
        let data = ensure_float(self.data());
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(
                a.mapv(|x| {
                    let mut v = x;
                    if let Some(mn) = a_min {
                        if v < mn as f32 {
                            v = mn as f32;
                        }
                    }
                    if let Some(mx) = a_max {
                        if v > mx as f32 {
                            v = mx as f32;
                        }
                    }
                    v
                })
                .into_shared(),
            ),
            ArrayData::Float64(a) => ArrayData::Float64(
                a.mapv(|x| {
                    let mut v = x;
                    if let Some(mn) = a_min {
                        if v < mn {
                            v = mn;
                        }
                    }
                    if let Some(mx) = a_max {
                        if v > mx {
                            v = mx;
                        }
                    }
                    v
                })
                .into_shared(),
            ),
            _ => data, // bool/int/string pass through
        };
        NdArray::from_data(result)
    }

    /// Clip complex array values using lexicographic comparison.
    pub fn clip_complex(
        &self,
        a_min: Option<num_complex::Complex<f64>>,
        a_max: Option<num_complex::Complex<f64>>,
    ) -> NdArray {
        use crate::ops::comparison::complex_cmp;
        match self.data() {
            ArrayData::Complex128(a) => {
                let result = a.mapv(|x| {
                    let mut v = x;
                    if let Some(mn) = a_min {
                        if complex_cmp(&v, &mn) == std::cmp::Ordering::Less {
                            v = mn;
                        }
                    }
                    if let Some(mx) = a_max {
                        if complex_cmp(&v, &mx) == std::cmp::Ordering::Greater {
                            v = mx;
                        }
                    }
                    v
                });
                NdArray::from_data(ArrayData::Complex128(result.into_shared()))
            }
            ArrayData::Complex64(a) => {
                let mn32 = a_min.map(|c| num_complex::Complex::new(c.re as f32, c.im as f32));
                let mx32 = a_max.map(|c| num_complex::Complex::new(c.re as f32, c.im as f32));
                let result = a.mapv(|x| {
                    let mut v = x;
                    if let Some(mn) = mn32 {
                        if complex_cmp(&v, &mn) == std::cmp::Ordering::Less {
                            v = mn;
                        }
                    }
                    if let Some(mx) = mx32 {
                        if complex_cmp(&v, &mx) == std::cmp::Ordering::Greater {
                            v = mx;
                        }
                    }
                    v
                });
                NdArray::from_data(ArrayData::Complex64(result.into_shared()))
            }
            _ => {
                // Non-complex: convert bounds to f64 and use regular clip
                self.clip(a_min.map(|c| c.re), a_max.map(|c| c.re))
            }
        }
    }
}

impl NdArray {
    /// Replace NaN, +Inf, -Inf with finite values. Integer arrays pass through unchanged.
    /// nan: replacement for NaN (default 0.0)
    /// posinf: replacement for +Inf (default f64::MAX)
    /// neginf: replacement for -Inf (default -f64::MAX)
    pub fn nan_to_num(&self, nan: f64, posinf: f64, neginf: f64) -> NdArray {
        // Integer/bool/string arrays cannot have NaN or Inf — pass through unchanged
        match self.dtype() {
            crate::dtype::DType::Bool
            | crate::dtype::DType::Int32
            | crate::dtype::DType::Int64
            | crate::dtype::DType::Str => return NdArray::from_data(self.data().deep_copy()),
            _ => {}
        }
        let data = ensure_float(self.data());
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(
                a.mapv(|x| {
                    if x.is_nan() {
                        nan as f32
                    } else if x == f32::INFINITY {
                        posinf as f32
                    } else if x == f32::NEG_INFINITY {
                        neginf as f32
                    } else {
                        x
                    }
                })
                .into_shared(),
            ),
            ArrayData::Float64(a) => ArrayData::Float64(
                a.mapv(|x| {
                    if x.is_nan() {
                        nan
                    } else if x.is_infinite() && x > 0.0 {
                        posinf
                    } else if x.is_infinite() && x < 0.0 {
                        neginf
                    } else {
                        x
                    }
                })
                .into_shared(),
            ),
            ArrayData::Complex64(a) => ArrayData::Complex64(
                a.mapv(|x| {
                    let re = if x.re.is_nan() {
                        nan as f32
                    } else if x.re == f32::INFINITY {
                        posinf as f32
                    } else if x.re == f32::NEG_INFINITY {
                        neginf as f32
                    } else {
                        x.re
                    };
                    let im = if x.im.is_nan() {
                        nan as f32
                    } else if x.im == f32::INFINITY {
                        posinf as f32
                    } else if x.im == f32::NEG_INFINITY {
                        neginf as f32
                    } else {
                        x.im
                    };
                    num_complex::Complex::new(re, im)
                })
                .into_shared(),
            ),
            ArrayData::Complex128(a) => ArrayData::Complex128(
                a.mapv(|x| {
                    let re = if x.re.is_nan() {
                        nan
                    } else if x.re.is_infinite() && x.re > 0.0 {
                        posinf
                    } else if x.re.is_infinite() && x.re < 0.0 {
                        neginf
                    } else {
                        x.re
                    };
                    let im = if x.im.is_nan() {
                        nan
                    } else if x.im.is_infinite() && x.im > 0.0 {
                        posinf
                    } else if x.im.is_infinite() && x.im < 0.0 {
                        neginf
                    } else {
                        x.im
                    };
                    num_complex::Complex::new(re, im)
                })
                .into_shared(),
            ),
            _ => {
                // Bool/Int/Str already handled above; this shouldn't happen
                return NdArray::from_data(self.data().deep_copy());
            }
        };
        NdArray::from_data(result)
    }

    /// Distance between x and the nearest adjacent floating-point number.
    pub fn spacing(&self) -> Result<NdArray> {
        execute_real_unary(self, RealUnaryKernelOp::Spacing, "spacing")
    }

    /// Modified Bessel function of the first kind, order 0.
    /// Uses series expansion: I0(x) = Σ ((x/2)^k / k!)^2
    pub fn i0(&self) -> NdArray {
        execute_real_unary(self, RealUnaryKernelOp::I0, "i0").expect("i0 supports real inputs")
    }
}

impl NdArray {
    /// Decomposes each element into mantissa and base-2 exponent.
    /// Returns (mantissa: Float64, exponent: Int32), both same shape as input.
    pub fn frexp(&self) -> Result<(NdArray, NdArray)> {
        execute_real_decompose(self, DecomposeUnaryKernelOp::Frexp, "frexp")
    }

    /// Splits each element into fractional and integer parts.
    /// Returns (fractional, integer), both same dtype as input (float).
    pub fn modf(&self) -> Result<(NdArray, NdArray)> {
        execute_real_decompose(self, DecomposeUnaryKernelOp::Modf, "modf")
    }
}

/// Helper: scan Float32/Float64 array for out-of-domain values per predicate.
/// If any found, casts to Complex128; otherwise returns the float data unchanged.
fn maybe_complex(data: &ArrayData, is_out_of_domain: impl Fn(f64) -> bool) -> ArrayData {
    let should_complex = match data {
        ArrayData::Float32(a) => a.iter().any(|&x| is_out_of_domain(x as f64)),
        ArrayData::Float64(a) => a.iter().any(|&x| is_out_of_domain(x)),
        _ => false,
    };
    if should_complex {
        cast_array_data(data, DType::Complex128)
    } else {
        data.clone()
    }
}

fn execute_scimath_unary(
    input: &NdArray,
    op: MathUnaryKernelOp,
    is_out_of_domain: impl Fn(f64) -> bool,
) -> NdArray {
    let data = ensure_float(input.data());
    let data = maybe_complex(&data, is_out_of_domain);
    execute_math_unary_on_data(data, op)
}

impl NdArray {
    pub fn scimath_sqrt(&self) -> NdArray {
        execute_scimath_unary(self, MathUnaryKernelOp::Sqrt, |x| x < 0.0)
    }

    pub fn scimath_log(&self) -> NdArray {
        execute_scimath_unary(self, MathUnaryKernelOp::Log, |x| x < 0.0)
    }

    pub fn scimath_log2(&self) -> NdArray {
        execute_scimath_unary(self, MathUnaryKernelOp::Log2, |x| x < 0.0)
    }

    pub fn scimath_log10(&self) -> NdArray {
        execute_scimath_unary(self, MathUnaryKernelOp::Log10, |x| x < 0.0)
    }

    pub fn scimath_arcsin(&self) -> NdArray {
        execute_scimath_unary(self, MathUnaryKernelOp::ArcSin, |x| x.abs() > 1.0)
    }

    pub fn scimath_arccos(&self) -> NdArray {
        execute_scimath_unary(self, MathUnaryKernelOp::ArcCos, |x| x.abs() > 1.0)
    }

    pub fn scimath_arctanh(&self) -> NdArray {
        execute_scimath_unary(self, MathUnaryKernelOp::ArcTanh, |x| x.abs() > 1.0)
    }

    /// Complex-safe power: negative base → complex via powc.
    pub fn scimath_power(&self, exp_arr: &NdArray) -> Result<NdArray> {
        let data_a = ensure_float(self.data());
        let data_e = ensure_float(exp_arr.data());
        let out_shape = broadcast_shape(self.shape(), exp_arr.shape())?;
        let data_a = broadcast_array_data(&data_a, &out_shape);
        let data_e = broadcast_array_data(&data_e, &out_shape);
        // Check if any base is negative (needs complex)
        let needs_complex = match &data_a {
            ArrayData::Float32(a) => a.iter().any(|&x| x < 0.0),
            ArrayData::Float64(a) => a.iter().any(|&x| x < 0.0),
            _ => false,
        };
        if needs_complex {
            let c_a = cast_array_data(&data_a, DType::Complex128);
            let c_e = cast_array_data(&data_e, DType::Complex128);
            if let (ArrayData::Complex128(a), ArrayData::Complex128(e)) = (c_a, c_e) {
                return Ok(NdArray::from_data(ArrayData::Complex128(
                    ndarray::Zip::from(&a)
                        .and(&e)
                        .map_collect(|&b, &p| b.powc(p))
                        .into_shared(),
                )));
            }
        }
        // Normal float power
        let result = match (data_a, data_e) {
            (ArrayData::Float32(a), ArrayData::Float32(e)) => ArrayData::Float32(
                ndarray::Zip::from(&a)
                    .and(&e)
                    .map_collect(|&x, &p| x.powf(p))
                    .into_shared(),
            ),
            (ArrayData::Float64(a), ArrayData::Float64(e)) => ArrayData::Float64(
                ndarray::Zip::from(&a)
                    .and(&e)
                    .map_collect(|&x, &p| x.powf(p))
                    .into_shared(),
            ),
            _ => unreachable!(),
        };
        Ok(NdArray::from_data(result))
    }
}

#[cfg(test)]
mod tests {
    use crate::array_data::ArrayData;
    use crate::{DType, NdArray};
    use num_complex::Complex;

    /// Extract f64 values from a Float64 NdArray for assertions.
    fn f64_vals(r: &NdArray) -> Vec<f64> {
        let ArrayData::Float64(a) = r.data() else {
            panic!("expected Float64, got {:?}", r.dtype())
        };
        a.iter().copied().collect()
    }

    /// Create a Float64 NdArray from a vector of f64 values.
    fn arr(v: Vec<f64>) -> NdArray {
        NdArray::from_vec(v)
    }

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
    fn test_set_real_complex_scalar_broadcast() {
        let mut a =
            NdArray::from_complex128_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        a.set_real(&NdArray::from_vec(vec![9.0_f64]));

        let ArrayData::Complex128(arr) = a.data() else {
            panic!();
        };
        assert_eq!(arr[[0]].re, 9.0);
        assert_eq!(arr[[0]].im, 2.0);
        assert_eq!(arr[[1]].re, 9.0);
        assert_eq!(arr[[1]].im, 4.0);
    }

    #[test]
    fn test_set_imag_complex_elementwise() {
        let mut a =
            NdArray::from_complex64_vec(vec![Complex::new(1.0f32, 2.0), Complex::new(3.0, 4.0)]);
        a.set_imag(&NdArray::from_vec(vec![7.0_f32, 8.0]));

        let ArrayData::Complex64(arr) = a.data() else {
            panic!();
        };
        assert_eq!(arr[[0]].re, 1.0);
        assert_eq!(arr[[0]].im, 7.0);
        assert_eq!(arr[[1]].re, 3.0);
        assert_eq!(arr[[1]].im, 8.0);
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

    #[test]
    fn test_arcsinh_arccosh_arctanh() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let _ = a.arcsinh();
        let b = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let _ = b.arccosh();
        let c = NdArray::from_vec(vec![0.0_f64, 0.5]);
        let _ = c.arctanh();
    }

    #[test]
    fn test_trunc() {
        let a = NdArray::from_vec(vec![1.7_f64, -1.7, 0.5]);
        let b = a.trunc().unwrap();
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_trunc_complex_fails() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        assert!(a.trunc().is_err());
    }

    #[test]
    fn test_cbrt() {
        let a = arr(vec![8.0_f64, -27.0, 0.0]);
        let r = a.cbrt().unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 2.0).abs() < 1e-10, "cbrt(8) = {}", vals[0]);
        assert!((vals[1] - (-3.0)).abs() < 1e-10, "cbrt(-27) = {}", vals[1]);
        assert_eq!(vals[2], 0.0);
    }

    #[test]
    fn test_gamma() {
        let a = arr(vec![1.0_f64, 2.0, 5.0]);
        let r = a.gamma().unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 1.0).abs() < 1e-10); // gamma(1) = 1
        assert!((vals[1] - 1.0).abs() < 1e-10); // gamma(2) = 1
        assert!((vals[2] - 24.0).abs() < 1e-10); // gamma(5) = 24
    }

    #[test]
    fn test_erf() {
        let a = arr(vec![0.0_f64, 1.0, -1.0]);
        let r = a.erf().unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals[0], 0.0);
        assert!((vals[1] - 0.842_700_792_9).abs() < 1e-8);
        assert!((vals[2] + 0.842_700_792_9).abs() < 1e-8);
    }

    #[test]
    fn test_j0() {
        let a = arr(vec![0.0_f64, 1.0]);
        let r = a.j0().unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 1.0).abs() < 1e-10); // j0(0) = 1
        assert!((vals[1] - 0.765_197_686_6).abs() < 1e-8);
    }

    #[test]
    fn test_arctan2() {
        use crate::array_data::ArrayData;
        let y = NdArray::from_vec(vec![1.0_f64, 0.0, -1.0]);
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 0.0]);
        let result = y.arctan2(&x).unwrap();
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((arr[[1]] - 0.0).abs() < 1e-10);
        assert!((arr[[2]] - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-10);
    }

    #[test]
    fn test_arctan2_mixed_float_promotes_to_float64() {
        let y = NdArray::from_vec(vec![1.0_f32, -1.0]);
        let x = NdArray::from_vec(vec![1.0_f64, 1.0]);
        let result = y.arctan2(&x).unwrap();
        assert_eq!(result.dtype(), DType::Float64);
        let vals = f64_vals(&result);
        assert!((vals[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!((vals[1] + std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    }

    #[test]
    fn test_ldexp() {
        let a = arr(vec![1.5, 0.25]);
        let e = NdArray::from_vec(vec![2_i32, 3]);
        let result = a.ldexp(&e).unwrap();
        let vals = f64_vals(&result);
        assert!((vals[0] - 6.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip() {
        use crate::array_data::ArrayData;
        let a = NdArray::from_vec(vec![1.0_f64, 5.0, 10.0, -3.0]);
        let result = a.clip(Some(0.0), Some(7.0));
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]] - 1.0).abs() < 1e-10);
        assert!((arr[[1]] - 5.0).abs() < 1e-10);
        assert!((arr[[2]] - 7.0).abs() < 1e-10);
        assert!((arr[[3]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_copysign() {
        let a = arr(vec![1.0, -2.0, 3.0]);
        let b = arr(vec![-1.0, 1.0, -1.0]);
        let r = a.copysign(&b).unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_hypot() {
        let a = arr(vec![3.0, 0.0]);
        let b = arr(vec![4.0, 5.0]);
        let r = a.hypot(&b).unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 5.0).abs() < 1e-10);
        assert!((vals[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_maximum_nan_propagation() {
        let a = arr(vec![f64::NAN, 1.0, 3.0]);
        let b = arr(vec![1.0, f64::NAN, 2.0]);
        let r = a.maximum(&b).unwrap();
        let vals = f64_vals(&r);
        assert!(vals[0].is_nan()); // NAN propagates
        assert!(vals[1].is_nan()); // NAN propagates
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn test_fmax_nan_ignoring() {
        let a = arr(vec![f64::NAN, 1.0, 3.0]);
        let b = arr(vec![1.0, f64::NAN, 2.0]);
        let r = a.fmax(&b).unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals[0], 1.0); // NaN ignored
        assert_eq!(vals[1], 1.0); // NaN ignored
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn test_logaddexp() {
        // logaddexp(1, 2) = log(e^1 + e^2) ≈ 2.3132617
        let a = arr(vec![1.0]);
        let b = arr(vec![2.0]);
        let r = a.logaddexp(&b).unwrap();
        let vals = f64_vals(&r);
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((vals[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_frexp() {
        use crate::array_data::ArrayData;
        let a = arr(vec![12.0, 0.5]);
        let (m, e) = a.frexp().unwrap();
        let mv = f64_vals(&m);
        // Exponent array is Int32; extract values via ArrayData pattern match
        let ArrayData::Int32(e_arr) = e.data() else {
            panic!("expected Int32")
        };
        let ev: Vec<i32> = e_arr.iter().copied().collect();
        assert!((mv[0] - 0.75).abs() < 1e-10); // 12 = 0.75 * 2^4
        assert_eq!(ev[0], 4_i32);
        assert!((mv[1] - 0.5).abs() < 1e-10); // 0.5 = 0.5 * 2^0
        assert_eq!(ev[1], 0_i32);
    }

    #[test]
    fn test_modf() {
        let a = arr(vec![3.7, -2.5, 0.0]);
        let (frac, intg) = a.modf().unwrap();
        let fv = f64_vals(&frac);
        let iv = f64_vals(&intg);
        assert!((fv[0] - 0.7).abs() < 1e-10);
        assert_eq!(iv[0], 3.0);
        assert!((fv[1] - (-0.5)).abs() < 1e-10);
        assert_eq!(iv[1], -2.0);
    }

    #[test]
    fn test_nan_to_num() {
        use crate::array_data::ArrayData;
        let a = NdArray::from_data(ArrayData::Float64(
            ndarray::array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]
                .into_dyn()
                .into_shared(),
        ));
        let r = a.nan_to_num(0.0, 1e308, -1e308);
        let vals = f64_vals(&r);
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 1e308);
        assert_eq!(vals[2], -1e308);
        assert_eq!(vals[3], 1.0);
    }

    #[test]
    fn test_nan_to_num_integer_passthrough() {
        use crate::array_data::ArrayData;
        let a = NdArray::from_data(ArrayData::Int64(
            ndarray::array![1_i64, 2, 3].into_dyn().into_shared(),
        ));
        let r = a.nan_to_num(0.0, 1e308, -1e308);
        // Integer arrays must pass through unchanged
        assert!(matches!(r.data(), ArrayData::Int64(_)));
    }

    #[test]
    fn test_i0() {
        let a = arr(vec![0.0, 1.0]);
        let r = a.i0();
        let vals = f64_vals(&r);
        assert!((vals[0] - 1.0).abs() < 1e-10); // I0(0) = 1
        assert!((vals[1] - 1.2660658778).abs() < 1e-8); // I0(1) ≈ 1.2660658778
    }

    #[test]
    fn test_scimath_sqrt_positive() {
        let a = arr(vec![4.0, 9.0]);
        let r = a.scimath_sqrt();
        // All positive: result should be real Float64
        assert!(matches!(r.data(), ArrayData::Float64(_)));
        let vals = f64_vals(&r);
        assert!((vals[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_scimath_sqrt_negative() {
        let a = arr(vec![-4.0, 9.0]);
        let r = a.scimath_sqrt();
        // Has negative: result must be Complex128
        assert!(matches!(r.data(), ArrayData::Complex128(_)));
        // sqrt(-4) = 2i
        if let ArrayData::Complex128(arr) = r.data() {
            let v: Vec<_> = arr.iter().collect();
            assert!((v[0].im - 2.0).abs() < 1e-10, "sqrt(-4).im = {}", v[0].im);
        }
    }

    #[test]
    fn test_scimath_log_negative() {
        let a = arr(vec![-1.0]);
        let r = a.scimath_log();
        // log(-1) = iπ
        assert!(matches!(r.data(), ArrayData::Complex128(_)));
        if let ArrayData::Complex128(arr) = r.data() {
            let v = arr.iter().next().unwrap();
            assert!((v.re - 0.0).abs() < 1e-10);
            assert!((v.im - std::f64::consts::PI).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scimath_power_negative_base_promotes_to_complex() {
        let a = arr(vec![-4.0]);
        let e = arr(vec![0.5]);
        let r = a.scimath_power(&e).unwrap();
        assert!(matches!(r.data(), ArrayData::Complex128(_)));
        if let ArrayData::Complex128(arr) = r.data() {
            let v = arr.iter().next().unwrap();
            assert!(v.re.abs() < 1e-10);
            assert!((v.im - 2.0).abs() < 1e-10);
        }
    }
}
