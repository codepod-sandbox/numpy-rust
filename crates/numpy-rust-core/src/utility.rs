use crate::array_data::ArrayD;
use ndarray::IxDyn;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::error::{NumpyError, Result};
use crate::{DType, NdArray};

/// Convert IEEE 754 half-precision (16-bit) bit pattern to f32.
fn half_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0x1f {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            // NaN: preserve some mantissa bits
            f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13))
        }
    } else if exp == 0 {
        if mant == 0 {
            // Signed zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: value = (-1)^sign * 2^-14 * (mant / 1024)
            let mut m = mant;
            let mut e: i32 = -14;
            // Normalize
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff; // remove leading 1
            let f32_exp = ((e + 127) as u32) & 0xff;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else {
        // Normal: exp_f32 = exp_f16 - 15 + 127 = exp_f16 + 112
        let f32_exp = exp + 112;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Convert f32 to IEEE 754 half-precision (16-bit) bit pattern.
fn f32_to_half_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        // Inf or NaN
        if mant == 0 {
            (sign << 15) | 0x7c00
        } else {
            // NaN
            (sign << 15) | 0x7c00 | ((mant >> 13) as u16).max(1)
        }
    } else if exp > 142 {
        // Overflow -> Inf
        (sign << 15) | 0x7c00
    } else if exp < 103 {
        // Underflow -> zero
        sign << 15
    } else if exp < 113 {
        // Subnormal
        let shift = 113 - exp;
        let m = (mant | 0x800000) >> (shift + 13);
        (sign << 15) | (m as u16)
    } else {
        // Normal
        let h_exp = ((exp - 112) as u16) & 0x1f;
        let h_mant = (mant >> 13) as u16;
        (sign << 15) | (h_exp << 10) | h_mant
    }
}

/// Element-wise ternary: select from `x` where `cond` is true, else from `y`.
/// Like `numpy.where(cond, x, y)`.
pub fn where_cond(cond: &NdArray, x: &NdArray, y: &NdArray) -> Result<NdArray> {
    let bool_cond = match &cond.data {
        ArrayData::Bool(a) => a.clone(),
        _ => return Err(NumpyError::TypeError("condition must be boolean".into())),
    };

    // Promote x and y to common dtype (skip promotion for string+string)
    let (xd, yd) = if x.dtype().is_string() && y.dtype().is_string() {
        (x.data.clone(), y.data.clone())
    } else if x.dtype().is_string() || y.dtype().is_string() {
        return Err(NumpyError::TypeError(
            "where: cannot mix string and numeric arrays".into(),
        ));
    } else {
        let common_dtype = x.dtype().promote(y.dtype());
        (
            cast_array_data(&x.data, common_dtype),
            cast_array_data(&y.data, common_dtype),
        )
    };

    // Broadcast all three to common shape
    let shape_xy = broadcast_shape(x.shape(), y.shape())?;
    let out_shape = broadcast_shape(cond.shape(), &shape_xy)?;

    let cond_b = broadcast_array_data(&ArrayData::Bool(bool_cond), &out_shape);
    let xb = broadcast_array_data(&xd, &out_shape);
    let yb = broadcast_array_data(&yd, &out_shape);

    let cond_arr = match cond_b {
        ArrayData::Bool(a) => a,
        _ => unreachable!(),
    };

    macro_rules! do_where {
        ($xa:expr, $ya:expr, $variant:ident) => {{
            let result = ndarray::Zip::from(&cond_arr)
                .and($xa)
                .and($ya)
                .map_collect(|&c, &xv, &yv| if c { xv } else { yv });
            ArrayData::$variant(result.into_shared())
        }};
    }

    let data = match (xb, yb) {
        (ArrayData::Float64(xa), ArrayData::Float64(ya)) => do_where!(&xa, &ya, Float64),
        (ArrayData::Float32(xa), ArrayData::Float32(ya)) => do_where!(&xa, &ya, Float32),
        (ArrayData::Int64(xa), ArrayData::Int64(ya)) => do_where!(&xa, &ya, Int64),
        (ArrayData::Int32(xa), ArrayData::Int32(ya)) => do_where!(&xa, &ya, Int32),
        (ArrayData::Bool(xa), ArrayData::Bool(ya)) => do_where!(&xa, &ya, Bool),
        (ArrayData::Complex64(xa), ArrayData::Complex64(ya)) => {
            do_where!(&xa, &ya, Complex64)
        }
        (ArrayData::Complex128(xa), ArrayData::Complex128(ya)) => {
            do_where!(&xa, &ya, Complex128)
        }
        (ArrayData::Str(xa), ArrayData::Str(ya)) => {
            let result = ndarray::Zip::from(&cond_arr)
                .and(&xa)
                .and(&ya)
                .map_collect(|&c, xv, yv| if c { xv.clone() } else { yv.clone() });
            ArrayData::Str(result.into_shared())
        }
        _ => unreachable!("promotion ensures matching types"),
    };

    Ok(NdArray::from_data(data))
}

impl NdArray {
    /// Returns a Bool array: true where elements are NaN.
    /// For integer/bool types, always returns all-false.
    /// For complex types, true if either component is NaN.
    pub fn isnan(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_nan()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_nan()).into_shared()),
            ArrayData::Complex64(a) => {
                ArrayData::Bool(a.mapv(|x| x.re.is_nan() || x.im.is_nan()).into_shared())
            }
            ArrayData::Complex128(a) => {
                ArrayData::Bool(a.mapv(|x| x.re.is_nan() || x.im.is_nan()).into_shared())
            }
            _ => ArrayData::Bool(ArrayD::from_elem(IxDyn(self.shape()), false).into_shared()),
        };
        NdArray::from_data(data)
    }

    /// Returns a Bool array: true where elements are finite (not NaN or Inf).
    /// For integer/bool types, always returns all-true.
    /// For complex types, true if both components are finite.
    pub fn isfinite(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_finite()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_finite()).into_shared()),
            ArrayData::Complex64(a) => ArrayData::Bool(
                a.mapv(|x| x.re.is_finite() && x.im.is_finite())
                    .into_shared(),
            ),
            ArrayData::Complex128(a) => ArrayData::Bool(
                a.mapv(|x| x.re.is_finite() && x.im.is_finite())
                    .into_shared(),
            ),
            _ => ArrayData::Bool(ArrayD::from_elem(IxDyn(self.shape()), true).into_shared()),
        };
        NdArray::from_data(data)
    }

    /// Returns a Bool array: true where elements are infinite.
    /// For integer/bool types, always returns all-false.
    pub fn isinf(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_infinite()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_infinite()).into_shared()),
            ArrayData::Complex64(a) => ArrayData::Bool(
                a.mapv(|x| x.re.is_infinite() || x.im.is_infinite())
                    .into_shared(),
            ),
            ArrayData::Complex128(a) => ArrayData::Bool(
                a.mapv(|x| x.re.is_infinite() || x.im.is_infinite())
                    .into_shared(),
            ),
            _ => ArrayData::Bool(ArrayD::from_elem(IxDyn(self.shape()), false).into_shared()),
        };
        NdArray::from_data(data)
    }

    /// Reinterpret the raw data as a different dtype (view).
    /// Only supports same-itemsize reinterpretation for now.
    pub fn view_as_dtype(&self, target: DType) -> Result<NdArray> {
        let src_dt = self.dtype();
        if src_dt == target {
            return Ok(self.clone());
        }
        let src_size = src_dt.itemsize();
        let tgt_size = target.itemsize();
        let shape = self.shape().to_vec();

        // Same itemsize: reinterpret bytes 1:1
        if src_size == tgt_size {
            match (&self.data, target) {
                // Bool (1 byte) <-> Int8 (1 byte)
                (ArrayData::Bool(a), DType::Int8) => {
                    let raw: Vec<i32> = a.iter().map(|&b| if b { 1 } else { 0 }).collect();
                    Ok(NdArray::from_data(ArrayData::Int32(
                        ArrayD::from_shape_vec(IxDyn(&shape), raw)
                            .unwrap()
                            .into_shared(),
                    ))
                    .astype(DType::Int8))
                }
                (ArrayData::Int32(a), DType::Bool) if src_dt == DType::Int8 => {
                    let raw: Vec<bool> = a.iter().map(|&v| v != 0).collect();
                    Ok(NdArray::from_data(ArrayData::Bool(
                        ArrayD::from_shape_vec(IxDyn(&shape), raw)
                            .unwrap()
                            .into_shared(),
                    )))
                }
                // UInt16/Int16 (stored as Int32) <-> Float16 (stored as Float32)
                (ArrayData::Int32(a), DType::Float16)
                    if src_dt == DType::UInt16 || src_dt == DType::Int16 =>
                {
                    let raw: Vec<f32> = a.iter().map(|&v| half_bits_to_f32(v as u16)).collect();
                    Ok(NdArray {
                        data: ArrayData::Float32(
                            ArrayD::from_shape_vec(IxDyn(&shape), raw)
                                .unwrap()
                                .into_shared(),
                        ),
                        declared_dtype: Some(DType::Float16),
                    })
                }
                (ArrayData::Float32(a), DType::UInt16) if src_dt == DType::Float16 => {
                    let raw: Vec<i32> = a.iter().map(|&f| f32_to_half_bits(f) as i32).collect();
                    Ok(NdArray {
                        data: ArrayData::Int32(
                            ArrayD::from_shape_vec(IxDyn(&shape), raw)
                                .unwrap()
                                .into_shared(),
                        ),
                        declared_dtype: Some(DType::UInt16),
                    })
                }
                (ArrayData::Float32(a), DType::Int16) if src_dt == DType::Float16 => {
                    let raw: Vec<i32> = a
                        .iter()
                        .map(|&f| f32_to_half_bits(f) as i16 as i32)
                        .collect();
                    Ok(NdArray {
                        data: ArrayData::Int32(
                            ArrayD::from_shape_vec(IxDyn(&shape), raw)
                                .unwrap()
                                .into_shared(),
                        ),
                        declared_dtype: Some(DType::Int16),
                    })
                }
                // Float32 (4 bytes) <-> Int32 (4 bytes)
                (ArrayData::Float32(a), DType::Int32) => {
                    let raw: Vec<i32> = a.iter().map(|&f| f.to_bits() as i32).collect();
                    Ok(NdArray::from_data(ArrayData::Int32(
                        ArrayD::from_shape_vec(IxDyn(&shape), raw)
                            .unwrap()
                            .into_shared(),
                    )))
                }
                (ArrayData::Int32(a), DType::Float32) => {
                    let raw: Vec<f32> = a.iter().map(|&i| f32::from_bits(i as u32)).collect();
                    Ok(NdArray::from_data(ArrayData::Float32(
                        ArrayD::from_shape_vec(IxDyn(&shape), raw)
                            .unwrap()
                            .into_shared(),
                    )))
                }
                // Float64 (8 bytes) <-> Int64 (8 bytes)
                (ArrayData::Float64(a), DType::Int64) => {
                    let raw: Vec<i64> = a.iter().map(|&f| f.to_bits() as i64).collect();
                    Ok(NdArray::from_data(ArrayData::Int64(
                        ArrayD::from_shape_vec(IxDyn(&shape), raw)
                            .unwrap()
                            .into_shared(),
                    )))
                }
                (ArrayData::Int64(a), DType::Float64) => {
                    let raw: Vec<f64> = a.iter().map(|&i| f64::from_bits(i as u64)).collect();
                    Ok(NdArray::from_data(ArrayData::Float64(
                        ArrayD::from_shape_vec(IxDyn(&shape), raw)
                            .unwrap()
                            .into_shared(),
                    )))
                }
                _ => {
                    // Fall back to astype for unsupported view combinations
                    Ok(self.astype(target))
                }
            }
        } else {
            // Different itemsizes: fall back to astype
            Ok(self.astype(target))
        }
    }

    /// Check if any element is infinite.
    pub fn has_inf(&self) -> bool {
        match &self.data {
            ArrayData::Float32(a) => a.iter().any(|x| x.is_infinite()),
            ArrayData::Float64(a) => a.iter().any(|x| x.is_infinite()),
            ArrayData::Complex64(a) => a.iter().any(|x| x.re.is_infinite() || x.im.is_infinite()),
            ArrayData::Complex128(a) => a.iter().any(|x| x.re.is_infinite() || x.im.is_infinite()),
            _ => false,
        }
    }

    pub fn has_nan(&self) -> bool {
        match &self.data {
            ArrayData::Float32(a) => a.iter().any(|x| x.is_nan()),
            ArrayData::Float64(a) => a.iter().any(|x| x.is_nan()),
            ArrayData::Complex64(a) => a.iter().any(|x| x.re.is_nan() || x.im.is_nan()),
            ArrayData::Complex128(a) => a.iter().any(|x| x.re.is_nan() || x.im.is_nan()),
            _ => false,
        }
    }

    /// Deep copy of the array.
    pub fn copy(&self) -> NdArray {
        NdArray {
            data: self.data.deep_copy(),
            declared_dtype: self.declared_dtype,
        }
    }

    /// Check if this array shares its underlying buffer with another.
    pub fn shares_memory_with(&self, other: &NdArray) -> bool {
        self.data.shares_memory_with(&other.data)
    }
}

/// Return the indices of non-zero elements as an (N, ndim) Int64 array.
pub fn argwhere(a: &NdArray) -> NdArray {
    let shape = a.shape().to_vec();
    let ndim = a.ndim();
    let flat = a.astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else {
        unreachable!()
    };

    let mut coords: Vec<i64> = Vec::new();
    let mut count = 0usize;

    for (linear_idx, &val) in arr.iter().enumerate() {
        if val != 0.0 {
            let mut remaining = linear_idx;
            let mut coord = vec![0i64; ndim];
            for d in (0..ndim).rev() {
                coord[d] = (remaining % shape[d]) as i64;
                remaining /= shape[d];
            }
            coords.extend_from_slice(&coord);
            count += 1;
        }
    }

    let result_shape = if ndim == 0 {
        vec![count, 0]
    } else {
        vec![count, ndim]
    };
    NdArray::from_data(ArrayData::Int64(
        ArrayD::from_shape_vec(IxDyn(&result_shape), coords)
            .expect("coords match result shape")
            .into_shared(),
    ))
}

/// Return the indices of non-zero elements as a tuple of arrays, one per dimension.
/// This matches NumPy's convention: `nonzero(a)` returns `(idx_dim0, idx_dim1, ...)`.
pub fn nonzero(a: &NdArray) -> Vec<NdArray> {
    let shape = a.shape().to_vec();
    let ndim = a.ndim().max(1); // at least 1-D
    let flat = a.astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else {
        unreachable!()
    };

    let mut indices: Vec<Vec<i64>> = vec![Vec::new(); ndim];
    for (linear_idx, &val) in arr.iter().enumerate() {
        if val != 0.0 {
            let mut remaining = linear_idx;
            for d in (0..ndim).rev() {
                let dim_size = if d < shape.len() { shape[d] } else { 1 };
                indices[d].push((remaining % dim_size) as i64);
                remaining /= dim_size;
            }
        }
    }

    indices
        .into_iter()
        .map(|idx| {
            NdArray::from_data(ArrayData::Int64(
                ArrayD::from_shape_vec(IxDyn(&[idx.len()]), idx)
                    .unwrap()
                    .into_shared(),
            ))
        })
        .collect()
}

/// Count the number of non-zero elements.
pub fn count_nonzero(a: &NdArray) -> usize {
    let flat = a.astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else {
        unreachable!()
    };
    arr.iter().filter(|&&x| x != 0.0).count()
}

/// Dot product of two arrays.
/// - 1-D x 1-D: inner product (scalar result)
/// - 2-D x 2-D: matrix multiply
/// - 2-D x 1-D: matrix-vector multiply
pub fn dot(a: &NdArray, b: &NdArray) -> Result<NdArray> {
    let common = a.dtype().promote(b.dtype());
    let ad = cast_array_data(&a.data, common);
    let bd = cast_array_data(&b.data, common);

    match (a.ndim(), b.ndim()) {
        (1, 1) => dot_1d_1d(&ad, &bd),
        (2, 2) => matmul_2d_2d(&ad, &bd),
        (2, 1) => matmul_2d_1d(&ad, &bd),
        _ => Err(NumpyError::ValueError(format!(
            "dot not supported for {}D x {}D arrays",
            a.ndim(),
            b.ndim()
        ))),
    }
}

fn dot_1d_1d(a: &ArrayData, b: &ArrayData) -> Result<NdArray> {
    macro_rules! do_dot {
        ($a:expr, $b:expr, $variant:ident) => {{
            let s = $a.iter().zip($b.iter()).map(|(&x, &y)| x * y).sum();
            ArrayData::$variant(ArrayD::from_elem(IxDyn(&[]), s).into_shared())
        }};
    }

    let data = match (a, b) {
        (ArrayData::Float64(a), ArrayData::Float64(b)) => do_dot!(a, b, Float64),
        (ArrayData::Float32(a), ArrayData::Float32(b)) => do_dot!(a, b, Float32),
        (ArrayData::Int64(a), ArrayData::Int64(b)) => do_dot!(a, b, Int64),
        (ArrayData::Int32(a), ArrayData::Int32(b)) => do_dot!(a, b, Int32),
        (ArrayData::Complex64(a), ArrayData::Complex64(b)) => do_dot!(a, b, Complex64),
        (ArrayData::Complex128(a), ArrayData::Complex128(b)) => do_dot!(a, b, Complex128),
        _ => unreachable!(),
    };
    Ok(NdArray::from_data(data))
}

fn matmul_2d_2d(a: &ArrayData, b: &ArrayData) -> Result<NdArray> {
    macro_rules! do_matmul {
        ($a:expr, $b:expr, $variant:ident) => {{
            let a2 = $a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let b2 = $b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let result = a2.dot(&b2).into_dyn().into_shared();
            ArrayData::$variant(result)
        }};
    }

    let data = match (a, b) {
        (ArrayData::Float64(a), ArrayData::Float64(b)) => do_matmul!(a, b, Float64),
        (ArrayData::Float32(a), ArrayData::Float32(b)) => do_matmul!(a, b, Float32),
        (ArrayData::Int64(a), ArrayData::Int64(b)) => do_matmul!(a, b, Int64),
        (ArrayData::Int32(a), ArrayData::Int32(b)) => do_matmul!(a, b, Int32),
        (ArrayData::Complex64(a), ArrayData::Complex64(b)) => do_matmul!(a, b, Complex64),
        (ArrayData::Complex128(a), ArrayData::Complex128(b)) => do_matmul!(a, b, Complex128),
        _ => unreachable!(),
    };
    Ok(NdArray::from_data(data))
}

fn matmul_2d_1d(a: &ArrayData, b: &ArrayData) -> Result<NdArray> {
    macro_rules! do_matvec {
        ($a:expr, $b:expr, $variant:ident) => {{
            let a2 = $a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let b1 = $b.view().into_dimensionality::<ndarray::Ix1>().unwrap();
            let result = a2.dot(&b1).into_dyn().into_shared();
            ArrayData::$variant(result)
        }};
    }

    let data = match (a, b) {
        (ArrayData::Float64(a), ArrayData::Float64(b)) => do_matvec!(a, b, Float64),
        (ArrayData::Float32(a), ArrayData::Float32(b)) => do_matvec!(a, b, Float32),
        (ArrayData::Int64(a), ArrayData::Int64(b)) => do_matvec!(a, b, Int64),
        (ArrayData::Int32(a), ArrayData::Int32(b)) => do_matvec!(a, b, Int32),
        (ArrayData::Complex64(a), ArrayData::Complex64(b)) => do_matvec!(a, b, Complex64),
        (ArrayData::Complex128(a), ArrayData::Complex128(b)) => do_matvec!(a, b, Complex128),
        _ => unreachable!(),
    };
    Ok(NdArray::from_data(data))
}

/// Extract diagonal from a 2D array.
pub fn diagonal(a: &NdArray, offset: i64) -> Result<NdArray> {
    if a.ndim() != 2 {
        return Err(NumpyError::ValueError("diagonal requires 2-d array".into()));
    }
    let rows = a.shape()[0];
    let cols = a.shape()[1];
    let (start_r, start_c) = if offset >= 0 {
        (0, offset as usize)
    } else {
        ((-offset) as usize, 0)
    };

    let n = if offset >= 0 {
        std::cmp::min(rows, cols.saturating_sub(offset as usize))
    } else {
        std::cmp::min(rows.saturating_sub((-offset) as usize), cols)
    };

    if n == 0 {
        return Ok(NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&[0]), vec![])
                .unwrap()
                .into_shared(),
        )));
    }

    // Extract elements along diagonal
    let flat = a.astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else {
        unreachable!()
    };
    let arr2 = arr.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let vals: Vec<f64> = (0..n).map(|i| arr2[[start_r + i, start_c + i]]).collect();

    Ok(NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[n]), vals)
            .unwrap()
            .into_shared(),
    )))
}

/// Compute outer product of two arrays (flattened).
pub fn outer(a: &NdArray, b: &NdArray) -> NdArray {
    let a_flat = a.flatten().astype(crate::DType::Float64);
    let b_flat = b.flatten().astype(crate::DType::Float64);
    let ArrayData::Float64(aa) = &a_flat.data else {
        unreachable!()
    };
    let ArrayData::Float64(bb) = &b_flat.data else {
        unreachable!()
    };

    let m = aa.len();
    let n = bb.len();
    let mut result = Vec::with_capacity(m * n);
    for &ai in aa.iter() {
        for &bi in bb.iter() {
            result.push(ai * bi);
        }
    }
    NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[m, n]), result)
            .unwrap()
            .into_shared(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_where_cond() {
        let cond = NdArray::from_vec(vec![true, false, true]);
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0]);
        let result = where_cond(&cond, &x, &y).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.dtype(), DType::Float64);
    }

    #[test]
    fn test_where_cond_broadcast() {
        let cond = NdArray::from_vec(vec![true, false, true]);
        let x = NdArray::full_f64(&[], 1.0); // scalar
        let y = NdArray::full_f64(&[], 0.0); // scalar
        let result = where_cond(&cond, &x, &y).unwrap();
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_where_cond_type_promotion() {
        let cond = NdArray::from_vec(vec![true, false]);
        let x = NdArray::from_vec(vec![1_i32, 2]);
        let y = NdArray::from_vec(vec![3.0_f64, 4.0]);
        let result = where_cond(&cond, &x, &y).unwrap();
        assert_eq!(result.dtype(), DType::Float64);
    }

    #[test]
    fn test_isnan() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0]);
        let b = a.isnan();
        assert_eq!(b.dtype(), DType::Bool);
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_isnan_int() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = a.isnan();
        assert_eq!(b.dtype(), DType::Bool);
        // All false for integers
        assert!(!b.any());
    }

    #[test]
    fn test_isfinite() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::INFINITY, f64::NAN]);
        let b = a.isfinite();
        assert_eq!(b.dtype(), DType::Bool);
    }

    #[test]
    fn test_isfinite_int() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = a.isfinite();
        assert!(b.all());
    }

    #[test]
    fn test_copy() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = a.copy();
        assert_eq!(b.shape(), a.shape());
        assert_eq!(b.dtype(), a.dtype());
    }

    #[test]
    fn test_dot_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = dot(&a, &b).unwrap();
        assert_eq!(c.shape(), &[]); // scalar
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_dot_2d() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[3, 4], DType::Float64);
        let c = dot(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_dot_2d_1d() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let b = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = dot(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2]);
    }

    #[test]
    fn test_dot_type_promotion() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = dot(&a, &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_argwhere_1d() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0, 0.0, 3.0, 0.0]);
        let result = argwhere(&a);
        assert_eq!(result.shape(), &[2, 1]);
    }

    #[test]
    fn test_argwhere_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 0.0, 0.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let result = argwhere(&a);
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_argwhere_all_zero() {
        let a = NdArray::from_vec(vec![0.0_f64, 0.0, 0.0]);
        let result = argwhere(&a);
        assert_eq!(result.shape(), &[0, 1]);
    }

    #[test]
    fn test_isinf() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
        let b = a.isinf();
        assert_eq!(b.dtype(), DType::Bool);
        assert_eq!(b.shape(), &[4]);
    }

    #[test]
    fn test_isinf_int() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = a.isinf();
        assert_eq!(b.dtype(), DType::Bool);
        // All false for integers
        assert!(!b.any());
    }

    #[test]
    fn test_nonzero_1d() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0, 0.0, 3.0, 0.0]);
        let result = nonzero(&a);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[2]);
        assert_eq!(result[0].dtype(), DType::Int64);
    }

    #[test]
    fn test_nonzero_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 0.0, 0.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let result = nonzero(&a);
        assert_eq!(result.len(), 2);
        // Two non-zero elements: (0,0) and (1,1)
        assert_eq!(result[0].shape(), &[2]);
        assert_eq!(result[1].shape(), &[2]);
    }

    #[test]
    fn test_count_nonzero() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0, 0.0, 3.0, 0.0]);
        let count = count_nonzero(&a);
        assert_eq!(count, 2);
    }

    // --- diagonal tests ---

    #[test]
    fn test_diagonal_main() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let d = diagonal(&a, 0).unwrap();
        assert_eq!(d.shape(), &[2]);
        use crate::indexing::Scalar;
        assert_eq!(d.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(d.get(&[1]).unwrap(), Scalar::Float64(4.0));
    }

    #[test]
    fn test_diagonal_positive_offset() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let d = diagonal(&a, 1).unwrap();
        assert_eq!(d.shape(), &[2]);
        use crate::indexing::Scalar;
        assert_eq!(d.get(&[0]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(d.get(&[1]).unwrap(), Scalar::Float64(6.0));
    }

    #[test]
    fn test_diagonal_negative_offset() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[3, 2])
            .unwrap();
        let d = diagonal(&a, -1).unwrap();
        assert_eq!(d.shape(), &[2]);
        use crate::indexing::Scalar;
        assert_eq!(d.get(&[0]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(d.get(&[1]).unwrap(), Scalar::Float64(6.0));
    }

    #[test]
    fn test_diagonal_1d_fails() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(diagonal(&a, 0).is_err());
    }

    #[test]
    fn test_diagonal_large_offset_empty() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let d = diagonal(&a, 5).unwrap();
        assert_eq!(d.shape(), &[0]);
    }

    // --- outer tests ---

    #[test]
    fn test_outer() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0]);
        let c = outer(&a, &b);
        assert_eq!(c.shape(), &[3, 2]);
        use crate::indexing::Scalar;
        assert_eq!(c.get(&[0, 0]).unwrap(), Scalar::Float64(4.0));
        assert_eq!(c.get(&[0, 1]).unwrap(), Scalar::Float64(5.0));
        assert_eq!(c.get(&[1, 0]).unwrap(), Scalar::Float64(8.0));
        assert_eq!(c.get(&[2, 1]).unwrap(), Scalar::Float64(15.0));
    }

    #[test]
    fn test_outer_scalar() {
        let a = NdArray::from_vec(vec![2.0_f64]);
        let b = NdArray::from_vec(vec![3.0_f64, 4.0]);
        let c = outer(&a, &b);
        assert_eq!(c.shape(), &[1, 2]);
        use crate::indexing::Scalar;
        assert_eq!(c.get(&[0, 0]).unwrap(), Scalar::Float64(6.0));
        assert_eq!(c.get(&[0, 1]).unwrap(), Scalar::Float64(8.0));
    }
}
