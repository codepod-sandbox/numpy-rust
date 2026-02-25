use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::error::{NumpyError, Result};
use crate::NdArray;

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

    let cond_b = broadcast_array_data(
        &ArrayData::Bool(bool_cond),
        &out_shape,
    );
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
            ArrayData::$variant(result)
        }};
    }

    let data = match (xb, yb) {
        (ArrayData::Float64(xa), ArrayData::Float64(ya)) => do_where!(&xa, &ya, Float64),
        (ArrayData::Float32(xa), ArrayData::Float32(ya)) => do_where!(&xa, &ya, Float32),
        (ArrayData::Int64(xa), ArrayData::Int64(ya)) => do_where!(&xa, &ya, Int64),
        (ArrayData::Int32(xa), ArrayData::Int32(ya)) => do_where!(&xa, &ya, Int32),
        (ArrayData::Bool(xa), ArrayData::Bool(ya)) => do_where!(&xa, &ya, Bool),
        (ArrayData::Str(xa), ArrayData::Str(ya)) => {
            let result = ndarray::Zip::from(&cond_arr)
                .and(&xa)
                .and(&ya)
                .map_collect(|&c, xv, yv| if c { xv.clone() } else { yv.clone() });
            ArrayData::Str(result)
        }
        _ => unreachable!("promotion ensures matching types"),
    };

    Ok(NdArray::from_data(data))
}

impl NdArray {
    /// Returns a Bool array: true where elements are NaN.
    /// For integer/bool types, always returns all-false.
    pub fn isnan(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_nan())),
            ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_nan())),
            _ => ArrayData::Bool(ArrayD::from_elem(IxDyn(self.shape()), false)),
        };
        NdArray::from_data(data)
    }

    /// Returns a Bool array: true where elements are finite (not NaN or Inf).
    /// For integer/bool types, always returns all-true.
    pub fn isfinite(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_finite())),
            ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_finite())),
            _ => ArrayData::Bool(ArrayD::from_elem(IxDyn(self.shape()), true)),
        };
        NdArray::from_data(data)
    }

    /// Deep copy of the array.
    pub fn copy(&self) -> NdArray {
        self.clone()
    }
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
            ArrayData::$variant(ArrayD::from_elem(IxDyn(&[]), s))
        }};
    }

    let data = match (a, b) {
        (ArrayData::Float64(a), ArrayData::Float64(b)) => do_dot!(a, b, Float64),
        (ArrayData::Float32(a), ArrayData::Float32(b)) => do_dot!(a, b, Float32),
        (ArrayData::Int64(a), ArrayData::Int64(b)) => do_dot!(a, b, Int64),
        (ArrayData::Int32(a), ArrayData::Int32(b)) => do_dot!(a, b, Int32),
        _ => unreachable!(),
    };
    Ok(NdArray::from_data(data))
}

fn matmul_2d_2d(a: &ArrayData, b: &ArrayData) -> Result<NdArray> {
    macro_rules! do_matmul {
        ($a:expr, $b:expr, $variant:ident) => {{
            let a2 = $a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let b2 = $b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let result = a2.dot(&b2).into_dyn();
            ArrayData::$variant(result)
        }};
    }

    let data = match (a, b) {
        (ArrayData::Float64(a), ArrayData::Float64(b)) => do_matmul!(a, b, Float64),
        (ArrayData::Float32(a), ArrayData::Float32(b)) => do_matmul!(a, b, Float32),
        (ArrayData::Int64(a), ArrayData::Int64(b)) => do_matmul!(a, b, Int64),
        (ArrayData::Int32(a), ArrayData::Int32(b)) => do_matmul!(a, b, Int32),
        _ => unreachable!(),
    };
    Ok(NdArray::from_data(data))
}

fn matmul_2d_1d(a: &ArrayData, b: &ArrayData) -> Result<NdArray> {
    macro_rules! do_matvec {
        ($a:expr, $b:expr, $variant:ident) => {{
            let a2 = $a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            let b1 = $b.view().into_dimensionality::<ndarray::Ix1>().unwrap();
            let result = a2.dot(&b1).into_dyn();
            ArrayData::$variant(result)
        }};
    }

    let data = match (a, b) {
        (ArrayData::Float64(a), ArrayData::Float64(b)) => do_matvec!(a, b, Float64),
        (ArrayData::Float32(a), ArrayData::Float32(b)) => do_matvec!(a, b, Float32),
        (ArrayData::Int64(a), ArrayData::Int64(b)) => do_matvec!(a, b, Int64),
        (ArrayData::Int32(a), ArrayData::Int32(b)) => do_matvec!(a, b, Int32),
        _ => unreachable!(),
    };
    Ok(NdArray::from_data(data))
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
}
