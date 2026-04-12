use crate::array_data::ArrayD;
use ndarray::IxDyn;

use crate::array_data::ArrayData;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::error::Result;
use crate::storage::ArrayStorage;
use crate::NdArray;

/// Create a 1-D array with evenly spaced values within [start, stop).
pub fn arange(start: f64, stop: f64, step: f64, dtype: Option<DType>) -> NdArray {
    let mut values = Vec::new();
    let mut v = start;
    if step > 0.0 {
        while v < stop {
            values.push(v);
            let next = v + step;
            if next == v {
                break; // step too small relative to v, avoid infinite loop
            }
            v = next;
        }
    } else if step < 0.0 {
        while v > stop {
            values.push(v);
            let next = v + step;
            if next == v {
                break; // step too small relative to v, avoid infinite loop
            }
            v = next;
        }
    }
    let len = values.len();
    let arr = NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[len]), values).expect("vec length matches shape"),
    ));
    match dtype {
        Some(dt) => arr.astype(dt),
        None => arr,
    }
}

/// Create a 1-D array with `num` evenly spaced values from start to stop (inclusive).
pub fn linspace(start: f64, stop: f64, num: usize) -> NdArray {
    linspace_with_step(start, stop, num).0
}

/// Same as linspace but also returns the step size.
pub fn linspace_with_step(start: f64, stop: f64, num: usize) -> (NdArray, f64) {
    let (values, step) = if num == 0 {
        (Vec::new(), 0.0)
    } else if num == 1 {
        (vec![start], 0.0)
    } else {
        let step = (stop - start) / (num - 1) as f64;
        let vals: Vec<f64> = (0..num).map(|i| start + step * i as f64).collect();
        (vals, step)
    };
    let len = values.len();
    let arr = NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[len]), values).expect("vec length matches shape"),
    ));
    (arr, step)
}

/// Create a 2-D array with ones on the k-th diagonal and zeros elsewhere.
///
/// - `n` -- number of rows
/// - `m` -- number of columns (defaults to `n` when `None`)
/// - `k` -- diagonal offset: 0 = main, positive = superdiagonal, negative = subdiagonal
pub fn eye(n: usize, m: Option<usize>, k: isize, dtype: DType) -> Result<NdArray> {
    let cols = m.unwrap_or(n);
    let descriptor = descriptor_for_dtype(dtype);
    Ok(NdArray::from_parts(
        ArrayStorage::eye(n, cols, k, descriptor)?,
        descriptor,
    ))
}

/// Create an array filled with a given value.
pub fn full(shape: &[usize], value: f64, dtype: DType) -> NdArray {
    let descriptor = descriptor_for_dtype(dtype);
    NdArray::from_parts(ArrayStorage::full_f64(shape, descriptor, value), descriptor)
}

/// Create a string array filled with a given string value.
pub fn full_str(shape: &[usize], value: &str) -> NdArray {
    let sh = IxDyn(shape);
    NdArray::from_data(ArrayData::Str(
        ArrayD::from_elem(sh, value.to_string()).into_shared(),
    ))
}

/// Create an array of zeros with the same shape and dtype as the given array.
pub fn zeros_like(a: &NdArray) -> NdArray {
    NdArray::zeros(a.shape(), a.dtype())
}

/// Create an array of ones with the same shape and dtype as the given array.
pub fn ones_like(a: &NdArray) -> NdArray {
    NdArray::ones(a.shape(), a.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_arange() {
        let a = arange(0.0, 5.0, 1.0, None);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_arange_fractional_step() {
        let a = arange(0.0, 1.0, 0.5, None);
        assert_eq!(a.shape(), &[2]);
    }

    #[test]
    fn test_arange_negative_step() {
        let a = arange(5.0, 0.0, -1.0, None);
        assert_eq!(a.shape(), &[5]);
    }

    #[test]
    fn test_arange_empty() {
        let a = arange(5.0, 0.0, 1.0, None);
        assert_eq!(a.shape(), &[0]);
    }

    #[test]
    fn test_arange_with_dtype() {
        let a = arange(0.0, 5.0, 1.0, Some(DType::Int32));
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.dtype(), DType::Int32);
    }

    #[test]
    fn test_linspace() {
        let a = linspace(0.0, 1.0, 5);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_linspace_single() {
        let a = linspace(3.0, 3.0, 1);
        assert_eq!(a.shape(), &[1]);
    }

    #[test]
    fn test_linspace_empty() {
        let a = linspace(0.0, 1.0, 0);
        assert_eq!(a.shape(), &[0]);
    }

    #[test]
    fn test_linspace_with_step() {
        let (a, step) = linspace_with_step(0.0, 1.0, 5);
        assert_eq!(a.shape(), &[5]);
        assert!((step - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_eye() {
        let a = eye(3, None, 0, DType::Float64).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_eye_i32() {
        let a = eye(2, None, 0, DType::Int32).unwrap();
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.dtype(), DType::Int32);
    }

    #[test]
    fn test_eye_rectangular() {
        let a = eye(3, Some(4), 0, DType::Float64).unwrap();
        assert_eq!(a.shape(), &[3, 4]);
    }

    #[test]
    fn test_eye_offset() {
        let a = eye(3, None, 1, DType::Float64).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
    }

    #[test]
    fn test_eye_negative_offset() {
        let a = eye(3, None, -1, DType::Float64).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
    }

    #[test]
    fn test_eye_complex() {
        let a = eye(3, None, 0, DType::Complex128).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(a.dtype(), DType::Complex128);
    }

    #[test]
    fn test_full() {
        let a = full(&[2, 3], 7.0, DType::Float64);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_full_i32() {
        let a = full(&[4], 5.0, DType::Int32);
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.dtype(), DType::Int32);
    }

    #[test]
    fn test_full_complex() {
        let a = full(&[3], 2.0, DType::Complex128);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), DType::Complex128);
    }

    #[test]
    fn test_zeros_like() {
        let a = NdArray::ones(&[2, 3], DType::Int32);
        let b = zeros_like(&a);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.dtype(), DType::Int32);
    }

    #[test]
    fn test_ones_like() {
        let a = NdArray::zeros(&[4, 5], DType::Float64);
        let b = ones_like(&a);
        assert_eq!(b.shape(), &[4, 5]);
        assert_eq!(b.dtype(), DType::Float64);
    }
}
