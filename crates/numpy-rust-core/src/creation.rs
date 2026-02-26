use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::NdArray;

/// Create a 1-D array with evenly spaced values within [start, stop).
pub fn arange(start: f64, stop: f64, step: f64) -> NdArray {
    let mut values = Vec::new();
    let mut v = start;
    if step > 0.0 {
        while v < stop {
            values.push(v);
            v += step;
        }
    } else if step < 0.0 {
        while v > stop {
            values.push(v);
            v += step;
        }
    }
    let len = values.len();
    NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[len]), values).expect("vec length matches shape"),
    ))
}

/// Create a 1-D array with `num` evenly spaced values from start to stop (inclusive).
pub fn linspace(start: f64, stop: f64, num: usize) -> NdArray {
    let values: Vec<f64> = if num == 0 {
        Vec::new()
    } else if num == 1 {
        vec![start]
    } else {
        let step = (stop - start) / (num - 1) as f64;
        (0..num).map(|i| start + step * i as f64).collect()
    };
    let len = values.len();
    NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[len]), values).expect("vec length matches shape"),
    ))
}

/// Create a 2-D array with ones on the k-th diagonal and zeros elsewhere.
///
/// - `n` — number of rows
/// - `m` — number of columns (defaults to `n` when `None`)
/// - `k` — diagonal offset: 0 = main, positive = superdiagonal, negative = subdiagonal
pub fn eye(n: usize, m: Option<usize>, k: isize, dtype: DType) -> NdArray {
    if dtype.is_string() {
        panic!("eye() not supported for string dtype");
    }
    let cols = m.unwrap_or(n);
    let mut arr = NdArray::zeros(&[n, cols], dtype);
    match &mut arr.data {
        ArrayData::Bool(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = true;
                }
            }
        }
        ArrayData::Int32(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1;
                }
            }
        }
        ArrayData::Int64(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1;
                }
            }
        }
        ArrayData::Float32(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1.0;
                }
            }
        }
        ArrayData::Float64(a) => {
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1.0;
                }
            }
        }
        ArrayData::Str(_) => unreachable!(),
    }
    arr
}

/// Create an array filled with a given value.
pub fn full(shape: &[usize], value: f64, dtype: DType) -> NdArray {
    let sh = IxDyn(shape);
    let data = match dtype {
        DType::Bool => ArrayData::Bool(ArrayD::from_elem(sh, value != 0.0)),
        DType::Int32 => ArrayData::Int32(ArrayD::from_elem(sh, value as i32)),
        DType::Int64 => ArrayData::Int64(ArrayD::from_elem(sh, value as i64)),
        DType::Float32 => ArrayData::Float32(ArrayD::from_elem(sh, value as f32)),
        DType::Float64 => ArrayData::Float64(ArrayD::from_elem(sh, value)),
        DType::Str => ArrayData::Str(ArrayD::from_elem(sh, value.to_string())),
    };
    NdArray::from_data(data)
}

/// Create a string array filled with a given string value.
pub fn full_str(shape: &[usize], value: &str) -> NdArray {
    let sh = IxDyn(shape);
    NdArray::from_data(ArrayData::Str(ArrayD::from_elem(sh, value.to_string())))
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
        let a = arange(0.0, 5.0, 1.0);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_arange_fractional_step() {
        let a = arange(0.0, 1.0, 0.5);
        assert_eq!(a.shape(), &[2]);
    }

    #[test]
    fn test_arange_negative_step() {
        let a = arange(5.0, 0.0, -1.0);
        assert_eq!(a.shape(), &[5]);
    }

    #[test]
    fn test_arange_empty() {
        let a = arange(5.0, 0.0, 1.0);
        assert_eq!(a.shape(), &[0]);
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
    fn test_eye() {
        let a = eye(3, None, 0, DType::Float64);
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_eye_i32() {
        let a = eye(2, None, 0, DType::Int32);
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.dtype(), DType::Int32);
    }

    #[test]
    fn test_eye_rectangular() {
        let a = eye(3, Some(4), 0, DType::Float64);
        assert_eq!(a.shape(), &[3, 4]);
    }

    #[test]
    fn test_eye_offset() {
        let a = eye(3, None, 1, DType::Float64);
        assert_eq!(a.shape(), &[3, 3]);
    }

    #[test]
    fn test_eye_negative_offset() {
        let a = eye(3, None, -1, DType::Float64);
        assert_eq!(a.shape(), &[3, 3]);
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
