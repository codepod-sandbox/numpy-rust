use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::dtype::DType;

/// The main N-dimensional array type, analogous to `numpy.ndarray`.
#[derive(Debug, Clone)]
pub struct NdArray {
    pub(crate) data: ArrayData,
}

// --- Constructors ---

impl NdArray {
    /// Create an NdArray from existing ArrayData.
    pub fn from_data(data: ArrayData) -> Self {
        Self { data }
    }

    /// Create an array filled with zeros.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let sh = IxDyn(shape);
        let data = match dtype {
            DType::Bool => ArrayData::Bool(ArrayD::from_elem(sh, false)),
            DType::Int32 => ArrayData::Int32(ArrayD::zeros(sh)),
            DType::Int64 => ArrayData::Int64(ArrayD::zeros(sh)),
            DType::Float32 => ArrayData::Float32(ArrayD::zeros(sh)),
            DType::Float64 => ArrayData::Float64(ArrayD::zeros(sh)),
            DType::Complex64 => {
                ArrayData::Complex64(ArrayD::from_elem(sh, Complex::new(0.0f32, 0.0)))
            }
            DType::Complex128 => {
                ArrayData::Complex128(ArrayD::from_elem(sh, Complex::new(0.0f64, 0.0)))
            }
            DType::Str => ArrayData::Str(ArrayD::from_elem(sh, String::new())),
        };
        Self { data }
    }

    /// Create an array filled with ones.
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        if dtype == DType::Str {
            panic!("ones() not supported for string dtype");
        }
        let sh = IxDyn(shape);
        let data = match dtype {
            DType::Bool => ArrayData::Bool(ArrayD::from_elem(sh, true)),
            DType::Int32 => ArrayData::Int32(ArrayD::ones(sh)),
            DType::Int64 => ArrayData::Int64(ArrayD::ones(sh)),
            DType::Float32 => ArrayData::Float32(ArrayD::ones(sh)),
            DType::Float64 => ArrayData::Float64(ArrayD::ones(sh)),
            DType::Complex64 => {
                ArrayData::Complex64(ArrayD::from_elem(sh, Complex::new(1.0f32, 0.0)))
            }
            DType::Complex128 => {
                ArrayData::Complex128(ArrayD::from_elem(sh, Complex::new(1.0f64, 0.0)))
            }
            DType::Str => unreachable!(),
        };
        Self { data }
    }

    /// Create an array filled with a given f64 value.
    pub fn full_f64(shape: &[usize], value: f64) -> Self {
        let sh = IxDyn(shape);
        Self {
            data: ArrayData::Float64(ArrayD::from_elem(sh, value)),
        }
    }

    /// Create a 0-dimensional (scalar) array from an f64 value.
    pub fn from_scalar(value: f64) -> Self {
        Self {
            data: ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), value)),
        }
    }
}

// --- Attributes ---

impl NdArray {
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn size(&self) -> usize {
        self.data.size()
    }

    pub fn data(&self) -> &ArrayData {
        &self.data
    }

    /// Cast this array to a different dtype, following NumPy's astype semantics.
    pub fn astype(&self, dtype: DType) -> Self {
        Self {
            data: crate::casting::cast_array_data(&self.data, dtype),
        }
    }
}

// --- Trait for converting Vec<T> to ArrayData ---

pub trait IntoArrayData {
    fn into_array_data(self) -> ArrayData;
}

impl IntoArrayData for Vec<f64> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Float64(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<f32> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Float32(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<i32> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Int32(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<i64> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Int64(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<bool> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Bool(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<String> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Str(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<Complex<f32>> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Complex64(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<Complex<f64>> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Complex128(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl NdArray {
    pub fn from_vec<V: IntoArrayData>(vec: V) -> Self {
        Self {
            data: vec.into_array_data(),
        }
    }

    /// Convenience constructor for a 1-D Complex128 array.
    pub fn from_complex128_vec(vec: Vec<Complex<f64>>) -> Self {
        Self::from_vec(vec)
    }

    /// Convenience constructor for a 1-D Complex64 array.
    pub fn from_complex64_vec(vec: Vec<Complex<f32>>) -> Self {
        Self::from_vec(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f64_vec() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.dtype(), DType::Float64);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_from_i32_vec() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        assert_eq!(a.dtype(), DType::Int32);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_from_bool_vec() {
        let a = NdArray::from_vec(vec![true, false, true]);
        assert_eq!(a.dtype(), DType::Bool);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_zeros() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.size(), 6);
    }

    #[test]
    fn test_zeros_complex() {
        let a = NdArray::zeros(&[2, 3], DType::Complex128);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.dtype(), DType::Complex128);
    }

    #[test]
    fn test_ones() {
        let a = NdArray::ones(&[3], DType::Int32);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), DType::Int32);
    }

    #[test]
    fn test_ones_complex() {
        let a = NdArray::ones(&[3], DType::Complex128);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), DType::Complex128);
    }

    #[test]
    fn test_full_f64() {
        let a = NdArray::full_f64(&[2, 2], 3.14);
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.dtype(), DType::Float64);
        assert_eq!(a.size(), 4);
    }

    #[test]
    fn test_from_data() {
        let data = ArrayData::Float32(ArrayD::zeros(IxDyn(&[5])));
        let a = NdArray::from_data(data);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.dtype(), DType::Float32);
    }

    #[test]
    fn test_from_complex_vec() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(a.dtype(), DType::Complex128);
        assert_eq!(a.size(), 2);
    }
}
