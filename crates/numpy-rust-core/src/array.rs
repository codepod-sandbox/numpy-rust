use ndarray::{ArrayD, IxDyn};

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
        };
        Self { data }
    }

    /// Create an array filled with ones.
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let sh = IxDyn(shape);
        let data = match dtype {
            DType::Bool => ArrayData::Bool(ArrayD::from_elem(sh, true)),
            DType::Int32 => ArrayData::Int32(ArrayD::ones(sh)),
            DType::Int64 => ArrayData::Int64(ArrayD::ones(sh)),
            DType::Float32 => ArrayData::Float32(ArrayD::ones(sh)),
            DType::Float64 => ArrayData::Float64(ArrayD::ones(sh)),
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

impl NdArray {
    pub fn from_vec<V: IntoArrayData>(vec: V) -> Self {
        Self {
            data: vec.into_array_data(),
        }
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
    fn test_ones() {
        let a = NdArray::ones(&[3], DType::Int32);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), DType::Int32);
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
}
