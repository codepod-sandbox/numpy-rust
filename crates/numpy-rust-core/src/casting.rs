use ndarray::ArrayD;

use crate::array_data::ArrayData;
use crate::dtype::DType;

/// Cast an ArrayData to a different dtype.
/// If already the target dtype, returns a clone.
pub fn cast_array_data(data: &ArrayData, target: DType) -> ArrayData {
    if data.dtype() == target {
        return data.clone();
    }

    match target {
        DType::Bool => ArrayData::Bool(cast_to_bool(data)),
        DType::Int32 => ArrayData::Int32(cast_to_i32(data)),
        DType::Int64 => ArrayData::Int64(cast_to_i64(data)),
        DType::Float32 => ArrayData::Float32(cast_to_f32(data)),
        DType::Float64 => ArrayData::Float64(cast_to_f64(data)),
    }
}

fn cast_to_bool(data: &ArrayData) -> ArrayD<bool> {
    match data {
        ArrayData::Bool(a) => a.clone(),
        ArrayData::Int32(a) => a.mapv(|x| x != 0),
        ArrayData::Int64(a) => a.mapv(|x| x != 0),
        ArrayData::Float32(a) => a.mapv(|x| x != 0.0),
        ArrayData::Float64(a) => a.mapv(|x| x != 0.0),
    }
}

fn cast_to_i32(data: &ArrayData) -> ArrayD<i32> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| x as i32),
        ArrayData::Int32(a) => a.clone(),
        ArrayData::Int64(a) => a.mapv(|x| x as i32),
        ArrayData::Float32(a) => a.mapv(|x| x as i32),
        ArrayData::Float64(a) => a.mapv(|x| x as i32),
    }
}

fn cast_to_i64(data: &ArrayData) -> ArrayD<i64> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| x as i64),
        ArrayData::Int32(a) => a.mapv(|x| x as i64),
        ArrayData::Int64(a) => a.clone(),
        ArrayData::Float32(a) => a.mapv(|x| x as i64),
        ArrayData::Float64(a) => a.mapv(|x| x as i64),
    }
}

fn cast_to_f32(data: &ArrayData) -> ArrayD<f32> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| if x { 1.0 } else { 0.0 }),
        ArrayData::Int32(a) => a.mapv(|x| x as f32),
        ArrayData::Int64(a) => a.mapv(|x| x as f32),
        ArrayData::Float32(a) => a.clone(),
        ArrayData::Float64(a) => a.mapv(|x| x as f32),
    }
}

fn cast_to_f64(data: &ArrayData) -> ArrayD<f64> {
    match data {
        ArrayData::Bool(a) => a.mapv(|x| if x { 1.0 } else { 0.0 }),
        ArrayData::Int32(a) => a.mapv(|x| x as f64),
        ArrayData::Int64(a) => a.mapv(|x| x as f64),
        ArrayData::Float32(a) => a.mapv(|x| x as f64),
        ArrayData::Float64(a) => a.clone(),
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
}
