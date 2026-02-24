use ndarray::ArrayD;

use crate::dtype::DType;

/// Type-erased array storage. Each variant holds a concrete `ArrayD<T>`.
#[derive(Debug, Clone)]
pub enum ArrayData {
    Bool(ArrayD<bool>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
}

impl ArrayData {
    pub fn dtype(&self) -> DType {
        match self {
            ArrayData::Bool(_) => DType::Bool,
            ArrayData::Int32(_) => DType::Int32,
            ArrayData::Int64(_) => DType::Int64,
            ArrayData::Float32(_) => DType::Float32,
            ArrayData::Float64(_) => DType::Float64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayData::Bool(a) => a.shape(),
            ArrayData::Int32(a) => a.shape(),
            ArrayData::Int64(a) => a.shape(),
            ArrayData::Float32(a) => a.shape(),
            ArrayData::Float64(a) => a.shape(),
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            ArrayData::Bool(a) => a.ndim(),
            ArrayData::Int32(a) => a.ndim(),
            ArrayData::Int64(a) => a.ndim(),
            ArrayData::Float32(a) => a.ndim(),
            ArrayData::Float64(a) => a.ndim(),
        }
    }

    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Dispatch over all ArrayData variants, binding the inner ArrayD to `$name`.
#[macro_export]
macro_rules! dispatch_unary {
    ($data:expr, $name:ident, $body:expr) => {
        match $data {
            $crate::ArrayData::Bool($name) => $body,
            $crate::ArrayData::Int32($name) => $body,
            $crate::ArrayData::Int64($name) => $body,
            $crate::ArrayData::Float32($name) => $body,
            $crate::ArrayData::Float64($name) => $body,
        }
    };
}
