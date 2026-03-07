use ndarray::{ArcArray, IxDyn};
use num_complex::Complex;

use crate::dtype::DType;

/// Shared (reference-counted) dynamic-dimensional array.
/// `clone()` is O(1) (Arc refcount increment). Mutation triggers copy-on-write.
/// Re-exported as `ArrayD` so existing code compiles without changes.
pub type ArrayD<T> = ArcArray<T, IxDyn>;

/// Alias for clarity in contexts where sharing semantics matter.
pub type SharedArrayD<T> = ArrayD<T>;

/// Type-erased array storage. Each variant holds a shared `ArcArray<T, IxDyn>`.
/// Clone is O(1) — arrays share their underlying buffer via Arc.
/// Mutation automatically triggers copy-on-write when the buffer is shared.
#[derive(Debug, Clone)]
pub enum ArrayData {
    Bool(SharedArrayD<bool>),
    Int32(SharedArrayD<i32>),
    Int64(SharedArrayD<i64>),
    Float32(SharedArrayD<f32>),
    Float64(SharedArrayD<f64>),
    Complex64(SharedArrayD<Complex<f32>>),
    Complex128(SharedArrayD<Complex<f64>>),
    Str(SharedArrayD<String>),
}

impl ArrayData {
    pub fn dtype(&self) -> DType {
        match self {
            ArrayData::Bool(_) => DType::Bool,
            ArrayData::Int32(_) => DType::Int32,
            ArrayData::Int64(_) => DType::Int64,
            ArrayData::Float32(_) => DType::Float32,
            ArrayData::Float64(_) => DType::Float64,
            ArrayData::Complex64(_) => DType::Complex64,
            ArrayData::Complex128(_) => DType::Complex128,
            ArrayData::Str(_) => DType::Str,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayData::Bool(a) => a.shape(),
            ArrayData::Int32(a) => a.shape(),
            ArrayData::Int64(a) => a.shape(),
            ArrayData::Float32(a) => a.shape(),
            ArrayData::Float64(a) => a.shape(),
            ArrayData::Complex64(a) => a.shape(),
            ArrayData::Complex128(a) => a.shape(),
            ArrayData::Str(a) => a.shape(),
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            ArrayData::Bool(a) => a.ndim(),
            ArrayData::Int32(a) => a.ndim(),
            ArrayData::Int64(a) => a.ndim(),
            ArrayData::Float32(a) => a.ndim(),
            ArrayData::Float64(a) => a.ndim(),
            ArrayData::Complex64(a) => a.ndim(),
            ArrayData::Complex128(a) => a.ndim(),
            ArrayData::Str(a) => a.ndim(),
        }
    }

    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Deep copy: creates an independent copy of the data (not just Arc refcount++).
    pub fn deep_copy(&self) -> Self {
        match self {
            ArrayData::Bool(a) => ArrayData::Bool(a.to_owned().into_shared()),
            ArrayData::Int32(a) => ArrayData::Int32(a.to_owned().into_shared()),
            ArrayData::Int64(a) => ArrayData::Int64(a.to_owned().into_shared()),
            ArrayData::Float32(a) => ArrayData::Float32(a.to_owned().into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.to_owned().into_shared()),
            ArrayData::Complex64(a) => ArrayData::Complex64(a.to_owned().into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.to_owned().into_shared()),
            ArrayData::Str(a) => ArrayData::Str(a.to_owned().into_shared()),
        }
    }

    /// Check if two ArrayData share the same underlying buffer (Arc pointer equality).
    pub fn shares_memory_with(&self, other: &Self) -> bool {
        match (self, other) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => std::ptr::eq(a.as_ptr(), b.as_ptr()),
            (ArrayData::Int32(a), ArrayData::Int32(b)) => std::ptr::eq(a.as_ptr(), b.as_ptr()),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => std::ptr::eq(a.as_ptr(), b.as_ptr()),
            (ArrayData::Float32(a), ArrayData::Float32(b)) => std::ptr::eq(a.as_ptr(), b.as_ptr()),
            (ArrayData::Float64(a), ArrayData::Float64(b)) => std::ptr::eq(a.as_ptr(), b.as_ptr()),
            (ArrayData::Complex64(a), ArrayData::Complex64(b)) => {
                std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            (ArrayData::Complex128(a), ArrayData::Complex128(b)) => {
                std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            (ArrayData::Str(a), ArrayData::Str(b)) => std::ptr::eq(a.as_ptr(), b.as_ptr()),
            _ => false, // different dtypes can't share memory
        }
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
            $crate::ArrayData::Complex64($name) => $body,
            $crate::ArrayData::Complex128($name) => $body,
        }
    };
}
