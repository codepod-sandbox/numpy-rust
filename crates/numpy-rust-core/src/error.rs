use thiserror::Error;

#[derive(Error, Debug)]
pub enum NumpyError {
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("invalid axis {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("cannot broadcast shapes {0:?} and {1:?}")]
    BroadcastError(Vec<usize>, Vec<usize>),

    #[error("cannot reshape array of size {from} into shape {to:?}")]
    ReshapeError { from: usize, to: Vec<usize> },

    #[error("type error: {0}")]
    TypeError(String),

    #[error("value error: {0}")]
    ValueError(String),
}

pub type Result<T> = std::result::Result<T, NumpyError>;

/// Maximum total number of elements we allow in an array.
/// This prevents panics from ndarray when shapes overflow.
const MAX_ARRAY_SIZE: usize = if usize::BITS >= 64 { 1 << 48 } else { 1 << 30 };

/// Validate that a shape's total element count doesn't overflow or exceed limits.
/// Returns the total size on success.
pub fn validate_shape(shape: &[usize]) -> Result<usize> {
    let mut total: usize = 1;
    for &dim in shape {
        total = total.checked_mul(dim).ok_or_else(|| {
            NumpyError::ValueError(format!(
                "array is too big; shape {:?} would overflow",
                shape
            ))
        })?;
    }
    if total > MAX_ARRAY_SIZE {
        return Err(NumpyError::ValueError(format!(
            "array is too big; shape {:?} requires {} elements",
            shape, total
        )));
    }
    Ok(total)
}
