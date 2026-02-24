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
