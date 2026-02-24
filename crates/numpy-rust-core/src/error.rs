use thiserror::Error;

#[derive(Error, Debug)]
pub enum NumpyError {
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
}

pub type Result<T> = std::result::Result<T, NumpyError>;
