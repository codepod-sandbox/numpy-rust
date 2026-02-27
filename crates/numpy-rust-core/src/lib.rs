pub mod array;
pub mod array_data;
pub mod broadcasting;
pub mod casting;
pub mod creation;
pub mod dtype;
pub mod error;
pub mod fft;
pub mod indexing;
pub mod linalg;
pub mod manipulation;
pub mod ops;
pub mod random;
pub mod utility;

pub use array::NdArray;
pub use array_data::ArrayData;
pub use creation::{arange, eye, full, linspace, linspace_with_step, ones_like, zeros_like};
pub use dtype::DType;
pub use error::{NumpyError, Result};
pub use manipulation::{
    concatenate, hsplit, hstack, split, stack, unique, vsplit, vstack, SplitSpec,
};
pub use ops::einsum::einsum;
pub use ops::selection::{choose, intersect1d, isin, setdiff1d, union1d};
pub use utility::{argwhere, count_nonzero, diagonal, dot, nonzero, outer, where_cond};
