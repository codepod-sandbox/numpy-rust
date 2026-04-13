pub mod array;
pub mod array_data;
pub mod broadcasting;
pub mod casting;
pub mod creation;
pub mod descriptor;
pub mod dtype;
pub mod error;
pub mod fft;
pub mod indexing;
pub mod kernel;
pub mod linalg;
pub mod manipulation;
pub mod ops;
pub mod random;
pub mod resolver;
pub mod storage;
pub mod struct_array;
pub mod utility;
pub use struct_array::{FieldSpec, StructArrayData};

pub use array::NdArray;
pub use array_data::ArrayData;
pub use creation::{arange, eye, full, linspace, linspace_with_step, ones_like, zeros_like};
pub use descriptor::{descriptor_for_dtype, DTypeDescriptor};
pub use dtype::DType;
pub use error::{validate_shape, NumpyError, Result};
pub use indexing::{ravel_multi_index, unravel_index, LogicalScalar};
pub use manipulation::{
    column_stack, concatenate, dstack, hsplit, hstack, meshgrid, pad_constant, split, stack,
    unique, vsplit, vstack, SplitSpec,
};
pub use ops::einsum::einsum;
pub use ops::selection::{choose, intersect1d, isin, setdiff1d, union1d};
pub use resolver::{
    resolve_binary_op, resolve_dot_op, resolve_where_op, BinaryOp, BinaryOpPlan, DotOp, DotOpPlan,
    WhereOp, WhereOpPlan,
};
pub use utility::{argwhere, count_nonzero, diagonal, dot, nonzero, outer, where_cond};
