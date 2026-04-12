use ndarray::{ArcArray, Dimension, IxDyn, SliceInfoElem};
use num_complex::Complex;

use crate::dtype::DType;
use crate::error::{NumpyError, Result};

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

    /// Return the strides of the underlying array (in units of elements, not bytes).
    pub fn strides(&self) -> Vec<isize> {
        match self {
            ArrayData::Bool(a) => a.strides().to_vec(),
            ArrayData::Int32(a) => a.strides().to_vec(),
            ArrayData::Int64(a) => a.strides().to_vec(),
            ArrayData::Float32(a) => a.strides().to_vec(),
            ArrayData::Float64(a) => a.strides().to_vec(),
            ArrayData::Complex64(a) => a.strides().to_vec(),
            ArrayData::Complex128(a) => a.strides().to_vec(),
            ArrayData::Str(a) => a.strides().to_vec(),
        }
    }

    /// Check if the array is C-contiguous (row-major, standard layout).
    pub fn is_c_contiguous(&self) -> bool {
        match self {
            ArrayData::Bool(a) => a.is_standard_layout(),
            ArrayData::Int32(a) => a.is_standard_layout(),
            ArrayData::Int64(a) => a.is_standard_layout(),
            ArrayData::Float32(a) => a.is_standard_layout(),
            ArrayData::Float64(a) => a.is_standard_layout(),
            ArrayData::Complex64(a) => a.is_standard_layout(),
            ArrayData::Complex128(a) => a.is_standard_layout(),
            ArrayData::Str(a) => a.is_standard_layout(),
        }
    }

    /// Check if the array is Fortran-contiguous (column-major).
    pub fn is_f_contiguous(&self) -> bool {
        fn check_f_contiguous(shape: &[usize], strides: &[isize]) -> bool {
            let ndim = shape.len();
            if ndim <= 1 {
                return true;
            }
            let mut expected_stride: isize = 1;
            for d in 0..ndim {
                if shape[d] > 1 && strides[d] != expected_stride {
                    return false;
                }
                expected_stride *= shape[d] as isize;
            }
            true
        }
        match self {
            ArrayData::Bool(a) => check_f_contiguous(a.shape(), a.strides()),
            ArrayData::Int32(a) => check_f_contiguous(a.shape(), a.strides()),
            ArrayData::Int64(a) => check_f_contiguous(a.shape(), a.strides()),
            ArrayData::Float32(a) => check_f_contiguous(a.shape(), a.strides()),
            ArrayData::Float64(a) => check_f_contiguous(a.shape(), a.strides()),
            ArrayData::Complex64(a) => check_f_contiguous(a.shape(), a.strides()),
            ArrayData::Complex128(a) => check_f_contiguous(a.shape(), a.strides()),
            ArrayData::Str(a) => check_f_contiguous(a.shape(), a.strides()),
        }
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

    /// Check if two ArrayData share the same underlying buffer.
    /// Uses memory range overlap detection to handle views/slices correctly.
    pub fn shares_memory_with(&self, other: &Self) -> bool {
        /// Check if two ArcArrays' memory ranges overlap.
        fn ranges_overlap<T>(
            a: &ArcArray<T, ndarray::IxDyn>,
            b: &ArcArray<T, ndarray::IxDyn>,
        ) -> bool {
            if a.is_empty() || b.is_empty() {
                return false;
            }
            let a_ptr = a.as_ptr() as usize;
            let b_ptr = b.as_ptr() as usize;
            // Compute the extent: max offset from as_ptr() across all elements
            let a_end = a_ptr + memory_extent(a);
            let b_end = b_ptr + memory_extent(b);
            a_ptr < b_end && b_ptr < a_end
        }

        /// Compute the byte extent of an array's data in memory.
        fn memory_extent<T>(a: &ArcArray<T, ndarray::IxDyn>) -> usize {
            let elem_size = std::mem::size_of::<T>();
            if elem_size == 0 || a.is_empty() {
                return 0;
            }
            // The extent is the max offset of any element from the base pointer
            let mut max_offset: isize = 0;
            for (ax, &stride) in a.strides().iter().enumerate() {
                let dim = a.shape()[ax];
                if dim > 1 {
                    let end_offset = stride * (dim as isize - 1);
                    if end_offset > 0 {
                        max_offset += end_offset;
                    }
                }
            }
            (max_offset as usize + 1) * elem_size
        }

        match (self, other) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => ranges_overlap(a, b),
            (ArrayData::Int32(a), ArrayData::Int32(b)) => ranges_overlap(a, b),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => ranges_overlap(a, b),
            (ArrayData::Float32(a), ArrayData::Float32(b)) => ranges_overlap(a, b),
            (ArrayData::Float64(a), ArrayData::Float64(b)) => ranges_overlap(a, b),
            (ArrayData::Complex64(a), ArrayData::Complex64(b)) => ranges_overlap(a, b),
            (ArrayData::Complex128(a), ArrayData::Complex128(b)) => ranges_overlap(a, b),
            (ArrayData::Str(a), ArrayData::Str(b)) => ranges_overlap(a, b),
            _ => false, // different dtypes can't share memory
        }
    }

    pub fn reshape_clone(&self, shape: IxDyn) -> Result<Self> {
        let target_shape = shape.slice().to_vec();
        macro_rules! reshape_or_copy {
            ($a:expr, $shape:expr) => {{
                let arr = $a.clone();
                match arr.into_shape_with_order($shape.clone()) {
                    Ok(reshaped) => reshaped.into_shared(),
                    Err(_) => {
                        let contiguous = $a.as_standard_layout().into_owned();
                        let original_len = contiguous.len();
                        contiguous
                            .into_shape_with_order($shape)
                            .map_err(|_| NumpyError::ReshapeError {
                                from: original_len,
                                to: target_shape.clone(),
                            })?
                            .into_shared()
                    }
                }
            }};
        }

        Ok(match self {
            ArrayData::Bool(a) => ArrayData::Bool(reshape_or_copy!(a, shape)),
            ArrayData::Int32(a) => ArrayData::Int32(reshape_or_copy!(a, shape)),
            ArrayData::Int64(a) => ArrayData::Int64(reshape_or_copy!(a, shape)),
            ArrayData::Float32(a) => ArrayData::Float32(reshape_or_copy!(a, shape)),
            ArrayData::Float64(a) => ArrayData::Float64(reshape_or_copy!(a, shape)),
            ArrayData::Complex64(a) => ArrayData::Complex64(reshape_or_copy!(a, shape)),
            ArrayData::Complex128(a) => ArrayData::Complex128(reshape_or_copy!(a, shape)),
            ArrayData::Str(a) => ArrayData::Str(reshape_or_copy!(a, shape)),
        })
    }

    pub fn reversed_axes_view(&self) -> Self {
        match self {
            ArrayData::Bool(a) => ArrayData::Bool(a.clone().reversed_axes()),
            ArrayData::Int32(a) => ArrayData::Int32(a.clone().reversed_axes()),
            ArrayData::Int64(a) => ArrayData::Int64(a.clone().reversed_axes()),
            ArrayData::Float32(a) => ArrayData::Float32(a.clone().reversed_axes()),
            ArrayData::Float64(a) => ArrayData::Float64(a.clone().reversed_axes()),
            ArrayData::Complex64(a) => ArrayData::Complex64(a.clone().reversed_axes()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.clone().reversed_axes()),
            ArrayData::Str(a) => ArrayData::Str(a.clone().reversed_axes()),
        }
    }

    pub fn permuted_axes_view(&self, axes: Vec<usize>) -> Self {
        macro_rules! permute {
            ($a:expr, $axes:expr) => {
                $a.clone().permuted_axes($axes)
            };
        }

        match self {
            ArrayData::Bool(a) => ArrayData::Bool(permute!(a, axes)),
            ArrayData::Int32(a) => ArrayData::Int32(permute!(a, axes)),
            ArrayData::Int64(a) => ArrayData::Int64(permute!(a, axes)),
            ArrayData::Float32(a) => ArrayData::Float32(permute!(a, axes)),
            ArrayData::Float64(a) => ArrayData::Float64(permute!(a, axes)),
            ArrayData::Complex64(a) => ArrayData::Complex64(permute!(a, axes)),
            ArrayData::Complex128(a) => ArrayData::Complex128(permute!(a, axes)),
            ArrayData::Str(a) => ArrayData::Str(permute!(a, axes)),
        }
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> Self {
        let target = IxDyn(target_shape);
        match self {
            ArrayData::Bool(a) => {
                ArrayData::Bool(a.broadcast(target).unwrap().to_owned().into_shared())
            }
            ArrayData::Int32(a) => {
                ArrayData::Int32(a.broadcast(target).unwrap().to_owned().into_shared())
            }
            ArrayData::Int64(a) => {
                ArrayData::Int64(a.broadcast(target).unwrap().to_owned().into_shared())
            }
            ArrayData::Float32(a) => {
                ArrayData::Float32(a.broadcast(target).unwrap().to_owned().into_shared())
            }
            ArrayData::Float64(a) => {
                ArrayData::Float64(a.broadcast(target).unwrap().to_owned().into_shared())
            }
            ArrayData::Complex64(a) => {
                ArrayData::Complex64(a.broadcast(target).unwrap().to_owned().into_shared())
            }
            ArrayData::Complex128(a) => {
                ArrayData::Complex128(a.broadcast(target).unwrap().to_owned().into_shared())
            }
            ArrayData::Str(a) => {
                ArrayData::Str(a.broadcast(target).unwrap().to_owned().into_shared())
            }
        }
    }

    pub fn slice_view(&self, info: &[SliceInfoElem]) -> Result<Self> {
        macro_rules! do_slice {
            ($arr:expr, $variant:ident) => {{
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    $arr.clone().slice_move(info)
                }));
                match result {
                    Ok(v) => ArrayData::$variant(v),
                    Err(panic_info) => {
                        let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else {
                            "internal error during slicing".to_string()
                        };
                        return Err(NumpyError::ValueError(msg));
                    }
                }
            }};
        }

        Ok(match self {
            ArrayData::Bool(a) => do_slice!(a, Bool),
            ArrayData::Int32(a) => do_slice!(a, Int32),
            ArrayData::Int64(a) => do_slice!(a, Int64),
            ArrayData::Float32(a) => do_slice!(a, Float32),
            ArrayData::Float64(a) => do_slice!(a, Float64),
            ArrayData::Complex64(a) => do_slice!(a, Complex64),
            ArrayData::Complex128(a) => do_slice!(a, Complex128),
            ArrayData::Str(a) => do_slice!(a, Str),
        })
    }

    pub fn index_select(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        macro_rules! do_select {
            ($arr:expr, $variant:ident) => {{
                let views: Vec<_> = indices
                    .iter()
                    .map(|&i| $arr.index_axis(ndarray::Axis(axis), i))
                    .collect();
                let view_refs: Vec<_> = views.iter().map(|v| v.view()).collect();
                ArrayData::$variant(
                    ndarray::stack(ndarray::Axis(axis), &view_refs)
                        .map_err(|e| NumpyError::ShapeMismatch(e.to_string()))?
                        .into_shared(),
                )
            }};
        }

        Ok(match self {
            ArrayData::Bool(a) => do_select!(a, Bool),
            ArrayData::Int32(a) => do_select!(a, Int32),
            ArrayData::Int64(a) => do_select!(a, Int64),
            ArrayData::Float32(a) => do_select!(a, Float32),
            ArrayData::Float64(a) => do_select!(a, Float64),
            ArrayData::Complex64(a) => do_select!(a, Complex64),
            ArrayData::Complex128(a) => do_select!(a, Complex128),
            ArrayData::Str(a) => do_select!(a, Str),
        })
    }

    pub fn assign_indexed(
        &mut self,
        axis: usize,
        indices: &[usize],
        values: &ArrayData,
    ) -> Result<()> {
        macro_rules! do_set {
            ($dst:expr, $src:expr) => {{
                if $src.len() == 1 {
                    let val = $src.iter().next().unwrap().clone();
                    for &idx in indices {
                        let mut dst_view = $dst.index_axis_mut(ndarray::Axis(axis), idx);
                        dst_view.fill(val.clone());
                    }
                } else {
                    for (pos, &idx) in indices.iter().enumerate() {
                        let src_view = $src.index_axis(ndarray::Axis(axis), pos);
                        let mut dst_view = $dst.index_axis_mut(ndarray::Axis(axis), idx);
                        dst_view.assign(&src_view);
                    }
                }
                Ok(())
            }};
        }

        match (self, values) {
            (ArrayData::Bool(dst), ArrayData::Bool(src)) => do_set!(dst, src),
            (ArrayData::Int32(dst), ArrayData::Int32(src)) => do_set!(dst, src),
            (ArrayData::Int64(dst), ArrayData::Int64(src)) => do_set!(dst, src),
            (ArrayData::Float32(dst), ArrayData::Float32(src)) => do_set!(dst, src),
            (ArrayData::Float64(dst), ArrayData::Float64(src)) => do_set!(dst, src),
            (ArrayData::Complex64(dst), ArrayData::Complex64(src)) => do_set!(dst, src),
            (ArrayData::Complex128(dst), ArrayData::Complex128(src)) => do_set!(dst, src),
            (ArrayData::Str(dst), ArrayData::Str(src)) => do_set!(dst, src),
            _ => Err(NumpyError::TypeError(
                "cannot assign array of incompatible dtype".into(),
            )),
        }
    }

    pub fn assign_slice(&mut self, info: &[SliceInfoElem], values: &ArrayData) -> Result<()> {
        macro_rules! do_set_slice {
            ($dst:expr, $src:expr) => {{
                let mut view = $dst.slice_mut(info);
                if view.shape() == $src.shape() {
                    view.assign($src);
                } else if $src.len() == 1 {
                    let val = $src.iter().next().unwrap().clone();
                    view.fill(val);
                } else {
                    return Err(NumpyError::ShapeMismatch(format!(
                        "could not broadcast input array from shape {:?} into shape {:?}",
                        $src.shape(),
                        view.shape()
                    )));
                }
                Ok(())
            }};
        }

        match (self, values) {
            (ArrayData::Bool(dst), ArrayData::Bool(src)) => do_set_slice!(dst, src),
            (ArrayData::Int32(dst), ArrayData::Int32(src)) => do_set_slice!(dst, src),
            (ArrayData::Int64(dst), ArrayData::Int64(src)) => do_set_slice!(dst, src),
            (ArrayData::Float32(dst), ArrayData::Float32(src)) => do_set_slice!(dst, src),
            (ArrayData::Float64(dst), ArrayData::Float64(src)) => do_set_slice!(dst, src),
            (ArrayData::Complex64(dst), ArrayData::Complex64(src)) => do_set_slice!(dst, src),
            (ArrayData::Complex128(dst), ArrayData::Complex128(src)) => do_set_slice!(dst, src),
            (ArrayData::Str(dst), ArrayData::Str(src)) => do_set_slice!(dst, src),
            _ => Err(NumpyError::TypeError(
                "cannot assign array of incompatible dtype".into(),
            )),
        }
    }

    pub fn assign_masked(&mut self, flat_mask: &[bool], values: &ArrayData) -> Result<()> {
        macro_rules! do_mask_set {
            ($dst:expr, $src:expr) => {{
                let mut val_idx = 0;
                let src_flat: Vec<_> = $src.iter().cloned().collect();
                if let Some(flat) = $dst.as_slice_mut() {
                    for (i, &m) in flat_mask.iter().enumerate() {
                        if m {
                            flat[i] = if src_flat.len() == 1 {
                                src_flat[0].clone()
                            } else {
                                src_flat[val_idx].clone()
                            };
                            val_idx += 1;
                        }
                    }
                } else {
                    let shape = $dst.shape().to_vec();
                    for (i, &m) in flat_mask.iter().enumerate() {
                        if m {
                            let mut rem = i;
                            let mut coord = vec![0usize; shape.len()];
                            for d in (0..shape.len()).rev() {
                                coord[d] = rem % shape[d];
                                rem /= shape[d];
                            }
                            let val = if src_flat.len() == 1 {
                                src_flat[0].clone()
                            } else {
                                src_flat[val_idx].clone()
                            };
                            $dst[ndarray::IxDyn(&coord)] = val;
                            val_idx += 1;
                        }
                    }
                }
                Ok(())
            }};
        }

        match (self, values) {
            (ArrayData::Bool(dst), ArrayData::Bool(src)) => do_mask_set!(dst, src),
            (ArrayData::Int32(dst), ArrayData::Int32(src)) => do_mask_set!(dst, src),
            (ArrayData::Int64(dst), ArrayData::Int64(src)) => do_mask_set!(dst, src),
            (ArrayData::Float32(dst), ArrayData::Float32(src)) => do_mask_set!(dst, src),
            (ArrayData::Float64(dst), ArrayData::Float64(src)) => do_mask_set!(dst, src),
            (ArrayData::Complex64(dst), ArrayData::Complex64(src)) => do_mask_set!(dst, src),
            (ArrayData::Complex128(dst), ArrayData::Complex128(src)) => do_mask_set!(dst, src),
            (ArrayData::Str(dst), ArrayData::Str(src)) => do_mask_set!(dst, src),
            _ => Err(NumpyError::TypeError(
                "cannot assign values of incompatible dtype".into(),
            )),
        }
    }

    pub fn select_masked(&self, flat_mask: &[bool]) -> Self {
        macro_rules! do_mask {
            ($arr:expr, $variant:ident) => {{
                let flat: Vec<_> = $arr.iter().cloned().collect();
                let selected: Vec<_> = flat
                    .into_iter()
                    .zip(flat_mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(v, _)| v)
                    .collect();
                let len = selected.len();
                ArrayData::$variant(
                    ndarray::ArrayD::from_shape_vec(IxDyn(&[len]), selected)
                        .unwrap()
                        .into_shared(),
                )
            }};
        }

        match self {
            ArrayData::Bool(a) => do_mask!(a, Bool),
            ArrayData::Int32(a) => do_mask!(a, Int32),
            ArrayData::Int64(a) => do_mask!(a, Int64),
            ArrayData::Float32(a) => do_mask!(a, Float32),
            ArrayData::Float64(a) => do_mask!(a, Float64),
            ArrayData::Complex64(a) => do_mask!(a, Complex64),
            ArrayData::Complex128(a) => do_mask!(a, Complex128),
            ArrayData::Str(a) => do_mask!(a, Str),
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
