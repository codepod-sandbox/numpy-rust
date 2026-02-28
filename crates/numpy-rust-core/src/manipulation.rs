use ndarray::{ArrayD, Axis, IxDyn, Slice};

use crate::array_data::ArrayData;
use crate::casting::cast_array_data;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Return a new array with the given shape. Total size must match.
    pub fn reshape(&self, shape: &[usize]) -> Result<NdArray> {
        let new_size: usize = shape.iter().product();
        if new_size != self.size() {
            return Err(NumpyError::ReshapeError {
                from: self.size(),
                to: shape.to_vec(),
            });
        }
        let sh = IxDyn(shape);
        let data = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Int32(a) => ArrayData::Int32(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Int64(a) => ArrayData::Int64(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Float32(a) => ArrayData::Float32(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Float64(a) => ArrayData::Float64(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Complex64(a) => ArrayData::Complex64(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Complex128(a) => ArrayData::Complex128(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
            ArrayData::Str(a) => ArrayData::Str(
                a.clone()
                    .into_shape_with_order(sh)
                    .expect("size validated above"),
            ),
        };
        Ok(NdArray::from_data(data))
    }

    /// Transpose the array (reverse axes).
    pub fn transpose(&self) -> NdArray {
        let data = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(a.t().to_owned()),
            ArrayData::Int32(a) => ArrayData::Int32(a.t().to_owned()),
            ArrayData::Int64(a) => ArrayData::Int64(a.t().to_owned()),
            ArrayData::Float32(a) => ArrayData::Float32(a.t().to_owned()),
            ArrayData::Float64(a) => ArrayData::Float64(a.t().to_owned()),
            ArrayData::Complex64(a) => ArrayData::Complex64(a.t().to_owned()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.t().to_owned()),
            ArrayData::Str(a) => ArrayData::Str(a.t().to_owned()),
        };
        NdArray::from_data(data)
    }

    /// Swap two axes of the array.
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> Result<NdArray> {
        let ndim = self.ndim();
        if axis1 >= ndim || axis2 >= ndim {
            return Err(NumpyError::ValueError(format!(
                "axis {} or {} out of bounds for array of dimension {}",
                axis1, axis2, ndim
            )));
        }
        if axis1 == axis2 {
            return Ok(self.clone());
        }
        // Build permutation: identity with axis1 and axis2 swapped
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(axis1, axis2);

        macro_rules! do_swap {
            ($arr:expr) => {{
                let view = $arr.view().permuted_axes(perm.clone());
                view.to_owned()
            }};
        }

        let data = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(do_swap!(a)),
            ArrayData::Int32(a) => ArrayData::Int32(do_swap!(a)),
            ArrayData::Int64(a) => ArrayData::Int64(do_swap!(a)),
            ArrayData::Float32(a) => ArrayData::Float32(do_swap!(a)),
            ArrayData::Float64(a) => ArrayData::Float64(do_swap!(a)),
            ArrayData::Complex64(a) => ArrayData::Complex64(do_swap!(a)),
            ArrayData::Complex128(a) => ArrayData::Complex128(do_swap!(a)),
            ArrayData::Str(a) => ArrayData::Str(do_swap!(a)),
        };
        Ok(NdArray::from_data(data))
    }

    /// Return a 1-D copy of the array.
    pub fn flatten(&self) -> NdArray {
        self.reshape(&[self.size()])
            .expect("flatten reshape cannot fail")
    }

    /// Return a contiguous flattened array (same as flatten for our purposes).
    pub fn ravel(&self) -> NdArray {
        self.flatten()
    }

    /// Insert a new axis of size 1 at the given position.
    pub fn expand_dims(&self, axis: usize) -> Result<NdArray> {
        if axis > self.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis,
                ndim: self.ndim() + 1,
            });
        }
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);
        self.reshape(&new_shape)
    }

    /// Reverse elements along the given axis. If axis is None, flip the flattened array.
    pub fn flip(&self, axis: Option<usize>) -> Result<NdArray> {
        match axis {
            None => {
                let flat = self.flatten();
                flat.flip(Some(0))
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::ValueError(format!(
                        "axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }

                macro_rules! do_flip {
                    ($arr:expr) => {
                        $arr.slice_axis(Axis(ax), Slice::new(0, None, -1))
                            .to_owned()
                    };
                }

                let data = match &self.data {
                    ArrayData::Bool(a) => ArrayData::Bool(do_flip!(a)),
                    ArrayData::Int32(a) => ArrayData::Int32(do_flip!(a)),
                    ArrayData::Int64(a) => ArrayData::Int64(do_flip!(a)),
                    ArrayData::Float32(a) => ArrayData::Float32(do_flip!(a)),
                    ArrayData::Float64(a) => ArrayData::Float64(do_flip!(a)),
                    ArrayData::Complex64(a) => ArrayData::Complex64(do_flip!(a)),
                    ArrayData::Complex128(a) => ArrayData::Complex128(do_flip!(a)),
                    ArrayData::Str(a) => ArrayData::Str(do_flip!(a)),
                };
                Ok(NdArray::from_data(data))
            }
        }
    }

    /// Rotate array 90 degrees in the plane of the first two axes.
    /// k=1: transpose then flip axis 1
    /// k=2: flip both axes
    /// k=3: transpose then flip axis 0
    pub fn rot90(&self, k: i32) -> Result<NdArray> {
        if self.ndim() < 2 {
            return Err(NumpyError::ValueError(
                "rot90 requires at least 2-D array".into(),
            ));
        }
        let k = k.rem_euclid(4);
        match k {
            0 => Ok(self.clone()),
            1 => {
                // rot90 k=1: transpose then flip axis 1
                self.transpose().flip(Some(0))
            }
            2 => {
                // flip both axes
                self.flip(Some(0))?.flip(Some(1))
            }
            3 => {
                // flip axis 0 then transpose
                self.flip(Some(0))?.transpose().flip(Some(1))
            }
            _ => unreachable!(),
        }
    }

    /// Circular shift of elements. If axis is None, flatten first then reshape back.
    pub fn roll(&self, shift: i64, axis: Option<usize>) -> Result<NdArray> {
        match axis {
            None => {
                let flat = self.flatten();
                let rolled = flat.roll(shift, Some(0))?;
                rolled.reshape(self.shape())
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::ValueError(format!(
                        "axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }
                let n = self.shape()[ax] as i64;
                if n == 0 {
                    return Ok(self.clone());
                }
                let shift = ((shift % n) + n) % n;
                if shift == 0 {
                    return Ok(self.clone());
                }
                // Build rolled indices
                let indices: Vec<usize> = (0..n as usize)
                    .map(|i| (i + (n as usize - shift as usize)) % n as usize)
                    .collect();
                self.index_select(ax, &indices)
            }
        }
    }

    /// Select elements by indices. If axis is None, flatten first.
    pub fn take(&self, indices: &[usize], axis: Option<usize>) -> Result<NdArray> {
        match axis {
            None => {
                let flat = self.flatten();
                flat.index_select(0, indices)
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::ValueError(format!(
                        "axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }
                self.index_select(ax, indices)
            }
        }
    }

    /// Remove dimensions of size 1.
    /// If `axis` is Some, only remove that specific axis (error if it's not size 1).
    /// If `axis` is None, remove all size-1 dimensions.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<NdArray> {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                    });
                }
                if self.shape()[ax] != 1 {
                    return Err(NumpyError::ValueError(format!(
                        "cannot squeeze axis {ax} with size {}",
                        self.shape()[ax]
                    )));
                }
                let new_shape: Vec<usize> = self
                    .shape()
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                if new_shape.is_empty() {
                    self.reshape(&[])
                } else {
                    self.reshape(&new_shape)
                }
            }
            None => {
                let new_shape: Vec<usize> =
                    self.shape().iter().copied().filter(|&s| s != 1).collect();
                if new_shape.is_empty() {
                    self.reshape(&[])
                } else {
                    self.reshape(&new_shape)
                }
            }
        }
    }
}

/// Concatenate arrays along an axis. All arrays must have the same shape
/// except in the concatenation dimension.
pub fn concatenate(arrays: &[&NdArray], axis: usize) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to concatenate".into(),
        ));
    }

    let ndim = arrays[0].ndim();
    if axis >= ndim {
        return Err(NumpyError::InvalidAxis { axis, ndim });
    }

    // Promote all to common dtype
    let common_dtype = arrays
        .iter()
        .skip(1)
        .fold(arrays[0].dtype(), |acc, a| acc.promote(a.dtype()));

    let promoted: Vec<_> = arrays
        .iter()
        .map(|a| cast_array_data(&a.data, common_dtype))
        .collect();

    let ax = Axis(axis);

    macro_rules! concat_variant {
        ($variant:ident) => {{
            let views: Vec<_> = promoted
                .iter()
                .map(|d| match d {
                    ArrayData::$variant(a) => a.view(),
                    _ => unreachable!(),
                })
                .collect();
            ArrayData::$variant(
                ndarray::concatenate(ax, &views)
                    .map_err(|e| NumpyError::ShapeMismatch(e.to_string()))?,
            )
        }};
    }

    let data = match common_dtype {
        crate::DType::Bool => concat_variant!(Bool),
        crate::DType::Int32 => concat_variant!(Int32),
        crate::DType::Int64 => concat_variant!(Int64),
        crate::DType::Float32 => concat_variant!(Float32),
        crate::DType::Float64 => concat_variant!(Float64),
        crate::DType::Complex64 => concat_variant!(Complex64),
        crate::DType::Complex128 => concat_variant!(Complex128),
        crate::DType::Str => concat_variant!(Str),
        _ => unreachable!("promote() returns canonical storage types"),
    };

    Ok(NdArray::from_data(data))
}

/// Stack arrays along a new axis.
pub fn stack(arrays: &[&NdArray], axis: usize) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to stack".into(),
        ));
    }

    // Each array gets a new axis inserted at `axis`, then concatenate
    let expanded: Vec<NdArray> = arrays
        .iter()
        .map(|a| {
            let mut new_shape = a.shape().to_vec();
            new_shape.insert(axis, 1);
            a.reshape(&new_shape)
                .expect("insert-axis reshape cannot fail")
        })
        .collect();

    let refs: Vec<&NdArray> = expanded.iter().collect();
    concatenate(&refs, axis)
}

/// Vertical stack -- concatenate along axis 0.
pub fn vstack(arrays: &[&NdArray]) -> Result<NdArray> {
    concatenate(arrays, 0)
}

/// Horizontal stack -- concatenate along axis 1 (or axis 0 for 1-D arrays).
pub fn hstack(arrays: &[&NdArray]) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to hstack".into(),
        ));
    }
    if arrays[0].ndim() == 1 {
        concatenate(arrays, 0)
    } else {
        concatenate(arrays, 1)
    }
}

/// Column stack -- for 1-D arrays: reshape each to (N, 1) then concatenate along axis 1.
/// For 2-D+ arrays: concatenate along axis 1.
pub fn column_stack(arrays: &[&NdArray]) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to column_stack".into(),
        ));
    }

    // Check if all inputs are 1-D
    let all_1d = arrays.iter().all(|a| a.ndim() == 1);

    if all_1d {
        // Reshape each from (N,) to (N, 1), then concatenate along axis 1
        let reshaped: Vec<NdArray> = arrays
            .iter()
            .map(|a| {
                let n = a.shape()[0];
                a.reshape(&[n, 1]).expect("reshape to column cannot fail")
            })
            .collect();
        let refs: Vec<&NdArray> = reshaped.iter().collect();
        concatenate(&refs, 1)
    } else {
        concatenate(arrays, 1)
    }
}

/// Depth stack -- expand each array to at least 3-D, then concatenate along axis 2.
/// 1-D (N,) -> (1, N, 1)
/// 2-D (M, N) -> (M, N, 1)
/// 3-D+ -> unchanged
pub fn dstack(arrays: &[&NdArray]) -> Result<NdArray> {
    if arrays.is_empty() {
        return Err(NumpyError::ValueError(
            "need at least one array to dstack".into(),
        ));
    }

    let expanded: Vec<NdArray> = arrays
        .iter()
        .map(|a| match a.ndim() {
            1 => {
                let n = a.shape()[0];
                a.reshape(&[1, n, 1])
                    .expect("reshape 1-D to 3-D cannot fail")
            }
            2 => {
                let m = a.shape()[0];
                let n = a.shape()[1];
                a.reshape(&[m, n, 1])
                    .expect("reshape 2-D to 3-D cannot fail")
            }
            _ => (*a).clone(),
        })
        .collect();

    let refs: Vec<&NdArray> = expanded.iter().collect();
    concatenate(&refs, 2)
}

/// Specification for how to split an array.
pub enum SplitSpec {
    /// Split into N equal sections.
    NSections(usize),
    /// Split at the given indices.
    Indices(Vec<usize>),
}

/// Split array along an axis.
pub fn split(a: &NdArray, spec: &SplitSpec, axis: usize) -> Result<Vec<NdArray>> {
    if axis >= a.ndim() {
        return Err(NumpyError::InvalidAxis {
            axis,
            ndim: a.ndim(),
        });
    }
    let axis_len = a.shape()[axis];

    let indices = match spec {
        SplitSpec::NSections(n) => {
            if *n == 0 {
                return Err(NumpyError::ValueError(
                    "number of sections must be > 0".into(),
                ));
            }
            if !axis_len.is_multiple_of(*n) {
                return Err(NumpyError::ValueError(format!(
                    "array split does not result in an equal division: {} into {}",
                    axis_len, n
                )));
            }
            let section_size = axis_len / n;
            (1..*n).map(|i| i * section_size).collect::<Vec<_>>()
        }
        SplitSpec::Indices(idx) => idx.clone(),
    };

    let mut boundaries = Vec::with_capacity(indices.len() + 2);
    boundaries.push(0usize);
    boundaries.extend_from_slice(&indices);
    boundaries.push(axis_len);

    let mut result = Vec::with_capacity(boundaries.len() - 1);
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        let mut args = Vec::with_capacity(a.ndim());
        for i in 0..a.ndim() {
            if i == axis {
                args.push(crate::indexing::SliceArg::Range {
                    start: Some(start as isize),
                    stop: Some(end as isize),
                    step: 1,
                });
            } else {
                args.push(crate::indexing::SliceArg::Full);
            }
        }
        result.push(a.slice(&args)?);
    }

    Ok(result)
}

/// Split along axis 0.
pub fn vsplit(a: &NdArray, spec: &SplitSpec) -> Result<Vec<NdArray>> {
    split(a, spec, 0)
}

/// Split along axis 1 (or axis 0 for 1-D arrays).
pub fn hsplit(a: &NdArray, spec: &SplitSpec) -> Result<Vec<NdArray>> {
    if a.ndim() == 1 {
        split(a, spec, 0)
    } else {
        split(a, spec, 1)
    }
}

/// Repeat elements of an array.
/// axis=None: flatten then repeat. axis=Some: repeat along that axis.
pub fn repeat(a: &NdArray, repeats: usize, axis: Option<usize>) -> Result<NdArray> {
    match axis {
        None => {
            let flat = a.flatten();
            repeat_along_axis(&flat, repeats, 0)
        }
        Some(ax) => {
            if ax >= a.ndim() {
                return Err(NumpyError::InvalidAxis {
                    axis: ax,
                    ndim: a.ndim(),
                });
            }
            repeat_along_axis(a, repeats, ax)
        }
    }
}

fn repeat_along_axis(a: &NdArray, repeats: usize, axis: usize) -> Result<NdArray> {
    let axis_len = a.shape()[axis];
    let mut parts = Vec::with_capacity(axis_len * repeats);
    for i in 0..axis_len {
        let slice = a.index_select(axis, &[i])?;
        for _ in 0..repeats {
            parts.push(slice.clone());
        }
    }
    let refs: Vec<&NdArray> = parts.iter().collect();
    concatenate(&refs, axis)
}

/// Tile an array by repeating it along each axis.
pub fn tile(a: &NdArray, reps: &[usize]) -> Result<NdArray> {
    if reps.is_empty() {
        return Ok(a.clone());
    }
    let ndim = a.ndim().max(reps.len());
    let mut arr = a.clone();

    // Prepend size-1 dims if reps has more dims
    while arr.ndim() < ndim {
        arr = arr.expand_dims(0)?;
    }

    // Pad reps with 1s if array has more dims
    let mut full_reps = vec![1usize; ndim];
    let offset = ndim - reps.len();
    for (i, &r) in reps.iter().enumerate() {
        full_reps[offset + i] = r;
    }

    // Concatenate along each axis
    for (ax, &r) in full_reps.iter().enumerate() {
        if r > 1 {
            let copies: Vec<NdArray> = (0..r).map(|_| arr.clone()).collect();
            let refs: Vec<&NdArray> = copies.iter().collect();
            arr = concatenate(&refs, ax)?;
        }
    }

    Ok(arr)
}

/// Return sorted unique values of the flattened array.
pub fn unique(a: &NdArray) -> NdArray {
    let flat = a.flatten().astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else {
        unreachable!()
    };
    let mut vals: Vec<f64> = arr.iter().copied().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup();
    NdArray::from_data(ArrayData::Float64(
        ndarray::ArrayD::from_shape_vec(IxDyn(&[vals.len()]), vals).unwrap(),
    ))
}

/// Create coordinate matrices from coordinate vectors.
/// indexing="xy" (default): first output has shape (len(y), len(x)) -- x varies along columns
/// indexing="ij": first output has shape (len(x), len(y)) -- standard matrix indexing
pub fn meshgrid(arrays: &[&NdArray], indexing: &str) -> Result<Vec<NdArray>> {
    if arrays.is_empty() {
        return Ok(Vec::new());
    }

    let ndim = arrays.len();
    let shapes: Vec<usize> = arrays.iter().map(|a| a.size()).collect();

    // For xy indexing, swap first two dimensions
    let mut output_shape = shapes.clone();
    if indexing == "xy" && ndim >= 2 {
        output_shape.swap(0, 1);
    }

    let mut result = Vec::with_capacity(ndim);

    for (i, arr) in arrays.iter().enumerate() {
        let flat = arr.astype(DType::Float64).flatten();
        let ArrayData::Float64(data) = &flat.data else {
            unreachable!()
        };
        let values: Vec<f64> = data.iter().copied().collect();

        // Determine which axis this array varies along
        let vary_axis = if indexing == "xy" && ndim >= 2 {
            if i == 0 {
                1
            } else if i == 1 {
                0
            } else {
                i
            }
        } else {
            i
        };

        // Build the output array
        let total: usize = output_shape.iter().product();
        let mut out = vec![0.0_f64; total];

        // Compute strides for the output shape
        let mut strides = vec![1_usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * output_shape[d + 1];
        }

        // Fill: for each position in the output, use the coordinate along vary_axis
        for (idx, val) in out.iter_mut().enumerate() {
            let coord = (idx / strides[vary_axis]) % output_shape[vary_axis];
            *val = values[coord];
        }

        let grid = NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&output_shape), out).expect("shape matches"),
        ));
        result.push(grid);
    }

    Ok(result)
}

/// Pad an array with constant values.
/// pad_width is a Vec of (before, after) tuples, one per axis.
pub fn pad_constant(
    arr: &NdArray,
    pad_width: &[(usize, usize)],
    constant_value: f64,
) -> Result<NdArray> {
    let ndim = arr.ndim();
    if pad_width.len() != ndim {
        return Err(NumpyError::ValueError(format!(
            "pad_width has {} entries but array has {} dimensions",
            pad_width.len(),
            ndim
        )));
    }

    let f = arr.astype(DType::Float64);
    let ArrayData::Float64(data) = &f.data else {
        unreachable!()
    };

    // Compute new shape
    let new_shape: Vec<usize> = data
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &s)| s + pad_width[i].0 + pad_width[i].1)
        .collect();

    // Create output filled with constant
    let mut out = ArrayD::<f64>::from_elem(IxDyn(&new_shape), constant_value);

    // Copy original data into the right position using flat iteration
    let old_shape = data.shape().to_vec();
    let total: usize = old_shape.iter().product();

    for flat_idx in 0..total {
        // Convert flat index to multi-index in old shape
        let mut remaining = flat_idx;
        let mut old_coords = vec![0_usize; ndim];
        for d in (0..ndim).rev() {
            old_coords[d] = remaining % old_shape[d];
            remaining /= old_shape[d];
        }

        // Offset by pad_width
        let new_coords: Vec<usize> = old_coords
            .iter()
            .enumerate()
            .map(|(d, &c)| c + pad_width[d].0)
            .collect();

        out[IxDyn(&new_coords)] = data[IxDyn(&old_coords)];
    }

    Ok(NdArray::from_data(ArrayData::Float64(out)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_reshape() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.reshape(&[2, 3]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_invalid() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(a.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn test_transpose() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = a.transpose();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose_3d() {
        let a = NdArray::zeros(&[2, 3, 4], DType::Float64);
        let b = a.transpose();
        assert_eq!(b.shape(), &[4, 3, 2]);
    }

    #[test]
    fn test_flatten() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = a.flatten();
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_ravel() {
        let a = NdArray::zeros(&[2, 3, 4], DType::Int32);
        let b = a.ravel();
        assert_eq!(b.shape(), &[24]);
    }

    #[test]
    fn test_concatenate_axis0() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 3], DType::Float64);
        let c = concatenate(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[4, 3]);
    }

    #[test]
    fn test_concatenate_axis1() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 4], DType::Float64);
        let c = concatenate(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 7]);
    }

    #[test]
    fn test_concatenate_type_promotion() {
        let a = NdArray::from_vec(vec![1_i32, 2]);
        let b = NdArray::from_vec(vec![3.0_f64, 4.0]);
        let c = concatenate(&[&a, &b], 0).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
        assert_eq!(c.shape(), &[4]);
    }

    #[test]
    fn test_stack() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_stack_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = stack(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
    }

    #[test]
    fn test_vstack() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[1, 3], DType::Float64);
        let c = vstack(&[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[3, 3]);
    }

    #[test]
    fn test_hstack_2d() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 1], DType::Float64);
        let c = hstack(&[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_hstack_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = NdArray::from_vec(vec![3.0_f64, 4.0, 5.0]);
        let c = hstack(&[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[5]);
    }

    #[test]
    fn test_expand_dims() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = a.expand_dims(0).unwrap();
        assert_eq!(b.shape(), &[1, 3]);
        let c = a.expand_dims(1).unwrap();
        assert_eq!(c.shape(), &[3, 1]);
    }

    #[test]
    fn test_expand_dims_invalid() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(a.expand_dims(3).is_err()); // ndim=1, max valid axis=1
    }

    #[test]
    fn test_squeeze() {
        let a = NdArray::zeros(&[1, 3, 1], DType::Float64);
        let b = a.squeeze(None).unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_squeeze_specific_axis() {
        let a = NdArray::zeros(&[1, 3, 1], DType::Float64);
        let b = a.squeeze(Some(0)).unwrap();
        assert_eq!(b.shape(), &[3, 1]);
    }

    #[test]
    fn test_squeeze_non_unit_axis_fails() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        assert!(a.squeeze(Some(0)).is_err());
    }

    #[test]
    fn test_split_equal() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let parts = super::split(&a, &super::SplitSpec::NSections(3), 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[2].shape(), &[2]);
    }

    #[test]
    fn test_split_indices() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let parts = super::split(&a, &super::SplitSpec::Indices(vec![2, 4]), 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[2].shape(), &[1]);
    }

    #[test]
    fn test_split_2d_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .reshape(&[2, 4])
            .unwrap();
        let parts = super::split(&a, &super::SplitSpec::NSections(2), 1).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_repeat_flat() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let r = super::repeat(&a, 2, None).unwrap();
        assert_eq!(r.shape(), &[6]);
    }

    #[test]
    fn test_repeat_axis0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let r = super::repeat(&a, 3, Some(0)).unwrap();
        assert_eq!(r.shape(), &[6, 2]);
    }

    #[test]
    fn test_tile_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let t = super::tile(&a, &[3]).unwrap();
        assert_eq!(t.shape(), &[6]);
    }

    #[test]
    fn test_tile_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let t = super::tile(&a, &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[4, 6]);
    }

    // --- flip tests ---

    #[test]
    fn test_flip_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let b = a.flip(Some(0)).unwrap();
        assert_eq!(b.shape(), &[5]);
        // Check values are reversed
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0]).unwrap(), Scalar::Float64(5.0));
        assert_eq!(b.get(&[4]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_flip_2d_axis0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let b = a.flip(Some(0)).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        // First row should become last row
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0, 0]).unwrap(), Scalar::Float64(4.0));
        assert_eq!(b.get(&[1, 0]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_flip_2d_axis1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let b = a.flip(Some(1)).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0, 0]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(b.get(&[0, 2]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_flip_none_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = a.flip(None).unwrap();
        assert_eq!(b.shape(), &[3]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0]).unwrap(), Scalar::Float64(3.0));
    }

    #[test]
    fn test_flip_invalid_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.flip(Some(5)).is_err());
    }

    // --- rot90 tests ---

    #[test]
    fn test_rot90_k1() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = a.rot90(1).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_rot90_k0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = a.rot90(0).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0, 0]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_rot90_k2() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = a.rot90(2).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
        // k=2 reverses both axes: [[4,3],[2,1]]
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0, 0]).unwrap(), Scalar::Float64(4.0));
        assert_eq!(b.get(&[1, 1]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_rot90_k4_identity() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = a.rot90(4).unwrap();
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0, 0]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_rot90_1d_fails() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.rot90(1).is_err());
    }

    // --- unique tests ---

    #[test]
    fn test_unique() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0, 1.0, 3.0, 2.0]);
        let b = super::unique(&a);
        assert_eq!(b.shape(), &[3]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(b.get(&[1]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(b.get(&[2]).unwrap(), Scalar::Float64(3.0));
    }

    #[test]
    fn test_unique_already_unique() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = super::unique(&a);
        assert_eq!(b.shape(), &[3]);
    }

    // --- roll tests ---

    #[test]
    fn test_roll_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let b = a.roll(2, Some(0)).unwrap();
        assert_eq!(b.shape(), &[5]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0]).unwrap(), Scalar::Float64(4.0));
        assert_eq!(b.get(&[1]).unwrap(), Scalar::Float64(5.0));
        assert_eq!(b.get(&[2]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_roll_negative() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let b = a.roll(-1, Some(0)).unwrap();
        assert_eq!(b.shape(), &[5]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(b.get(&[4]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_roll_none_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = a.roll(1, None).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_roll_invalid_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.roll(1, Some(5)).is_err());
    }

    // --- take tests ---

    #[test]
    fn test_take_1d() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
        let b = a.take(&[0, 2, 4], Some(0)).unwrap();
        assert_eq!(b.shape(), &[3]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0]).unwrap(), Scalar::Float64(10.0));
        assert_eq!(b.get(&[1]).unwrap(), Scalar::Float64(30.0));
        assert_eq!(b.get(&[2]).unwrap(), Scalar::Float64(50.0));
    }

    #[test]
    fn test_take_none_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = a.take(&[0, 3], None).unwrap();
        assert_eq!(b.shape(), &[2]);
        use crate::indexing::Scalar;
        assert_eq!(b.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(b.get(&[1]).unwrap(), Scalar::Float64(4.0));
    }

    #[test]
    fn test_take_2d_axis0() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[3, 2])
            .unwrap();
        let b = a.take(&[0, 2], Some(0)).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_take_invalid_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.take(&[0], Some(5)).is_err());
    }

    // --- column_stack tests ---

    #[test]
    fn test_column_stack_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let r = column_stack(&[&a, &b]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
    }

    #[test]
    fn test_column_stack_2d() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 1], DType::Float64);
        let r = column_stack(&[&a, &b]).unwrap();
        assert_eq!(r.shape(), &[2, 4]);
    }

    // --- dstack tests ---

    #[test]
    fn test_dstack_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let r = dstack(&[&a, &b]).unwrap();
        assert_eq!(r.shape(), &[1, 3, 2]);
    }

    #[test]
    fn test_dstack_2d() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 3], DType::Float64);
        let r = dstack(&[&a, &b]).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2]);
    }

    // --- meshgrid tests ---

    #[test]
    fn test_meshgrid_2d_xy() {
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![4.0_f64, 5.0]);
        let grids = meshgrid(&[&x, &y], "xy").unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[2, 3]); // x varies along columns
        assert_eq!(grids[1].shape(), &[2, 3]); // y varies along rows
    }

    #[test]
    fn test_meshgrid_2d_ij() {
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![4.0_f64, 5.0]);
        let grids = meshgrid(&[&x, &y], "ij").unwrap();
        assert_eq!(grids[0].shape(), &[3, 2]);
        assert_eq!(grids[1].shape(), &[3, 2]);
    }

    // --- pad tests ---

    #[test]
    fn test_pad_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let padded = pad_constant(&a, &[(2, 1)], 0.0).unwrap();
        assert_eq!(padded.shape(), &[6]); // 2+3+1
    }

    #[test]
    fn test_pad_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let padded = pad_constant(&a, &[(1, 1), (1, 1)], 0.0).unwrap();
        assert_eq!(padded.shape(), &[4, 4]);
    }
}
