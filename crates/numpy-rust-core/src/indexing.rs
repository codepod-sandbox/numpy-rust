use ndarray::{IxDyn, SliceInfoElem};
use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Describes how to slice one axis.
#[derive(Debug, Clone)]
pub enum SliceArg {
    /// Select a single index (removes the dimension).
    Index(isize),
    /// Slice with start, stop, step (all optional, like Python's a[start:stop:step]).
    Range {
        start: Option<isize>,
        stop: Option<isize>,
        step: isize,
    },
    /// Select all elements along this axis (equivalent to `:`).
    Full,
}

/// A scalar value extracted from an array.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Complex64(Complex<f32>),
    Complex128(Complex<f64>),
    Str(String),
}

impl NdArray {
    /// Get a single element by multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Result<Scalar> {
        let idx = IxDyn(index);
        match &self.data {
            ArrayData::Bool(a) => a
                .get(idx)
                .map(|&v| Scalar::Bool(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Int32(a) => a
                .get(idx)
                .map(|&v| Scalar::Int32(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Int64(a) => a
                .get(idx)
                .map(|&v| Scalar::Int64(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Float32(a) => a
                .get(idx)
                .map(|&v| Scalar::Float32(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Float64(a) => a
                .get(idx)
                .map(|&v| Scalar::Float64(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Complex64(a) => a
                .get(idx)
                .map(|&v| Scalar::Complex64(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Complex128(a) => a
                .get(idx)
                .map(|&v| Scalar::Complex128(v))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
            ArrayData::Str(a) => a
                .get(idx)
                .map(|v| Scalar::Str(v.clone()))
                .ok_or_else(|| NumpyError::ValueError("index out of bounds".into())),
        }
    }

    /// Slice the array using SliceArg descriptors for each axis.
    /// Axes not specified are treated as Full.
    pub fn slice(&self, args: &[SliceArg]) -> Result<NdArray> {
        let ndim = self.ndim();
        let shape = self.shape();

        // Build ndarray SliceInfoElem for each axis
        let mut slice_elems: Vec<SliceInfoElem> = Vec::with_capacity(ndim);
        let mut remove_axes: Vec<usize> = Vec::new();

        for i in 0..ndim {
            let arg = if i < args.len() {
                &args[i]
            } else {
                &SliceArg::Full
            };
            match arg {
                SliceArg::Index(idx) => {
                    let resolved = resolve_index(*idx, shape[i])?;
                    slice_elems.push(SliceInfoElem::Index(resolved as isize));
                    remove_axes.push(i);
                }
                SliceArg::Range { start, stop, step } => {
                    let s = start.map(|v| v).unwrap_or(0);
                    let e = stop.map(|v| v).unwrap_or(shape[i] as isize);
                    slice_elems.push(SliceInfoElem::Slice {
                        start: s,
                        end: Some(e),
                        step: *step,
                    });
                }
                SliceArg::Full => {
                    slice_elems.push(SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    });
                }
            }
        }

        let info: Vec<SliceInfoElem> = slice_elems;

        macro_rules! do_slice {
            ($arr:expr) => {{
                let view = $arr.slice(info.as_slice());
                view.to_owned()
            }};
        }

        let data = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(do_slice!(a)),
            ArrayData::Int32(a) => ArrayData::Int32(do_slice!(a)),
            ArrayData::Int64(a) => ArrayData::Int64(do_slice!(a)),
            ArrayData::Float32(a) => ArrayData::Float32(do_slice!(a)),
            ArrayData::Float64(a) => ArrayData::Float64(do_slice!(a)),
            ArrayData::Complex64(a) => ArrayData::Complex64(do_slice!(a)),
            ArrayData::Complex128(a) => ArrayData::Complex128(do_slice!(a)),
            ArrayData::Str(a) => ArrayData::Str(do_slice!(a)),
        };

        Ok(NdArray::from_data(data))
    }

    /// Select elements along an axis by integer indices.
    pub fn index_select(&self, axis: usize, indices: &[usize]) -> Result<NdArray> {
        if axis >= self.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        macro_rules! do_select {
            ($arr:expr) => {{
                let views: Vec<_> = indices
                    .iter()
                    .map(|&i| $arr.index_axis(ndarray::Axis(axis), i))
                    .collect();
                let view_refs: Vec<_> = views.iter().map(|v| v.view()).collect();
                ndarray::stack(ndarray::Axis(axis), &view_refs).unwrap()
            }};
        }

        let data = match &self.data {
            ArrayData::Bool(a) => ArrayData::Bool(do_select!(a)),
            ArrayData::Int32(a) => ArrayData::Int32(do_select!(a)),
            ArrayData::Int64(a) => ArrayData::Int64(do_select!(a)),
            ArrayData::Float32(a) => ArrayData::Float32(do_select!(a)),
            ArrayData::Float64(a) => ArrayData::Float64(do_select!(a)),
            ArrayData::Complex64(a) => ArrayData::Complex64(do_select!(a)),
            ArrayData::Complex128(a) => ArrayData::Complex128(do_select!(a)),
            ArrayData::Str(a) => ArrayData::Str(do_select!(a)),
        };

        Ok(NdArray::from_data(data))
    }

    /// Set elements along an axis by integer indices from a values array.
    /// Like `a[[0, 2]] = values` in NumPy.
    pub fn index_set(&mut self, axis: usize, indices: &[usize], values: &NdArray) -> Result<()> {
        if axis >= self.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        macro_rules! do_set {
            ($dst:expr, $src:expr) => {{
                for (pos, &idx) in indices.iter().enumerate() {
                    let src_view = $src.index_axis(ndarray::Axis(axis), pos);
                    let mut dst_view = $dst.index_axis_mut(ndarray::Axis(axis), idx);
                    dst_view.assign(&src_view);
                }
            }};
        }

        // Cast values to match self dtype if needed
        let cast_values = if values.dtype() != self.dtype() {
            values.astype(self.dtype())
        } else {
            values.clone()
        };

        match (&mut self.data, &cast_values.data) {
            (ArrayData::Bool(dst), ArrayData::Bool(src)) => do_set!(dst, src),
            (ArrayData::Int32(dst), ArrayData::Int32(src)) => do_set!(dst, src),
            (ArrayData::Int64(dst), ArrayData::Int64(src)) => do_set!(dst, src),
            (ArrayData::Float32(dst), ArrayData::Float32(src)) => do_set!(dst, src),
            (ArrayData::Float64(dst), ArrayData::Float64(src)) => do_set!(dst, src),
            (ArrayData::Complex64(dst), ArrayData::Complex64(src)) => do_set!(dst, src),
            (ArrayData::Complex128(dst), ArrayData::Complex128(src)) => do_set!(dst, src),
            (ArrayData::Str(dst), ArrayData::Str(src)) => do_set!(dst, src),
            _ => {
                return Err(NumpyError::TypeError(
                    "cannot assign array of incompatible dtype".into(),
                ));
            }
        }
        Ok(())
    }

    /// Set a single element by multi-dimensional index.
    pub fn set(&mut self, index: &[usize], value: Scalar) -> Result<()> {
        let idx = IxDyn(index);
        match (&mut self.data, value) {
            (ArrayData::Bool(a), Scalar::Bool(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            (ArrayData::Int32(a), Scalar::Int32(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            (ArrayData::Int64(a), Scalar::Int64(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            (ArrayData::Float32(a), Scalar::Float32(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            (ArrayData::Float64(a), Scalar::Float64(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            (ArrayData::Complex64(a), Scalar::Complex64(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            (ArrayData::Complex128(a), Scalar::Complex128(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            // Allow cross-type numeric assignment (cast to target dtype)
            (ArrayData::Float64(a), Scalar::Int64(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v as f64;
            }
            (ArrayData::Float64(a), Scalar::Int32(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v as f64;
            }
            (ArrayData::Float64(a), Scalar::Float32(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v as f64;
            }
            (ArrayData::Int64(a), Scalar::Int32(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v as i64;
            }
            (ArrayData::Complex128(a), Scalar::Float64(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = Complex::new(v, 0.0);
            }
            (ArrayData::Complex128(a), Scalar::Complex64(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = Complex::new(v.re as f64, v.im as f64);
            }
            (ArrayData::Str(a), Scalar::Str(v)) => {
                let elem = a
                    .get_mut(idx)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *elem = v;
            }
            _ => {
                return Err(NumpyError::TypeError(
                    "cannot assign scalar of incompatible dtype".into(),
                ));
            }
        }
        Ok(())
    }

    /// Set a slice of the array from values in another array.
    pub fn set_slice(&mut self, args: &[SliceArg], values: &NdArray) -> Result<()> {
        let ndim = self.ndim();
        let shape = self.shape().to_vec();

        // Build ndarray SliceInfoElem for each axis
        let mut slice_elems: Vec<SliceInfoElem> = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let arg = if i < args.len() {
                &args[i]
            } else {
                &SliceArg::Full
            };
            match arg {
                SliceArg::Index(idx) => {
                    let resolved = resolve_index(*idx, shape[i])?;
                    slice_elems.push(SliceInfoElem::Index(resolved as isize));
                }
                SliceArg::Range { start, stop, step } => {
                    let s = start.unwrap_or(0);
                    let e = stop.unwrap_or(shape[i] as isize);
                    slice_elems.push(SliceInfoElem::Slice {
                        start: s,
                        end: Some(e),
                        step: *step,
                    });
                }
                SliceArg::Full => {
                    slice_elems.push(SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    });
                }
            }
        }

        let info: Vec<SliceInfoElem> = slice_elems;

        macro_rules! do_set_slice {
            ($dst:expr, $src:expr) => {{
                let mut view = $dst.slice_mut(info.as_slice());
                if view.shape() != $src.shape() {
                    return Err(NumpyError::ShapeMismatch(format!(
                        "could not broadcast input array from shape {:?} into shape {:?}",
                        $src.shape(),
                        view.shape()
                    )));
                }
                view.assign($src);
            }};
        }

        match (&mut self.data, &values.data) {
            (ArrayData::Bool(dst), ArrayData::Bool(src)) => do_set_slice!(dst, src),
            (ArrayData::Int32(dst), ArrayData::Int32(src)) => do_set_slice!(dst, src),
            (ArrayData::Int64(dst), ArrayData::Int64(src)) => do_set_slice!(dst, src),
            (ArrayData::Float32(dst), ArrayData::Float32(src)) => do_set_slice!(dst, src),
            (ArrayData::Float64(dst), ArrayData::Float64(src)) => do_set_slice!(dst, src),
            (ArrayData::Complex64(dst), ArrayData::Complex64(src)) => do_set_slice!(dst, src),
            (ArrayData::Complex128(dst), ArrayData::Complex128(src)) => do_set_slice!(dst, src),
            (ArrayData::Str(dst), ArrayData::Str(src)) => do_set_slice!(dst, src),
            // Allow float64 array to accept cast from other numeric types
            (ArrayData::Float64(dst), _) => {
                let cast = values.astype(crate::DType::Float64);
                if let ArrayData::Float64(src) = &cast.data {
                    do_set_slice!(dst, src);
                }
            }
            // Allow complex128 array to accept cast from other numeric types
            (ArrayData::Complex128(dst), _) => {
                let cast = values.astype(crate::DType::Complex128);
                if let ArrayData::Complex128(src) = &cast.data {
                    do_set_slice!(dst, src);
                }
            }
            _ => {
                return Err(NumpyError::TypeError(
                    "cannot assign array of incompatible dtype".into(),
                ));
            }
        }
        Ok(())
    }

    /// Set elements where mask is true to the corresponding values.
    pub fn mask_set(&mut self, mask: &NdArray, values: &NdArray) -> Result<()> {
        let bool_mask = match &mask.data {
            ArrayData::Bool(m) => m,
            _ => return Err(NumpyError::TypeError("mask must be a boolean array".into())),
        };

        if bool_mask.len() != self.size() {
            return Err(NumpyError::ShapeMismatch(format!(
                "mask size {} does not match array size {}",
                bool_mask.len(),
                self.size()
            )));
        }

        let flat_mask: Vec<bool> = bool_mask.iter().copied().collect();
        let true_count: usize = flat_mask.iter().filter(|&&m| m).count();

        if values.size() != true_count && values.size() != 1 {
            return Err(NumpyError::ShapeMismatch(format!(
                "values size {} does not match number of True entries {}",
                values.size(),
                true_count
            )));
        }

        macro_rules! do_mask_set {
            ($dst:expr, $src:expr) => {{
                let mut val_idx = 0;
                let src_flat: Vec<_> = $src.iter().copied().collect();
                for (i, &m) in flat_mask.iter().enumerate() {
                    if m {
                        let flat = $dst.as_slice_mut().unwrap();
                        flat[i] = if src_flat.len() == 1 {
                            src_flat[0]
                        } else {
                            src_flat[val_idx]
                        };
                        val_idx += 1;
                    }
                }
            }};
        }

        match (&mut self.data, &values.data) {
            (ArrayData::Bool(dst), ArrayData::Bool(src)) => do_mask_set!(dst, src),
            (ArrayData::Int32(dst), ArrayData::Int32(src)) => do_mask_set!(dst, src),
            (ArrayData::Int64(dst), ArrayData::Int64(src)) => do_mask_set!(dst, src),
            (ArrayData::Float32(dst), ArrayData::Float32(src)) => do_mask_set!(dst, src),
            (ArrayData::Float64(dst), ArrayData::Float64(src)) => do_mask_set!(dst, src),
            (ArrayData::Complex64(dst), ArrayData::Complex64(src)) => do_mask_set!(dst, src),
            (ArrayData::Complex128(dst), ArrayData::Complex128(src)) => do_mask_set!(dst, src),
            (ArrayData::Float64(dst), _) => {
                let cast = values.astype(crate::DType::Float64);
                if let ArrayData::Float64(src) = &cast.data {
                    do_mask_set!(dst, src);
                }
            }
            _ => {
                return Err(NumpyError::TypeError(
                    "cannot assign values of incompatible dtype".into(),
                ));
            }
        }
        Ok(())
    }

    /// Select elements where mask is true, returning a 1-D array.
    /// Like `a[mask]` in NumPy where mask is a boolean array.
    pub fn mask_select(&self, mask: &NdArray) -> Result<NdArray> {
        // Mask must be Bool dtype
        let bool_mask = match &mask.data {
            ArrayData::Bool(m) => m,
            _ => return Err(NumpyError::TypeError("mask must be a boolean array".into())),
        };

        // Flatten both to 1-D for element-wise selection
        if bool_mask.len() != self.size() {
            return Err(NumpyError::ShapeMismatch(format!(
                "mask size {} does not match array size {}",
                bool_mask.len(),
                self.size()
            )));
        }

        let flat_mask: Vec<bool> = bool_mask.iter().copied().collect();

        macro_rules! do_mask {
            ($arr:expr, $variant:ident) => {{
                let flat: Vec<_> = $arr.iter().copied().collect();
                let selected: Vec<_> = flat
                    .into_iter()
                    .zip(flat_mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(v, _)| v)
                    .collect();
                let len = selected.len();
                ArrayData::$variant(
                    ndarray::ArrayD::from_shape_vec(IxDyn(&[len]), selected).unwrap(),
                )
            }};
        }

        let data = match &self.data {
            ArrayData::Bool(a) => do_mask!(a, Bool),
            ArrayData::Int32(a) => do_mask!(a, Int32),
            ArrayData::Int64(a) => do_mask!(a, Int64),
            ArrayData::Float32(a) => do_mask!(a, Float32),
            ArrayData::Float64(a) => do_mask!(a, Float64),
            ArrayData::Complex64(a) => do_mask!(a, Complex64),
            ArrayData::Complex128(a) => do_mask!(a, Complex128),
            ArrayData::Str(a) => {
                let flat: Vec<_> = a.iter().cloned().collect();
                let selected: Vec<_> = flat
                    .into_iter()
                    .zip(flat_mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(v, _)| v)
                    .collect();
                let len = selected.len();
                ArrayData::Str(ndarray::ArrayD::from_shape_vec(IxDyn(&[len]), selected).unwrap())
            }
        };

        Ok(NdArray::from_data(data))
    }
}

/// Resolve a possibly-negative index to a positive one.
pub(crate) fn resolve_index(idx: isize, dim_size: usize) -> Result<usize> {
    let resolved = if idx < 0 {
        dim_size as isize + idx
    } else {
        idx
    };
    if resolved < 0 || resolved as usize >= dim_size {
        return Err(NumpyError::ValueError(format!(
            "index {idx} is out of bounds for axis with size {dim_size}"
        )));
    }
    Ok(resolved as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_get_1d() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0]);
        assert_eq!(a.get(&[0]).unwrap(), Scalar::Float64(10.0));
        assert_eq!(a.get(&[2]).unwrap(), Scalar::Float64(30.0));
    }

    #[test]
    fn test_get_2d() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3, 4, 5, 6])
            .reshape(&[2, 3])
            .unwrap();
        assert_eq!(a.get(&[0, 0]).unwrap(), Scalar::Int32(1));
        assert_eq!(a.get(&[1, 2]).unwrap(), Scalar::Int32(6));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.get(&[5]).is_err());
    }

    #[test]
    fn test_slice_full() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let b = a.slice(&[SliceArg::Full, SliceArg::Full]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_slice_range() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let b = a
            .slice(&[SliceArg::Range {
                start: Some(1),
                stop: Some(4),
                step: 1,
            }])
            .unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_slice_step() {
        let a = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = a
            .slice(&[SliceArg::Range {
                start: Some(0),
                stop: Some(6),
                step: 2,
            }])
            .unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_slice_index_reduces_dim() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = a.slice(&[SliceArg::Index(0), SliceArg::Full]).unwrap();
        assert_eq!(b.shape(), &[4]);
    }

    #[test]
    fn test_slice_negative_index() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = a.slice(&[SliceArg::Index(-1), SliceArg::Full]).unwrap();
        assert_eq!(b.shape(), &[4]);
    }

    #[test]
    fn test_index_select() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
        let b = a.index_select(0, &[0, 2, 4]).unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_index_select_2d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[3, 2])
            .unwrap();
        let b = a.index_select(0, &[0, 2]).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
    }

    #[test]
    fn test_index_select_invalid_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.index_select(5, &[0]).is_err());
    }

    #[test]
    fn test_mask_select() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
        let mask = NdArray::from_vec(vec![true, false, true, false, true]);
        let b = a.mask_select(&mask).unwrap();
        assert_eq!(b.shape(), &[3]);
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_mask_select_all_false() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let mask = NdArray::from_vec(vec![false, false, false]);
        let b = a.mask_select(&mask).unwrap();
        assert_eq!(b.shape(), &[0]);
    }

    #[test]
    fn test_mask_select_all_true() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let mask = NdArray::from_vec(vec![true, true, true]);
        let b = a.mask_select(&mask).unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_mask_select_size_mismatch() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let mask = NdArray::from_vec(vec![true, false]);
        assert!(a.mask_select(&mask).is_err());
    }

    #[test]
    fn test_mask_select_non_bool_mask() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let mask = NdArray::from_vec(vec![1_i32, 0]);
        assert!(a.mask_select(&mask).is_err());
    }

    #[test]
    fn test_set_1d() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        a.set(&[0], Scalar::Float64(99.0)).unwrap();
        assert_eq!(a.get(&[0]).unwrap(), Scalar::Float64(99.0));
        assert_eq!(a.get(&[1]).unwrap(), Scalar::Float64(2.0));
    }

    #[test]
    fn test_set_2d() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        a.set(&[0, 1], Scalar::Float64(99.0)).unwrap();
        assert_eq!(a.get(&[0, 1]).unwrap(), Scalar::Float64(99.0));
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.set(&[5], Scalar::Float64(99.0)).is_err());
    }

    #[test]
    fn test_set_slice_1d() {
        let mut a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let vals = NdArray::from_vec(vec![88.0_f64, 77.0]);
        a.set_slice(
            &[SliceArg::Range {
                start: Some(1),
                stop: Some(3),
                step: 1,
            }],
            &vals,
        )
        .unwrap();
        assert_eq!(a.get(&[1]).unwrap(), Scalar::Float64(88.0));
        assert_eq!(a.get(&[2]).unwrap(), Scalar::Float64(77.0));
        assert_eq!(a.get(&[0]).unwrap(), Scalar::Float64(1.0));
    }
}
