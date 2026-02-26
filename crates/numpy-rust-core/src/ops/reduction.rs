use ndarray::{ArrayD, Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Validate an axis index against the array's dimensionality.
fn validate_axis(axis: usize, ndim: usize) -> Result<()> {
    if axis >= ndim {
        return Err(NumpyError::InvalidAxis { axis, ndim });
    }
    Ok(())
}

/// If `keepdims` is true and a specific axis was reduced, re-insert a size-1
/// dimension at `axis` so the output rank matches the input rank.
fn maybe_keepdims(result: NdArray, axis: Option<usize>, keepdims: bool) -> NdArray {
    if !keepdims {
        return result;
    }
    if let Some(ax) = axis {
        let mut new_shape = result.shape().to_vec();
        new_shape.insert(ax, 1);
        result
            .reshape(&new_shape)
            .expect("keepdims reshape cannot fail")
    } else {
        result
    }
}

impl NdArray {
    /// Sum of array elements over a given axis, or all elements if axis is None.
    pub fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "sum not supported for string arrays".into(),
            ));
        }
        let result = match axis {
            None => self.reduce_all_sum(),
            Some(ax) => self.reduce_axis_sum(ax),
        }?;
        Ok(maybe_keepdims(result, axis, keepdims))
    }

    /// Mean of array elements. Always returns Float64.
    pub fn mean(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "mean not supported for string arrays".into(),
            ));
        }
        let sum = self.astype(DType::Float64).sum(axis, false)?;
        let count = match axis {
            None => self.size(),
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                self.shape()[ax]
            }
        };
        let divisor = NdArray::full_f64(sum.shape(), count as f64);
        let result = (&sum / &divisor)?;
        Ok(maybe_keepdims(result, axis, keepdims))
    }

    /// Minimum element over a given axis, or global minimum.
    pub fn min(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let result = match axis {
            None => self.reduce_all_min(),
            Some(ax) => self.reduce_axis_fold(ax, ReduceOp::Min),
        }?;
        Ok(maybe_keepdims(result, axis, keepdims))
    }

    /// Maximum element over a given axis, or global maximum.
    pub fn max(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let result = match axis {
            None => self.reduce_all_max(),
            Some(ax) => self.reduce_axis_fold(ax, ReduceOp::Max),
        }?;
        Ok(maybe_keepdims(result, axis, keepdims))
    }

    /// Standard deviation. Always returns Float64.
    pub fn std(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "std not supported for string arrays".into(),
            ));
        }
        let var = self.var(axis, ddof, false)?;
        let result = var.sqrt();
        Ok(maybe_keepdims(result, axis, keepdims))
    }

    /// Variance. Always returns Float64.
    pub fn var(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "var not supported for string arrays".into(),
            ));
        }
        // var = mean(x^2) - mean(x)^2
        let float_self = self.astype(DType::Float64);
        let x_sq = (&float_self * &float_self)?;
        let mean_x_sq = x_sq.mean(axis, false)?;
        let mean_x = float_self.mean(axis, false)?;
        let mean_x_squared = (&mean_x * &mean_x)?;
        let result = (&mean_x_sq - &mean_x_squared)?;
        if ddof > 0 {
            let n = match axis {
                None => self.size(),
                Some(ax) => {
                    validate_axis(ax, self.ndim())?;
                    self.shape()[ax]
                }
            };
            if ddof >= n {
                let nan_val = NdArray::full_f64(result.shape(), f64::NAN);
                return Ok(maybe_keepdims(nan_val, axis, keepdims));
            }
            let correction = NdArray::full_f64(result.shape(), n as f64 / (n - ddof) as f64);
            let corrected = (&result * &correction)?;
            return Ok(maybe_keepdims(corrected, axis, keepdims));
        }
        Ok(maybe_keepdims(result, axis, keepdims))
    }

    /// Index of minimum element (flattened).
    pub fn argmin(&self) -> Result<usize> {
        self.reduce_all_argmin()
    }

    /// Index of maximum element (flattened).
    pub fn argmax(&self) -> Result<usize> {
        self.reduce_all_argmax()
    }

    /// True if all elements are truthy.
    pub fn all(&self) -> bool {
        match &self.data {
            ArrayData::Bool(a) => a.iter().all(|&x| x),
            ArrayData::Int32(a) => a.iter().all(|&x| x != 0),
            ArrayData::Int64(a) => a.iter().all(|&x| x != 0),
            ArrayData::Float32(a) => a.iter().all(|&x| x != 0.0),
            ArrayData::Float64(a) => a.iter().all(|&x| x != 0.0),
            ArrayData::Str(a) => a.iter().all(|x| !x.is_empty()),
        }
    }

    /// True if any element is truthy.
    pub fn any(&self) -> bool {
        match &self.data {
            ArrayData::Bool(a) => a.iter().any(|&x| x),
            ArrayData::Int32(a) => a.iter().any(|&x| x != 0),
            ArrayData::Int64(a) => a.iter().any(|&x| x != 0),
            ArrayData::Float32(a) => a.iter().any(|&x| x != 0.0),
            ArrayData::Float64(a) => a.iter().any(|&x| x != 0.0),
            ArrayData::Str(a) => a.iter().any(|x| !x.is_empty()),
        }
    }

    // --- Internal helpers ---

    fn reduce_all_sum(&self) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "sum not supported for string arrays".into(),
            ));
        }
        let data = match &self.data {
            ArrayData::Bool(a) => {
                let s: i32 = a.iter().map(|&x| x as i32).sum();
                ArrayData::Int32(ArrayD::from_elem(IxDyn(&[]), s))
            }
            ArrayData::Int32(a) => {
                let s: i32 = a.iter().sum();
                ArrayData::Int32(ArrayD::from_elem(IxDyn(&[]), s))
            }
            ArrayData::Int64(a) => {
                let s: i64 = a.iter().sum();
                ArrayData::Int64(ArrayD::from_elem(IxDyn(&[]), s))
            }
            ArrayData::Float32(a) => {
                let s: f32 = a.iter().sum();
                ArrayData::Float32(ArrayD::from_elem(IxDyn(&[]), s))
            }
            ArrayData::Float64(a) => {
                let s: f64 = a.iter().sum();
                ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), s))
            }
            ArrayData::Str(_) => unreachable!(),
        };
        Ok(NdArray::from_data(data))
    }

    fn reduce_axis_sum(&self, axis: usize) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "sum not supported for string arrays".into(),
            ));
        }
        validate_axis(axis, self.ndim())?;
        let ax = Axis(axis);
        let data = match &self.data {
            ArrayData::Bool(a) => {
                let int_arr = a.mapv(|x| x as i32);
                ArrayData::Int32(int_arr.sum_axis(ax))
            }
            ArrayData::Int32(a) => ArrayData::Int32(a.sum_axis(ax)),
            ArrayData::Int64(a) => ArrayData::Int64(a.sum_axis(ax)),
            ArrayData::Float32(a) => ArrayData::Float32(a.sum_axis(ax)),
            ArrayData::Float64(a) => ArrayData::Float64(a.sum_axis(ax)),
            ArrayData::Str(_) => unreachable!(),
        };
        Ok(NdArray::from_data(data))
    }

    fn reduce_all_min(&self) -> Result<NdArray> {
        let data = match &self.data {
            ArrayData::Bool(a) => {
                let v = *a
                    .iter()
                    .min()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Bool(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Int32(a) => {
                let v = *a
                    .iter()
                    .min()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Int32(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Int64(a) => {
                let v = *a
                    .iter()
                    .min()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Int64(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Float32(a) => {
                let v = a
                    .iter()
                    .copied()
                    .reduce(f32::min)
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Float32(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Float64(a) => {
                let v = a
                    .iter()
                    .copied()
                    .reduce(f64::min)
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Str(a) => {
                let v = a
                    .iter()
                    .min()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?
                    .clone();
                ArrayData::Str(ArrayD::from_elem(IxDyn(&[]), v))
            }
        };
        Ok(NdArray::from_data(data))
    }

    fn reduce_all_max(&self) -> Result<NdArray> {
        let data = match &self.data {
            ArrayData::Bool(a) => {
                let v = *a
                    .iter()
                    .max()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Bool(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Int32(a) => {
                let v = *a
                    .iter()
                    .max()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Int32(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Int64(a) => {
                let v = *a
                    .iter()
                    .max()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Int64(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Float32(a) => {
                let v = a
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Float32(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Float64(a) => {
                let v = a
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?;
                ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), v))
            }
            ArrayData::Str(a) => {
                let v = a
                    .iter()
                    .max()
                    .ok_or_else(|| NumpyError::ValueError("empty array".into()))?
                    .clone();
                ArrayData::Str(ArrayD::from_elem(IxDyn(&[]), v))
            }
        };
        Ok(NdArray::from_data(data))
    }

    fn reduce_all_argmin(&self) -> Result<usize> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "argmin not supported for string arrays".into(),
            ));
        }
        match &self.data {
            ArrayData::Float64(a) => Ok(a
                .iter()
                .enumerate()
                .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?),
            _ => {
                let f = self.astype(DType::Float64);
                f.reduce_all_argmin()
            }
        }
    }

    fn reduce_all_argmax(&self) -> Result<usize> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "argmax not supported for string arrays".into(),
            ));
        }
        match &self.data {
            ArrayData::Float64(a) => Ok(a
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .ok_or_else(|| NumpyError::ValueError("empty array".into()))?),
            _ => {
                let f = self.astype(DType::Float64);
                f.reduce_all_argmax()
            }
        }
    }

    fn reduce_axis_fold(&self, axis: usize, op: ReduceOp) -> Result<NdArray> {
        validate_axis(axis, self.ndim())?;
        let ax = Axis(axis);
        macro_rules! fold_axis {
            ($arr:expr, $init:expr, $fold:expr) => {
                $arr.fold_axis(ax, $init, $fold)
            };
        }
        let data = match (&self.data, op) {
            (ArrayData::Int32(a), ReduceOp::Min) => {
                ArrayData::Int32(fold_axis!(a, i32::MAX, |&acc, &x| acc.min(x)))
            }
            (ArrayData::Int32(a), ReduceOp::Max) => {
                ArrayData::Int32(fold_axis!(a, i32::MIN, |&acc, &x| acc.max(x)))
            }
            (ArrayData::Int64(a), ReduceOp::Min) => {
                ArrayData::Int64(fold_axis!(a, i64::MAX, |&acc, &x| acc.min(x)))
            }
            (ArrayData::Int64(a), ReduceOp::Max) => {
                ArrayData::Int64(fold_axis!(a, i64::MIN, |&acc, &x| acc.max(x)))
            }
            (ArrayData::Float32(a), ReduceOp::Min) => {
                ArrayData::Float32(fold_axis!(a, f32::INFINITY, |&acc, &x| acc.min(x)))
            }
            (ArrayData::Float32(a), ReduceOp::Max) => {
                ArrayData::Float32(fold_axis!(a, f32::NEG_INFINITY, |&acc, &x| acc.max(x)))
            }
            (ArrayData::Float64(a), ReduceOp::Min) => {
                ArrayData::Float64(fold_axis!(a, f64::INFINITY, |&acc, &x| acc.min(x)))
            }
            (ArrayData::Float64(a), ReduceOp::Max) => {
                ArrayData::Float64(fold_axis!(a, f64::NEG_INFINITY, |&acc, &x| acc.max(x)))
            }
            (ArrayData::Bool(a), ReduceOp::Min) => {
                ArrayData::Bool(fold_axis!(a, true, |&acc, &x| acc && x))
            }
            (ArrayData::Bool(a), ReduceOp::Max) => {
                ArrayData::Bool(fold_axis!(a, false, |&acc, &x| acc || x))
            }
            (ArrayData::Str(a), ReduceOp::Min) => {
                // fold_axis with String requires Clone-based fold
                let result = a.fold_axis(ax, String::from("\u{10FFFF}"), |acc, x| {
                    if x < acc {
                        x.clone()
                    } else {
                        acc.clone()
                    }
                });
                ArrayData::Str(result)
            }
            (ArrayData::Str(a), ReduceOp::Max) => {
                let result = a.fold_axis(ax, String::new(), |acc, x| {
                    if x > acc {
                        x.clone()
                    } else {
                        acc.clone()
                    }
                });
                ArrayData::Str(result)
            }
        };
        Ok(NdArray::from_data(data))
    }
}

#[derive(Clone, Copy)]
enum ReduceOp {
    Min,
    Max,
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_sum_all() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let s = a.sum(None, false).unwrap();
        assert_eq!(s.size(), 1);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.dtype(), DType::Float64);
    }

    #[test]
    fn test_sum_axis() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let s = a.sum(Some(0), false).unwrap();
        assert_eq!(s.shape(), &[4]);
    }

    #[test]
    fn test_sum_axis_1() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let s = a.sum(Some(1), false).unwrap();
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_sum_invalid_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.sum(Some(5), false).is_err());
    }

    #[test]
    fn test_mean() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let m = a.mean(None, false).unwrap();
        assert_eq!(m.size(), 1);
        assert_eq!(m.dtype(), DType::Float64);
    }

    #[test]
    fn test_mean_axis() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let m = a.mean(Some(0), false).unwrap();
        assert_eq!(m.shape(), &[4]);
    }

    #[test]
    fn test_min_max_all() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0]);
        let mn = a.min(None, false).unwrap();
        let mx = a.max(None, false).unwrap();
        assert_eq!(mn.size(), 1);
        assert_eq!(mx.size(), 1);
    }

    #[test]
    fn test_min_axis() {
        let a = NdArray::ones(&[3, 4], DType::Int32);
        let mn = a.min(Some(0), false).unwrap();
        assert_eq!(mn.shape(), &[4]);
    }

    #[test]
    fn test_var_std() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let v = a.var(None, 0, false).unwrap();
        let s = a.std(None, 0, false).unwrap();
        assert_eq!(v.dtype(), DType::Float64);
        assert_eq!(s.dtype(), DType::Float64);
    }

    #[test]
    fn test_argmin_argmax() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0]);
        assert_eq!(a.argmin().unwrap(), 1);
        assert_eq!(a.argmax().unwrap(), 0);
    }

    #[test]
    fn test_all_any() {
        let all_true = NdArray::from_vec(vec![true, true, true]);
        let some_false = NdArray::from_vec(vec![true, false, true]);
        let all_false = NdArray::from_vec(vec![false, false, false]);

        assert!(all_true.all());
        assert!(!some_false.all());
        assert!(some_false.any());
        assert!(!all_false.any());
    }

    #[test]
    fn test_all_any_numeric() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        assert!(a.all());
        assert!(a.any());

        let b = NdArray::from_vec(vec![0_i32, 0, 0]);
        assert!(!b.all());
        assert!(!b.any());
    }

    #[test]
    fn test_sum_bool() {
        let a = NdArray::from_vec(vec![true, false, true]);
        let s = a.sum(None, false).unwrap();
        assert_eq!(s.dtype(), DType::Int32);
    }

    #[test]
    fn test_sum_keepdims() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let s = a.sum(Some(0), true).unwrap();
        assert_eq!(s.shape(), &[1, 4]);
    }

    #[test]
    fn test_mean_keepdims() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let m = a.mean(Some(1), true).unwrap();
        assert_eq!(m.shape(), &[3, 1]);
    }

    #[test]
    fn test_var_ddof() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let v0 = a.var(None, 0, false).unwrap();
        let v1 = a.var(None, 1, false).unwrap();
        assert_eq!(v0.dtype(), DType::Float64);
        assert_eq!(v1.dtype(), DType::Float64);
    }
}
