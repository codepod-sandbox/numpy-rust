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
fn maybe_keepdims(
    result: NdArray,
    axis: Option<usize>,
    keepdims: bool,
    original_ndim: usize,
) -> NdArray {
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
        // axis=None reduced everything to scalar -- wrap in shape (1, 1, ..., 1)
        let new_shape = vec![1; original_ndim];
        result
            .reshape(&new_shape)
            .expect("keepdims reshape cannot fail")
    }
}

/// Helper: extract the Float64 inner array from an NdArray that has been cast.
fn as_f64(arr: &NdArray) -> &ArrayD<f64> {
    match &arr.data {
        ArrayData::Float64(a) => a,
        _ => unreachable!("expected Float64 after astype"),
    }
}

impl NdArray {
    /// Sum of array elements, ignoring NaN values.
    /// NaN values are treated as zero.
    pub fn nansum(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let s: f64 = arr.iter().filter(|x| !x.is_nan()).sum();
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), s)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let result_arr =
                    arr.map_axis(Axis(ax), |lane| lane.iter().filter(|x| !x.is_nan()).sum());
                NdArray::from_data(ArrayData::Float64(result_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Mean of array elements, ignoring NaN values.
    /// If all values along the reduction are NaN, the result is NaN.
    pub fn nanmean(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let (sum, count) = arr.iter().fold((0.0_f64, 0_usize), |(s, c), &x| {
                    if x.is_nan() {
                        (s, c)
                    } else {
                        (s + x, c + 1)
                    }
                });
                let mean = if count == 0 {
                    f64::NAN
                } else {
                    sum / count as f64
                };
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), mean)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let result_arr = arr.map_axis(Axis(ax), |lane| {
                    let (sum, count) = lane.iter().fold((0.0_f64, 0_usize), |(s, c), &x| {
                        if x.is_nan() {
                            (s, c)
                        } else {
                            (s + x, c + 1)
                        }
                    });
                    if count == 0 {
                        f64::NAN
                    } else {
                        sum / count as f64
                    }
                });
                NdArray::from_data(ArrayData::Float64(result_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Variance of array elements, ignoring NaN values.
    /// Uses the formula: sum((x - mean)^2) / (n - ddof) on NaN-filtered values.
    /// If all values are NaN, or n <= ddof, the result is NaN.
    pub fn nanvar(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let val = nanvar_slice(arr.iter().copied(), ddof);
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), val)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let result_arr =
                    arr.map_axis(Axis(ax), |lane| nanvar_slice(lane.iter().copied(), ddof));
                NdArray::from_data(ArrayData::Float64(result_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Standard deviation of array elements, ignoring NaN values.
    /// This is the square root of nanvar.
    pub fn nanstd(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let val = nanvar_slice(arr.iter().copied(), ddof).sqrt();
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), val)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let result_arr = arr.map_axis(Axis(ax), |lane| {
                    nanvar_slice(lane.iter().copied(), ddof).sqrt()
                });
                NdArray::from_data(ArrayData::Float64(result_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Minimum of array elements, ignoring NaN values.
    /// Returns an error if all values along the reduction are NaN.
    pub fn nanmin(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let min = arr
                    .iter()
                    .copied()
                    .filter(|x| !x.is_nan())
                    .fold(None, |acc: Option<f64>, x| {
                        Some(match acc {
                            Some(m) => m.min(x),
                            None => x,
                        })
                    })
                    .ok_or_else(|| {
                        NumpyError::ValueError("All-NaN slice encountered in nanmin".into())
                    })?;
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), min)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                // Check for all-NaN lanes before building the result.
                let mut results: Vec<f64> = Vec::new();
                for lane in arr.lanes(Axis(ax)) {
                    let min = lane
                        .iter()
                        .copied()
                        .filter(|x| !x.is_nan())
                        .fold(None, |acc: Option<f64>, x| {
                            Some(match acc {
                                Some(m) => m.min(x),
                                None => x,
                            })
                        })
                        .ok_or_else(|| {
                            NumpyError::ValueError("All-NaN slice encountered in nanmin".into())
                        })?;
                    results.push(min);
                }
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() {
                    IxDyn(&[])
                } else {
                    IxDyn(&result_shape)
                };
                let result_arr =
                    ArrayD::from_shape_vec(result_dim, results).expect("shape matches lanes count");
                NdArray::from_data(ArrayData::Float64(result_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Maximum of array elements, ignoring NaN values.
    /// Returns an error if all values along the reduction are NaN.
    pub fn nanmax(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let max = arr
                    .iter()
                    .copied()
                    .filter(|x| !x.is_nan())
                    .fold(None, |acc: Option<f64>, x| {
                        Some(match acc {
                            Some(m) => m.max(x),
                            None => x,
                        })
                    })
                    .ok_or_else(|| {
                        NumpyError::ValueError("All-NaN slice encountered in nanmax".into())
                    })?;
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), max)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut results: Vec<f64> = Vec::new();
                for lane in arr.lanes(Axis(ax)) {
                    let max = lane
                        .iter()
                        .copied()
                        .filter(|x| !x.is_nan())
                        .fold(None, |acc: Option<f64>, x| {
                            Some(match acc {
                                Some(m) => m.max(x),
                                None => x,
                            })
                        })
                        .ok_or_else(|| {
                            NumpyError::ValueError("All-NaN slice encountered in nanmax".into())
                        })?;
                    results.push(max);
                }
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() {
                    IxDyn(&[])
                } else {
                    IxDyn(&result_shape)
                };
                let result_arr =
                    ArrayD::from_shape_vec(result_dim, results).expect("shape matches lanes count");
                NdArray::from_data(ArrayData::Float64(result_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }

    /// Index of the minimum value, ignoring NaN values.
    /// Returns an error if all values along the reduction are NaN.
    pub fn nanargmin(&self, axis: Option<usize>) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);

        match axis {
            None => {
                let mut min_idx: Option<usize> = None;
                let mut min_val = f64::INFINITY;
                for (i, &v) in arr.iter().enumerate() {
                    if !v.is_nan() && v < min_val {
                        min_val = v;
                        min_idx = Some(i);
                    }
                }
                let idx = min_idx.ok_or_else(|| {
                    NumpyError::ValueError("All-NaN slice encountered in nanargmin".into())
                })?;
                Ok(NdArray::from_data(ArrayData::Int64(ArrayD::from_elem(
                    IxDyn(&[]),
                    idx as i64,
                ))))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut results: Vec<i64> = Vec::new();
                for lane in arr.lanes(Axis(ax)) {
                    let mut min_idx: Option<usize> = None;
                    let mut min_val = f64::INFINITY;
                    for (i, &v) in lane.iter().enumerate() {
                        if !v.is_nan() && v < min_val {
                            min_val = v;
                            min_idx = Some(i);
                        }
                    }
                    let idx = min_idx.ok_or_else(|| {
                        NumpyError::ValueError("All-NaN slice encountered in nanargmin".into())
                    })?;
                    results.push(idx as i64);
                }
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() {
                    IxDyn(&[])
                } else {
                    IxDyn(&result_shape)
                };
                let result_arr =
                    ArrayD::from_shape_vec(result_dim, results).expect("shape matches lanes count");
                Ok(NdArray::from_data(ArrayData::Int64(result_arr)))
            }
        }
    }

    /// Index of the maximum value, ignoring NaN values.
    /// Returns an error if all values along the reduction are NaN.
    pub fn nanargmax(&self, axis: Option<usize>) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);

        match axis {
            None => {
                let mut max_idx: Option<usize> = None;
                let mut max_val = f64::NEG_INFINITY;
                for (i, &v) in arr.iter().enumerate() {
                    if !v.is_nan() && v > max_val {
                        max_val = v;
                        max_idx = Some(i);
                    }
                }
                let idx = max_idx.ok_or_else(|| {
                    NumpyError::ValueError("All-NaN slice encountered in nanargmax".into())
                })?;
                Ok(NdArray::from_data(ArrayData::Int64(ArrayD::from_elem(
                    IxDyn(&[]),
                    idx as i64,
                ))))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let mut results: Vec<i64> = Vec::new();
                for lane in arr.lanes(Axis(ax)) {
                    let mut max_idx: Option<usize> = None;
                    let mut max_val = f64::NEG_INFINITY;
                    for (i, &v) in lane.iter().enumerate() {
                        if !v.is_nan() && v > max_val {
                            max_val = v;
                            max_idx = Some(i);
                        }
                    }
                    let idx = max_idx.ok_or_else(|| {
                        NumpyError::ValueError("All-NaN slice encountered in nanargmax".into())
                    })?;
                    results.push(idx as i64);
                }
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() {
                    IxDyn(&[])
                } else {
                    IxDyn(&result_shape)
                };
                let result_arr =
                    ArrayD::from_shape_vec(result_dim, results).expect("shape matches lanes count");
                Ok(NdArray::from_data(ArrayData::Int64(result_arr)))
            }
        }
    }

    /// Product of array elements, ignoring NaN values.
    /// NaN values are treated as 1.
    pub fn nanprod(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let f = self.astype(DType::Float64);
        let arr = as_f64(&f);
        let original_ndim = self.ndim();

        let result = match axis {
            None => {
                let p: f64 = arr
                    .iter()
                    .map(|&x| if x.is_nan() { 1.0 } else { x })
                    .product();
                NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), p)))
            }
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                let result_arr = arr.map_axis(Axis(ax), |lane| {
                    lane.iter()
                        .map(|&x| if x.is_nan() { 1.0 } else { x })
                        .product()
                });
                NdArray::from_data(ArrayData::Float64(result_arr))
            }
        };
        Ok(maybe_keepdims(result, axis, keepdims, original_ndim))
    }
}

/// Compute variance of an iterator of f64 values, ignoring NaN.
/// Returns NaN if count <= ddof or count == 0.
fn nanvar_slice(values: impl Iterator<Item = f64>, ddof: usize) -> f64 {
    let filtered: Vec<f64> = values.filter(|x| !x.is_nan()).collect();
    let n = filtered.len();
    if n == 0 || n <= ddof {
        return f64::NAN;
    }
    let mean = filtered.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = filtered.iter().map(|&x| (x - mean) * (x - mean)).sum();
    sum_sq / (n - ddof) as f64
}

#[cfg(test)]
mod tests {
    use crate::{ArrayData, DType, NdArray};
    use ndarray::{ArrayD, IxDyn};

    /// Helper to create a 1-D Float64 NdArray from a slice.
    fn arr1(vals: &[f64]) -> NdArray {
        NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&[vals.len()]), vals.to_vec()).unwrap(),
        ))
    }

    /// Helper to create a 2-D Float64 NdArray from shape and flat data.
    fn arr2(rows: usize, cols: usize, vals: &[f64]) -> NdArray {
        NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&[rows, cols]), vals.to_vec()).unwrap(),
        ))
    }

    /// Extract the scalar f64 value from a 0-d Float64 NdArray.
    fn scalar_f64(a: &NdArray) -> f64 {
        match a.data() {
            ArrayData::Float64(arr) => arr[[]], // 0-d indexing
            _ => panic!("expected Float64"),
        }
    }

    /// Extract f64 values as a Vec from a Float64 NdArray.
    fn to_f64_vec(a: &NdArray) -> Vec<f64> {
        match a.data() {
            ArrayData::Float64(arr) => arr.iter().copied().collect(),
            _ => panic!("expected Float64"),
        }
    }

    /// Extract the scalar i64 value from a 0-d Int64 NdArray.
    fn scalar_i64(a: &NdArray) -> i64 {
        match a.data() {
            ArrayData::Int64(arr) => arr[[]],
            _ => panic!("expected Int64"),
        }
    }

    /// Extract i64 values as a Vec.
    fn to_i64_vec(a: &NdArray) -> Vec<i64> {
        match a.data() {
            ArrayData::Int64(arr) => arr.iter().copied().collect(),
            _ => panic!("expected Int64"),
        }
    }

    // ===== nansum tests =====

    #[test]
    fn test_nansum_basic() {
        let a = arr1(&[1.0, f64::NAN, 3.0]);
        let r = a.nansum(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 4.0);
    }

    #[test]
    fn test_nansum_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        let r = a.nansum(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 0.0);
    }

    #[test]
    fn test_nansum_no_nan() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let r = a.nansum(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 6.0);
    }

    #[test]
    fn test_nansum_2d_axis0() {
        // [[1, NaN], [3, 4]]
        let a = arr2(2, 2, &[1.0, f64::NAN, 3.0, 4.0]);
        let r = a.nansum(Some(0), false).unwrap();
        assert_eq!(r.shape(), &[2]);
        let vals = to_f64_vec(&r);
        assert_eq!(vals[0], 4.0);
        assert_eq!(vals[1], 4.0);
    }

    #[test]
    fn test_nansum_2d_axis1() {
        // [[1, NaN], [3, 4]]
        let a = arr2(2, 2, &[1.0, f64::NAN, 3.0, 4.0]);
        let r = a.nansum(Some(1), false).unwrap();
        assert_eq!(r.shape(), &[2]);
        let vals = to_f64_vec(&r);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 7.0);
    }

    #[test]
    fn test_nansum_keepdims() {
        let a = arr2(2, 3, &[1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nansum(Some(1), true).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
    }

    #[test]
    fn test_nansum_keepdims_none() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nansum(None, true).unwrap();
        assert_eq!(r.shape(), &[1, 1]);
    }

    // ===== nanmean tests =====

    #[test]
    fn test_nanmean_basic() {
        let a = arr1(&[1.0, f64::NAN, 3.0]);
        let r = a.nanmean(None, false).unwrap();
        assert!((scalar_f64(&r) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanmean_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        let r = a.nanmean(None, false).unwrap();
        assert!(scalar_f64(&r).is_nan());
    }

    #[test]
    fn test_nanmean_no_nan() {
        let a = arr1(&[2.0, 4.0, 6.0]);
        let r = a.nanmean(None, false).unwrap();
        assert!((scalar_f64(&r) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanmean_2d_axis0() {
        // [[1, NaN], [3, 4]]
        let a = arr2(2, 2, &[1.0, f64::NAN, 3.0, 4.0]);
        let r = a.nanmean(Some(0), false).unwrap();
        let vals = to_f64_vec(&r);
        assert!((vals[0] - 2.0).abs() < 1e-10); // mean(1,3) = 2
        assert!((vals[1] - 4.0).abs() < 1e-10); // mean(4) = 4, NaN filtered
    }

    #[test]
    fn test_nanmean_keepdims() {
        let a = arr2(2, 3, &[1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nanmean(Some(0), true).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
    }

    // ===== nanvar tests =====

    #[test]
    fn test_nanvar_basic() {
        // [1, NaN, 3]: filtered = [1, 3], mean = 2, var = ((1-2)^2 + (3-2)^2)/2 = 1.0
        let a = arr1(&[1.0, f64::NAN, 3.0]);
        let r = a.nanvar(None, 0, false).unwrap();
        assert!((scalar_f64(&r) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanvar_ddof1() {
        // [1, NaN, 3]: filtered = [1, 3], mean = 2, var = 2/(2-1) = 2.0
        let a = arr1(&[1.0, f64::NAN, 3.0]);
        let r = a.nanvar(None, 1, false).unwrap();
        assert!((scalar_f64(&r) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanvar_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        let r = a.nanvar(None, 0, false).unwrap();
        assert!(scalar_f64(&r).is_nan());
    }

    #[test]
    fn test_nanvar_ddof_exceeds_count() {
        // [1.0, NaN]: filtered count = 1, ddof = 1 => n <= ddof => NaN
        let a = arr1(&[1.0, f64::NAN]);
        let r = a.nanvar(None, 1, false).unwrap();
        assert!(scalar_f64(&r).is_nan());
    }

    #[test]
    fn test_nanvar_no_nan() {
        // [1, 2, 3]: mean = 2, var = (1+0+1)/3 = 2/3
        let a = arr1(&[1.0, 2.0, 3.0]);
        let r = a.nanvar(None, 0, false).unwrap();
        assert!((scalar_f64(&r) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanvar_2d_axis1() {
        // [[1, NaN, 3], [4, 5, 6]]
        let a = arr2(2, 3, &[1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nanvar(Some(1), 0, false).unwrap();
        let vals = to_f64_vec(&r);
        // row 0: [1, 3], mean=2, var=1.0
        assert!((vals[0] - 1.0).abs() < 1e-10);
        // row 1: [4, 5, 6], mean=5, var = (1+0+1)/3 = 2/3
        assert!((vals[1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanvar_keepdims() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nanvar(Some(1), 0, true).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
    }

    // ===== nanstd tests =====

    #[test]
    fn test_nanstd_basic() {
        // [1, NaN, 3]: var = 1.0, std = 1.0
        let a = arr1(&[1.0, f64::NAN, 3.0]);
        let r = a.nanstd(None, 0, false).unwrap();
        assert!((scalar_f64(&r) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanstd_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        let r = a.nanstd(None, 0, false).unwrap();
        assert!(scalar_f64(&r).is_nan());
    }

    #[test]
    fn test_nanstd_ddof1() {
        // [1, NaN, 3]: var(ddof=1) = 2.0, std = sqrt(2)
        let a = arr1(&[1.0, f64::NAN, 3.0]);
        let r = a.nanstd(None, 1, false).unwrap();
        assert!((scalar_f64(&r) - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_nanstd_2d_axis0() {
        // [[1, 2], [3, 4]]
        let a = arr2(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let r = a.nanstd(Some(0), 0, false).unwrap();
        let vals = to_f64_vec(&r);
        // col 0: [1, 3], std = 1.0; col 1: [2, 4], std = 1.0
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nanstd_keepdims() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nanstd(Some(0), 0, true).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
    }

    // ===== nanmin tests =====

    #[test]
    fn test_nanmin_basic() {
        let a = arr1(&[3.0, f64::NAN, 1.0]);
        let r = a.nanmin(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 1.0);
    }

    #[test]
    fn test_nanmin_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        assert!(a.nanmin(None, false).is_err());
    }

    #[test]
    fn test_nanmin_no_nan() {
        let a = arr1(&[5.0, 2.0, 8.0]);
        let r = a.nanmin(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 2.0);
    }

    #[test]
    fn test_nanmin_2d_axis0() {
        // [[NaN, 2], [3, 1]]
        let a = arr2(2, 2, &[f64::NAN, 2.0, 3.0, 1.0]);
        let r = a.nanmin(Some(0), false).unwrap();
        let vals = to_f64_vec(&r);
        assert_eq!(vals[0], 3.0); // min(NaN, 3) ignoring NaN = 3
        assert_eq!(vals[1], 1.0); // min(2, 1) = 1
    }

    #[test]
    fn test_nanmin_2d_axis1() {
        // [[NaN, 2], [3, 1]]
        let a = arr2(2, 2, &[f64::NAN, 2.0, 3.0, 1.0]);
        let r = a.nanmin(Some(1), false).unwrap();
        let vals = to_f64_vec(&r);
        assert_eq!(vals[0], 2.0); // row 0: min(NaN, 2) = 2
        assert_eq!(vals[1], 1.0); // row 1: min(3, 1) = 1
    }

    #[test]
    fn test_nanmin_all_nan_axis() {
        // [[NaN, NaN], [3, 1]]
        let a = arr2(2, 2, &[f64::NAN, f64::NAN, 3.0, 1.0]);
        // axis=1: row 0 is all NaN -> error
        assert!(a.nanmin(Some(1), false).is_err());
    }

    #[test]
    fn test_nanmin_keepdims() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nanmin(Some(1), true).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
    }

    // ===== nanmax tests =====

    #[test]
    fn test_nanmax_basic() {
        let a = arr1(&[1.0, f64::NAN, 3.0]);
        let r = a.nanmax(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 3.0);
    }

    #[test]
    fn test_nanmax_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        assert!(a.nanmax(None, false).is_err());
    }

    #[test]
    fn test_nanmax_no_nan() {
        let a = arr1(&[5.0, 2.0, 8.0]);
        let r = a.nanmax(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 8.0);
    }

    #[test]
    fn test_nanmax_2d_axis0() {
        // [[NaN, 2], [3, 1]]
        let a = arr2(2, 2, &[f64::NAN, 2.0, 3.0, 1.0]);
        let r = a.nanmax(Some(0), false).unwrap();
        let vals = to_f64_vec(&r);
        assert_eq!(vals[0], 3.0); // max(NaN, 3) ignoring NaN = 3
        assert_eq!(vals[1], 2.0); // max(2, 1) = 2
    }

    #[test]
    fn test_nanmax_all_nan_axis() {
        let a = arr2(2, 2, &[f64::NAN, f64::NAN, 3.0, 1.0]);
        // axis=1: row 0 is all NaN -> error
        assert!(a.nanmax(Some(1), false).is_err());
    }

    #[test]
    fn test_nanmax_keepdims() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nanmax(Some(0), true).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
    }

    // ===== nanargmin tests =====

    #[test]
    fn test_nanargmin_basic() {
        let a = arr1(&[3.0, f64::NAN, 1.0, 2.0]);
        let r = a.nanargmin(None).unwrap();
        assert_eq!(scalar_i64(&r), 2);
    }

    #[test]
    fn test_nanargmin_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        assert!(a.nanargmin(None).is_err());
    }

    #[test]
    fn test_nanargmin_no_nan() {
        let a = arr1(&[5.0, 2.0, 8.0]);
        let r = a.nanargmin(None).unwrap();
        assert_eq!(scalar_i64(&r), 1);
    }

    #[test]
    fn test_nanargmin_2d_axis1() {
        // [[3, NaN, 1], [6, 4, 5]]
        let a = arr2(2, 3, &[3.0, f64::NAN, 1.0, 6.0, 4.0, 5.0]);
        let r = a.nanargmin(Some(1)).unwrap();
        let vals = to_i64_vec(&r);
        assert_eq!(vals[0], 2); // row 0: min of [3, 1] at index 2
        assert_eq!(vals[1], 1); // row 1: min of [6, 4, 5] at index 1
    }

    #[test]
    fn test_nanargmin_all_nan_axis() {
        let a = arr2(2, 2, &[f64::NAN, f64::NAN, 3.0, 1.0]);
        // axis=1: row 0 is all NaN -> error
        assert!(a.nanargmin(Some(1)).is_err());
    }

    // ===== nanargmax tests =====

    #[test]
    fn test_nanargmax_basic() {
        let a = arr1(&[1.0, f64::NAN, 3.0, 2.0]);
        let r = a.nanargmax(None).unwrap();
        assert_eq!(scalar_i64(&r), 2);
    }

    #[test]
    fn test_nanargmax_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        assert!(a.nanargmax(None).is_err());
    }

    #[test]
    fn test_nanargmax_no_nan() {
        let a = arr1(&[5.0, 8.0, 2.0]);
        let r = a.nanargmax(None).unwrap();
        assert_eq!(scalar_i64(&r), 1);
    }

    #[test]
    fn test_nanargmax_2d_axis0() {
        // [[1, NaN], [3, 4]]
        let a = arr2(2, 2, &[1.0, f64::NAN, 3.0, 4.0]);
        let r = a.nanargmax(Some(0)).unwrap();
        let vals = to_i64_vec(&r);
        assert_eq!(vals[0], 1); // col 0: max(1, 3) at row 1
        assert_eq!(vals[1], 1); // col 1: max(NaN, 4) ignoring NaN => row 1
    }

    #[test]
    fn test_nanargmax_all_nan_axis() {
        let a = arr2(2, 2, &[f64::NAN, f64::NAN, 3.0, 1.0]);
        assert!(a.nanargmax(Some(1)).is_err());
    }

    // ===== nanprod tests =====

    #[test]
    fn test_nanprod_basic() {
        let a = arr1(&[2.0, f64::NAN, 3.0]);
        let r = a.nanprod(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 6.0);
    }

    #[test]
    fn test_nanprod_all_nan() {
        let a = arr1(&[f64::NAN, f64::NAN]);
        let r = a.nanprod(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 1.0); // product of empty = 1
    }

    #[test]
    fn test_nanprod_no_nan() {
        let a = arr1(&[2.0, 3.0, 4.0]);
        let r = a.nanprod(None, false).unwrap();
        assert_eq!(scalar_f64(&r), 24.0);
    }

    #[test]
    fn test_nanprod_2d_axis0() {
        // [[2, NaN], [3, 4]]
        let a = arr2(2, 2, &[2.0, f64::NAN, 3.0, 4.0]);
        let r = a.nanprod(Some(0), false).unwrap();
        let vals = to_f64_vec(&r);
        assert_eq!(vals[0], 6.0); // 2 * 3
        assert_eq!(vals[1], 4.0); // 1 * 4 (NaN treated as 1)
    }

    #[test]
    fn test_nanprod_keepdims() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = a.nanprod(Some(0), true).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
    }

    // ===== Invalid axis tests =====

    #[test]
    fn test_nansum_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nansum(Some(5), false).is_err());
    }

    #[test]
    fn test_nanmean_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanmean(Some(5), false).is_err());
    }

    #[test]
    fn test_nanvar_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanvar(Some(5), 0, false).is_err());
    }

    #[test]
    fn test_nanstd_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanstd(Some(5), 0, false).is_err());
    }

    #[test]
    fn test_nanmin_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanmin(Some(5), false).is_err());
    }

    #[test]
    fn test_nanmax_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanmax(Some(5), false).is_err());
    }

    #[test]
    fn test_nanargmin_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanargmin(Some(5)).is_err());
    }

    #[test]
    fn test_nanargmax_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanargmax(Some(5)).is_err());
    }

    #[test]
    fn test_nanprod_invalid_axis() {
        let a = arr1(&[1.0, 2.0]);
        assert!(a.nanprod(Some(5), false).is_err());
    }

    // ===== Integer input (casts to Float64) =====

    #[test]
    fn test_nansum_int_input() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let r = a.nansum(None, false).unwrap();
        assert_eq!(r.dtype(), DType::Float64);
        assert_eq!(scalar_f64(&r), 6.0);
    }

    #[test]
    fn test_nanmean_int_input() {
        let a = NdArray::from_vec(vec![2_i64, 4, 6]);
        let r = a.nanmean(None, false).unwrap();
        assert!((scalar_f64(&r) - 4.0).abs() < 1e-10);
    }

    // ===== Matching regular reductions when no NaN =====

    #[test]
    fn test_nansum_matches_sum_no_nan() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let nansum = scalar_f64(&a.nansum(None, false).unwrap());
        let sum = scalar_f64(&a.sum(None, false).unwrap());
        assert!((nansum - sum).abs() < 1e-10);
    }

    #[test]
    fn test_nanprod_matches_prod_no_nan() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let nanprod = scalar_f64(&a.nanprod(None, false).unwrap());
        let prod = scalar_f64(&a.prod(None, false).unwrap());
        assert!((nanprod - prod).abs() < 1e-10);
    }

    #[test]
    fn test_nanmin_matches_min_no_nan() {
        let a = arr1(&[5.0, 1.0, 3.0]);
        let nanmin = scalar_f64(&a.nanmin(None, false).unwrap());
        let min = scalar_f64(&a.min(None, false).unwrap());
        assert!((nanmin - min).abs() < 1e-10);
    }

    #[test]
    fn test_nanmax_matches_max_no_nan() {
        let a = arr1(&[5.0, 1.0, 3.0]);
        let nanmax = scalar_f64(&a.nanmax(None, false).unwrap());
        let max = scalar_f64(&a.max(None, false).unwrap());
        assert!((nanmax - max).abs() < 1e-10);
    }
}
