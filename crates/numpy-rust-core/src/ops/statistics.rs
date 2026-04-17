use crate::array_data::ArrayD;
use ndarray::{Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

fn to_float64(array: &NdArray) -> ArrayD<f64> {
    let cast = array.astype(DType::Float64);
    let ArrayData::Float64(arr) = cast.data() else {
        unreachable!("float64 cast must produce float64 storage")
    };
    arr.clone()
}

fn flatten_to_float64_vec(array: &NdArray) -> Vec<f64> {
    to_float64(&array.flatten()).iter().copied().collect()
}

fn flatten_to_int64_vec(array: &NdArray) -> Vec<i64> {
    let cast = array.flatten().astype(DType::Int64);
    let ArrayData::Int64(arr) = cast.data() else {
        unreachable!("int64 cast must produce int64 storage")
    };
    arr.iter().copied().collect()
}

fn validate_weights(weights: &[f64]) -> Result<()> {
    if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(NumpyError::ValueError(
            "Weights included NaN, inf or were all zero".into(),
        ));
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(NumpyError::ValueError(
            "Weights included NaN, inf or were all zero".into(),
        ));
    }
    Ok(())
}

fn weighted_inverted_cdf_1d(values: &[f64], weights: &[f64], q: f64) -> Result<f64> {
    if values.len() != weights.len() {
        return Err(NumpyError::ValueError(
            "values and weights must have the same length".into(),
        ));
    }
    if values.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    validate_weights(weights)?;

    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter(|(_, w)| *w > 0.0)
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total: f64 = pairs.iter().map(|(_, w)| *w).sum();
    let threshold = q * total;
    let mut cumulative = 0.0;
    for (value, weight) in pairs.iter().copied() {
        cumulative += weight;
        if cumulative >= threshold {
            return Ok(value);
        }
    }
    Ok(pairs
        .last()
        .map(|(value, _)| *value)
        .unwrap_or(f64::NAN))
}

fn prepare_numeric_float64_input(
    array: &NdArray,
    axis: Option<usize>,
    op_name: &str,
) -> Result<ArrayD<f64>> {
    if array.dtype().is_string() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for string arrays"
        )));
    }
    if array.dtype().is_complex() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for complex arrays"
        )));
    }

    let cast = to_float64(array);
    if let Some(ax) = axis {
        if ax >= cast.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis: ax,
                ndim: cast.ndim(),
            });
        }
    }

    Ok(cast)
}

fn execute_float64_axis_stat<FAll, FAxis>(
    array: &NdArray,
    axis: Option<usize>,
    op_name: &str,
    reduce_all: FAll,
    reduce_axis: FAxis,
) -> Result<NdArray>
where
    FAll: FnOnce(&ArrayD<f64>) -> Result<f64>,
    FAxis: FnOnce(&ArrayD<f64>, usize) -> Result<ArrayD<f64>>,
{
    let arr = prepare_numeric_float64_input(array, axis, op_name)?;

    Ok(match axis {
        None => NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(
            IxDyn(&[]),
            reduce_all(&arr)?,
        ))),
        Some(ax) => NdArray::from_data(ArrayData::Float64(reduce_axis(&arr, ax)?)),
    })
}

impl NdArray {
    /// Compute the q-th quantile (0.0 to 1.0) using linear interpolation.
    /// axis=None: compute over flattened array.
    pub fn quantile(&self, q: f64, axis: Option<usize>) -> Result<NdArray> {
        if !(0.0..=1.0).contains(&q) {
            return Err(NumpyError::ValueError(format!(
                "quantile q must be between 0 and 1, got {}",
                q
            )));
        }
        execute_float64_axis_stat(
            self,
            axis,
            "quantile",
            |arr| {
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                if flat.is_empty() {
                    return Err(NumpyError::ValueError("empty array".into()));
                }
                flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                Ok(interpolate_quantile(&flat, q))
            },
            |arr, ax| {
                let mut result_shape = arr.shape().to_vec();
                result_shape.remove(ax);
                let result_dim = if result_shape.is_empty() {
                    IxDyn(&[])
                } else {
                    IxDyn(&result_shape)
                };
                let mut result = ArrayD::<f64>::zeros(result_dim);
                for (lane, result_elem) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                    let mut v: Vec<f64> = lane.iter().copied().collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    *result_elem = interpolate_quantile(&v, q);
                }
                Ok(result)
            },
        )
    }

    /// Compute the q-th percentile (0 to 100).
    pub fn percentile(&self, q: f64, axis: Option<usize>) -> Result<NdArray> {
        self.quantile(q / 100.0, axis)
    }

    pub fn weighted_inverted_cdf_quantile(
        &self,
        q: f64,
        axis: Option<usize>,
        weights: &NdArray,
    ) -> Result<NdArray> {
        if !(0.0..=1.0).contains(&q) {
            return Err(NumpyError::ValueError(format!(
                "quantile q must be between 0 and 1, got {}",
                q
            )));
        }

        let arr = prepare_numeric_float64_input(self, axis, "quantile")?;
        let w_arr = prepare_numeric_float64_input(weights, axis, "quantile")?;

        match axis {
            None => {
                if arr.shape() != w_arr.shape() {
                    return Err(NumpyError::ValueError(
                        "Shape of weights must match shape of a when axis=None".into(),
                    ));
                }
                let vals: Vec<f64> = arr.iter().copied().collect();
                let wts: Vec<f64> = w_arr.iter().copied().collect();
                let result = weighted_inverted_cdf_1d(&vals, &wts, q)?;
                Ok(NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(
                    IxDyn(&[]),
                    result,
                ))))
            }
            Some(ax) => {
                if ax >= arr.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: arr.ndim(),
                    });
                }

                let result_shape: Vec<usize> = arr
                    .shape()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &dim)| if i == ax { None } else { Some(dim) })
                    .collect();
                let result_dim = if result_shape.is_empty() {
                    IxDyn(&[])
                } else {
                    IxDyn(&result_shape)
                };
                let mut result = ArrayD::<f64>::zeros(result_dim);

                if w_arr.ndim() == 1 {
                    if w_arr.len() != arr.shape()[ax] {
                        return Err(NumpyError::ValueError(
                            "Shape of weights must be consistent with shape of a along the reduction axis"
                                .into(),
                        ));
                    }
                    let shared_weights: Vec<f64> = w_arr.iter().copied().collect();
                    validate_weights(&shared_weights)?;
                    for (lane, result_elem) in arr.lanes(Axis(ax)).into_iter().zip(result.iter_mut()) {
                        let vals: Vec<f64> = lane.iter().copied().collect();
                        *result_elem = weighted_inverted_cdf_1d(&vals, &shared_weights, q)?;
                    }
                } else {
                    if arr.shape() != w_arr.shape() {
                        return Err(NumpyError::ValueError(
                            "Shape of weights must match shape of a or be 1-D along the reduction axis"
                                .into(),
                        ));
                    }
                    for ((lane, weight_lane), result_elem) in arr
                        .lanes(Axis(ax))
                        .into_iter()
                        .zip(w_arr.lanes(Axis(ax)).into_iter())
                        .zip(result.iter_mut())
                    {
                        let vals: Vec<f64> = lane.iter().copied().collect();
                        let wts: Vec<f64> = weight_lane.iter().copied().collect();
                        *result_elem = weighted_inverted_cdf_1d(&vals, &wts, q)?;
                    }
                }

                Ok(NdArray::from_data(ArrayData::Float64(result)))
            }
        }
    }

    /// Compute the histogram of a dataset.
    ///
    /// Flattens the input, casts to Float64, creates `bins` uniform bins
    /// from lo to hi, counts elements in each bin (NaN values skipped),
    /// and the right edge of the last bin is inclusive.
    ///
    /// Returns `(counts: Int64 array of length bins, bin_edges: Float64 array of length bins+1)`.
    pub fn histogram(&self, bins: usize, range: Option<(f64, f64)>) -> Result<(NdArray, NdArray)> {
        if bins == 0 {
            return Err(NumpyError::ValueError("bins must be > 0".into()));
        }

        let values: Vec<f64> = flatten_to_float64_vec(self)
            .into_iter()
            .filter(|v| !v.is_nan())
            .collect();

        let (lo, hi) = match range {
            Some((l, h)) => (l, h),
            None => {
                if values.is_empty() {
                    (0.0, 1.0)
                } else {
                    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    if min_val == max_val {
                        (min_val, min_val + 1.0)
                    } else {
                        (min_val, max_val)
                    }
                }
            }
        };

        let bin_width = (hi - lo) / bins as f64;

        // Build bin edges: bins+1 edges
        let edges: Vec<f64> = (0..=bins).map(|i| lo + i as f64 * bin_width).collect();

        // Count elements in each bin
        let mut counts = vec![0i64; bins];
        for &v in &values {
            if v < lo || v > hi {
                continue;
            }
            let idx = ((v - lo) / bin_width).floor() as usize;
            // Right edge of last bin is inclusive
            let idx = if idx >= bins { bins - 1 } else { idx };
            counts[idx] += 1;
        }

        let counts_arr = NdArray::from_vec(counts);
        let edges_arr = NdArray::from_vec(edges);
        Ok((counts_arr, edges_arr))
    }

    /// Count occurrences of each value in an array of non-negative integers.
    ///
    /// - Cast to Int64, flatten.
    /// - Validates all values are non-negative.
    /// - If weights provided, sums weights instead of counting (returns Float64).
    /// - Without weights, returns Int64.
    /// - `minlength` sets minimum output length.
    pub fn bincount(&self, weights: Option<&NdArray>, minlength: usize) -> Result<NdArray> {
        let flat = flatten_to_int64_vec(self);

        // Validate non-negative
        for &v in &flat {
            if v < 0 {
                return Err(NumpyError::ValueError(
                    "bincount: all values must be non-negative".into(),
                ));
            }
        }

        let max_val = flat.iter().copied().max().unwrap_or(-1);
        let out_len = std::cmp::max(minlength, (max_val + 1) as usize);

        match weights {
            Some(w) => {
                let weights = flatten_to_float64_vec(w);
                if weights.len() != flat.len() {
                    return Err(NumpyError::ValueError(
                        "bincount: weights must have the same length as input".into(),
                    ));
                }
                let mut result = vec![0.0f64; out_len];
                for (i, &v) in flat.iter().enumerate() {
                    result[v as usize] += weights[i];
                }
                Ok(NdArray::from_vec(result))
            }
            None => {
                let mut result = vec![0i64; out_len];
                for &v in &flat {
                    result[v as usize] += 1;
                }
                Ok(NdArray::from_vec(result))
            }
        }
    }
}

/// Linear interpolation for quantile on a sorted slice.
fn interpolate_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= n {
        sorted[n - 1]
    } else {
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}

#[cfg(test)]
mod tests {
    use crate::indexing::Scalar;
    use crate::NdArray;

    #[test]
    fn test_quantile_median() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let q = a.quantile(0.5, None).unwrap();
        assert_eq!(q.shape(), &[]);
        assert_eq!(q.get(&[]).unwrap(), Scalar::Float64(3.0));
    }

    #[test]
    fn test_quantile_min_max() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let q0 = a.quantile(0.0, None).unwrap();
        let q1 = a.quantile(1.0, None).unwrap();
        assert_eq!(q0.get(&[]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(q1.get(&[]).unwrap(), Scalar::Float64(3.0));
    }

    #[test]
    fn test_quantile_interpolated() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let q = a.quantile(0.25, None).unwrap();
        assert_eq!(q.get(&[]).unwrap(), Scalar::Float64(1.75));
    }

    #[test]
    fn test_quantile_axis() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0, 6.0, 4.0, 5.0])
            .reshape(&[2, 3])
            .unwrap();
        let q = a.quantile(0.5, Some(1)).unwrap();
        assert_eq!(q.shape(), &[2]);
    }

    #[test]
    fn test_percentile() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let p = a.percentile(25.0, None).unwrap();
        assert_eq!(p.get(&[]).unwrap(), Scalar::Float64(1.75));
    }

    #[test]
    fn test_histogram_5_bins() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let (counts, edges) = a.histogram(5, None).unwrap();
        assert_eq!(counts.shape(), &[5]);
        assert_eq!(edges.shape(), &[6]);
        // All 5 values should be distributed across 5 bins, totalling 5
        let total: i64 = (0..5)
            .map(|i| match counts.get(&[i]).unwrap() {
                Scalar::Int64(v) => v,
                _ => panic!("expected Int64"),
            })
            .sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_histogram_explicit_range() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let (counts, edges) = a.histogram(5, Some((0.0, 10.0))).unwrap();
        assert_eq!(counts.shape(), &[5]);
        assert_eq!(edges.shape(), &[6]);
        // With range 0..10 and 5 bins, bin width is 2.0
        // First edge should be 0.0, last edge should be 10.0
        assert_eq!(edges.get(&[0]).unwrap(), Scalar::Float64(0.0));
        assert_eq!(edges.get(&[5]).unwrap(), Scalar::Float64(10.0));
    }

    #[test]
    fn test_bincount_basic() {
        let a = NdArray::from_vec(vec![0i64, 1, 1, 2, 2, 2]);
        let c = a.bincount(None, 0).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.get(&[0]).unwrap(), Scalar::Int64(1));
        assert_eq!(c.get(&[1]).unwrap(), Scalar::Int64(2));
        assert_eq!(c.get(&[2]).unwrap(), Scalar::Int64(3));
    }

    #[test]
    fn test_bincount_minlength() {
        let a = NdArray::from_vec(vec![0i64, 1, 1]);
        let c = a.bincount(None, 5).unwrap();
        assert_eq!(c.shape(), &[5]);
        assert_eq!(c.get(&[0]).unwrap(), Scalar::Int64(1));
        assert_eq!(c.get(&[1]).unwrap(), Scalar::Int64(2));
        assert_eq!(c.get(&[2]).unwrap(), Scalar::Int64(0));
        assert_eq!(c.get(&[3]).unwrap(), Scalar::Int64(0));
        assert_eq!(c.get(&[4]).unwrap(), Scalar::Int64(0));
    }

    #[test]
    fn test_bincount_weights() {
        let a = NdArray::from_vec(vec![0i64, 1, 1, 2, 2, 2]);
        let w = NdArray::from_vec(vec![0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0]);
        let c = a.bincount(Some(&w), 0).unwrap();
        assert_eq!(c.shape(), &[3]);
        // bin 0: 0.5, bin 1: 1.0+1.5=2.5, bin 2: 2.0+2.5+3.0=7.5
        assert_eq!(c.get(&[0]).unwrap(), Scalar::Float64(0.5));
        assert_eq!(c.get(&[1]).unwrap(), Scalar::Float64(2.5));
        assert_eq!(c.get(&[2]).unwrap(), Scalar::Float64(7.5));
    }

    #[test]
    fn test_bincount_negative_error() {
        let a = NdArray::from_vec(vec![0i64, -1, 2]);
        let result = a.bincount(None, 0);
        assert!(result.is_err());
    }
}
