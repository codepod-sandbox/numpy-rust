use ndarray::{ArrayD, Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

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
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "quantile not supported for string arrays".into(),
            ));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "quantile not supported for complex arrays".into(),
            ));
        }

        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else {
            unreachable!()
        };

        match axis {
            None => {
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                if flat.is_empty() {
                    return Err(NumpyError::ValueError("empty array".into()));
                }
                flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let val = interpolate_quantile(&flat, q);
                Ok(NdArray::from_data(ArrayData::Float64(ArrayD::from_elem(
                    IxDyn(&[]),
                    val,
                ))))
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                    });
                }
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
                Ok(NdArray::from_data(ArrayData::Float64(result)))
            }
        }
    }

    /// Compute the q-th percentile (0 to 100).
    pub fn percentile(&self, q: f64, axis: Option<usize>) -> Result<NdArray> {
        self.quantile(q / 100.0, axis)
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

        let f = self.flatten().astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else {
            unreachable!()
        };

        // Collect non-NaN values
        let values: Vec<f64> = arr.iter().copied().filter(|v| !v.is_nan()).collect();

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
        let int_arr = self.flatten().astype(DType::Int64);
        let ArrayData::Int64(arr) = &int_arr.data else {
            unreachable!()
        };

        let flat: Vec<i64> = arr.iter().copied().collect();

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
                let w_flat = w.flatten().astype(DType::Float64);
                let ArrayData::Float64(w_arr) = &w_flat.data else {
                    unreachable!()
                };
                if w_arr.len() != flat.len() {
                    return Err(NumpyError::ValueError(
                        "bincount: weights must have the same length as input".into(),
                    ));
                }
                let mut result = vec![0.0f64; out_len];
                for (i, &v) in flat.iter().enumerate() {
                    result[v as usize] += w_arr[i];
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
