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
}
