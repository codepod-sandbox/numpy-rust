use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Compute the covariance matrix.
    ///
    /// When `rowvar` is true (the default), each row represents a variable
    /// and each column an observation.  For a 1-D input the array is treated
    /// as a single variable (1 × N).
    ///
    /// `ddof` is the "delta degrees of freedom" divisor: the result is
    /// normalised by `N - ddof` where N is the number of observations.
    pub fn cov(&self, rowvar: bool, ddof: usize) -> Result<NdArray> {
        // Cast to Float64
        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else {
            return Err(NumpyError::TypeError("cov requires numeric input".into()));
        };

        // Determine the 2-D observation matrix (variables × observations).
        let mat = match arr.ndim() {
            0 => {
                return Err(NumpyError::ValueError(
                    "cov requires at least a 1-D array".into(),
                ));
            }
            1 => {
                // Treat as a single variable (1 × N).
                let n = arr.len();
                arr.clone()
                    .into_shape_with_order(IxDyn(&[1, n]))
                    .map_err(|e| NumpyError::ValueError(e.to_string()))?
            }
            2 => {
                if rowvar {
                    arr.clone()
                } else {
                    // columns = variables → transpose so rows = variables
                    arr.t().to_owned().into_dimensionality().unwrap()
                }
            }
            _ => {
                return Err(NumpyError::ValueError(
                    "cov requires a 1-D or 2-D array".into(),
                ));
            }
        };

        let num_vars = mat.shape()[0]; // number of variables
        let num_obs = mat.shape()[1]; // number of observations

        if num_obs < 1 {
            return Err(NumpyError::ValueError(
                "cov requires at least one observation".into(),
            ));
        }
        if ddof >= num_obs {
            return Err(NumpyError::ValueError(format!(
                "ddof ({}) must be less than the number of observations ({})",
                ddof, num_obs
            )));
        }

        // Center: subtract mean of each row (variable).
        let mut centered = mat.clone();
        for i in 0..num_vars {
            let mut row_sum = 0.0;
            for j in 0..num_obs {
                row_sum += mat[IxDyn(&[i, j])];
            }
            let row_mean = row_sum / num_obs as f64;
            for j in 0..num_obs {
                centered[IxDyn(&[i, j])] -= row_mean;
            }
        }

        // Compute covariance: C = centered @ centered^T / (N - ddof)
        let norm = (num_obs - ddof) as f64;
        let mut cov_mat = ArrayD::<f64>::zeros(IxDyn(&[num_vars, num_vars]));
        for i in 0..num_vars {
            for j in i..num_vars {
                let mut sum = 0.0;
                for k in 0..num_obs {
                    sum += centered[IxDyn(&[i, k])] * centered[IxDyn(&[j, k])];
                }
                let val = sum / norm;
                cov_mat[IxDyn(&[i, j])] = val;
                cov_mat[IxDyn(&[j, i])] = val;
            }
        }

        Ok(NdArray::from_data(ArrayData::Float64(cov_mat)))
    }

    /// Compute the Pearson correlation coefficient matrix.
    ///
    /// Each row represents a variable, each column an observation
    /// (when `rowvar` is true).
    ///
    /// Internally this computes `cov(ddof=0)` and normalises:
    /// `R[i,j] = C[i,j] / sqrt(C[i,i] * C[j,j])`.
    pub fn corrcoef(&self, rowvar: bool) -> Result<NdArray> {
        let cov_arr = self.cov(rowvar, 0)?;
        let ArrayData::Float64(c) = &cov_arr.data else {
            unreachable!();
        };

        let n = c.shape()[0];
        let mut r = ArrayD::<f64>::zeros(IxDyn(&[n, n]));

        for i in 0..n {
            for j in 0..n {
                let denom = (c[IxDyn(&[i, i])] * c[IxDyn(&[j, j])]).sqrt();
                if denom == 0.0 {
                    r[IxDyn(&[i, j])] = if i == j { 1.0 } else { f64::NAN };
                } else {
                    // Clamp to [-1, 1] to avoid floating-point artefacts
                    let val = (c[IxDyn(&[i, j])] / denom).clamp(-1.0, 1.0);
                    r[IxDyn(&[i, j])] = val;
                }
            }
        }

        Ok(NdArray::from_data(ArrayData::Float64(r)))
    }
}

/// Compute the covariance matrix for two 1-D arrays.
///
/// Stacks `x` and `y` into a 2 × N matrix and calls `NdArray::cov`.
pub fn cov_xy(x: &NdArray, y: &NdArray, ddof: usize) -> Result<NdArray> {
    let x_f = x.astype(DType::Float64);
    let y_f = y.astype(DType::Float64);

    if x_f.ndim() != 1 || y_f.ndim() != 1 {
        return Err(NumpyError::ValueError("cov_xy requires 1-D arrays".into()));
    }
    if x_f.size() != y_f.size() {
        return Err(NumpyError::ValueError(
            "cov_xy requires arrays of the same length".into(),
        ));
    }

    let n = x_f.size();
    let stacked = crate::concatenate(&[&x_f, &y_f], 0)?;
    let mat = stacked.reshape(&[2, n])?;
    mat.cov(true, ddof)
}

/// Compute the Pearson correlation coefficient matrix for two 1-D arrays.
///
/// Stacks `x` and `y` into a 2 × N matrix and calls `NdArray::corrcoef`.
pub fn corrcoef_xy(x: &NdArray, y: &NdArray) -> Result<NdArray> {
    let x_f = x.astype(DType::Float64);
    let y_f = y.astype(DType::Float64);

    if x_f.ndim() != 1 || y_f.ndim() != 1 {
        return Err(NumpyError::ValueError(
            "corrcoef_xy requires 1-D arrays".into(),
        ));
    }
    if x_f.size() != y_f.size() {
        return Err(NumpyError::ValueError(
            "corrcoef_xy requires arrays of the same length".into(),
        ));
    }

    let n = x_f.size();
    let stacked = crate::concatenate(&[&x_f, &y_f], 0)?;
    let mat = stacked.reshape(&[2, n])?;
    mat.corrcoef(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {} ≈ {} (tol={}), diff={}",
            a,
            b,
            tol,
            (a - b).abs()
        );
    }

    #[test]
    fn test_cov_known_data() {
        // x = [1, 2, 3], y = [4, 5, 6]
        // cov(x, x) = 1.0, cov(y, y) = 1.0, cov(x, y) = 1.0
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = cov_xy(&x, &y, 1).unwrap();
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        assert_eq!(c.shape(), &[2, 2]);
        approx_eq(arr[IxDyn(&[0, 0])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[0, 1])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 0])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 1])], 1.0, 1e-10);
    }

    #[test]
    fn test_corrcoef_perfectly_correlated() {
        // x = [1, 2, 3], y = [2, 4, 6]  -> perfect positive correlation
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let c = corrcoef_xy(&x, &y).unwrap();
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        assert_eq!(c.shape(), &[2, 2]);
        approx_eq(arr[IxDyn(&[0, 0])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[0, 1])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 0])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 1])], 1.0, 1e-10);
    }

    #[test]
    fn test_cov_xy_free_function() {
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let y = NdArray::from_vec(vec![5.0_f64, 4.0, 3.0, 2.0, 1.0]);
        let c = cov_xy(&x, &y, 1).unwrap();
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        // x and y are perfectly negatively correlated
        // var(x) = 2.5, cov(x,y) = -2.5
        approx_eq(arr[IxDyn(&[0, 0])], 2.5, 1e-10);
        approx_eq(arr[IxDyn(&[0, 1])], -2.5, 1e-10);
        approx_eq(arr[IxDyn(&[1, 0])], -2.5, 1e-10);
        approx_eq(arr[IxDyn(&[1, 1])], 2.5, 1e-10);
    }

    #[test]
    fn test_corrcoef_xy_free_function() {
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let y = NdArray::from_vec(vec![5.0_f64, 4.0, 3.0, 2.0, 1.0]);
        let c = corrcoef_xy(&x, &y).unwrap();
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        // Perfect negative correlation
        approx_eq(arr[IxDyn(&[0, 0])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[0, 1])], -1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 0])], -1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 1])], 1.0, 1e-10);
    }

    #[test]
    fn test_cov_1d_single_variable() {
        // 1-D input -> treated as single variable, result is 1x1 matrix
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = x.cov(true, 1).unwrap();
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        assert_eq!(c.shape(), &[1, 1]);
        approx_eq(arr[IxDyn(&[0, 0])], 1.0, 1e-10); // var([1,2,3]) = 1.0
    }

    #[test]
    fn test_cov_2d_rowvar_true() {
        // 2x3 matrix, rows=variables
        let data = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let c = data.cov(true, 1).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        approx_eq(arr[IxDyn(&[0, 0])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 1])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[0, 1])], 1.0, 1e-10);
    }

    #[test]
    fn test_corrcoef_1d() {
        // 1-D input should give 1x1 matrix with value 1.0
        let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = x.corrcoef(true).unwrap();
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        assert_eq!(c.shape(), &[1, 1]);
        approx_eq(arr[IxDyn(&[0, 0])], 1.0, 1e-10);
    }
}
