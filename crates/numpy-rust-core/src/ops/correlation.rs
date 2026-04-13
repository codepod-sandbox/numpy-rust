use crate::array_data::ArrayD;
use ndarray::IxDyn;
use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

fn to_float64(array: &NdArray) -> Result<ArrayD<f64>> {
    let cast = array.astype(DType::Float64);
    let ArrayData::Float64(arr) = cast.data() else {
        return Err(NumpyError::TypeError("cov requires numeric input".into()));
    };
    Ok(arr.clone())
}

fn to_complex128(array: &NdArray) -> ArrayD<Complex<f64>> {
    let cast = array.astype(DType::Complex128);
    let ArrayData::Complex128(arr) = cast.data() else {
        unreachable!("complex128 cast must produce complex128 storage");
    };
    arr.clone()
}

fn normalize_matrix_f64(array: &NdArray, rowvar: bool) -> Result<ArrayD<f64>> {
    let arr = to_float64(array)?;
    match arr.ndim() {
        0 => Err(NumpyError::ValueError(
            "cov requires at least a 1-D array".into(),
        )),
        1 => {
            let n = arr.len();
            arr.into_shape_with_order(IxDyn(&[1, n]))
                .map_err(|e| NumpyError::ValueError(e.to_string()))
        }
        2 => {
            if rowvar {
                Ok(arr)
            } else {
                Ok(arr.t().to_owned().into())
            }
        }
        _ => Err(NumpyError::ValueError(
            "cov requires a 1-D or 2-D array".into(),
        )),
    }
}

fn normalize_matrix_complex128(array: &NdArray, rowvar: bool) -> Result<ArrayD<Complex<f64>>> {
    let arr = to_complex128(array);
    match arr.ndim() {
        0 => Err(NumpyError::ValueError(
            "cov requires at least a 1-D array".into(),
        )),
        1 => {
            let n = arr.len();
            arr.into_shape_with_order(IxDyn(&[1, n]))
                .map_err(|e| NumpyError::ValueError(e.to_string()))
        }
        2 => {
            if rowvar {
                Ok(arr)
            } else {
                Ok(arr.t().to_owned().into())
            }
        }
        _ => Err(NumpyError::ValueError(
            "cov requires a 1-D or 2-D array".into(),
        )),
    }
}

impl NdArray {
    /// Compute the covariance matrix.
    ///
    /// When `rowvar` is true (the default), each row represents a variable
    /// and each column an observation.  For a 1-D input the array is treated
    /// as a single variable (1 × N).
    ///
    /// `ddof` is the "delta degrees of freedom" divisor: the result is
    /// normalised by `N - ddof` where N is the number of observations.
    /// If ddof >= N, returns a matrix of ±Inf (matching NumPy behaviour).
    pub fn cov(&self, rowvar: bool, ddof: i64) -> Result<NdArray> {
        // Handle complex input separately
        if matches!(self.dtype(), DType::Complex64 | DType::Complex128) {
            return self.cov_complex(rowvar, ddof);
        }

        let mat = normalize_matrix_f64(self, rowvar)?;

        let num_vars = mat.shape()[0]; // number of variables
        let num_obs = mat.shape()[1]; // number of observations

        // Empty array: return NaN (matching NumPy behaviour)
        if num_obs == 0 {
            if num_vars == 0 {
                let empty = ArrayD::<f64>::zeros(IxDyn(&[0, 0]));
                return Ok(NdArray::from_data(ArrayData::Float64(empty)));
            }
            let nan_mat = ArrayD::<f64>::from_elem(IxDyn(&[num_vars, num_vars]), f64::NAN);
            return Ok(NdArray::from_data(ArrayData::Float64(nan_mat)));
        }

        // When ddof >= num_obs, result is ±Inf (NumPy emits a RuntimeWarning)
        let norm = num_obs as i64 - ddof;
        let norm_f = norm as f64;
        let degenerate = norm <= 0;

        // Center: subtract mean of each row (variable).
        let mut centered = mat.to_owned();
        for i in 0..num_vars {
            let mut row_sum = 0.0_f64;
            for j in 0..num_obs {
                row_sum += mat[IxDyn(&[i, j])];
            }
            let row_mean = row_sum / num_obs as f64;
            for j in 0..num_obs {
                centered[IxDyn(&[i, j])] -= row_mean;
            }
        }

        // Compute covariance: C = centered @ centered^T / (N - ddof)
        // When norm <= 0 (ddof >= N), return ±Inf based on sign of cross-product sum
        let mut cov_mat = ArrayD::<f64>::zeros(IxDyn(&[num_vars, num_vars]));
        for i in 0..num_vars {
            for j in i..num_vars {
                let mut sum = 0.0_f64;
                for k in 0..num_obs {
                    sum += centered[IxDyn(&[i, k])] * centered[IxDyn(&[j, k])];
                }
                let val = if degenerate {
                    if sum == 0.0 {
                        f64::NAN
                    } else if sum > 0.0 {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    }
                } else {
                    sum / norm_f
                };
                cov_mat[IxDyn(&[i, j])] = val;
                cov_mat[IxDyn(&[j, i])] = val;
            }
        }

        Ok(NdArray::from_data(ArrayData::Float64(cov_mat)))
    }

    fn cov_complex(&self, rowvar: bool, ddof: i64) -> Result<NdArray> {
        let mat = normalize_matrix_complex128(self, rowvar)?;

        let num_vars = mat.shape()[0];
        let num_obs = mat.shape()[1];

        if num_obs == 0 {
            if num_vars == 0 {
                let empty = ArrayD::<Complex<f64>>::zeros(IxDyn(&[0, 0]));
                return Ok(NdArray::from_data(ArrayData::Complex128(empty)));
            }
            let nan = Complex::new(f64::NAN, f64::NAN);
            let nan_mat = ArrayD::<Complex<f64>>::from_elem(IxDyn(&[num_vars, num_vars]), nan);
            return Ok(NdArray::from_data(ArrayData::Complex128(nan_mat)));
        }

        let norm = num_obs as i64 - ddof;
        let norm_f = norm as f64;
        let degenerate = norm <= 0;

        // Center each row
        let mut centered = mat.to_owned();
        for i in 0..num_vars {
            let mut row_sum = Complex::new(0.0_f64, 0.0);
            for j in 0..num_obs {
                row_sum += mat[IxDyn(&[i, j])];
            }
            let row_mean = row_sum / num_obs as f64;
            for j in 0..num_obs {
                centered[IxDyn(&[i, j])] -= row_mean;
            }
        }

        // C[i,j] = sum_k(centered[i,k] * conj(centered[j,k])) / norm
        // When norm <= 0, return ±Inf based on sign of sum components
        let mut cov_mat = ArrayD::<Complex<f64>>::zeros(IxDyn(&[num_vars, num_vars]));
        for i in 0..num_vars {
            for j in 0..num_vars {
                let mut sum = Complex::new(0.0_f64, 0.0);
                for k in 0..num_obs {
                    sum += centered[IxDyn(&[i, k])] * centered[IxDyn(&[j, k])].conj();
                }
                cov_mat[IxDyn(&[i, j])] = if degenerate {
                    let re = if sum.re == 0.0 {
                        f64::NAN
                    } else if sum.re > 0.0 {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                    let im = if sum.im == 0.0 {
                        0.0
                    } else if sum.im > 0.0 {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                    Complex::new(re, im)
                } else {
                    sum / norm_f
                };
            }
        }

        Ok(NdArray::from_data(ArrayData::Complex128(cov_mat)))
    }

    /// Compute the Pearson correlation coefficient matrix.
    ///
    /// Each row represents a variable, each column an observation
    /// (when `rowvar` is true).
    ///
    /// Internally this computes `cov(ddof=0)` and normalises:
    /// `R[i,j] = C[i,j] / (sqrt(C[i,i]) * sqrt(C[j,j]))`.
    /// Using two separate sqrt calls avoids overflow for extreme values.
    pub fn corrcoef(&self, rowvar: bool) -> Result<NdArray> {
        // Handle complex input
        if matches!(self.dtype(), DType::Complex64 | DType::Complex128) {
            return self.corrcoef_complex(rowvar);
        }

        let cov_arr = self.cov(rowvar, 0)?;
        let ArrayData::Float64(c) = cov_arr.data() else {
            unreachable!();
        };

        let n = c.shape()[0];
        let mut r = ArrayD::<f64>::zeros(IxDyn(&[n, n]));

        for i in 0..n {
            for j in 0..n {
                let si = c[IxDyn(&[i, i])].sqrt();
                let sj = c[IxDyn(&[j, j])].sqrt();
                let denom = si * sj;
                if denom == 0.0 || !denom.is_finite() {
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

    fn corrcoef_complex(&self, rowvar: bool) -> Result<NdArray> {
        let cov_arr = self.cov_complex(rowvar, 0)?;
        let ArrayData::Complex128(c) = cov_arr.data() else {
            unreachable!();
        };

        let n = c.shape()[0];
        let mut r = ArrayD::<Complex<f64>>::zeros(IxDyn(&[n, n]));

        for i in 0..n {
            for j in 0..n {
                // Diagonal elements of cov are real for Hermitian matrices
                let si = c[IxDyn(&[i, i])].re.sqrt();
                let sj = c[IxDyn(&[j, j])].re.sqrt();
                let denom = si * sj;
                if denom == 0.0 || !denom.is_finite() {
                    r[IxDyn(&[i, j])] = if i == j {
                        Complex::new(1.0, 0.0)
                    } else {
                        Complex::new(f64::NAN, f64::NAN)
                    };
                } else {
                    r[IxDyn(&[i, j])] = c[IxDyn(&[i, j])] / denom;
                }
            }
        }

        Ok(NdArray::from_data(ArrayData::Complex128(r)))
    }
}

/// Normalise an array for cov/corrcoef: promote to 2D (rowvar=true), vstack with optional y.
/// Accepts 1-D or 2-D inputs.  For 2-D input, uses rows as variables.
fn to_matrix(x: &NdArray, y: Option<&NdArray>, rowvar: bool) -> Result<NdArray> {
    let xf = x.astype(
        if matches!(x.dtype(), DType::Complex64 | DType::Complex128) {
            DType::Complex128
        } else {
            DType::Float64
        },
    );

    // Promote to 2-D
    let xm = match xf.ndim() {
        0 => {
            return Err(NumpyError::ValueError(
                "cov requires at least a 1-D array".into(),
            ))
        }
        1 => {
            let n = xf.size();
            xf.reshape(&[1, n])?
        }
        2 => {
            if rowvar {
                xf
            } else {
                xf.transpose()
            }
        }
        _ => {
            return Err(NumpyError::ValueError(
                "cov requires a 1-D or 2-D array".into(),
            ))
        }
    };

    if let Some(y) = y {
        let yf = y.astype(
            if matches!(y.dtype(), DType::Complex64 | DType::Complex128) {
                DType::Complex128
            } else {
                xm.dtype()
            },
        );
        let ym = match yf.ndim() {
            0 => {
                return Err(NumpyError::ValueError(
                    "cov requires at least a 1-D array".into(),
                ))
            }
            1 => {
                let n = yf.size();
                yf.reshape(&[1, n])?
            }
            2 => {
                if rowvar {
                    yf
                } else {
                    yf.transpose()
                }
            }
            _ => {
                return Err(NumpyError::ValueError(
                    "cov requires a 1-D or 2-D array".into(),
                ))
            }
        };
        crate::concatenate(&[&xm, &ym], 0)
    } else {
        Ok(xm)
    }
}

/// Compute the covariance matrix, accepting 1-D or 2-D x and optional y.
pub fn cov_xy(x: &NdArray, y: Option<&NdArray>, rowvar: bool, ddof: i64) -> Result<NdArray> {
    let mat = to_matrix(x, y, rowvar)?;
    mat.cov(true, ddof)
}

/// Compute the Pearson correlation coefficient matrix, accepting 1-D or 2-D x and optional y.
pub fn corrcoef_xy(x: &NdArray, y: Option<&NdArray>, rowvar: bool) -> Result<NdArray> {
    let mat = to_matrix(x, y, rowvar)?;
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
        let c = cov_xy(&x, Some(&y), true, 1).unwrap();
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
        let c = corrcoef_xy(&x, Some(&y), true).unwrap();
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
        let c = cov_xy(&x, Some(&y), true, 1).unwrap();
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
        let c = corrcoef_xy(&x, Some(&y), true).unwrap();
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

    #[test]
    fn test_corrcoef_extreme_values() {
        // Values like 1e-100 and 1e100 should not overflow during normalization
        // Diagonal elements must be 1.0
        let data = NdArray::from_vec(vec![1e-100_f64, 1e100, 1e100, 1e-100])
            .reshape(&[2, 2])
            .unwrap();
        let c = data.corrcoef(true).unwrap();
        let ArrayData::Float64(arr) = c.data() else {
            panic!("expected Float64");
        };
        approx_eq(arr[IxDyn(&[0, 0])], 1.0, 1e-10);
        approx_eq(arr[IxDyn(&[1, 1])], 1.0, 1e-10);
    }
}
