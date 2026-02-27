use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::NdArray;

/// Evaluate polynomial with coefficients `p` at points `x` using Horner's method.
///
/// `p` is ordered highest-degree first: `p[0]*x^(n-1) + p[1]*x^(n-2) + ... + p[n-1]`.
pub fn polyval(p: &NdArray, x: &NdArray) -> NdArray {
    let p_f = p.astype(DType::Float64).flatten();
    let x_f = x.astype(DType::Float64).flatten();

    let ArrayData::Float64(p_arr) = &p_f.data else {
        unreachable!()
    };
    let ArrayData::Float64(x_arr) = &x_f.data else {
        unreachable!()
    };

    let coeffs: Vec<f64> = p_arr.iter().copied().collect();
    let n = coeffs.len();

    let mut result = Vec::with_capacity(x_arr.len());
    for &xi in x_arr.iter() {
        if n == 0 {
            result.push(0.0);
        } else {
            // Horner's method
            let mut val = coeffs[0];
            for c in &coeffs[1..] {
                val = val * xi + c;
            }
            result.push(val);
        }
    }

    NdArray::from_vec(result)
}

/// Fit a polynomial of degree `deg` to data points (x, y) via least-squares.
///
/// Builds a Vandermonde matrix V where V\[i,j\] = x\[i\]^(deg-j) and solves
/// V @ p = y using SVD. Returns coefficients ordered highest-degree first,
/// matching NumPy's `polyfit` convention.
#[cfg(feature = "linalg")]
pub fn polyfit(x: &NdArray, y: &NdArray, deg: usize) -> crate::error::Result<NdArray> {
    use crate::error::NumpyError;
    use nalgebra::DMatrix;

    let x_f = x.astype(DType::Float64).flatten();
    let y_f = y.astype(DType::Float64).flatten();

    let ArrayData::Float64(x_arr) = &x_f.data else {
        unreachable!()
    };
    let ArrayData::Float64(y_arr) = &y_f.data else {
        unreachable!()
    };

    let m = x_arr.len();
    if m != y_arr.len() {
        return Err(NumpyError::ValueError(
            "x and y must have the same length".into(),
        ));
    }
    if m == 0 {
        return Err(NumpyError::ValueError("x must not be empty".into()));
    }
    if deg + 1 > m {
        return Err(NumpyError::ValueError(
            "degree is too large for the number of data points".into(),
        ));
    }

    let ncols = deg + 1;

    // Build Vandermonde matrix: V[i, j] = x[i]^(deg - j)
    let v = DMatrix::from_fn(m, ncols, |i, j| {
        let exp = (deg - j) as i32;
        x_arr[i].powi(exp)
    });

    // Build RHS column vector
    let b = DMatrix::from_fn(m, 1, |i, _| y_arr[i]);

    // Solve via SVD least-squares
    let svd = v.svd(true, true);
    let solution = svd
        .solve(&b, 1e-14)
        .map_err(|_| NumpyError::ValueError("SVD solve failed".into()))?;

    // Extract coefficients into a 1-D array
    let coeffs: Vec<f64> = (0..ncols).map(|j| solution[(j, 0)]).collect();

    Ok(NdArray::from_vec(coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyval_constant() {
        // p(x) = 5
        let p = NdArray::from_vec(vec![5.0_f64]);
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0]);
        let result = polyval(&p, &x);
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]] - 5.0).abs() < 1e-10);
        assert!((arr[[1]] - 5.0).abs() < 1e-10);
        assert!((arr[[2]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyval_linear() {
        // p(x) = 2x + 1
        let p = NdArray::from_vec(vec![2.0_f64, 1.0]);
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 3.0]);
        let result = polyval(&p, &x);
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]] - 1.0).abs() < 1e-10);
        assert!((arr[[1]] - 3.0).abs() < 1e-10);
        assert!((arr[[2]] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyval_quadratic() {
        // p(x) = x^2 - 2x + 3
        let p = NdArray::from_vec(vec![1.0_f64, -2.0, 3.0]);
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0]);
        let result = polyval(&p, &x);
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]] - 3.0).abs() < 1e-10); // 0 - 0 + 3
        assert!((arr[[1]] - 2.0).abs() < 1e-10); // 1 - 2 + 3
        assert!((arr[[2]] - 3.0).abs() < 1e-10); // 4 - 4 + 3
    }

    #[test]
    fn test_polyval_empty_coeffs() {
        let p = NdArray::from_vec(Vec::<f64>::new());
        let x = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let result = polyval(&p, &x);
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]]).abs() < 1e-10);
        assert!((arr[[1]]).abs() < 1e-10);
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_polyfit_linear() {
        // y = 2x + 1
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0]);
        let y = NdArray::from_vec(vec![1.0_f64, 3.0, 5.0, 7.0, 9.0]);
        let p = polyfit(&x, &y, 1).unwrap();
        let ArrayData::Float64(arr) = p.data() else {
            panic!()
        };
        assert!((arr[[0]] - 2.0).abs() < 1e-10); // slope
        assert!((arr[[1]] - 1.0).abs() < 1e-10); // intercept
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_polyfit_quadratic() {
        // y = x^2
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0]);
        let y = NdArray::from_vec(vec![0.0_f64, 1.0, 4.0, 9.0, 16.0]);
        let p = polyfit(&x, &y, 2).unwrap();
        let ArrayData::Float64(arr) = p.data() else {
            panic!()
        };
        assert!((arr[[0]] - 1.0).abs() < 1e-8); // x^2 coeff
        assert!((arr[[1]] - 0.0).abs() < 1e-8); // x coeff
        assert!((arr[[2]] - 0.0).abs() < 1e-8); // constant
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_polyfit_length_mismatch() {
        let x = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let y = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0]);
        assert!(polyfit(&x, &y, 1).is_err());
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_polyfit_degree_too_large() {
        let x = NdArray::from_vec(vec![0.0_f64, 1.0]);
        let y = NdArray::from_vec(vec![0.0_f64, 1.0]);
        assert!(polyfit(&x, &y, 2).is_err()); // 3 coefficients > 2 points
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_polyfit_polyval_roundtrip() {
        // Fit and evaluate should reproduce original data
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0]);
        let y = NdArray::from_vec(vec![1.0_f64, 0.0, 1.0, 4.0]); // y = x^2 - 2x + 1
        let p = polyfit(&x, &y, 2).unwrap();
        let y_hat = polyval(&p, &x);
        let ArrayData::Float64(y_arr) = &y.data else {
            panic!()
        };
        let ArrayData::Float64(yh_arr) = y_hat.data() else {
            panic!()
        };
        for i in 0..4 {
            assert!((yh_arr[[i]] - y_arr[[i]]).abs() < 1e-8);
        }
    }
}
