//! Linear algebra operations backed by faer.
//! Only available with the `linalg` feature.

#[cfg(feature = "linalg")]
mod inner {
    use faer::linalg::solvers::{DenseSolveCore, Solve};
    use faer::{Mat, Side};
    use ndarray::{ArrayD, IxDyn};

    use crate::array_data::ArrayData;
    use crate::casting::cast_array_data;
    use crate::dtype::DType;
    use crate::error::{NumpyError, Result};
    use crate::NdArray;

    /// Convert a 2-D NdArray to faer::Mat<f64>.
    fn to_faer(data: &ArrayData) -> Result<Mat<f64>> {
        let f64_data = cast_array_data(data, DType::Float64);
        match f64_data {
            ArrayData::Float64(a) => {
                if a.ndim() != 2 {
                    return Err(NumpyError::ValueError(format!(
                        "expected 2-D array, got {}-D",
                        a.ndim()
                    )));
                }
                let (m, n) = (a.shape()[0], a.shape()[1]);
                Ok(Mat::from_fn(m, n, |i, j| a[[i, j]]))
            }
            _ => unreachable!(),
        }
    }

    /// Convert faer::Mat<f64> back to NdArray.
    fn from_faer(mat: &Mat<f64>) -> NdArray {
        let (m, n) = (mat.nrows(), mat.ncols());
        let data = ArrayD::from_shape_fn(IxDyn(&[m, n]), |idx| mat[(idx[0], idx[1])]);
        NdArray::from_data(ArrayData::Float64(data))
    }

    /// Matrix multiplication: (m x k) @ (k x n) -> (m x n).
    pub fn matmul(a: &NdArray, b: &NdArray) -> Result<NdArray> {
        let am = to_faer(&a.data)?;
        let bm = to_faer(&b.data)?;
        if am.ncols() != bm.nrows() {
            return Err(NumpyError::ShapeMismatch(format!(
                "matmul: shapes {:?} and {:?} not aligned",
                a.shape(),
                b.shape()
            )));
        }
        let result = &am * &bm;
        Ok(from_faer(&result))
    }

    /// Matrix inverse via LU decomposition.
    pub fn inv(a: &NdArray) -> Result<NdArray> {
        let m = to_faer(&a.data)?;
        check_square(&m)?;
        let lu = m.partial_piv_lu();
        let result = lu.inverse();
        Ok(from_faer(&result))
    }

    /// Solve Ax = b.
    pub fn solve(a: &NdArray, b: &NdArray) -> Result<NdArray> {
        let am = to_faer(&a.data)?;
        check_square(&am)?;
        let bm = to_faer(&b.data)?;
        let lu = am.partial_piv_lu();
        let result = lu.solve(&bm);
        Ok(from_faer(&result))
    }

    /// Determinant via eigenvalues.
    pub fn det(a: &NdArray) -> Result<f64> {
        let m = to_faer(&a.data)?;
        check_square(&m)?;
        let eigenvalues = m
            .eigenvalues_from_real()
            .map_err(|_| NumpyError::ValueError("eigenvalue computation failed".into()))?;
        let mut d_re = 1.0_f64;
        let mut d_im = 0.0_f64;
        for ev in &eigenvalues {
            let new_re = d_re * ev.re - d_im * ev.im;
            let new_im = d_re * ev.im + d_im * ev.re;
            d_re = new_re;
            d_im = new_im;
        }
        Ok(d_re)
    }

    /// Eigendecomposition for symmetric matrices.
    /// Returns (eigenvalues as 1-D array, eigenvectors as 2-D array).
    pub fn eig(a: &NdArray) -> Result<(NdArray, NdArray)> {
        let m = to_faer(&a.data)?;
        check_square(&m)?;
        let eigen = m
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| NumpyError::ValueError("eigendecomposition failed".into()))?;

        let n = m.nrows();
        let s = eigen.S();
        let eigenvalues =
            ArrayD::from_shape_fn(IxDyn(&[n]), |idx| s.column_vector()[idx[0]]);
        let u = eigen.U();
        let eigenvectors = Mat::from_fn(u.nrows(), u.ncols(), |i, j| u[(i, j)]);

        Ok((
            NdArray::from_data(ArrayData::Float64(eigenvalues)),
            from_faer(&eigenvectors),
        ))
    }

    /// Singular Value Decomposition.
    /// Returns (U, S, Vt) where A ≈ U @ diag(S) @ Vt.
    pub fn svd(a: &NdArray) -> Result<(NdArray, NdArray, NdArray)> {
        let m = to_faer(&a.data)?;
        let decomp = m
            .svd()
            .map_err(|_| NumpyError::ValueError("SVD computation failed".into()))?;

        let u_ref = decomp.U();
        let u_mat = Mat::from_fn(u_ref.nrows(), u_ref.ncols(), |i, j| u_ref[(i, j)]);
        let u = from_faer(&u_mat);

        let s_diag = decomp.S();
        let k = s_diag.column_vector().nrows();
        let s = ArrayD::from_shape_fn(IxDyn(&[k]), |idx| s_diag.column_vector()[idx[0]]);

        let v_ref = decomp.V();
        // NumPy returns Vt (V transposed)
        let vt_mat = Mat::from_fn(v_ref.ncols(), v_ref.nrows(), |i, j| v_ref[(j, i)]);

        Ok((
            u,
            NdArray::from_data(ArrayData::Float64(s)),
            from_faer(&vt_mat),
        ))
    }

    /// QR decomposition. Returns (Q, R).
    pub fn qr(a: &NdArray) -> Result<(NdArray, NdArray)> {
        let m = to_faer(&a.data)?;
        let decomp = m.qr();

        #[allow(non_snake_case)]
        let thin_Q = decomp.compute_thin_Q();
        #[allow(non_snake_case)]
        let thin_R_ref = decomp.thin_R();
        #[allow(non_snake_case)]
        let thin_R = Mat::from_fn(thin_R_ref.nrows(), thin_R_ref.ncols(), |i, j| {
            thin_R_ref[(i, j)]
        });

        Ok((from_faer(&thin_Q), from_faer(&thin_R)))
    }

    /// Frobenius norm.
    pub fn norm(a: &NdArray) -> Result<f64> {
        let f64_data = cast_array_data(&a.data, DType::Float64);
        match f64_data {
            ArrayData::Float64(arr) => {
                let sum_sq: f64 = arr.iter().map(|&x| x * x).sum();
                Ok(sum_sq.sqrt())
            }
            _ => unreachable!(),
        }
    }

    /// Cholesky decomposition for symmetric positive-definite matrices.
    /// Returns L such that A = L @ L^T.
    pub fn cholesky(a: &NdArray) -> Result<NdArray> {
        let m = to_faer(&a.data)?;
        check_square(&m)?;
        let llt = m
            .llt(Side::Lower)
            .map_err(|_| NumpyError::ValueError("matrix is not positive definite".into()))?;
        let l_ref = llt.L();
        let l_mat = Mat::from_fn(l_ref.nrows(), l_ref.ncols(), |i, j| l_ref[(i, j)]);
        Ok(from_faer(&l_mat))
    }

    fn check_square(m: &Mat<f64>) -> Result<()> {
        if m.nrows() != m.ncols() {
            return Err(NumpyError::ValueError(format!(
                "expected square matrix, got {}x{}",
                m.nrows(),
                m.ncols()
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "linalg")]
pub use inner::*;

#[cfg(test)]
#[cfg(feature = "linalg")]
mod tests {
    use super::*;
    use crate::{DType, NdArray};

    fn eye_3x3() -> NdArray {
        crate::creation::eye(3, DType::Float64)
    }

    #[test]
    fn test_matmul() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[3, 4], DType::Float64);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let b = NdArray::ones(&[2, 4], DType::Float64);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_inv_identity() {
        let a = eye_3x3();
        let a_inv = inv(&a).unwrap();
        assert_eq!(a_inv.shape(), &[3, 3]);
    }

    #[test]
    fn test_solve() {
        let a = eye_3x3();
        let b = NdArray::ones(&[3, 1], DType::Float64);
        let x = solve(&a, &b).unwrap();
        assert_eq!(x.shape(), &[3, 1]);
    }

    #[test]
    fn test_det_identity() {
        let a = eye_3x3();
        let d = det(&a).unwrap();
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eig() {
        let a = eye_3x3();
        let (vals, vecs) = eig(&a).unwrap();
        assert_eq!(vals.shape(), &[3]);
        assert_eq!(vecs.shape(), &[3, 3]);
    }

    #[test]
    fn test_svd() {
        let a = NdArray::ones(&[3, 2], DType::Float64);
        let (u, s, vt) = svd(&a).unwrap();
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);
    }

    #[test]
    fn test_qr() {
        let a = NdArray::ones(&[3, 2], DType::Float64);
        let (q, r) = qr(&a).unwrap();
        assert_eq!(q.shape()[0], 3);
        assert_eq!(r.shape()[1], 2);
    }

    #[test]
    fn test_norm() {
        let a = eye_3x3();
        let n = norm(&a).unwrap();
        assert!((n - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        let data = vec![4.0_f64, 2.0, 2.0, 3.0];
        let a = NdArray::from_vec(data).reshape(&[2, 2]).unwrap();
        let l = cholesky(&a).unwrap();
        assert_eq!(l.shape(), &[2, 2]);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        let data = vec![1.0_f64, 2.0, 2.0, 1.0];
        let a = NdArray::from_vec(data).reshape(&[2, 2]).unwrap();
        assert!(cholesky(&a).is_err());
    }

    #[test]
    fn test_inv_non_square() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        assert!(inv(&a).is_err());
    }
}
