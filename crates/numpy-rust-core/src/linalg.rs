//! Linear algebra operations backed by nalgebra.
//! Only available with the `linalg` feature.

#[cfg(feature = "linalg")]
mod inner {
    use nalgebra::DMatrix;
    use ndarray::{ArrayD, IxDyn};

    use crate::array_data::ArrayData;
    use crate::casting::cast_array_data;
    use crate::dtype::DType;
    use crate::error::{NumpyError, Result};
    use crate::NdArray;

    /// Convert a 2-D NdArray to nalgebra DMatrix<f64>.
    fn to_nalgebra(data: &ArrayData) -> Result<DMatrix<f64>> {
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
                Ok(DMatrix::from_fn(m, n, |i, j| a[[i, j]]))
            }
            _ => unreachable!(),
        }
    }

    /// Convert nalgebra DMatrix<f64> back to NdArray.
    fn from_nalgebra(mat: &DMatrix<f64>) -> NdArray {
        let (m, n) = mat.shape();
        let data = ArrayD::from_shape_fn(IxDyn(&[m, n]), |idx| mat[(idx[0], idx[1])]);
        NdArray::from_data(ArrayData::Float64(data))
    }

    /// Matrix multiplication: (m x k) @ (k x n) -> (m x n).
    pub fn matmul(a: &NdArray, b: &NdArray) -> Result<NdArray> {
        let am = to_nalgebra(&a.data)?;
        let bm = to_nalgebra(&b.data)?;
        if am.ncols() != bm.nrows() {
            return Err(NumpyError::ShapeMismatch(format!(
                "matmul: shapes {:?} and {:?} not aligned",
                a.shape(),
                b.shape()
            )));
        }
        let result = &am * &bm;
        Ok(from_nalgebra(&result))
    }

    /// Matrix inverse via LU decomposition.
    pub fn inv(a: &NdArray) -> Result<NdArray> {
        let m = to_nalgebra(&a.data)?;
        check_square(&m)?;
        let result = m
            .clone()
            .try_inverse()
            .ok_or_else(|| NumpyError::ValueError("singular matrix".into()))?;
        Ok(from_nalgebra(&result))
    }

    /// Solve Ax = b.
    pub fn solve(a: &NdArray, b: &NdArray) -> Result<NdArray> {
        let am = to_nalgebra(&a.data)?;
        check_square(&am)?;
        let bm = to_nalgebra(&b.data)?;
        let lu = am.lu();
        let result = lu
            .solve(&bm)
            .ok_or_else(|| NumpyError::ValueError("singular matrix".into()))?;
        Ok(from_nalgebra(&result))
    }

    /// Determinant via LU decomposition.
    pub fn det(a: &NdArray) -> Result<f64> {
        let m = to_nalgebra(&a.data)?;
        check_square(&m)?;
        Ok(m.determinant())
    }

    /// Eigendecomposition for symmetric matrices.
    /// Returns (eigenvalues as 1-D array, eigenvectors as 2-D array).
    pub fn eig(a: &NdArray) -> Result<(NdArray, NdArray)> {
        let m = to_nalgebra(&a.data)?;
        check_square(&m)?;
        let symmetric =
            nalgebra::DMatrix::from_fn(m.nrows(), m.ncols(), |i, j| (m[(i, j)] + m[(j, i)]) / 2.0);
        let eigen = nalgebra::SymmetricEigen::new(symmetric);

        let n = m.nrows();
        let eigenvalues = ArrayD::from_shape_fn(IxDyn(&[n]), |idx| eigen.eigenvalues[idx[0]]);
        let eigenvectors = from_nalgebra(&eigen.eigenvectors);

        Ok((
            NdArray::from_data(ArrayData::Float64(eigenvalues)),
            eigenvectors,
        ))
    }

    /// Singular Value Decomposition.
    /// Returns (U, S, Vt) where A ≈ U @ diag(S) @ Vt.
    pub fn svd(a: &NdArray) -> Result<(NdArray, NdArray, NdArray)> {
        let m = to_nalgebra(&a.data)?;
        let decomp = m.svd(true, true);

        let u_mat = decomp
            .u
            .ok_or_else(|| NumpyError::ValueError("SVD computation failed (U)".into()))?;
        let u = from_nalgebra(&u_mat);

        let k = decomp.singular_values.len();
        let s = ArrayD::from_shape_fn(IxDyn(&[k]), |idx| decomp.singular_values[idx[0]]);

        let v_t = decomp
            .v_t
            .ok_or_else(|| NumpyError::ValueError("SVD computation failed (Vt)".into()))?;

        Ok((
            u,
            NdArray::from_data(ArrayData::Float64(s)),
            from_nalgebra(&v_t),
        ))
    }

    /// QR decomposition. Returns (Q, R).
    pub fn qr(a: &NdArray) -> Result<(NdArray, NdArray)> {
        let m = to_nalgebra(&a.data)?;
        let decomp = m.qr();
        let q = decomp.q();
        let r = decomp.r();
        Ok((from_nalgebra(&q), from_nalgebra(&r)))
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
        let m = to_nalgebra(&a.data)?;
        check_square(&m)?;
        let chol = nalgebra::linalg::Cholesky::new(m)
            .ok_or_else(|| NumpyError::ValueError("matrix is not positive definite".into()))?;
        let l = chol.l();
        Ok(from_nalgebra(&l))
    }

    fn check_square(m: &DMatrix<f64>) -> Result<()> {
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
        // nalgebra returns thin SVD: U is (m x min(m,n)), Vt is (min(m,n) x n)
        assert_eq!(u.shape()[0], 3);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape()[1], 2);
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
