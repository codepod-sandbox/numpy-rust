//! FFT operations backed by RustFFT.
//! Only available with the `fft` feature.

#[cfg(feature = "fft")]
mod inner {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    use ndarray::{ArrayD, IxDyn};

    use crate::array_data::ArrayData;
    use crate::casting::cast_array_data;
    use crate::dtype::DType;
    use crate::error::{NumpyError, Result};
    use crate::NdArray;

    /// Convert a 1-D NdArray to Vec<Complex<f64>>.
    fn to_complex(data: &ArrayData) -> Result<Vec<Complex<f64>>> {
        let f64_data = cast_array_data(data, DType::Float64);
        match f64_data {
            ArrayData::Float64(a) => {
                if a.ndim() != 1 {
                    return Err(NumpyError::ValueError(format!(
                        "expected 1-D array, got {}-D",
                        a.ndim()
                    )));
                }
                Ok(a.iter().map(|&x| Complex { re: x, im: 0.0 }).collect())
            }
            _ => unreachable!(),
        }
    }

    /// Convert Vec<Complex<f64>> to a 2-D NdArray with shape [n, 2] (real, imag columns).
    fn from_complex(data: &[Complex<f64>]) -> NdArray {
        let n = data.len();
        let flat: Vec<f64> = data.iter().flat_map(|c| [c.re, c.im]).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, 2]), flat).unwrap();
        NdArray::from_data(ArrayData::Float64(arr))
    }

    /// 1-D discrete Fourier Transform.
    /// Input: real-valued 1-D array.
    /// Output: complex array as shape [n, 2] where columns are (real, imag).
    pub fn fft(a: &NdArray) -> Result<NdArray> {
        let mut buffer = to_complex(&a.data)?;
        let n = buffer.len();
        if n == 0 {
            return Ok(from_complex(&buffer));
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);
        Ok(from_complex(&buffer))
    }

    /// 1-D inverse discrete Fourier Transform.
    /// Input: complex array as shape [n, 2].
    /// Output: complex array as shape [n, 2] (normalized by 1/n).
    pub fn ifft(a: &NdArray) -> Result<NdArray> {
        let mut buffer = complex_from_array(a)?;
        let n = buffer.len();
        if n == 0 {
            return Ok(from_complex(&buffer));
        }
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);
        ifft.process(&mut buffer);
        // Normalize by 1/n
        let inv_n = 1.0 / n as f64;
        for c in &mut buffer {
            c.re *= inv_n;
            c.im *= inv_n;
        }
        Ok(from_complex(&buffer))
    }

    /// Real-input FFT. Returns only the positive-frequency half: n/2+1 complex values.
    /// Input: real-valued 1-D array of length n.
    /// Output: complex array as shape [n/2+1, 2].
    pub fn rfft(a: &NdArray) -> Result<NdArray> {
        let mut buffer = to_complex(&a.data)?;
        let n = buffer.len();
        if n == 0 {
            return Ok(from_complex(&buffer));
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);
        // Keep only positive frequencies: indices 0..=n/2
        let half = n / 2 + 1;
        buffer.truncate(half);
        Ok(from_complex(&buffer))
    }

    /// Inverse of rfft. Takes n/2+1 complex values and returns a real 1-D array of length n.
    /// `n` is the output length (needed because rfft output is ambiguous for even/odd).
    pub fn irfft(a: &NdArray, n: usize) -> Result<NdArray> {
        let half_complex = complex_from_array(a)?;
        let half = half_complex.len();
        if half == 0 || n == 0 {
            return Ok(NdArray::from_data(ArrayData::Float64(
                ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
            )));
        }

        // Reconstruct full spectrum using conjugate symmetry
        let mut buffer = Vec::with_capacity(n);
        for i in 0..n {
            if i < half {
                buffer.push(half_complex[i]);
            } else {
                // Conjugate symmetry: X[k] = conj(X[n-k])
                let mirror = n - i;
                if mirror < half {
                    buffer.push(Complex {
                        re: half_complex[mirror].re,
                        im: -half_complex[mirror].im,
                    });
                } else {
                    buffer.push(Complex { re: 0.0, im: 0.0 });
                }
            }
        }

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);
        ifft.process(&mut buffer);

        // Normalize and take real part
        let inv_n = 1.0 / n as f64;
        let real_data: Vec<f64> = buffer.iter().map(|c| c.re * inv_n).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), real_data).unwrap();
        Ok(NdArray::from_data(ArrayData::Float64(arr)))
    }

    /// Helper: parse a [n, 2] complex array back into Vec<Complex<f64>>.
    fn complex_from_array(a: &NdArray) -> Result<Vec<Complex<f64>>> {
        let f64_data = cast_array_data(&a.data, DType::Float64);
        match f64_data {
            ArrayData::Float64(arr) => {
                if arr.ndim() != 2 || arr.shape()[1] != 2 {
                    return Err(NumpyError::ValueError(
                        "expected complex array with shape [n, 2]".into(),
                    ));
                }
                let n = arr.shape()[0];
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    result.push(Complex {
                        re: arr[[i, 0]],
                        im: arr[[i, 1]],
                    });
                }
                Ok(result)
            }
            _ => unreachable!(),
        }
    }

    /// FFT frequency bins for a signal of length n with sample spacing d.
    /// Like numpy.fft.fftfreq.
    pub fn fftfreq(n: usize, d: f64) -> NdArray {
        let mut freqs = Vec::with_capacity(n);
        let val = 1.0 / (n as f64 * d);
        for i in 0..n {
            let k = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            freqs.push(k * val);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), freqs).unwrap();
        NdArray::from_data(ArrayData::Float64(arr))
    }
}

#[cfg(feature = "fft")]
pub use inner::*;

#[cfg(test)]
#[cfg(feature = "fft")]
mod tests {
    use super::*;
    use crate::{DType, NdArray};

    #[test]
    fn test_fft_basic() {
        let a = NdArray::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
        let result = fft(&a).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
    }

    #[test]
    fn test_fft_ones() {
        // FFT of [1,1,1,1] should be [4,0,0,0]
        let a = NdArray::from_vec(vec![1.0_f64, 1.0, 1.0, 1.0]);
        let result = fft(&a).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        // First element should be (4.0, 0.0)
        let val = result.get(&[0, 0]).unwrap();
        match val {
            crate::indexing::Scalar::Float64(v) => assert!((v - 4.0).abs() < 1e-10),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let freq = fft(&a).unwrap();
        let recovered = ifft(&freq).unwrap();
        assert_eq!(recovered.shape(), &[4, 2]);
        // Real parts should match original, imaginary should be ~0
        for i in 0..4 {
            let re = match recovered.get(&[i, 0]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            let im = match recovered.get(&[i, 1]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            assert!((re - (i as f64 + 1.0)).abs() < 1e-10);
            assert!(im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_rfft() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let result = rfft(&a).unwrap();
        // rfft of length 4 should return 3 complex values (4/2+1)
        assert_eq!(result.shape(), &[3, 2]);
    }

    #[test]
    fn test_irfft() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let freq = rfft(&a).unwrap();
        let recovered = irfft(&freq, 4).unwrap();
        assert_eq!(recovered.shape(), &[4]);
        // Should match original
        for i in 0..4 {
            let v = match recovered.get(&[i]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            assert!((v - (i as f64 + 1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fftfreq() {
        let freqs = fftfreq(4, 1.0);
        assert_eq!(freqs.shape(), &[4]);
        // Expected: [0.0, 0.25, -0.5, -0.25] for n=4, d=1.0
        // Actually: [0.0, 0.25, 0.5, -0.25] — index 2 = n/2 is positive
        let f0 = match freqs.get(&[0]).unwrap() {
            crate::indexing::Scalar::Float64(v) => v,
            _ => panic!(),
        };
        assert!((f0 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fft_non_1d_error() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        assert!(fft(&a).is_err());
    }

    #[test]
    fn test_ifft_bad_shape_error() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(ifft(&a).is_err());
    }
}
