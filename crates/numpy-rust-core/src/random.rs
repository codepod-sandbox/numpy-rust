//! Random number generation backed by `rand` and `rand_distr`.
//! Only available with the `random` feature.

#[cfg(feature = "random")]
mod inner {
    use std::sync::Mutex;

    use ndarray::{ArrayD, IxDyn};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, Normal, Uniform};

    use crate::array_data::ArrayData;
    use crate::error::{NumpyError, Result};
    use crate::NdArray;

    static GLOBAL_RNG: Mutex<Option<StdRng>> = Mutex::new(None);

    /// Get or initialize the global RNG.
    fn with_rng<F, R>(f: F) -> R
    where
        F: FnOnce(&mut StdRng) -> R,
    {
        let mut guard = GLOBAL_RNG.lock().unwrap();
        if guard.is_none() {
            *guard = Some(StdRng::from_os_rng());
        }
        f(guard.as_mut().unwrap())
    }

    /// Set the global RNG seed for reproducibility.
    pub fn seed(n: u64) {
        let mut guard = GLOBAL_RNG.lock().unwrap();
        *guard = Some(StdRng::seed_from_u64(n));
    }

    /// Generate an array of uniform random values in [0, 1).
    /// Like `numpy.random.rand(*shape)`.
    pub fn rand(shape: &[usize]) -> NdArray {
        let size: usize = shape.iter().product();
        let data: Vec<f64> = with_rng(|rng| (0..size).map(|_| rng.random::<f64>()).collect());
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        NdArray::from_data(ArrayData::Float64(arr))
    }

    /// Generate an array of standard normal (mean=0, std=1) random values.
    /// Like `numpy.random.randn(*shape)`.
    pub fn randn(shape: &[usize]) -> NdArray {
        let size: usize = shape.iter().product();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = with_rng(|rng| (0..size).map(|_| dist.sample(rng)).collect());
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        NdArray::from_data(ArrayData::Float64(arr))
    }

    /// Generate an array of random integers in [low, high).
    /// Like `numpy.random.randint(low, high, size)`.
    pub fn randint(low: i64, high: i64, shape: &[usize]) -> Result<NdArray> {
        if low >= high {
            return Err(NumpyError::ValueError(format!(
                "low ({low}) must be less than high ({high})"
            )));
        }
        let size: usize = shape.iter().product();
        let dist = Uniform::new(low, high).unwrap();
        let data: Vec<i64> = with_rng(|rng| (0..size).map(|_| dist.sample(rng)).collect());
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        Ok(NdArray::from_data(ArrayData::Int64(arr)))
    }

    /// Generate an array of normal-distributed values.
    /// Like `numpy.random.normal(loc, scale, size)`.
    pub fn normal(mean: f64, std_dev: f64, shape: &[usize]) -> Result<NdArray> {
        let dist = Normal::new(mean, std_dev)
            .map_err(|_| NumpyError::ValueError("invalid normal distribution parameters".into()))?;
        let size: usize = shape.iter().product();
        let data: Vec<f64> = with_rng(|rng| (0..size).map(|_| dist.sample(rng)).collect());
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        Ok(NdArray::from_data(ArrayData::Float64(arr)))
    }

    /// Generate an array of uniform-distributed values in [low, high).
    /// Like `numpy.random.uniform(low, high, size)`.
    pub fn uniform(low: f64, high: f64, shape: &[usize]) -> Result<NdArray> {
        if low >= high {
            return Err(NumpyError::ValueError(format!(
                "low ({low}) must be less than high ({high})"
            )));
        }
        let dist = Uniform::new(low, high).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = with_rng(|rng| (0..size).map(|_| dist.sample(rng)).collect());
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        Ok(NdArray::from_data(ArrayData::Float64(arr)))
    }

    /// Choose random elements from a 1-D array.
    /// Like `numpy.random.choice(a, size, replace)`.
    pub fn choice(a: &NdArray, size: usize, replace: bool) -> Result<NdArray> {
        if a.ndim() != 1 {
            return Err(NumpyError::ValueError(
                "choice requires a 1-D array".into(),
            ));
        }
        let n = a.size();
        if n == 0 {
            return Err(NumpyError::ValueError("cannot choose from empty array".into()));
        }
        if !replace && size > n {
            return Err(NumpyError::ValueError(format!(
                "cannot take a sample of size {size} from array of size {n} without replacement"
            )));
        }

        let indices: Vec<usize> = if replace {
            with_rng(|rng| (0..size).map(|_| rng.random_range(0..n)).collect())
        } else {
            // Fisher-Yates shuffle on indices, take first `size`
            let mut pool: Vec<usize> = (0..n).collect();
            with_rng(|rng| {
                for i in 0..size {
                    let j = rng.random_range(i..n);
                    pool.swap(i, j);
                }
            });
            pool.into_iter().take(size).collect()
        };

        a.index_select(0, &indices)
    }
}

#[cfg(feature = "random")]
pub use inner::*;

#[cfg(test)]
#[cfg(feature = "random")]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_seed_deterministic() {
        seed(42);
        let a = rand(&[5]);
        seed(42);
        let b = rand(&[5]);
        // Same seed should produce same values
        for i in 0..5 {
            let va = match a.get(&[i]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            let vb = match b.get(&[i]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            assert_eq!(va, vb);
        }
    }

    #[test]
    fn test_rand_shape() {
        let a = rand(&[3, 4]);
        assert_eq!(a.shape(), &[3, 4]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_rand_range() {
        seed(123);
        let a = rand(&[100]);
        // All values should be in [0, 1)
        for i in 0..100 {
            let v = match a.get(&[i]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_randn_shape() {
        let a = randn(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.dtype(), DType::Float64);
    }

    #[test]
    fn test_randint() {
        seed(42);
        let a = randint(0, 10, &[20]).unwrap();
        assert_eq!(a.shape(), &[20]);
        assert_eq!(a.dtype(), DType::Int64);
        // All values in [0, 10)
        for i in 0..20 {
            let v = match a.get(&[i]).unwrap() {
                crate::indexing::Scalar::Int64(v) => v,
                _ => panic!(),
            };
            assert!(v >= 0 && v < 10);
        }
    }

    #[test]
    fn test_randint_invalid_range() {
        assert!(randint(10, 5, &[1]).is_err());
    }

    #[test]
    fn test_normal() {
        seed(42);
        let a = normal(5.0, 0.001, &[100]).unwrap();
        assert_eq!(a.shape(), &[100]);
        // With very small std, all values should be close to mean
        for i in 0..100 {
            let v = match a.get(&[i]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            assert!((v - 5.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_uniform() {
        seed(42);
        let a = uniform(2.0, 5.0, &[100]).unwrap();
        assert_eq!(a.shape(), &[100]);
        for i in 0..100 {
            let v = match a.get(&[i]).unwrap() {
                crate::indexing::Scalar::Float64(v) => v,
                _ => panic!(),
            };
            assert!(v >= 2.0 && v < 5.0);
        }
    }

    #[test]
    fn test_uniform_invalid_range() {
        assert!(uniform(5.0, 2.0, &[1]).is_err());
    }

    #[test]
    fn test_choice_with_replacement() {
        seed(42);
        let a = crate::NdArray::from_vec(vec![10.0_f64, 20.0, 30.0]);
        let b = choice(&a, 5, true).unwrap();
        assert_eq!(b.shape(), &[5]);
    }

    #[test]
    fn test_choice_without_replacement() {
        seed(42);
        let a = crate::NdArray::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
        let b = choice(&a, 3, false).unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_choice_too_large_no_replace() {
        let a = crate::NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(choice(&a, 5, false).is_err());
    }

    #[test]
    fn test_choice_non_1d_error() {
        let a = crate::NdArray::zeros(&[2, 3], DType::Float64);
        assert!(choice(&a, 2, true).is_err());
    }
}
