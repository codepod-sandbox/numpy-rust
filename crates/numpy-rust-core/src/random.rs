//! Random number generation backed by `rand` and `rand_distr`.
//! Only available with the `random` feature.

#[cfg(feature = "random")]
mod inner {
    use crate::array_data::ArrayD;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    use ndarray::IxDyn;

    use crate::array_data::ArrayData;
    use crate::error::{NumpyError, Result};
    use crate::NdArray;

    const SPLITMIX64_GAMMA: u64 = 0x9E3779B97F4A7C15;
    static ENTROPY_COUNTER: AtomicU64 = AtomicU64::new(1);

    #[derive(Clone, Debug)]
    pub struct StatefulRng {
        state: u64,
    }

    impl StatefulRng {
        pub fn new(seed: Option<u64>) -> Self {
            Self {
                state: seed.unwrap_or_else(entropy_seed),
            }
        }

        pub fn from_state(state: u64) -> Self {
            Self { state }
        }

        pub fn state(&self) -> u64 {
            self.state
        }

        pub fn set_state(&mut self, state: u64) {
            self.state = state;
        }

        pub fn advance(&mut self, delta: u64) {
            self.state = mix64(self.state.wrapping_add(delta.rotate_left(17)));
        }

        pub fn jumped(&self) -> Self {
            let mut jumped = self.clone();
            jumped.advance(1);
            jumped
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
            mix64(self.state)
        }

        pub fn randbelow(&mut self, upper: u64) -> Result<u64> {
            if upper == 0 {
                return Err(NumpyError::ValueError(
                    "upper bound must be positive".into(),
                ));
            }
            let zone = u64::MAX - (u64::MAX % upper);
            loop {
                let candidate = self.next_u64();
                if candidate < zone {
                    return Ok(candidate % upper);
                }
            }
        }

        pub fn random_scalar(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
        }

        fn standard_normal_scalar(&mut self) -> f64 {
            let u1 = (1.0 - self.random_scalar()).max(f64::MIN_POSITIVE);
            let u2 = self.random_scalar();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }

        pub fn rand(&mut self, shape: &[usize]) -> NdArray {
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size).map(|_| self.random_scalar()).collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            NdArray::from_data(ArrayData::Float64(arr))
        }

        pub fn randn(&mut self, shape: &[usize]) -> NdArray {
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size).map(|_| self.standard_normal_scalar()).collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            NdArray::from_data(ArrayData::Float64(arr))
        }

        pub fn randint_scalar(&mut self, low: i64, high: i64) -> Result<i64> {
            if low >= high {
                return Err(NumpyError::ValueError(format!(
                    "low ({low}) must be less than high ({high})"
                )));
            }
            let span = (high as i128 - low as i128) as u128;
            let offset = self.randbelow(span as u64)? as i128;
            Ok((low as i128 + offset) as i64)
        }

        pub fn randint(&mut self, low: i64, high: i64, shape: &[usize]) -> Result<NdArray> {
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                data.push(self.randint_scalar(low, high)?);
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Int64(arr)))
        }

        pub fn normal(&mut self, mean: f64, std_dev: f64, shape: &[usize]) -> Result<NdArray> {
            if std_dev < 0.0 {
                return Err(NumpyError::ValueError(
                    "invalid normal distribution parameters".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| mean + std_dev * self.standard_normal_scalar())
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Result<NdArray> {
            if low >= high {
                return Err(NumpyError::ValueError(format!(
                    "low ({low}) must be less than high ({high})"
                )));
            }
            let size: usize = shape.iter().product();
            let scale = high - low;
            let data: Vec<f64> = (0..size)
                .map(|_| low + scale * self.random_scalar())
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn choice(&mut self, a: &NdArray, size: usize, replace: bool) -> Result<NdArray> {
            if a.ndim() != 1 {
                return Err(NumpyError::ValueError("choice requires a 1-D array".into()));
            }
            let n = a.size();
            if n == 0 {
                return Err(NumpyError::ValueError(
                    "cannot choose from empty array".into(),
                ));
            }
            if !replace && size > n {
                return Err(NumpyError::ValueError(format!(
                    "cannot take a sample of size {size} from array of size {n} without replacement"
                )));
            }
            let indices: Vec<usize> = if replace {
                let mut out = Vec::with_capacity(size);
                for _ in 0..size {
                    out.push(self.randbelow(n as u64)? as usize);
                }
                out
            } else {
                let mut pool: Vec<usize> = (0..n).collect();
                for i in 0..size {
                    let j = i + self.randbelow((n - i) as u64)? as usize;
                    pool.swap(i, j);
                }
                pool.into_iter().take(size).collect()
            };
            a.index_select(0, &indices)
        }

        pub fn randbits(&mut self, bits: usize) -> u64 {
            if bits == 0 {
                return 0;
            }
            if bits >= 64 {
                return self.next_u64();
            }
            self.next_u64() >> (64 - bits)
        }
    }

    fn entropy_seed() -> u64 {
        let counter = ENTROPY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        mix64(now ^ counter.rotate_left(23))
    }

    fn mix64(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(crate) static GLOBAL_RNG: Mutex<Option<StatefulRng>> = Mutex::new(None);

    /// Get or initialize the global RNG.
    fn with_rng<F, R>(f: F) -> R
    where
        F: FnOnce(&mut StatefulRng) -> R,
    {
        let mut guard = GLOBAL_RNG.lock().unwrap();
        if guard.is_none() {
            *guard = Some(StatefulRng::new(None));
        }
        f(guard.as_mut().unwrap())
    }

    /// Set the global RNG seed for reproducibility.
    pub fn seed(n: u64) {
        let mut guard = GLOBAL_RNG.lock().unwrap();
        *guard = Some(StatefulRng::new(Some(n)));
    }

    /// Generate an array of uniform random values in [0, 1).
    /// Like `numpy.random.rand(*shape)`.
    pub fn rand(shape: &[usize]) -> NdArray {
        with_rng(|rng| rng.rand(shape))
    }

    /// Generate an array of standard normal (mean=0, std=1) random values.
    /// Like `numpy.random.randn(*shape)`.
    pub fn randn(shape: &[usize]) -> NdArray {
        with_rng(|rng| rng.randn(shape))
    }

    /// Generate an array of random integers in [low, high).
    /// Like `numpy.random.randint(low, high, size)`.
    pub fn randint(low: i64, high: i64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.randint(low, high, shape))
    }

    /// Generate an array of normal-distributed values.
    /// Like `numpy.random.normal(loc, scale, size)`.
    pub fn normal(mean: f64, std_dev: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.normal(mean, std_dev, shape))
    }

    /// Generate an array of uniform-distributed values in [low, high).
    /// Like `numpy.random.uniform(low, high, size)`.
    pub fn uniform(low: f64, high: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.uniform(low, high, shape))
    }

    /// Choose random elements from a 1-D array.
    /// Like `numpy.random.choice(a, size, replace)`.
    pub fn choice(a: &NdArray, size: usize, replace: bool) -> Result<NdArray> {
        with_rng(|rng| rng.choice(a, size, replace))
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
        use super::inner::GLOBAL_RNG;

        let mut guard = GLOBAL_RNG.lock().unwrap();
        *guard = Some(StatefulRng::new(Some(99)));
        let vals_a: Vec<f64> = (0..5).map(|_| guard.as_mut().unwrap().random_scalar()).collect();
        *guard = Some(StatefulRng::new(Some(99)));
        let vals_b: Vec<f64> = (0..5).map(|_| guard.as_mut().unwrap().random_scalar()).collect();
        drop(guard);

        assert_eq!(vals_a, vals_b);
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
