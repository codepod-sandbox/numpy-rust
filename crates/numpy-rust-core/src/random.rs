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

        pub fn exponential(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray> {
            if scale < 0.0 {
                return Err(NumpyError::ValueError(
                    "scale < 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| -scale * (1.0 - self.random_scalar()).ln())
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn weibull(&mut self, a: f64, shape: &[usize]) -> Result<NdArray> {
            if a <= 0.0 {
                return Err(NumpyError::ValueError(
                    "a <= 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| (-(1.0 - self.random_scalar()).ln()).powf(1.0 / a))
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn rayleigh(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray> {
            if scale < 0.0 {
                return Err(NumpyError::ValueError(
                    "scale < 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| scale * (-2.0 * (1.0 - self.random_scalar()).ln()).sqrt())
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn power(&mut self, a: f64, shape: &[usize]) -> Result<NdArray> {
            if a <= 0.0 {
                return Err(NumpyError::ValueError(
                    "a <= 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| self.random_scalar().powf(1.0 / a))
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn pareto(&mut self, a: f64, shape: &[usize]) -> Result<NdArray> {
            if a <= 0.0 {
                return Err(NumpyError::ValueError(
                    "a <= 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| (1.0 - self.random_scalar()).powf(-1.0 / a) - 1.0)
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn laplace(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
            if scale < 0.0 {
                return Err(NumpyError::ValueError(
                    "scale < 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| {
                    let shifted = self.random_scalar() - 0.5;
                    if shifted == 0.0 {
                        loc
                    } else {
                        let sign = if shifted > 0.0 { 1.0 } else { -1.0 };
                        loc - scale * sign * (1.0 - 2.0 * shifted.abs()).ln()
                    }
                })
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn logistic(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
            if scale < 0.0 {
                return Err(NumpyError::ValueError(
                    "scale < 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| {
                    let u = self.random_scalar().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
                    loc + scale * (u / (1.0 - u)).ln()
                })
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn gumbel(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
            if scale < 0.0 {
                return Err(NumpyError::ValueError(
                    "scale < 0".into(),
                ));
            }
            let size: usize = shape.iter().product();
            let data: Vec<f64> = (0..size)
                .map(|_| {
                    let u = self.random_scalar().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
                    loc - scale * (-(u.ln())).ln()
                })
                .collect();
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        fn gamma_scalar(&mut self, shape: f64, scale: f64) -> Result<f64> {
            if shape <= 0.0 {
                return Err(NumpyError::ValueError("shape <= 0".into()));
            }
            if scale < 0.0 {
                return Err(NumpyError::ValueError("scale < 0".into()));
            }

            let (alpha, boost) = if shape < 1.0 {
                let u = self.random_scalar().clamp(f64::MIN_POSITIVE, 1.0);
                (shape + 1.0, u.powf(1.0 / shape))
            } else {
                (shape, 1.0)
            };

            let d = alpha - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                let x = self.standard_normal_scalar();
                let v = (1.0 + c * x).powi(3);
                if v <= 0.0 {
                    continue;
                }
                let u = self.random_scalar();
                if u < 1.0 - 0.0331 * x.powi(4) {
                    return Ok(d * v * scale * boost);
                }
                if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                    return Ok(d * v * scale * boost);
                }
            }
        }

        pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                data.push(self.gamma_scalar(shape_param, scale)?);
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn beta(&mut self, a: f64, b: f64, shape: &[usize]) -> Result<NdArray> {
            if a <= 0.0 {
                return Err(NumpyError::ValueError("a <= 0".into()));
            }
            if b <= 0.0 {
                return Err(NumpyError::ValueError("b <= 0".into()));
            }
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                let x = self.gamma_scalar(a, 1.0)?;
                let y = self.gamma_scalar(b, 1.0)?;
                data.push(x / (x + y));
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn dirichlet(&mut self, alpha: &[f64], sample_shape: &[usize]) -> Result<NdArray> {
            if alpha.is_empty() {
                return Err(NumpyError::ValueError("alpha must be non-empty".into()));
            }
            for &a in alpha {
                if a <= 0.0 {
                    return Err(NumpyError::ValueError("alpha <= 0".into()));
                }
            }

            let samples: usize = if sample_shape.is_empty() {
                1
            } else {
                sample_shape.iter().product()
            };
            let k = alpha.len();
            let mut data = Vec::with_capacity(samples * k);
            for _ in 0..samples {
                let mut draws = Vec::with_capacity(k);
                let mut total = 0.0;
                for &a in alpha {
                    let g = self.gamma_scalar(a, 1.0)?;
                    total += g;
                    draws.push(g);
                }
                for g in draws {
                    data.push(g / total);
                }
            }

            let mut shape = sample_shape.to_vec();
            shape.push(k);
            let arr = ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        fn poisson_scalar(&mut self, lam: f64) -> Result<f64> {
            if lam < 0.0 {
                return Err(NumpyError::ValueError("lam < 0".into()));
            }
            let l = (-lam).exp();
            let mut k = 0usize;
            let mut p = 1.0f64;
            loop {
                k += 1;
                p *= self.random_scalar();
                if p <= l {
                    return Ok((k - 1) as f64);
                }
            }
        }

        pub fn poisson(&mut self, lam: f64, shape: &[usize]) -> Result<NdArray> {
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                data.push(self.poisson_scalar(lam)?);
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        fn binomial_scalar(&mut self, n: i64, p: f64) -> Result<f64> {
            if n < 0 {
                return Err(NumpyError::ValueError("n < 0".into()));
            }
            if !(0.0..=1.0).contains(&p) {
                return Err(NumpyError::ValueError("p < 0, p > 1 or p is NaN".into()));
            }
            let mut successes = 0i64;
            for _ in 0..n {
                if self.random_scalar() < p {
                    successes += 1;
                }
            }
            Ok(successes as f64)
        }

        pub fn binomial(&mut self, n: i64, p: f64, shape: &[usize]) -> Result<NdArray> {
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                data.push(self.binomial_scalar(n, p)?);
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        fn geometric_scalar(&mut self, p: f64) -> Result<f64> {
            if !(0.0 < p && p <= 1.0) {
                return Err(NumpyError::ValueError("p <= 0, p > 1 or p is NaN".into()));
            }
            let u = self.random_scalar().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
            Ok((u.ln() / (1.0 - p).ln()).ceil())
        }

        pub fn geometric(&mut self, p: f64, shape: &[usize]) -> Result<NdArray> {
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                data.push(self.geometric_scalar(p)?);
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn multinomial(
            &mut self,
            n: i64,
            pvals: &[f64],
            sample_shape: &[usize],
        ) -> Result<NdArray> {
            if n < 0 {
                return Err(NumpyError::ValueError("n < 0".into()));
            }
            if pvals.is_empty() {
                return Err(NumpyError::ValueError("pvals must not be empty".into()));
            }
            if pvals.iter().any(|&p| p < 0.0 || p.is_nan()) {
                return Err(NumpyError::ValueError("pvals < 0, pvals > 1 or pvals contains NaNs".into()));
            }

            let samples: usize = if sample_shape.is_empty() {
                1
            } else {
                sample_shape.iter().product()
            };
            let k = pvals.len();
            let mut out = Vec::with_capacity(samples * k);

            for _ in 0..samples {
                let mut counts = vec![0_i64; k];
                for _ in 0..n {
                    let r = self.random_scalar();
                    let mut cumsum = 0.0;
                    let mut placed = false;
                    for (j, p) in pvals.iter().enumerate() {
                        cumsum += *p;
                        if r < cumsum {
                            counts[j] += 1;
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        counts[k - 1] += 1;
                    }
                }
                out.extend(counts);
            }

            let mut shape = sample_shape.to_vec();
            shape.push(k);
            let arr = ArrayD::from_shape_vec(IxDyn(&shape), out).unwrap();
            Ok(NdArray::from_data(ArrayData::Int64(arr)))
        }

        pub fn negative_binomial(&mut self, n: i64, p: f64, shape: &[usize]) -> Result<NdArray> {
            if n < 0 {
                return Err(NumpyError::ValueError("n < 0".into()));
            }
            if !(0.0..=1.0).contains(&p) {
                return Err(NumpyError::ValueError("p < 0, p > 1 or p is NaN".into()));
            }
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                let mut failures = 0_i64;
                let mut successes = 0_i64;
                while successes < n {
                    if self.random_scalar() < p {
                        successes += 1;
                    } else {
                        failures += 1;
                    }
                }
                data.push(failures as f64);
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn hypergeometric(
            &mut self,
            ngood: i64,
            nbad: i64,
            nsample: i64,
            shape: &[usize],
        ) -> Result<NdArray> {
            if ngood < 0 || nbad < 0 || nsample < 0 {
                return Err(NumpyError::ValueError("parameters must be non-negative".into()));
            }
            if nsample > ngood + nbad {
                return Err(NumpyError::ValueError("nsample > ngood + nbad".into()));
            }
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                let mut count = 0_i64;
                let mut rg = ngood as f64;
                let mut rt = (ngood + nbad) as f64;
                for _ in 0..nsample {
                    if self.random_scalar() < (rg / rt) {
                        count += 1;
                        rg -= 1.0;
                    }
                    rt -= 1.0;
                }
                data.push(count as f64);
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn triangular(
            &mut self,
            left: f64,
            mode: f64,
            right: f64,
            shape: &[usize],
        ) -> Result<NdArray> {
            if !(left <= mode && mode <= right) || left == right {
                return Err(NumpyError::ValueError(
                    "left <= mode <= right and left < right required".into(),
                ));
            }
            let fc = (mode - left) / (right - left);
            let size: usize = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                let u = self.random_scalar();
                if u < fc {
                    data.push(left + ((right - left) * (mode - left) * u).sqrt());
                } else {
                    data.push(right - ((right - left) * (right - mode) * (1.0 - u)).sqrt());
                }
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn multivariate_normal_from_cholesky(
            &mut self,
            mean: &[f64],
            chol: &[f64],
            sample_shape: &[usize],
        ) -> Result<NdArray> {
            let n = mean.len();
            if n == 0 {
                return Err(NumpyError::ValueError("mean must be non-empty".into()));
            }
            if chol.len() != n * n {
                return Err(NumpyError::ValueError(
                    "cholesky factor has incompatible shape".into(),
                ));
            }
            let samples: usize = if sample_shape.is_empty() {
                1
            } else {
                sample_shape.iter().product()
            };
            let mut out = Vec::with_capacity(samples * n);
            let mut z = vec![0.0; n];
            for _ in 0..samples {
                for zi in &mut z {
                    *zi = self.standard_normal_scalar();
                }
                for i in 0..n {
                    let mut val = mean[i];
                    let row = &chol[i * n..(i + 1) * n];
                    for j in 0..n {
                        val += row[j] * z[j];
                    }
                    out.push(val);
                }
            }
            let mut shape = sample_shape.to_vec();
            shape.push(n);
            let arr = ArrayD::from_shape_vec(IxDyn(&shape), out).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn vonmises(&mut self, mu: f64, kappa: f64, shape: &[usize]) -> Result<NdArray> {
            if kappa < 0.0 {
                return Err(NumpyError::ValueError("kappa < 0".into()));
            }
            let size: usize = shape.iter().product();
            let mut out = Vec::with_capacity(size);
            if kappa < 1e-6 {
                for _ in 0..size {
                    out.push(-std::f64::consts::PI + 2.0 * std::f64::consts::PI * self.random_scalar());
                }
            } else {
                let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
                let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
                let r = (1.0 + rho * rho) / (2.0 * rho);
                while out.len() < size {
                    let u1 = self.random_scalar();
                    let z = (std::f64::consts::PI * u1).cos();
                    let f = (1.0 + r * z) / (r + z);
                    let c = kappa * (r - f);
                    let u2 = self.random_scalar();
                    if u2 < c * (2.0 - c) || u2 <= c * (1.0 - c).exp() {
                        let u3 = self.random_scalar();
                        let theta = mu
                            + if u3 > 0.5 { 1.0 } else { -1.0 } * f.acos();
                        out.push(theta);
                    }
                }
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), out).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn wald(&mut self, mean: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
            if mean <= 0.0 || scale <= 0.0 {
                return Err(NumpyError::ValueError("mean <= 0 or scale <= 0".into()));
            }
            let size: usize = shape.iter().product();
            let mut out = Vec::with_capacity(size);
            for _ in 0..size {
                let v = self.standard_normal_scalar();
                let y = v * v;
                let x = mean
                    + (mean * mean * y) / (2.0 * scale)
                    - (mean / (2.0 * scale))
                        * (4.0 * mean * scale * y + mean * mean * y * y).sqrt();
                let u = self.random_scalar();
                if u <= mean / (mean + x) {
                    out.push(x);
                } else {
                    out.push(mean * mean / x);
                }
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), out).unwrap();
            Ok(NdArray::from_data(ArrayData::Float64(arr)))
        }

        pub fn zipf(&mut self, a: f64, shape: &[usize]) -> Result<NdArray> {
            if a <= 1.0 {
                return Err(NumpyError::ValueError("a <= 1".into()));
            }
            let size: usize = shape.iter().product();
            let am1 = a - 1.0;
            let b = 2.0_f64.powf(am1);
            let mut out = Vec::with_capacity(size);
            while out.len() < size {
                let u = self.random_scalar();
                if u <= 0.0 {
                    continue;
                }
                let v = self.random_scalar();
                let mut x = (u.powf(-1.0 / am1)) as i64;
                if x < 1 {
                    x = 1;
                }
                let xf = x as f64;
                let t = (1.0 + 1.0 / xf).powf(am1);
                if v * xf * (t - 1.0) / (b - 1.0) <= t / b {
                    out.push(xf);
                }
            }
            let arr = ArrayD::from_shape_vec(IxDyn(shape), out).unwrap();
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

    /// Generate an array of exponential-distributed values.
    pub fn exponential(scale: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.exponential(scale, shape))
    }

    pub fn weibull(a: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.weibull(a, shape))
    }

    pub fn rayleigh(scale: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.rayleigh(scale, shape))
    }

    pub fn power(a: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.power(a, shape))
    }

    pub fn pareto(a: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.pareto(a, shape))
    }

    pub fn laplace(loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.laplace(loc, scale, shape))
    }

    pub fn logistic(loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.logistic(loc, scale, shape))
    }

    pub fn gumbel(loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.gumbel(loc, scale, shape))
    }

    pub fn gamma(shape_param: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.gamma(shape_param, scale, shape))
    }

    pub fn beta(a: f64, b: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.beta(a, b, shape))
    }

    pub fn dirichlet(alpha: &[f64], sample_shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.dirichlet(alpha, sample_shape))
    }

    pub fn poisson(lam: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.poisson(lam, shape))
    }

    pub fn binomial(n: i64, p: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.binomial(n, p, shape))
    }

    pub fn geometric(p: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.geometric(p, shape))
    }

    pub fn multinomial(n: i64, pvals: &[f64], sample_shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.multinomial(n, pvals, sample_shape))
    }

    pub fn negative_binomial(n: i64, p: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.negative_binomial(n, p, shape))
    }

    pub fn hypergeometric(
        ngood: i64,
        nbad: i64,
        nsample: i64,
        shape: &[usize],
    ) -> Result<NdArray> {
        with_rng(|rng| rng.hypergeometric(ngood, nbad, nsample, shape))
    }

    pub fn triangular(left: f64, mode: f64, right: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.triangular(left, mode, right, shape))
    }

    pub fn multivariate_normal_from_cholesky(
        mean: &[f64],
        chol: &[f64],
        sample_shape: &[usize],
    ) -> Result<NdArray> {
        with_rng(|rng| rng.multivariate_normal_from_cholesky(mean, chol, sample_shape))
    }

    pub fn vonmises(mu: f64, kappa: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.vonmises(mu, kappa, shape))
    }

    pub fn wald(mean: f64, scale: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.wald(mean, scale, shape))
    }

    pub fn zipf(a: f64, shape: &[usize]) -> Result<NdArray> {
        with_rng(|rng| rng.zipf(a, shape))
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
        let vals_a: Vec<f64> = (0..5)
            .map(|_| guard.as_mut().unwrap().random_scalar())
            .collect();
        *guard = Some(StatefulRng::new(Some(99)));
        let vals_b: Vec<f64> = (0..5)
            .map(|_| guard.as_mut().unwrap().random_scalar())
            .collect();
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
