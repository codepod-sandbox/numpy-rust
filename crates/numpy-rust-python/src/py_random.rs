use rustpython_vm as vm;
use vm::convert::ToPyObject;
use vm::{PyObjectRef, PyRef, PyResult, VirtualMachine};

use crate::py_array::{extract_shape, obj_to_ndarray, PyNdArray};
use numpy_rust_core::{ArrayData, DType};

fn err(e: numpy_rust_core::NumpyError, vm: &VirtualMachine) -> vm::builtins::PyBaseExceptionRef {
    vm.new_value_error(e.to_string())
}

fn parse_float64_vec(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<f64>> {
    let arr = obj_to_ndarray(obj, vm)?;
    let flat = arr.flatten().astype(DType::Float64);
    let ArrayData::Float64(data) = flat.data() else {
        return Err(vm.new_type_error("expected float-convertible array-like".to_owned()));
    };
    Ok(data.iter().copied().collect())
}

fn parse_float64_array(
    obj: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let arr = obj_to_ndarray(obj, vm)?;
    let shape = arr.shape().to_vec();
    let flat = arr.flatten().astype(DType::Float64);
    let ArrayData::Float64(data) = flat.data() else {
        return Err(vm.new_type_error("expected float-convertible array-like".to_owned()));
    };
    Ok((data.iter().copied().collect(), shape))
}

/// Convert variadic positional args into a shape.
/// Supports: rand(5, 3) -> (5, 3), rand((5, 3)) -> (5, 3), rand() -> (1,)
fn shape_from_varargs(args: &vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    if args.args.is_empty() {
        return Ok(vec![1]);
    }
    if args.args.len() == 1 {
        return extract_shape(&args.args[0], vm);
    }
    // Multiple positional args: treat each as a dimension
    let mut shape = Vec::with_capacity(args.args.len());
    for arg in &args.args {
        let n: i64 = arg.clone().try_into_value(vm)?;
        shape.push(n as usize);
    }
    Ok(shape)
}

#[vm::pymodule]
mod _random {
    use super::*;

    fn make_state_tuple(state: u64, value: PyObjectRef, vm: &VirtualMachine) -> PyObjectRef {
        vm.ctx
            .new_tuple(vec![vm.ctx.new_int(state).into(), value])
            .into()
    }

    #[pyfunction]
    fn seed(n: u64, _vm: &VirtualMachine) {
        numpy_rust_core::random::seed(n);
    }

    #[pyfunction]
    fn rand(args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = shape_from_varargs(&args, vm)?;
        Ok(PyNdArray::from_core(numpy_rust_core::random::rand(&sh)))
    }

    #[pyfunction]
    fn randn(args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = shape_from_varargs(&args, vm)?;
        Ok(PyNdArray::from_core(numpy_rust_core::random::randn(&sh)))
    }

    #[pyfunction]
    fn randint(args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        // Accept: randint(high) or randint(low, high) or randint(low, high, size)
        // Also accepts size= and dtype= as keyword args (dtype is ignored, always int64)
        let first = args
            .args
            .first()
            .ok_or_else(|| vm.new_type_error("randint() missing argument: 'low'".to_owned()))?
            .clone();
        let first_val: i64 = first.try_into_value(vm)?;

        let (low, high) = if let Some(second) = args.args.get(1) {
            let second_val: i64 = second.clone().try_into_value(vm)?;
            (first_val, second_val)
        } else if let Some(kw_high) = args.kwargs.get("high") {
            let high_val: i64 = kw_high.clone().try_into_value(vm)?;
            (first_val, high_val)
        } else {
            // Single arg: randint(high) means [0, high)
            (0i64, first_val)
        };

        // size can be positional (3rd arg) or keyword 'size'
        let size_obj = if let Some(pos) = args.args.get(2) {
            Some(pos.clone())
        } else {
            args.kwargs.get("size").cloned()
        };

        match size_obj {
            None => {
                // Scalar: return single value
                let sh = vec![1usize];
                numpy_rust_core::random::randint(low, high, &sh)
                    .map(PyNdArray::from_core)
                    .map_err(|e| err(e, vm))
            }
            Some(size_obj) => {
                let sh = extract_shape(&size_obj, vm)?;
                numpy_rust_core::random::randint(low, high, &sh)
                    .map(PyNdArray::from_core)
                    .map_err(|e| err(e, vm))
            }
        }
    }

    #[pyfunction]
    fn normal(mean: f64, std: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::normal(mean, std, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn uniform(
        low: f64,
        high: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::uniform(low, high, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn exponential(scale: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::exponential(scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn weibull(a: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::weibull(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn rayleigh(scale: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::rayleigh(scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn power(a: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::power(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn pareto(a: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::pareto(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn laplace(loc: f64, scale: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::laplace(loc, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn logistic(loc: f64, scale: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::logistic(loc, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn gumbel(loc: f64, scale: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::gumbel(loc, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn gamma(shape_param: f64, scale: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::gamma(shape_param, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn beta(a: f64, b: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::beta(a, b, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn dirichlet(alpha: PyObjectRef, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let alpha = parse_float64_vec(&alpha, vm)?;
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::dirichlet(&alpha, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn poisson(lam: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::poisson(lam, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn binomial(n: i64, p: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::binomial(n, p, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn geometric(p: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::geometric(p, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn multivariate_normal_from_cholesky(
        mean: PyObjectRef,
        chol: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let mean = parse_float64_vec(&mean, vm)?;
        let (chol, chol_shape) = parse_float64_array(&chol, vm)?;
        if chol_shape != vec![mean.len(), mean.len()] {
            return Err(vm.new_value_error("cholesky factor has incompatible shape".to_owned()));
        }
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::multivariate_normal_from_cholesky(&mean, &chol, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn vonmises(mu: f64, kappa: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::vonmises(mu, kappa, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn wald(mean: f64, scale: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::wald(mean, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn zipf(a: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::zipf(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn multinomial(
        n: i64,
        pvals: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let pvals = parse_float64_vec(&pvals, vm)?;
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::multinomial(n, &pvals, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn negative_binomial(n: i64, p: f64, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::negative_binomial(n, p, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn hypergeometric(
        ngood: i64,
        nbad: i64,
        nsample: i64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::hypergeometric(ngood, nbad, nsample, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn triangular(
        left: f64,
        mode: f64,
        right: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        numpy_rust_core::random::triangular(left, mode, right, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn choice(
        a: PyRef<PyNdArray>,
        size: usize,
        replace: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let replace = replace.unwrap_or(true);
        let a_inner = a.inner();
        numpy_rust_core::random::choice(&a_inner, size, replace)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn stateful_seed(seed: vm::function::OptionalArg<u64>, vm: &VirtualMachine) -> PyObjectRef {
        let rng = numpy_rust_core::random::StatefulRng::new(seed.into_option());
        vm.ctx.new_int(rng.state()).into()
    }

    #[pyfunction]
    fn rand_with_state(
        state: u64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = PyNdArray::from_core(rng.rand(&sh)).to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn randn_with_state(
        state: u64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = PyNdArray::from_core(rng.randn(&sh)).to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn uniform_with_state(
        state: u64,
        low: f64,
        high: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .uniform(low, high, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn normal_with_state(
        state: u64,
        mean: f64,
        std: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .normal(mean, std, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn exponential_with_state(
        state: u64,
        scale: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .exponential(scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn weibull_with_state(
        state: u64,
        a: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .weibull(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn rayleigh_with_state(
        state: u64,
        scale: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .rayleigh(scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn power_with_state(
        state: u64,
        a: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .power(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn pareto_with_state(
        state: u64,
        a: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .pareto(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn laplace_with_state(
        state: u64,
        loc: f64,
        scale: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .laplace(loc, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn logistic_with_state(
        state: u64,
        loc: f64,
        scale: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .logistic(loc, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn gumbel_with_state(
        state: u64,
        loc: f64,
        scale: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .gumbel(loc, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn gamma_with_state(
        state: u64,
        shape_param: f64,
        scale: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .gamma(shape_param, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn beta_with_state(
        state: u64,
        a: f64,
        b: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .beta(a, b, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn dirichlet_with_state(
        state: u64,
        alpha: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let alpha = parse_float64_vec(&alpha, vm)?;
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .dirichlet(&alpha, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn poisson_with_state(
        state: u64,
        lam: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .poisson(lam, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn binomial_with_state(
        state: u64,
        n: i64,
        p: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .binomial(n, p, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn geometric_with_state(
        state: u64,
        p: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .geometric(p, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn multivariate_normal_from_cholesky_with_state(
        state: u64,
        mean: PyObjectRef,
        chol: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let mean = parse_float64_vec(&mean, vm)?;
        let (chol, chol_shape) = parse_float64_array(&chol, vm)?;
        if chol_shape != vec![mean.len(), mean.len()] {
            return Err(vm.new_value_error("cholesky factor has incompatible shape".to_owned()));
        }
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .multivariate_normal_from_cholesky(&mean, &chol, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn vonmises_with_state(
        state: u64,
        mu: f64,
        kappa: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .vonmises(mu, kappa, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn wald_with_state(
        state: u64,
        mean: f64,
        scale: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .wald(mean, scale, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn zipf_with_state(
        state: u64,
        a: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .zipf(a, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn multinomial_with_state(
        state: u64,
        n: i64,
        pvals: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let pvals = parse_float64_vec(&pvals, vm)?;
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .multinomial(n, &pvals, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn negative_binomial_with_state(
        state: u64,
        n: i64,
        p: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .negative_binomial(n, p, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn hypergeometric_with_state(
        state: u64,
        ngood: i64,
        nbad: i64,
        nsample: i64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .hypergeometric(ngood, nbad, nsample, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn triangular_with_state(
        state: u64,
        left: f64,
        mode: f64,
        right: f64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .triangular(left, mode, right, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn randint_with_state(
        state: u64,
        low: i64,
        high: i64,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sh = extract_shape(&shape, vm)?;
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let arr = rng
            .randint(low, high, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))?
            .to_pyobject(vm);
        Ok(make_state_tuple(rng.state(), arr, vm))
    }

    #[pyfunction]
    fn randint_scalar_with_state(
        state: u64,
        low: i64,
        high: i64,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        let value = rng.randint_scalar(low, high).map_err(|e| err(e, vm))?;
        Ok(make_state_tuple(
            rng.state(),
            vm.ctx.new_int(value).into(),
            vm,
        ))
    }

    #[pyfunction]
    fn randbits_with_state(state: u64, bits: usize, vm: &VirtualMachine) -> PyObjectRef {
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        make_state_tuple(rng.state(), vm.ctx.new_int(rng.randbits(bits)).into(), vm)
    }

    #[pyfunction]
    fn advance_state(state: u64, delta: u64, vm: &VirtualMachine) -> PyObjectRef {
        let mut rng = numpy_rust_core::random::StatefulRng::from_state(state);
        rng.advance(delta);
        vm.ctx.new_int(rng.state()).into()
    }

    #[pyfunction]
    fn jumped_state(state: u64, vm: &VirtualMachine) -> PyObjectRef {
        let rng = numpy_rust_core::random::StatefulRng::from_state(state).jumped();
        vm.ctx.new_int(rng.state()).into()
    }
}

pub fn make_module(vm: &VirtualMachine) -> PyObjectRef {
    _random::module_def(&vm.ctx)
        .create_module(vm)
        .unwrap()
        .into()
}
