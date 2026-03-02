use rustpython_vm as vm;
use vm::{PyObjectRef, PyRef, PyResult, VirtualMachine};

use crate::py_array::{extract_shape, PyNdArray};

fn err(e: numpy_rust_core::NumpyError, vm: &VirtualMachine) -> vm::builtins::PyBaseExceptionRef {
    vm.new_value_error(e.to_string())
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
        // Accept: randint(low, high, size) or randint(low, high, size=N)
        let low: i64 = args
            .args
            .first()
            .ok_or_else(|| vm.new_type_error("randint() missing argument: 'low'".to_owned()))?
            .clone()
            .try_into_value(vm)?;
        let high: i64 = args
            .args
            .get(1)
            .ok_or_else(|| vm.new_type_error("randint() missing argument: 'high'".to_owned()))?
            .clone()
            .try_into_value(vm)?;
        // size can be positional (3rd arg) or keyword 'size'
        let size_obj = if let Some(pos) = args.args.get(2) {
            pos.clone()
        } else if let Some(kw) = args.kwargs.get("size") {
            kw.clone()
        } else {
            // Scalar: return single value
            let sh = vec![1usize];
            return numpy_rust_core::random::randint(low, high, &sh)
                .map(PyNdArray::from_core)
                .map_err(|e| err(e, vm));
        };
        let sh = extract_shape(&size_obj, vm)?;
        numpy_rust_core::random::randint(low, high, &sh)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
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
}

pub fn make_module(vm: &VirtualMachine) -> PyObjectRef {
    _random::module_def(&vm.ctx)
        .create_module(vm)
        .unwrap()
        .into()
}
