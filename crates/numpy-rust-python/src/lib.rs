pub mod py_array;
pub mod py_creation;
pub mod py_fft;
pub mod py_linalg;
pub mod py_random;

use rustpython_vm as vm;

/// Return the numpy module definition for registration with the interpreter builder.
pub fn numpy_module_def(ctx: &vm::Context) -> &'static vm::builtins::PyModuleDef {
    numpy::module_def(ctx)
}

#[vm::pymodule]
pub mod numpy {
    use super::*;
    use crate::py_array::PyNdArray;
    use vm::class::PyClassImpl;
    use vm::{PyObjectRef, PyResult, VirtualMachine};

    // Register the ndarray class type
    #[pyattr]
    fn ndarray(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyNdArray::make_class(&vm.ctx)
    }

    // --- Creation functions ---

    #[pyfunction]
    fn array(data: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        py_creation::py_array(data, vm)
    }

    #[pyfunction]
    fn zeros(shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        py_creation::py_zeros(&shape, vm)
    }

    #[pyfunction]
    fn ones(shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        py_creation::py_ones(&shape, vm)
    }

    #[pyfunction]
    fn arange(
        start: f64,
        stop: f64,
        step: vm::function::OptionalArg<f64>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        let step = step.unwrap_or(1.0);
        PyNdArray::from_core(numpy_rust_core::creation::arange(start, stop, step))
    }

    #[pyfunction]
    fn linspace(start: f64, stop: f64, num: usize, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::creation::linspace(start, stop, num))
    }

    #[pyfunction]
    fn eye(n: usize, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::creation::eye(
            n,
            numpy_rust_core::DType::Float64,
        ))
    }

    #[pyfunction]
    fn dot(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        numpy_rust_core::dot(a.inner(), b.inner())
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction(name = "where_")]
    fn where_fn(
        cond: vm::PyRef<PyNdArray>,
        x: vm::PyRef<PyNdArray>,
        y: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        numpy_rust_core::where_cond(cond.inner(), x.inner(), y.inner())
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn concatenate(
        arrays: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        py_creation::py_concatenate(arrays, axis.unwrap_or(0), vm)
    }

    // --- Submodules (registered as attributes) ---

    #[pyattr]
    fn linalg(vm: &VirtualMachine) -> PyObjectRef {
        py_linalg::make_module(vm)
    }

    #[pyattr]
    fn fft(vm: &VirtualMachine) -> PyObjectRef {
        py_fft::make_module(vm)
    }

    #[pyattr]
    fn random(vm: &VirtualMachine) -> PyObjectRef {
        py_random::make_module(vm)
    }
}
