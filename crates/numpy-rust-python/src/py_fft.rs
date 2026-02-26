use rustpython_vm as vm;
use vm::{PyObjectRef, PyRef, PyResult, VirtualMachine};

use crate::py_array::PyNdArray;

fn err(e: numpy_rust_core::NumpyError, vm: &VirtualMachine) -> vm::builtins::PyBaseExceptionRef {
    vm.new_value_error(e.to_string())
}

#[vm::pymodule]
mod _fft {
    use super::*;

    #[pyfunction]
    fn fft(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a_inner = a.inner();
        numpy_rust_core::fft::fft(&a_inner)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn ifft(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a_inner = a.inner();
        numpy_rust_core::fft::ifft(&a_inner)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn rfft(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a_inner = a.inner();
        numpy_rust_core::fft::rfft(&a_inner)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn irfft(a: PyRef<PyNdArray>, n: usize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a_inner = a.inner();
        numpy_rust_core::fft::irfft(&a_inner, n)
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn fftfreq(n: usize, d: vm::function::OptionalArg<f64>, _vm: &VirtualMachine) -> PyNdArray {
        let d = d.unwrap_or(1.0);
        PyNdArray::from_core(numpy_rust_core::fft::fftfreq(n, d))
    }
}

pub fn make_module(vm: &VirtualMachine) -> PyObjectRef {
    _fft::module_def(&vm.ctx).create_module(vm).unwrap().into()
}
