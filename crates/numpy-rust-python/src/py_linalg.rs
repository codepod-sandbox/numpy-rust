use rustpython_vm as vm;
use vm::convert::ToPyObject;
use vm::{PyObjectRef, PyRef, PyResult, VirtualMachine};

use crate::py_array::PyNdArray;

fn err(e: numpy_rust_core::NumpyError, vm: &VirtualMachine) -> vm::builtins::PyBaseExceptionRef {
    vm.new_value_error(e.to_string())
}

#[vm::pymodule]
mod _linalg {
    use super::*;

    #[pyfunction]
    fn matmul(
        a: PyRef<PyNdArray>,
        b: PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        numpy_rust_core::linalg::matmul(a.inner(), b.inner())
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn inv(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        numpy_rust_core::linalg::inv(a.inner())
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn solve(
        a: PyRef<PyNdArray>,
        b: PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        numpy_rust_core::linalg::solve(a.inner(), b.inner())
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn det(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<f64> {
        numpy_rust_core::linalg::det(a.inner()).map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn eig(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<vm::builtins::PyTupleRef> {
        let (vals, vecs) = numpy_rust_core::linalg::eig(a.inner()).map_err(|e| err(e, vm))?;
        Ok(vm::builtins::PyTuple::new_ref(
            vec![
                PyNdArray::from_core(vals).to_pyobject(vm),
                PyNdArray::from_core(vecs).to_pyobject(vm),
            ],
            &vm.ctx,
        ))
    }

    #[pyfunction]
    fn svd(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<vm::builtins::PyTupleRef> {
        let (u, s, vt) = numpy_rust_core::linalg::svd(a.inner()).map_err(|e| err(e, vm))?;
        Ok(vm::builtins::PyTuple::new_ref(
            vec![
                PyNdArray::from_core(u).to_pyobject(vm),
                PyNdArray::from_core(s).to_pyobject(vm),
                PyNdArray::from_core(vt).to_pyobject(vm),
            ],
            &vm.ctx,
        ))
    }

    #[pyfunction]
    fn qr(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<vm::builtins::PyTupleRef> {
        let (q, r) = numpy_rust_core::linalg::qr(a.inner()).map_err(|e| err(e, vm))?;
        Ok(vm::builtins::PyTuple::new_ref(
            vec![
                PyNdArray::from_core(q).to_pyobject(vm),
                PyNdArray::from_core(r).to_pyobject(vm),
            ],
            &vm.ctx,
        ))
    }

    #[pyfunction]
    fn norm(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<f64> {
        numpy_rust_core::linalg::norm(a.inner()).map_err(|e| err(e, vm))
    }

    #[pyfunction]
    fn cholesky(a: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        numpy_rust_core::linalg::cholesky(a.inner())
            .map(PyNdArray::from_core)
            .map_err(|e| err(e, vm))
    }
}

pub fn make_module(vm: &VirtualMachine) -> PyObjectRef {
    _linalg::module_def(&vm.ctx).create_module(vm).unwrap().into()
}
