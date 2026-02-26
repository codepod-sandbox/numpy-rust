pub mod py_array;
pub mod py_creation;
#[cfg(feature = "fft")]
pub mod py_fft;
#[cfg(feature = "linalg")]
pub mod py_linalg;
#[cfg(feature = "random")]
pub mod py_random;

use rustpython_vm as vm;

/// Return the native numpy module definition for registration with the interpreter builder.
pub fn numpy_module_def(ctx: &vm::Context) -> &'static vm::builtins::PyModuleDef {
    _numpy_native::module_def(ctx)
}

#[vm::pymodule]
pub mod _numpy_native {
    use super::*;
    use crate::py_array::{parse_optional_axis, PyNdArray, PyNdArrayIter};
    use vm::class::PyClassImpl;
    use vm::{PyObjectRef, PyResult, VirtualMachine};

    // Register the ndarray class type
    #[pyattr]
    fn ndarray(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyNdArray::make_class(&vm.ctx)
    }

    #[pyattr]
    fn ndarray_iterator(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyNdArrayIter::make_class(&vm.ctx)
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
        PyNdArray::from_core(numpy_rust_core::creation::arange(start, stop, step, None))
    }

    #[pyfunction]
    fn linspace(start: f64, stop: f64, num: usize, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::creation::linspace(start, stop, num))
    }

    #[pyfunction]
    fn eye(
        n: usize,
        m: vm::function::OptionalArg<usize>,
        k: vm::function::OptionalArg<isize>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        let m_val = m.into_option();
        let k_val = k.unwrap_or(0);
        PyNdArray::from_core(numpy_rust_core::creation::eye(
            n,
            m_val,
            k_val,
            numpy_rust_core::DType::Float64,
        ))
    }

    #[pyfunction]
    fn dot(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let a_inner = a.inner();
        let b_inner = b.inner();
        numpy_rust_core::dot(&a_inner, &b_inner)
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
        let cond_inner = cond.inner();
        let x_inner = x.inner();
        let y_inner = y.inner();
        numpy_rust_core::where_cond(&cond_inner, &x_inner, &y_inner)
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

    // --- Module-level math functions ---

    #[pyfunction]
    fn abs(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().abs())
    }

    #[pyfunction]
    fn sqrt(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().sqrt())
    }

    #[pyfunction]
    fn exp(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().exp())
    }

    #[pyfunction]
    fn log(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().log())
    }

    #[pyfunction]
    fn sin(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().sin())
    }

    #[pyfunction]
    fn cos(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().cos())
    }

    #[pyfunction]
    fn tan(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().tan())
    }

    #[pyfunction]
    fn floor(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .floor()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn ceil(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .ceil()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn round(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .round()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Module-level reduction functions ---

    #[pyfunction]
    fn sum(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .sum(ax, false)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn mean(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .mean(ax, false)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn min(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .min(ax, false)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn max(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .max(ax, false)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn std(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .std(ax, 0, false)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn var(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .var(ax, 0, false)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Einsum ---

    #[pyfunction]
    fn einsum(
        subscripts: vm::PyRef<vm::builtins::PyStr>,
        args: vm::function::PosArgs<vm::PyRef<PyNdArray>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let operands: Vec<numpy_rust_core::NdArray> =
            args.iter().map(|a| a.inner().clone()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = operands.iter().collect();
        numpy_rust_core::einsum(subscripts.as_str(), &refs)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Cumulative / Diff ---

    #[pyfunction]
    fn cumsum(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .cumsum(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn cumprod(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .cumprod(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn diff(
        a: vm::PyRef<PyNdArray>,
        n: vm::function::OptionalArg<usize>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let n = n.unwrap_or(1);
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .diff(n, axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Sort / Argsort ---

    #[pyfunction]
    fn sort(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .sort(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn argsort(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .argsort(ax)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Selection / Search ---

    #[pyfunction]
    fn searchsorted(
        a: vm::PyRef<PyNdArray>,
        v: vm::PyRef<PyNdArray>,
        side: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let side_str = side.as_ref().map(|s| s.as_str()).unwrap_or("left");
        a.inner()
            .searchsorted(&v.inner(), side_str)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn compress(
        condition: vm::PyRef<PyNdArray>,
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .compress(&condition.inner(), ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn choose(
        a: vm::PyRef<PyNdArray>,
        choices: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let list = choices
            .downcast_ref::<vm::builtins::PyList>()
            .ok_or_else(|| vm.new_type_error("choose requires a list of arrays".to_owned()))?;
        let items = list.borrow_vec();
        let py_arrays: Vec<vm::PyRef<PyNdArray>> = items
            .iter()
            .map(|item| item.clone().try_into_value::<vm::PyRef<PyNdArray>>(vm))
            .collect::<PyResult<Vec<_>>>()?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            py_arrays.iter().map(|c| c.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        numpy_rust_core::choose(&a.inner(), &refs)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- String (char) operations ---

    #[pyfunction]
    fn char_upper(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .str_upper()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn char_lower(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .str_lower()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn char_capitalize(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .str_capitalize()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn char_strip(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .str_strip()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn char_str_len(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .str_len()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn char_startswith(
        a: vm::PyRef<PyNdArray>,
        prefix: vm::PyRef<vm::builtins::PyStr>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        a.inner()
            .str_startswith(prefix.as_str())
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn char_endswith(
        a: vm::PyRef<PyNdArray>,
        suffix: vm::PyRef<vm::builtins::PyStr>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        a.inner()
            .str_endswith(suffix.as_str())
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn char_replace(
        a: vm::PyRef<PyNdArray>,
        old: vm::PyRef<vm::builtins::PyStr>,
        new: vm::PyRef<vm::builtins::PyStr>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        a.inner()
            .str_replace(old.as_str(), new.as_str())
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    // --- Submodules (registered as attributes, feature-gated) ---

    #[cfg(feature = "linalg")]
    #[pyattr]
    fn linalg(vm: &VirtualMachine) -> PyObjectRef {
        py_linalg::make_module(vm)
    }

    #[cfg(feature = "fft")]
    #[pyattr]
    fn fft(vm: &VirtualMachine) -> PyObjectRef {
        py_fft::make_module(vm)
    }

    #[cfg(feature = "random")]
    #[pyattr]
    fn random(vm: &VirtualMachine) -> PyObjectRef {
        py_random::make_module(vm)
    }
}
