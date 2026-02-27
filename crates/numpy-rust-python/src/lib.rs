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
    use crate::py_array::{obj_to_ndarray, parse_optional_axis, PyNdArray, PyNdArrayIter};
    use vm::class::PyClassImpl;
    use vm::{PyObjectRef, PyPayload, PyResult, VirtualMachine};

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
    fn zeros(
        shape: PyObjectRef,
        dtype: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dt = match dtype.into_option() {
            Some(s) => Some(py_array::parse_dtype(s.as_str(), vm)?),
            None => None,
        };
        py_creation::py_zeros(&shape, dt, vm)
    }

    #[pyfunction]
    fn ones(
        shape: PyObjectRef,
        dtype: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dt = match dtype.into_option() {
            Some(s) => Some(py_array::parse_dtype(s.as_str(), vm)?),
            None => None,
        };
        py_creation::py_ones(&shape, dt, vm)
    }

    #[pyfunction]
    fn arange(
        start: f64,
        stop: f64,
        step: vm::function::OptionalArg<f64>,
        dtype: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let step = step.unwrap_or(1.0);
        let dt = match dtype.into_option() {
            Some(s) => Some(py_array::parse_dtype(s.as_str(), vm)?),
            None => None,
        };
        Ok(PyNdArray::from_core(numpy_rust_core::creation::arange(
            start, stop, step, dt,
        )))
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
        dtype: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let m_val = m.into_option();
        let k_val = k.unwrap_or(0);
        let dt = match dtype.into_option() {
            Some(s) => py_array::parse_dtype(s.as_str(), vm)?,
            None => numpy_rust_core::DType::Float64,
        };
        Ok(PyNdArray::from_core(numpy_rust_core::creation::eye(
            n, m_val, k_val, dt,
        )))
    }

    #[pyfunction]
    fn full(
        shape: PyObjectRef,
        fill_value: f64,
        dtype: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let shape_vec = py_array::extract_shape(&shape, vm)?;
        let dt = match dtype.into_option() {
            Some(s) => py_array::parse_dtype(s.as_str(), vm)?,
            None => numpy_rust_core::DType::Float64,
        };
        Ok(PyNdArray::from_core(numpy_rust_core::creation::full(
            &shape_vec, fill_value, dt,
        )))
    }

    #[pyfunction]
    fn promote_types(
        type1: vm::PyRef<vm::builtins::PyStr>,
        type2: vm::PyRef<vm::builtins::PyStr>,
        vm: &VirtualMachine,
    ) -> PyResult<String> {
        let dt1 = py_array::parse_dtype(type1.as_str(), vm)?;
        let dt2 = py_array::parse_dtype(type2.as_str(), vm)?;
        Ok(dt1.promote(dt2).to_string())
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
    fn argwhere(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::argwhere(&a.inner()))
    }

    #[pyfunction]
    fn concatenate(
        arrays: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        py_creation::py_concatenate(arrays, axis.unwrap_or(0), vm)
    }

    #[pyfunction]
    fn split(
        a: vm::PyRef<PyNdArray>,
        indices_or_sections: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = axis.unwrap_or(0);
        let spec = if let Ok(n) = indices_or_sections.clone().try_into_value::<usize>(vm) {
            numpy_rust_core::SplitSpec::NSections(n)
        } else if let Some(list) = indices_or_sections.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            let indices: Vec<usize> = items
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?;
            numpy_rust_core::SplitSpec::Indices(indices)
        } else {
            return Err(vm.new_type_error("indices_or_sections must be int or list".to_owned()));
        };
        let parts = numpy_rust_core::split(&a.inner(), &spec, axis)
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_parts: Vec<PyObjectRef> = parts
            .into_iter()
            .map(|p| PyNdArray::from_core(p).into_pyobject(vm))
            .collect();
        Ok(vm.ctx.new_list(py_parts).into())
    }

    #[pyfunction]
    fn repeat(
        a: vm::PyRef<PyNdArray>,
        repeats: usize,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        numpy_rust_core::manipulation::repeat(&a.inner(), repeats, axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn tile(
        a: vm::PyRef<PyNdArray>,
        reps: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let reps_vec = if let Ok(n) = reps.clone().try_into_value::<usize>(vm) {
            vec![n]
        } else if let Some(tuple) = reps.downcast_ref::<vm::builtins::PyTuple>() {
            tuple
                .as_slice()
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        } else if let Some(list) = reps.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            return Err(vm.new_type_error("reps must be int, tuple, or list".to_owned()));
        };
        numpy_rust_core::manipulation::tile(&a.inner(), &reps_vec)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
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

    #[pyfunction]
    fn log10(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().log10())
    }

    #[pyfunction]
    fn log2(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().log2())
    }

    #[pyfunction]
    fn sinh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().sinh())
    }

    #[pyfunction]
    fn cosh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().cosh())
    }

    #[pyfunction]
    fn tanh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().tanh())
    }

    #[pyfunction]
    fn arcsin(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().arcsin())
    }

    #[pyfunction]
    fn arccos(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().arccos())
    }

    #[pyfunction]
    fn arctan(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().arctan())
    }

    #[pyfunction]
    fn sign(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().sign())
    }

    #[pyfunction]
    fn log1p(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .log1p()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn expm1(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .expm1()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn deg2rad(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .deg2rad()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pyfunction]
    fn rad2deg(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .rad2deg()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    // --- Element-wise check functions ---

    #[pyfunction]
    fn isnan(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().isnan())
    }

    #[pyfunction]
    fn isinf(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().isinf())
    }

    #[pyfunction]
    fn isfinite(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().isfinite())
    }

    #[pyfunction]
    fn around(
        a: vm::PyRef<PyNdArray>,
        decimals: vm::function::OptionalArg<i32>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        let d = decimals.unwrap_or(0);
        PyNdArray::from_core(a.inner().around(d))
    }

    #[pyfunction]
    fn signbit(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().signbit())
    }

    #[pyfunction]
    fn logical_not(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().logical_not())
    }

    #[pyfunction]
    fn power(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.pow(&b)
            .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nonzero(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult {
        let result = numpy_rust_core::nonzero(&a.inner());
        let py_arrays: Vec<PyObjectRef> = result
            .into_iter()
            .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
            .collect();
        Ok(vm.ctx.new_tuple(py_arrays).into())
    }

    #[pyfunction]
    fn count_nonzero(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> usize {
        numpy_rust_core::count_nonzero(&a.inner())
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

    // --- NaN-safe reduction functions ---

    #[pyfunction]
    fn nansum(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nansum(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanmean(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanmean(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanstd(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let ddof = ddof.unwrap_or(0);
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanstd(ax, ddof, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanvar(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let ddof = ddof.unwrap_or(0);
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanvar(ax, ddof, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanmin(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanmin(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanmax(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanmax(ax, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanargmin(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .nanargmin(ax)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanargmax(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        a.inner()
            .nanargmax(ax)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nanprod(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .nanprod(ax, keepdims)
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
    fn nancumsum(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .nancumsum(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nancumprod(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .nancumprod(axis)
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

    // --- Product reduction ---

    #[pyfunction]
    fn prod(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        a.inner()
            .prod(axis, keepdims)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
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

    // --- Set Operations ---

    #[pyfunction]
    fn intersect1d(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::intersect1d(&a.inner(), &b.inner()))
    }

    #[pyfunction]
    fn union1d(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::union1d(&a.inner(), &b.inner()))
    }

    #[pyfunction]
    fn setdiff1d(
        a: vm::PyRef<PyNdArray>,
        b: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::setdiff1d(&a.inner(), &b.inner()))
    }

    #[pyfunction]
    fn isin(
        element: vm::PyRef<PyNdArray>,
        test_elements: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(numpy_rust_core::isin(
            &element.inner(),
            &test_elements.inner(),
        ))
    }

    // --- Histogram / Bincount ---

    #[pyfunction]
    fn histogram(
        a: vm::PyRef<PyNdArray>,
        bins: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let bins = bins.unwrap_or(10);
        let (counts, edges) = a
            .inner()
            .histogram(bins, None)
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_counts = PyNdArray::from_core(counts).into_pyobject(vm);
        let py_edges = PyNdArray::from_core(edges).into_pyobject(vm);
        Ok(vm.ctx.new_tuple(vec![py_counts, py_edges]).into())
    }

    #[pyfunction]
    fn bincount(
        x: vm::PyRef<PyNdArray>,
        weights: PyObjectRef,
        minlength: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let minlength = minlength.unwrap_or(0);
        if vm.is_none(&weights) {
            x.inner()
                .bincount(None, minlength)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string()))
        } else {
            let w_arr: vm::PyRef<PyNdArray> = weights.try_into_value(vm)?;
            let w_guard = w_arr.inner();
            x.inner()
                .bincount(Some(&*w_guard), minlength)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string()))
        }
    }

    // --- Quantile / Percentile ---

    #[pyfunction]
    fn quantile(
        a: vm::PyRef<PyNdArray>,
        q: f64,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .quantile(q, axis)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn percentile(
        a: vm::PyRef<PyNdArray>,
        q: f64,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .percentile(q, axis)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Correlation / Covariance ---

    #[pyfunction]
    fn cov(
        m: vm::PyRef<PyNdArray>,
        y: PyObjectRef,
        rowvar: vm::function::OptionalArg<bool>,
        ddof: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let rowvar = rowvar.unwrap_or(true);
        let ddof = ddof.unwrap_or(1);
        if vm.is_none(&y) {
            m.inner()
                .cov(rowvar, ddof)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string()))
        } else {
            let y_arr: vm::PyRef<PyNdArray> = y.try_into_value(vm)?;
            let m_clone = m.inner().clone();
            let y_clone = y_arr.inner().clone();
            numpy_rust_core::ops::correlation::cov_xy(&m_clone, &y_clone, ddof)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string()))
        }
    }

    #[pyfunction]
    fn corrcoef(
        x: vm::PyRef<PyNdArray>,
        y: PyObjectRef,
        rowvar: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let rowvar = rowvar.unwrap_or(true);
        if vm.is_none(&y) {
            x.inner()
                .corrcoef(rowvar)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string()))
        } else {
            let y_arr: vm::PyRef<PyNdArray> = y.try_into_value(vm)?;
            let x_clone = x.inner().clone();
            let y_clone = y_arr.inner().clone();
            numpy_rust_core::ops::correlation::corrcoef_xy(&x_clone, &y_clone)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string()))
        }
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

    // --- Array manipulation: flip, flipud, fliplr, rot90, unique, roll, take, diagonal, outer ---

    #[pyfunction]
    fn flip(
        a: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let data = a.inner();
        let ax = parse_optional_axis(axis, vm)?;
        data.flip(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn flipud(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let data = a.inner();
        data.flip(Some(0))
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn fliplr(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let data = a.inner();
        data.flip(Some(1))
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn rot90(
        a: vm::PyRef<PyNdArray>,
        k: vm::function::OptionalArg<i32>,
        _vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let data = a.inner();
        let k_val = k.unwrap_or(1);
        data.rot90(k_val)
            .map(PyNdArray::from_core)
            .map_err(|e| _vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn unique(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        let data = a.inner();
        PyNdArray::from_core(numpy_rust_core::unique(&data))
    }

    #[pyfunction]
    fn diagonal(
        a: vm::PyRef<PyNdArray>,
        offset: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let data = a.inner();
        let off = offset.unwrap_or(0);
        numpy_rust_core::diagonal(&data, off)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn outer(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let a_arr = obj_to_ndarray(&a, vm)?;
        let b_arr = obj_to_ndarray(&b, vm)?;
        Ok(PyNdArray::from_core(numpy_rust_core::outer(&a_arr, &b_arr)).into_pyobject(vm))
    }

    #[pyfunction]
    fn roll(
        a: vm::PyRef<PyNdArray>,
        shift: i64,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let data = a.inner();
        let ax = parse_optional_axis(axis, vm)?;
        data.roll(shift, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn take(
        a: vm::PyRef<PyNdArray>,
        indices: PyObjectRef,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        // Parse indices from list, tuple, or ndarray
        let idx: Vec<usize> = if let Some(arr) = indices.downcast_ref::<PyNdArray>() {
            // Extract from ndarray
            let inner = arr.inner();
            let flat = inner.flatten().astype(numpy_rust_core::DType::Int64);
            let numpy_rust_core::ArrayData::Int64(data) = flat.data() else {
                return Err(vm.new_type_error("indices must be integer type".to_owned()));
            };
            data.iter().map(|&v| v as usize).collect()
        } else if let Some(list) = indices.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        } else if let Some(tuple) = indices.downcast_ref::<vm::builtins::PyTuple>() {
            tuple
                .as_slice()
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        } else if let Ok(single) = indices.clone().try_into_value::<usize>(vm) {
            vec![single]
        } else {
            return Err(
                vm.new_type_error("indices must be list, tuple, ndarray, or int".to_owned())
            );
        };

        let data = a.inner();
        let ax = parse_optional_axis(axis, vm)?;
        data.take(&idx, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Index Utilities ---

    fn parse_shape_tuple(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
        if let Some(tuple) = obj.downcast_ref::<vm::builtins::PyTuple>() {
            tuple
                .as_slice()
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()
        } else if let Some(list) = obj.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| item.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()
        } else if let Ok(n) = obj.clone().try_into_value::<usize>(vm) {
            Ok(vec![n])
        } else {
            Err(vm.new_type_error("shape must be tuple, list, or int".to_owned()))
        }
    }

    #[pyfunction]
    fn unravel_index(
        indices: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        // Parse indices: could be int or ndarray
        let idx_arr = obj_to_ndarray(&indices, vm)?;
        // Parse shape: tuple/list of ints
        let shape_vec = parse_shape_tuple(&shape, vm)?;
        let result = numpy_rust_core::indexing::unravel_index(&idx_arr, &shape_vec)
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_arrays: Vec<PyObjectRef> = result
            .into_iter()
            .map(|a| py_array::ndarray_or_scalar(a, vm))
            .collect();
        Ok(vm.ctx.new_tuple(py_arrays).into())
    }

    #[pyfunction]
    fn ravel_multi_index(
        multi_index: PyObjectRef,
        dims: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        // Parse multi_index: tuple of arrays/ints
        let tuple = multi_index
            .downcast_ref::<vm::builtins::PyTuple>()
            .ok_or_else(|| vm.new_type_error("multi_index must be a tuple".to_owned()))?;
        let arrs: Vec<numpy_rust_core::NdArray> = tuple
            .as_slice()
            .iter()
            .map(|item| obj_to_ndarray(item, vm))
            .collect::<PyResult<Vec<_>>>()?;
        let refs: Vec<&numpy_rust_core::NdArray> = arrs.iter().collect();
        let dims_vec = parse_shape_tuple(&dims, vm)?;
        numpy_rust_core::indexing::ravel_multi_index(&refs, &dims_vec)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Stacking helpers ---

    fn extract_ndarray_list(
        obj: &PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<vm::PyRef<PyNdArray>>> {
        if let Some(list) = obj.downcast_ref::<vm::builtins::PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| item.clone().try_into_value::<vm::PyRef<PyNdArray>>(vm))
                .collect::<PyResult<Vec<_>>>()
        } else if let Some(tuple) = obj.downcast_ref::<vm::builtins::PyTuple>() {
            tuple
                .as_slice()
                .iter()
                .map(|item| item.clone().try_into_value::<vm::PyRef<PyNdArray>>(vm))
                .collect::<PyResult<Vec<_>>>()
        } else {
            Err(vm.new_type_error("expected list or tuple of arrays".to_owned()))
        }
    }

    #[pyfunction]
    fn stack_native(
        arrays: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = axis.unwrap_or(0);
        let arr_list = extract_ndarray_list(&arrays, vm)?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            arr_list.iter().map(|a| a.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        numpy_rust_core::stack(&refs, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn column_stack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr_list = extract_ndarray_list(&arrays, vm)?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            arr_list.iter().map(|a| a.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        numpy_rust_core::column_stack(&refs)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn dstack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr_list = extract_ndarray_list(&arrays, vm)?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            arr_list.iter().map(|a| a.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        numpy_rust_core::dstack(&refs)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- meshgrid / pad ---

    #[pyfunction]
    fn meshgrid(
        arrays: PyObjectRef,
        indexing: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let idx = indexing.as_ref().map(|s| s.as_str()).unwrap_or("xy");
        let arr_list = extract_ndarray_list(&arrays, vm)?;
        let borrowed: Vec<std::sync::RwLockReadGuard<'_, numpy_rust_core::NdArray>> =
            arr_list.iter().map(|a| a.inner()).collect();
        let refs: Vec<&numpy_rust_core::NdArray> = borrowed.iter().map(|r| &**r).collect();
        let result =
            numpy_rust_core::meshgrid(&refs, idx).map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_arrays: Vec<PyObjectRef> = result
            .into_iter()
            .map(|a| PyNdArray::from_core(a).into_pyobject(vm))
            .collect();
        Ok(vm.ctx.new_tuple(py_arrays).into())
    }

    fn parse_pad_width(
        obj: &PyObjectRef,
        ndim: usize,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<(usize, usize)>> {
        // Single int: apply to all axes, both sides
        if let Ok(n) = obj.clone().try_into_value::<usize>(vm) {
            return Ok(vec![(n, n); ndim]);
        }
        // Tuple of two ints: (before, after) for all axes
        if let Some(tuple) = obj.downcast_ref::<vm::builtins::PyTuple>() {
            let items = tuple.as_slice();
            if items.len() == 2 {
                if let (Ok(a), Ok(b)) = (
                    items[0].clone().try_into_value::<usize>(vm),
                    items[1].clone().try_into_value::<usize>(vm),
                ) {
                    return Ok(vec![(a, b); ndim]);
                }
            }
            // List of tuples
            let mut result = Vec::with_capacity(items.len());
            for item in items {
                if let Some(inner) = item.downcast_ref::<vm::builtins::PyTuple>() {
                    let inner_items = inner.as_slice();
                    if inner_items.len() == 2 {
                        let a: usize = inner_items[0].clone().try_into_value(vm)?;
                        let b: usize = inner_items[1].clone().try_into_value(vm)?;
                        result.push((a, b));
                    } else {
                        return Err(
                            vm.new_value_error("pad_width tuples must have 2 elements".to_owned())
                        );
                    }
                } else {
                    return Err(vm.new_type_error(
                        "pad_width must be int, (int, int), or tuple of (int, int)".to_owned(),
                    ));
                }
            }
            return Ok(result);
        }
        Err(vm.new_type_error("pad_width must be int, tuple, or list".to_owned()))
    }

    #[pyfunction]
    fn pad(
        a: vm::PyRef<PyNdArray>,
        pad_width: PyObjectRef,
        constant_values: vm::function::OptionalArg<f64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let cv = constant_values.unwrap_or(0.0);
        let pw = parse_pad_width(&pad_width, a.inner().ndim(), vm)?;
        numpy_rust_core::pad_constant(&a.inner(), &pw, cv)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- arctan2 & clip ---

    #[pyfunction]
    fn arctan2(
        y: vm::PyRef<PyNdArray>,
        x: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        y.inner()
            .arctan2(&x.inner())
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn clip(
        a: vm::PyRef<PyNdArray>,
        a_min: vm::function::OptionalArg<f64>,
        a_max: vm::function::OptionalArg<f64>,
        _vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        Ok(PyNdArray::from_core(
            a.inner().clip(a_min.into_option(), a_max.into_option()),
        ))
    }

    // --- Interpolation & gradient ---

    #[pyfunction]
    fn interp(
        x: vm::PyRef<PyNdArray>,
        xp: vm::PyRef<PyNdArray>,
        fp: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        numpy_rust_core::ops::numerical::interp(&x.inner(), &xp.inner(), &fp.inner())
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn gradient(
        f: vm::PyRef<PyNdArray>,
        spacing: vm::function::OptionalArg<f64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let sp = spacing.unwrap_or(1.0);
        let data = f.inner();
        if data.ndim() <= 1 {
            numpy_rust_core::ops::numerical::gradient_1d(&data, sp)
                .map(|arr| PyNdArray::from_core(arr).into_pyobject(vm))
                .map_err(|e| vm.new_value_error(e.to_string()))
        } else {
            let grads = numpy_rust_core::ops::numerical::gradient_nd(&data, sp)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            let py_arrays: Vec<PyObjectRef> = grads
                .into_iter()
                .map(|a| PyNdArray::from_core(a).into_pyobject(vm))
                .collect();
            Ok(vm.ctx.new_list(py_arrays).into())
        }
    }

    // --- Polynomial ---

    #[cfg(feature = "linalg")]
    #[pyfunction]
    fn polyfit(
        x: vm::PyRef<PyNdArray>,
        y: vm::PyRef<PyNdArray>,
        deg: usize,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        numpy_rust_core::ops::polynomial::polyfit(&x.inner(), &y.inner(), deg)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn polyval(
        p: vm::PyRef<PyNdArray>,
        x: vm::PyRef<PyNdArray>,
        _vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        Ok(PyNdArray::from_core(
            numpy_rust_core::ops::polynomial::polyval(&p.inner(), &x.inner()),
        ))
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
