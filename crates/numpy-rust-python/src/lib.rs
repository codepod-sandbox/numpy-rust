pub mod py_array;
pub mod py_creation;
#[cfg(feature = "fft")]
pub mod py_fft;
#[cfg(feature = "linalg")]
pub mod py_linalg;
#[cfg(feature = "random")]
pub mod py_random;
pub mod py_struct_array;

use rustpython_vm as vm;

/// Return the native numpy module definition for registration with the interpreter builder.
pub fn numpy_module_def(ctx: &vm::Context) -> &'static vm::builtins::PyModuleDef {
    _numpy_native::module_def(ctx)
}

#[vm::pymodule]
pub mod _numpy_native {
    use super::*;
    use crate::py_array::{
        obj_to_ndarray, parse_optional_axis, PyFlagsObj, PyFlatIter, PyNdArray, PyNdArrayIter,
    };
    use numpy_rust_core::NdArray;
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

    #[pyattr]
    fn flatiter(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyFlatIter::make_class(&vm.ctx)
    }

    #[pyattr]
    fn flagsobj(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyFlagsObj::make_class(&vm.ctx)
    }

    use crate::py_struct_array::PyStructuredArray;

    #[allow(non_snake_case)]
    #[pyattr]
    fn StructuredArray(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyStructuredArray::make_class(&vm.ctx)
    }

    // --- Creation functions ---

    fn py_sequence_items(
        obj: &PyObjectRef,
        vm: &VirtualMachine,
        expected: &str,
    ) -> PyResult<Vec<PyObjectRef>> {
        if let Some(tuple) = obj.downcast_ref::<vm::builtins::PyTuple>() {
            Ok(tuple.as_slice().to_vec())
        } else if let Some(list) = obj.downcast_ref::<vm::builtins::PyList>() {
            Ok(list.borrow_vec().to_vec())
        } else {
            Err(vm.new_type_error(expected.to_owned()))
        }
    }

    fn parse_usize_values(
        obj: &PyObjectRef,
        vm: &VirtualMachine,
        expected: &str,
    ) -> PyResult<Vec<usize>> {
        if let Ok(n) = obj.clone().try_into_value::<usize>(vm) {
            return Ok(vec![n]);
        }

        let items = py_sequence_items(obj, vm, expected)?;
        items
            .into_iter()
            .map(|item| item.try_into_value::<usize>(vm))
            .collect::<PyResult<Vec<_>>>()
    }

    fn parse_indices_arg(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
        if let Some(arr) = obj.downcast_ref::<PyNdArray>() {
            let inner = arr.inner();
            let flat = inner.flatten().astype(numpy_rust_core::DType::Int64);
            let numpy_rust_core::ArrayData::Int64(data) = flat.data() else {
                return Err(vm.new_type_error("indices must be integer type".to_owned()));
            };
            Ok(data.iter().map(|&v| v as usize).collect())
        } else {
            parse_usize_values(obj, vm, "indices must be list, tuple, ndarray, or int")
        }
    }

    fn extract_core_array_sequence(
        obj: &PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<numpy_rust_core::NdArray>> {
        let items = py_sequence_items(obj, vm, "expected list or tuple of array-like values")?;
        items
            .into_iter()
            .map(|item| obj_to_ndarray(&item, vm))
            .collect::<PyResult<Vec<_>>>()
    }

    fn parse_index_array_sequence(
        obj: &PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<numpy_rust_core::NdArray>> {
        let items = py_sequence_items(obj, vm, "multi_index must be a list or tuple")?;
        items
            .into_iter()
            .map(|item| obj_to_ndarray(&item, vm))
            .collect::<PyResult<Vec<_>>>()
    }

    fn parse_float_edge_sequence(
        obj: &PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<Vec<f64>>> {
        let items = py_sequence_items(obj, vm, "edges must be a list or tuple")?;
        items
            .into_iter()
            .map(|item| {
                if let Some(arr) = item.downcast_ref::<PyNdArray>() {
                    let cast = arr
                        .inner()
                        .flatten()
                        .astype(numpy_rust_core::DType::Float64);
                    let numpy_rust_core::ArrayData::Float64(values) = cast.data() else {
                        unreachable!("float64 cast must produce float64 storage")
                    };
                    Ok(values.iter().copied().collect())
                } else {
                    let edge_items = py_sequence_items(
                        &item,
                        vm,
                        "each edge array must be a list, tuple, or ndarray",
                    )?;
                    edge_items
                        .into_iter()
                        .map(|edge| edge.try_into_value::<f64>(vm))
                        .collect::<PyResult<Vec<_>>>()
                }
            })
            .collect::<PyResult<Vec<_>>>()
    }

    fn parse_usize_pair(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<(usize, usize)> {
        let items = py_sequence_items(obj, vm, "expected pair of integers")?;
        if items.len() != 2 {
            return Err(vm.new_value_error("pad_width tuples must have 2 elements".to_owned()));
        }
        Ok((
            items[0].clone().try_into_value::<usize>(vm)?,
            items[1].clone().try_into_value::<usize>(vm)?,
        ))
    }

    fn parse_f64_pair(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<(f64, f64)> {
        let items = py_sequence_items(obj, vm, "expected pair of floats")?;
        if items.len() != 2 {
            return Err(vm.new_value_error("range tuples must have 2 elements".to_owned()));
        }
        Ok((
            items[0].clone().try_into_value::<f64>(vm)?,
            items[1].clone().try_into_value::<f64>(vm)?,
        ))
    }

    fn parse_gradient_axes(
        axes: &vm::PyRef<vm::builtins::PyList>,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<usize>> {
        axes.borrow_vec()
            .iter()
            .map(|obj| obj.clone().try_into_value::<i64>(vm).map(|v| v as usize))
            .collect::<PyResult<Vec<_>>>()
            .map_err(|_| vm.new_type_error("axis must be an integer".to_string()))
    }

    fn parse_gradient_spacings(
        spacings: &vm::PyRef<vm::builtins::PyList>,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<numpy_rust_core::ops::numerical::GradientSpacing>> {
        use numpy_rust_core::ops::numerical::GradientSpacing;

        spacings
            .borrow_vec()
            .iter()
            .map(|obj| {
                if let Some(arr_ref) = obj.downcast_ref::<PyNdArray>() {
                    let arr_inner = arr_ref.inner();
                    let cast = arr_inner.astype(numpy_rust_core::DType::Float64);
                    if let numpy_rust_core::ArrayData::Float64(a) = cast.data() {
                        let coords: Vec<f64> = a.iter().copied().collect();
                        if coords.len() <= 1 {
                            Ok(GradientSpacing::Uniform(
                                coords.first().copied().unwrap_or(1.0),
                            ))
                        } else {
                            Ok(GradientSpacing::Coordinates(coords))
                        }
                    } else {
                        Ok(GradientSpacing::Uniform(1.0))
                    }
                } else if let Ok(v) = obj.clone().try_into_value::<f64>(vm) {
                    Ok(GradientSpacing::Uniform(v))
                } else if let Ok(v) = obj.clone().try_into_value::<i64>(vm) {
                    Ok(GradientSpacing::Uniform(v as f64))
                } else {
                    Ok(GradientSpacing::Uniform(1.0))
                }
            })
            .collect()
    }

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
        Ok(PyNdArray::from_core(
            numpy_rust_core::creation::eye(n, m_val, k_val, dt)
                .map_err(|e| py_array::numpy_err(e, vm))?,
        ))
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
        if dt1 == numpy_rust_core::DType::Str || dt2 == numpy_rust_core::DType::Str {
            return Err(
                vm.new_type_error("Cannot promote string dtype with numeric dtype".to_owned())
            );
        }
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
        } else {
            numpy_rust_core::SplitSpec::Indices(parse_usize_values(
                &indices_or_sections,
                vm,
                "indices_or_sections must be int or list/tuple",
            )?)
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
        let reps_vec = parse_usize_values(&reps, vm, "reps must be int, tuple, or list")?;
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
    fn arcsinh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().arcsinh())
    }

    #[pyfunction]
    fn arccosh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().arccosh())
    }

    #[pyfunction]
    fn arctanh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().arctanh())
    }

    #[pyfunction]
    fn trunc(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .trunc()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
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

    // --- libm-backed unary functions ---

    #[pyfunction]
    fn cbrt(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .cbrt()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn gamma(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .gamma()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn lgamma(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .lgamma()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn erf(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .erf()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn erfc(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .erfc()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn j0(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .j0()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn j1(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .j1()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn y0(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .y0()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn y1(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .y1()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Binary math functions ---

    #[pyfunction]
    fn copysign(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.copysign(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn hypot(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.hypot(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn fmod(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.fmod(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn ldexp(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.ldexp(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn frexp(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<(PyNdArray, PyNdArray)> {
        let (mantissa, exponent) = a
            .inner()
            .frexp()
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        Ok((
            PyNdArray::from_core(mantissa),
            PyNdArray::from_core(exponent),
        ))
    }

    #[pyfunction]
    fn modf(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<(PyNdArray, PyNdArray)> {
        let (frac, int_part) = a
            .inner()
            .modf()
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        Ok((PyNdArray::from_core(frac), PyNdArray::from_core(int_part)))
    }

    #[pyfunction]
    fn nextafter(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.nextafter(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nan_to_num(
        a: vm::PyRef<PyNdArray>,
        nan: f64,
        posinf: f64,
        neginf: f64,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(a.inner().nan_to_num(nan, posinf, neginf))
    }

    #[pyfunction]
    fn spacing(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .spacing()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn i0(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().i0())
    }

    // --- Scimath complex-safe functions ---

    #[pyfunction]
    fn scimath_sqrt(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_sqrt())
    }

    #[pyfunction]
    fn scimath_log(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_log())
    }

    #[pyfunction]
    fn scimath_log2(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_log2())
    }

    #[pyfunction]
    fn scimath_log10(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_log10())
    }

    #[pyfunction]
    fn scimath_arcsin(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_arcsin())
    }

    #[pyfunction]
    fn scimath_arccos(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_arccos())
    }

    #[pyfunction]
    fn scimath_arctanh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_arctanh())
    }

    #[pyfunction]
    fn scimath_power(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.scimath_power(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn logaddexp(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.logaddexp(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn logaddexp2(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.logaddexp2(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn maximum(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.maximum(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn minimum(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.minimum(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn fmax(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.fmax(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn fmin(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.fmin(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn bitwise_and(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.bitwise_and(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn bitwise_or(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.bitwise_or(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn bitwise_xor(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.bitwise_xor(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn left_shift(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.left_shift(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn right_shift(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.right_shift(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn bitwise_not(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_ndarray(&a, vm)?;
        arr.bitwise_not()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn invert(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_ndarray(&a, vm)?;
        arr.bitwise_not()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn bitwise_count(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_ndarray(&a, vm)?;
        arr.bitwise_count()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn packbits(
        a: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        bitorder: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_ndarray(&a, vm)?;
        let little = match bitorder {
            vm::function::OptionalArg::Present(s) => match s.as_str() {
                "big" => false,
                "little" => true,
                other => {
                    return Err(vm.new_value_error(format!(
                        "bitorder must be either 'little' or 'big', got '{}'",
                        other
                    )))
                }
            },
            vm::function::OptionalArg::Missing => false,
        };
        arr.packbits(axis.into_option(), little)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn unpackbits(
        a: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        count: vm::function::OptionalArg<i64>,
        bitorder: vm::function::OptionalArg<vm::PyRef<vm::builtins::PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_ndarray(&a, vm)?;
        let little = match bitorder {
            vm::function::OptionalArg::Present(s) => match s.as_str() {
                "big" => false,
                "little" => true,
                other => {
                    return Err(vm.new_value_error(format!(
                        "bitorder must be either 'little' or 'big', got '{}'",
                        other
                    )))
                }
            },
            vm::function::OptionalArg::Missing => false,
        };
        arr.unpackbits(
            axis.into_option(),
            count.into_option().map(|v| v as isize),
            little,
        )
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn logical_and(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.logical_and(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn logical_or(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.logical_or(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn logical_xor(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.logical_xor(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
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
    fn putmask(
        a: vm::PyRef<PyNdArray>,
        mask: vm::PyRef<PyNdArray>,
        values: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let mut arr = a.inner().clone();
        arr.mask_set_repeat(&mask.inner(), &values.inner())
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        a.replace_inner(arr);
        Ok(())
    }

    #[pyfunction]
    fn choose(
        a: vm::PyRef<PyNdArray>,
        choices: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arrays = extract_core_array_sequence(&choices, vm).map_err(|_| {
            vm.new_type_error("choose requires a list or tuple of array-like values".to_owned())
        })?;
        let refs: Vec<&numpy_rust_core::NdArray> = arrays.iter().collect();
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
        range: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let bins = bins.unwrap_or(10);
        let range = match range {
            vm::function::OptionalArg::Present(obj) => Some(parse_f64_pair(&obj, vm)?),
            vm::function::OptionalArg::Missing => None,
        };
        let (counts, edges) = a
            .inner()
            .histogram(bins, range)
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        let py_counts = PyNdArray::from_core(counts).into_pyobject(vm);
        let py_edges = PyNdArray::from_core(edges).into_pyobject(vm);
        Ok(vm.ctx.new_tuple(vec![py_counts, py_edges]).into())
    }

    #[pyfunction]
    fn histogramdd_counts(
        sample: vm::PyRef<PyNdArray>,
        edges: PyObjectRef,
        weights: vm::function::OptionalArg<vm::PyRef<PyNdArray>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let edge_vecs = parse_float_edge_sequence(&edges, vm)?;
        let sample_inner = sample.inner();
        let counts = match weights.into_option() {
            Some(weights_arr) => {
                let weights_inner = weights_arr.inner();
                sample_inner
                    .histogramdd_counts(&edge_vecs, Some(&weights_inner))
                    .map_err(|e| vm.new_value_error(e.to_string()))?
            }
            None => sample_inner
                .histogramdd_counts(&edge_vecs, None)
                .map_err(|e| vm.new_value_error(e.to_string()))?,
        };
        Ok(PyNdArray::from_core(counts))
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

    #[pyfunction]
    fn weighted_inverted_cdf_quantile(
        a: vm::PyRef<PyNdArray>,
        q: f64,
        weights: vm::PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        a.inner()
            .weighted_inverted_cdf_quantile(q, axis, &weights.inner())
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Correlation / Covariance ---

    #[pyfunction]
    fn cov(
        m: vm::PyRef<PyNdArray>,
        y: PyObjectRef,
        rowvar: vm::function::OptionalArg<bool>,
        ddof: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let rowvar = rowvar.unwrap_or(true);
        let ddof = ddof.unwrap_or(1);
        let m_clone = m.inner().clone();
        let y_clone: Option<NdArray> = if vm.is_none(&y) {
            None
        } else {
            let y_arr: vm::PyRef<PyNdArray> = y.try_into_value(vm)?;
            let cloned = y_arr.inner().clone();
            Some(cloned)
        };
        numpy_rust_core::ops::correlation::cov_xy(&m_clone, y_clone.as_ref(), rowvar, ddof)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn corrcoef(
        x: vm::PyRef<PyNdArray>,
        y: PyObjectRef,
        rowvar: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let rowvar = rowvar.unwrap_or(true);
        let x_clone = x.inner().clone();
        let y_clone: Option<NdArray> = if vm.is_none(&y) {
            None
        } else {
            let y_arr: vm::PyRef<PyNdArray> = y.try_into_value(vm)?;
            let cloned = y_arr.inner().clone();
            Some(cloned)
        };
        numpy_rust_core::ops::correlation::corrcoef_xy(&x_clone, y_clone.as_ref(), rowvar)
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
        let idx = parse_indices_arg(&indices, vm)?;

        let data = a.inner();
        let ax = parse_optional_axis(axis, vm)?;
        data.take(&idx, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Index Utilities ---

    fn parse_shape_tuple(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
        parse_usize_values(obj, vm, "shape must be tuple, list, or int")
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
        let arrs = parse_index_array_sequence(&multi_index, vm)?;
        let refs: Vec<&numpy_rust_core::NdArray> = arrs.iter().collect();
        let dims_vec = parse_shape_tuple(&dims, vm)?;
        numpy_rust_core::indexing::ravel_multi_index(&refs, &dims_vec)
            .map(|arr| py_array::ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Stacking helpers ---

    #[pyfunction]
    fn stack_native(
        arrays: PyObjectRef,
        axis: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = axis.unwrap_or(0);
        let arr_list = extract_core_array_sequence(&arrays, vm)?;
        let refs: Vec<&numpy_rust_core::NdArray> = arr_list.iter().collect();
        numpy_rust_core::stack(&refs, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn column_stack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr_list = extract_core_array_sequence(&arrays, vm)?;
        let refs: Vec<&numpy_rust_core::NdArray> = arr_list.iter().collect();
        numpy_rust_core::column_stack(&refs)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn dstack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr_list = extract_core_array_sequence(&arrays, vm)?;
        let refs: Vec<&numpy_rust_core::NdArray> = arr_list.iter().collect();
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
        let arr_list = extract_core_array_sequence(&arrays, vm)?;
        let refs: Vec<&numpy_rust_core::NdArray> = arr_list.iter().collect();
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
        if let Ok(n) = obj.clone().try_into_value::<usize>(vm) {
            return Ok(vec![(n, n); ndim]);
        }

        if let Ok(pair) = parse_usize_pair(obj, vm) {
            return Ok(vec![pair; ndim]);
        }

        let items = py_sequence_items(
            obj,
            vm,
            "pad_width must be int, (int, int), or tuple/list of (int, int)",
        )?;
        items
            .into_iter()
            .map(|item| parse_usize_pair(&item, vm))
            .collect::<PyResult<Vec<_>>>()
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
    fn interp_with_options(
        x: vm::PyRef<PyNdArray>,
        xp: vm::PyRef<PyNdArray>,
        fp: vm::PyRef<PyNdArray>,
        left: vm::function::OptionalArg<f64>,
        right: vm::function::OptionalArg<f64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        numpy_rust_core::ops::numerical::interp_with_options(
            &x.inner(),
            &xp.inner(),
            &fp.inner(),
            left.into_option(),
            right.into_option(),
        )
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn interp_periodic(
        x: vm::PyRef<PyNdArray>,
        xp: vm::PyRef<PyNdArray>,
        fp: vm::PyRef<PyNdArray>,
        period: f64,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        numpy_rust_core::ops::numerical::interp_periodic(
            &x.inner(),
            &xp.inner(),
            &fp.inner(),
            period,
        )
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
    }

    /// gradient(f, spacings, edge_order, axes) -> list of arrays
    #[pyfunction]
    fn gradient(
        f: vm::PyRef<PyNdArray>,
        spacings: vm::PyRef<vm::builtins::PyList>,
        edge_order: i64,
        axes: vm::PyRef<vm::builtins::PyList>,
        vm: &VirtualMachine,
    ) -> PyResult<vm::PyRef<vm::builtins::PyList>> {
        let f_inner = f.inner();
        let edge_order = edge_order as usize;
        let axes_vec = parse_gradient_axes(&axes, vm)?;
        let sp_vec = parse_gradient_spacings(&spacings, vm)?;

        let results = numpy_rust_core::ops::numerical::gradient_full(
            &f_inner, &sp_vec, edge_order, &axes_vec,
        )
        .map_err(|e| vm.new_value_error(e.to_string()))?;

        let py_list: Vec<vm::PyObjectRef> = results
            .into_iter()
            .map(|arr| PyNdArray::from_core(arr).into_pyobject(vm))
            .collect();
        Ok(vm::builtins::PyList::from(py_list).into_ref(&vm.ctx))
    }

    #[pyfunction]
    fn trapz(
        y: vm::PyRef<PyNdArray>,
        dx: vm::function::OptionalArg<f64>,
        axis: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dx_val = dx.unwrap_or(1.0);
        let axis_val = axis.into_option();
        numpy_rust_core::ops::numerical::trapz(&y.inner(), None, dx_val, axis_val)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn trapz_x(
        y: vm::PyRef<PyNdArray>,
        x: vm::PyRef<PyNdArray>,
        dx: vm::function::OptionalArg<f64>,
        axis: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dx_val = dx.unwrap_or(1.0);
        let axis_val = axis.into_option();
        let x_inner = x.inner();
        numpy_rust_core::ops::numerical::trapz(&y.inner(), Some(&x_inner), dx_val, axis_val)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn cumulative_trapezoid(
        y: vm::PyRef<PyNdArray>,
        dx: vm::function::OptionalArg<f64>,
        axis: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dx_val = dx.unwrap_or(1.0);
        let axis_val = axis.into_option();
        numpy_rust_core::ops::numerical::cumulative_trapezoid(&y.inner(), None, dx_val, axis_val)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn cumulative_trapezoid_x(
        y: vm::PyRef<PyNdArray>,
        x: vm::PyRef<PyNdArray>,
        dx: vm::function::OptionalArg<f64>,
        axis: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dx_val = dx.unwrap_or(1.0);
        let axis_val = axis.into_option();
        let x_inner = x.inner();
        numpy_rust_core::ops::numerical::cumulative_trapezoid(
            &y.inner(),
            Some(&x_inner),
            dx_val,
            axis_val,
        )
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
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
