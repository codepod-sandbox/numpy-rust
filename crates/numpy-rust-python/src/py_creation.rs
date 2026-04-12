use num_complex::Complex;
use rustpython_vm as vm;
use vm::builtins::{PyList, PyStr, PyTuple};
use vm::{AsObject, PyObjectRef, PyRef, PyResult, VirtualMachine};

use numpy_rust_core::indexing::Scalar;
use numpy_rust_core::{DType, NdArray};

use crate::py_array::{extract_shape, parse_dtype, PyNdArray};

enum SequenceKind {
    Strings,
    Bools,
    Floats,
    Complexes,
}

fn scalar_ndarray_with_dtype(
    obj: &vm::PyObject,
    dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    let dtype = if let Some(dtype) = dtype {
        Some(dtype)
    } else if let Ok(dtype_name_obj) = obj.get_attr("_numpy_dtype_name", vm) {
        let dtype_name = dtype_name_obj.try_into_value::<String>(vm)?;
        Some(parse_dtype(dtype_name.as_str(), vm)?)
    } else {
        None
    };

    if obj.class().is(vm.ctx.types.bool_type) {
        let value = obj.to_owned().try_into_value::<bool>(vm)?;
        let arr = NdArray::from_vec(vec![value])
            .reshape(&[])
            .map_err(|e| vm.new_type_error(e.to_string()))?;
        return Ok(match dtype {
            Some(dt) if dt != DType::Bool => arr.astype(dt),
            _ => arr,
        });
    }

    if let Some(s) = obj.downcast_ref::<vm::builtins::PyStr>() {
        let arr = NdArray::from_vec(vec![s.as_str().to_owned()])
            .reshape(&[])
            .map_err(|e| vm.new_type_error(e.to_string()))?;
        return Ok(match dtype {
            Some(dt) => arr.astype(dt),
            None => arr,
        });
    }

    let wants_complex = matches!(dtype, Some(dt) if dt.is_complex())
        || obj.downcast_ref::<vm::builtins::PyComplex>().is_some();
    if wants_complex {
        if let Ok(c) = obj
            .to_owned()
            .try_into_value::<vm::function::ArgIntoComplex>(vm)
        {
            let value = c.into_complex();
            let arr = NdArray::from_complex128_vec(vec![Complex::new(value.re, value.im)])
                .reshape(&[])
                .map_err(|e| vm.new_type_error(e.to_string()))?;
            return Ok(match dtype {
                Some(dt) => arr.astype(dt),
                None => arr,
            });
        }
    }

    if let Ok(f) = obj.to_owned().try_into_value::<f64>(vm) {
        let arr = NdArray::from_scalar(f);
        return Ok(match dtype {
            Some(dt) => arr.astype(dt),
            None => arr,
        });
    }

    Err(vm.new_type_error(format!("cannot convert {} to array", obj.class().name())))
}

fn classify_sequence(items: &[PyObjectRef], vm: &VirtualMachine) -> PyResult<SequenceKind> {
    if items.is_empty() {
        return Ok(SequenceKind::Floats);
    }

    let has_string = items
        .iter()
        .any(|item| item.downcast_ref::<PyStr>().is_some());
    if has_string {
        if items
            .iter()
            .all(|item| item.downcast_ref::<PyStr>().is_some())
        {
            return Ok(SequenceKind::Strings);
        }
        return Err(vm.new_type_error("mixed types in sequence".to_owned()));
    }

    let scalars = items
        .iter()
        .map(|item| object_to_scalar(item, vm))
        .collect::<PyResult<Vec<_>>>()?;

    if scalars
        .iter()
        .all(|scalar| matches!(scalar, Scalar::Bool(_)))
    {
        return Ok(SequenceKind::Bools);
    }

    if scalars
        .iter()
        .any(|scalar| matches!(scalar, Scalar::Complex64(_) | Scalar::Complex128(_)))
    {
        return Ok(SequenceKind::Complexes);
    }

    Ok(SequenceKind::Floats)
}

fn build_sequence_array(
    items: &[PyObjectRef],
    kind: SequenceKind,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    match kind {
        SequenceKind::Strings => {
            let values = items
                .iter()
                .map(|item| {
                    item.downcast_ref::<PyStr>()
                        .map(|s| s.as_str().to_owned())
                        .ok_or_else(|| vm.new_type_error("mixed types in sequence".to_owned()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(NdArray::from_vec(values))
        }
        SequenceKind::Bools => {
            let values = items
                .iter()
                .map(|item| object_to_scalar(item, vm))
                .collect::<PyResult<Vec<_>>>()?
                .into_iter()
                .map(|scalar| match scalar {
                    Scalar::Bool(value) => Ok(value),
                    _ => Err(vm.new_type_error("mixed types in sequence".to_owned())),
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(NdArray::from_vec(values))
        }
        SequenceKind::Floats => {
            let values = items
                .iter()
                .map(|item| item.clone().try_into_value::<f64>(vm))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(NdArray::from_vec(values))
        }
        SequenceKind::Complexes => {
            let values = items
                .iter()
                .map(|item| object_to_scalar(item, vm))
                .collect::<PyResult<Vec<_>>>()?
                .into_iter()
                .map(|scalar| match scalar {
                    Scalar::Bool(value) => Ok(Complex::new(if value { 1.0 } else { 0.0 }, 0.0)),
                    Scalar::Int32(value) => Ok(Complex::new(value as f64, 0.0)),
                    Scalar::Int64(value) => Ok(Complex::new(value as f64, 0.0)),
                    Scalar::Float32(value) => Ok(Complex::new(value as f64, 0.0)),
                    Scalar::Float64(value) => Ok(Complex::new(value, 0.0)),
                    Scalar::Complex64(value) => Ok(Complex::new(value.re as f64, value.im as f64)),
                    Scalar::Complex128(value) => Ok(value),
                    Scalar::Str(_) => Err(vm.new_type_error("mixed types in sequence".to_owned())),
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(NdArray::from_complex128_vec(values))
        }
    }
}

fn flat_sequence_to_ndarray(items: &[PyObjectRef], vm: &VirtualMachine) -> PyResult<NdArray> {
    let kind = classify_sequence(items, vm)?;
    build_sequence_array(items, kind, vm)
}

fn flatten_nested_sequence(
    items: &[PyObjectRef],
    vm: &VirtualMachine,
) -> PyResult<(Vec<PyObjectRef>, usize, usize)> {
    let nrows = items.len();
    let mut flat = Vec::new();
    let mut ncols = None;

    for item in items {
        let row = if let Some(list) = item.downcast_ref::<PyList>() {
            list.borrow_vec().to_vec()
        } else if let Some(tuple) = item.downcast_ref::<PyTuple>() {
            tuple.as_slice().to_vec()
        } else {
            return Err(vm.new_type_error("expected sequence of sequences".to_owned()));
        };

        match ncols {
            None => ncols = Some(row.len()),
            Some(c) if c != row.len() => {
                return Err(vm.new_value_error("inconsistent row lengths".to_owned()));
            }
            _ => {}
        }

        for elem in row {
            flat.push(elem);
        }
    }

    Ok((flat, nrows, ncols.unwrap_or(0)))
}

fn nested_sequence_to_ndarray(items: &[PyObjectRef], vm: &VirtualMachine) -> PyResult<NdArray> {
    let (flat, nrows, ncols) = flatten_nested_sequence(items, vm)?;
    let kind = classify_sequence(&flat, vm)?;
    build_sequence_array(&flat, kind, vm)?
        .reshape(&[nrows, ncols])
        .map_err(|e| vm.new_value_error(e.to_string()))
}

pub fn object_to_ndarray(data: &vm::PyObject, vm: &VirtualMachine) -> PyResult<NdArray> {
    if let Some(arr) = data.downcast_ref::<PyNdArray>() {
        return Ok(arr.inner().clone());
    }

    if let Some(list) = data.downcast_ref::<PyList>() {
        let items = list.borrow_vec();
        if !items.is_empty()
            && (items[0].downcast_ref::<PyList>().is_some()
                || items[0].downcast_ref::<PyTuple>().is_some())
        {
            return nested_sequence_to_ndarray(&items, vm);
        }
        return flat_sequence_to_ndarray(&items, vm);
    }

    if let Some(tuple) = data.downcast_ref::<PyTuple>() {
        let items = tuple.as_slice();
        if !items.is_empty()
            && (items[0].downcast_ref::<PyList>().is_some()
                || items[0].downcast_ref::<PyTuple>().is_some())
        {
            return nested_sequence_to_ndarray(items, vm);
        }
        return flat_sequence_to_ndarray(items, vm);
    }

    scalar_ndarray_with_dtype(data, None, vm)
}

pub fn is_array_like_object(obj: &vm::PyObject, vm: &VirtualMachine) -> bool {
    obj.downcast_ref::<PyNdArray>().is_some() || obj.get_attr("_numpy_dtype_name", vm).is_ok()
}

pub fn object_to_ndarray_weak(
    obj: &vm::PyObject,
    target_dtype: DType,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    if is_array_like_object(obj, vm) || obj.class().is(vm.ctx.types.bool_type) {
        return object_to_ndarray(obj, vm);
    }

    if obj.class().is(vm.ctx.types.float_type) {
        if target_dtype.is_float() || target_dtype.is_complex() {
            if let Ok(value) = obj.to_owned().try_into_value::<f64>(vm) {
                return Ok(NdArray::from_scalar(value).astype(target_dtype));
            }
        }
        return object_to_ndarray(obj, vm);
    }

    if obj.class().is(vm.ctx.types.int_type) {
        if let Some(value) = obj.downcast_ref::<vm::builtins::PyInt>() {
            if let Ok(int_value) = value.try_to_primitive::<i64>(vm) {
                return Ok(NdArray::from_scalar(int_value as f64).astype(target_dtype));
            }
        }
    }

    object_to_ndarray(obj, vm)
}

pub fn object_to_scalar(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Scalar> {
    if let Some(tuple) = obj.downcast_ref::<PyTuple>() {
        let elems = tuple.as_slice();
        if elems.len() == 2 {
            let re = elems[0].clone().try_into_value::<f64>(vm).unwrap_or(0.0);
            let im = elems[1].clone().try_into_value::<f64>(vm).unwrap_or(0.0);
            return Ok(Scalar::Complex128(Complex::new(re, im)));
        }
    }

    let arr = object_to_ndarray(obj, vm)?;
    if arr.size() != 1 {
        return Err(vm.new_type_error(format!(
            "cannot convert value to array scalar (shape={:?}, size={})",
            arr.shape(),
            arr.size()
        )));
    }

    let coord = vec![0; arr.ndim()];
    arr.get(&coord)
        .map_err(|e| vm.new_type_error(e.to_string()))
}

/// numpy.array(data) — convert a Python list to an NdArray.
pub fn py_array(data: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    object_to_ndarray(&data, vm).map(PyNdArray::from_core)
}

/// numpy.zeros(shape, dtype)
pub fn py_zeros(
    shape_obj: &PyObjectRef,
    dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let shape = extract_shape(shape_obj, vm)?;
    Ok(PyNdArray::from_core(NdArray::zeros(
        &shape,
        dtype.unwrap_or(DType::Float64),
    )))
}

/// numpy.ones(shape, dtype)
pub fn py_ones(
    shape_obj: &PyObjectRef,
    dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    Ok(PyNdArray::from_core(NdArray::ones(
        &extract_shape(shape_obj, vm)?,
        dtype.unwrap_or(DType::Float64),
    )))
}

/// numpy.concatenate(arrays, axis)
pub fn py_concatenate(
    arrays_obj: PyObjectRef,
    axis: usize,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let list = arrays_obj
        .downcast_ref::<PyList>()
        .ok_or_else(|| vm.new_type_error("concatenate requires a list of arrays".to_owned()))?;
    let items = list.borrow_vec();
    let py_arrays: Vec<PyRef<PyNdArray>> = items
        .iter()
        .map(|item| item.clone().try_into_value::<PyRef<PyNdArray>>(vm))
        .collect::<PyResult<Vec<_>>>()?;
    let borrowed: Vec<std::sync::RwLockReadGuard<'_, NdArray>> =
        py_arrays.iter().map(|a| a.inner()).collect();
    let core_refs: Vec<&NdArray> = borrowed.iter().map(|r| &**r).collect();
    numpy_rust_core::concatenate(&core_refs, axis)
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
}
