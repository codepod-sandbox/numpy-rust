use num_complex::Complex;
use rustpython_vm as vm;
use vm::builtins::{PyList, PyStr, PyTuple};
use vm::{AsObject, PyObjectRef, PyRef, PyResult, VirtualMachine};

use numpy_rust_core::indexing::Scalar;
use numpy_rust_core::{BoxedObjectScalar, BoxedScalar, BoxedTemporalScalar, DType, NdArray};

use crate::py_array::{extract_shape, parse_dtype, PyNdArray};

enum SequenceKind {
    Strings,
    Bools,
    Floats,
    Complexes,
    Objects,
    Datetimes,
    Timedeltas,
}

fn extract_temporal_boxed_scalar(
    obj: &vm::PyObject,
    dtype: DType,
    unit_hint: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<BoxedScalar> {
    if let Some(i) = obj.downcast_ref::<vm::builtins::PyInt>() {
        if let Ok(value) = i.try_to_primitive::<i64>(vm) {
            let scalar = BoxedTemporalScalar {
                value,
                unit: unit_hint.unwrap_or("ns").to_owned(),
                is_nat: false,
            };
            return Ok(match dtype {
                DType::Datetime64 => BoxedScalar::Datetime(scalar),
                DType::Timedelta64 => BoxedScalar::Timedelta(scalar),
                _ => unreachable!("temporal boxed scalar requires temporal dtype"),
            });
        }
    }
    let value = obj.get_attr("_value", vm)?.try_into_value::<i64>(vm)?;
    let unit = obj.get_attr("_unit", vm)?.try_into_value::<String>(vm)?;
    let is_nat = obj.get_attr("_is_nat", vm)?.try_into_value::<bool>(vm)?;
    let scalar = BoxedTemporalScalar {
        value,
        unit,
        is_nat,
    };
    Ok(match dtype {
        DType::Datetime64 => BoxedScalar::Datetime(scalar),
        DType::Timedelta64 => BoxedScalar::Timedelta(scalar),
        _ => unreachable!("temporal boxed scalar requires temporal dtype"),
    })
}

fn object_scalar_to_boxed(obj: &vm::PyObject, vm: &VirtualMachine) -> PyResult<BoxedScalar> {
    if obj.class().is(vm.ctx.types.bool_type) {
        return Ok(BoxedScalar::Object(BoxedObjectScalar::Bool(
            obj.to_owned().try_into_value::<bool>(vm)?,
        )));
    }
    if let Some(c) = obj.downcast_ref::<vm::builtins::PyComplex>() {
        let value = c.to_complex();
        return Ok(BoxedScalar::Object(BoxedObjectScalar::Complex(
            Complex::new(value.re, value.im),
        )));
    }
    if let Some(s) = obj.downcast_ref::<PyStr>() {
        return Ok(BoxedScalar::Object(BoxedObjectScalar::Text(
            s.as_str().to_owned(),
        )));
    }
    if let Ok(c) = obj
        .to_owned()
        .try_into_value::<vm::function::ArgIntoComplex>(vm)
    {
        let value = c.into_complex();
        if value.im != 0.0 {
            return Ok(BoxedScalar::Object(BoxedObjectScalar::Complex(
                Complex::new(value.re, value.im),
            )));
        }
    }
    if let Some(i) = obj.downcast_ref::<vm::builtins::PyInt>() {
        if let Ok(value) = i.try_to_primitive::<i64>(vm) {
            return Ok(BoxedScalar::Object(BoxedObjectScalar::Int(value)));
        }
    }
    if let Ok(f) = obj.to_owned().try_into_value::<f64>(vm) {
        return Ok(BoxedScalar::Object(BoxedObjectScalar::Float(f)));
    }
    Err(vm.new_type_error(format!(
        "object dtype does not yet support values of type {}",
        obj.class().name()
    )))
}

fn boxed_scalar_for_dtype(
    obj: &vm::PyObject,
    dtype: DType,
    temporal_unit: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<BoxedScalar> {
    match dtype {
        DType::Object => object_scalar_to_boxed(obj, vm),
        DType::Datetime64 => {
            extract_temporal_boxed_scalar(obj, DType::Datetime64, temporal_unit, vm)
        }
        DType::Timedelta64 => {
            extract_temporal_boxed_scalar(obj, DType::Timedelta64, temporal_unit, vm)
        }
        _ => Err(vm.new_type_error(format!(
            "boxed scalar conversion requires boxed dtype, got {dtype}"
        ))),
    }
}

fn infer_temporal_dtype(obj: &vm::PyObject, vm: &VirtualMachine) -> Option<DType> {
    if obj
        .get_attr("_is_datetime64", vm)
        .ok()
        .and_then(|v| v.try_into_value::<bool>(vm).ok())
        == Some(true)
    {
        return Some(DType::Datetime64);
    }
    if obj
        .get_attr("_is_timedelta64", vm)
        .ok()
        .and_then(|v| v.try_into_value::<bool>(vm).ok())
        == Some(true)
    {
        return Some(DType::Timedelta64);
    }
    None
}

fn boxed_scalar_ndarray_with_dtype(
    obj: &vm::PyObject,
    dtype: DType,
    temporal_unit: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    NdArray::from_boxed_scalars(
        vec![boxed_scalar_for_dtype(obj, dtype, temporal_unit, vm)?],
        &[],
        dtype,
    )
    .map_err(|e| vm.new_type_error(e.to_string()))
}

fn scalar_ndarray_with_dtype(
    obj: &vm::PyObject,
    dtype: Option<DType>,
    temporal_unit: Option<&str>,
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

    if let Some(dtype) = dtype {
        if dtype.is_boxed() {
            return boxed_scalar_ndarray_with_dtype(obj, dtype, temporal_unit, vm);
        }
    } else if let Some(temporal) = infer_temporal_dtype(obj, vm) {
        return boxed_scalar_ndarray_with_dtype(obj, temporal, temporal_unit, vm);
    }

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

    if let Some(i) = obj.downcast_ref::<vm::builtins::PyInt>() {
        if let Ok(value) = i.try_to_primitive::<i64>(vm) {
            let arr = NdArray::from_vec(vec![value])
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

fn classify_sequence(
    items: &[PyObjectRef],
    forced_dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyResult<SequenceKind> {
    if let Some(dtype) = forced_dtype {
        return match dtype {
            DType::Object => Ok(SequenceKind::Objects),
            DType::Datetime64 => Ok(SequenceKind::Datetimes),
            DType::Timedelta64 => Ok(SequenceKind::Timedeltas),
            _ => classify_sequence(items, None, vm),
        };
    }

    if items.is_empty() {
        return Ok(SequenceKind::Floats);
    }

    if items
        .iter()
        .all(|item| infer_temporal_dtype(item, vm) == Some(DType::Datetime64))
    {
        return Ok(SequenceKind::Datetimes);
    }

    if items
        .iter()
        .all(|item| infer_temporal_dtype(item, vm) == Some(DType::Timedelta64))
    {
        return Ok(SequenceKind::Timedeltas);
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
    forced_dtype: Option<DType>,
    temporal_unit: Option<&str>,
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
                .map(|item| object_to_scalar(item, vm))
                .collect::<PyResult<Vec<_>>>()?
                .into_iter()
                .map(|scalar| match scalar {
                    Scalar::Bool(value) => Ok(if value { 1.0 } else { 0.0 }),
                    Scalar::Int32(value) => Ok(value as f64),
                    Scalar::Int64(value) => Ok(value as f64),
                    Scalar::Float32(value) => Ok(value as f64),
                    Scalar::Float64(value) => Ok(value),
                    Scalar::Complex64(_) | Scalar::Complex128(_) => {
                        Err(vm.new_type_error("mixed types in sequence".to_owned()))
                    }
                    Scalar::Str(_) => Err(vm.new_type_error("mixed types in sequence".to_owned())),
                })
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
        SequenceKind::Objects => NdArray::from_boxed_scalars(
            items
                .iter()
                .map(|item| boxed_scalar_for_dtype(item, DType::Object, temporal_unit, vm))
                .collect::<PyResult<Vec<_>>>()?,
            &[items.len()],
            DType::Object,
        )
        .map_err(|e| vm.new_type_error(e.to_string())),
        SequenceKind::Datetimes => {
            let dtype = forced_dtype.unwrap_or(DType::Datetime64);
            NdArray::from_boxed_scalars(
                items
                    .iter()
                    .map(|item| boxed_scalar_for_dtype(item, DType::Datetime64, temporal_unit, vm))
                    .collect::<PyResult<Vec<_>>>()?,
                &[items.len()],
                dtype,
            )
            .map_err(|e| vm.new_type_error(e.to_string()))
        }
        SequenceKind::Timedeltas => {
            let dtype = forced_dtype.unwrap_or(DType::Timedelta64);
            NdArray::from_boxed_scalars(
                items
                    .iter()
                    .map(|item| boxed_scalar_for_dtype(item, DType::Timedelta64, temporal_unit, vm))
                    .collect::<PyResult<Vec<_>>>()?,
                &[items.len()],
                dtype,
            )
            .map_err(|e| vm.new_type_error(e.to_string()))
        }
    }
}

fn flat_sequence_to_ndarray(
    items: &[PyObjectRef],
    dtype: Option<DType>,
    temporal_unit: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    let kind = classify_sequence(items, dtype, vm)?;
    build_sequence_array(items, kind, dtype, temporal_unit, vm)
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

fn nested_sequence_to_ndarray(
    items: &[PyObjectRef],
    dtype: Option<DType>,
    temporal_unit: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    let (flat, nrows, ncols) = flatten_nested_sequence(items, vm)?;
    let kind = classify_sequence(&flat, dtype, vm)?;
    build_sequence_array(&flat, kind, dtype, temporal_unit, vm)?
        .reshape(&[nrows, ncols])
        .map_err(|e| vm.new_value_error(e.to_string()))
}

pub(crate) fn object_to_ndarray_with_dtype_and_unit(
    data: &vm::PyObject,
    dtype: Option<DType>,
    temporal_unit: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    if let Some(arr) = data.downcast_ref::<PyNdArray>() {
        return Ok(match dtype {
            Some(target) if target != arr.inner().dtype() => arr.inner().astype(target),
            _ => arr.inner().clone(),
        });
    }

    if let Some(list) = data.downcast_ref::<PyList>() {
        let items = list.borrow_vec();
        if !items.is_empty()
            && (items[0].downcast_ref::<PyList>().is_some()
                || items[0].downcast_ref::<PyTuple>().is_some())
        {
            return nested_sequence_to_ndarray(&items, dtype, temporal_unit, vm);
        }
        return flat_sequence_to_ndarray(&items, dtype, temporal_unit, vm);
    }

    if let Some(tuple) = data.downcast_ref::<PyTuple>() {
        let items = tuple.as_slice();
        if !items.is_empty()
            && (items[0].downcast_ref::<PyList>().is_some()
                || items[0].downcast_ref::<PyTuple>().is_some())
        {
            return nested_sequence_to_ndarray(items, dtype, temporal_unit, vm);
        }
        return flat_sequence_to_ndarray(items, dtype, temporal_unit, vm);
    }

    scalar_ndarray_with_dtype(data, dtype, temporal_unit, vm)
}

pub fn object_to_ndarray_with_dtype(
    data: &vm::PyObject,
    dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    object_to_ndarray_with_dtype_and_unit(data, dtype, None, vm)
}

pub fn object_to_ndarray(data: &vm::PyObject, vm: &VirtualMachine) -> PyResult<NdArray> {
    object_to_ndarray_with_dtype(data, None, vm)
}

pub fn is_array_like_object(obj: &vm::PyObject, vm: &VirtualMachine) -> bool {
    obj.downcast_ref::<PyNdArray>().is_some() || obj.get_attr("_numpy_dtype_name", vm).is_ok()
}

pub fn object_to_ndarray_weak(
    obj: &vm::PyObject,
    target_dtype: DType,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    if target_dtype.is_boxed() {
        return object_to_ndarray_with_dtype(obj, Some(target_dtype), vm);
    }
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
pub fn py_array(
    data: PyObjectRef,
    dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    object_to_ndarray_with_dtype(&data, dtype, vm).map(PyNdArray::from_core)
}

fn extract_temporal_unit(dtype_name: &str) -> Option<&str> {
    let start = dtype_name.find('[')?;
    let end = dtype_name[start + 1..].find(']')?;
    Some(&dtype_name[start + 1..start + 1 + end])
}

pub fn py_array_with_dtype_name(
    data: PyObjectRef,
    dtype_name: &str,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let dtype = parse_dtype(dtype_name, vm)?;
    let temporal_unit = if matches!(dtype, DType::Datetime64 | DType::Timedelta64) {
        extract_temporal_unit(dtype_name)
    } else {
        None
    };
    object_to_ndarray_with_dtype_and_unit(&data, Some(dtype), temporal_unit, vm)
        .map(PyNdArray::from_core)
}

fn temporal_fill_array(
    shape: &[usize],
    dtype: DType,
    unit: Option<&str>,
    value: i64,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    let total = shape.iter().product::<usize>();
    let scalar = BoxedTemporalScalar {
        value,
        unit: unit.unwrap_or("generic").to_owned(),
        is_nat: false,
    };
    let elements = match dtype {
        DType::Datetime64 => vec![BoxedScalar::Datetime(scalar); total],
        DType::Timedelta64 => vec![BoxedScalar::Timedelta(scalar); total],
        _ => unreachable!("temporal fill array requires temporal dtype"),
    };
    NdArray::from_boxed_scalars(elements, shape, dtype)
        .map_err(|e| vm.new_type_error(e.to_string()))
}

/// numpy.zeros(shape, dtype)
pub fn py_zeros(
    shape_obj: &PyObjectRef,
    dtype_name: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let shape = extract_shape(shape_obj, vm)?;
    let (dtype, temporal_unit) = match dtype_name {
        Some(name) => {
            let parsed = parse_dtype(name, vm)?;
            let unit = if matches!(parsed, DType::Datetime64 | DType::Timedelta64) {
                extract_temporal_unit(name)
            } else {
                None
            };
            (parsed, unit)
        }
        None => (DType::Float64, None),
    };
    let arr = match dtype {
        DType::Datetime64 | DType::Timedelta64 => {
            temporal_fill_array(&shape, dtype, temporal_unit, 0, vm)?
        }
        _ => NdArray::zeros(&shape, dtype),
    };
    Ok(PyNdArray::from_core(arr))
}

/// numpy.ones(shape, dtype)
pub fn py_ones(
    shape_obj: &PyObjectRef,
    dtype_name: Option<&str>,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let shape = extract_shape(shape_obj, vm)?;
    let (dtype, temporal_unit) = match dtype_name {
        Some(name) => {
            let parsed = parse_dtype(name, vm)?;
            let unit = if matches!(parsed, DType::Datetime64 | DType::Timedelta64) {
                extract_temporal_unit(name)
            } else {
                None
            };
            (parsed, unit)
        }
        None => (DType::Float64, None),
    };
    let arr = match dtype {
        DType::Datetime64 | DType::Timedelta64 => {
            temporal_fill_array(&shape, dtype, temporal_unit, 1, vm)?
        }
        _ => NdArray::ones(&shape, dtype),
    };
    Ok(PyNdArray::from_core(arr))
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
