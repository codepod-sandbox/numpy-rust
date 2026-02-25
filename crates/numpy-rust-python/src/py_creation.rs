use rustpython_vm as vm;
use vm::builtins::PyList;
use vm::{PyObjectRef, PyRef, PyResult, VirtualMachine};

use numpy_rust_core::{DType, NdArray};

use crate::py_array::{extract_shape, PyNdArray};

/// numpy.array(data) — convert a Python list to an NdArray.
pub fn py_array(data: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    if let Some(list) = data.payload::<PyList>() {
        let items = list.borrow_vec();
        if items.is_empty() {
            return Ok(PyNdArray::from_core(NdArray::from_vec(Vec::<f64>::new())));
        }

        // Check if first element is a list (nested)
        if items[0].payload::<PyList>().is_some() {
            return parse_nested_list(&items, vm);
        }

        // Flat list — try to extract as floats
        let mut floats = Vec::with_capacity(items.len());
        for item in items.iter() {
            let f: f64 = item.clone().try_into_value(vm)?;
            floats.push(f);
        }
        Ok(PyNdArray::from_core(NdArray::from_vec(floats)))
    } else {
        Err(vm.new_type_error("array() requires a list".into()))
    }
}

fn parse_nested_list(items: &[PyObjectRef], vm: &VirtualMachine) -> PyResult<PyNdArray> {
    let nrows = items.len();
    let mut flat = Vec::new();
    let mut ncols = None;

    for item in items {
        let row = item
            .payload::<PyList>()
            .ok_or_else(|| vm.new_type_error("expected list of lists".into()))?;
        let row_items = row.borrow_vec();
        match ncols {
            None => ncols = Some(row_items.len()),
            Some(c) if c != row_items.len() => {
                return Err(vm.new_value_error("inconsistent row lengths".into()));
            }
            _ => {}
        }
        for elem in row_items.iter() {
            let f: f64 = elem.clone().try_into_value(vm)?;
            flat.push(f);
        }
    }

    let ncols = ncols.unwrap_or(0);
    let arr = NdArray::from_vec(flat)
        .reshape(&[nrows, ncols])
        .map_err(|e| vm.new_value_error(e.to_string()))?;
    Ok(PyNdArray::from_core(arr))
}

/// numpy.zeros(shape)
pub fn py_zeros(shape_obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    let shape = extract_shape(shape_obj, vm)?;
    Ok(PyNdArray::from_core(NdArray::zeros(&shape, DType::Float64)))
}

/// numpy.ones(shape)
pub fn py_ones(shape_obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    Ok(PyNdArray::from_core(NdArray::ones(
        &extract_shape(shape_obj, vm)?,
        DType::Float64,
    )))
}

/// numpy.concatenate(arrays, axis)
pub fn py_concatenate(
    arrays_obj: PyObjectRef,
    axis: usize,
    vm: &VirtualMachine,
) -> PyResult<PyNdArray> {
    let list = arrays_obj
        .payload::<PyList>()
        .ok_or_else(|| vm.new_type_error("concatenate requires a list of arrays".into()))?;
    let items = list.borrow_vec();
    let py_arrays: Vec<PyRef<PyNdArray>> = items
        .iter()
        .map(|item| item.clone().try_into_value::<PyRef<PyNdArray>>(vm))
        .collect::<PyResult<Vec<_>>>()?;
    let core_refs: Vec<&NdArray> = py_arrays.iter().map(|a| a.inner()).collect();
    numpy_rust_core::concatenate(&core_refs, axis)
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
}
