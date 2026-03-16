use std::sync::RwLock;

use rustpython_vm as vm;
use vm::builtins::PyStr;
use vm::protocol::PyMappingMethods;
use vm::types::AsMapping;
use vm::{atomic_func, AsObject, PyObjectRef, PyPayload, PyResult, VirtualMachine};

use numpy_rust_core::{FieldSpec, NdArray, StructArrayData};

use crate::py_array::{numpy_err, scalar_to_py, PyNdArray};

/// Python-visible structured array class backed by columnar Rust storage.
/// Each field is stored as a separate ArrayData column.
/// Constructor: _native.StructuredArray(fields, shape, dtype_json)
///   fields: list of [name_str, PyNdArray]
///   shape: list of int
///   dtype_json: JSON string [["x","float64"],["y","int32"]]
#[vm::pyclass(module = "numpy", name = "StructuredArray")]
#[derive(Debug, PyPayload)]
pub struct PyStructuredArray {
    inner: RwLock<StructArrayData>,
    /// JSON string: [["x","float64"],["y","int32"]]
    /// Returned verbatim by .dtype so Python can json.loads() it.
    dtype_json: String,
}

impl PyStructuredArray {
    pub fn new_from_data(sa: StructArrayData, dtype_json: String) -> Self {
        Self {
            inner: RwLock::new(sa),
            dtype_json,
        }
    }
}

#[vm::pyclass(with(AsMapping))]
impl PyStructuredArray {
    #[pyslot]
    fn slot_new(
        cls: vm::builtins::PyTypeRef,
        args: vm::function::FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult {
        if args.args.len() < 3 {
            return Err(vm.new_type_error(
                "StructuredArray requires 3 positional args: fields, shape, dtype_json".to_owned(),
            ));
        }
        let fields_obj = args.args[0].clone();
        let shape_obj = args.args[1].clone();
        let dtype_json_obj = args.args[2].clone();

        // Parse shape: list of int
        let shape: Vec<usize> = {
            let list = shape_obj.try_into_value::<Vec<PyObjectRef>>(vm)?;
            list.iter()
                .map(|o| o.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        };
        // Parse dtype_json string
        let dtype_json = dtype_json_obj
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("dtype_json must be a str".to_owned()))?
            .as_str()
            .to_owned();
        // Parse fields: list of [name_str, PyNdArray]
        let fields_list = fields_obj.try_into_value::<Vec<PyObjectRef>>(vm)?;
        let mut fields: Vec<FieldSpec> = Vec::with_capacity(fields_list.len());
        for item in fields_list {
            let tup = item.try_into_value::<Vec<PyObjectRef>>(vm)?;
            if tup.len() != 2 {
                return Err(
                    vm.new_value_error("each field must be a [name, ndarray] pair".to_owned())
                );
            }
            let name = tup[0]
                .downcast_ref::<PyStr>()
                .ok_or_else(|| vm.new_type_error("field name must be a str".to_owned()))?
                .as_str()
                .to_owned();
            let py_arr = tup[1].downcast_ref::<PyNdArray>().ok_or_else(|| {
                vm.new_type_error(format!("field '{}': value must be ndarray", name))
            })?;
            // O(1) clone: increments Arc refcount only
            let array_data = py_arr.inner().data().clone();
            if array_data.shape() != shape.as_slice() {
                return Err(vm.new_value_error(format!(
                    "field '{}' has shape {:?}, expected {:?}",
                    name,
                    array_data.shape(),
                    shape
                )));
            }
            fields.push(FieldSpec {
                name,
                data: array_data,
            });
        }
        if shape.iter().product::<usize>() > 0 && fields.is_empty() {
            return Err(vm.new_value_error(
                "structured array with non-zero size must have at least one field".to_owned(),
            ));
        }
        let sa = StructArrayData::new(fields, shape);
        Ok(Self::new_from_data(sa, dtype_json)
            .into_ref_with_type(vm, cls)?
            .into())
    }

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        let inner = self.inner.read().unwrap();
        let shape_vec: Vec<PyObjectRef> = inner
            .shape
            .iter()
            .map(|&d| vm.ctx.new_int(d as i64).into())
            .collect();
        vm.ctx.new_tuple(shape_vec).into()
    }

    #[pygetset]
    fn ndim(&self) -> usize {
        self.inner.read().unwrap().shape.len()
    }

    /// Returns the JSON dtype string — Python parses with json.loads().
    #[pygetset]
    fn dtype(&self) -> String {
        self.dtype_json.clone()
    }

    #[pymethod]
    fn field_names(&self, vm: &VirtualMachine) -> PyObjectRef {
        let inner = self.inner.read().unwrap();
        let names: Vec<PyObjectRef> = inner
            .field_names()
            .into_iter()
            .map(|n| vm.ctx.new_str(n).into())
            .collect();
        vm.ctx.new_list(names).into()
    }

    #[pymethod]
    fn __len__(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    #[pymethod]
    fn __getitem__(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let inner = self.inner.read().unwrap();

        // String key → return PyNdArray column
        if let Some(key_str) = key.downcast_ref::<PyStr>() {
            let name = key_str.as_str();
            let col_data = inner
                .field(name)
                .ok_or_else(|| vm.new_key_error(vm.ctx.new_str(name).into()))?;
            let nd = NdArray::from_data(col_data.clone());
            return Ok(PyNdArray::from_core(nd).to_py(vm));
        }

        // List of strings → new PyStructuredArray with subset of fields
        if let Ok(key_list) = key.clone().try_into_value::<Vec<PyObjectRef>>(vm) {
            if !key_list.is_empty() && key_list.iter().all(|k| k.downcast_ref::<PyStr>().is_some())
            {
                let names: Vec<&str> = key_list
                    .iter()
                    .map(|k| k.downcast_ref::<PyStr>().unwrap().as_str())
                    .collect();
                for &n in &names {
                    if inner.field(n).is_none() {
                        return Err(vm.new_key_error(vm.ctx.new_str(n).into()));
                    }
                }
                let subset_fields: Vec<FieldSpec> = names
                    .iter()
                    .map(|&n| FieldSpec {
                        name: n.to_owned(),
                        data: inner.field(n).unwrap().clone(),
                    })
                    .collect();
                let all_fields = parse_dtype_json_to_vec(&self.dtype_json, vm)?;
                let mut subset_parts: Vec<String> = Vec::new();
                for &n in &names {
                    if let Some((_, dt)) = all_fields.iter().find(|(k, _)| k == n) {
                        subset_parts.push(format!("[\"{}\",\"{}\"]", n, dt));
                    } else {
                        return Err(vm.new_value_error(format!(
                            "field '{}' missing from dtype_json — internal error",
                            n
                        )));
                    }
                }
                let subset_dtype_json = format!("[{}]", subset_parts.join(","));
                let sub_sa = StructArrayData::new(subset_fields, inner.shape.clone());
                return Ok(Self::new_from_data(sub_sa, subset_dtype_json).into_pyobject(vm));
            }
        }

        // Integer key → return list of scalars in field order.
        // Python StructuredArray.__getitem__ zips with dtype.names to build void.
        if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
            let row_scalars = inner.get_row(idx as isize).map_err(|e| numpy_err(e, vm))?;
            let values: Vec<PyObjectRef> = row_scalars
                .into_iter()
                .map(|s| scalar_to_py(s, vm))
                .collect();
            return Ok(vm.ctx.new_list(values).into());
        }

        Err(vm.new_type_error(format!(
            "StructuredArray indices must be string, list of strings, or integer, not {}",
            key.class().name()
        )))
    }

    #[pymethod]
    fn __setitem__(
        &self,
        key: PyObjectRef,
        value: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        if let Some(key_str) = key.downcast_ref::<PyStr>() {
            let name = key_str.as_str();
            let mut inner = self.inner.write().unwrap();
            let py_arr = value.downcast_ref::<PyNdArray>().ok_or_else(|| {
                vm.new_type_error("field assignment value must be ndarray".to_owned())
            })?;
            let array_data = py_arr.inner().data().clone();
            inner
                .set_field(name, array_data)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        }
        Err(vm.new_type_error("StructuredArray field assignment key must be a string".to_owned()))
    }
}

impl AsMapping for PyStructuredArray {
    fn as_mapping() -> &'static PyMappingMethods {
        use once_cell::sync::Lazy;
        static AS_MAPPING: Lazy<PyMappingMethods> = Lazy::new(|| PyMappingMethods {
            length: atomic_func!(|mapping, _vm| {
                let zelf = PyStructuredArray::mapping_downcast(mapping);
                Ok(zelf.__len__())
            }),
            subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                let zelf = PyStructuredArray::mapping_downcast(mapping);
                zelf.__getitem__(needle.to_owned(), vm)
            }),
            ass_subscript: atomic_func!(|mapping, needle: &vm::PyObject, value, vm| {
                let zelf = PyStructuredArray::mapping_downcast(mapping);
                let value = value.ok_or_else(|| {
                    vm.new_type_error("StructuredArray does not support item deletion".to_owned())
                })?;
                zelf.__setitem__(needle.to_owned(), value.to_owned(), vm)?;
                Ok(())
            }),
        });
        &AS_MAPPING
    }
}

/// Parse dtype_json into Vec<(name, dtype_str)>. Format: [["x","float64"],["y","int32"]]
fn parse_dtype_json_to_vec(
    dtype_json: &str,
    vm: &VirtualMachine,
) -> PyResult<Vec<(String, String)>> {
    let s = dtype_json.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(vm.new_value_error(format!("invalid dtype_json: {}", dtype_json)));
    }
    let inner = &s[1..s.len() - 1];
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    let pairs: Vec<&str> = inner.split("],[").collect();
    let mut result = Vec::new();
    for pair in pairs {
        let clean = pair.trim_matches(|c| c == '[' || c == ']');
        let parts: Vec<&str> = clean.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err(vm.new_value_error(format!("bad pair in dtype_json: {}", pair)));
        }
        let name = parts[0].trim().trim_matches('"').to_owned();
        let dtype_str = parts[1].trim().trim_matches('"').to_owned();
        result.push((name, dtype_str));
    }
    Ok(result)
}
