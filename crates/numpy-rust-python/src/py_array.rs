use rustpython_vm as vm;
use vm::atomic_func;
use vm::builtins::{PyList, PySlice, PyStr, PyTuple};
use vm::protocol::{PyMappingMethods, PyNumberMethods, PySequenceMethods};
use vm::types::{AsMapping, AsNumber, AsSequence};
use vm::{PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

use numpy_rust_core::indexing::{Scalar, SliceArg};
use numpy_rust_core::{DType, NdArray};

/// Python-visible ndarray class wrapping the core NdArray.
#[vm::pyclass(module = "numpy", name = "ndarray")]
#[derive(Debug, PyPayload)]
pub struct PyNdArray {
    data: NdArray,
}

impl Clone for PyNdArray {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl PyNdArray {
    pub fn from_core(data: NdArray) -> Self {
        Self { data }
    }

    pub fn inner(&self) -> &NdArray {
        &self.data
    }

    pub fn to_py(self, vm: &VirtualMachine) -> PyObjectRef {
        self.into_pyobject(vm)
    }
}

/// Convert a NumpyError to a Python exception.
fn numpy_err(
    e: numpy_rust_core::NumpyError,
    vm: &VirtualMachine,
) -> vm::builtins::PyBaseExceptionRef {
    vm.new_value_error(e.to_string())
}

/// Convert a Scalar to a Python object.
fn scalar_to_py(s: Scalar, vm: &VirtualMachine) -> PyObjectRef {
    match s {
        Scalar::Bool(v) => vm.ctx.new_bool(v).into(),
        Scalar::Int32(v) => vm.ctx.new_int(v).into(),
        Scalar::Int64(v) => vm.ctx.new_int(v).into(),
        Scalar::Float32(v) => vm.ctx.new_float(v as f64).into(),
        Scalar::Float64(v) => vm.ctx.new_float(v).into(),
        Scalar::Str(v) => vm.ctx.new_str(v).into(),
    }
}

/// Convert an NdArray result to PyObjectRef, returning a scalar for 0-D arrays.
pub fn ndarray_or_scalar(arr: NdArray, vm: &VirtualMachine) -> PyObjectRef {
    if arr.ndim() == 0 {
        let s = arr.get(&[]).unwrap();
        scalar_to_py(s, vm)
    } else {
        PyNdArray::from_core(arr).into_pyobject(vm)
    }
}

/// Parse a Python dtype string to DType.
fn parse_dtype(s: &str, vm: &VirtualMachine) -> PyResult<DType> {
    match s {
        "bool" => Ok(DType::Bool),
        "int32" | "i32" => Ok(DType::Int32),
        "int64" | "i64" | "int" => Ok(DType::Int64),
        "float32" | "f32" => Ok(DType::Float32),
        "float64" | "f64" | "float" => Ok(DType::Float64),
        "str" | "U" => Ok(DType::Str),
        _ if s.starts_with('S') || s.starts_with('U') => Ok(DType::Str),
        _ => Err(vm.new_type_error(format!("unsupported dtype: {s}"))),
    }
}

/// Extract a shape tuple from a Python object (tuple or list of ints).
pub fn extract_shape(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    if let Some(tuple) = obj.downcast_ref::<PyTuple>() {
        let mut shape = Vec::new();
        for item in tuple.as_slice() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        Ok(shape)
    } else if let Some(list) = obj.downcast_ref::<PyList>() {
        let mut shape = Vec::new();
        for item in list.borrow_vec().iter() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        Ok(shape)
    } else if let Ok(n) = obj.clone().try_into_value::<i64>(vm) {
        Ok(vec![n as usize])
    } else {
        Err(vm.new_type_error("shape must be a tuple, list, or integer".to_owned()))
    }
}

#[vm::pyclass(flags(BASETYPE), with(AsNumber, AsMapping, AsSequence))]
impl PyNdArray {
    // --- Properties ---

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        let shape: Vec<PyObjectRef> = self
            .data
            .shape()
            .iter()
            .map(|&s| vm.ctx.new_int(s).into())
            .collect();
        PyTuple::new_ref(shape, &vm.ctx).into()
    }

    #[pygetset]
    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    #[pygetset]
    fn size(&self) -> usize {
        self.data.size()
    }

    #[pygetset]
    fn dtype(&self) -> String {
        self.data.dtype().to_string()
    }

    #[pygetset(name = "T")]
    fn transpose_prop(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.transpose())
    }

    // --- Methods ---

    #[pymethod]
    fn reshape(&self, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        self.data
            .reshape(&sh)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn flatten(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.flatten())
    }

    #[pymethod]
    fn ravel(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.ravel())
    }

    #[pymethod]
    fn copy(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.copy())
    }

    #[pymethod]
    fn astype(&self, dtype: PyRef<PyStr>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let dt = parse_dtype(dtype.as_str(), vm)?;
        Ok(PyNdArray::from_core(self.data.astype(dt)))
    }

    #[pymethod]
    fn sum(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .sum(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn mean(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .mean(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn min(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .min(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn max(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .max(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn std(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .std(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn var(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .var(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmin(&self, vm: &VirtualMachine) -> PyResult<usize> {
        self.data.argmin().map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmax(&self, vm: &VirtualMachine) -> PyResult<usize> {
        self.data.argmax().map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn all(&self) -> bool {
        self.data.all()
    }

    #[pymethod]
    fn any(&self) -> bool {
        self.data.any()
    }

    // --- Unary math ---

    #[pymethod]
    fn abs(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.abs())
    }

    #[pymethod]
    fn sqrt(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.sqrt())
    }

    // --- Scalar conversion ---

    #[pymethod]
    fn float(&self, vm: &VirtualMachine) -> PyResult<f64> {
        if self.data.size() != 1 {
            return Err(vm.new_type_error(
                "only size-1 arrays can be converted to Python scalars".to_owned(),
            ));
        }
        let s = self
            .data
            .get(&vec![0; self.data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(match s {
            Scalar::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
            Scalar::Int32(v) => v as f64,
            Scalar::Int64(v) => v as f64,
            Scalar::Float32(v) => v as f64,
            Scalar::Float64(v) => v,
            Scalar::Str(v) => v.parse::<f64>().map_err(|_| {
                vm.new_value_error(format!("could not convert string to float: '{v}'"))
            })?,
        })
    }

    #[pymethod]
    fn int(&self, vm: &VirtualMachine) -> PyResult<i64> {
        if self.data.size() != 1 {
            return Err(vm.new_type_error(
                "only size-1 arrays can be converted to Python scalars".to_owned(),
            ));
        }
        let s = self
            .data
            .get(&vec![0; self.data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(match s {
            Scalar::Bool(v) => {
                if v {
                    1
                } else {
                    0
                }
            }
            Scalar::Int32(v) => v as i64,
            Scalar::Int64(v) => v,
            Scalar::Float32(v) => v as i64,
            Scalar::Float64(v) => v as i64,
            Scalar::Str(v) => v.parse::<i64>().map_err(|_| {
                vm.new_value_error(format!("could not convert string to int: '{v}'"))
            })?,
        })
    }

    #[pymethod]
    fn bool(&self, vm: &VirtualMachine) -> PyResult<bool> {
        if self.data.size() != 1 {
            return Err(vm.new_value_error(
                "The truth value of an array with more than one element is ambiguous".to_owned(),
            ));
        }
        let s = self
            .data
            .get(&vec![0; self.data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(match s {
            Scalar::Bool(v) => v,
            Scalar::Int32(v) => v != 0,
            Scalar::Int64(v) => v != 0,
            Scalar::Float32(v) => v != 0.0,
            Scalar::Float64(v) => v != 0.0,
            Scalar::Str(v) => !v.is_empty(),
        })
    }

    // --- Operators ---

    // Arithmetic operators are implemented via AsNumber below.
    // Reverse ops (radd, rsub, etc.) are handled by the number protocol.

    // Comparison operators are implemented via the Comparable trait below.

    // --- Indexing ---

    #[pymethod]
    fn __getitem__(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Integer index -> scalar or sub-array
        if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
            let resolved = if idx < 0 {
                (self.data.shape()[0] as i64 + idx) as usize
            } else {
                idx as usize
            };

            if self.data.ndim() == 1 {
                let s = self.data.get(&[resolved]).map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py(s, vm));
            } else {
                let result = self
                    .data
                    .slice(&[SliceArg::Index(idx as isize)])
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        // Single slice -> e.g. a[0:3], a[::2], a[:]
        if let Some(slice) = key.downcast_ref::<PySlice>() {
            let arg = py_slice_to_slice_arg(slice, vm)?;
            let result = self.data.slice(&[arg]).map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // Tuple index -> multi-dimensional indexing (integers and/or slices)
        if let Some(tuple) = key.downcast_ref::<PyTuple>() {
            let items = tuple.as_slice();
            let has_slice = items
                .iter()
                .any(|item| item.downcast_ref::<PySlice>().is_some());

            if has_slice {
                // Mixed integers and slices — use SliceArg for each element
                let args: Vec<SliceArg> = items
                    .iter()
                    .map(|item| py_obj_to_slice_arg(item, vm))
                    .collect::<PyResult<Vec<_>>>()?;
                let result = self.data.slice(&args).map_err(|e| numpy_err(e, vm))?;
                return Ok(ndarray_or_scalar(result, vm));
            }

            // All integers
            let mut indices = Vec::new();
            for item in items {
                let i: i64 = item.clone().try_into_value(vm)?;
                indices.push(i as isize);
            }
            if indices.len() == self.data.ndim() {
                let usize_indices: Vec<usize> = indices
                    .iter()
                    .enumerate()
                    .map(|(axis, &i)| {
                        if i < 0 {
                            (self.data.shape()[axis] as isize + i) as usize
                        } else {
                            i as usize
                        }
                    })
                    .collect();
                let s = self
                    .data
                    .get(&usize_indices)
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py(s, vm));
            }
            let args: Vec<SliceArg> = indices.iter().map(|&i| SliceArg::Index(i)).collect();
            let result = self.data.slice(&args).map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // Boolean mask
        if let Some(mask) = key.downcast_ref::<PyNdArray>() {
            if mask.data.dtype() == DType::Bool {
                let result = self
                    .data
                    .mask_select(&mask.data)
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        Err(vm.new_type_error("unsupported index type".to_owned()))
    }

    // --- Comparison (slot-based, returns ndarray like NumPy) ---

    #[pyslot]
    fn slot_richcompare(
        zelf: &vm::PyObject,
        other: &vm::PyObject,
        op: vm::types::PyComparisonOp,
        vm: &VirtualMachine,
    ) -> PyResult<vm::function::Either<PyObjectRef, vm::function::PyComparisonValue>> {
        let zelf = zelf
            .downcast_ref::<PyNdArray>()
            .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
        let other = match obj_to_ndarray(other, vm) {
            Ok(arr) => arr,
            Err(_) => {
                return Ok(vm::function::Either::B(
                    vm::function::PyComparisonValue::NotImplemented,
                ))
            }
        };
        let result = match op {
            vm::types::PyComparisonOp::Eq => zelf.data.eq(&other),
            vm::types::PyComparisonOp::Ne => zelf.data.ne(&other),
            vm::types::PyComparisonOp::Lt => zelf.data.lt(&other),
            vm::types::PyComparisonOp::Le => zelf.data.le(&other),
            vm::types::PyComparisonOp::Gt => zelf.data.gt(&other),
            vm::types::PyComparisonOp::Ge => zelf.data.ge(&other),
        };
        match result {
            Ok(arr) => Ok(vm::function::Either::A(
                PyNdArray::from_core(arr).into_pyobject(vm),
            )),
            Err(e) => Err(numpy_err(e, vm)),
        }
    }

    // --- String representations ---

    #[pymethod]
    fn __repr__(&self) -> String {
        format!(
            "ndarray(shape={:?}, dtype={})",
            self.data.shape(),
            self.data.dtype()
        )
    }

    #[pymethod]
    fn __str__(&self) -> String {
        format!(
            "ndarray(shape={:?}, dtype={})",
            self.data.shape(),
            self.data.dtype()
        )
    }

    #[pymethod]
    fn __len__(&self) -> usize {
        if self.data.ndim() == 0 {
            0
        } else {
            self.data.shape()[0]
        }
    }
}

/// Try to get an NdArray from a PyObject, auto-wrapping scalars (int/float/bool/str).
fn obj_to_ndarray(obj: &vm::PyObject, vm: &VirtualMachine) -> PyResult<NdArray> {
    if let Some(arr) = obj.downcast_ref::<PyNdArray>() {
        return Ok(arr.data.clone());
    }
    // Try float (PyFloat)
    if let Some(f) = obj.downcast_ref::<vm::builtins::PyFloat>() {
        return Ok(NdArray::from_scalar(f.to_f64()));
    }
    // Try int (PyInt) — extract i64 then convert to f64
    if let Some(i) = obj.downcast_ref::<vm::builtins::PyInt>() {
        let val: i64 = i.try_to_primitive(vm)?;
        return Ok(NdArray::from_scalar(val as f64));
    }
    // Try str (PyStr) — create 0-D string array
    if let Some(s) = obj.downcast_ref::<vm::builtins::PyStr>() {
        return Ok(NdArray::from_vec(vec![s.as_str().to_owned()]));
    }
    Err(vm.new_type_error("expected ndarray or scalar".to_owned()))
}

// --- AsNumber implementation for operator dispatch ---

fn number_bin_op(
    a: &vm::PyObject,
    b: &vm::PyObject,
    op: fn(&NdArray, &NdArray) -> numpy_rust_core::Result<NdArray>,
    vm: &VirtualMachine,
) -> PyResult {
    let a_arr = obj_to_ndarray(a, vm)?;
    let b_arr = obj_to_ndarray(b, vm)?;
    op(&a_arr, &b_arr)
        .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
        .map_err(|e| vm.new_value_error(e.to_string()))
}

fn number_neg(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    Ok(PyNdArray::from_core(a.data.neg()).into_pyobject(vm))
}

fn number_float(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    if a.data.size() != 1 {
        return Err(
            vm.new_type_error("only size-1 arrays can be converted to Python scalars".to_owned())
        );
    }
    let s = a
        .data
        .get(&vec![0; a.data.ndim()])
        .map_err(|e| numpy_err(e, vm))?;
    let v = match s {
        Scalar::Bool(v) => {
            if v {
                1.0
            } else {
                0.0
            }
        }
        Scalar::Int32(v) => v as f64,
        Scalar::Int64(v) => v as f64,
        Scalar::Float32(v) => v as f64,
        Scalar::Float64(v) => v,
        Scalar::Str(v) => v
            .parse::<f64>()
            .map_err(|_| vm.new_value_error(format!("could not convert string to float: '{v}'")))?,
    };
    Ok(vm.ctx.new_float(v).into())
}

fn number_int(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    if a.data.size() != 1 {
        return Err(
            vm.new_type_error("only size-1 arrays can be converted to Python scalars".to_owned())
        );
    }
    let s = a
        .data
        .get(&vec![0; a.data.ndim()])
        .map_err(|e| numpy_err(e, vm))?;
    let v = match s {
        Scalar::Bool(v) => {
            if v {
                1i64
            } else {
                0
            }
        }
        Scalar::Int32(v) => v as i64,
        Scalar::Int64(v) => v,
        Scalar::Float32(v) => v as i64,
        Scalar::Float64(v) => v as i64,
        Scalar::Str(v) => v
            .parse::<i64>()
            .map_err(|_| vm.new_value_error(format!("could not convert string to int: '{v}'")))?,
    };
    Ok(vm.ctx.new_int(v).into())
}

impl PyNdArray {
    const AS_NUMBER: PyNumberMethods = PyNumberMethods {
        add: Some(|a, b, vm| number_bin_op(a, b, |x, y| x + y, vm)),
        subtract: Some(|a, b, vm| number_bin_op(a, b, |x, y| x - y, vm)),
        multiply: Some(|a, b, vm| number_bin_op(a, b, |x, y| x * y, vm)),
        true_divide: Some(|a, b, vm| number_bin_op(a, b, |x, y| x / y, vm)),
        negative: Some(number_neg),
        int: Some(number_int),
        float: Some(number_float),
        ..PyNumberMethods::NOT_IMPLEMENTED
    };
}

impl AsNumber for PyNdArray {
    fn as_number() -> &'static PyNumberMethods {
        static AS_NUMBER: PyNumberMethods = PyNdArray::AS_NUMBER;
        &AS_NUMBER
    }
}

impl AsMapping for PyNdArray {
    fn as_mapping() -> &'static PyMappingMethods {
        use once_cell::sync::Lazy;
        static AS_MAPPING: Lazy<PyMappingMethods> = Lazy::new(|| PyMappingMethods {
            length: atomic_func!(|mapping, _vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                Ok(if zelf.data.ndim() == 0 {
                    0
                } else {
                    zelf.data.shape()[0]
                })
            }),
            subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                zelf.__getitem__(needle.to_owned(), vm)
            }),
            ..PyMappingMethods::NOT_IMPLEMENTED
        });
        &AS_MAPPING
    }
}

impl AsSequence for PyNdArray {
    fn as_sequence() -> &'static PySequenceMethods {
        use once_cell::sync::Lazy;
        static AS_SEQUENCE: Lazy<PySequenceMethods> = Lazy::new(|| PySequenceMethods {
            length: atomic_func!(|seq, _vm| {
                let zelf = PyNdArray::sequence_downcast(seq);
                Ok(if zelf.data.ndim() == 0 {
                    0
                } else {
                    zelf.data.shape()[0]
                })
            }),
            ..PySequenceMethods::NOT_IMPLEMENTED
        });
        &AS_SEQUENCE
    }
}

/// Convert a RustPython `PySlice` to a core `SliceArg`.
fn py_slice_to_slice_arg(slice: &PySlice, vm: &VirtualMachine) -> PyResult<SliceArg> {
    let start = match &slice.start {
        Some(obj) if !vm.is_none(obj) => {
            let v: i64 = obj.clone().try_into_value(vm)?;
            Some(v as isize)
        }
        _ => None,
    };
    let stop = if vm.is_none(&slice.stop) {
        None
    } else {
        let v: i64 = slice.stop.clone().try_into_value(vm)?;
        Some(v as isize)
    };
    let step = match &slice.step {
        Some(obj) if !vm.is_none(obj) => {
            let v: i64 = obj.clone().try_into_value(vm)?;
            v as isize
        }
        _ => 1,
    };
    Ok(SliceArg::Range { start, stop, step })
}

/// Convert a Python object to a `SliceArg` (either a slice or an integer index).
fn py_obj_to_slice_arg(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<SliceArg> {
    if let Some(slice) = obj.downcast_ref::<PySlice>() {
        return py_slice_to_slice_arg(slice, vm);
    }
    if let Ok(i) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(SliceArg::Index(i as isize));
    }
    Err(vm.new_type_error("index must be an integer or slice".to_owned()))
}

/// Parse an optional axis argument (None means reduce all).
fn parse_optional_axis(
    arg: vm::function::OptionalArg<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<Option<usize>> {
    match arg.into_option() {
        None => Ok(None),
        Some(obj) => {
            if vm.is_none(&obj) {
                Ok(None)
            } else {
                let axis: i64 = obj.try_into_value(vm)?;
                Ok(Some(axis as usize))
            }
        }
    }
}
