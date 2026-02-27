use std::sync::RwLock;

use rustpython_vm as vm;
use vm::atomic_func;
use vm::builtins::{PyList, PySlice, PyStr, PyTuple};
use vm::protocol::{PyIterReturn, PyMappingMethods, PyNumberMethods, PySequenceMethods};
use vm::types::{AsMapping, AsNumber, AsSequence, IterNext, Iterable, Representable, SelfIter};
use vm::{Py, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

use numpy_rust_core::indexing::{Scalar, SliceArg};
use numpy_rust_core::{DType, NdArray};

/// Python-visible ndarray class wrapping the core NdArray.
/// Uses `RwLock` for interior mutability so `__setitem__` can work
/// through RustPython's `&self` method signatures.
#[vm::pyclass(module = "numpy", name = "ndarray")]
#[derive(Debug, PyPayload)]
pub struct PyNdArray {
    data: RwLock<NdArray>,
}

impl Clone for PyNdArray {
    fn clone(&self) -> Self {
        Self {
            data: RwLock::new(self.data.read().unwrap().clone()),
        }
    }
}

impl PyNdArray {
    pub fn from_core(data: NdArray) -> Self {
        Self {
            data: RwLock::new(data),
        }
    }

    pub fn inner(&self) -> std::sync::RwLockReadGuard<'_, NdArray> {
        self.data.read().unwrap()
    }

    pub fn to_py(self, vm: &VirtualMachine) -> PyObjectRef {
        self.into_pyobject(vm)
    }
}

/// Iterator over the first axis of an ndarray.
#[vm::pyclass(module = "numpy", name = "ndarray_iterator")]
#[derive(Debug, PyPayload)]
pub struct PyNdArrayIter {
    array: PyRef<PyNdArray>,
    index: std::sync::atomic::AtomicUsize,
    length: usize,
}

#[vm::pyclass(flags(DISALLOW_INSTANTIATION), with(IterNext, Iterable))]
impl PyNdArrayIter {}

impl SelfIter for PyNdArrayIter {}

impl IterNext for PyNdArrayIter {
    fn next(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<PyIterReturn> {
        let idx = zelf
            .index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if idx >= zelf.length {
            return Ok(PyIterReturn::StopIteration(None));
        }
        let data = zelf.array.data.read().unwrap();
        if data.ndim() == 1 {
            let s = data.get(&[idx]).map_err(|e| numpy_err(e, vm))?;
            Ok(PyIterReturn::Return(scalar_to_py(s, vm)))
        } else {
            let result = data
                .slice(&[SliceArg::Index(idx as isize)])
                .map_err(|e| numpy_err(e, vm))?;
            Ok(PyIterReturn::Return(
                PyNdArray::from_core(result).into_pyobject(vm),
            ))
        }
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
        Scalar::Complex64(v) => {
            let tup = vm.ctx.new_tuple(vec![
                vm.ctx.new_float(v.re as f64).into(),
                vm.ctx.new_float(v.im as f64).into(),
            ]);
            tup.into()
        }
        Scalar::Complex128(v) => {
            let tup = vm.ctx.new_tuple(vec![
                vm.ctx.new_float(v.re).into(),
                vm.ctx.new_float(v.im).into(),
            ]);
            tup.into()
        }
        Scalar::Str(v) => vm.ctx.new_str(v).into(),
    }
}

/// Convert a Python object to a Scalar matching the target array dtype.
fn py_obj_to_scalar(obj: &PyObjectRef, dtype: DType, vm: &VirtualMachine) -> PyResult<Scalar> {
    if let Ok(f) = obj.clone().try_into_value::<f64>(vm) {
        return Ok(match dtype {
            DType::Float64 => Scalar::Float64(f),
            DType::Float32 => Scalar::Float32(f as f32),
            DType::Int64 => Scalar::Int64(f as i64),
            DType::Int32 => Scalar::Int32(f as i32),
            DType::Bool => Scalar::Bool(f != 0.0),
            DType::Complex64 => Scalar::Complex64(num_complex::Complex::new(f as f32, 0.0)),
            DType::Complex128 => Scalar::Complex128(num_complex::Complex::new(f, 0.0)),
            DType::Str => Scalar::Str(f.to_string()),
        });
    }
    if let Ok(i) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(match dtype {
            DType::Float64 => Scalar::Float64(i as f64),
            DType::Float32 => Scalar::Float32(i as f32),
            DType::Int64 => Scalar::Int64(i),
            DType::Int32 => Scalar::Int32(i as i32),
            DType::Bool => Scalar::Bool(i != 0),
            DType::Complex64 => Scalar::Complex64(num_complex::Complex::new(i as f32, 0.0)),
            DType::Complex128 => Scalar::Complex128(num_complex::Complex::new(i as f64, 0.0)),
            DType::Str => Scalar::Str(i.to_string()),
        });
    }
    Err(vm.new_type_error("cannot convert value to array scalar".to_owned()))
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
        "complex64" | "c64" => Ok(DType::Complex64),
        "complex128" | "c128" | "complex" => Ok(DType::Complex128),
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

#[vm::pyclass(
    flags(BASETYPE),
    with(AsNumber, AsMapping, AsSequence, Representable, Iterable)
)]
impl PyNdArray {
    // --- Properties ---

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        let data = self.data.read().unwrap();
        let shape: Vec<PyObjectRef> = data
            .shape()
            .iter()
            .map(|&s| vm.ctx.new_int(s).into())
            .collect();
        PyTuple::new_ref(shape, &vm.ctx).into()
    }

    #[pygetset]
    fn ndim(&self) -> usize {
        self.data.read().unwrap().ndim()
    }

    #[pygetset]
    fn size(&self) -> usize {
        self.data.read().unwrap().size()
    }

    #[pygetset]
    fn dtype(&self) -> String {
        self.data.read().unwrap().dtype().to_string()
    }

    #[pygetset(name = "T")]
    fn transpose_prop(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().transpose())
    }

    #[pygetset]
    fn real(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().real())
    }

    #[pygetset]
    fn imag(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().imag())
    }

    // --- Methods ---

    #[pymethod]
    fn reshape(&self, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        self.data
            .read()
            .unwrap()
            .reshape(&sh)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn flatten(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().flatten())
    }

    #[pymethod]
    fn ravel(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().ravel())
    }

    #[pymethod]
    fn expand_dims(&self, axis: usize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .expand_dims(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn squeeze(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .squeeze(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn copy(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().copy())
    }

    #[pymethod]
    fn astype(&self, dtype: PyRef<PyStr>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let dt = parse_dtype(dtype.as_str(), vm)?;
        Ok(PyNdArray::from_core(self.data.read().unwrap().astype(dt)))
    }

    #[pymethod]
    fn sum(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let kd = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .sum(ax, kd)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn mean(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let kd = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .mean(ax, kd)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn min(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let kd = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .min(ax, kd)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn max(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let kd = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .max(ax, kd)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn std(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let dd = ddof.unwrap_or(0);
        let kd = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .std(ax, dd, kd)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn var(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let dd = ddof.unwrap_or(0);
        let kd = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .var(ax, dd, kd)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmin(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .argmin(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmax(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .argmax(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn all(&self) -> bool {
        self.data.read().unwrap().all()
    }

    #[pymethod]
    fn any(&self) -> bool {
        self.data.read().unwrap().any()
    }

    // --- Unary math ---

    #[pymethod]
    fn abs(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().abs())
    }

    #[pymethod]
    fn sqrt(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sqrt())
    }

    #[pymethod]
    fn exp(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().exp())
    }

    #[pymethod]
    fn log(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().log())
    }

    #[pymethod]
    fn sin(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sin())
    }

    #[pymethod]
    fn cos(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().cos())
    }

    #[pymethod]
    fn tan(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().tan())
    }

    #[pymethod]
    fn floor(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let result = self
            .data
            .read()
            .unwrap()
            .floor()
            .map_err(|e| numpy_err(e, vm))?;
        Ok(PyNdArray::from_core(result))
    }

    #[pymethod]
    fn ceil(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let result = self
            .data
            .read()
            .unwrap()
            .ceil()
            .map_err(|e| numpy_err(e, vm))?;
        Ok(PyNdArray::from_core(result))
    }

    #[pymethod]
    fn round(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let result = self
            .data
            .read()
            .unwrap()
            .round()
            .map_err(|e| numpy_err(e, vm))?;
        Ok(PyNdArray::from_core(result))
    }

    #[pymethod]
    fn log10(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().log10())
    }

    #[pymethod]
    fn log2(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().log2())
    }

    #[pymethod]
    fn sinh(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sinh())
    }

    #[pymethod]
    fn cosh(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().cosh())
    }

    #[pymethod]
    fn tanh(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().tanh())
    }

    #[pymethod]
    fn arcsin(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().arcsin())
    }

    #[pymethod]
    fn arccos(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().arccos())
    }

    #[pymethod]
    fn arctan(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().arctan())
    }

    #[pymethod]
    fn log1p(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .log1p()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn expm1(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .expm1()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn deg2rad(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .deg2rad()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn rad2deg(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .rad2deg()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn sign(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sign())
    }

    #[pymethod]
    fn conj(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().conj())
    }

    #[pymethod]
    fn conjugate(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().conj())
    }

    #[pymethod]
    fn angle(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().angle())
    }

    // --- Element-wise checks ---

    #[pymethod]
    fn isnan(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().isnan())
    }

    #[pymethod]
    fn isinf(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().isinf())
    }

    #[pymethod]
    fn isfinite(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().isfinite())
    }

    #[pymethod]
    fn around(&self, decimals: vm::function::OptionalArg<i32>) -> PyNdArray {
        let d = decimals.unwrap_or(0);
        PyNdArray::from_core(self.data.read().unwrap().around(d))
    }

    // --- Scalar conversion ---

    #[pymethod]
    fn float(&self, vm: &VirtualMachine) -> PyResult<f64> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_type_error(
                "only size-1 arrays can be converted to Python scalars".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
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
            Scalar::Complex64(v) => v.re as f64,
            Scalar::Complex128(v) => v.re,
            Scalar::Str(v) => v.parse::<f64>().map_err(|_| {
                vm.new_value_error(format!("could not convert string to float: '{v}'"))
            })?,
        })
    }

    #[pymethod]
    fn int(&self, vm: &VirtualMachine) -> PyResult<i64> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_type_error(
                "only size-1 arrays can be converted to Python scalars".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
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
            Scalar::Complex64(v) => v.re as i64,
            Scalar::Complex128(v) => v.re as i64,
            Scalar::Str(v) => v.parse::<i64>().map_err(|_| {
                vm.new_value_error(format!("could not convert string to int: '{v}'"))
            })?,
        })
    }

    #[pymethod]
    fn bool(&self, vm: &VirtualMachine) -> PyResult<bool> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_value_error(
                "The truth value of an array with more than one element is ambiguous".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(match s {
            Scalar::Bool(v) => v,
            Scalar::Int32(v) => v != 0,
            Scalar::Int64(v) => v != 0,
            Scalar::Float32(v) => v != 0.0,
            Scalar::Float64(v) => v != 0.0,
            Scalar::Complex64(v) => v.re != 0.0 || v.im != 0.0,
            Scalar::Complex128(v) => v.re != 0.0 || v.im != 0.0,
            Scalar::Str(v) => !v.is_empty(),
        })
    }

    // --- Sort / Argsort ---

    #[pymethod]
    fn sort(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .sort(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argsort(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .argsort(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    // --- Cumulative operations ---

    #[pymethod]
    fn cumsum(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .cumsum(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pymethod]
    fn cumprod(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .cumprod(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Product reduction ---

    #[pymethod]
    fn prod(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = parse_optional_axis(axis, vm)?;
        let keepdims = keepdims.unwrap_or(false);
        self.data
            .read()
            .unwrap()
            .prod(axis, keepdims)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Operators ---

    // Arithmetic operators are implemented via AsNumber below.
    // Reverse ops (radd, rsub, etc.) are handled by the number protocol.

    // Comparison operators are implemented via the Comparable trait below.

    // --- Indexing ---

    #[pymethod]
    fn __getitem__(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();

        // Integer index -> scalar or sub-array
        if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
            let resolved = if idx < 0 {
                (data.shape()[0] as i64 + idx) as usize
            } else {
                idx as usize
            };

            if data.ndim() == 1 {
                let s = data.get(&[resolved]).map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py(s, vm));
            } else {
                let result = data
                    .slice(&[SliceArg::Index(idx as isize)])
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        // Single slice -> e.g. a[0:3], a[::2], a[:]
        if let Some(slice) = key.downcast_ref::<PySlice>() {
            let arg = py_slice_to_slice_arg(slice, vm)?;
            let result = data.slice(&[arg]).map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // Tuple index -> multi-dimensional indexing (integers and/or slices)
        if let Some(tuple) = key.downcast_ref::<PyTuple>() {
            let items = tuple.as_slice();
            let has_slice = items
                .iter()
                .any(|item| item.downcast_ref::<PySlice>().is_some());

            if has_slice {
                let args: Vec<SliceArg> = items
                    .iter()
                    .map(|item| py_obj_to_slice_arg(item, vm))
                    .collect::<PyResult<Vec<_>>>()?;
                let result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
                return Ok(ndarray_or_scalar(result, vm));
            }

            // All integers
            let mut indices = Vec::new();
            for item in items {
                let i: i64 = item.clone().try_into_value(vm)?;
                indices.push(i as isize);
            }
            if indices.len() == data.ndim() {
                let shape = data.shape();
                let usize_indices: Vec<usize> = indices
                    .iter()
                    .enumerate()
                    .map(|(axis, &i)| {
                        if i < 0 {
                            (shape[axis] as isize + i) as usize
                        } else {
                            i as usize
                        }
                    })
                    .collect();
                let s = data.get(&usize_indices).map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py(s, vm));
            }
            let args: Vec<SliceArg> = indices.iter().map(|&i| SliceArg::Index(i)).collect();
            let result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // NdArray index: boolean mask or integer fancy indexing
        if let Some(arr) = key.downcast_ref::<PyNdArray>() {
            let arr_data = arr.data.read().unwrap();
            if arr_data.dtype() == DType::Bool {
                let result = data.mask_select(&arr_data).map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            } else if arr_data.dtype().is_integer() || arr_data.dtype().is_float() {
                let indices = extract_int_indices(&arr_data, data.shape()[0], vm)?;
                let result = data
                    .index_select(0, &indices)
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        // List of integers -> fancy indexing: a[[0, 2, 4]]
        if let Some(list) = key.downcast_ref::<PyList>() {
            let items = list.borrow_vec();
            let dim_size = data.shape()[0];
            let mut indices = Vec::with_capacity(items.len());
            for item in items.iter() {
                let i: i64 = item.clone().try_into_value(vm)?;
                let resolved = if i < 0 {
                    (dim_size as i64 + i) as usize
                } else {
                    i as usize
                };
                indices.push(resolved);
            }
            let result = data
                .index_select(0, &indices)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
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
        let data = zelf.data.read().unwrap();
        let result = match op {
            vm::types::PyComparisonOp::Eq => data.eq(&other),
            vm::types::PyComparisonOp::Ne => data.ne(&other),
            vm::types::PyComparisonOp::Lt => data.lt(&other),
            vm::types::PyComparisonOp::Le => data.le(&other),
            vm::types::PyComparisonOp::Gt => data.gt(&other),
            vm::types::PyComparisonOp::Ge => data.ge(&other),
        };
        match result {
            Ok(arr) => Ok(vm::function::Either::A(
                PyNdArray::from_core(arr).into_pyobject(vm),
            )),
            Err(e) => Err(numpy_err(e, vm)),
        }
    }

    // --- New methods: tolist, item ---

    #[pymethod]
    fn tolist(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        Ok(ndarray_to_pylist(&data, vm))
    }

    #[pymethod]
    fn item(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_value_error(
                "can only convert an array of size 1 to a Python scalar".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(scalar_to_py(s, vm))
    }

    // --- String representations ---

    #[pymethod]
    fn __repr__(&self) -> String {
        let data = self.data.read().unwrap();
        format_array(&data, true)
    }

    #[pymethod]
    fn __str__(&self) -> String {
        let data = self.data.read().unwrap();
        format_array(&data, false)
    }

    #[pymethod]
    fn __len__(&self) -> usize {
        let data = self.data.read().unwrap();
        if data.ndim() == 0 {
            0
        } else {
            data.shape()[0]
        }
    }
}

/// Try to get an NdArray from a PyObject, auto-wrapping scalars (int/float/bool/str).
pub fn obj_to_ndarray(obj: &vm::PyObject, vm: &VirtualMachine) -> PyResult<NdArray> {
    if let Some(arr) = obj.downcast_ref::<PyNdArray>() {
        return Ok(arr.data.read().unwrap().clone());
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
    Ok(PyNdArray::from_core(a.data.read().unwrap().neg()).into_pyobject(vm))
}

fn number_invert(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let result = a
        .data
        .read()
        .unwrap()
        .bitwise_not()
        .map_err(|e| numpy_err(e, vm))?;
    Ok(PyNdArray::from_core(result).into_pyobject(vm))
}

fn number_absolute(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    Ok(PyNdArray::from_core(a.data.read().unwrap().abs()).into_pyobject(vm))
}

fn number_boolean(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult<bool> {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let data = a.data.read().unwrap();
    if data.size() != 1 {
        return Err(vm.new_value_error(
            "The truth value of an array with more than one element is ambiguous".to_owned(),
        ));
    }
    let s = data
        .get(&vec![0; data.ndim()])
        .map_err(|e| numpy_err(e, vm))?;
    Ok(match s {
        Scalar::Bool(v) => v,
        Scalar::Int32(v) => v != 0,
        Scalar::Int64(v) => v != 0,
        Scalar::Float32(v) => v != 0.0,
        Scalar::Float64(v) => v != 0.0,
        Scalar::Complex64(v) => v.re != 0.0 || v.im != 0.0,
        Scalar::Complex128(v) => v.re != 0.0 || v.im != 0.0,
        Scalar::Str(v) => !v.is_empty(),
    })
}

fn number_float(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let data = a.data.read().unwrap();
    if data.size() != 1 {
        return Err(
            vm.new_type_error("only size-1 arrays can be converted to Python scalars".to_owned())
        );
    }
    let s = data
        .get(&vec![0; data.ndim()])
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
        Scalar::Complex64(v) => v.re as f64,
        Scalar::Complex128(v) => v.re,
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
    let data = a.data.read().unwrap();
    if data.size() != 1 {
        return Err(
            vm.new_type_error("only size-1 arrays can be converted to Python scalars".to_owned())
        );
    }
    let s = data
        .get(&vec![0; data.ndim()])
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
        Scalar::Complex64(v) => v.re as i64,
        Scalar::Complex128(v) => v.re as i64,
        Scalar::Str(v) => v
            .parse::<i64>()
            .map_err(|_| vm.new_value_error(format!("could not convert string to int: '{v}'")))?,
    };
    Ok(vm.ctx.new_int(v).into())
}

fn number_inplace_bin_op(
    a: &vm::PyObject,
    b: &vm::PyObject,
    op: fn(&NdArray, &NdArray) -> numpy_rust_core::Result<NdArray>,
    vm: &VirtualMachine,
) -> PyResult {
    let a_py = a
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let b_arr = obj_to_ndarray(b, vm)?;
    let a_inner = a_py.data.read().unwrap().clone();
    let result = op(&a_inner, &b_arr).map_err(|e| vm.new_value_error(e.to_string()))?;
    *a_py.data.write().unwrap() = result;
    Ok(a.to_owned())
}

impl PyNdArray {
    const AS_NUMBER: PyNumberMethods = PyNumberMethods {
        add: Some(|a, b, vm| number_bin_op(a, b, |x, y| x + y, vm)),
        subtract: Some(|a, b, vm| number_bin_op(a, b, |x, y| x - y, vm)),
        multiply: Some(|a, b, vm| number_bin_op(a, b, |x, y| x * y, vm)),
        true_divide: Some(|a, b, vm| number_bin_op(a, b, |x, y| x / y, vm)),
        floor_divide: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.floor_div(y), vm)),
        remainder: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.remainder(y), vm)),
        power: Some(|a, b, _modulo, vm| {
            let a_arr = obj_to_ndarray(a, vm)?;
            let b_arr = obj_to_ndarray(b, vm)?;
            a_arr
                .pow(&b_arr)
                .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
                .map_err(|e| vm.new_value_error(e.to_string()))
        }),
        negative: Some(number_neg),
        int: Some(number_int),
        float: Some(number_float),
        and: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.bitwise_and(y), vm)),
        or: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.bitwise_or(y), vm)),
        invert: Some(number_invert),
        matrix_multiply: Some(|a, b, vm| {
            let a_arr = obj_to_ndarray(a, vm)?;
            let b_arr = obj_to_ndarray(b, vm)?;
            numpy_rust_core::dot(&a_arr, &b_arr)
                .map(|r| ndarray_or_scalar(r, vm))
                .map_err(|e| vm.new_value_error(e.to_string()))
        }),
        inplace_add: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x + y, vm)),
        inplace_subtract: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x - y, vm)),
        inplace_multiply: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x * y, vm)),
        inplace_true_divide: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x / y, vm)),
        inplace_floor_divide: Some(|a, b, vm| {
            number_inplace_bin_op(a, b, |x, y| x.floor_div(y), vm)
        }),
        inplace_remainder: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.remainder(y), vm)),
        inplace_power: Some(|a, b, _modulo, vm| {
            let a_py = a
                .downcast_ref::<PyNdArray>()
                .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
            let b_arr = obj_to_ndarray(b, vm)?;
            let a_inner = a_py.data.read().unwrap().clone();
            let result = a_inner
                .pow(&b_arr)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            *a_py.data.write().unwrap() = result;
            Ok(a.to_owned())
        }),
        inplace_matrix_multiply: Some(|a, b, vm| {
            let a_py = a
                .downcast_ref::<PyNdArray>()
                .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
            let b_arr = obj_to_ndarray(b, vm)?;
            let a_inner = a_py.data.read().unwrap().clone();
            let result = numpy_rust_core::dot(&a_inner, &b_arr)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            *a_py.data.write().unwrap() = result;
            Ok(a.to_owned())
        }),
        xor: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.bitwise_xor(y), vm)),
        lshift: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.left_shift(y), vm)),
        rshift: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.right_shift(y), vm)),
        inplace_and: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.bitwise_and(y), vm)),
        inplace_or: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.bitwise_or(y), vm)),
        inplace_xor: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.bitwise_xor(y), vm)),
        inplace_lshift: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.left_shift(y), vm)),
        inplace_rshift: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.right_shift(y), vm)),
        absolute: Some(number_absolute),
        boolean: Some(number_boolean),
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
                let data = zelf.data.read().unwrap();
                Ok(if data.ndim() == 0 { 0 } else { data.shape()[0] })
            }),
            subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                zelf.__getitem__(needle.to_owned(), vm)
            }),
            ass_subscript: atomic_func!(|mapping, needle: &vm::PyObject, value, vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                match value {
                    Some(value) => setitem_impl(zelf, needle.to_owned(), value.to_owned(), vm),
                    None => {
                        Err(vm.new_type_error("ndarray does not support item deletion".to_owned()))
                    }
                }
            }),
        });
        &AS_MAPPING
    }
}

impl Representable for PyNdArray {
    fn repr_str(zelf: &vm::Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        let data = zelf.data.read().unwrap();
        Ok(format_array(&data, true))
    }
}

impl AsSequence for PyNdArray {
    fn as_sequence() -> &'static PySequenceMethods {
        use once_cell::sync::Lazy;
        static AS_SEQUENCE: Lazy<PySequenceMethods> = Lazy::new(|| PySequenceMethods {
            length: atomic_func!(|seq, _vm| {
                let zelf = PyNdArray::sequence_downcast(seq);
                let data = zelf.data.read().unwrap();
                Ok(if data.ndim() == 0 { 0 } else { data.shape()[0] })
            }),
            ..PySequenceMethods::NOT_IMPLEMENTED
        });
        &AS_SEQUENCE
    }
}

impl Iterable for PyNdArray {
    fn iter(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult {
        let data = zelf.data.read().unwrap();
        let length = if data.ndim() == 0 {
            return Err(vm.new_type_error("iteration over a 0-d array".to_owned()));
        } else {
            data.shape()[0]
        };
        drop(data);
        Ok(PyNdArrayIter {
            array: zelf,
            index: std::sync::atomic::AtomicUsize::new(0),
            length,
        }
        .into_pyobject(vm))
    }
}

/// Extract integer indices from an NdArray, resolving negative indices.
fn extract_int_indices(
    idx_arr: &NdArray,
    dim_size: usize,
    vm: &VirtualMachine,
) -> PyResult<Vec<usize>> {
    let flat = idx_arr.flatten();
    let mut out = Vec::with_capacity(flat.size());
    for i in 0..flat.size() {
        let s = flat.get(&[i]).map_err(|e| numpy_err(e, vm))?;
        let v: i64 = match s {
            Scalar::Int32(v) => v as i64,
            Scalar::Int64(v) => v,
            Scalar::Float32(v) => v as i64,
            Scalar::Float64(v) => v as i64,
            Scalar::Bool(v) => v as i64,
            _ => return Err(vm.new_type_error("expected integer index".to_owned())),
        };
        let resolved = if v < 0 {
            (dim_size as i64 + v) as usize
        } else {
            v as usize
        };
        if resolved >= dim_size {
            return Err(vm.new_value_error(format!(
                "index {v} is out of bounds for axis with size {dim_size}"
            )));
        }
        out.push(resolved);
    }
    Ok(out)
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
pub fn parse_optional_axis(
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

// --- __setitem__ implementation ---

fn setitem_impl(
    zelf: &PyNdArray,
    key: PyObjectRef,
    value: PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<()> {
    // Integer key → set single element or row
    if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
        let data = zelf.data.read().unwrap();
        let shape = data.shape().to_vec();
        let ndim = data.ndim();
        let dtype = data.dtype();
        drop(data);

        let resolved = if idx < 0 {
            (shape[0] as i64 + idx) as usize
        } else {
            idx as usize
        };

        if ndim == 1 {
            // Scalar assignment: a[i] = value
            let scalar = py_obj_to_scalar(&value, dtype, vm)?;
            zelf.data
                .write()
                .unwrap()
                .set(&[resolved], scalar)
                .map_err(|e| numpy_err(e, vm))?;
        } else {
            // Row assignment: a[i] = array_value
            let value_arr = obj_to_ndarray(&value, vm)?;
            zelf.data
                .write()
                .unwrap()
                .set_slice(&[SliceArg::Index(idx as isize)], &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
        }
        return Ok(());
    }

    // Slice key → a[start:stop:step] = values
    if let Some(slice) = key.downcast_ref::<PySlice>() {
        let arg = py_slice_to_slice_arg(slice, vm)?;
        let value_arr = obj_to_ndarray(&value, vm)?;
        zelf.data
            .write()
            .unwrap()
            .set_slice(&[arg], &value_arr)
            .map_err(|e| numpy_err(e, vm))?;
        return Ok(());
    }

    // Tuple key → multi-dimensional assignment
    if let Some(tuple) = key.downcast_ref::<PyTuple>() {
        let items = tuple.as_slice();
        let has_slice = items
            .iter()
            .any(|item| item.downcast_ref::<PySlice>().is_some());

        if has_slice {
            let args: Vec<SliceArg> = items
                .iter()
                .map(|item| py_obj_to_slice_arg(item, vm))
                .collect::<PyResult<Vec<_>>>()?;
            let value_arr = obj_to_ndarray(&value, vm)?;
            zelf.data
                .write()
                .unwrap()
                .set_slice(&args, &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        }

        // All integers → scalar assignment at multi-dim index
        let mut indices = Vec::new();
        for item in items {
            let i: i64 = item.clone().try_into_value(vm)?;
            indices.push(i as isize);
        }
        let data = zelf.data.read().unwrap();
        let shape = data.shape().to_vec();
        let dtype = data.dtype();
        drop(data);

        let usize_indices: Vec<usize> = indices
            .iter()
            .enumerate()
            .map(|(axis, &i)| {
                if i < 0 {
                    (shape[axis] as isize + i) as usize
                } else {
                    i as usize
                }
            })
            .collect();
        let scalar = py_obj_to_scalar(&value, dtype, vm)?;
        zelf.data
            .write()
            .unwrap()
            .set(&usize_indices, scalar)
            .map_err(|e| numpy_err(e, vm))?;
        return Ok(());
    }

    // NdArray key: boolean mask or integer fancy indexing
    if let Some(arr) = key.downcast_ref::<PyNdArray>() {
        let arr_data = arr.data.read().unwrap();
        if arr_data.dtype() == DType::Bool {
            let value_arr = obj_to_ndarray(&value, vm)?;
            let mask_owned = arr_data.clone();
            drop(arr_data);
            zelf.data
                .write()
                .unwrap()
                .mask_set(&mask_owned, &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        } else if arr_data.dtype().is_integer() || arr_data.dtype().is_float() {
            let dim_size = zelf.data.read().unwrap().shape()[0];
            let indices = extract_int_indices(&arr_data, dim_size, vm)?;
            drop(arr_data);
            let value_arr = obj_to_ndarray(&value, vm)?;
            zelf.data
                .write()
                .unwrap()
                .index_set(0, &indices, &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        }
    }

    // List of integers -> fancy indexing assignment: a[[0, 2]] = values
    if let Some(list) = key.downcast_ref::<PyList>() {
        let items = list.borrow_vec();
        let dim_size = zelf.data.read().unwrap().shape()[0];
        let mut indices = Vec::with_capacity(items.len());
        for item in items.iter() {
            let i: i64 = item.clone().try_into_value(vm)?;
            let resolved = if i < 0 {
                (dim_size as i64 + i) as usize
            } else {
                i as usize
            };
            indices.push(resolved);
        }
        let value_arr = obj_to_ndarray(&value, vm)?;
        zelf.data
            .write()
            .unwrap()
            .index_set(0, &indices, &value_arr)
            .map_err(|e| numpy_err(e, vm))?;
        return Ok(());
    }

    Err(vm.new_type_error("unsupported index type for assignment".to_owned()))
}

// --- tolist helper ---

fn ndarray_to_pylist(data: &NdArray, vm: &VirtualMachine) -> PyObjectRef {
    let ndim = data.ndim();
    let shape = data.shape();

    if ndim == 0 {
        // 0-D: return plain scalar
        let s = data.get(&[]).unwrap();
        return scalar_to_py(s, vm);
    }

    if ndim == 1 {
        // 1-D: build flat list
        let items: Vec<PyObjectRef> = (0..shape[0])
            .map(|i| {
                let s = data.get(&[i]).unwrap();
                scalar_to_py(s, vm)
            })
            .collect();
        return PyList::from(items).into_ref(&vm.ctx).into();
    }

    // N-D: recursively slice along first axis
    let items: Vec<PyObjectRef> = (0..shape[0])
        .map(|i| {
            let sub = data.slice(&[SliceArg::Index(i as isize)]).unwrap();
            ndarray_to_pylist(&sub, vm)
        })
        .collect();
    PyList::from(items).into_ref(&vm.ctx).into()
}

// --- repr/str formatting ---

/// Format a scalar value for repr output.
fn format_scalar(s: &Scalar) -> String {
    match s {
        Scalar::Bool(v) => {
            if *v {
                " True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        Scalar::Int32(v) => v.to_string(),
        Scalar::Int64(v) => v.to_string(),
        Scalar::Float32(v) => format_float(*v as f64),
        Scalar::Float64(v) => format_float(*v),
        Scalar::Complex64(v) => {
            if v.im >= 0.0 {
                format!("({}+{}j)", v.re, v.im)
            } else {
                format!("({}{}j)", v.re, v.im)
            }
        }
        Scalar::Complex128(v) => {
            if v.im >= 0.0 {
                format!("({}+{}j)", v.re, v.im)
            } else {
                format!("({}{}j)", v.re, v.im)
            }
        }
        Scalar::Str(v) => format!("'{v}'"),
    }
}

/// Format a float like NumPy: use "." suffix for integer-valued floats.
fn format_float(v: f64) -> String {
    if v.is_nan() {
        "nan".to_owned()
    } else if v.is_infinite() {
        if v > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        }
    } else if v == v.trunc() && v.abs() < 1e16 {
        // Integer-valued float: show as "5." not "5.0"
        format!("{v:.1}")
    } else {
        // Use default float formatting, trim trailing zeros
        let s = format!("{v}");
        s
    }
}

/// Max elements to show per axis edge before truncating.
const EDGE_ITEMS: usize = 3;
/// Threshold for total elements to trigger truncation.
const THRESHOLD: usize = 1000;

fn format_array(data: &NdArray, with_array_prefix: bool) -> String {
    let ndim = data.ndim();

    if ndim == 0 {
        let s = data.get(&[]).unwrap();
        let val = format_scalar(&s);
        return if with_array_prefix {
            format!("array({val})")
        } else {
            val
        };
    }

    let body = format_array_inner(data, 0);

    if with_array_prefix {
        format!("array({body})")
    } else {
        body
    }
}

fn format_array_inner(data: &NdArray, depth: usize) -> String {
    let ndim = data.ndim();
    let shape = data.shape();

    if ndim == 1 {
        // 1-D array
        let n = shape[0];
        let truncate = data.size() > THRESHOLD && n > EDGE_ITEMS * 2;

        let mut parts = Vec::new();
        if truncate {
            for i in 0..EDGE_ITEMS {
                let s = data.get(&[i]).unwrap();
                parts.push(format_scalar(&s));
            }
            parts.push("...".to_owned());
            for i in (n - EDGE_ITEMS)..n {
                let s = data.get(&[i]).unwrap();
                parts.push(format_scalar(&s));
            }
        } else {
            for i in 0..n {
                let s = data.get(&[i]).unwrap();
                parts.push(format_scalar(&s));
            }
        }
        return format!("[{}]", parts.join(", "));
    }

    // N-D: recurse along first axis
    let n = shape[0];
    let total_size = data.size();
    let truncate = total_size > THRESHOLD && n > EDGE_ITEMS * 2;

    let indent = " ".repeat(depth + 1);
    let mut parts = Vec::new();

    let indices: Vec<usize> = if truncate {
        let mut v: Vec<usize> = (0..EDGE_ITEMS).collect();
        v.push(usize::MAX); // sentinel for "..."
        v.extend((n - EDGE_ITEMS)..n);
        v
    } else {
        (0..n).collect()
    };

    for &i in &indices {
        if i == usize::MAX {
            parts.push(format!("{indent}..."));
            continue;
        }
        let sub = data.slice(&[SliceArg::Index(i as isize)]).unwrap();
        let sub_str = format_array_inner(&sub, depth + 1);
        parts.push(format!("{indent}{sub_str}"));
    }

    let sep = if ndim == 2 { ",\n" } else { ",\n\n" };
    format!("[{}]", parts.join(sep))
}
