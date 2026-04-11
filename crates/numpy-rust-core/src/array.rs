use crate::array_data::ArrayD;
use ndarray::{IxDyn, SliceInfoElem};
use num_complex::Complex;

use crate::array_data::ArrayData;
use crate::descriptor::{descriptor_for_dtype, DTypeDescriptor};
use crate::dtype::DType;
use crate::resolver::{BinaryOpPlan, CastPlan};
use crate::storage::ArrayStorage;

/// The main N-dimensional array type, analogous to `numpy.ndarray`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayFlags {
    pub c_contiguous: bool,
    pub f_contiguous: bool,
    pub writeable: bool,
}

impl ArrayFlags {
    fn from_data(data: &ArrayData) -> Self {
        Self {
            c_contiguous: data.is_c_contiguous(),
            f_contiguous: data.is_f_contiguous(),
            writeable: true,
        }
    }
}

pub(crate) struct StorageMutationGuard(());

impl StorageMutationGuard {
    fn new() -> Self {
        Self(())
    }
}

struct ValidatedRuntimeState {
    descriptor: &'static DTypeDescriptor,
    shape: Vec<usize>,
    strides: Vec<isize>,
    flags: ArrayFlags,
}

impl ValidatedRuntimeState {
    fn new(storage: &ArrayStorage, descriptor: &'static DTypeDescriptor) -> Self {
        let data = storage.data();
        assert_eq!(
            data.dtype(),
            descriptor.id.storage_dtype(),
            "descriptor storage dtype must match backing storage"
        );
        Self {
            descriptor,
            shape: data.shape().to_vec(),
            strides: data.strides(),
            flags: ArrayFlags::from_data(data),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NdArray {
    storage: ArrayStorage,
    descriptor: &'static DTypeDescriptor,
    shape: Vec<usize>,
    strides: Vec<isize>,
    flags: ArrayFlags,
}

// --- Constructors ---

impl NdArray {
    /// Create an NdArray from existing ArrayData.
    pub fn from_data(data: ArrayData) -> Self {
        let descriptor = descriptor_for_dtype(data.dtype());
        Self::from_parts(ArrayStorage::from_array_data(data), descriptor)
    }

    fn from_parts(storage: ArrayStorage, descriptor: &'static DTypeDescriptor) -> Self {
        let runtime = ValidatedRuntimeState::new(&storage, descriptor);
        Self {
            descriptor: runtime.descriptor,
            shape: runtime.shape,
            strides: runtime.strides,
            flags: runtime.flags,
            storage,
        }
    }

    /// Set a declared dtype override (for narrow types stored as wider types).
    pub fn with_declared_dtype(mut self, dtype: DType) -> Self {
        self.set_declared_dtype(dtype);
        self
    }

    /// Preserve declared_dtype from another NdArray (used in shape-only operations).
    pub fn with_preserved_dtype(mut self, source: &NdArray) -> Self {
        self.preserve_descriptor_from(source);
        self
    }

    /// Create an array filled with zeros.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let descriptor = descriptor_for_dtype(dtype);
        let storage = ArrayStorage::zeros(shape, descriptor);
        Self::from_parts(storage, descriptor)
    }

    /// Create an array filled with ones.
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let descriptor = descriptor_for_dtype(dtype);
        let storage = ArrayStorage::ones(shape, descriptor);
        Self::from_parts(storage, descriptor)
    }

    /// Create an array filled with a given f64 value.
    pub fn full_f64(shape: &[usize], value: f64) -> Self {
        let sh = IxDyn(shape);
        Self::from_data(ArrayData::Float64(
            ArrayD::from_elem(sh, value).into_shared(),
        ))
    }

    /// Create a 0-dimensional (scalar) array from an f64 value.
    pub fn from_scalar(value: f64) -> Self {
        Self::from_data(ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), value)))
    }
}

// --- Attributes ---

impl NdArray {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn dtype(&self) -> DType {
        self.descriptor.id
    }

    pub fn declared_dtype(&self) -> Option<DType> {
        logical_dtype_override(self.descriptor.id)
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn data(&self) -> &ArrayData {
        self.storage.data()
    }

    pub fn descriptor(&self) -> &'static DTypeDescriptor {
        self.descriptor
    }

    pub fn storage(&self) -> &ArrayStorage {
        &self.storage
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn flags(&self) -> ArrayFlags {
        self.flags
    }

    /// Cast this array to a different dtype, following NumPy's astype semantics.
    pub fn astype(&self, dtype: DType) -> Self {
        let storage = dtype.storage_dtype();
        let data = crate::casting::cast_array_data(self.data(), storage);
        let data = if dtype.is_narrow() {
            crate::casting::narrow_truncate(data, dtype)
        } else {
            data
        };
        let descriptor = descriptor_for_dtype(dtype);
        Self::from_parts(ArrayStorage::from_array_data(data), descriptor)
    }

    pub(crate) fn cast_for_execution(&self, plan: CastPlan) -> ArrayData {
        crate::casting::cast_array_data(self.data(), plan.execution_storage_dtype())
    }

    pub(crate) fn from_binary_plan_result(mut data: ArrayData, plan: BinaryOpPlan) -> Self {
        if plan.requires_output_narrowing() {
            data = crate::casting::narrow_truncate(data, plan.output_dtype());
        }

        let mut result = Self::from_data(data);
        if plan.requires_output_narrowing() {
            result.set_declared_dtype(plan.output_dtype());
        }
        result
    }

    pub fn slice_axis_for_test(&self, start: usize, end: usize) -> Self {
        let info = [SliceInfoElem::Slice {
            start: start as isize,
            end: Some(end as isize),
            step: 1,
        }];

        macro_rules! do_slice {
            ($arr:expr, $variant:ident) => {
                ArrayData::$variant($arr.clone().slice_move(info.as_slice()))
            };
        }

        let data = match self.data() {
            ArrayData::Bool(a) => do_slice!(a, Bool),
            ArrayData::Int32(a) => do_slice!(a, Int32),
            ArrayData::Int64(a) => do_slice!(a, Int64),
            ArrayData::Float32(a) => do_slice!(a, Float32),
            ArrayData::Float64(a) => do_slice!(a, Float64),
            ArrayData::Complex64(a) => do_slice!(a, Complex64),
            ArrayData::Complex128(a) => do_slice!(a, Complex128),
            ArrayData::Str(a) => do_slice!(a, Str),
        };

        Self::from_data(data).with_preserved_dtype(self)
    }

    pub(crate) fn set_declared_dtype(&mut self, dtype: DType) {
        let runtime = ValidatedRuntimeState::new(&self.storage, descriptor_for_dtype(dtype));
        self.apply_runtime_state(runtime);
    }

    pub(crate) fn preserve_descriptor_from(&mut self, source: &NdArray) {
        let runtime = ValidatedRuntimeState::new(&self.storage, source.descriptor());
        self.apply_runtime_state(runtime);
    }

    /// Applies an in-place mutation and then re-synchronizes the stored runtime metadata.
    /// The callback must preserve the storage dtype that matches the current descriptor.
    pub(crate) fn mutate_data<R>(&mut self, mutate: impl FnOnce(&mut ArrayData) -> R) -> R {
        let mut storage = self.storage.clone();
        let result = crate::storage::mutate_storage_with_guard(
            &mut storage,
            StorageMutationGuard::new(),
            mutate,
        );
        let runtime = ValidatedRuntimeState::new(&storage, self.descriptor);
        self.commit_runtime_state(storage, runtime);
        result
    }

    pub(crate) fn replace_data_with_dtype(&mut self, data: ArrayData, dtype: DType) {
        let storage = ArrayStorage::from_array_data(data);
        let runtime = ValidatedRuntimeState::new(&storage, descriptor_for_dtype(dtype));
        self.commit_runtime_state(storage, runtime);
    }

    fn apply_runtime_state(&mut self, runtime: ValidatedRuntimeState) {
        self.descriptor = runtime.descriptor;
        self.shape = runtime.shape;
        self.strides = runtime.strides;
        self.flags = runtime.flags;
    }

    fn commit_runtime_state(&mut self, storage: ArrayStorage, runtime: ValidatedRuntimeState) {
        self.storage = storage;
        self.apply_runtime_state(runtime);
    }
}

fn logical_dtype_override(dtype: DType) -> Option<DType> {
    (dtype.storage_dtype() != dtype).then_some(dtype)
}

// --- Trait for converting Vec<T> to ArrayData ---

pub trait IntoArrayData {
    fn into_array_data(self) -> ArrayData;
}

impl IntoArrayData for Vec<f64> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Float64(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<f32> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Float32(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<i32> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Int32(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<i64> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Int64(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<bool> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Bool(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<String> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Str(ArrayD::from_shape_vec(IxDyn(&[len]), self).unwrap())
    }
}

impl IntoArrayData for Vec<Complex<f32>> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Complex64(
            ArrayD::from_shape_vec(IxDyn(&[len]), self).expect("vec length matches 1-D shape"),
        )
    }
}

impl IntoArrayData for Vec<Complex<f64>> {
    fn into_array_data(self) -> ArrayData {
        let len = self.len();
        ArrayData::Complex128(
            ArrayD::from_shape_vec(IxDyn(&[len]), self).expect("vec length matches 1-D shape"),
        )
    }
}

impl NdArray {
    pub fn from_vec<V: IntoArrayData>(vec: V) -> Self {
        Self::from_data(vec.into_array_data())
    }

    /// Convenience constructor for a 1-D Complex128 array.
    pub fn from_complex128_vec(vec: Vec<Complex<f64>>) -> Self {
        Self::from_vec(vec)
    }

    /// Convenience constructor for a 1-D Complex64 array.
    pub fn from_complex64_vec(vec: Vec<Complex<f32>>) -> Self {
        Self::from_vec(vec)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::*;

    #[test]
    fn test_from_f64_vec() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.dtype(), DType::Float64);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_from_i32_vec() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        assert_eq!(a.dtype(), DType::Int32);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_from_bool_vec() {
        let a = NdArray::from_vec(vec![true, false, true]);
        assert_eq!(a.dtype(), DType::Bool);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_zeros() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.size(), 6);
    }

    #[test]
    fn test_zeros_complex() {
        let a = NdArray::zeros(&[2, 3], DType::Complex128);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.dtype(), DType::Complex128);
    }

    #[test]
    fn test_ones() {
        let a = NdArray::ones(&[3], DType::Int32);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), DType::Int32);
    }

    #[test]
    fn test_ones_complex() {
        let a = NdArray::ones(&[3], DType::Complex128);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), DType::Complex128);
    }

    #[test]
    fn test_full_f64() {
        let a = NdArray::full_f64(&[2, 2], 3.125);
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.dtype(), DType::Float64);
        assert_eq!(a.size(), 4);
    }

    #[test]
    fn test_from_data() {
        let data = ArrayData::Float32(ArrayD::zeros(IxDyn(&[5])));
        let a = NdArray::from_data(data);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.dtype(), DType::Float32);
    }

    #[test]
    fn test_from_data_stores_runtime_spine_fields() {
        let data = ArrayData::Float32(ArrayD::zeros(IxDyn(&[5])));
        let a = NdArray::from_data(data);
        assert!(matches!(a.storage().data(), ArrayData::Float32(_)));
        assert_eq!(a.descriptor().id, DType::Float32);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.strides(), &[1]);
        assert!(a.flags().c_contiguous);
        assert!(a.flags().f_contiguous);
    }

    #[test]
    fn test_mutate_data_refreshes_runtime_metadata() {
        let mut a = NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        ));

        a.mutate_data(|data| match data {
            ArrayData::Float64(values) => {
                *values = values
                    .clone()
                    .into_shape_with_order(IxDyn(&[2, 3]))
                    .unwrap()
                    .into_shared();
            }
            other => panic!("expected Float64 storage, got {other:?}"),
        });

        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.strides(), &[3, 1]);
        assert!(a.flags().c_contiguous);
    }

    #[test]
    #[should_panic(expected = "descriptor storage dtype")]
    fn test_with_declared_dtype_rejects_incompatible_storage() {
        let _ = NdArray::from_vec(vec![1.0f64]).with_declared_dtype(DType::Int32);
    }

    #[test]
    #[should_panic(expected = "descriptor storage dtype")]
    fn test_with_preserved_dtype_rejects_incompatible_storage() {
        let source = NdArray::from_vec(vec![1_i32, 2, 3]);
        let _ = NdArray::from_vec(vec![1.0f64, 2.0, 3.0]).with_preserved_dtype(&source);
    }

    #[test]
    fn test_mutate_data_panic_leaves_runtime_state_consistent() {
        let mut a = NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        ));

        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            a.mutate_data(|data| match data {
                ArrayData::Float64(values) => {
                    *values = values
                        .clone()
                        .into_shape_with_order(IxDyn(&[2, 3]))
                        .unwrap()
                        .into_shared();
                    panic!("boom");
                }
                other => panic!("expected Float64 storage, got {other:?}"),
            });
        }));

        assert!(panic_result.is_err());
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.data().shape(), &[6]);
        assert_eq!(a.shape(), a.data().shape());
        assert_eq!(a.strides(), a.data().strides());
        assert_eq!(a.dtype().storage_dtype(), a.data().dtype());
    }

    #[test]
    fn test_replace_data_with_dtype_panic_leaves_runtime_state_consistent() {
        let mut a = NdArray::from_vec(vec![1.0f64]);

        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            a.replace_data_with_dtype(
                ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                        .unwrap()
                        .into_shared(),
                ),
                DType::Int32,
            );
        }));

        assert!(panic_result.is_err());
        assert_eq!(a.shape(), &[1]);
        assert_eq!(a.data().shape(), &[1]);
        assert_eq!(a.shape(), a.data().shape());
        assert_eq!(a.strides(), a.data().strides());
        assert_eq!(a.dtype().storage_dtype(), a.data().dtype());
    }

    #[test]
    fn test_from_complex_vec() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(a.dtype(), DType::Complex128);
        assert_eq!(a.size(), 2);
    }

    #[test]
    fn test_arcarray_shared_clone() {
        // ArrayD is now ArcArray — clone shares buffer (O(1))
        let a: ArrayD<f64> =
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = a.clone();
        assert_eq!(a.as_ptr(), b.as_ptr(), "clone shares buffer");

        // Mutation triggers CoW
        let mut c = a.clone();
        c[[0]] = 99.0;
        assert_ne!(c.as_ptr(), a.as_ptr(), "CoW on mutation");
        assert_eq!(a[[0]], 1.0, "original unchanged");
    }
}
