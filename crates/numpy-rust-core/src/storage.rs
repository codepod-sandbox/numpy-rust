use ndarray::IxDyn;
use num_complex::Complex;

use crate::array::StorageMutationGuard;
use crate::array_data::{ArrayD, ArrayData};
use crate::descriptor::DTypeDescriptor;
use crate::dtype::DType;

#[derive(Debug, Clone)]
pub enum StorageKind {
    Backend(ArrayData),
}

#[derive(Debug, Clone)]
pub struct ArrayStorage {
    kind: StorageKind,
}

impl ArrayStorage {
    pub fn new() -> Self {
        Self::from_array_data(ArrayData::Float64(ArrayD::zeros(IxDyn(&[])).into_shared()))
    }

    pub fn from_array_data(data: ArrayData) -> Self {
        Self {
            kind: StorageKind::Backend(data),
        }
    }

    pub fn into_array_data(self) -> ArrayData {
        match self.kind {
            StorageKind::Backend(data) => data,
        }
    }

    pub fn data(&self) -> &ArrayData {
        match &self.kind {
            StorageKind::Backend(data) => data,
        }
    }

    fn with_data_mut<R>(&mut self, mutate: impl FnOnce(&mut ArrayData) -> R) -> R {
        match &mut self.kind {
            StorageKind::Backend(data) => mutate(data),
        }
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        matches!(
            (&self.kind, &other.kind),
            (StorageKind::Backend(a), StorageKind::Backend(b)) if a.shares_memory_with(b)
        )
    }

    pub fn zeros(shape: &[usize], descriptor: &'static DTypeDescriptor) -> Self {
        Self::from_array_data(fill(shape, descriptor.id.storage_dtype(), FillValue::Zero))
    }

    pub fn ones(shape: &[usize], descriptor: &'static DTypeDescriptor) -> Self {
        Self::from_array_data(fill(shape, descriptor.id.storage_dtype(), FillValue::One))
    }
}

impl Default for ArrayStorage {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn mutate_storage_with_guard<R>(
    storage: &mut ArrayStorage,
    _guard: StorageMutationGuard,
    mutate: impl FnOnce(&mut ArrayData) -> R,
) -> R {
    storage.with_data_mut(mutate)
}

#[derive(Clone, Copy)]
enum FillValue {
    Zero,
    One,
}

fn fill(shape: &[usize], dtype: DType, fill: FillValue) -> ArrayData {
    let sh = IxDyn(shape);

    match dtype {
        DType::Bool => {
            ArrayData::Bool(ArrayD::from_elem(sh, matches!(fill, FillValue::One)).into_shared())
        }
        DType::Int32 => ArrayData::Int32(match fill {
            FillValue::Zero => ArrayD::zeros(sh).into_shared(),
            FillValue::One => ArrayD::ones(sh).into_shared(),
        }),
        DType::Int64 => ArrayData::Int64(match fill {
            FillValue::Zero => ArrayD::zeros(sh).into_shared(),
            FillValue::One => ArrayD::ones(sh).into_shared(),
        }),
        DType::Float32 => ArrayData::Float32(match fill {
            FillValue::Zero => ArrayD::zeros(sh).into_shared(),
            FillValue::One => ArrayD::ones(sh).into_shared(),
        }),
        DType::Float64 => ArrayData::Float64(match fill {
            FillValue::Zero => ArrayD::zeros(sh).into_shared(),
            FillValue::One => ArrayD::ones(sh).into_shared(),
        }),
        DType::Complex64 => ArrayData::Complex64(
            ArrayD::from_elem(
                sh,
                match fill {
                    FillValue::Zero => Complex::new(0.0f32, 0.0),
                    FillValue::One => Complex::new(1.0f32, 0.0),
                },
            )
            .into_shared(),
        ),
        DType::Complex128 => ArrayData::Complex128(
            ArrayD::from_elem(
                sh,
                match fill {
                    FillValue::Zero => Complex::new(0.0f64, 0.0),
                    FillValue::One => Complex::new(1.0f64, 0.0),
                },
            )
            .into_shared(),
        ),
        DType::Str => ArrayData::Str(
            ArrayD::from_elem(
                sh,
                match fill {
                    FillValue::Zero => String::new(),
                    FillValue::One => "1".to_string(),
                },
            )
            .into_shared(),
        ),
        _ => unreachable!("storage dtypes are canonical backend types"),
    }
}
