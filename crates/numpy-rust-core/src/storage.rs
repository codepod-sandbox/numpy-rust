use ndarray::IxDyn;
use num_complex::Complex;

use crate::array::StorageMutationGuard;
use crate::array_data::{ArrayD, ArrayData};
use crate::descriptor::DTypeDescriptor;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};

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

    pub fn full_f64(shape: &[usize], descriptor: &'static DTypeDescriptor, value: f64) -> Self {
        let data = fill_f64(shape, descriptor.id, value);
        Self::from_array_data(data)
    }

    pub fn eye(
        rows: usize,
        cols: usize,
        k: isize,
        descriptor: &'static DTypeDescriptor,
    ) -> Result<Self> {
        if descriptor.id.is_string() {
            return Err(NumpyError::TypeError(
                "eye() not supported for string dtype".into(),
            ));
        }

        let mut data = fill(
            &[rows, cols],
            descriptor.id.storage_dtype(),
            FillValue::Zero,
        );
        write_eye_diagonal(&mut data, rows, cols, k)?;
        Ok(Self::from_array_data(data))
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

fn fill_f64(shape: &[usize], dtype: DType, value: f64) -> ArrayData {
    let sh = IxDyn(shape);
    let storage_dtype = dtype.storage_dtype();
    let data = match storage_dtype {
        DType::Bool => ArrayData::Bool(ArrayD::from_elem(sh, value != 0.0).into_shared()),
        DType::Int32 => ArrayData::Int32(ArrayD::from_elem(sh, value as i32).into_shared()),
        DType::Int64 => ArrayData::Int64(ArrayD::from_elem(sh, value as i64).into_shared()),
        DType::Float32 => ArrayData::Float32(ArrayD::from_elem(sh, value as f32).into_shared()),
        DType::Float64 => ArrayData::Float64(ArrayD::from_elem(sh, value).into_shared()),
        DType::Complex64 => ArrayData::Complex64(
            ArrayD::from_elem(sh, Complex::new(value as f32, 0.0)).into_shared(),
        ),
        DType::Complex128 => {
            ArrayData::Complex128(ArrayD::from_elem(sh, Complex::new(value, 0.0)).into_shared())
        }
        DType::Str => ArrayData::Str(ArrayD::from_elem(sh, value.to_string()).into_shared()),
        _ => unreachable!("storage dtypes are canonical backend types"),
    };

    if dtype.is_narrow() {
        crate::casting::narrow_truncate(data, dtype)
    } else {
        data
    }
}

fn write_eye_diagonal(data: &mut ArrayData, rows: usize, cols: usize, k: isize) -> Result<()> {
    match data {
        ArrayData::Bool(a) => {
            for i in 0..rows {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = true;
                }
            }
        }
        ArrayData::Int32(a) => {
            for i in 0..rows {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1;
                }
            }
        }
        ArrayData::Int64(a) => {
            for i in 0..rows {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1;
                }
            }
        }
        ArrayData::Float32(a) => {
            for i in 0..rows {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1.0;
                }
            }
        }
        ArrayData::Float64(a) => {
            for i in 0..rows {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = 1.0;
                }
            }
        }
        ArrayData::Complex64(a) => {
            for i in 0..rows {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = Complex::new(1.0f32, 0.0);
                }
            }
        }
        ArrayData::Complex128(a) => {
            for i in 0..rows {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < cols {
                    a[[i, j as usize]] = Complex::new(1.0f64, 0.0);
                }
            }
        }
        ArrayData::Str(_) => {
            return Err(NumpyError::TypeError(
                "eye() not supported for string dtype".into(),
            ));
        }
    }

    Ok(())
}
