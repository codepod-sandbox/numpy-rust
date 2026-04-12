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
    string_width: Option<usize>,
}

impl ArrayStorage {
    pub fn new() -> Self {
        Self::from_array_data(ArrayData::Float64(ArrayD::zeros(IxDyn(&[])).into_shared()))
    }

    pub fn from_array_data(data: ArrayData) -> Self {
        let string_width = string_width_from_data(&data);
        Self {
            kind: StorageKind::Backend(data),
            string_width,
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

    pub(crate) fn string_width(&self) -> Option<usize> {
        self.string_width
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

pub(crate) fn scalar_to_array_data(value: &crate::indexing::Scalar) -> ArrayData {
    match value {
        crate::indexing::Scalar::Bool(v) => {
            ArrayData::Bool(ArrayD::from_elem(IxDyn(&[]), *v).into_shared())
        }
        crate::indexing::Scalar::Int32(v) => {
            ArrayData::Int32(ArrayD::from_elem(IxDyn(&[]), *v).into_shared())
        }
        crate::indexing::Scalar::Int64(v) => {
            ArrayData::Int64(ArrayD::from_elem(IxDyn(&[]), *v).into_shared())
        }
        crate::indexing::Scalar::Float32(v) => {
            ArrayData::Float32(ArrayD::from_elem(IxDyn(&[]), *v).into_shared())
        }
        crate::indexing::Scalar::Float64(v) => {
            ArrayData::Float64(ArrayD::from_elem(IxDyn(&[]), *v).into_shared())
        }
        crate::indexing::Scalar::Complex64(v) => {
            ArrayData::Complex64(ArrayD::from_elem(IxDyn(&[]), *v).into_shared())
        }
        crate::indexing::Scalar::Complex128(v) => {
            ArrayData::Complex128(ArrayD::from_elem(IxDyn(&[]), *v).into_shared())
        }
        crate::indexing::Scalar::Str(v) => {
            ArrayData::Str(ArrayD::from_elem(IxDyn(&[]), v.clone()).into_shared())
        }
    }
}

pub(crate) fn array_data_to_scalar(data: &ArrayData) -> crate::indexing::Scalar {
    match data {
        ArrayData::Bool(a) => crate::indexing::Scalar::Bool(a[IxDyn(&[])]),
        ArrayData::Int32(a) => crate::indexing::Scalar::Int32(a[IxDyn(&[])]),
        ArrayData::Int64(a) => crate::indexing::Scalar::Int64(a[IxDyn(&[])]),
        ArrayData::Float32(a) => crate::indexing::Scalar::Float32(a[IxDyn(&[])]),
        ArrayData::Float64(a) => crate::indexing::Scalar::Float64(a[IxDyn(&[])]),
        ArrayData::Complex64(a) => crate::indexing::Scalar::Complex64(a[IxDyn(&[])]),
        ArrayData::Complex128(a) => crate::indexing::Scalar::Complex128(a[IxDyn(&[])]),
        ArrayData::Str(a) => crate::indexing::Scalar::Str(a[IxDyn(&[])].clone()),
    }
}

pub(crate) fn normalize_string_assignment(target: &ArrayStorage, value: ArrayData) -> ArrayData {
    let Some(width) = target.string_width() else {
        return value;
    };

    match value {
        ArrayData::Str(values) => ArrayData::Str(
            values
                .mapv(|s| truncate_string_to_width(s, width))
                .into_shared(),
        ),
        other => other,
    }
}

fn string_width_from_data(data: &ArrayData) -> Option<usize> {
    match data {
        ArrayData::Str(values) => Some(values.iter().map(|s| s.chars().count()).max().unwrap_or(0)),
        _ => None,
    }
}

fn truncate_string_to_width(value: String, width: usize) -> String {
    value.chars().take(width).collect()
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
