use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use ndarray::{IxDyn, SliceInfoElem};
use num_complex::Complex;

#[cfg(test)]
use crate::array::StorageMutationGuard;
use crate::array_data::{ArrayD, ArrayData};
use crate::descriptor::DTypeDescriptor;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};

#[derive(Debug)]
struct SharedBackend {
    data: Arc<RwLock<ArrayData>>,
    version: Arc<AtomicUsize>,
}

impl Clone for SharedBackend {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            version: Arc::clone(&self.version),
        }
    }
}

#[derive(Debug, Clone)]
struct ViewCache {
    version: usize,
    data: ArrayData,
}

#[derive(Debug)]
struct ViewStorage {
    base: Box<ArrayStorage>,
    info: Vec<SliceInfoElem>,
    cache: RwLock<ViewCache>,
}

impl Clone for ViewStorage {
    fn clone(&self) -> Self {
        let cache = self.cache.read().unwrap();
        Self {
            base: Box::new((*self.base).clone()),
            info: self.info.clone(),
            cache: RwLock::new(cache.clone()),
        }
    }
}

#[derive(Debug, Clone)]
enum StorageKind {
    Backend(SharedBackend),
    View(ViewStorage),
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
        Self::from_array_data_with_string_width(data, string_width)
    }

    pub(crate) fn from_array_data_with_string_width(
        data: ArrayData,
        string_width: Option<usize>,
    ) -> Self {
        Self {
            kind: StorageKind::Backend(SharedBackend {
                data: Arc::new(RwLock::new(data)),
                version: Arc::new(AtomicUsize::new(0)),
            }),
            string_width,
        }
    }

    pub fn into_array_data(self) -> ArrayData {
        match self.kind {
            StorageKind::Backend(backend) => backend.data.read().unwrap().clone(),
            StorageKind::View(view) => view.cache.into_inner().unwrap().data,
        }
    }

    pub fn data(&self) -> ArrayData {
        match &self.kind {
            StorageKind::Backend(backend) => backend.data.read().unwrap().clone(),
            StorageKind::View(view) => {
                let parent_version = view.base.version();
                let needs_refresh = {
                    let cache = view.cache.read().unwrap();
                    cache.version != parent_version
                };
                if needs_refresh {
                    let refreshed = view.base.data().slice_view(view.info.as_slice()).unwrap();
                    let mut cache = view.cache.write().unwrap();
                    cache.version = parent_version;
                    cache.data = refreshed;
                }
                view.cache.read().unwrap().data.clone()
            }
        }
    }

    pub(crate) fn string_width(&self) -> Option<usize> {
        self.string_width
    }

    fn with_data_mut<R>(&mut self, mutate: impl FnOnce(&mut ArrayData) -> R) -> R {
        match &mut self.kind {
            StorageKind::Backend(backend) => {
                let mut data = backend.data.write().unwrap();
                let result = mutate(&mut data);
                backend.version.fetch_add(1, Ordering::Relaxed);
                result
            }
            StorageKind::View(_) => {
                panic!("generic storage mutation is not supported for view storage")
            }
        }
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.data().shares_memory_with(&other.data())
    }

    pub(crate) fn version(&self) -> usize {
        match &self.kind {
            StorageKind::Backend(backend) => backend.version.load(Ordering::Relaxed),
            StorageKind::View(view) => view.base.version(),
        }
    }

    pub(crate) fn logical_offset_elements(&self) -> isize {
        match &self.kind {
            StorageKind::Backend(_) => 0,
            StorageKind::View(view) => {
                let base_data = view.base.data();
                let base_shape = base_data.shape().to_vec();
                let base_strides = base_data.strides().to_vec();
                let mut axis = 0usize;
                let mut offset = 0isize;
                for elem in &view.info {
                    match *elem {
                        SliceInfoElem::Index(index) => {
                            let len = base_shape[axis] as isize;
                            let idx = if index < 0 { index + len } else { index };
                            offset += idx * base_strides[axis];
                            axis += 1;
                        }
                        SliceInfoElem::Slice { start, .. } => {
                            let len = base_shape[axis] as isize;
                            let raw = start;
                            let idx = if raw < 0 { raw + len } else { raw };
                            offset += idx * base_strides[axis];
                            axis += 1;
                        }
                        SliceInfoElem::NewAxis => {}
                    }
                }
                view.base.logical_offset_elements() + offset
            }
        }
    }

    pub(crate) fn slice_view(&self, info: &[SliceInfoElem]) -> Result<Self> {
        let data = self.data().slice_view(info)?;
        Ok(Self {
            kind: StorageKind::View(ViewStorage {
                base: Box::new(self.clone()),
                info: info.to_vec(),
                cache: RwLock::new(ViewCache {
                    version: self.version(),
                    data,
                }),
            }),
            string_width: self.string_width,
        })
    }

    pub(crate) fn assign_element(
        &mut self,
        index: &[usize],
        value: crate::indexing::Scalar,
    ) -> Result<()> {
        match &mut self.kind {
            StorageKind::Backend(_) => {
                self.with_data_mut(|data| set_array_element(data, index, value))
            }
            StorageKind::View(view) => {
                let mapped = map_index_through_view(view.info.as_slice(), index)?;
                let result = view.base.assign_element(&mapped, value);
                if result.is_ok() {
                    refresh_view_cache(view)?;
                }
                result
            }
        }
    }

    pub(crate) fn assign_slice_values(
        &mut self,
        info: &[SliceInfoElem],
        values: &ArrayData,
    ) -> Result<()> {
        match &mut self.kind {
            StorageKind::Backend(_) => self.with_data_mut(|data| data.assign_slice(info, values)),
            StorageKind::View(view) => {
                let target = {
                    let cache = view.cache.read().unwrap();
                    cache.data.slice_view(info)?
                };
                assign_materialized_selection(
                    &mut view.base,
                    view.info.as_slice(),
                    &target,
                    values,
                )?;
                refresh_view_cache(view)
            }
        }
    }

    pub(crate) fn assign_indexed_values(
        &mut self,
        axis: usize,
        indices: &[usize],
        values: &ArrayData,
    ) -> Result<()> {
        match &mut self.kind {
            StorageKind::Backend(_) => {
                self.with_data_mut(|data| data.assign_indexed(axis, indices, values))
            }
            StorageKind::View(view) => {
                let target = {
                    let cache = view.cache.read().unwrap();
                    cache.data.index_select(axis, indices)?
                };
                assign_materialized_selection(
                    &mut view.base,
                    view.info.as_slice(),
                    &target,
                    values,
                )?;
                refresh_view_cache(view)
            }
        }
    }

    pub(crate) fn assign_masked_values(
        &mut self,
        flat_mask: &[bool],
        values: &ArrayData,
    ) -> Result<()> {
        match &mut self.kind {
            StorageKind::Backend(_) => {
                self.with_data_mut(|data| data.assign_masked(flat_mask, values))
            }
            StorageKind::View(view) => {
                let current = {
                    let cache = view.cache.read().unwrap();
                    cache.data.clone()
                };
                let coords = iter_array_scalars(&current)
                    .zip(flat_mask.iter().copied())
                    .filter_map(|(entry, keep)| keep.then_some(entry))
                    .map(|(coord, _)| coord)
                    .collect::<Vec<_>>();
                let target = current.select_masked(flat_mask);
                assign_scalar_list(
                    &mut view.base,
                    view.info.as_slice(),
                    coords.as_slice(),
                    &target,
                    values,
                )?;
                refresh_view_cache(view)
            }
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxedDType {
    Object,
    Datetime64,
    Timedelta64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoxedObjectScalar {
    Bool(bool),
    Int(i64),
    Float(f64),
    Complex(Complex<f64>),
    Text(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxedTemporalScalar {
    pub value: i64,
    pub unit: String,
    pub is_nat: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoxedScalar {
    Object(BoxedObjectScalar),
    Datetime(BoxedTemporalScalar),
    Timedelta(BoxedTemporalScalar),
}

#[derive(Debug, Clone)]
struct SharedBoxedBackend {
    data: Arc<RwLock<Vec<BoxedScalar>>>,
    version: Arc<AtomicUsize>,
}

#[derive(Debug, Clone)]
struct BoxedViewStorage {
    base: Box<BoxedStorage>,
    info: Vec<SliceInfoElem>,
}

#[derive(Debug, Clone)]
enum BoxedStorageKind {
    Backend(SharedBoxedBackend),
    View(BoxedViewStorage),
}

#[derive(Debug, Clone)]
pub struct BoxedStorage {
    kind: BoxedStorageKind,
    dtype: BoxedDType,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

impl BoxedStorage {
    pub fn from_scalars(
        elements: Vec<BoxedScalar>,
        shape: &[usize],
        dtype: BoxedDType,
    ) -> Result<Self> {
        let total = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| NumpyError::ValueError(format!("shape {:?} would overflow", shape)))?;
        if total != elements.len() {
            return Err(NumpyError::ShapeMismatch(format!(
                "boxed values length {} does not match shape {:?}",
                elements.len(),
                shape
            )));
        }
        for element in &elements {
            if !boxed_scalar_matches_dtype(element, dtype) {
                return Err(NumpyError::TypeError(format!(
                    "boxed scalar {element:?} does not match boxed dtype {dtype:?}"
                )));
            }
        }
        Ok(Self {
            kind: BoxedStorageKind::Backend(SharedBoxedBackend {
                data: Arc::new(RwLock::new(elements)),
                version: Arc::new(AtomicUsize::new(0)),
            }),
            dtype,
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
        })
    }

    pub fn dtype(&self) -> BoxedDType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn slice_view(&self, info: &[SliceInfoElem]) -> Result<Self> {
        let (shape, strides) = boxed_slice_shape_and_strides(&self.shape, &self.strides, info)?;
        Ok(Self {
            kind: BoxedStorageKind::View(BoxedViewStorage {
                base: Box::new(self.clone()),
                info: info.to_vec(),
            }),
            dtype: self.dtype,
            shape,
            strides,
        })
    }

    pub fn get(&self, index: &[usize]) -> Result<BoxedScalar> {
        match &self.kind {
            BoxedStorageKind::Backend(backend) => {
                let flat = row_major_flat_index(&self.shape, index)?;
                backend
                    .data
                    .read()
                    .unwrap()
                    .get(flat)
                    .cloned()
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))
            }
            BoxedStorageKind::View(view) => {
                let mapped = boxed_map_index_through_view(view.info.as_slice(), index)?;
                view.base.get(&mapped)
            }
        }
    }

    pub fn set(&mut self, index: &[usize], value: BoxedScalar) -> Result<()> {
        if !boxed_scalar_matches_dtype(&value, self.dtype) {
            return Err(NumpyError::TypeError(format!(
                "boxed scalar {value:?} does not match boxed dtype {:?}",
                self.dtype
            )));
        }
        match &mut self.kind {
            BoxedStorageKind::Backend(backend) => {
                let flat = row_major_flat_index(&self.shape, index)?;
                let mut data = backend.data.write().unwrap();
                let slot = data
                    .get_mut(flat)
                    .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
                *slot = value;
                backend.version.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            BoxedStorageKind::View(view) => {
                let mapped = boxed_map_index_through_view(view.info.as_slice(), index)?;
                view.base.set(&mapped, value)
            }
        }
    }

    pub fn elements(&self) -> Result<Vec<BoxedScalar>> {
        let mut values = Vec::with_capacity(self.size());
        for coord in iter_boxed_coords(self.shape()) {
            values.push(self.get(&coord)?);
        }
        Ok(values)
    }
}

fn boxed_scalar_matches_dtype(value: &BoxedScalar, dtype: BoxedDType) -> bool {
    matches!(
        (value, dtype),
        (BoxedScalar::Object(_), BoxedDType::Object)
            | (BoxedScalar::Datetime(_), BoxedDType::Datetime64)
            | (BoxedScalar::Timedelta(_), BoxedDType::Timedelta64)
    )
}

fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

fn row_major_flat_index(shape: &[usize], index: &[usize]) -> Result<usize> {
    if index.len() != shape.len() {
        return Err(NumpyError::ValueError(
            "index rank does not match array rank".into(),
        ));
    }
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (&coord, &dim) in index.iter().rev().zip(shape.iter().rev()) {
        if coord >= dim {
            return Err(NumpyError::ValueError("index out of bounds".into()));
        }
        flat = flat
            .checked_add(
                coord
                    .checked_mul(stride)
                    .ok_or_else(|| NumpyError::ValueError("index computation overflowed".into()))?,
            )
            .ok_or_else(|| NumpyError::ValueError("index computation overflowed".into()))?;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| NumpyError::ValueError("index computation overflowed".into()))?;
    }
    Ok(flat)
}

fn boxed_map_index_through_view(info: &[SliceInfoElem], index: &[usize]) -> Result<Vec<usize>> {
    let mut mapped = Vec::with_capacity(info.len());
    let mut view_axis = 0usize;
    for elem in info {
        match elem {
            SliceInfoElem::Index(idx) => mapped.push(*idx as usize),
            SliceInfoElem::Slice { start, step, .. } => {
                let coord = index.get(view_axis).ok_or_else(|| {
                    NumpyError::ValueError("view index rank does not match destination".into())
                })?;
                let mapped_idx = *start + (*coord as isize) * *step;
                if mapped_idx < 0 {
                    return Err(NumpyError::ValueError("negative mapped index".into()));
                }
                mapped.push(mapped_idx as usize);
                view_axis += 1;
            }
            SliceInfoElem::NewAxis => {}
        }
    }
    Ok(mapped)
}

fn boxed_slice_shape_and_strides(
    shape: &[usize],
    strides: &[isize],
    info: &[SliceInfoElem],
) -> Result<(Vec<usize>, Vec<isize>)> {
    let mut out_shape = Vec::with_capacity(shape.len());
    let mut out_strides = Vec::with_capacity(shape.len());
    let mut axis = 0usize;

    for elem in info {
        match *elem {
            SliceInfoElem::Index(_) => {
                if axis >= shape.len() {
                    return Err(NumpyError::ValueError(
                        "too many indices for boxed array".into(),
                    ));
                }
                axis += 1;
            }
            SliceInfoElem::Slice { start, end, step } => {
                if axis >= shape.len() {
                    return Err(NumpyError::ValueError(
                        "too many indices for boxed array".into(),
                    ));
                }
                if step == 0 {
                    return Err(NumpyError::ValueError("slice step cannot be zero".into()));
                }
                let dim = shape[axis] as isize;
                let start = normalize_slice_bound(start, dim);
                let end = normalize_slice_end(end.unwrap_or(dim), dim);
                let len = if step > 0 {
                    if end <= start {
                        0
                    } else {
                        ((end - start - 1) / step + 1) as usize
                    }
                } else {
                    0
                };
                out_shape.push(len);
                out_strides.push(strides[axis] * step);
                axis += 1;
            }
            SliceInfoElem::NewAxis => {
                out_shape.push(1);
                out_strides.push(0);
            }
        }
    }

    while axis < shape.len() {
        out_shape.push(shape[axis]);
        out_strides.push(strides[axis]);
        axis += 1;
    }

    Ok((out_shape, out_strides))
}

fn normalize_slice_bound(bound: isize, dim: isize) -> isize {
    if bound < 0 {
        (bound + dim).max(0).min(dim)
    } else {
        bound.min(dim)
    }
}

fn normalize_slice_end(bound: isize, dim: isize) -> isize {
    normalize_slice_bound(bound, dim)
}

fn iter_boxed_coords(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    let size: usize = shape.iter().product();
    (0..size).map(move |i| linear_to_coord(i, shape))
}

fn refresh_view_cache(view: &ViewStorage) -> Result<()> {
    let mut cache = view.cache.write().unwrap();
    cache.version = view.base.version();
    cache.data = view.base.data().slice_view(view.info.as_slice())?;
    Ok(())
}

fn assign_materialized_selection(
    base: &mut ArrayStorage,
    view_info: &[SliceInfoElem],
    target: &ArrayData,
    values: &ArrayData,
) -> Result<()> {
    let coords = iter_array_scalars(target)
        .map(|(coord, _)| coord)
        .collect::<Vec<_>>();
    assign_scalar_list(base, view_info, coords.as_slice(), target, values)
}

fn assign_scalar_list(
    base: &mut ArrayStorage,
    view_info: &[SliceInfoElem],
    coords: &[Vec<usize>],
    target: &ArrayData,
    values: &ArrayData,
) -> Result<()> {
    if target.shape() == values.shape() {
        let scalars = iter_array_scalars(values)
            .map(|(_, scalar)| scalar)
            .collect::<Vec<_>>();
        for (coord, scalar) in coords.iter().zip(scalars.into_iter()) {
            let mapped = map_index_through_view(view_info, coord.as_slice())?;
            base.assign_element(&mapped, scalar)?;
        }
        Ok(())
    } else if values.size() == 1 {
        let scalar = iter_array_scalars(values)
            .next()
            .ok_or_else(|| NumpyError::ValueError("cannot assign from empty values".into()))?
            .1;
        for coord in coords {
            let mapped = map_index_through_view(view_info, coord.as_slice())?;
            base.assign_element(&mapped, scalar.clone())?;
        }
        Ok(())
    } else {
        Err(NumpyError::ShapeMismatch(format!(
            "could not broadcast input array from shape {:?} into shape {:?}",
            values.shape(),
            target.shape()
        )))
    }
}

fn map_index_through_view(info: &[SliceInfoElem], index: &[usize]) -> Result<Vec<usize>> {
    let mut mapped = Vec::with_capacity(info.len());
    let mut view_axis = 0usize;
    for elem in info {
        match elem {
            SliceInfoElem::Index(idx) => mapped.push(*idx as usize),
            SliceInfoElem::Slice { start, step, .. } => {
                let coord = index.get(view_axis).ok_or_else(|| {
                    NumpyError::ValueError("view index rank does not match destination".into())
                })?;
                let mapped_idx = *start + (*coord as isize) * *step;
                if mapped_idx < 0 {
                    return Err(NumpyError::ValueError("negative mapped index".into()));
                }
                mapped.push(mapped_idx as usize);
                view_axis += 1;
            }
            SliceInfoElem::NewAxis => {}
        }
    }
    Ok(mapped)
}

fn set_array_element(
    data: &mut ArrayData,
    index: &[usize],
    value: crate::indexing::Scalar,
) -> Result<()> {
    let idx = IxDyn(index);
    match (data, value) {
        (ArrayData::Bool(a), crate::indexing::Scalar::Bool(v)) => set_elem(a, idx, v),
        (ArrayData::Int32(a), crate::indexing::Scalar::Int32(v)) => set_elem(a, idx, v),
        (ArrayData::Int64(a), crate::indexing::Scalar::Int64(v)) => set_elem(a, idx, v),
        (ArrayData::Float32(a), crate::indexing::Scalar::Float32(v)) => set_elem(a, idx, v),
        (ArrayData::Float64(a), crate::indexing::Scalar::Float64(v)) => set_elem(a, idx, v),
        (ArrayData::Complex64(a), crate::indexing::Scalar::Complex64(v)) => set_elem(a, idx, v),
        (ArrayData::Complex128(a), crate::indexing::Scalar::Complex128(v)) => set_elem(a, idx, v),
        (ArrayData::Str(a), crate::indexing::Scalar::Str(v)) => set_elem(a, idx, v),
        _ => Err(NumpyError::TypeError(
            "cannot assign scalar of incompatible dtype".into(),
        )),
    }
}

fn set_elem<T: Clone>(array: &mut ArrayD<T>, idx: IxDyn, value: T) -> Result<()> {
    let elem = array
        .get_mut(idx)
        .ok_or_else(|| NumpyError::ValueError("index out of bounds".into()))?;
    *elem = value;
    Ok(())
}

fn iter_array_scalars(
    data: &ArrayData,
) -> impl Iterator<Item = (Vec<usize>, crate::indexing::Scalar)> + '_ {
    let shape = data.shape().to_vec();
    (0..data.size()).map(move |i| {
        let coord = linear_to_coord(i, shape.as_slice());
        let scalar = array_data_to_scalar(
            &data
                .slice_view(
                    &coord
                        .iter()
                        .map(|&idx| SliceInfoElem::Index(idx as isize))
                        .collect::<Vec<_>>(),
                )
                .expect("valid scalar slice"),
        );
        (coord, scalar)
    })
}

fn linear_to_coord(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut coord = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        coord[d] = idx % shape[d];
        idx /= shape[d];
    }
    coord
}

#[cfg(test)]
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
        ArrayData::Str(values) => Some(
            values
                .iter()
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0)
                .max(1),
        ),
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
        DType::Complex64 => ArrayData::Complex64(match fill {
            FillValue::Zero => ArrayD::from_elem(sh, Complex::<f32>::new(0.0, 0.0)).into_shared(),
            FillValue::One => ArrayD::from_elem(sh, Complex::<f32>::new(1.0, 0.0)).into_shared(),
        }),
        DType::Complex128 => ArrayData::Complex128(match fill {
            FillValue::Zero => ArrayD::from_elem(sh, Complex::<f64>::new(0.0, 0.0)).into_shared(),
            FillValue::One => ArrayD::from_elem(sh, Complex::<f64>::new(1.0, 0.0)).into_shared(),
        }),
        DType::Str => ArrayData::Str(ArrayD::from_elem(sh, String::new()).into_shared()),
        DType::Int8
        | DType::Int16
        | DType::UInt8
        | DType::UInt16
        | DType::UInt32
        | DType::UInt64 => {
            panic!("storage fill only supports concrete storage dtypes, got {dtype:?}")
        }
        DType::Float16 => panic!("storage fill should receive Float32 backing for Float16"),
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
            ArrayD::from_elem(sh, Complex::<f32>::new(value as f32, 0.0)).into_shared(),
        ),
        DType::Complex128 => ArrayData::Complex128(
            ArrayD::from_elem(sh, Complex::<f64>::new(value, 0.0)).into_shared(),
        ),
        DType::Str => ArrayData::Str(ArrayD::from_elem(sh, value.to_string()).into_shared()),
        _ => unreachable!("storage fill_f64 should only see canonical storage dtypes"),
    };

    if dtype.is_narrow() {
        crate::casting::narrow_truncate(data, dtype)
    } else {
        data
    }
}

fn write_eye_diagonal(data: &mut ArrayData, rows: usize, cols: usize, k: isize) -> Result<()> {
    let (start_row, start_col) = if k >= 0 {
        (0usize, k as usize)
    } else {
        ((-k) as usize, 0usize)
    };

    if start_row >= rows || start_col >= cols {
        return Ok(());
    }

    let diag_len = (rows - start_row).min(cols - start_col);

    macro_rules! write_diag {
        ($arr:expr, $value:expr) => {{
            for i in 0..diag_len {
                $arr[[start_row + i, start_col + i]] = $value;
            }
        }};
    }

    match data {
        ArrayData::Bool(arr) => write_diag!(arr, true),
        ArrayData::Int32(arr) => write_diag!(arr, 1_i32),
        ArrayData::Int64(arr) => write_diag!(arr, 1_i64),
        ArrayData::Float32(arr) => write_diag!(arr, 1.0_f32),
        ArrayData::Float64(arr) => write_diag!(arr, 1.0_f64),
        ArrayData::Complex64(arr) => write_diag!(arr, Complex::<f32>::new(1.0, 0.0)),
        ArrayData::Complex128(arr) => write_diag!(arr, Complex::<f64>::new(1.0, 0.0)),
        ArrayData::Str(_) => {
            return Err(NumpyError::TypeError(
                "eye() not supported for string dtype".into(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::SliceInfoElem;

    use crate::array::{
        BoxedArray, BoxedDType, BoxedObjectScalar, BoxedScalar, BoxedTemporalScalar,
    };

    use super::BoxedStorage;

    #[test]
    fn boxed_view_write_updates_base() {
        let base = BoxedStorage::from_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Text("a".into())),
                BoxedScalar::Object(BoxedObjectScalar::Text("b".into())),
                BoxedScalar::Object(BoxedObjectScalar::Text("c".into())),
            ],
            &[3],
            BoxedDType::Object,
        )
        .unwrap();
        let mut view = base
            .slice_view(&[SliceInfoElem::Slice {
                start: 1,
                end: Some(3),
                step: 1,
            }])
            .unwrap();
        view.set(
            &[0],
            BoxedScalar::Object(BoxedObjectScalar::Text("z".into())),
        )
        .unwrap();
        assert_eq!(
            base.get(&[1]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Text("z".into()))
        );
    }

    #[test]
    fn boxed_temporal_scalar_round_trip_preserves_nat() {
        let arr = BoxedArray::from_boxed_scalars(
            vec![BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: 0,
                unit: "generic".into(),
                is_nat: true,
            })],
            &[1],
            BoxedDType::Timedelta64,
        )
        .unwrap();
        assert!(matches!(
            arr.get_boxed(&[0]).unwrap(),
            BoxedScalar::Timedelta(value) if value.is_nat
        ));
    }
}
