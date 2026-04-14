use crate::array::{BoxedScalar, BoxedTemporalScalar};
use crate::array_data::ArrayD;
use std::cmp::Ordering;

use ndarray::{Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::ops::comparison::{compare_boxed_scalars, iter_boxed_coords};
use crate::NdArray;

fn f64_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

fn boxed_cmp(lhs: &BoxedScalar, rhs: &BoxedScalar) -> Result<Ordering> {
    if let Some(ord) = compare_temporal_nat_order(lhs, rhs) {
        return Ok(ord);
    }
    Ok(compare_boxed_scalars(lhs, rhs)?.unwrap_or(Ordering::Equal))
}

fn compare_temporal_nat_order(lhs: &BoxedScalar, rhs: &BoxedScalar) -> Option<Ordering> {
    fn nat_order(lhs: &BoxedTemporalScalar, rhs: &BoxedTemporalScalar) -> Option<Ordering> {
        match (lhs.is_nat, rhs.is_nat) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Greater),
            (false, true) => Some(Ordering::Less),
            (false, false) => None,
        }
    }

    match (lhs, rhs) {
        (BoxedScalar::Datetime(lhs), BoxedScalar::Datetime(rhs))
        | (BoxedScalar::Timedelta(lhs), BoxedScalar::Timedelta(rhs)) => nat_order(lhs, rhs),
        _ => None,
    }
}

fn try_argsort_boxed_values(values: &[BoxedScalar]) -> Result<Vec<usize>> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    for i in 1..indices.len() {
        let mut j = i;
        while j > 0 {
            let ord = boxed_cmp(&values[indices[j - 1]], &values[indices[j]])?;
            if ord == Ordering::Greater {
                indices.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
    Ok(indices)
}

fn row_major_index(shape: &[usize], coord: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (&idx, &dim) in coord.iter().rev().zip(shape.iter().rev()) {
        flat += idx * stride;
        stride *= dim;
    }
    flat
}

fn lane_coord(axis: usize, outer_coord: &[usize], index: usize, ndim: usize) -> Vec<usize> {
    let mut coord = Vec::with_capacity(ndim);
    let mut outer_axis = 0usize;
    for current_axis in 0..ndim {
        if current_axis == axis {
            coord.push(index);
        } else {
            coord.push(outer_coord[outer_axis]);
            outer_axis += 1;
        }
    }
    coord
}

fn validate_axis(array: &NdArray, axis: Option<usize>, op_name: &str) -> Result<Option<usize>> {
    if let Some(ax) = axis {
        if ax >= array.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis: ax,
                ndim: array.ndim(),
            });
        }
        return Ok(Some(ax));
    }
    if array.dtype().is_string() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for string arrays"
        )));
    }
    Ok(None)
}

fn execute_boxed_sort(array: &NdArray, axis: Option<usize>) -> Result<NdArray> {
    let axis = validate_axis(array, axis, "sort")?;
    match axis {
        None => {
            let mut values = Vec::with_capacity(array.size());
            for coord in iter_boxed_coords(array.shape()) {
                values.push(array.get_boxed(&coord)?);
            }
            let order = try_argsort_boxed_values(&values)?;
            let sorted = order
                .into_iter()
                .map(|idx| values[idx].clone())
                .collect::<Vec<_>>();
            NdArray::from_boxed_scalars(sorted, &[values.len()], array.dtype())
        }
        Some(ax) => {
            let shape = array.shape().to_vec();
            let axis_len = shape[ax];
            let outer_shape = shape
                .iter()
                .enumerate()
                .filter_map(|(idx, &dim)| (idx != ax).then_some(dim))
                .collect::<Vec<_>>();
            let mut out = Vec::with_capacity(array.size());
            for coord in iter_boxed_coords(&shape) {
                out.push(array.get_boxed(&coord)?);
            }
            for outer_coord in iter_boxed_coords(&outer_shape) {
                let lane_coords = (0..axis_len)
                    .map(|idx| lane_coord(ax, &outer_coord, idx, shape.len()))
                    .collect::<Vec<_>>();
                let mut lane = Vec::with_capacity(axis_len);
                for coord in &lane_coords {
                    lane.push(array.get_boxed(coord)?);
                }
                let order = try_argsort_boxed_values(&lane)?;
                for (dest_idx, source_idx) in order.into_iter().enumerate() {
                    let flat = row_major_index(&shape, &lane_coords[dest_idx]);
                    out[flat] = lane[source_idx].clone();
                }
            }
            NdArray::from_boxed_scalars(out, &shape, array.dtype())
        }
    }
}

fn execute_boxed_argsort(array: &NdArray, axis: Option<usize>) -> Result<NdArray> {
    let axis = validate_axis(array, axis, "argsort")?;
    match axis {
        None => {
            let mut values = Vec::with_capacity(array.size());
            for coord in iter_boxed_coords(array.shape()) {
                values.push(array.get_boxed(&coord)?);
            }
            let order = try_argsort_boxed_values(&values)?
                .into_iter()
                .map(|idx| idx as i64)
                .collect::<Vec<_>>();
            Ok(NdArray::from_data(ArrayData::Int64(
                ArrayD::from_shape_vec(IxDyn(&[order.len()]), order)
                    .expect("flat vec matches shape"),
            )))
        }
        Some(ax) => {
            let shape = array.shape().to_vec();
            let axis_len = shape[ax];
            let outer_shape = shape
                .iter()
                .enumerate()
                .filter_map(|(idx, &dim)| (idx != ax).then_some(dim))
                .collect::<Vec<_>>();
            let mut out = ArrayD::<i64>::zeros(IxDyn(&shape));
            for outer_coord in iter_boxed_coords(&outer_shape) {
                let lane_coords = (0..axis_len)
                    .map(|idx| lane_coord(ax, &outer_coord, idx, shape.len()))
                    .collect::<Vec<_>>();
                let mut lane = Vec::with_capacity(axis_len);
                for coord in &lane_coords {
                    lane.push(array.get_boxed(coord)?);
                }
                let order = try_argsort_boxed_values(&lane)?;
                for (dest_idx, source_idx) in order.into_iter().enumerate() {
                    out[IxDyn(lane_coords[dest_idx].as_slice())] = source_idx as i64;
                }
            }
            Ok(NdArray::from_data(ArrayData::Int64(out)))
        }
    }
}

fn prepare_sort_input(array: &NdArray, axis: Option<usize>, op_name: &str) -> Result<ArrayD<f64>> {
    if array.dtype().is_string() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for string arrays"
        )));
    }
    if array.dtype().is_complex() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for complex arrays"
        )));
    }

    let cast = array.astype(DType::Float64);
    if let Some(ax) = axis {
        if ax >= cast.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis: ax,
                ndim: cast.ndim(),
            });
        }
    }

    let ArrayData::Float64(arr) = cast.data() else {
        unreachable!()
    };

    Ok(arr.clone())
}

fn execute_sort_like<TFlat, TAxis>(
    array: &NdArray,
    axis: Option<usize>,
    op_name: &str,
    reduce_flat: TFlat,
    reduce_axis: TAxis,
) -> Result<NdArray>
where
    TFlat: FnOnce(&ArrayD<f64>) -> NdArray,
    TAxis: FnOnce(ArrayD<f64>, usize) -> NdArray,
{
    let arr = prepare_sort_input(array, axis, op_name)?;

    Ok(match axis {
        None => reduce_flat(&arr),
        Some(ax) => reduce_axis(arr, ax),
    })
}

impl NdArray {
    /// Return a new sorted array.
    /// axis=None: flatten then sort. axis=Some(ax): sort along that axis.
    pub fn sort(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_boxed() {
            return execute_boxed_sort(self, axis);
        }
        execute_sort_like(
            self,
            axis,
            "sort",
            |arr| {
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                flat.sort_by(f64_cmp);
                NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat)
                        .expect("flat vec matches shape"),
                ))
            },
            |arr, ax| {
                let mut out = arr.clone();
                for mut lane in out.lanes_mut(Axis(ax)) {
                    let mut v: Vec<f64> = lane.iter().copied().collect();
                    v.sort_by(f64_cmp);
                    for (dest, src) in lane.iter_mut().zip(v.iter()) {
                        *dest = *src;
                    }
                }
                NdArray::from_data(ArrayData::Float64(out))
            },
        )
    }

    /// Return the indices that would sort the array.
    /// axis=None: flatten then argsort. axis=Some(ax): argsort along that axis.
    pub fn argsort(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_boxed() {
            return execute_boxed_argsort(self, axis);
        }
        execute_sort_like(
            self,
            axis,
            "argsort",
            |arr| {
                let flat: Vec<f64> = arr.iter().copied().collect();
                let mut indices: Vec<i64> = (0..flat.len() as i64).collect();
                indices.sort_by(|&a, &b| f64_cmp(&flat[a as usize], &flat[b as usize]));
                NdArray::from_data(ArrayData::Int64(
                    ArrayD::from_shape_vec(IxDyn(&[indices.len()]), indices)
                        .expect("flat vec matches shape"),
                ))
            },
            |arr, ax| {
                let mut result = ArrayD::<i64>::zeros(arr.raw_dim());
                for (lane_in, mut lane_out) in arr
                    .lanes(Axis(ax))
                    .into_iter()
                    .zip(result.lanes_mut(Axis(ax)))
                {
                    let v: Vec<f64> = lane_in.iter().copied().collect();
                    let mut indices: Vec<i64> = (0..v.len() as i64).collect();
                    indices.sort_by(|&a, &b| f64_cmp(&v[a as usize], &v[b as usize]));
                    for (dest, &src) in lane_out.iter_mut().zip(indices.iter()) {
                        *dest = src;
                    }
                }
                NdArray::from_data(ArrayData::Int64(result))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{BoxedObjectScalar, BoxedScalar, BoxedTemporalScalar};
    use crate::NdArray;

    #[test]
    fn test_sort_1d() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0]);
        let s = a.sort(None).unwrap();
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_argsort_1d() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0]);
        let idx = a.argsort(None).unwrap();
        assert_eq!(idx.shape(), &[3]);
    }

    #[test]
    fn test_sort_axis() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0, 6.0, 4.0, 5.0])
            .reshape(&[2, 3])
            .unwrap();
        let s = a.sort(Some(1)).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    fn test_sort_boxed_object() {
        let a = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Int(3)),
                BoxedScalar::Object(BoxedObjectScalar::Int(1)),
                BoxedScalar::Object(BoxedObjectScalar::Int(2)),
            ],
            &[3],
            crate::DType::Object,
        )
        .unwrap();
        let s = a.sort(None).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(
            vec![
                s.get_boxed(&[0]).unwrap(),
                s.get_boxed(&[1]).unwrap(),
                s.get_boxed(&[2]).unwrap(),
            ],
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Int(1)),
                BoxedScalar::Object(BoxedObjectScalar::Int(2)),
                BoxedScalar::Object(BoxedObjectScalar::Int(3)),
            ]
        );
    }

    #[test]
    fn test_argsort_boxed_object() {
        let a = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Int(3)),
                BoxedScalar::Object(BoxedObjectScalar::Int(1)),
                BoxedScalar::Object(BoxedObjectScalar::Int(2)),
            ],
            &[3],
            crate::DType::Object,
        )
        .unwrap();
        let idx = a.argsort(None).unwrap();
        assert_eq!(idx.shape(), &[3]);
    }

    #[test]
    fn test_sort_boxed_temporal_nat_last() {
        let nat = BoxedScalar::Datetime(BoxedTemporalScalar {
            value: 0,
            unit: "ns".to_string(),
            is_nat: true,
        });
        let one = BoxedScalar::Datetime(BoxedTemporalScalar {
            value: 1,
            unit: "ns".to_string(),
            is_nat: false,
        });
        let two = BoxedScalar::Datetime(BoxedTemporalScalar {
            value: 2,
            unit: "ns".to_string(),
            is_nat: false,
        });
        let a = NdArray::from_boxed_scalars(
            vec![nat.clone(), two.clone(), one.clone()],
            &[3],
            crate::DType::Datetime64,
        )
        .unwrap();
        let s = a.sort(None).unwrap();
        assert_eq!(s.get_boxed(&[0]).unwrap(), one);
        assert_eq!(s.get_boxed(&[1]).unwrap(), two);
        assert_eq!(s.get_boxed(&[2]).unwrap(), nat);
    }
}
