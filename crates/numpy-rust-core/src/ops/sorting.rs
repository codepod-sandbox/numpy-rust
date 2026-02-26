use std::cmp::Ordering;

use ndarray::{ArrayD, Axis, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

fn f64_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

impl NdArray {
    /// Return a new sorted array.
    /// axis=None: flatten then sort. axis=Some(ax): sort along that axis.
    pub fn sort(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "sort not supported for string arrays".into(),
            ));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "sort not supported for complex arrays".into(),
            ));
        }
        let f = self.astype(DType::Float64);
        match axis {
            None => {
                let ArrayData::Float64(arr) = &f.data else {
                    unreachable!()
                };
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                flat.sort_by(f64_cmp);
                Ok(NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat)
                        .expect("flat vec matches shape"),
                )))
            }
            Some(ax) => {
                if ax >= f.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: f.ndim(),
                    });
                }
                let ArrayData::Float64(arr) = &f.data else {
                    unreachable!()
                };
                let mut out = arr.clone();
                for mut lane in out.lanes_mut(Axis(ax)) {
                    let mut v: Vec<f64> = lane.iter().copied().collect();
                    v.sort_by(f64_cmp);
                    for (dest, src) in lane.iter_mut().zip(v.iter()) {
                        *dest = *src;
                    }
                }
                Ok(NdArray::from_data(ArrayData::Float64(out)))
            }
        }
    }

    /// Return the indices that would sort the array.
    /// axis=None: flatten then argsort. axis=Some(ax): argsort along that axis.
    pub fn argsort(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "argsort not supported for string arrays".into(),
            ));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "argsort not supported for complex arrays".into(),
            ));
        }
        let f = self.astype(DType::Float64);
        match axis {
            None => {
                let ArrayData::Float64(arr) = &f.data else {
                    unreachable!()
                };
                let flat: Vec<f64> = arr.iter().copied().collect();
                let mut indices: Vec<i64> = (0..flat.len() as i64).collect();
                indices.sort_by(|&a, &b| f64_cmp(&flat[a as usize], &flat[b as usize]));
                Ok(NdArray::from_data(ArrayData::Int64(
                    ArrayD::from_shape_vec(IxDyn(&[indices.len()]), indices)
                        .expect("flat vec matches shape"),
                )))
            }
            Some(ax) => {
                if ax >= f.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: f.ndim(),
                    });
                }
                let ArrayData::Float64(arr) = &f.data else {
                    unreachable!()
                };
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
                Ok(NdArray::from_data(ArrayData::Int64(result)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
}
