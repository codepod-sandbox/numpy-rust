use ndarray::{ArrayD, Axis, IxDyn, SliceInfoElem};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Cumulative sum along an axis.
    /// If `axis` is `None`, the array is flattened first and the cumulative sum
    /// is computed over the flat array, returning a 1-D result.
    pub fn cumsum(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "cumsum not supported for string arrays".into(),
            ));
        }
        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else {
            unreachable!()
        };
        match axis {
            None => {
                let flat: Vec<f64> = arr.iter().copied().collect();
                let mut cumulated = Vec::with_capacity(flat.len());
                let mut acc = 0.0_f64;
                for v in &flat {
                    acc += v;
                    cumulated.push(acc);
                }
                Ok(NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[cumulated.len()]), cumulated)
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
                let mut out = arr.clone();
                for mut lane in out.lanes_mut(Axis(ax)) {
                    let mut acc = 0.0_f64;
                    for elem in lane.iter_mut() {
                        acc += *elem;
                        *elem = acc;
                    }
                }
                Ok(NdArray::from_data(ArrayData::Float64(out)))
            }
        }
    }

    /// Cumulative product along an axis.
    /// If `axis` is `None`, the array is flattened first and the cumulative product
    /// is computed over the flat array, returning a 1-D result.
    pub fn cumprod(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "cumprod not supported for string arrays".into(),
            ));
        }
        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else {
            unreachable!()
        };
        match axis {
            None => {
                let flat: Vec<f64> = arr.iter().copied().collect();
                let mut cumulated = Vec::with_capacity(flat.len());
                let mut acc = 1.0_f64;
                for v in &flat {
                    acc *= v;
                    cumulated.push(acc);
                }
                Ok(NdArray::from_data(ArrayData::Float64(
                    ArrayD::from_shape_vec(IxDyn(&[cumulated.len()]), cumulated)
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
                let mut out = arr.clone();
                for mut lane in out.lanes_mut(Axis(ax)) {
                    let mut acc = 1.0_f64;
                    for elem in lane.iter_mut() {
                        acc *= *elem;
                        *elem = acc;
                    }
                }
                Ok(NdArray::from_data(ArrayData::Float64(out)))
            }
        }
    }

    /// N-th discrete difference along an axis.
    /// If `axis` is `None`, the array is flattened first.
    /// The result shape has `shape[axis] - n` along the diff axis.
    pub fn diff(&self, n: usize, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "diff not supported for string arrays".into(),
            ));
        }
        let f = self.astype(DType::Float64);
        let ArrayData::Float64(arr) = &f.data else {
            unreachable!()
        };

        let (work, ax) = match axis {
            None => {
                // Flatten to 1-D
                let flat: Vec<f64> = arr.iter().copied().collect();
                let flat_arr =
                    ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).expect("flat vec matches");
                (flat_arr, 0)
            }
            Some(ax) => {
                if ax >= f.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: f.ndim(),
                    });
                }
                (arr.clone(), ax)
            }
        };

        if n == 0 {
            return Ok(NdArray::from_data(ArrayData::Float64(work)));
        }

        let mut current = work;
        for _ in 0..n {
            if current.shape()[ax] < 2 {
                return Err(NumpyError::ValueError(format!(
                    "diff requires at least 2 elements along axis {ax}, got {}",
                    current.shape()[ax]
                )));
            }
            current = diff_once(&current, ax);
        }

        Ok(NdArray::from_data(ArrayData::Float64(current)))
    }
}

/// Compute a single discrete difference along axis `ax`.
/// result[..., i, ...] = arr[..., i+1, ...] - arr[..., i, ...]
fn diff_once(arr: &ArrayD<f64>, ax: usize) -> ArrayD<f64> {
    let ndim = arr.ndim();
    let len = arr.shape()[ax];

    // Build slice info for hi = arr[..., 1:, ...] along axis `ax`
    let mut hi_slices: Vec<SliceInfoElem> = Vec::with_capacity(ndim);
    let mut lo_slices: Vec<SliceInfoElem> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        if i == ax {
            // hi: [1:]
            hi_slices.push(SliceInfoElem::Slice {
                start: 1,
                end: Some(len as isize),
                step: 1,
            });
            // lo: [:-1]
            lo_slices.push(SliceInfoElem::Slice {
                start: 0,
                end: Some((len - 1) as isize),
                step: 1,
            });
        } else {
            hi_slices.push(SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            });
            lo_slices.push(SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            });
        }
    }

    let hi = arr.slice(hi_slices.as_slice());
    let lo = arr.slice(lo_slices.as_slice());
    (&hi - &lo).into_owned()
}

#[cfg(test)]
mod tests {
    use crate::indexing::Scalar;
    use crate::NdArray;

    #[test]
    fn test_cumsum_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let c = a.cumsum(None).unwrap();
        assert_eq!(c.shape(), &[4]);
        assert_eq!(c.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(c.get(&[1]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(c.get(&[2]).unwrap(), Scalar::Float64(6.0));
        assert_eq!(c.get(&[3]).unwrap(), Scalar::Float64(10.0));
    }

    #[test]
    fn test_cumsum_2d_axis0() {
        // [[1,2],[3,4]] axis=0 → [[1,2],[4,6]]
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = a.cumsum(Some(0)).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(c.get(&[0, 1]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(c.get(&[1, 0]).unwrap(), Scalar::Float64(4.0));
        assert_eq!(c.get(&[1, 1]).unwrap(), Scalar::Float64(6.0));
    }

    #[test]
    fn test_cumsum_2d_axis1() {
        // [[1,2],[3,4]] axis=1 → [[1,3],[3,7]]
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = a.cumsum(Some(1)).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(c.get(&[0, 1]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(c.get(&[1, 0]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(c.get(&[1, 1]).unwrap(), Scalar::Float64(7.0));
    }

    #[test]
    fn test_cumprod_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let c = a.cumprod(None).unwrap();
        assert_eq!(c.shape(), &[4]);
        assert_eq!(c.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(c.get(&[1]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(c.get(&[2]).unwrap(), Scalar::Float64(6.0));
        assert_eq!(c.get(&[3]).unwrap(), Scalar::Float64(24.0));
    }

    #[test]
    fn test_cumprod_2d_axis0() {
        // [[1,2],[3,4]] axis=0 → [[1,2],[3,8]]
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = a.cumprod(Some(0)).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(c.get(&[0, 1]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(c.get(&[1, 0]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(c.get(&[1, 1]).unwrap(), Scalar::Float64(8.0));
    }

    #[test]
    fn test_diff_1d() {
        // [1,3,6,10] → [2,3,4]
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 6.0, 10.0]);
        let d = a.diff(1, None).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.get(&[0]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(d.get(&[1]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(d.get(&[2]).unwrap(), Scalar::Float64(4.0));
    }

    #[test]
    fn test_diff_n2() {
        // [1,3,6,10] with n=2 → [1,1]
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 6.0, 10.0]);
        let d = a.diff(2, None).unwrap();
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(d.get(&[1]).unwrap(), Scalar::Float64(1.0));
    }

    #[test]
    fn test_diff_2d_axis1() {
        // [[1,2,4,7],[3,5,9,15]] axis=1 → shape [2,3]
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 4.0, 7.0, 3.0, 5.0, 9.0, 15.0])
            .reshape(&[2, 4])
            .unwrap();
        let d = a.diff(1, Some(1)).unwrap();
        assert_eq!(d.shape(), &[2, 3]);
        // Row 0: [2-1, 4-2, 7-4] = [1, 2, 3]
        assert_eq!(d.get(&[0, 0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(d.get(&[0, 1]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(d.get(&[0, 2]).unwrap(), Scalar::Float64(3.0));
        // Row 1: [5-3, 9-5, 15-9] = [2, 4, 6]
        assert_eq!(d.get(&[1, 0]).unwrap(), Scalar::Float64(2.0));
        assert_eq!(d.get(&[1, 1]).unwrap(), Scalar::Float64(4.0));
        assert_eq!(d.get(&[1, 2]).unwrap(), Scalar::Float64(6.0));
    }

    #[test]
    fn test_diff_2d_axis0() {
        // [[1,2,3],[4,5,6]] axis=0 → shape [1,3]
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let d = a.diff(1, Some(0)).unwrap();
        assert_eq!(d.shape(), &[1, 3]);
        // [4-1, 5-2, 6-3] = [3, 3, 3]
        assert_eq!(d.get(&[0, 0]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(d.get(&[0, 1]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(d.get(&[0, 2]).unwrap(), Scalar::Float64(3.0));
    }
}
