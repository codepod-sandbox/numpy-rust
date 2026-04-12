use crate::array_data::ArrayD;
use ndarray::{Axis, IxDyn, SliceInfoElem};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

fn prepare_float64_axis_op(
    array: &NdArray,
    axis: Option<usize>,
    op_name: &str,
) -> Result<ArrayD<f64>> {
    if array.dtype().is_string() {
        return Err(NumpyError::TypeError(format!(
            "{op_name} not supported for string arrays"
        )));
    }

    let cast = array.astype(DType::Float64);
    let ArrayData::Float64(arr) = cast.data() else {
        unreachable!()
    };

    if let Some(ax) = axis {
        if ax >= cast.ndim() {
            return Err(NumpyError::InvalidAxis {
                axis: ax,
                ndim: cast.ndim(),
            });
        }
    }

    Ok(arr.to_owned().into_shared())
}

fn execute_float64_cumulative<F>(
    array: &NdArray,
    axis: Option<usize>,
    op_name: &str,
    init: f64,
    mut step: F,
) -> Result<NdArray>
where
    F: FnMut(&mut f64, f64) -> f64,
{
    let arr = prepare_float64_axis_op(array, axis, op_name)?;

    let out = match axis {
        None => {
            let flat: Vec<f64> = arr.iter().copied().collect();
            let mut cumulated = Vec::with_capacity(flat.len());
            let mut acc = init;
            for v in flat {
                let next = step(&mut acc, v);
                cumulated.push(next);
            }
            ArrayD::from_shape_vec(IxDyn(&[cumulated.len()]), cumulated)
                .expect("flat vec matches shape")
                .into_shared()
        }
        Some(ax) => {
            let mut out = arr;
            for mut lane in out.lanes_mut(Axis(ax)) {
                let mut acc = init;
                for elem in lane.iter_mut() {
                    *elem = step(&mut acc, *elem);
                }
            }
            out.into_shared()
        }
    };

    Ok(NdArray::from_data(ArrayData::Float64(out)))
}

fn prepare_float64_diff_input(
    array: &NdArray,
    axis: Option<usize>,
) -> Result<(ndarray::ArrayD<f64>, usize)> {
    let arr = prepare_float64_axis_op(array, axis, "diff")?;

    match axis {
        None => {
            let flat: Vec<f64> = arr.iter().copied().collect();
            let flat_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat)
                .expect("flat vec matches");
            Ok((flat_arr, 0))
        }
        Some(ax) => Ok((arr.to_owned(), ax)),
    }
}

impl NdArray {
    /// Cumulative sum along an axis.
    /// If `axis` is `None`, the array is flattened first and the cumulative sum
    /// is computed over the flat array, returning a 1-D result.
    pub fn cumsum(&self, axis: Option<usize>) -> Result<NdArray> {
        execute_float64_cumulative(self, axis, "cumsum", 0.0, |acc, value| {
            *acc += value;
            *acc
        })
    }

    /// Cumulative product along an axis.
    /// If `axis` is `None`, the array is flattened first and the cumulative product
    /// is computed over the flat array, returning a 1-D result.
    pub fn cumprod(&self, axis: Option<usize>) -> Result<NdArray> {
        execute_float64_cumulative(self, axis, "cumprod", 1.0, |acc, value| {
            *acc *= value;
            *acc
        })
    }

    /// Cumulative sum, treating NaN as zero.
    pub fn nancumsum(&self, axis: Option<usize>) -> Result<NdArray> {
        execute_float64_cumulative(self, axis, "nancumsum", 0.0, |acc, value| {
            if !value.is_nan() {
                *acc += value;
            }
            *acc
        })
    }

    /// Cumulative product, treating NaN as one.
    pub fn nancumprod(&self, axis: Option<usize>) -> Result<NdArray> {
        execute_float64_cumulative(self, axis, "nancumprod", 1.0, |acc, value| {
            if !value.is_nan() {
                *acc *= value;
            }
            *acc
        })
    }

    /// N-th discrete difference along an axis.
    /// If `axis` is `None`, the array is flattened first.
    /// The result shape has `shape[axis] - n` along the diff axis.
    pub fn diff(&self, n: usize, axis: Option<usize>) -> Result<NdArray> {
        let (work, ax) = prepare_float64_diff_input(self, axis)?;

        if n == 0 {
            return Ok(NdArray::from_data(ArrayData::Float64(work.into_shared())));
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

        Ok(NdArray::from_data(ArrayData::Float64(
            current.into_shared(),
        )))
    }
}

/// Compute a single discrete difference along axis `ax`.
/// result[..., i, ...] = arr[..., i+1, ...] - arr[..., i, ...]
fn diff_once(arr: &ndarray::ArrayD<f64>, ax: usize) -> ndarray::ArrayD<f64> {
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
    fn test_nancumsum_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0, 4.0]);
        let c = a.nancumsum(None).unwrap();
        assert_eq!(c.shape(), &[4]);
        assert_eq!(c.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(c.get(&[1]).unwrap(), Scalar::Float64(1.0)); // NaN skipped
        assert_eq!(c.get(&[2]).unwrap(), Scalar::Float64(4.0));
        assert_eq!(c.get(&[3]).unwrap(), Scalar::Float64(8.0));
    }

    #[test]
    fn test_nancumprod_1d() {
        let a = NdArray::from_vec(vec![1.0_f64, f64::NAN, 3.0, 4.0]);
        let c = a.nancumprod(None).unwrap();
        assert_eq!(c.shape(), &[4]);
        assert_eq!(c.get(&[0]).unwrap(), Scalar::Float64(1.0));
        assert_eq!(c.get(&[1]).unwrap(), Scalar::Float64(1.0)); // NaN skipped
        assert_eq!(c.get(&[2]).unwrap(), Scalar::Float64(3.0));
        assert_eq!(c.get(&[3]).unwrap(), Scalar::Float64(12.0));
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
