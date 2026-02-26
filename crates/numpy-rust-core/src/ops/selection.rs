use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

impl NdArray {
    /// Binary search in a sorted 1-D array. Returns Int64 array of insertion indices.
    /// side="left": leftmost insertion point. side="right": rightmost.
    pub fn searchsorted(&self, values: &NdArray, side: &str) -> Result<NdArray> {
        if self.ndim() != 1 {
            return Err(NumpyError::ValueError(
                "searchsorted requires a 1-D sorted array".into(),
            ));
        }
        let sorted = self.astype(DType::Float64);
        let vals = values.astype(DType::Float64);
        let ArrayData::Float64(sorted_arr) = &sorted.data else {
            unreachable!()
        };
        let ArrayData::Float64(vals_arr) = &vals.data else {
            unreachable!()
        };
        let sorted_slice: Vec<f64> = sorted_arr.iter().copied().collect();
        let left = match side {
            "left" => true,
            "right" => false,
            _ => {
                return Err(NumpyError::ValueError(format!(
                    "searchsorted: invalid side '{side}', must be 'left' or 'right'"
                )))
            }
        };

        let mut indices = Vec::with_capacity(vals_arr.len());
        for &v in vals_arr.iter() {
            let idx = if left {
                sorted_slice.partition_point(|&x| x < v)
            } else {
                sorted_slice.partition_point(|&x| x <= v)
            };
            indices.push(idx as i64);
        }

        Ok(NdArray::from_data(ArrayData::Int64(
            ArrayD::from_shape_vec(vals_arr.raw_dim(), indices)
                .expect("output shape matches values shape"),
        )))
    }

    /// Select slices along `axis` where `condition` is true.
    pub fn compress(&self, condition: &NdArray, axis: Option<usize>) -> Result<NdArray> {
        let cond = condition.astype(DType::Bool);
        let ArrayData::Bool(mask) = &cond.data else {
            unreachable!()
        };

        match axis {
            None => {
                // Flatten self, apply mask
                let flat = self.flatten();
                flat.mask_select(&cond)
            }
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumpyError::InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                    });
                }
                let indices: Vec<usize> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &b)| b)
                    .map(|(i, _)| i)
                    .collect();
                self.index_select(ax, &indices)
            }
        }
    }
}

/// Select from `choices` arrays based on integer index array `a`.
pub fn choose(a: &NdArray, choices: &[&NdArray]) -> Result<NdArray> {
    if choices.is_empty() {
        return Err(NumpyError::ValueError(
            "choose requires at least one choice array".into(),
        ));
    }
    let idx = a.astype(DType::Int64);
    let flat_idx = idx.flatten();
    let ArrayData::Int64(idx_arr) = &flat_idx.data else {
        unreachable!()
    };

    let n_choices = choices.len();

    // Flatten all choices to Vec<f64> for O(1) indexing
    let flat_vecs: Vec<Vec<f64>> = choices
        .iter()
        .map(|c| {
            let f = c.astype(DType::Float64).flatten();
            let ArrayData::Float64(arr) = &f.data else {
                unreachable!()
            };
            arr.iter().copied().collect()
        })
        .collect();

    let mut result = Vec::with_capacity(idx_arr.len());
    for (pos, &i) in idx_arr.iter().enumerate() {
        if i < 0 {
            return Err(NumpyError::ValueError(format!(
                "invalid entry {i} in choice array (negative indices not supported)"
            )));
        }
        let choice_idx = i as usize;
        if choice_idx >= n_choices {
            return Err(NumpyError::ValueError(format!(
                "invalid entry {i} in choice array (out of range for {n_choices} choices)"
            )));
        }
        let vec = &flat_vecs[choice_idx];
        if pos >= vec.len() {
            return Err(NumpyError::ValueError(format!(
                "shape mismatch: index array has {} elements but choice {} has {}",
                idx_arr.len(),
                choice_idx,
                vec.len()
            )));
        }
        result.push(vec[pos]);
    }

    let out_shape = a.shape().to_vec();
    Ok(NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&out_shape), result).expect("shape matches"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_searchsorted_left() {
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 5.0, 7.0]);
        let v = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let idx = a.searchsorted(&v, "left").unwrap();
        assert_eq!(idx.shape(), &[3]);
        let ArrayData::Int64(arr) = idx.data() else {
            panic!("expected Int64");
        };
        assert_eq!(arr[[0]], 1); // 2.0 goes at index 1
        assert_eq!(arr[[1]], 2); // 4.0 goes at index 2
        assert_eq!(arr[[2]], 3); // 6.0 goes at index 3
    }

    #[test]
    fn test_searchsorted_right() {
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 3.0, 5.0]);
        let v = NdArray::from_vec(vec![3.0_f64]);
        let idx = a.searchsorted(&v, "right").unwrap();
        let ArrayData::Int64(arr) = idx.data() else {
            panic!("expected Int64");
        };
        assert_eq!(arr[[0]], 3);
    }

    #[test]
    fn test_compress() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let cond = NdArray::from_vec(vec![true, false, true, false]);
        let result = a.compress(&cond, None).unwrap();
        assert_eq!(result.shape(), &[2]);
    }

    #[test]
    fn test_choose_basic() {
        let a = NdArray::from_vec(vec![0_i64, 1, 0, 1]);
        let c0 = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0]);
        let c1 = NdArray::from_vec(vec![50.0_f64, 60.0, 70.0, 80.0]);
        let result = choose(&a, &[&c0, &c1]).unwrap();
        assert_eq!(result.shape(), &[4]);
        let ArrayData::Float64(arr) = result.data() else {
            panic!("expected Float64");
        };
        assert_eq!(arr[[0]], 10.0);
        assert_eq!(arr[[1]], 60.0);
        assert_eq!(arr[[2]], 30.0);
        assert_eq!(arr[[3]], 80.0);
    }
}
