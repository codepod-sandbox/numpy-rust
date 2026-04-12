use crate::array_data::ArrayD;
use std::collections::HashSet;

use ndarray::IxDyn;

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;
use num_complex::Complex;

fn prepare_float64_array(array: &NdArray) -> ArrayD<f64> {
    let cast = array.astype(DType::Float64);
    let ArrayData::Float64(arr) = cast.data() else {
        unreachable!()
    };
    arr.clone()
}

fn prepare_float64_flat_vec(array: &NdArray) -> Vec<f64> {
    prepare_float64_array(&array.flatten())
        .iter()
        .copied()
        .collect()
}

fn prepare_int64_flat_array(array: &NdArray) -> ArrayD<i64> {
    let cast = array.astype(DType::Int64);
    let flat = cast.flatten();
    let ArrayData::Int64(arr) = flat.data() else {
        unreachable!()
    };
    arr.clone()
}

fn prepare_choice_array(choice: &NdArray, dtype: DType, target_shape: &[usize]) -> NdArray {
    let cast = choice.astype(dtype);
    if cast.shape() == target_shape {
        cast
    } else {
        NdArray::from_data(crate::broadcasting::broadcast_array_data(
            cast.data(),
            target_shape,
        ))
    }
}

fn choose_choice_index(choice: i64, n_choices: usize) -> Result<usize> {
    if choice < 0 {
        return Err(NumpyError::ValueError(format!(
            "invalid entry {choice} in choice array (negative indices not supported)"
        )));
    }

    let choice_idx = choice as usize;
    if choice_idx >= n_choices {
        return Err(NumpyError::ValueError(format!(
            "invalid entry {choice} in choice array (out of range for {n_choices} choices)"
        )));
    }

    Ok(choice_idx)
}

fn choose_from_flat_choices<T: Copy>(
    idx_arr: &ArrayD<i64>,
    flat_choices: &[Vec<T>],
    n_choices: usize,
) -> Result<Vec<T>> {
    let mut result = Vec::with_capacity(idx_arr.len());
    for (pos, &choice) in idx_arr.iter().enumerate() {
        let choice_idx = choose_choice_index(choice, n_choices)?;
        let values = &flat_choices[choice_idx];
        if pos >= values.len() {
            return Err(NumpyError::ValueError(format!(
                "shape mismatch: index array has {} elements but choice {} has {}",
                idx_arr.len(),
                choice_idx,
                values.len()
            )));
        }
        result.push(values[pos]);
    }
    Ok(result)
}

fn prepare_float64_choice_vecs(choices: &[&NdArray], target_shape: &[usize]) -> Vec<Vec<f64>> {
    choices
        .iter()
        .map(|choice| {
            let prepared = prepare_choice_array(choice, DType::Float64, target_shape);
            let flattened = prepared.flatten();
            let ArrayData::Float64(arr) = flattened.data() else {
                unreachable!()
            };
            arr.iter().copied().collect()
        })
        .collect()
}

fn prepare_complex128_choice_vecs(
    choices: &[&NdArray],
    target_shape: &[usize],
) -> Vec<Vec<Complex<f64>>> {
    choices
        .iter()
        .map(|choice| {
            let prepared = prepare_choice_array(choice, DType::Complex128, target_shape);
            let flattened = prepared.flatten();
            let ArrayData::Complex128(arr) = flattened.data() else {
                unreachable!()
            };
            arr.iter().copied().collect()
        })
        .collect()
}

impl NdArray {
    /// Binary search in a sorted 1-D array. Returns Int64 array of insertion indices.
    /// side="left": leftmost insertion point. side="right": rightmost.
    pub fn searchsorted(&self, values: &NdArray, side: &str) -> Result<NdArray> {
        if self.ndim() != 1 {
            return Err(NumpyError::ValueError(
                "searchsorted requires a 1-D sorted array".into(),
            ));
        }
        let sorted_arr = prepare_float64_array(self);
        let vals_arr = prepare_float64_array(values);
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
        let ArrayData::Bool(mask) = cond.data() else {
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
    let idx_arr = prepare_int64_flat_array(a);

    let n_choices = choices.len();
    let target_shape = a.shape();

    let any_complex = choices.iter().any(|c| c.dtype().is_complex());
    let out_shape = a.shape().to_vec();

    if any_complex {
        let flat_choices = prepare_complex128_choice_vecs(choices, target_shape);
        let result = choose_from_flat_choices(&idx_arr, &flat_choices, n_choices)?;
        Ok(NdArray::from_data(ArrayData::Complex128(
            ArrayD::from_shape_vec(IxDyn(&out_shape), result).expect("shape matches"),
        )))
    } else {
        let flat_choices = prepare_float64_choice_vecs(choices, target_shape);
        let result = choose_from_flat_choices(&idx_arr, &flat_choices, n_choices)?;
        Ok(NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&out_shape), result).expect("shape matches"),
        )))
    }
}

/// Return sorted unique values present in both arrays.
pub fn intersect1d(a: &NdArray, b: &NdArray) -> NdArray {
    let a_values = prepare_float64_flat_vec(a);
    let b_values = prepare_float64_flat_vec(b);

    let b_set: HashSet<u64> = b_values.iter().map(|v| v.to_bits()).collect();
    let mut result: Vec<f64> = a_values
        .iter()
        .copied()
        .filter(|v| b_set.contains(&v.to_bits()))
        .collect();
    // Sort and dedup
    result.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup();

    NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).expect("shape matches"),
    ))
}

/// Return sorted unique values from either array.
pub fn union1d(a: &NdArray, b: &NdArray) -> NdArray {
    let a_values = prepare_float64_flat_vec(a);
    let b_values = prepare_float64_flat_vec(b);

    let mut result: Vec<f64> = a_values.iter().chain(b_values.iter()).copied().collect();
    result.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup();

    NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).expect("shape matches"),
    ))
}

/// Return sorted values in `a` that are NOT in `b`.
pub fn setdiff1d(a: &NdArray, b: &NdArray) -> NdArray {
    let a_values = prepare_float64_flat_vec(a);
    let b_values = prepare_float64_flat_vec(b);

    let b_set: HashSet<u64> = b_values.iter().map(|v| v.to_bits()).collect();
    let mut result: Vec<f64> = a_values
        .iter()
        .copied()
        .filter(|v| !b_set.contains(&v.to_bits()))
        .collect();
    result.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    result.dedup();

    NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).expect("shape matches"),
    ))
}

/// Return boolean array with same shape as `element`, true where value exists in `test_elements`.
pub fn isin(element: &NdArray, test_elements: &NdArray) -> NdArray {
    let elem_arr = prepare_float64_array(element);
    let test_arr = prepare_float64_flat_vec(test_elements);

    let test_set: HashSet<u64> = test_arr.iter().map(|v| v.to_bits()).collect();
    let shape: Vec<usize> = element.shape().to_vec();
    let result: Vec<bool> = elem_arr
        .iter()
        .map(|v| test_set.contains(&v.to_bits()))
        .collect();

    NdArray::from_data(ArrayData::Bool(
        ArrayD::from_shape_vec(IxDyn(&shape), result).expect("shape matches"),
    ))
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

    #[test]
    fn test_choose_broadcast_choice() {
        let a = NdArray::from_vec(vec![0_i64, 1, 0, 1]);
        let c0 = NdArray::from_vec(vec![10.0_f64]);
        let c1 = NdArray::from_vec(vec![50.0_f64, 60.0, 70.0, 80.0]);
        let result = choose(&a, &[&c0, &c1]).unwrap();
        let ArrayData::Float64(arr) = result.data() else {
            panic!("expected Float64");
        };
        assert_eq!(arr[[0]], 10.0);
        assert_eq!(arr[[1]], 60.0);
        assert_eq!(arr[[2]], 10.0);
        assert_eq!(arr[[3]], 80.0);
    }

    #[test]
    fn test_choose_complex_output() {
        let a = NdArray::from_vec(vec![0_i64, 1]);
        let c0 = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let c1 = NdArray::from_vec(vec![Complex::new(3.0_f64, 4.0), Complex::new(5.0_f64, 6.0)]);
        let result = choose(&a, &[&c0, &c1]).unwrap();
        let ArrayData::Complex128(arr) = result.data() else {
            panic!("expected Complex128");
        };
        assert_eq!(arr[[0]], Complex::new(1.0, 0.0));
        assert_eq!(arr[[1]], Complex::new(5.0, 6.0));
    }

    #[test]
    fn test_intersect1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let r = intersect1d(&a, &b);
        assert_eq!(r.shape(), &[2]);
        let ArrayData::Float64(arr) = r.data() else {
            panic!()
        };
        assert_eq!(arr[[0]], 2.0);
        assert_eq!(arr[[1]], 4.0);
    }

    #[test]
    fn test_intersect1d_no_overlap() {
        let a = NdArray::from_vec(vec![1.0_f64, 3.0, 5.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let r = intersect1d(&a, &b);
        assert_eq!(r.shape(), &[0]);
    }

    #[test]
    fn test_union1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 4.0, 5.0]);
        let r = union1d(&a, &b);
        assert_eq!(r.shape(), &[5]);
        let ArrayData::Float64(arr) = r.data() else {
            panic!()
        };
        assert_eq!(arr[[0]], 1.0);
        assert_eq!(arr[[1]], 2.0);
        assert_eq!(arr[[2]], 3.0);
        assert_eq!(arr[[3]], 4.0);
        assert_eq!(arr[[4]], 5.0);
    }

    #[test]
    fn test_setdiff1d() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 4.0]);
        let r = setdiff1d(&a, &b);
        assert_eq!(r.shape(), &[3]);
        let ArrayData::Float64(arr) = r.data() else {
            panic!()
        };
        assert_eq!(arr[[0]], 1.0);
        assert_eq!(arr[[1]], 3.0);
        assert_eq!(arr[[2]], 5.0);
    }

    #[test]
    fn test_isin_basic() {
        let element = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let test_elements = NdArray::from_vec(vec![2.0_f64, 4.0]);
        let r = isin(&element, &test_elements);
        assert_eq!(r.shape(), &[5]);
        let ArrayData::Bool(arr) = r.data() else {
            panic!()
        };
        assert!(!arr[[0]]); // 1.0 not in test
        assert!(arr[[1]]); // 2.0 in test
        assert!(!arr[[2]]); // 3.0 not in test
        assert!(arr[[3]]); // 4.0 in test
        assert!(!arr[[4]]); // 5.0 not in test
    }

    #[test]
    fn test_isin_preserves_shape() {
        // 2x3 array
        let element = NdArray::from_data(ArrayData::Float64(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ));
        let test_elements = NdArray::from_vec(vec![2.0_f64, 4.0, 6.0]);
        let r = isin(&element, &test_elements);
        assert_eq!(r.shape(), &[2, 3]);
        let ArrayData::Bool(arr) = r.data() else {
            panic!()
        };
        assert!(!arr[[0, 0]]); // 1.0
        assert!(arr[[0, 1]]); // 2.0
        assert!(!arr[[0, 2]]); // 3.0
        assert!(arr[[1, 0]]); // 4.0
        assert!(!arr[[1, 1]]); // 5.0
        assert!(arr[[1, 2]]); // 6.0
    }
}
