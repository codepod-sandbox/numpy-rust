use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, IxDyn};

use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

fn parse_subscripts(s: &str) -> Result<(Vec<Vec<char>>, Vec<char>)> {
    let parts: Vec<&str> = s.split("->").collect();
    if parts.len() != 2 {
        return Err(NumpyError::ValueError(
            "einsum requires explicit '->' output subscripts".into(),
        ));
    }
    let inputs: Vec<Vec<char>> = parts[0]
        .split(',')
        .map(|p| p.trim().chars().collect())
        .collect();
    let output: Vec<char> = parts[1].trim().chars().collect();
    Ok((inputs, output))
}

pub fn einsum(subscripts: &str, operands: &[&NdArray]) -> Result<NdArray> {
    let (input_subs, output_sub) = parse_subscripts(subscripts)?;
    if input_subs.len() != operands.len() {
        return Err(NumpyError::ValueError(format!(
            "einsum: {} operands but subscripts specify {}",
            operands.len(),
            input_subs.len()
        )));
    }

    // Cast all inputs to Float64
    let float_ops: Vec<NdArray> = operands.iter().map(|a| a.astype(DType::Float64)).collect();
    let arrays: Vec<&ArrayD<f64>> = float_ops
        .iter()
        .map(|a| match &a.data {
            ArrayData::Float64(arr) => arr,
            _ => unreachable!(),
        })
        .collect();

    // Build index→size map
    let mut index_sizes: HashMap<char, usize> = HashMap::new();
    for (subs, arr) in input_subs.iter().zip(arrays.iter()) {
        if subs.len() != arr.ndim() {
            return Err(NumpyError::ValueError(format!(
                "einsum: operand has {} dims but subscript has {} indices",
                arr.ndim(),
                subs.len()
            )));
        }
        for (&c, &dim) in subs.iter().zip(arr.shape().iter()) {
            if let Some(&existing) = index_sizes.get(&c) {
                if existing != dim {
                    return Err(NumpyError::ShapeMismatch(format!(
                        "einsum: index '{c}' has size {existing} and {dim}"
                    )));
                }
            } else {
                index_sizes.insert(c, dim);
            }
        }
    }

    // Identify contracted indices (in input but not in output)
    let output_set: HashSet<char> = output_sub.iter().copied().collect();
    let all_indices: HashSet<char> = input_subs.iter().flat_map(|s| s.iter().copied()).collect();
    let contracted: Vec<char> = all_indices.difference(&output_set).copied().collect();

    // Build output shape
    let output_shape: Vec<usize> = output_sub
        .iter()
        .map(|&c| {
            *index_sizes
                .get(&c)
                .expect("output index must exist in inputs")
        })
        .collect();
    let output_size: usize = output_shape.iter().product::<usize>().max(1);
    let mut result_data = vec![0.0f64; output_size];

    let output_ranges: Vec<usize> = output_sub.iter().map(|c| index_sizes[c]).collect();
    let contracted_ranges: Vec<usize> = contracted.iter().map(|c| index_sizes[c]).collect();

    fn multi_index_iter(ranges: &[usize]) -> Vec<Vec<usize>> {
        if ranges.is_empty() {
            return vec![vec![]];
        }
        let mut result = Vec::new();
        let sub = multi_index_iter(&ranges[1..]);
        for i in 0..ranges[0] {
            for s in &sub {
                let mut v = vec![i];
                v.extend_from_slice(s);
                result.push(v);
            }
        }
        result
    }

    let output_indices = multi_index_iter(&output_ranges);
    let contract_indices = multi_index_iter(&contracted_ranges);

    // Iterate over output+contracted index combinations, sum products
    for (flat_idx, out_idx) in output_indices.iter().enumerate() {
        let mut sum = 0.0f64;
        let mut idx_map: HashMap<char, usize> = HashMap::new();
        for (i, &c) in output_sub.iter().enumerate() {
            idx_map.insert(c, out_idx[i]);
        }
        for cont_idx in &contract_indices {
            for (i, &c) in contracted.iter().enumerate() {
                idx_map.insert(c, cont_idx[i]);
            }
            let mut product = 1.0f64;
            for (subs, arr) in input_subs.iter().zip(arrays.iter()) {
                let arr_idx: Vec<usize> = subs.iter().map(|c| idx_map[c]).collect();
                product *= arr[IxDyn(&arr_idx)];
            }
            sum += product;
        }
        result_data[flat_idx] = sum;
    }

    let out_shape = if output_shape.is_empty() {
        IxDyn(&[])
    } else {
        IxDyn(&output_shape)
    };
    Ok(NdArray::from_data(ArrayData::Float64(
        ArrayD::from_shape_vec(out_shape, result_data).expect("output shape matches"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_einsum_matmul() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = NdArray::from_vec(vec![5.0, 6.0, 7.0, 8.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_einsum_trace() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = einsum("ii->", &[&a]).unwrap();
        assert_eq!(c.shape(), &[]);
    }

    #[test]
    fn test_einsum_transpose() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let c = einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
    }

    #[test]
    fn test_einsum_outer() {
        let a = NdArray::from_vec(vec![1.0, 2.0]);
        let b = NdArray::from_vec(vec![3.0, 4.0, 5.0]);
        let c = einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }
}
