use crate::array_data::ArrayD;
use crate::array_data::ArrayData;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Helper: ensure the array is of string dtype.
fn require_string_data(arr: &NdArray) -> Result<&ArrayD<String>> {
    match arr.data() {
        ArrayData::Str(a) => Ok(a),
        _ => Err(NumpyError::TypeError(
            "string operation requires string array".into(),
        )),
    }
}

fn map_string_output<F>(arr: &NdArray, op: F) -> Result<NdArray>
where
    F: Fn(&String) -> String + Copy,
{
    let data = require_string_data(arr)?;
    Ok(NdArray::from_data(ArrayData::Str(
        data.map(op).into_owned().into_shared(),
    )))
}

fn map_bool_output<F>(arr: &NdArray, op: F) -> Result<NdArray>
where
    F: Fn(&String) -> bool + Copy,
{
    let data = require_string_data(arr)?;
    Ok(NdArray::from_data(ArrayData::Bool(
        data.map(op).into_owned().into_shared(),
    )))
}

fn map_int_output<F>(arr: &NdArray, op: F) -> Result<NdArray>
where
    F: Fn(&String) -> i64 + Copy,
{
    let data = require_string_data(arr)?;
    Ok(NdArray::from_data(ArrayData::Int64(
        data.map(op).into_owned().into_shared(),
    )))
}

impl NdArray {
    /// Convert each string element to uppercase.
    pub fn str_upper(&self) -> Result<NdArray> {
        map_string_output(self, |s| s.to_uppercase())
    }

    /// Convert each string element to lowercase.
    pub fn str_lower(&self) -> Result<NdArray> {
        map_string_output(self, |s| s.to_lowercase())
    }

    /// Capitalize the first character of each string element.
    pub fn str_capitalize(&self) -> Result<NdArray> {
        map_string_output(self, |s| {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    upper + &chars.as_str().to_lowercase()
                }
            }
        })
    }

    /// Strip leading and trailing whitespace from each element.
    pub fn str_strip(&self) -> Result<NdArray> {
        map_string_output(self, |s| s.trim().to_string())
    }

    /// Return the length of each string element as an Int64 array.
    pub fn str_len(&self) -> Result<NdArray> {
        map_int_output(self, |s| s.chars().count() as i64)
    }

    /// Test whether each string element starts with the given prefix, returning a Bool array.
    pub fn str_startswith(&self, prefix: &str) -> Result<NdArray> {
        let prefix = prefix.to_string();
        map_bool_output(self, |s| s.starts_with(prefix.as_str()))
    }

    /// Test whether each string element ends with the given suffix, returning a Bool array.
    pub fn str_endswith(&self, suffix: &str) -> Result<NdArray> {
        let suffix = suffix.to_string();
        map_bool_output(self, |s| s.ends_with(suffix.as_str()))
    }

    /// Replace occurrences of `old` with `new` in each string element.
    pub fn str_replace(&self, old: &str, new: &str) -> Result<NdArray> {
        let old = old.to_string();
        let new = new.to_string();
        map_string_output(self, |s| s.replace(old.as_str(), new.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_str_upper() {
        let a = NdArray::from_vec(vec!["hello".to_string(), "world".to_string()]);
        let b = a.str_upper().unwrap();
        assert_eq!(b.dtype(), DType::Str);
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "HELLO");
            assert_eq!(arr[[1]], "WORLD");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_lower() {
        let a = NdArray::from_vec(vec!["HELLO".to_string(), "WORLD".to_string()]);
        let b = a.str_lower().unwrap();
        assert_eq!(b.dtype(), DType::Str);
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "hello");
            assert_eq!(arr[[1]], "world");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_capitalize() {
        let a = NdArray::from_vec(vec!["hello world".to_string(), "fOO".to_string()]);
        let b = a.str_capitalize().unwrap();
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "Hello world");
            assert_eq!(arr[[1]], "Foo");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_strip() {
        let a = NdArray::from_vec(vec!["  hello  ".to_string(), "\tworld\n".to_string()]);
        let b = a.str_strip().unwrap();
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "hello");
            assert_eq!(arr[[1]], "world");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_len() {
        let a = NdArray::from_vec(vec!["hi".to_string(), "hello".to_string()]);
        let b = a.str_len().unwrap();
        assert_eq!(b.dtype(), DType::Int64);
        if let crate::ArrayData::Int64(arr) = b.data() {
            assert_eq!(arr[[0]], 2);
            assert_eq!(arr[[1]], 5);
        } else {
            panic!("expected Int64");
        }
    }

    #[test]
    fn test_str_startswith_endswith() {
        let a = NdArray::from_vec(vec!["hello".to_string(), "world".to_string()]);
        let sw = a.str_startswith("he").unwrap();
        assert_eq!(sw.dtype(), DType::Bool);
        if let crate::ArrayData::Bool(arr) = sw.data() {
            assert!(arr[[0]]);
            assert!(!arr[[1]]);
        } else {
            panic!("expected Bool");
        }

        let ew = a.str_endswith("ld").unwrap();
        if let crate::ArrayData::Bool(arr) = ew.data() {
            assert!(!arr[[0]]);
            assert!(arr[[1]]);
        } else {
            panic!("expected Bool");
        }
    }

    #[test]
    fn test_str_replace() {
        let a = NdArray::from_vec(vec!["hello world".to_string(), "foo bar".to_string()]);
        let b = a.str_replace("o", "0").unwrap();
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "hell0 w0rld");
            assert_eq!(arr[[1]], "f00 bar");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_non_string_error() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(a.str_upper().is_err());
        assert!(a.str_lower().is_err());
        assert!(a.str_capitalize().is_err());
        assert!(a.str_strip().is_err());
        assert!(a.str_len().is_err());
        assert!(a.str_startswith("x").is_err());
        assert!(a.str_endswith("x").is_err());
        assert!(a.str_replace("a", "b").is_err());
    }

    #[test]
    fn test_string_helpers_share_non_string_type_error_message() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let errors = [
            a.str_upper().unwrap_err().to_string(),
            a.str_lower().unwrap_err().to_string(),
            a.str_capitalize().unwrap_err().to_string(),
            a.str_strip().unwrap_err().to_string(),
            a.str_len().unwrap_err().to_string(),
            a.str_startswith("x").unwrap_err().to_string(),
            a.str_endswith("x").unwrap_err().to_string(),
            a.str_replace("a", "b").unwrap_err().to_string(),
        ];

        for error in errors {
            assert_eq!(error, "type error: string operation requires string array");
        }
    }
}
