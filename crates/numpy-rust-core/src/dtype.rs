/// Supported data types, mirroring NumPy's core numeric dtypes plus strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,  // Complex<f32>
    Complex128, // Complex<f64>
    Str,
}

impl DType {
    /// Returns the common type that both `self` and `other` can be promoted to,
    /// following NumPy's type promotion rules.
    ///
    /// Promotion lattice (uses storage types for narrow dtypes):
    ///   Bool -> Int8 -> Int16 -> Int32 -> Int64 -> Float64
    ///   UInt8 -> UInt16 -> UInt32 -> UInt64 -> Float64
    ///   Float16 -> Float32 -> Float64
    ///   Complex64 / Complex128 follow float promotion lifted to complex
    pub fn promote(self, other: DType) -> DType {
        // Use the actual (logical) dtypes, not storage types, so that
        // narrow dtypes like Int8 are preserved when both operands match
        // (NumPy "same-kind" promotion: int8 + int8 -> int8).
        let a = self;
        let b = other;
        if a == b {
            return a;
        }
        if a == DType::Str || b == DType::Str {
            return DType::Str;
        }
        // If either is complex, result is complex
        if a.is_complex() || b.is_complex() {
            if a.is_complex() && b.is_complex() {
                return if a.rank() >= b.rank() { a } else { b };
            }
            let real_type = if a.is_complex() { b } else { a };
            let complex_type = if a.is_complex() { a } else { b };
            return match (complex_type, real_type) {
                (DType::Complex64, DType::Float64 | DType::Int64) => DType::Complex128,
                (DType::Complex128, _) => DType::Complex128,
                (DType::Complex64, _) => DType::Complex64,
                _ => DType::Complex128,
            };
        }
        let (hi, lo) = if a.rank() >= b.rank() { (a, b) } else { (b, a) };

        // Bool + integer/float -> the non-bool type (NumPy: bool_ treated as int8 for promotion)
        if lo == DType::Bool {
            return hi;
        }

        // Both integer: use NumPy rules
        if hi.is_integer() && lo.is_integer() {
            let hi_signed = hi.is_signed_int();
            let lo_signed = lo.is_signed_int();
            if hi_signed == lo_signed {
                // Same signedness: pick the wider one
                return hi;
            }
            // Mixed signed/unsigned: if signed bits > unsigned bits, use signed
            let hi_bits = hi.bit_width();
            let lo_bits = lo.bit_width();
            let (s_bits, u_bits) = if hi_signed {
                (hi_bits, lo_bits)
            } else {
                (lo_bits, hi_bits)
            };
            if s_bits > u_bits {
                // Signed type can hold unsigned values
                return if hi_signed { hi } else { lo };
            }
            // Need wider signed type
            return match u_bits {
                8 => DType::Int16,
                16 => DType::Int32,
                32 => DType::Int64,
                _ => DType::Float64,
            };
        }

        // Special case: mixing i64 with f32 promotes to f64 to avoid precision loss
        if (hi == DType::Float32 && lo == DType::Int64)
            || (hi == DType::Int64 && lo == DType::Float32)
        {
            return DType::Float64;
        }

        // Integer + float -> float (pick the float type, or widen if needed)
        // Float + float -> wider float
        hi
    }

    /// Map narrow dtypes to their internal storage type.
    /// Canonical types (Bool, Int32, Int64, Float32, Float64, Complex64, Complex128, Str)
    /// map to themselves.
    pub fn storage_dtype(self) -> DType {
        match self {
            DType::Int8 | DType::Int16 => DType::Int32,
            DType::UInt8 | DType::UInt16 => DType::Int32,
            DType::UInt32 | DType::UInt64 => DType::Int64,
            DType::Float16 => DType::Float32,
            other => other,
        }
    }

    /// Returns true if this is a narrow dtype that requires internal widening.
    pub fn is_narrow(self) -> bool {
        self.storage_dtype() != self
    }

    /// Numeric rank for promotion ordering.
    fn rank(self) -> u8 {
        match self {
            DType::Bool => 0,
            DType::Int8 | DType::UInt8 => 1,
            DType::Int16 | DType::UInt16 => 2,
            DType::Int32 | DType::UInt32 => 3,
            DType::Int64 | DType::UInt64 => 4,
            DType::Float16 => 5,
            DType::Float32 => 6,
            DType::Float64 => 7,
            DType::Complex64 => 8,
            DType::Complex128 => 9,
            DType::Str => 255,
        }
    }

    /// Size in bytes of a single element.
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Int16 | DType::UInt16 | DType::Float16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 4,
            DType::Int64 | DType::UInt64 | DType::Float64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
            DType::Str => 0, // variable-length
        }
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, DType::Float16 | DType::Float32 | DType::Float64)
    }

    /// Returns true if this is an integer type (signed or unsigned).
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64
        )
    }

    /// Returns true if this is an unsigned integer type.
    pub fn is_unsigned(self) -> bool {
        matches!(
            self,
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64
        )
    }

    /// Returns true if this is a complex type.
    pub fn is_complex(self) -> bool {
        matches!(self, DType::Complex64 | DType::Complex128)
    }

    /// Returns true if this is a signed integer type.
    pub fn is_signed_int(self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    /// Returns the bit width for integer types. Returns 0 for non-integer types.
    pub fn bit_width(self) -> u32 {
        match self {
            DType::Int8 | DType::UInt8 => 8,
            DType::Int16 | DType::UInt16 => 16,
            DType::Int32 | DType::UInt32 => 32,
            DType::Int64 | DType::UInt64 => 64,
            _ => 0,
        }
    }

    /// Returns true if this is a string type.
    pub fn is_string(self) -> bool {
        matches!(self, DType::Str)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::Int8 => write!(f, "int8"),
            DType::Int16 => write!(f, "int16"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::UInt8 => write!(f, "uint8"),
            DType::UInt16 => write!(f, "uint16"),
            DType::UInt32 => write!(f, "uint32"),
            DType::UInt64 => write!(f, "uint64"),
            DType::Float16 => write!(f, "float16"),
            DType::Float32 => write!(f, "float32"),
            DType::Float64 => write!(f, "float64"),
            DType::Complex64 => write!(f, "complex64"),
            DType::Complex128 => write!(f, "complex128"),
            DType::Str => write!(f, "str"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_same_type() {
        assert_eq!(DType::Float64.promote(DType::Float64), DType::Float64);
        assert_eq!(DType::Int32.promote(DType::Int32), DType::Int32);
        assert_eq!(DType::Complex64.promote(DType::Complex64), DType::Complex64);
        assert_eq!(
            DType::Complex128.promote(DType::Complex128),
            DType::Complex128
        );
    }

    #[test]
    fn test_promote_bool_widens() {
        assert_eq!(DType::Bool.promote(DType::Int32), DType::Int32);
        assert_eq!(DType::Bool.promote(DType::Float64), DType::Float64);
    }

    #[test]
    fn test_promote_int_to_float() {
        assert_eq!(DType::Int32.promote(DType::Float32), DType::Float32);
        assert_eq!(DType::Int64.promote(DType::Float64), DType::Float64);
    }

    #[test]
    fn test_promote_int32_int64() {
        assert_eq!(DType::Int32.promote(DType::Int64), DType::Int64);
    }

    #[test]
    fn test_promote_mixed_int_float() {
        // i64 + f32 -> f64 (to avoid precision loss)
        assert_eq!(DType::Int64.promote(DType::Float32), DType::Float64);
    }

    #[test]
    fn test_promote_is_symmetric() {
        let pairs = [
            (DType::Int32, DType::Float64),
            (DType::Bool, DType::Int64),
            (DType::Float32, DType::Int32),
            (DType::Int64, DType::Float32),
            (DType::Float32, DType::Complex64),
            (DType::Float64, DType::Complex128),
            (DType::Int32, DType::Complex64),
        ];
        for (a, b) in pairs {
            assert_eq!(
                a.promote(b),
                b.promote(a),
                "promote({a:?}, {b:?}) not symmetric"
            );
        }
    }

    #[test]
    fn test_promote_complex() {
        // Float32 + Complex64 -> Complex64
        assert_eq!(DType::Float32.promote(DType::Complex64), DType::Complex64);
        // Float64 + Complex64 -> Complex128
        assert_eq!(DType::Float64.promote(DType::Complex64), DType::Complex128);
        // Int64 + Complex64 -> Complex128
        assert_eq!(DType::Int64.promote(DType::Complex64), DType::Complex128);
        // Int32 + Complex64 -> Complex64
        assert_eq!(DType::Int32.promote(DType::Complex64), DType::Complex64);
        // Complex64 + Complex128 -> Complex128
        assert_eq!(
            DType::Complex64.promote(DType::Complex128),
            DType::Complex128
        );
        // Bool + Complex64 -> Complex64
        assert_eq!(DType::Bool.promote(DType::Complex64), DType::Complex64);
        // Float64 + Complex128 -> Complex128
        assert_eq!(DType::Float64.promote(DType::Complex128), DType::Complex128);
    }

    #[test]
    fn test_itemsize() {
        assert_eq!(DType::Bool.itemsize(), 1);
        assert_eq!(DType::Int32.itemsize(), 4);
        assert_eq!(DType::Float64.itemsize(), 8);
        assert_eq!(DType::Complex64.itemsize(), 8);
        assert_eq!(DType::Complex128.itemsize(), 16);
    }

    #[test]
    fn test_is_float() {
        assert!(DType::Float32.is_float());
        assert!(DType::Float64.is_float());
        assert!(!DType::Int32.is_float());
        assert!(!DType::Bool.is_float());
        assert!(!DType::Complex64.is_float());
        assert!(!DType::Complex128.is_float());
    }

    #[test]
    fn test_is_complex() {
        assert!(DType::Complex64.is_complex());
        assert!(DType::Complex128.is_complex());
        assert!(!DType::Float64.is_complex());
        assert!(!DType::Int32.is_complex());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", DType::Float64), "float64");
        assert_eq!(format!("{}", DType::Bool), "bool");
        assert_eq!(format!("{}", DType::Complex64), "complex64");
        assert_eq!(format!("{}", DType::Complex128), "complex128");
    }
}
