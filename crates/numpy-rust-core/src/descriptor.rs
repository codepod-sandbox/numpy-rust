use crate::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeKind {
    Bool,
    SignedInteger,
    UnsignedInteger,
    Float,
    Complex,
    String,
}

#[derive(Debug)]
pub struct DTypeDescriptor {
    pub id: DType,
    pub name: &'static str,
    pub itemsize: usize,
    pub kind: DTypeKind,
}

impl DTypeDescriptor {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    pub fn kind(&self) -> DTypeKind {
        self.kind
    }
}

static BOOL_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Bool,
    name: "bool",
    itemsize: 1,
    kind: DTypeKind::Bool,
};

static INT8_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int8,
    name: "int8",
    itemsize: 1,
    kind: DTypeKind::SignedInteger,
};

static INT16_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int16,
    name: "int16",
    itemsize: 2,
    kind: DTypeKind::SignedInteger,
};

static INT32_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int32,
    name: "int32",
    itemsize: 4,
    kind: DTypeKind::SignedInteger,
};

static INT64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int64,
    name: "int64",
    itemsize: 8,
    kind: DTypeKind::SignedInteger,
};

static UINT8_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt8,
    name: "uint8",
    itemsize: 1,
    kind: DTypeKind::UnsignedInteger,
};

static UINT16_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt16,
    name: "uint16",
    itemsize: 2,
    kind: DTypeKind::UnsignedInteger,
};

static UINT32_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt32,
    name: "uint32",
    itemsize: 4,
    kind: DTypeKind::UnsignedInteger,
};

static UINT64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt64,
    name: "uint64",
    itemsize: 8,
    kind: DTypeKind::UnsignedInteger,
};

static FLOAT16_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Float16,
    name: "float16",
    itemsize: 2,
    kind: DTypeKind::Float,
};

static FLOAT32_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Float32,
    name: "float32",
    itemsize: 4,
    kind: DTypeKind::Float,
};

static FLOAT64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Float64,
    name: "float64",
    itemsize: 8,
    kind: DTypeKind::Float,
};

static COMPLEX64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Complex64,
    name: "complex64",
    itemsize: 8,
    kind: DTypeKind::Complex,
};

static COMPLEX128_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Complex128,
    name: "complex128",
    itemsize: 16,
    kind: DTypeKind::Complex,
};

static STR_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Str,
    name: "str",
    itemsize: 0,
    kind: DTypeKind::String,
};

pub fn descriptor_for_dtype(dtype: DType) -> &'static DTypeDescriptor {
    match dtype {
        DType::Bool => &BOOL_DESCRIPTOR,
        DType::Int8 => &INT8_DESCRIPTOR,
        DType::Int16 => &INT16_DESCRIPTOR,
        DType::Int32 => &INT32_DESCRIPTOR,
        DType::Int64 => &INT64_DESCRIPTOR,
        DType::UInt8 => &UINT8_DESCRIPTOR,
        DType::UInt16 => &UINT16_DESCRIPTOR,
        DType::UInt32 => &UINT32_DESCRIPTOR,
        DType::UInt64 => &UINT64_DESCRIPTOR,
        DType::Float16 => &FLOAT16_DESCRIPTOR,
        DType::Float32 => &FLOAT32_DESCRIPTOR,
        DType::Float64 => &FLOAT64_DESCRIPTOR,
        DType::Complex64 => &COMPLEX64_DESCRIPTOR,
        DType::Complex128 => &COMPLEX128_DESCRIPTOR,
        DType::Str => &STR_DESCRIPTOR,
    }
}
