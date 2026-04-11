use crate::DType;

#[derive(Debug)]
pub struct DTypeDescriptor {
    pub id: DType,
    pub name: &'static str,
    pub itemsize: usize,
}

impl DTypeDescriptor {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
}

static FLOAT64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Float64,
    name: "float64",
    itemsize: 8,
};

static INT32_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int32,
    name: "int32",
    itemsize: 4,
};

pub fn descriptor_for_dtype(dtype: DType) -> &'static DTypeDescriptor {
    match dtype {
        DType::Float64 => &FLOAT64_DESCRIPTOR,
        DType::Int32 => &INT32_DESCRIPTOR,
        _ => unimplemented!("register descriptor for {:?}", dtype),
    }
}
