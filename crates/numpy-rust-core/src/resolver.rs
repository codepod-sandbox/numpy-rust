use std::fmt;

use crate::descriptor::{descriptor_for_dtype, DTypeKind};
use crate::{DType, NumpyError, Result};

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastingRule {
    SameKind,
    Unsafe,
}

impl fmt::Display for CastingRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CastingRule::SameKind => write!(f, "same_kind"),
            CastingRule::Unsafe => write!(f, "unsafe"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastPlan {
    source_dtype: DType,
    source_storage_dtype: DType,
    target_dtype: DType,
    execution_storage_dtype: DType,
}

impl CastPlan {
    pub fn source_dtype(&self) -> DType {
        self.source_dtype
    }

    pub fn source_storage_dtype(&self) -> DType {
        self.source_storage_dtype
    }

    pub fn target_dtype(&self) -> DType {
        self.target_dtype
    }

    pub fn target_storage_dtype(&self) -> DType {
        self.execution_storage_dtype
    }

    pub fn execution_storage_dtype(&self) -> DType {
        self.execution_storage_dtype
    }

    pub fn requires_storage_cast(&self) -> bool {
        self.source_storage_dtype != self.execution_storage_dtype
    }

    pub fn requires_narrowing(&self) -> bool {
        self.target_dtype.is_narrow()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryOpPlan {
    lhs_cast: CastPlan,
    rhs_cast: CastPlan,
    result_cast: CastPlan,
    logical_output_dtype: DType,
    result_storage_dtype: DType,
    requires_output_narrowing: bool,
}

impl BinaryOpPlan {
    pub fn lhs_cast(&self) -> CastPlan {
        self.lhs_cast
    }

    pub fn rhs_cast(&self) -> CastPlan {
        self.rhs_cast
    }

    pub fn result_cast(&self) -> CastPlan {
        self.result_cast
    }

    pub fn logical_output_dtype(&self) -> DType {
        self.logical_output_dtype
    }

    pub fn output_dtype(&self) -> DType {
        self.logical_output_dtype
    }

    pub fn result_storage_dtype(&self) -> DType {
        self.result_storage_dtype
    }

    pub fn requires_output_narrowing(&self) -> bool {
        self.requires_output_narrowing
    }
}

pub fn resolve_cast(source: DType, target: DType, rule: CastingRule) -> Result<CastPlan> {
    if !is_cast_allowed(source, target, rule) {
        return Err(NumpyError::TypeError(format!(
            "cannot cast {source} to {target} under {rule}"
        )));
    }

    Ok(CastPlan {
        source_dtype: source,
        source_storage_dtype: source.storage_dtype(),
        target_dtype: target,
        execution_storage_dtype: target.storage_dtype(),
    })
}

pub fn resolve_binary_op(op: BinaryOp, lhs: DType, rhs: DType) -> Result<BinaryOpPlan> {
    match op {
        BinaryOp::Add => resolve_add(lhs, rhs),
    }
}

fn resolve_add(lhs: DType, rhs: DType) -> Result<BinaryOpPlan> {
    if lhs.is_string() || rhs.is_string() {
        return Err(NumpyError::TypeError(
            "arithmetic not supported for string arrays".into(),
        ));
    }

    let logical_output_dtype = resolve_add_output_dtype(lhs, rhs);
    let lhs_cast = resolve_cast(lhs, logical_output_dtype, CastingRule::Unsafe)?;
    let rhs_cast = resolve_cast(rhs, logical_output_dtype, CastingRule::Unsafe)?;
    let result_storage_dtype = logical_output_dtype.storage_dtype();
    let result_cast = resolve_cast(
        result_storage_dtype,
        logical_output_dtype,
        CastingRule::Unsafe,
    )?;

    Ok(BinaryOpPlan {
        lhs_cast,
        rhs_cast,
        result_cast,
        logical_output_dtype,
        result_storage_dtype,
        requires_output_narrowing: result_cast.requires_narrowing(),
    })
}

fn resolve_add_output_dtype(lhs: DType, rhs: DType) -> DType {
    if lhs.is_bool() && rhs.is_bool() {
        DType::Int8
    } else {
        lhs.promote(rhs)
    }
}

fn is_cast_allowed(source: DType, target: DType, rule: CastingRule) -> bool {
    if source == target {
        return true;
    }

    match rule {
        CastingRule::Unsafe => true,
        CastingRule::SameKind => same_kind_cast_allowed(source, target),
    }
}

fn same_kind_cast_allowed(source: DType, target: DType) -> bool {
    let source_kind = descriptor_for_dtype(source).kind();
    let target_kind = descriptor_for_dtype(target).kind();

    if matches!(source_kind, DTypeKind::String) || matches!(target_kind, DTypeKind::String) {
        return matches!(
            (source_kind, target_kind),
            (DTypeKind::String, DTypeKind::String)
        );
    }

    same_kind_rank(source) <= same_kind_rank(target)
}

fn same_kind_rank(dtype: DType) -> u8 {
    match descriptor_for_dtype(dtype).kind() {
        DTypeKind::Bool => 0,
        DTypeKind::UnsignedInteger => 1,
        DTypeKind::SignedInteger => 2,
        DTypeKind::Float => 3,
        DTypeKind::Complex => 4,
        DTypeKind::String => 5,
    }
}
