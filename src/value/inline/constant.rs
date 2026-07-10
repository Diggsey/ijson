//! The inline constant representations: `null` and the two booleans.
//!
//! Each is a single fixed bit pattern (`super::NULL` / `FALSE` / `TRUE`) with no
//! payload to decode. Cloning is a bit-copy and there is nothing to drop, so both
//! representations keep the inline defaults for everything except comparison,
//! debug formatting and destructuring.

use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};

use super::InlineValue;
use crate::value::{BoolMut, Destructured, DestructuredMut, DestructuredRef, IValue, ValueType};

pub(crate) struct NullRepr;
impl InlineValue for NullRepr {
    fn value_type(&self) -> ValueType {
        ValueType::Null
    }
    // clone/drop/hash/eq use the defaults (bit-copy / nothing / pointer word /
    // `raw_eq`), all correct for the single `null` value.
    unsafe fn partial_cmp(&self, _a: &IValue, _b: &IValue) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
    unsafe fn debug(&self, _v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("null")
    }
    fn destructure(&self, _v: IValue) -> Destructured {
        Destructured::Null
    }
    unsafe fn destructure_ref<'a>(&self, _v: &'a IValue) -> DestructuredRef<'a> {
        DestructuredRef::Null
    }
    unsafe fn destructure_mut<'a>(&self, _v: &'a mut IValue) -> DestructuredMut<'a> {
        DestructuredMut::Null
    }
}

pub(crate) struct BoolRepr;
impl InlineValue for BoolRepr {
    fn value_type(&self) -> ValueType {
        ValueType::Bool
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        a.is_true().partial_cmp(&b.is_true())
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&v.is_true(), f)
    }
    fn destructure(&self, v: IValue) -> Destructured {
        Destructured::Bool(v.is_true())
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        DestructuredRef::Bool(v.is_true())
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        DestructuredMut::Bool(BoolMut(v))
    }
    // `to_bool` is not a `ValueRepr`/`InlineValue` operation: `IValue::to_bool`
    // decodes the two constant bit patterns directly.
}
