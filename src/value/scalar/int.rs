//! The heap `i64` number representation (tag `NumberI64`): the eight payload bytes
//! are a signed 64-bit integer.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::{alloc, free, read};
use crate::number::INumber;
use crate::value::{
    number_cmp, Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, NumberRepr,
    TypeTag, ValueRepr, ValueType,
};

/// The heap `i64` number representation.
pub(crate) struct I64Repr;

impl I64Repr {
    /// Stores an `i64` as a heap scalar. Always succeeds — the heap holds any
    /// `i64` — so it is the total fallback in construction.
    pub(crate) fn store(value: i64) -> IValue {
        // Safety: `alloc` returns a fresh, aligned, non-null allocation.
        unsafe { IValue::new_ptr(alloc::<i64>(value), TypeTag::NumberI64) }
    }
}

impl ValueRepr for I64Repr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        self.num_val(v).hash(state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(self.num_val(a), b) == Some(Ordering::Equal)
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        number_cmp(self.num_val(a), b)
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.num_val(v))
    }
    fn destructure(&self, v: IValue) -> Destructured {
        Destructured::Number(INumber(v))
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        DestructuredRef::Number(v.as_number_unchecked())
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        DestructuredMut::Number(v.as_number_unchecked_mut())
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        Self::store(read::<i64>(v.ptr()))
    }
    unsafe fn drop(&self, v: &mut IValue) {
        free::<i64>(v.ptr());
    }
}

impl NumberRepr for I64Repr {
    /// Decodes the payload as an `i64`. Safety: `v` must be a live `NumberI64`.
    unsafe fn num_val(&self, v: &IValue) -> NumVal {
        NumVal::Int(read::<i64>(v.ptr()))
    }
    // `has_decimal_point` and the numeric conversions use the `NumberRepr` defaults
    // (an integer, never written with a decimal point; conversions via `num_val`).
}
