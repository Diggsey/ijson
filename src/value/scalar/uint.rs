//! The heap `u64` number representation (tag `NumberU64`): the eight payload bytes
//! are an unsigned 64-bit integer. Reached only for values above `i64::MAX` —
//! smaller ones canonicalise to the signed [`super::int`] representation.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::{alloc, free, read};
use crate::number::INumber;
use crate::value::{
    num_debug, num_hash, num_to_f64, num_to_f64_lossy, num_to_i64, num_to_u64, number_cmp,
    Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, TypeTag, ValueRepr, ValueType,
};

/// The heap `u64` number representation.
pub(crate) struct U64Repr;

impl U64Repr {
    /// Stores a `u64` as a heap scalar. Always succeeds — the heap holds any
    /// `u64` — so it is the total fallback in construction.
    pub(crate) fn store(value: u64) -> IValue {
        // Safety: `alloc` returns a fresh, aligned, non-null allocation.
        unsafe { IValue::new_ptr(alloc(value), TypeTag::NumberU64) }
    }

    /// Decodes the 8-byte payload as a `u64`.
    /// Safety: `v` must be a live `NumberU64` scalar.
    pub(super) unsafe fn num_val(v: &IValue) -> NumVal {
        NumVal::UInt(read(v.ptr()))
    }
}

impl ValueRepr for U64Repr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    fn has_decimal_point(&self, _v: &IValue) -> bool {
        false
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        num_hash(Self::num_val(v), state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(a, b) == Ordering::Equal
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        Some(number_cmp(a, b))
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        num_debug(Self::num_val(v), f)
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
    unsafe fn to_i64(&self, v: &IValue) -> Option<i64> {
        num_to_i64(Self::num_val(v))
    }
    unsafe fn to_u64(&self, v: &IValue) -> Option<u64> {
        num_to_u64(Self::num_val(v))
    }
    unsafe fn to_f64(&self, v: &IValue) -> Option<f64> {
        num_to_f64(Self::num_val(v))
    }
    unsafe fn to_f64_lossy(&self, v: &IValue) -> Option<f64> {
        Some(num_to_f64_lossy(Self::num_val(v)))
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        IValue::new_ptr(alloc(read(v.ptr())), TypeTag::NumberU64)
    }
    unsafe fn drop(&self, v: &mut IValue) {
        free(v.ptr());
    }
}
