//! The heap `u64` number representation (tag `NumberU64`): the eight payload bytes
//! are an unsigned 64-bit integer. Reached only for values above `i64::MAX` —
//! smaller ones canonicalise to the signed [`super::int`] representation.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::{alloc, free, read};
use crate::number::INumber;
use crate::value::{
    number_cmp, Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, ReprTag, ValueRepr,
    ValueType,
};

/// The heap `u64` number representation.
pub(crate) struct U64Repr;

impl U64Repr {
    /// Stores a `u64` as a heap scalar. Always succeeds — the heap holds any
    /// `u64` — so it is the total fallback in construction.
    pub(crate) fn store(value: u64) -> IValue {
        // Safety: `alloc` returns a fresh, aligned, non-null allocation.
        unsafe { IValue::new_ptr(ReprTag::NumberU64, alloc::<u64>(value)) }
    }

    /// Decodes the payload as a `NumVal`. Safety: `v` must be a live `NumberU64`.
    unsafe fn num_val(v: &IValue) -> NumVal {
        NumVal::from_u64(read::<u64>(v.ptr()))
    }
}

impl ValueRepr for U64Repr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        Self::num_val(v).hash(state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(Self::num_val(a), b) == Some(Ordering::Equal)
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        number_cmp(Self::num_val(a), b)
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", Self::num_val(v))
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
        Self::store(read::<u64>(v.ptr()))
    }
    unsafe fn drop(&self, v: &mut IValue) {
        free::<u64>(v.ptr());
    }
    unsafe fn num_val(&self, v: &IValue) -> Option<NumVal> {
        Some(Self::num_val(v))
    }
}
