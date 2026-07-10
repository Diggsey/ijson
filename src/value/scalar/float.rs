//! The heap `f64` number representation (tag `NumberF64`): the eight payload bytes
//! are the `f64` bit pattern. This is the only heap scalar with a decimal point.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::{alloc, free, read};
use crate::number::INumber;
use crate::value::{
    num_debug, num_hash, num_to_f64, num_to_f64_lossy, num_to_i64, num_to_u64, number_cmp,
    Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, TypeTag, ValueRepr, ValueType,
};

/// The heap `f64` number representation.
pub(crate) struct F64Repr;

impl F64Repr {
    /// Stores an `f64` as a heap scalar. Always succeeds — the heap holds any
    /// `f64` — so it is the total fallback in construction.
    pub(crate) fn store(value: f64) -> IValue {
        // Safety: `alloc` returns a fresh, aligned, non-null allocation.
        unsafe { IValue::new_ptr(alloc::<f64>(value), TypeTag::NumberF64) }
    }
}

impl ValueRepr for F64Repr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    fn has_decimal_point(&self, _v: &IValue) -> bool {
        true
    }
    /// Decodes the payload as an `f64`. Safety: `v` must be a live `NumberF64`.
    unsafe fn num_val(&self, v: &IValue) -> NumVal {
        NumVal::Float(read::<f64>(v.ptr()))
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        num_hash(self.num_val(v), state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(self.num_val(a), b) == Ordering::Equal
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        Some(number_cmp(self.num_val(a), b))
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        num_debug(self.num_val(v), f)
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
        num_to_i64(self.num_val(v))
    }
    unsafe fn to_u64(&self, v: &IValue) -> Option<u64> {
        num_to_u64(self.num_val(v))
    }
    unsafe fn to_f64(&self, v: &IValue) -> Option<f64> {
        num_to_f64(self.num_val(v))
    }
    unsafe fn to_f64_lossy(&self, v: &IValue) -> Option<f64> {
        Some(num_to_f64_lossy(self.num_val(v)))
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        Self::store(read::<f64>(v.ptr()))
    }
    unsafe fn drop(&self, v: &mut IValue) {
        free::<f64>(v.ptr());
    }
}
