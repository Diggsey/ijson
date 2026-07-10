//! The heap scalar-number representation: a bare 8-byte payload behind the
//! `NumberI64` / `NumberU64` / `NumberF64` (and reserved) tags. The tag alone
//! determines how the eight bytes are interpreted, so no header is needed.
//!
//! These operate on the raw (aligned) allocation pointer; applying and stripping
//! the tag is the caller's (`IValue`'s) responsibility.

use std::alloc::Layout;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::{self, Formatter};
use std::hash::Hasher;
use std::ptr::NonNull;

use super::{
    num_debug, num_hash, num_to_f64, num_to_f64_lossy, num_to_i64, num_to_u64, number_cmp,
    Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, NumberRepr, TypeTag, ValueRepr,
    ValueType,
};
use crate::alloc::{alloc_infallible, dealloc_infallible};
use crate::number::INumber;

fn layout() -> Layout {
    // An 8-byte payload, 8-aligned so the tag bits stay free.
    Layout::from_size_align(8, 8).unwrap()
}

/// Allocates a heap scalar holding `bits`, returning the aligned allocation.
fn alloc(bits: u64) -> NonNull<u8> {
    // Safety: freshly allocated, 8-aligned, non-null.
    unsafe {
        let ptr = alloc_infallible(layout()).cast::<u64>();
        ptr.as_ptr().write(bits);
        ptr.cast()
    }
}

/// The heap scalar is the total number representation: it stores any `i64`, `u64`,
/// or `f64` (each tag records how to read the eight bytes back), so it never
/// declines a value and is the final fallback in the construction chain.
impl NumberRepr for ScalarRepr {
    fn from_i64(value: i64) -> Option<IValue> {
        // Safety: `alloc` returns a fresh, aligned, non-null scalar allocation.
        Some(unsafe { IValue::new_ptr(alloc(value as u64), TypeTag::NumberI64) })
    }
    fn from_u64(value: u64) -> Option<IValue> {
        match i64::try_from(value) {
            // Prefer the signed tag when it fits, so the representation is canonical.
            Ok(v) => Self::from_i64(v),
            // Safety: `alloc` returns a fresh, aligned, non-null scalar allocation.
            Err(_) => Some(unsafe { IValue::new_ptr(alloc(value), TypeTag::NumberU64) }),
        }
    }
    fn from_f64(value: f64) -> Option<IValue> {
        // Safety: `alloc` returns a fresh, aligned, non-null scalar allocation.
        Some(unsafe { IValue::new_ptr(alloc(value.to_bits()), TypeTag::NumberF64) })
    }
}

/// Reads the raw payload bits. Safety: `ptr` must be a live scalar allocation.
pub(crate) unsafe fn read(ptr: NonNull<u8>) -> u64 {
    ptr.cast::<u64>().as_ptr().read()
}

/// Frees a scalar allocation. Safety: `ptr` must be a live scalar allocation.
pub(crate) unsafe fn free(ptr: NonNull<u8>) {
    dealloc_infallible(ptr, layout());
}

/// This heap scalar number reduced to a [`NumVal`] for the shared numeric
/// utilities. The tag alone determines how the eight bytes are interpreted.
///
/// Safety: `v` must be a heap scalar number.
pub(crate) unsafe fn num_val(v: &IValue) -> NumVal {
    match v.type_tag() {
        TypeTag::NumberI64 => NumVal::Int(read(v.ptr()) as i64),
        TypeTag::NumberU64 => NumVal::UInt(read(v.ptr())),
        _ => NumVal::Float(f64::from_bits(read(v.ptr()))),
    }
}

/// The heap scalar-number representation of a JSON number.
pub(crate) struct ScalarRepr;
impl ValueRepr for ScalarRepr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    fn has_decimal_point(&self, v: &IValue) -> bool {
        v.type_tag() == TypeTag::NumberF64
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        num_hash(num_val(v), state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(a, b) == Ordering::Equal
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        Some(number_cmp(a, b))
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        num_debug(num_val(v), f)
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
        num_to_i64(num_val(v))
    }
    unsafe fn to_u64(&self, v: &IValue) -> Option<u64> {
        num_to_u64(num_val(v))
    }
    unsafe fn to_f64(&self, v: &IValue) -> Option<f64> {
        num_to_f64(num_val(v))
    }
    unsafe fn to_f64_lossy(&self, v: &IValue) -> Option<f64> {
        Some(num_to_f64_lossy(num_val(v)))
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        IValue::new_ptr(alloc(read(v.ptr())), v.type_tag())
    }
    unsafe fn drop(&self, v: &mut IValue) {
        free(v.ptr());
    }
}
