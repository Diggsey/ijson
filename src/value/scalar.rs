//! The heap scalar-number representations: a bare 8-byte payload behind the
//! `NumberI64` / `NumberU64` / `NumberF64` tags. These are three separate
//! representations — one per tag — not one type that re-inspects the tag: the tag
//! alone determines how the eight bytes are read, so each owns its own decode and
//! construction. They share only the raw allocation helpers below.
//!
//! These operate on the raw (aligned) allocation pointer; applying and stripping
//! the tag is the caller's (`IValue`'s) responsibility.

use std::alloc::Layout;
use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;
use std::ptr::NonNull;

use super::{
    num_debug, num_hash, num_to_f64, num_to_f64_lossy, num_to_i64, num_to_u64, number_cmp,
    Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, TypeTag, ValueRepr, ValueType,
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

/// Reads the raw payload bits. Safety: `ptr` must be a live scalar allocation.
pub(crate) unsafe fn read(ptr: NonNull<u8>) -> u64 {
    ptr.cast::<u64>().as_ptr().read()
}

/// Frees a scalar allocation. Safety: `ptr` must be a live scalar allocation.
unsafe fn free(ptr: NonNull<u8>) {
    dealloc_infallible(ptr, layout());
}

/// Defines a heap scalar number representation: a zero-sized `ValueRepr` type that
/// stores its kind's value as the 8-byte payload behind `tag`. The value operations
/// are identical across the three kinds (they route through the shared numeric
/// utilities); only the payload `encode`/`decode` and whether it has a decimal
/// point differ, so those are the macro's parameters.
macro_rules! scalar_number {
    (
        $(#[$meta:meta])*
        $repr:ident : $val:ty,
        tag = $tag:ident,
        decimal_point = $dot:expr,
        encode = |$value:ident| $encode:expr,
        decode = |$bits:ident| $decode:expr,
    ) => {
        $(#[$meta])*
        pub(crate) struct $repr;

        impl $repr {
            /// Stores this value as a heap scalar. Always succeeds — the heap holds
            /// any value of this kind — so it is the total fallback in construction.
            pub(crate) fn store($value: $val) -> IValue {
                // Safety: `alloc` returns a fresh, aligned, non-null allocation.
                unsafe { IValue::new_ptr(alloc($encode), TypeTag::$tag) }
            }

            /// Decodes the 8-byte payload as this kind.
            /// Safety: `v` must be a live scalar of this kind.
            unsafe fn num_val(v: &IValue) -> NumVal {
                let $bits = read(v.ptr());
                $decode
            }
        }

        impl ValueRepr for $repr {
            fn value_type(&self, _v: &IValue) -> ValueType {
                ValueType::Number
            }
            fn has_decimal_point(&self, _v: &IValue) -> bool {
                $dot
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
                IValue::new_ptr(alloc(read(v.ptr())), TypeTag::$tag)
            }
            unsafe fn drop(&self, v: &mut IValue) {
                free(v.ptr());
            }
        }
    };
}

scalar_number! {
    /// The heap `i64` number representation.
    I64Repr: i64,
    tag = NumberI64,
    decimal_point = false,
    encode = |value| value as u64,
    decode = |bits| NumVal::Int(bits as i64),
}

scalar_number! {
    /// The heap `u64` number representation, reached only for values above
    /// `i64::MAX` (smaller ones canonicalise to the signed `i64` representation).
    U64Repr: u64,
    tag = NumberU64,
    decimal_point = false,
    encode = |value| value,
    decode = |bits| NumVal::UInt(bits),
}

scalar_number! {
    /// The heap `f64` number representation.
    F64Repr: f64,
    tag = NumberF64,
    decimal_point = true,
    encode = |value| value.to_bits(),
    decode = |bits| NumVal::Float(f64::from_bits(bits)),
}

/// A heap scalar number of *any* kind reduced to a [`NumVal`], selecting the
/// representation by tag. Used for the cross-representation comparison that has to
/// resolve either operand; each representation decodes its own kind directly.
///
/// Safety: `v` must be a heap scalar number.
pub(crate) unsafe fn num_val(v: &IValue) -> NumVal {
    match v.type_tag() {
        TypeTag::NumberI64 => I64Repr::num_val(v),
        TypeTag::NumberU64 => U64Repr::num_val(v),
        // `NumberF64` and the reserved tag (never produced) both read as `f64`.
        _ => F64Repr::num_val(v),
    }
}
