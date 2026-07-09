//! The inline value family (tag `Inline`).
//!
//! A value with the `Inline` tag stores its whole contents in the pointer-sized
//! [`IValue`] rather than behind a pointer. Bit 3 selects the sub-family:
//!
//!   - 0 => number: `mantissa * BASE^exp`, base 10 or 2 (see [`number`]).
//!   - 1 => string or constant, distinguished by bit 7 (`CONST_FLAG`):
//!       - bit 7 = 0 => string (see [`string`]).
//!       - bit 7 = 1 => constant: `null` / `false` / `true`.
//!
//! The all-zero value is never produced (the number exponent is biased so
//! integer zero is non-zero), reserving it as the `NonNull` niche.

pub(crate) mod constant;
pub(crate) mod number_binary;
pub(crate) mod number_decimal;
pub(crate) mod string;

// The two inline number representations — an exact base-10 decimal and a base-2
// binary float — are fully independent modules (sharing no code, so their bit
// layouts can diverge). Both are always compiled, so a single `cargo test`
// unit-tests both regardless of features; this alias just selects which one
// `IValue` construction and decoding actually use.
#[cfg(not(feature = "arbitrary_precision"))]
pub(crate) use number_binary as number;
#[cfg(feature = "arbitrary_precision")]
pub(crate) use number_decimal as number;

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use crate::value::{Destructured, DestructuredMut, DestructuredRef, IValue, ValueRepr, ValueType};

// Bit 3 of an inline value: set for the string/constant sub-family, clear for
// inline numbers.
const STR_FAMILY: usize = 1 << 3;
// Bit 7 of an inline string-family value: set for a constant (`null`/`false`/
// `true`), clear for an actual inline string.
const CONST_FLAG: usize = 1 << 7;

// Bit patterns of the inline constants (the `Inline` tag is 0, so these are the
// whole inline value). The constant is selected by bits 4-6 (0 = null,
// 1 = false, 2 = true).
pub(crate) const NULL: usize = STR_FAMILY | CONST_FLAG;
pub(crate) const FALSE: usize = NULL | (1 << 4);
pub(crate) const TRUE: usize = NULL | (2 << 4);

/// The JSON type of an inline value, from its raw bits.
pub(crate) fn value_type(bits: usize) -> ValueType {
    if is_number(bits) {
        ValueType::Number
    } else if is_string(bits) {
        ValueType::String
    } else if bits == NULL {
        ValueType::Null
    } else {
        ValueType::Bool
    }
}

/// `true` if the inline value is a number (number sub-family).
fn is_number(bits: usize) -> bool {
    bits & STR_FAMILY == 0
}

/// `true` if the inline value is a string (string sub-family, not a constant).
fn is_string(bits: usize) -> bool {
    bits & STR_FAMILY != 0 && bits & CONST_FLAG == 0
}

/// The per-type behaviour of an inline value. A single [`InlineRepr`] implements
/// [`ValueRepr`] for the whole inline family and decodes the family bits to pick
/// the right one of these; each sub-representation only overrides what it needs.
///
/// This mirrors the value operations of [`ValueRepr`] but omits `clone`/`drop`
/// (every inline value is a bit-copy with nothing to free) and `len` (an inline
/// value is never a collection); [`InlineRepr`] supplies those uniformly.
pub(crate) trait InlineValue {
    /// The JSON type this inline sub-representation stores.
    fn value_type(&self) -> ValueType;
    /// Hash by value. Default: the canonical pointer word (correct for the
    /// constants and inline strings). Inline numbers override to hash by value.
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        state.write_usize(v.ptr_usize());
    }
    /// Equality within a type. Default: the canonical bits. Numbers override.
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        a.raw_eq(b)
    }
    /// Ordering within a type. Default: unordered; every inline type overrides.
    unsafe fn partial_cmp(&self, _a: &IValue, _b: &IValue) -> Option<Ordering> {
        None
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result;
    fn destructure(&self, v: IValue) -> Destructured;
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a>;
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a>;
    fn to_bool(&self, _v: &IValue) -> Option<bool> {
        None
    }
    unsafe fn to_i64(&self, _v: &IValue) -> Option<i64> {
        None
    }
    unsafe fn to_u64(&self, _v: &IValue) -> Option<u64> {
        None
    }
    unsafe fn to_f64(&self, _v: &IValue) -> Option<f64> {
        None
    }
    unsafe fn to_f64_lossy(&self, _v: &IValue) -> Option<f64> {
        None
    }
    unsafe fn as_bytes<'a>(&self, _v: &'a IValue) -> Option<&'a [u8]> {
        None
    }
    fn has_decimal_point(&self, _v: &IValue) -> bool {
        false
    }
}

/// The single representation for the whole inline family. Every operation decodes
/// the family bits, selects the inline sub-representation, and delegates to it.
pub(crate) struct InlineRepr;

impl InlineRepr {
    /// Selects the inline sub-representation for `v` from its family bits.
    #[inline]
    fn inner(v: &IValue) -> &'static dyn InlineValue {
        match value_type(v.ptr_usize()) {
            ValueType::Null => &constant::NullRepr,
            ValueType::Bool => &constant::BoolRepr,
            ValueType::Number => &number::InlineNumberRepr,
            ValueType::String => &string::InlineStringRepr,
            // An inline value is never an array or object.
            ValueType::Array | ValueType::Object => unreachable!(),
        }
    }
}

impl ValueRepr for InlineRepr {
    // clone/drop/len use the `ValueRepr` defaults: every inline value is a
    // bit-copy to clone, has nothing to free, and is never a collection.
    fn value_type(&self, v: &IValue) -> ValueType {
        Self::inner(v).value_type()
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        Self::inner(v).hash(v, state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        Self::inner(a).eq(a, b)
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        Self::inner(a).partial_cmp(a, b)
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        Self::inner(v).debug(v, f)
    }
    fn destructure(&self, v: IValue) -> Destructured {
        Self::inner(&v).destructure(v)
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        Self::inner(v).destructure_ref(v)
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        let inner = Self::inner(v);
        inner.destructure_mut(v)
    }
    fn to_bool(&self, v: &IValue) -> Option<bool> {
        Self::inner(v).to_bool(v)
    }
    unsafe fn to_i64(&self, v: &IValue) -> Option<i64> {
        Self::inner(v).to_i64(v)
    }
    unsafe fn to_u64(&self, v: &IValue) -> Option<u64> {
        Self::inner(v).to_u64(v)
    }
    unsafe fn to_f64(&self, v: &IValue) -> Option<f64> {
        Self::inner(v).to_f64(v)
    }
    unsafe fn to_f64_lossy(&self, v: &IValue) -> Option<f64> {
        Self::inner(v).to_f64_lossy(v)
    }
    unsafe fn as_bytes<'a>(&self, v: &'a IValue) -> Option<&'a [u8]> {
        Self::inner(v).as_bytes(v)
    }
    fn has_decimal_point(&self, v: &IValue) -> bool {
        Self::inner(v).has_decimal_point(v)
    }
}
