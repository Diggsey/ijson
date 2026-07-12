//! The inline value family (tag `Inline`).
//!
//! A value with the `Inline` tag stores its whole contents in the pointer-sized
//! [`IValue`] rather than behind a pointer. The bits just above the tag pick the
//! sub-family (see [`InlineKind`] and [`InlineRepr::kind`]):
//!
//!   - bit 3 set   => number: `mantissa * BASE^exp`, base 10 or 2 (see
//!     [`number_decimal`]/[`number_binary`], one selected as [`InlineNumberRepr`]).
//!   - bit 3 clear => string or constant, distinguished by bit 4 (`IS_STRING`,
//!     adjacent to the number bit):
//!       - bit 4 set   => string (see [`string`]); bits 5-7 hold its length.
//!       - bit 4 clear => constant: `null` / `false` / `true` (see [`constant`]);
//!         bits 5-7 hold the constant's discriminant (numbered from 1).
//!
//! The all-zero word is never produced — a number sets bit 3, a string sets bit 4,
//! and constant discriminants start at 1 — reserving it as the `NonNull` niche. Each
//! sub-family is thus non-null on its own, so no representation has to shape its
//! encoding to dodge the niche.

pub(crate) mod constant;
pub(crate) mod number;
pub(crate) mod number_binary;
pub(crate) mod number_decimal;
pub(crate) mod string;

pub(crate) use constant::{FALSE, NULL, TRUE};
pub(crate) use number::{InlineNumber, InlineNumberError, NumberShape};

// The two inline number representations — an exact base-10 decimal and a base-2
// binary float — are fully independent modules (sharing no code, so their bit
// layouts can diverge). Both are always compiled, so a single `cargo test`
// unit-tests both regardless of features; this alias selects the active
// representation *type*, whose `InlineNumber` associated functions
// (`encode_int`/`encode_f64`/`from_str`) are how `IValue` builds inline numbers, and
// whose `NumberRepr` impl decodes them.
#[cfg(not(feature = "arbitrary_precision"))]
pub(crate) use number_binary::BinaryNumberRepr as InlineNumberRepr;
#[cfg(feature = "arbitrary_precision")]
pub(crate) use number_decimal::DecimalNumberRepr as InlineNumberRepr;

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use crate::value::{Destructured, DestructuredMut, DestructuredRef, IValue, ValueRepr, ValueType};

// Bit 3: set for an inline number, clear for a string or constant. A single positive
// predicate ("is a number") — and, because every number sets it, a number word is
// never all-zero, so the number codec needs no niche-avoidance of its own.
const IS_NUMBER: usize = 1 << 3;
// Bit 4 (only meaningful when `IS_NUMBER` is clear): set for an inline string, clear
// for a constant (`null`/`false`/`true`). A string always sets it, so even the empty
// inline string is non-zero.
const IS_STRING: usize = 1 << 4;
// Bits 5-7 carry the payload of a string or constant: an inline string's length or a
// constant's discriminant (see [`constant::Constant`]).
const PAYLOAD_SHIFT: u32 = 5;

// The `Inline` tag occupies the low 3 bits and is all-zero, so an encoded inline
// payload must leave them clear for the tag to survive `IValue::new_usize`. Every
// `encode` helper debug-asserts its result against this.
const TAG_MASK: usize = super::ALIGNMENT - 1;

/// Which of the three inline sub-families a value belongs to — exactly the
/// distinction the inline bits encode, unlike the six-way [`ValueType`]. (The null
/// vs. bool split within `Constant` is a further decode, done by [`constant`].)
enum InlineKind {
    Number,
    String,
    Constant,
}

/// The *universal* behaviour of an inline value — the [`ValueRepr`] operations that
/// [`InlineRepr`] delegates to a sub-representation, minus the `clone`/`drop` it
/// supplies uniformly (every inline value is a bit-copy with nothing to free). A
/// single [`InlineRepr`] implements [`ValueRepr`] for the whole inline family and
/// decodes the family bits to pick the right one of these; each sub-representation
/// only overrides what it needs.
///
/// The *type-specific* accessors are not here: the inline number and string
/// sub-representations implement the per-type traits (`NumberRepr` / `StringRepr`)
/// directly, reached through `IValue::number_repr` / `string_repr`, and inline
/// `bool`/`null` carry no accessor at all.
pub(crate) trait InlineValue {
    /// The JSON type this inline value stores. Takes `v` because one `ConstantRepr`
    /// serves both `null` and `bool`, decoding which from the bits.
    fn value_type(&self, v: &IValue) -> ValueType;
    /// Hash by value. Default: the canonical pointer word (correct for the
    /// constants and inline strings). Inline numbers override to hash by value.
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        state.write_usize(v.usize_());
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
}

/// The single representation for the whole inline family. Every operation decodes
/// the family bits, selects the inline sub-representation, and delegates to it.
pub(crate) struct InlineRepr;

impl InlineRepr {
    /// Which inline sub-family `v` belongs to.
    fn kind(v: &IValue) -> InlineKind {
        let bits = v.usize_();
        if bits & IS_NUMBER != 0 {
            InlineKind::Number
        } else if bits & IS_STRING != 0 {
            InlineKind::String
        } else {
            InlineKind::Constant
        }
    }

    /// Selects the inline sub-representation for `v` from its family bits.
    #[inline]
    fn inner(v: &IValue) -> &'static dyn InlineValue {
        match Self::kind(v) {
            InlineKind::Number => &InlineNumberRepr,
            InlineKind::String => &string::InlineStringRepr,
            InlineKind::Constant => &constant::ConstantRepr,
        }
    }
}

impl ValueRepr for InlineRepr {
    fn value_type(&self, v: &IValue) -> ValueType {
        Self::inner(v).value_type(v)
    }
    // Every inline value is stored entirely in the pointer word: cloning is a
    // bit-copy of that word, and there is no heap storage to release on drop.
    unsafe fn clone(&self, v: &IValue) -> IValue {
        v.raw_copy()
    }
    unsafe fn drop(&self, _v: &mut IValue) {}
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
        Self::inner(v).destructure_mut(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_classify_and_avoid_the_niche() {
        // Build each inline sub-family through its real constructors and check that
        // `kind()` classifies it and that no encoding is the all-zero niche word.
        // Integer zero is the case the old layout had to bias away from zero; here it
        // is non-zero structurally, because every number sets `IS_NUMBER`.
        let numbers = [
            IValue::new_i64(0),
            IValue::new_i64(1),
            IValue::new_i64(-1),
            IValue::new_f64(0.5).unwrap(),
            IValue::new_f64(0.0).unwrap(),
        ];
        let strings = [IValue::new_string(""), IValue::new_string("inline")];
        let constants = [IValue::NULL, IValue::TRUE, IValue::FALSE];

        for v in &numbers {
            assert!(matches!(InlineRepr::kind(v), InlineKind::Number), "{:?}", v);
        }
        for v in &strings {
            assert!(matches!(InlineRepr::kind(v), InlineKind::String), "{:?}", v);
        }
        for v in &constants {
            assert!(
                matches!(InlineRepr::kind(v), InlineKind::Constant),
                "{:?}",
                v
            );
        }
        for v in numbers.iter().chain(&strings).chain(&constants) {
            assert_ne!(v.usize_(), 0, "{:?} is the reserved all-zero niche", v);
        }
    }
}
