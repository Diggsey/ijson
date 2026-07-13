//! The inline constant representation: `null` and the two booleans.
//!
//! The three constants are the inline family with both the number bit (`IS_NUMBER`)
//! and the string bit (`IS_STRING`) clear (see [`super`]); the [`Constant`]
//! discriminant is packed into the payload bits. Discriminants are numbered from 1 so
//! that no constant is the all-zero niche word. A single [`ConstantRepr`] handles all
//! three, decoding the discriminant from the value's own bits. Each is a fixed bit
//! pattern with no payload to free, so it keeps the inline defaults for everything
//! except comparison, debug formatting and destructuring.

use std::cmp::Ordering;

use super::{PAYLOAD_SHIFT, TAG_MASK};
use crate::value::{BoolMut, Destructured, DestructuredMut, DestructuredRef, IValue, ValueType};

/// The three inline constants, numbered from one (zero is the reserved niche word).
/// The discriminant is stored in the payload bits and also gives the constants their
/// order (`null` < `false` < `true`; only the same-type comparisons — two `null`s or
/// two `bool`s — are ever observed).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub(crate) enum Constant {
    Null = 1,
    False = 2,
    True = 3,
}

/// The inline representation of the `null`/`false`/`true` constants.
pub(crate) struct ConstantRepr;

impl ConstantRepr {
    /// The inline bits for a constant: the number and string bits clear (so `kind()`
    /// classifies it as a constant), with the discriminant in the payload bits.
    const fn encode(c: Constant) -> usize {
        let bits = (c as usize) << PAYLOAD_SHIFT;
        debug_assert!(
            bits & TAG_MASK == 0,
            "inline constant must leave the tag bits clear"
        );
        bits
    }

    /// The constant an inline constant value holds.
    ///
    /// Total: the discriminant is one of exactly three values (zero is reserved, which is
    /// what keeps the all-zero word free as the `NonNull` niche), so the last needs no
    /// test. Written as a `match` with an `unreachable!()` arm instead, the impossible
    /// zero would put a panic — and the `Arguments` it formats — into *every* operation
    /// that reaches an inline constant, [`IValue::type_`] among them. `decode` is on that
    /// path, so it has to generate nothing.
    fn decode(bits: usize) -> Constant {
        let discriminant = (bits >> PAYLOAD_SHIFT) & 0b11;
        debug_assert!(
            (1..=3).contains(&discriminant),
            "only the three constants are ever encoded"
        );
        if discriminant == Constant::Null as usize {
            Constant::Null
        } else if discriminant == Constant::False as usize {
            Constant::False
        } else {
            Constant::True
        }
    }
}

// The whole inline value for each constant (the `Inline` tag is 0), re-exported by
// `super` so `IValue` can build and recognise them.
pub(crate) const NULL: usize = ConstantRepr::encode(Constant::Null);
pub(crate) const FALSE: usize = ConstantRepr::encode(Constant::False);
pub(crate) const TRUE: usize = ConstantRepr::encode(Constant::True);

impl super::InlineValue for ConstantRepr {
    fn value_type(&self, v: &IValue) -> ValueType {
        // A constant is `null`, or a `bool` for either boolean.
        match Self::decode(v.usize_()) {
            Constant::Null => ValueType::Null,
            Constant::False | Constant::True => ValueType::Bool,
        }
    }
    // clone/drop/hash/eq use the inline defaults (bit-copy / nothing / pointer word /
    // `raw_eq`), all correct for the fixed constant bit patterns.
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        // The caller guarantees `a` and `b` share a type, so this only orders two
        // `null`s (equal) or two `bool`s; the discriminant order gives both.
        Some(Self::decode(a.usize_()).cmp(&Self::decode(b.usize_())))
    }
    unsafe fn debug(&self, v: &IValue, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match Self::decode(v.usize_()) {
            Constant::Null => "null",
            Constant::False => "false",
            Constant::True => "true",
        })
    }
    fn destructure(&self, v: IValue) -> Destructured {
        match Self::decode(v.usize_()) {
            Constant::Null => Destructured::Null,
            Constant::False => Destructured::Bool(false),
            Constant::True => Destructured::Bool(true),
        }
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        match Self::decode(v.usize_()) {
            Constant::Null => DestructuredRef::Null,
            Constant::False => DestructuredRef::Bool(false),
            Constant::True => DestructuredRef::Bool(true),
        }
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        match Self::decode(v.usize_()) {
            Constant::Null => DestructuredMut::Null,
            Constant::False | Constant::True => DestructuredMut::Bool(BoolMut(v)),
        }
    }
}
