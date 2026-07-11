//! The inline constant representation: `null` and the two booleans.
//!
//! The three constants are the string/constant sub-family with the `CONST_FLAG` bit
//! set (see [`super`]); the [`Constant`] discriminant is packed into the payload
//! bits. A single [`ConstantRepr`] handles all three, decoding the discriminant from
//! the value's own bits. Each is a fixed bit pattern with no payload to free, so it
//! keeps the inline defaults for everything except comparison, debug formatting and
//! destructuring.

use std::cmp::Ordering;

use super::{CONST_FLAG, PAYLOAD_SHIFT, STR_FAMILY};
use crate::value::{BoolMut, Destructured, DestructuredMut, DestructuredRef, IValue, ValueType};

/// The three inline constants, numbered from zero. The discriminant is stored in the
/// payload bits and also gives the constants their order (`null` < `false` < `true`;
/// only the same-type comparisons — two `null`s or two `bool`s — are ever observed).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub(crate) enum Constant {
    Null = 0,
    False = 1,
    True = 2,
}

/// The inline representation of the `null`/`false`/`true` constants.
pub(crate) struct ConstantRepr;

impl ConstantRepr {
    /// The inline bits for a constant.
    const fn encode(c: Constant) -> usize {
        STR_FAMILY | CONST_FLAG | ((c as usize) << PAYLOAD_SHIFT)
    }

    /// The constant an inline constant value holds.
    fn decode(bits: usize) -> Constant {
        match (bits >> PAYLOAD_SHIFT) & 0b11 {
            0 => Constant::Null,
            1 => Constant::False,
            2 => Constant::True,
            _ => unreachable!("only the three constants are ever encoded"),
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
