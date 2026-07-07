//! The inline value family (tag `Inline`).
//!
//! A value with the `Inline` tag stores its whole contents in the pointer-sized
//! [`IValue`] rather than behind a pointer. Bit 3 selects the sub-family:
//!
//!   - 0 => number: a decimal `mantissa * 10^exp` (see [`number`]).
//!   - 1 => string or constant, distinguished by bit 7 (`CONST_FLAG`):
//!       - bit 7 = 0 => string (see [`string`]).
//!       - bit 7 = 1 => constant: `null` / `false` / `true`.
//!
//! The all-zero value is never produced (the number exponent is biased so
//! integer zero is non-zero), reserving it as the `NonNull` niche.

pub(crate) mod number;
pub(crate) mod string;

use std::hash::{Hash, Hasher};

use crate::value::ValueType;

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
pub(crate) fn is_number(bits: usize) -> bool {
    bits & STR_FAMILY == 0
}

/// `true` if the inline value is a string (string sub-family, not a constant).
pub(crate) fn is_string(bits: usize) -> bool {
    bits & STR_FAMILY != 0 && bits & CONST_FLAG == 0
}

/// Hashes an inline value. Numbers hash by numeric value (so `2` and `2.0`
/// agree); strings, `null` and the booleans have a canonical bit pattern and
/// hash by it.
pub(crate) fn hash<H: Hasher>(bits: usize, state: &mut H) {
    if is_number(bits) {
        number::hash(bits, state);
    } else {
        bits.hash(state);
    }
}
