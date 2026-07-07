//! The inline value family (tag `Inline`).
//!
//! A value with the `Inline` tag stores its whole contents in the pointer-sized
//! [`IValue`] rather than behind a pointer. Bit 3 selects the sub-family:
//!
//!   - 0 => number: a decimal `mantissa * 10^exp` (see [`number`]).
//!   - 1 => string or constant, distinguished by bit 7 ([`INLINE_CONST_FLAG`]):
//!       - bit 7 = 0 => string (see [`string`]).
//!       - bit 7 = 1 => constant: `null` / `false` / `true`.
//!
//! The all-zero value is never produced (the number exponent is biased so
//! integer zero is non-zero), reserving it as the `NonNull` niche.

pub(crate) mod number;
pub(crate) mod string;

use crate::value::{IValue, TypeTag};

/// Bit 3 of an inline value: set for the string/constant sub-family, clear for
/// inline numbers.
pub(crate) const INLINE_STR_FAMILY: usize = 1 << 3;
/// Bit 7 of an inline string-family value: set for a constant (`null`/`false`/
/// `true`), clear for an actual inline string.
pub(crate) const INLINE_CONST_FLAG: usize = 1 << 7;

// Bit patterns of the inline constants (the `Inline` tag is 0, so these equal
// the raw `ptr_usize()` of `NULL`/`FALSE`/`TRUE`). The constant is selected by
// bits 4-6 (0 = null, 1 = false, 2 = true). These are compared directly rather
// than materialising `IValue::NULL` etc., because those are droppable
// temporaries whose `Drop` would re-enter type classification and recurse.
pub(crate) const NULL_BITS: usize = INLINE_STR_FAMILY | INLINE_CONST_FLAG;
pub(crate) const FALSE_BITS: usize = NULL_BITS | (1 << 4);
pub(crate) const TRUE_BITS: usize = NULL_BITS | (2 << 4);

impl IValue {
    /// Returns `true` if this value is an inline number (tag `Inline`, number
    /// sub-family). See [`number`] for the decimal layout.
    pub(crate) fn is_inline_number(&self) -> bool {
        self.type_tag() == TypeTag::Inline && self.ptr_usize() & INLINE_STR_FAMILY == 0
    }

    /// Returns `true` if this value is a string stored inline rather than as a
    /// pointer to an interned heap allocation. Inline strings carry the `Inline`
    /// tag with the string sub-family bit set and the constant bit clear.
    pub(crate) fn is_inline_string(&self) -> bool {
        self.type_tag() == TypeTag::Inline
            && self.ptr_usize() & INLINE_STR_FAMILY != 0
            && self.ptr_usize() & INLINE_CONST_FLAG == 0
    }
}
