//! The inline short-string representation.
//!
//! A string of at most [`CAPACITY`] bytes is packed into the value with the
//! `Inline` tag and the string sub-family flag. The low byte is the control
//! byte — the `Inline` tag (bits 0-2), the string sub-family flag (bit 3), and
//! the length (bits 4-6), with bit 7 clear (set only for the `null`/`false`/
//! `true` constants). The remaining bytes hold the UTF-8 data.

use super::INLINE_STR_FAMILY;
use crate::value::{IValue, TypeTag};

/// The number of string bytes that fit inline in a pointer-sized [`IValue`]:
/// 7 on 64-bit platforms and 3 on 32-bit (one byte is the control byte).
pub(crate) const CAPACITY: usize = std::mem::size_of::<usize>() - 1;

const LEN_SHIFT: u32 = 4;
const LEN_MASK: usize = 0b111;

// Memory offsets of the control byte and the first character byte. The control
// byte must be the low byte of the integer value (so the tag and flags land in
// the low bits): offset 0 on little-endian, the top byte on big-endian. The
// characters follow in ascending memory order.
#[cfg(target_endian = "little")]
const CONTROL_OFFSET: usize = 0;
#[cfg(target_endian = "little")]
const CHAR_OFFSET: usize = 1;
#[cfg(target_endian = "big")]
const CONTROL_OFFSET: usize = std::mem::size_of::<usize>() - 1;
#[cfg(target_endian = "big")]
const CHAR_OFFSET: usize = 0;

/// Encodes a string of at most [`CAPACITY`] bytes directly into a value.
pub(crate) fn encode(s: &str) -> IValue {
    debug_assert!(s.len() <= CAPACITY);

    // Build the payload with the tag bits left clear; `IValue::new_inline` ORs
    // in the `Inline` tag (0). The control byte sets the string sub-family flag
    // and the length, and the remaining bytes carry the characters.
    let mut bytes = [0u8; std::mem::size_of::<usize>()];
    bytes[CONTROL_OFFSET] = INLINE_STR_FAMILY as u8 | ((s.len() as u8) << LEN_SHIFT);
    bytes[CHAR_OFFSET..CHAR_OFFSET + s.len()].copy_from_slice(s.as_bytes());

    // Safety: the string sub-family flag keeps the value non-zero, and the
    // payload leaves the tag bits clear.
    unsafe { IValue::new_inline(TypeTag::Inline, usize::from_ne_bytes(bytes)) }
}

impl IValue {
    /// The byte length of an inline string, read from the control byte.
    ///
    /// Only meaningful when this is an inline string.
    pub(crate) fn inline_string_len(&self) -> usize {
        (self.ptr_usize() >> LEN_SHIFT) & LEN_MASK
    }

    /// The UTF-8 bytes of an inline string, borrowed from within `self`.
    ///
    /// Only valid when this is an inline string.
    pub(crate) fn inline_string_bytes(&self) -> &[u8] {
        let len = self.inline_string_len();
        // Safety: an inline string keeps its characters within its own storage
        // at `CHAR_OFFSET`, and `len <= CAPACITY` fits within the value.
        unsafe {
            let base = (self as *const IValue).cast::<u8>();
            std::slice::from_raw_parts(base.add(CHAR_OFFSET), len)
        }
    }
}
