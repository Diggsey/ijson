//! The inline short-string representation.
//!
//! A string of at most [`CAPACITY`] bytes is packed into the value with the
//! `Inline` tag and the string sub-family flag. The low byte is the control
//! byte — the `Inline` tag (bits 0-2), the string sub-family flag (bit 3), and
//! the length (bits 4-6), with bit 7 clear (set only for the `null`/`false`/
//! `true` constants). The remaining bytes hold the UTF-8 data.
//!
//! These operate on the raw inline bits (and, for the borrowed bytes, a pointer
//! to the value's own storage); they do not know about `IValue`.

use std::ptr::NonNull;

use super::STR_FAMILY;

/// The number of string bytes that fit inline in a pointer-sized value:
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

/// The inline bits for `s` if it fits inline (at most [`CAPACITY`] bytes), or
/// `None` if it is too long and must be stored some other way. This is how the
/// value layer asks the inline representation whether it can hold a string.
pub(crate) fn try_encode(s: &str) -> Option<usize> {
    (s.len() <= CAPACITY).then(|| encode(s))
}

/// The inline bits for a string of at most [`CAPACITY`] bytes.
pub(crate) fn encode(s: &str) -> usize {
    debug_assert!(s.len() <= CAPACITY);

    // The control byte sets the string sub-family flag and the length (the
    // `Inline` tag bits are zero), and the remaining bytes carry the characters.
    let mut bytes = [0u8; std::mem::size_of::<usize>()];
    bytes[CONTROL_OFFSET] = STR_FAMILY as u8 | ((s.len() as u8) << LEN_SHIFT);
    bytes[CHAR_OFFSET..CHAR_OFFSET + s.len()].copy_from_slice(s.as_bytes());
    usize::from_ne_bytes(bytes)
}

/// The byte length of an inline string, read from the control byte.
pub(crate) fn len(bits: usize) -> usize {
    (bits >> LEN_SHIFT) & LEN_MASK
}

/// The UTF-8 bytes of an inline string, borrowed from `storage`.
///
/// Safety: `storage` must point at the value holding `bits` and remain valid
/// for `'a`.
pub(crate) unsafe fn bytes<'a>(storage: NonNull<u8>, bits: usize) -> &'a [u8] {
    std::slice::from_raw_parts(storage.as_ptr().add(CHAR_OFFSET), len(bits))
}
