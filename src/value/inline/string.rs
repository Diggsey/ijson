//! The inline short-string representation.
//!
//! A string of at most [`CAPACITY`] bytes is packed into the value with the
//! `Inline` tag and the string flag. The low byte is the control byte — the
//! `Inline` tag (bits 0-2), the number flag clear (bit 3), the string flag set
//! (bit 4), and the length (bits 5-7). The remaining bytes hold the UTF-8 data.
//!
//! The encode/decode helpers are associated functions of [`InlineStringRepr`] that
//! operate on the raw inline bits (and, for the borrowed bytes, a pointer to the
//! value's own storage); only its trait impls at the bottom know about `IValue`.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::ptr::NonNull;

use super::{InlineValue, IS_STRING, TAG_MASK};
use crate::string::IString;
use crate::value::{
    string_cmp, string_debug, Destructured, DestructuredMut, DestructuredRef, IValue, ValueType,
};

/// The number of string bytes that fit inline in a pointer-sized value:
/// 7 on 64-bit platforms and 3 on 32-bit (one byte is the control byte).
pub(crate) const CAPACITY: usize = std::mem::size_of::<usize>() - 1;

// The length occupies the shared payload bits (5-7) of the control byte.
const LEN_SHIFT: u32 = super::PAYLOAD_SHIFT;
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

/// The inline short-string representation of a JSON string.
pub(crate) struct InlineStringRepr;

impl InlineStringRepr {
    /// The inline bits for `s` if it fits inline (at most [`CAPACITY`] bytes), or
    /// `None` if it is too long and must be stored some other way. This is how the
    /// value layer asks the inline representation whether it can hold a string.
    pub(crate) fn try_encode(s: &str) -> Option<usize> {
        (s.len() <= CAPACITY).then(|| Self::encode(s))
    }

    /// The inline bits for a string of at most [`CAPACITY`] bytes.
    fn encode(s: &str) -> usize {
        debug_assert!(s.len() <= CAPACITY);

        // The control byte sets the string flag and the length (the number flag and
        // the `Inline` tag bits are zero); the remaining bytes carry the characters.
        let mut bytes = [0u8; std::mem::size_of::<usize>()];
        bytes[CONTROL_OFFSET] = IS_STRING as u8 | ((s.len() as u8) << LEN_SHIFT);
        bytes[CHAR_OFFSET..CHAR_OFFSET + s.len()].copy_from_slice(s.as_bytes());
        let bits = usize::from_ne_bytes(bytes);
        debug_assert_eq!(
            bits & TAG_MASK,
            0,
            "inline string must leave the tag bits clear"
        );
        bits
    }

    /// The byte length of an inline string, read from the control byte.
    fn len(bits: usize) -> usize {
        (bits >> LEN_SHIFT) & LEN_MASK
    }

    /// The UTF-8 bytes of an inline string, borrowed from `storage`.
    ///
    /// Safety: `storage` must point at the value holding `bits` and remain valid
    /// for `'a`.
    unsafe fn bytes<'a>(storage: NonNull<u8>, bits: usize) -> &'a [u8] {
        std::slice::from_raw_parts(storage.as_ptr().add(CHAR_OFFSET), Self::len(bits))
    }
}

impl InlineValue for InlineStringRepr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::String
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        Some(string_cmp(a, b))
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        string_debug(v, f)
    }
    fn destructure(&self, v: IValue) -> Destructured {
        Destructured::String(IString(v))
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        DestructuredRef::String(v.as_string_unchecked())
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        DestructuredMut::String(v.as_string_unchecked_mut())
    }
    // clone/drop/hash/eq use the defaults (bit-copy / nothing / pointer word /
    // `raw_eq`), all correct for an inline string.
    /// The inline UTF-8 bytes, borrowed from `v`'s own storage. Safety: `v` must be
    /// an inline string.
    unsafe fn as_bytes<'a>(&self, v: &'a IValue) -> Option<&'a [u8]> {
        Some(Self::bytes(NonNull::from(v).cast(), v.usize_()))
    }
}
