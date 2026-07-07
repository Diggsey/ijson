//! String-type logic spanning the inline and interned representations.
//!
//! A JSON string is stored either as an inline short string (see
//! [`super::inline::string`]) or as a heap interned string (see
//! [`super::interned`]). This module holds the logic that spans those
//! representations — construction, byte/str access, comparison and formatting —
//! as free functions on `&IValue`. `IValue`'s own trait impls delegate here, and
//! the public [`crate::IString`] wrapper does too.

use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::ptr::NonNull;

use super::inline::string as inl;
use super::interned;
use super::{IValue, TypeTag};

pub(crate) fn init_cache() {
    interned::init_cache();
}

pub(crate) fn new(s: &str) -> IValue {
    if s.len() <= inl::CAPACITY {
        // Safety: `encode` returns valid inline-string bits.
        unsafe { IValue::new_inline(TypeTag::Inline, inl::encode(s)) }
    } else {
        // Safety: `intern` returns a live, aligned interned header pointer.
        unsafe { IValue::new_ptr(interned::intern(s), TypeTag::String) }
    }
}

pub(crate) fn len(v: &IValue) -> usize {
    if v.is_inline() {
        inl::len(v.ptr_usize())
    } else {
        // Safety: not an inline string, so it is interned.
        unsafe { interned::len(v.ptr()) }
    }
}

pub(crate) fn bytes(v: &IValue) -> &[u8] {
    if v.is_inline() {
        // Safety: an inline string keeps its bytes within `v`'s storage.
        unsafe { inl::bytes(NonNull::from(v).cast(), v.ptr_usize()) }
    } else {
        // Safety: not an inline string, so it is interned.
        unsafe { interned::bytes(v.ptr()) }
    }
}

pub(crate) fn as_str(v: &IValue) -> &str {
    // Safety: inline and interned string bytes are both valid UTF-8.
    unsafe { std::str::from_utf8_unchecked(bytes(v)) }
}

pub(crate) fn cmp(a: &IValue, b: &IValue) -> Ordering {
    if a.raw_eq(b) {
        Ordering::Equal
    } else {
        as_str(a).cmp(as_str(b))
    }
}

pub(crate) fn debug(v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
    Debug::fmt(as_str(v), f)
}
