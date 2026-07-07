//! The heap scalar-number representation: a bare 8-byte payload behind the
//! `NumberI64` / `NumberU64` / `NumberF64` (and reserved) tags. The tag alone
//! determines how the eight bytes are interpreted, so no header is needed.
//!
//! These operate on the raw (aligned) allocation pointer; applying and stripping
//! the tag is the caller's (`IValue`'s) responsibility.

use std::alloc::Layout;
use std::ptr::NonNull;

use crate::alloc::{alloc_infallible, dealloc_infallible};

fn layout() -> Layout {
    // An 8-byte payload, 8-aligned so the tag bits stay free.
    Layout::from_size_align(8, 8).unwrap()
}

/// Allocates a heap scalar holding `bits`, returning the aligned allocation.
pub(crate) fn alloc(bits: u64) -> NonNull<u8> {
    // Safety: freshly allocated, 8-aligned, non-null.
    unsafe {
        let ptr = alloc_infallible(layout()).cast::<u64>();
        ptr.as_ptr().write(bits);
        ptr.cast()
    }
}

/// Reads the raw payload bits. Safety: `ptr` must be a live scalar allocation.
pub(crate) unsafe fn read(ptr: NonNull<u8>) -> u64 {
    ptr.cast::<u64>().as_ptr().read()
}

/// Frees a scalar allocation. Safety: `ptr` must be a live scalar allocation.
pub(crate) unsafe fn free(ptr: NonNull<u8>) {
    dealloc_infallible(ptr, layout());
}
