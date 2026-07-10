//! The heap scalar-number representations: a bare 8-byte payload behind the
//! `NumberI64` / `NumberU64` / `NumberF64` tags.
//!
//! These are three separate representations — one per tag, in [`int`] (`I64Repr`),
//! [`uint`] (`U64Repr`), and [`float`] (`F64Repr`) — not one type that re-inspects
//! the tag: the tag alone determines how the eight bytes are read, so each owns its
//! own decode and construction. They share only the raw allocation helpers here,
//! which operate on the raw (aligned) allocation pointer; applying and stripping
//! the tag is the caller's (`IValue`'s) responsibility.

mod float;
mod int;
mod uint;

pub(crate) use float::F64Repr;
pub(crate) use int::I64Repr;
pub(crate) use uint::U64Repr;

use std::alloc::Layout;
use std::ptr::NonNull;

use super::{IValue, NumVal, TypeTag};
use crate::alloc::{alloc_infallible, dealloc_infallible};

fn layout() -> Layout {
    // An 8-byte payload, 8-aligned so the tag bits stay free.
    Layout::from_size_align(8, 8).unwrap()
}

/// Allocates a heap scalar holding `bits`, returning the aligned allocation.
fn alloc(bits: u64) -> NonNull<u8> {
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
unsafe fn free(ptr: NonNull<u8>) {
    dealloc_infallible(ptr, layout());
}

/// A heap scalar number of *any* kind reduced to a [`NumVal`], selecting the
/// representation by tag. Used for the cross-representation comparison that has to
/// resolve either operand; each representation decodes its own kind directly.
///
/// Safety: `v` must be a heap scalar number.
pub(crate) unsafe fn num_val(v: &IValue) -> NumVal {
    match v.type_tag() {
        TypeTag::NumberI64 => I64Repr::num_val(v),
        TypeTag::NumberU64 => U64Repr::num_val(v),
        // `NumberF64` and the reserved tag (never produced) both read as `f64`.
        _ => F64Repr::num_val(v),
    }
}
