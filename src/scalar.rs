//! The heap scalar-number representation: a bare 8-byte payload behind the
//! `NumberI64` / `NumberU64` / `NumberF64` (and reserved) tags. The tag alone
//! determines how the eight bytes are interpreted, so no header is needed.

use std::alloc::Layout;

use crate::alloc::{alloc_infallible, dealloc_infallible};
use crate::value::{IValue, TypeTag};

fn layout() -> Layout {
    // An 8-byte payload, 8-aligned so the tag bits stay free.
    Layout::from_size_align(8, 8).unwrap()
}

/// Allocates a heap scalar with the given tag and raw payload bits.
pub(crate) fn new(tag: TypeTag, bits: u64) -> IValue {
    // Safety: freshly allocated, 8-aligned, non-null.
    unsafe {
        let ptr = alloc_infallible(layout()).cast::<u64>();
        ptr.as_ptr().write(bits);
        IValue::new_ptr(ptr.cast(), tag)
    }
}

impl IValue {
    /// Reads the raw payload bits of a heap scalar number.
    ///
    /// Safety: must be a heap scalar (tag `NumberI64`/`NumberU64`/`NumberF64`).
    pub(crate) unsafe fn scalar_bits(&self) -> u64 {
        self.ptr().cast::<u64>().as_ptr().read()
    }

    /// Clones a heap scalar by copying its payload into a fresh allocation.
    ///
    /// Safety: must be a heap scalar.
    pub(crate) unsafe fn scalar_clone(&self) -> IValue {
        new(self.type_tag(), self.scalar_bits())
    }

    /// Frees the allocation backing a heap scalar.
    ///
    /// Safety: must be a heap scalar.
    pub(crate) unsafe fn scalar_drop(&mut self) {
        dealloc_infallible(self.ptr(), layout());
    }
}
