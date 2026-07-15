use std::{
    alloc::{alloc, handle_alloc_error, Layout},
    ptr::NonNull,
};

/// Allocates for `layout`, aborting (rather than returning null) if the allocator fails.
///
/// # Safety
///
/// Like [`std::alloc::alloc`]: `layout` must have non-zero size. Every representation here
/// allocates a `Header` plus a trailing array, so the size is always non-zero; a zero-sized
/// `layout` is undefined behaviour, not an allocation failure. The returned block is
/// uninitialised — the caller must write it before reading.
#[inline]
pub(crate) unsafe fn alloc_infallible(layout: Layout) -> NonNull<u8> {
    let ptr = alloc(layout);
    if ptr.is_null() {
        handle_alloc_error(layout);
    }
    NonNull::new_unchecked(ptr)
}

/// Grows or shrinks an allocation, aborting (rather than returning null) on failure.
///
/// # Safety
///
/// Like [`std::alloc::realloc`]: `ptr` must be a block currently allocated by the global
/// allocator with `old_layout`, and `new_layout.size()` must be non-zero. `new_layout` must
/// share `old_layout`'s alignment — `realloc` cannot change it — which is `debug_assert`ed
/// below (every representation reallocs the same header-plus-array shape, so the alignment
/// is constant). The returned block holds the old contents up to the smaller of the two
/// sizes; any growth is uninitialised.
#[inline]
pub(crate) unsafe fn realloc_infallible(
    ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> NonNull<u8> {
    debug_assert_eq!(old_layout.align(), new_layout.align());

    let new_ptr = std::alloc::realloc(ptr.as_ptr(), old_layout, new_layout.size());
    if new_ptr.is_null() {
        handle_alloc_error(new_layout);
    }
    NonNull::new_unchecked(new_ptr)
}

/// Frees an allocation.
///
/// # Safety
///
/// Like [`std::alloc::dealloc`]: `ptr` must be a block currently allocated by the global
/// allocator with *exactly* `layout` (the same one it was allocated with — each
/// representation recomputes it from the stored capacity, so it matches). After this the
/// caller must not use `ptr`.
#[inline]
pub(crate) unsafe fn dealloc_infallible(ptr: NonNull<u8>, layout: Layout) {
    std::alloc::dealloc(ptr.as_ptr(), layout);
}
