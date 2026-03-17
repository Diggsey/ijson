use std::{
    alloc::{alloc, handle_alloc_error, Layout},
    ptr::NonNull,
};

#[inline]
pub unsafe fn alloc_infallible(layout: Layout) -> NonNull<u8> {
    let ptr = alloc(layout);
    if ptr.is_null() {
        handle_alloc_error(layout);
    }
    NonNull::new_unchecked(ptr)
}

#[inline]
pub unsafe fn realloc_infallible(
    ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> NonNull<u8> {
    debug_assert_eq!(old_layout.align(), new_layout.align());

    let new_ptr = std::alloc::realloc(ptr.as_ptr(), old_layout, new_layout.size());
    if new_ptr.is_null() {
        dealloc_infallible(ptr, old_layout);
        handle_alloc_error(new_layout);
    }
    NonNull::new_unchecked(new_ptr)
}

#[inline]
pub unsafe fn dealloc_infallible(ptr: NonNull<u8>, layout: Layout) {
    std::alloc::dealloc(ptr.as_ptr(), layout);
}
