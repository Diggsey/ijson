//! The JSON array representation (tag `Array`).
//!
//! An array is a single pointer to a heap allocation whose header stores the
//! length and capacity inline, followed by the [`IValue`] elements. This module
//! owns that layout and every operation on it, exposed as free functions that
//! operate directly on an `&IValue` (or `&mut IValue`) known to be an array. It
//! never refers to the public [`crate::IArray`] wrapper — `IArray` is a thin
//! facade that delegates *down* to these functions, as does `IValue` itself.
//!
//! # Safety
//!
//! Every `unsafe fn` in this module shares one precondition: the `IValue`
//! argument must have the `Array` tag. Callers (`IValue`'s trait impls and the
//! `IArray` facade) uphold this via the type/tag they already checked.

use std::alloc::{Layout, LayoutError};
use std::cmp::{self, Ordering};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;

use crate::alloc::{alloc_infallible, dealloc_infallible, realloc_infallible};
use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};

use super::{IValue, TypeTag};

#[repr(C)]
#[repr(align(8))]
struct Header {
    len: usize,
    cap: usize,
}

trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn array_ptr(&self) -> *const IValue {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr().add(1).cast::<IValue>() }
    }
    fn items_slice(&self) -> &'a [IValue] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts(self.array_ptr(), self.len) }
    }
}

trait HeaderMut<'a>: ThinMutExt<'a, Header> {
    fn array_ptr_mut(mut self) -> *mut IValue {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr_mut().add(1).cast::<IValue>() }
    }
    fn items_slice_mut(self) -> &'a mut [IValue] {
        // Safety: Header `len` must be accurate
        let len = self.len;
        unsafe { std::slice::from_raw_parts_mut(self.array_ptr_mut(), len) }
    }
    // Safety: Space must already be allocated for the item
    unsafe fn push(&mut self, item: IValue) {
        let index = self.len;
        self.reborrow().array_ptr_mut().add(index).write(item);
        self.len += 1;
    }
    fn pop(&mut self) -> Option<IValue> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            let index = self.len;

            // Safety: We just checked that an item exists
            unsafe { Some(self.reborrow().array_ptr_mut().add(index).read()) }
        }
    }
}

impl<'a, T: ThinRefExt<'a, Header>> HeaderRef<'a> for T {}
impl<'a, T: ThinMutExt<'a, Header>> HeaderMut<'a> for T {}

static EMPTY_HEADER: Header = Header { len: 0, cap: 0 };

fn layout(cap: usize) -> Result<Layout, LayoutError> {
    Ok(Layout::new::<Header>()
        .extend(Layout::array::<usize>(cap)?)?
        .0
        .pad_to_align())
}

fn alloc(cap: usize) -> NonNull<Header> {
    unsafe {
        let ptr = alloc_infallible(layout(cap).unwrap()).cast::<Header>();
        ptr.write(Header { len: 0, cap });
        ptr
    }
}

fn realloc(ptr: NonNull<Header>, new_cap: usize) -> NonNull<Header> {
    unsafe {
        let old_layout = layout(ptr.as_ref().cap).unwrap();
        let new_layout = layout(new_cap).unwrap();
        let mut ptr = realloc_infallible(ptr.cast(), old_layout, new_layout).cast::<Header>();
        ptr.as_mut().cap = new_cap;
        ptr
    }
}

fn dealloc(ptr: NonNull<Header>) {
    unsafe {
        let layout = layout(ptr.as_ref().cap).unwrap();
        dealloc_infallible(ptr.cast(), layout);
    }
}

// Safety (all header helpers): `v` must be an array.
unsafe fn header(v: &IValue) -> ThinRef<'_, Header> {
    ThinRef::new(v.ptr().cast())
}

// Safety: `v` must be an array and must not be the shared static empty header.
unsafe fn header_mut(v: &mut IValue) -> ThinMut<'_, Header> {
    ThinMut::new(v.ptr().cast())
}

// Safety: `v` must be an array. A static (capacity-0) array shares the immutable
// `EMPTY_HEADER` and so must never be mutated in place.
unsafe fn is_static(v: &IValue) -> bool {
    header(v).cap == 0
}

// Safety: `v` must be an array.
unsafe fn resize_internal(v: &mut IValue, cap: usize) {
    if is_static(v) || cap == 0 {
        *v = with_capacity(cap);
    } else {
        let new_ptr = realloc(v.ptr().cast(), cap);
        v.set_ptr(new_ptr.cast());
    }
}

/// Constructs a new empty array. Does not allocate.
pub(crate) fn new() -> IValue {
    // Safety: `EMPTY_HEADER` is a valid, aligned static header.
    unsafe { IValue::new_ref(&EMPTY_HEADER, TypeTag::Array) }
}

/// Constructs a new array with the given capacity.
pub(crate) fn with_capacity(cap: usize) -> IValue {
    if cap == 0 {
        new()
    } else {
        // Safety: `alloc` returns a freshly allocated, aligned header.
        unsafe { IValue::new_ptr(alloc(cap).cast(), TypeTag::Array) }
    }
}

pub(crate) unsafe fn capacity(v: &IValue) -> usize {
    header(v).cap
}

pub(crate) unsafe fn len(v: &IValue) -> usize {
    header(v).len
}

pub(crate) unsafe fn as_slice(v: &IValue) -> &[IValue] {
    header(v).items_slice()
}

pub(crate) unsafe fn as_mut_slice(v: &mut IValue) -> &mut [IValue] {
    if is_static(v) {
        &mut []
    } else {
        header_mut(v).items_slice_mut()
    }
}

pub(crate) unsafe fn reserve(v: &mut IValue, additional: usize) {
    let (current_capacity, len) = {
        let hd = header(v);
        (hd.cap, hd.len)
    };
    let desired_capacity = len.checked_add(additional).unwrap();
    if current_capacity >= desired_capacity {
        return;
    }
    resize_internal(v, cmp::max(current_capacity * 2, desired_capacity.max(4)));
}

pub(crate) unsafe fn truncate(v: &mut IValue, len: usize) {
    if is_static(v) {
        return;
    }
    let mut hd = header_mut(v);
    while hd.len > len {
        hd.pop();
    }
}

pub(crate) unsafe fn insert(v: &mut IValue, index: usize, item: IValue) {
    reserve(v, 1);

    // Safety: cannot be static after calling `reserve`
    let mut hd = header_mut(v);
    assert!(index <= hd.len);

    // Safety: We just reserved enough space for at least one extra item
    hd.push(item);
    if index < hd.len {
        hd.items_slice_mut()[index..].rotate_right(1);
    }
}

pub(crate) unsafe fn remove(v: &mut IValue, index: usize) -> Option<IValue> {
    if index < len(v) {
        // Safety: cannot be static if index < len
        let mut hd = header_mut(v);
        hd.reborrow().items_slice_mut()[index..].rotate_left(1);
        hd.pop()
    } else {
        None
    }
}

pub(crate) unsafe fn swap_remove(v: &mut IValue, index: usize) -> Option<IValue> {
    if index < len(v) {
        // Safety: cannot be static if index < len
        let mut hd = header_mut(v);
        let last_index = hd.len - 1;
        hd.reborrow().items_slice_mut().swap(index, last_index);
        hd.pop()
    } else {
        None
    }
}

pub(crate) unsafe fn push(v: &mut IValue, item: IValue) {
    reserve(v, 1);
    // Safety: We just reserved enough space for at least one extra item
    header_mut(v).push(item);
}

pub(crate) unsafe fn pop(v: &mut IValue) -> Option<IValue> {
    if is_static(v) {
        None
    } else {
        header_mut(v).pop()
    }
}

pub(crate) unsafe fn shrink_to_fit(v: &mut IValue) {
    resize_internal(v, len(v));
}

pub(crate) unsafe fn clone(v: &IValue) -> IValue {
    let src = header(v).items_slice();
    let l = src.len();
    let mut res = with_capacity(l);

    if l > 0 {
        // Safety: we cannot be static if len > 0
        let mut hd = header_mut(&mut res);
        for item in src {
            // Safety: we reserved enough space at the start
            hd.push(item.clone());
        }
    }
    res
}

pub(crate) unsafe fn drop(v: &mut IValue) {
    truncate(v, 0);
    if !is_static(v) {
        dealloc(v.ptr().cast());
        v.set_ref(&EMPTY_HEADER);
    }
}

pub(crate) unsafe fn hash<H: Hasher>(v: &IValue, state: &mut H) {
    // Recurses into each element through the standard slice/`IValue` `Hash` impl.
    as_slice(v).hash(state);
}

pub(crate) unsafe fn eq(a: &IValue, b: &IValue) -> bool {
    a.raw_eq(b) || as_slice(a) == as_slice(b)
}

pub(crate) unsafe fn cmp(a: &IValue, b: &IValue) -> Option<Ordering> {
    if a.raw_eq(b) {
        Some(Ordering::Equal)
    } else {
        as_slice(a).partial_cmp(as_slice(b))
    }
}

pub(crate) unsafe fn debug(v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
    Debug::fmt(as_slice(v), f)
}
