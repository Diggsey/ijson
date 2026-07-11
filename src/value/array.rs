//! The JSON array representation (tag `Array`).
//!
//! An array is a single pointer to a heap allocation whose header stores the
//! length and capacity inline, followed by the [`IValue`] elements. This module
//! owns that layout and every operation on it, exposed as associated functions of
//! the [`ArrayRepr`] representation type that operate on an `&IValue` (or
//! `&mut IValue`) known to be an array. It never refers to the public
//! [`crate::IArray`] wrapper — `IArray` is a thin facade that delegates *down* to
//! these functions, as does `IValue` itself.
//!
//! # Safety
//!
//! Every `unsafe fn` here shares one precondition: the `IValue` argument must have
//! the `Array` tag. Callers (`IValue`'s trait impls and the `IArray` facade) uphold
//! this via the type/tag they already checked.

use std::alloc::{Layout, LayoutError};
use std::cmp::{self, Ordering};
use std::fmt::{self, Debug, Formatter};
use std::hash::Hasher;
use std::ptr::NonNull;

use crate::alloc::{alloc_infallible, dealloc_infallible, realloc_infallible};
use crate::array::IArray;
use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};

use super::{
    Destructured, DestructuredMut, DestructuredRef, IValue, ReprTag, ValueRepr, ValueType,
};

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

/// The array representation.
pub(crate) struct ArrayRepr;

impl ArrayRepr {
    fn layout(cap: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<usize>(cap)?)?
            .0
            .pad_to_align())
    }

    fn alloc(cap: usize) -> NonNull<Header> {
        unsafe {
            let ptr = alloc_infallible(Self::layout(cap).unwrap()).cast::<Header>();
            ptr.write(Header { len: 0, cap });
            ptr
        }
    }

    fn realloc(ptr: NonNull<Header>, new_cap: usize) -> NonNull<Header> {
        unsafe {
            let old_layout = Self::layout(ptr.as_ref().cap).unwrap();
            let new_layout = Self::layout(new_cap).unwrap();
            let mut ptr = realloc_infallible(ptr.cast(), old_layout, new_layout).cast::<Header>();
            ptr.as_mut().cap = new_cap;
            ptr
        }
    }

    fn dealloc(ptr: NonNull<Header>) {
        unsafe {
            let layout = Self::layout(ptr.as_ref().cap).unwrap();
            dealloc_infallible(ptr.cast(), layout);
        }
    }

    // Safety (the header accessors): `v` must be an *allocated* array — never the
    // empty, unallocated form (`v.usize_() == 0`), whose pointer bits are zero. The
    // read accessors guard that case; mutators grow the array first.
    unsafe fn header(v: &IValue) -> ThinRef<'_, Header> {
        ThinRef::new(v.ptr().cast())
    }

    // Safety: `v` must be an allocated array (see `header`).
    unsafe fn header_mut(v: &mut IValue) -> ThinMut<'_, Header> {
        ThinMut::new(v.ptr().cast())
    }

    // Safety: `v` must be an array.
    unsafe fn resize_internal(v: &mut IValue, cap: usize) {
        if v.usize_() == 0 || cap == 0 {
            *v = Self::with_capacity(cap);
        } else {
            let new_ptr = Self::realloc(v.ptr().cast(), cap);
            v.set_ptr(new_ptr.cast());
        }
    }

    /// Constructs a new empty array. Does not allocate: an empty array is just the
    /// `Array` tag with no pointer.
    pub(crate) fn empty() -> IValue {
        // Safety: `Array` is a non-inline tag, so the tagged word is non-null.
        unsafe { IValue::new_usize(ReprTag::Array, 0) }
    }

    /// Constructs a new array with the given capacity.
    pub(crate) fn with_capacity(cap: usize) -> IValue {
        if cap == 0 {
            Self::empty()
        } else {
            // Safety: `alloc` returns a freshly allocated, aligned header.
            unsafe { IValue::new_ptr(ReprTag::Array, Self::alloc(cap).cast()) }
        }
    }

    pub(crate) unsafe fn capacity(v: &IValue) -> usize {
        if v.usize_() == 0 {
            0
        } else {
            Self::header(v).cap
        }
    }

    pub(crate) unsafe fn len(v: &IValue) -> usize {
        if v.usize_() == 0 {
            0
        } else {
            Self::header(v).len
        }
    }

    pub(crate) unsafe fn as_slice(v: &IValue) -> &[IValue] {
        if v.usize_() == 0 {
            &[]
        } else {
            Self::header(v).items_slice()
        }
    }

    pub(crate) unsafe fn as_mut_slice(v: &mut IValue) -> &mut [IValue] {
        if v.usize_() == 0 {
            &mut []
        } else {
            Self::header_mut(v).items_slice_mut()
        }
    }

    pub(crate) unsafe fn reserve(v: &mut IValue, additional: usize) {
        let (current_capacity, len) = if v.usize_() == 0 {
            (0, 0)
        } else {
            let hd = Self::header(v);
            (hd.cap, hd.len)
        };
        let desired_capacity = len.checked_add(additional).unwrap();
        if current_capacity >= desired_capacity {
            return;
        }
        Self::resize_internal(v, cmp::max(current_capacity * 2, desired_capacity.max(4)));
    }

    pub(crate) unsafe fn truncate(v: &mut IValue, len: usize) {
        if v.usize_() == 0 {
            return;
        }
        let mut hd = Self::header_mut(v);
        while hd.len > len {
            hd.pop();
        }
    }

    pub(crate) unsafe fn insert(v: &mut IValue, index: usize, item: IValue) {
        Self::reserve(v, 1);

        // Safety: cannot be the empty form after calling `reserve`
        let mut hd = Self::header_mut(v);
        assert!(index <= hd.len);

        // Safety: We just reserved enough space for at least one extra item
        hd.push(item);
        if index < hd.len {
            hd.items_slice_mut()[index..].rotate_right(1);
        }
    }

    pub(crate) unsafe fn remove(v: &mut IValue, index: usize) -> Option<IValue> {
        if index < Self::len(v) {
            // Safety: cannot be the empty form if index < len
            let mut hd = Self::header_mut(v);
            hd.reborrow().items_slice_mut()[index..].rotate_left(1);
            hd.pop()
        } else {
            None
        }
    }

    pub(crate) unsafe fn swap_remove(v: &mut IValue, index: usize) -> Option<IValue> {
        if index < Self::len(v) {
            // Safety: cannot be the empty form if index < len
            let mut hd = Self::header_mut(v);
            let last_index = hd.len - 1;
            hd.reborrow().items_slice_mut().swap(index, last_index);
            hd.pop()
        } else {
            None
        }
    }

    pub(crate) unsafe fn push(v: &mut IValue, item: IValue) {
        Self::reserve(v, 1);
        // Safety: We just reserved enough space for at least one extra item
        Self::header_mut(v).push(item);
    }

    pub(crate) unsafe fn pop(v: &mut IValue) -> Option<IValue> {
        if v.usize_() == 0 {
            None
        } else {
            Self::header_mut(v).pop()
        }
    }

    pub(crate) unsafe fn shrink_to_fit(v: &mut IValue) {
        Self::resize_internal(v, Self::len(v));
    }
}

impl ValueRepr for ArrayRepr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Array
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        let src = Self::as_slice(v);
        let l = src.len();
        let mut res = Self::with_capacity(l);

        if l > 0 {
            // Safety: `res` has capacity for every element, so it is not the empty form
            let mut hd = Self::header_mut(&mut res);
            for item in src {
                // Safety: we reserved enough space at the start
                hd.push(item.clone());
            }
        }
        res
    }
    unsafe fn drop(&self, v: &mut IValue) {
        Self::truncate(v, 0);
        if v.usize_() != 0 {
            Self::dealloc(v.ptr().cast());
            v.set_usize(0);
        }
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        let items = Self::as_slice(v);
        state.write_usize(items.len());
        // Order matters for arrays: feed each element in turn, delegating down to its
        // own representation via `repr()` (elements can be any value type).
        for item in items {
            item.repr().hash(item, state);
        }
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        a.raw_eq(b) || Self::as_slice(a) == Self::as_slice(b)
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        if a.raw_eq(b) {
            Some(Ordering::Equal)
        } else {
            Self::as_slice(a).partial_cmp(Self::as_slice(b))
        }
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(Self::as_slice(v), f)
    }
    fn destructure(&self, v: IValue) -> Destructured {
        Destructured::Array(IArray(v))
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        DestructuredRef::Array(v.as_array_unchecked())
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        DestructuredMut::Array(v.as_array_unchecked_mut())
    }
    unsafe fn len(&self, v: &IValue) -> Option<usize> {
        Some(Self::len(v))
    }
}
