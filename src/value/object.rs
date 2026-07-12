//! The JSON object representation (tag `Object`).
//!
//! An object is a single pointer to a heap allocation whose header stores the
//! length and capacity, followed by the insertion-ordered key/value pairs and a
//! Robin-Hood hash table indexing them. This module owns that layout and the
//! low-level machinery for manipulating it (the header, the split
//! item/table views, the hash probing). Every operation is an associated function
//! of the [`ObjectRepr`] representation type — the accessors and mutators directly,
//! the value-facing ones `IValue` needs (clone, drop, hash, equality, formatting)
//! through its [`ValueRepr`] impl. None refer to the public [`crate::IObject`]
//! wrapper.
//!
//! The public `IObject` type (and its `Entry`/iterator/index API) lives in the
//! top-level [`crate::object`] module. It is a thin facade that reuses the header
//! machinery exposed here.

use std::alloc::{Layout, LayoutError};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ptr::NonNull;

use crate::alloc::{alloc_infallible, dealloc_infallible};
use crate::string::IString;
use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};

use super::{
    Destructured, DestructuredMut, DestructuredRef, IValue, ReprTag, ValueRepr, ValueType,
};
use crate::object::IObject;

#[repr(C)]
#[repr(align(8))]
pub(crate) struct Header {
    pub(crate) len: usize,
    pub(crate) cap: usize,
}

#[repr(C)]
#[derive(Debug)]
pub(crate) struct KeyValuePair {
    pub(crate) key: IString,
    pub(crate) value: IValue,
}

pub(crate) struct SplitHeader<'a> {
    pub(crate) cap: usize,
    pub(crate) items: &'a [KeyValuePair],
    pub(crate) table: &'a [usize],
}

impl SplitHeader<'_> {
    pub(crate) fn find_bucket(&self, key: &IString) -> Result<usize, usize> {
        let hash_cap = ObjectRepr::hash_capacity(self.cap);
        let initial_bucket = ObjectRepr::hash_bucket(key, hash_cap);
        unsafe {
            // Linear search from expected bucket
            for i in 0..hash_cap {
                let bucket = (initial_bucket + i) % hash_cap;
                let index = *self.table.get_unchecked(bucket);

                // If we hit an empty bucket, we know the key is not present
                if index == usize::MAX {
                    return Err(bucket);
                }

                // If the bucket contains our key, we found the bucket
                let k = &self.items.get_unchecked(index).key;
                if k == key {
                    return Ok(bucket);
                }

                // If the bucket contains a different key, and its probe length is less than
                // ours, then we know our key is not present or we would have evicted this one.
                let key_dist =
                    (bucket + hash_cap - ObjectRepr::hash_bucket(k, hash_cap)) % hash_cap;
                if key_dist < i {
                    return Err(bucket);
                }
            }
        }
        Err(usize::MAX)
    }
    // Safety: index must be in bounds
    pub(crate) unsafe fn find_bucket_from_index(&self, index: usize) -> usize {
        let hash_cap = ObjectRepr::hash_capacity(self.cap);
        let key = &self.items.get_unchecked(index).key;
        let mut bucket = ObjectRepr::hash_bucket(key, hash_cap);

        // We don't bother with any early exit conditions, because
        // we know the item is present.
        while *self.table.get_unchecked(bucket) != index {
            bucket = (bucket + 1) % hash_cap;
        }

        bucket
    }
}

pub(crate) struct SplitHeaderMut<'a> {
    pub(crate) cap: usize,
    pub(crate) items: &'a mut [KeyValuePair],
    pub(crate) table: &'a mut [usize],
}

impl SplitHeaderMut<'_> {
    pub(crate) fn as_ref<'a>(&'a self) -> SplitHeader<'a> {
        SplitHeader {
            cap: self.cap,
            items: self.items,
            table: self.table,
        }
    }
    // Safety: Bucket must be valid and empty.
    //
    // Shifts elements up to fill the empty space if they are not at their ideal location.
    pub(crate) unsafe fn unshift(&mut self, initial_bucket: usize) {
        let hash_cap = ObjectRepr::hash_capacity(self.cap);
        let mut prev_bucket = initial_bucket;
        for i in 1..hash_cap {
            let bucket = (initial_bucket + i) % hash_cap;
            let index = *self.table.get_unchecked(bucket);

            // If we hit an empty bucket, we're done
            if index == usize::MAX {
                return;
            }

            // If the probe length is zero, we're done
            let k = &self.items.get_unchecked(index).key;
            if ObjectRepr::hash_bucket(k, hash_cap) == bucket {
                return;
            }

            // Shift this element back one
            self.table.swap(prev_bucket, bucket);
            prev_bucket = bucket;
        }
    }
    // Safety: item with this index must have just been pushed, and the bucket
    // index must be correct.
    //
    // Inserts an index into the table, shifting existing elements down until
    // there's an empty slot.
    pub(crate) unsafe fn shift(&mut self, initial_bucket: usize, mut index: usize) {
        let hash_cap = ObjectRepr::hash_capacity(self.cap);
        for i in 0..hash_cap {
            // If we hit an empty bucket, we're done
            if index == usize::MAX {
                return;
            }

            let bucket = (initial_bucket + i) % hash_cap;
            mem::swap(self.table.get_unchecked_mut(bucket), &mut index);
        }
    }
    // Safety: Bucket index must be in range and occupied
    pub(crate) unsafe fn remove_bucket(&mut self, bucket: usize) {
        // Remove the entry from the table
        let index = mem::replace(self.table.get_unchecked_mut(bucket), usize::MAX);

        // Unshift any displaced buckets, so the table is valid again
        self.unshift(bucket);

        // If the item being removed is not at the end of the array,
        // we need to do some book-keeping
        let last_index = self.items.len() - 1;
        if last_index != index {
            // Find the bucket containing the last item
            let bucket_to_update = self.as_ref().find_bucket_from_index(last_index);

            // Update it to point to the location where that item will be
            // after we swap it.
            *self.table.get_unchecked_mut(bucket_to_update) = index;

            // Swap the element to be removed to the back
            self.items.swap(index, last_index);
        }
    }
}

pub(crate) trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn items_ptr(&self) -> *const KeyValuePair {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr().add(1).cast() }
    }
    fn hashes_ptr(&self) -> *const usize {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.items_ptr().add(self.cap).cast() }
    }
    fn split(&self) -> SplitHeader<'a> {
        // Safety: Header `len` and `cap` must be accurate
        unsafe {
            SplitHeader {
                cap: self.cap,
                items: std::slice::from_raw_parts(self.items_ptr(), self.len),
                table: std::slice::from_raw_parts(
                    self.hashes_ptr(),
                    ObjectRepr::hash_capacity(self.cap),
                ),
            }
        }
    }
}

pub(crate) trait HeaderMut<'a>: ThinMutExt<'a, Header> {
    fn items_ptr_mut(&mut self) -> *mut KeyValuePair {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr_mut().add(1).cast() }
    }
    fn hashes_ptr_mut(&mut self) -> *mut usize {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.items_ptr_mut().add(self.cap).cast() }
    }
    fn split_mut(mut self) -> SplitHeaderMut<'a> {
        // Safety: Header `len` and `cap` must be accurate
        let len = self.len;
        let hash_cap = ObjectRepr::hash_capacity(self.cap);
        let item_ptr = self.items_ptr_mut();
        let hash_ptr = self.hashes_ptr_mut();
        unsafe {
            SplitHeaderMut {
                cap: self.cap,
                items: std::slice::from_raw_parts_mut(item_ptr as *mut _, len),
                table: std::slice::from_raw_parts_mut(hash_ptr as *mut _, hash_cap),
            }
        }
    }

    // Safety: Object must not be empty
    unsafe fn pop(&mut self) -> (IString, IValue) {
        self.len -= 1;
        let item = self.items_ptr_mut().add(self.len).read();
        (item.key, item.value)
    }
    unsafe fn push(&mut self, key: IString, value: IValue) -> usize {
        self.items_ptr_mut()
            .add(self.len)
            .write(KeyValuePair { key, value });
        let res = self.len;
        self.len += 1;
        res
    }
    fn clear(&mut self) {
        // Clear the table
        for item in self.reborrow().split_mut().table {
            *item = usize::MAX;
        }
        // Drop the items
        while self.len > 0 {
            // Safety: not empty
            unsafe {
                self.pop();
            }
        }
    }
}

impl<'a, T: ThinRefExt<'a, Header>> HeaderRef<'a> for T {}
impl<'a, T: ThinMutExt<'a, Header>> HeaderMut<'a> for T {}

/// The object representation.
pub(crate) struct ObjectRepr;

impl ObjectRepr {
    fn hash_capacity(cap: usize) -> usize {
        cap + cap / 4
    }

    fn hash_fn(s: &IString) -> usize {
        let v: &IValue = s.as_ref();
        // `usize_()` has masked the tag off, so the low 3 bits are always zero;
        // shift them out before mixing.
        let mut p = v.usize_() >> 3;
        p = p.wrapping_mul(202_529);
        p = p ^ (p >> 13);
        p.wrapping_mul(202_529)
    }

    fn hash_bucket(s: &IString, hash_cap: usize) -> usize {
        Self::hash_fn(s) % hash_cap
    }

    fn layout(cap: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<KeyValuePair>(cap)?)?
            .0
            .extend(Layout::array::<usize>(Self::hash_capacity(cap))?)?
            .0
            .pad_to_align())
    }

    fn alloc(cap: usize) -> NonNull<Header> {
        unsafe {
            let hd = alloc_infallible(Self::layout(cap).unwrap()).cast::<Header>();
            hd.write(Header { len: 0, cap });
            let mut hd_mut = ThinMut::new(hd);
            let hash_ptr = hd_mut.hashes_ptr_mut();
            for i in 0..Self::hash_capacity(cap) {
                hash_ptr.add(i).write(usize::MAX);
            }
            hd
        }
    }

    fn dealloc(ptr: NonNull<Header>) {
        unsafe {
            let layout = Self::layout(ptr.as_ref().cap).unwrap();
            dealloc_infallible(ptr.cast(), layout);
        }
    }

    // Safety (header accessors): `v` must be an *allocated* object — never the
    // empty, unallocated form (`v.usize_() == 0`), whose pointer bits are zero. The
    // read accessors guard that case; mutators grow the object first.
    pub(crate) unsafe fn header(v: &IValue) -> ThinRef<'_, Header> {
        ThinRef::new(v.ptr().cast())
    }

    // Safety: `v` must be an allocated object (see `header`).
    pub(crate) unsafe fn header_mut(v: &mut IValue) -> ThinMut<'_, Header> {
        ThinMut::new(v.ptr().cast())
    }

    // Safety: `v` must be an object.
    pub(crate) unsafe fn len(v: &IValue) -> usize {
        if v.usize_() == 0 {
            0
        } else {
            Self::header(v).len
        }
    }

    // Safety: `v` must be an object.
    pub(crate) unsafe fn capacity(v: &IValue) -> usize {
        if v.usize_() == 0 {
            0
        } else {
            Self::header(v).cap
        }
    }

    // The insertion-ordered entries; empty for the unallocated object.
    // Safety: `v` must be an object.
    pub(crate) unsafe fn items(v: &IValue) -> &[KeyValuePair] {
        if v.usize_() == 0 {
            &[]
        } else {
            Self::header(v).split().items
        }
    }

    /// Constructs a new empty object. Does not allocate: an empty object is just the
    /// `Object` tag with no pointer.
    pub(crate) fn empty() -> IValue {
        // Safety: `Object` is a non-inline tag, so the tagged word is non-null.
        unsafe { IValue::new_usize(ReprTag::Object, 0) }
    }

    /// Constructs a new object with the given capacity.
    pub(crate) fn with_capacity(cap: usize) -> IValue {
        if cap == 0 {
            Self::empty()
        } else {
            // Safety: `alloc` returns a freshly allocated, aligned header.
            unsafe { IValue::new_ptr(ReprTag::Object, Self::alloc(cap).cast()) }
        }
    }
}

impl ValueRepr for ObjectRepr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Object
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        if v.usize_() == 0 {
            return Self::empty();
        }
        let split = Self::header(v).split();
        let mut res = Self::with_capacity(split.items.len());

        if !split.items.is_empty() {
            // Safety: `res` has capacity for every entry, so it is allocated.
            let mut hd = Self::header_mut(&mut res);
            for kvp in split.items {
                // Keys in the source are unique, so every lookup is a fresh bucket.
                if let Err(bucket) = hd.split().find_bucket(&kvp.key) {
                    let index = hd.push(kvp.key.clone(), kvp.value.clone());
                    hd.reborrow().split_mut().shift(bucket, index);
                }
            }
        }
        res
    }
    unsafe fn drop(&self, v: &mut IValue) {
        if v.usize_() == 0 {
            return;
        }
        Self::header_mut(v).clear();
        Self::dealloc(v.ptr().cast());
        v.set_usize(0);
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        let entries = Self::items(v);
        state.write_usize(entries.len());

        // Order-independent: sum each entry's hash (computed with a local hasher), so
        // objects that differ only in insertion order still hash equal. Each entry
        // recurses through the standard `Hash` impls of its key and value; the value's
        // `IValue: Hash` in turn delegates down to its representation.
        let mut total_hash = 0_u64;
        for kvp in entries {
            let mut h = DefaultHasher::new();
            (&kvp.key, &kvp.value).hash(&mut h);
            total_hash = total_hash.wrapping_add(h.finish());
        }
        state.write_u64(total_hash);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        if a.raw_eq(b) {
            return true;
        }
        let len_a = Self::len(a);
        if len_a != Self::len(b) {
            return false;
        }
        if len_a == 0 {
            // Two empty objects are equal, and neither has a table to probe below.
            return true;
        }
        // Equal, non-zero lengths: both objects are allocated, so `header` is safe.
        let sa = Self::header(a).split();
        let sb = Self::header(b).split();
        for kvp in sa.items {
            // `sa` is non-empty here, so `sb` is too (equal lengths): `find_bucket`
            // is never invoked on a capacity-0 table.
            match sb.find_bucket(&kvp.key) {
                Ok(bucket) => {
                    let index = *sb.table.get_unchecked(bucket);
                    if sb.items.get_unchecked(index).value != kvp.value {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }
        true
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        // Objects have no ordering, but equal objects must still compare
        // `Some(Equal)` so `IValue`'s `PartialOrd` stays coherent with `PartialEq`
        // — including when an object is nested inside an array, whose element-wise
        // `partial_cmp` bottoms out here. This is the single boundary that owns the
        // `a == b => Some(Equal)` invariant for objects; the `IObject` wrapper
        // delegates to it. Mirrors `ArrayRepr::partial_cmp`.
        if self.eq(a, b) {
            Some(Ordering::Equal)
        } else {
            None
        }
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(Self::items(v).iter().map(|kvp| (&kvp.key, &kvp.value)))
            .finish()
    }
    fn destructure(&self, v: IValue) -> Destructured {
        Destructured::Object(IObject(v))
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        DestructuredRef::Object(v.as_object_unchecked())
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        DestructuredMut::Object(v.as_object_unchecked_mut())
    }
    unsafe fn len(&self, v: &IValue) -> Option<usize> {
        Some(Self::len(v))
    }
}
