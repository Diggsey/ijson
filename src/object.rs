use std::alloc::{alloc, dealloc, Layout, LayoutErr};
use std::cmp::{self, Ordering};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem::{self, MaybeUninit};
use std::ops::{Index, IndexMut};

use super::string::IString;
use super::value::{IValue, TypeTag};

#[repr(C)]
#[repr(align(4))]
struct Header {
    len: usize,
    cap: usize,
}

#[repr(C)]
struct KeyValuePair {
    key: IString,
    value: IValue,
}

fn hash_capacity(cap: usize) -> usize {
    cap + cap / 4
}

fn hash_fn(s: &IString) -> usize {
    let v: &IValue = s.as_ref();
    // We know the bottom two bits are always the same
    let mut p = v.ptr_usize() >> 2;
    p = p.wrapping_mul(202529);
    p = p ^ (p >> 13);
    p.wrapping_mul(202529)
}

fn hash_bucket(s: &IString, hash_cap: usize) -> usize {
    hash_fn(s) % hash_cap
}

struct SplitHeader<'a> {
    cap: &'a usize,
    items: &'a [KeyValuePair],
    table: &'a [usize],
}

impl<'a> SplitHeader<'a> {
    fn find_bucket(&self, key: &IString) -> Result<usize, usize> {
        let hash_cap = hash_capacity(*self.cap);
        let initial_bucket = hash_bucket(key, hash_cap);
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
                let key_dist = (hash_bucket(k, hash_cap) + hash_cap - bucket) % hash_cap;
                if key_dist < i {
                    return Err(bucket);
                }
            }
        }
        Err(usize::MAX)
    }
    // Safety: index must be in bounds
    unsafe fn find_bucket_from_index(&self, index: usize) -> usize {
        let hash_cap = hash_capacity(*self.cap);
        let key = &self.items.get_unchecked(index).key;
        let mut bucket = hash_bucket(key, hash_cap);

        // We don't bother with any early exit conditions, because
        // we know the item is present.
        while *self.table.get_unchecked(bucket) != index {
            bucket = (bucket + 1) % hash_cap;
        }

        bucket
    }
}

struct SplitHeaderMut<'a> {
    len: &'a mut usize,
    cap: &'a mut usize,
    items: &'a mut [KeyValuePair],
    table: &'a mut [usize],
}

impl<'a> SplitHeaderMut<'a> {
    fn as_ref(&self) -> SplitHeader {
        SplitHeader {
            cap: self.cap,
            items: self.items,
            table: self.table,
        }
    }
    unsafe fn unshift(&mut self, initial_bucket: usize) {
        let hash_cap = hash_capacity(*self.cap);
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
            let key_dist = (hash_bucket(k, hash_cap) + hash_cap - bucket) % hash_cap;
            if key_dist == 0 {
                return;
            }

            // Shift this element back one
            self.table.swap(prev_bucket, bucket);
            prev_bucket = bucket;
        }
    }
    // Safety: item with this index must have just been pushed, and the bucket
    // index must be correct.
    unsafe fn shift(&mut self, initial_bucket: usize, mut index: usize) {
        let hash_cap = hash_capacity(*self.cap);
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
    unsafe fn remove_bucket(&mut self, bucket: usize) {
        // Remove the entry from the table
        let index = mem::replace(self.table.get_unchecked_mut(bucket), usize::MAX);

        // Unshift any displaced buckets, so the table is valid again
        self.unshift(bucket);

        // If the item being removed is not at the end of the array,
        // we need to do some book-keeping
        let last_index = *self.len - 1;
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

impl Header {
    fn as_item_ptr(&self) -> *const KeyValuePair {
        // Safety: pointers to the end of structs are allowed
        unsafe { (self as *const Header).offset(1) as *const KeyValuePair }
    }
    fn as_hash_ptr(&self) -> *const usize {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.as_item_ptr().offset(self.cap as isize) as *const usize }
    }
    // Safety: len < cap
    unsafe fn end_item_mut(&mut self) -> &mut MaybeUninit<KeyValuePair> {
        &mut *(self.as_item_ptr().offset(self.len as isize) as *mut MaybeUninit<KeyValuePair>)
    }
    fn split(&self) -> SplitHeader {
        // Safety: Header `len` and `cap` must be accurate
        unsafe {
            SplitHeader {
                cap: &self.cap,
                items: std::slice::from_raw_parts(self.as_item_ptr(), self.len),
                table: std::slice::from_raw_parts(self.as_hash_ptr(), hash_capacity(self.cap)),
            }
        }
    }
    fn split_mut(&mut self) -> SplitHeaderMut {
        // Safety: Header `len` and `cap` must be accurate
        let len = self.len;
        let hash_cap = hash_capacity(self.cap);
        let item_ptr = self.as_item_ptr();
        let hash_ptr = self.as_hash_ptr();
        unsafe {
            SplitHeaderMut {
                len: &mut self.len,
                cap: &mut self.cap,
                items: std::slice::from_raw_parts_mut(item_ptr as *mut _, len),
                table: std::slice::from_raw_parts_mut(hash_ptr as *mut _, hash_cap),
            }
        }
    }
    fn as_mut_uninit_slice(&mut self) -> &mut [MaybeUninit<KeyValuePair>] {
        // Safety: Header `len` and `cap` must be accurate
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_item_ptr() as *mut MaybeUninit<KeyValuePair>,
                self.cap,
            )
        }
    }
    // Safety: Must ensure there's capacity for an extra element
    unsafe fn entry(&mut self, key: IString) -> Entry {
        match self.split().find_bucket(&key) {
            Err(bucket) => Entry::Vacant(VacantEntry {
                header: self,
                bucket,
                key,
            }),
            Ok(bucket) => Entry::Occupied(OccupiedEntry {
                header: self,
                bucket,
            }),
        }
    }
    // Safety: Must ensure there's capacity for an extra element
    unsafe fn entry_or_clone(&mut self, key: &IString) -> Entry {
        match self.split().find_bucket(key) {
            Err(bucket) => Entry::Vacant(VacantEntry {
                header: self,
                bucket,
                key: key.clone(),
            }),
            Ok(bucket) => Entry::Occupied(OccupiedEntry {
                header: self,
                bucket,
            }),
        }
    }
    unsafe fn pop(&mut self) -> (IString, IValue) {
        self.len -= 1;
        let item = self.end_item_mut().as_mut_ptr().read();
        (item.key, item.value)
    }
    unsafe fn push(&mut self, key: IString, value: IValue) -> usize {
        self.end_item_mut()
            .as_mut_ptr()
            .write(KeyValuePair { key, value });
        let res = self.len;
        self.len += 1;
        res
    }
    fn clear(&mut self) {
        // Clear the table
        for item in self.split_mut().table {
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

pub struct OccupiedEntry<'a> {
    header: &'a mut Header,
    bucket: usize,
}

impl<'a> OccupiedEntry<'a> {
    fn get_key_value(&self) -> (&IString, &IValue) {
        // Safety: Indices are known to be in range
        let split = self.header.split();
        unsafe {
            let index = *split.table.get_unchecked(self.bucket);
            let kvp = split.items.get_unchecked(index);
            (&kvp.key, &kvp.value)
        }
    }
    fn get_key_value_mut(&mut self) -> (&IString, &mut IValue) {
        // Safety: Indices are known to be in range
        let split = self.header.split_mut();
        unsafe {
            let index = *split.table.get_unchecked(self.bucket);
            let kvp = split.items.get_unchecked_mut(index);
            (&kvp.key, &mut kvp.value)
        }
    }
    fn into_get_key_value_mut(self) -> (&'a IString, &'a mut IValue) {
        // Safety: Indices are known to be in range
        let split = self.header.split_mut();
        unsafe {
            let index = *split.table.get_unchecked(self.bucket);
            let kvp = split.items.get_unchecked_mut(index);
            (&kvp.key, &mut kvp.value)
        }
    }
    pub fn key(&self) -> &IString {
        self.get_key_value().0
    }
    pub fn remove_entry(self) -> (IString, IValue) {
        // Safety: Bucket is known to be correct
        unsafe {
            self.header.split_mut().remove_bucket(self.bucket);
            self.header.pop()
        }
    }
    pub fn get(&self) -> &IValue {
        self.get_key_value().1
    }
    pub fn get_mut(&mut self) -> &mut IValue {
        self.get_key_value_mut().1
    }
    pub fn into_mut(self) -> &'a mut IValue {
        self.into_get_key_value_mut().1
    }
    pub fn insert(&mut self, value: impl Into<IValue>) -> IValue {
        mem::replace(self.get_mut(), value.into())
    }
    pub fn remove(self) -> IValue {
        self.remove_entry().1
    }
}

pub struct VacantEntry<'a> {
    header: &'a mut Header,
    bucket: usize,
    key: IString,
}

impl<'a> VacantEntry<'a> {
    pub fn key(&self) -> &IString {
        &self.key
    }
    pub fn into_key(self) -> IString {
        self.key
    }
    pub fn insert(self, value: impl Into<IValue>) -> &'a mut IValue {
        // Safety: we reserve space when the entry is initially created.
        // We know the bucket index is correct.
        unsafe {
            let index = self.header.push(self.key, value.into());
            let mut split = self.header.split_mut();
            split.shift(self.bucket, index);
            &mut split.items.last_mut().unwrap().value
        }
    }
}

pub enum Entry<'a> {
    Occupied(OccupiedEntry<'a>),
    Vacant(VacantEntry<'a>),
}

impl<'a> Entry<'a> {
    pub fn or_insert(self, default: IValue) -> &'a mut IValue {
        match self {
            Entry::Occupied(occ) => occ.into_mut(),
            Entry::Vacant(vac) => vac.insert(default),
        }
    }
    pub fn or_insert_with(self, default: impl FnOnce() -> IValue) -> &'a mut IValue {
        match self {
            Entry::Occupied(occ) => occ.into_mut(),
            Entry::Vacant(vac) => vac.insert(default()),
        }
    }
    pub fn key(&self) -> &IString {
        match self {
            Entry::Occupied(occ) => occ.key(),
            Entry::Vacant(vac) => vac.key(),
        }
    }
    pub fn and_modify(mut self, f: impl FnOnce(&mut IValue)) -> Self {
        if let Entry::Occupied(occ) = &mut self {
            f(occ.get_mut());
        }
        self
    }
}

pub struct IntoIter {
    header: *mut Header,
    index: usize,
}

impl Iterator for IntoIter {
    type Item = (IString, IValue);

    fn next(&mut self) -> Option<Self::Item> {
        if self.header.is_null() {
            None
        } else {
            // Safety: we set the pointer to null when it's deallocated
            unsafe {
                let len = (*self.header).len;
                let res = (*self.header)
                    .as_mut_uninit_slice()
                    .get_unchecked_mut(self.index)
                    .as_ptr()
                    .read();
                self.index += 1;
                if self.index >= len {
                    IObject::dealloc(self.header as *mut u8);
                    self.header = std::ptr::null_mut();
                }
                Some((res.key, res.value))
            }
        }
    }
}

impl ExactSizeIterator for IntoIter {
    fn len(&self) -> usize {
        if self.header.is_null() {
            0
        } else {
            // Safety: we set the pointer to null when it's deallocated
            unsafe { (*self.header).len - self.index }
        }
    }
}

impl Drop for IntoIter {
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct IObject(pub(crate) IValue);

value_subtype_impls!(IObject, into_object, as_object, as_object_mut);

static EMPTY_HEADER: Header = Header { len: 0, cap: 0 };

impl IObject {
    fn layout(cap: usize) -> Result<Layout, LayoutErr> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<KeyValuePair>(cap)?)?
            .0
            .extend(Layout::array::<usize>(hash_capacity(cap))?)?
            .0
            .pad_to_align())
    }

    fn alloc(cap: usize) -> *mut u8 {
        unsafe {
            let hd = &mut *(alloc(Self::layout(cap).unwrap()) as *mut Header);
            hd.len = 0;
            hd.cap = cap;
            for item in hd.split_mut().table {
                *item = usize::MAX;
            }
            hd as *mut _ as *mut u8
        }
    }

    fn dealloc(ptr: *mut u8) {
        unsafe {
            let layout = Self::layout((*(ptr as *const Header)).cap).unwrap();
            dealloc(ptr, layout);
        }
    }

    pub fn new() -> Self {
        unsafe { Self(IValue::new_ref(&EMPTY_HEADER, TypeTag::ObjectOrTrue)) }
    }

    pub fn with_capacity(cap: usize) -> Self {
        if cap == 0 {
            Self::new()
        } else {
            Self(unsafe { IValue::new_ptr(Self::alloc(cap), TypeTag::ObjectOrTrue) })
        }
    }

    fn header(&self) -> &Header {
        unsafe { &*(self.0.ptr() as *const Header) }
    }

    // Safety: must not be static
    unsafe fn header_mut(&mut self) -> &mut Header {
        &mut *(self.0.ptr() as *mut Header)
    }

    fn is_static(&self) -> bool {
        self.capacity() == 0
    }
    pub fn capacity(&self) -> usize {
        self.header().cap
    }
    pub fn len(&self) -> usize {
        self.header().len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn resize_internal(&mut self, cap: usize) {
        let old_obj = mem::replace(self, Self::with_capacity(cap));
        if !self.is_static() {
            unsafe {
                let hd = self.header_mut();
                for (k, v) in old_obj {
                    if let Err(bucket) = hd.split().find_bucket(&k) {
                        let index = hd.push(k, v);
                        hd.split_mut().shift(bucket, index);
                    }
                }
            }
        }
    }
    pub fn reserve(&mut self, additional: usize) {
        let hd = self.header();
        let current_capacity = hd.cap;
        let desired_capacity = hd.len.checked_add(additional).unwrap();
        if current_capacity >= desired_capacity {
            return;
        }
        self.resize_internal(cmp::max(current_capacity * 2, desired_capacity.max(4)));
    }

    pub fn entry(&mut self, key: impl Into<IString>) -> Entry {
        self.reserve(1);
        // Safety: cannot be static after reserving space
        unsafe { self.header_mut().entry(key.into()) }
    }
    pub fn entry_or_clone(&mut self, key: &IString) -> Entry {
        self.reserve(1);
        // Safety: cannot be static after reserving space
        unsafe { self.header_mut().entry_or_clone(key) }
    }
    pub fn keys(&self) -> impl Iterator<Item = &IString> {
        self.iter().map(|x| x.0)
    }
    pub fn values(&self) -> impl Iterator<Item = &IValue> {
        self.iter().map(|x| x.1)
    }
    pub fn iter(&self) -> Iter {
        Iter(self.header().split().items.iter())
    }
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut IValue> {
        self.iter_mut().map(|x| x.1)
    }
    pub fn iter_mut(&mut self) -> IterMut {
        IterMut(
            if self.is_static() {
                &mut []
            } else {
                // Safety: not static
                unsafe { self.header_mut().split_mut().items }
            }
            .iter_mut(),
        )
    }
    pub fn clear(&mut self) {
        if !self.is_static() {
            // Safety: not static
            unsafe {
                self.header_mut().clear();
            }
        }
    }
    pub fn get_key_value(&self, k: impl ObjectIndex) -> Option<(&IString, &IValue)> {
        k.index_into(self)
    }
    pub fn get_key_value_mut(&mut self, k: impl ObjectIndex) -> Option<(&IString, &mut IValue)> {
        k.index_into_mut(self)
    }
    pub fn get(&self, k: impl ObjectIndex) -> Option<&IValue> {
        self.get_key_value(k).map(|x| x.1)
    }
    pub fn get_mut(&mut self, k: impl ObjectIndex) -> Option<&mut IValue> {
        self.get_key_value_mut(k).map(|x| x.1)
    }
    pub fn insert(&mut self, k: impl Into<IString>, v: impl Into<IValue>) -> Option<IValue> {
        match self.entry(k) {
            Entry::Occupied(mut occ) => Some(occ.insert(v)),
            Entry::Vacant(vac) => {
                vac.insert(v);
                None
            }
        }
    }
    pub fn remove_entry(&mut self, k: impl ObjectIndex) -> Option<(IString, IValue)> {
        k.remove(self)
    }
    pub fn remove(&mut self, k: impl ObjectIndex) -> Option<IValue> {
        self.remove_entry(k).map(|x| x.1)
    }
    pub fn shrink_to_fit(&mut self) {
        self.resize_internal(self.len());
    }
    pub fn retain(&mut self, mut f: impl FnMut(&IString, &mut IValue) -> bool) {
        if self.is_static() {
            return;
        } else {
            // Safety: not static
            let hd = unsafe { self.header_mut() };
            let mut index = 0;
            while index < hd.len {
                let mut split = hd.split_mut();

                // Safety: Indices are in range
                unsafe {
                    let kvp = split.items.get_unchecked_mut(index);
                    if f(&kvp.key, &mut kvp.value) {
                        index += 1;
                    } else {
                        let bucket = split.as_ref().find_bucket_from_index(index);
                        split.remove_bucket(bucket);
                        hd.pop();
                    }
                }
            }
        }
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        let mut res = Self::with_capacity(self.len());
        for (k, v) in self.iter() {
            res.insert(k.clone(), v.clone());
        }

        res.0
    }
    pub(crate) fn drop_impl(&mut self) {
        self.clear();
        if !self.is_static() {
            unsafe {
                Self::dealloc(self.0.ptr());
                self.0.set_ref(&EMPTY_HEADER);
            }
        }
    }
}

impl IntoIterator for IObject {
    type Item = (IString, IValue);
    type IntoIter = IntoIter;

    fn into_iter(mut self) -> Self::IntoIter {
        if self.is_static() {
            IntoIter {
                header: std::ptr::null_mut(),
                index: 0,
            }
        } else {
            // Safety: not static
            unsafe {
                let header = self.header_mut() as *mut _;
                mem::forget(self);
                IntoIter { header, index: 0 }
            }
        }
    }
}

impl PartialEq for IObject {
    fn eq(&self, other: &Self) -> bool {
        if self.0.raw_eq(&other.0) {
            return true;
        }
        if self.len() != other.len() {
            return false;
        }
        for (k, v) in self.iter() {
            if other.get(k) != Some(v) {
                return false;
            }
        }
        true
    }
}

impl Eq for IObject {}
impl PartialOrd for IObject {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else {
            None
        }
    }
}

impl Hash for IObject {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);

        let mut total_hash = 0u64;
        for item in self.iter() {
            let mut h = DefaultHasher::new();
            item.hash(&mut h);
            total_hash = total_hash.wrapping_add(h.finish());
        }
        total_hash.hash(state);
    }
}

impl<K: Into<IString>, V: Into<IValue>> Extend<(K, V)> for IObject {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Into<IString>, V: Into<IValue>> FromIterator<(K, V)> for IObject {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut res = IObject::new();
        res.extend(iter);
        res
    }
}

impl<I: ObjectIndex> Index<I> for IObject {
    type Output = IValue;

    #[inline]
    fn index(&self, index: I) -> &IValue {
        index.index_into(self).unwrap().1
    }
}

impl<I: ObjectIndex> IndexMut<I> for IObject {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut IValue {
        index.index_or_insert(self)
    }
}

mod private {
    #[doc(hidden)]
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for &str {}
    impl Sealed for &super::IString {}
    impl<T: Sealed> Sealed for &T {}
}

pub trait ObjectIndex: private::Sealed + Copy {
    #[doc(hidden)]
    fn index_into(self, v: &IObject) -> Option<(&IString, &IValue)>;

    #[doc(hidden)]
    fn index_into_mut(self, v: &mut IObject) -> Option<(&IString, &mut IValue)>;

    #[doc(hidden)]
    fn index_or_insert(self, v: &mut IObject) -> &mut IValue;

    #[doc(hidden)]
    fn remove(self, v: &mut IObject) -> Option<(IString, IValue)>;
}

impl ObjectIndex for &str {
    fn index_into(self, v: &IObject) -> Option<(&IString, &IValue)> {
        IString::intern(self).index_into(v)
    }

    fn index_into_mut(self, v: &mut IObject) -> Option<(&IString, &mut IValue)> {
        IString::intern(self).index_into_mut(v)
    }

    fn index_or_insert(self, v: &mut IObject) -> &mut IValue {
        v.entry(IString::intern(self)).or_insert(IValue::NULL)
    }

    fn remove(self, v: &mut IObject) -> Option<(IString, IValue)> {
        IString::intern(self).remove(v)
    }
}

impl ObjectIndex for &IString {
    fn index_into(self, v: &IObject) -> Option<(&IString, &IValue)> {
        let hd = v.header().split();
        if let Ok(bucket) = hd.find_bucket(self) {
            // Safety: Bucket index is valid
            unsafe {
                let index = *hd.table.get_unchecked(bucket);
                let item = hd.items.get_unchecked(index);
                Some((&item.key, &item.value))
            }
        } else {
            None
        }
    }

    fn index_into_mut(self, v: &mut IObject) -> Option<(&IString, &mut IValue)> {
        if v.is_static() {
            return None;
        } else {
            // Safety: not static
            let hd = unsafe { v.header_mut().split_mut() };
            if let Ok(bucket) = hd.as_ref().find_bucket(self) {
                // Safety: Bucket index is valid
                unsafe {
                    let index = *hd.table.get_unchecked(bucket);
                    let item = hd.items.get_unchecked_mut(index);
                    Some((&item.key, &mut item.value))
                }
            } else {
                None
            }
        }
    }

    fn index_or_insert<'v>(self, v: &'v mut IObject) -> &'v mut IValue {
        v.entry_or_clone(self).or_insert(IValue::NULL)
    }

    fn remove(self, v: &mut IObject) -> Option<(IString, IValue)> {
        if v.is_static() {
            return None;
        } else {
            // Safety: not static
            let hd = unsafe { v.header_mut() };
            let mut split = hd.split_mut();
            if let Ok(bucket) = split.as_ref().find_bucket(self) {
                // Safety: Bucket index is valid
                unsafe {
                    split.remove_bucket(bucket);
                    Some(hd.pop())
                }
            } else {
                None
            }
        }
    }
}

impl<T: ObjectIndex> ObjectIndex for &T {
    fn index_into(self, v: &IObject) -> Option<(&IString, &IValue)> {
        (*self).index_into(v)
    }

    fn index_into_mut(self, v: &mut IObject) -> Option<(&IString, &mut IValue)> {
        (*self).index_into_mut(v)
    }

    fn index_or_insert(self, v: &mut IObject) -> &mut IValue {
        (*self).index_or_insert(v)
    }

    fn remove(self, v: &mut IObject) -> Option<(IString, IValue)> {
        (*self).remove(v)
    }
}

impl Debug for IObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

pub struct Iter<'a>(std::slice::Iter<'a, KeyValuePair>);

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a IString, &'a IValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| (&x.key, &x.value))
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

pub struct IterMut<'a>(std::slice::IterMut<'a, KeyValuePair>);

impl<'a> Iterator for IterMut<'a> {
    type Item = (&'a IString, &'a mut IValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| (&x.key, &mut x.value))
    }
}

impl<'a> ExactSizeIterator for IterMut<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a> IntoIterator for &'a IObject {
    type Item = (&'a IString, &'a IValue);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut IObject {
    type Item = (&'a IString, &'a mut IValue);
    type IntoIter = IterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Into<IString>, V: Into<IValue>> From<HashMap<K, V>> for IObject {
    fn from(other: HashMap<K, V>) -> Self {
        let mut res = Self::with_capacity(other.len());
        res.extend(other.into_iter().map(|(k, v)| (k.into(), v.into())));
        res
    }
}

impl<K: Into<IString>, V: Into<IValue>> From<BTreeMap<K, V>> for IObject {
    fn from(other: BTreeMap<K, V>) -> Self {
        let mut res = Self::with_capacity(other.len());
        res.extend(other.into_iter().map(|(k, v)| (k.into(), v.into())));
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[mockalloc::test]
    fn can_create() {
        let x = IObject::new();
        let y = IObject::with_capacity(10);

        assert_eq!(x, y);
    }

    #[mockalloc::test]
    fn can_collect() {
        let x = vec![
            ("a", IValue::NULL),
            ("b", IValue::TRUE),
            ("c", IValue::FALSE),
        ];
        let y: IObject = x.into_iter().collect();

        assert_eq!(y, y.clone());
        assert_eq!(y.len(), 3);
        assert_eq!(y["a"], IValue::NULL);
        assert_eq!(y["b"], IValue::TRUE);
        assert_eq!(y["c"], IValue::FALSE);
    }

    #[mockalloc::test]
    fn can_insert() {
        let mut x = IObject::new();
        x.insert("a", IValue::NULL);
        x.insert("b", IValue::TRUE);
        x.insert("c", IValue::FALSE);

        assert_eq!(x.len(), 3);
        assert_eq!(x["a"], IValue::NULL);
        assert_eq!(x["b"], IValue::TRUE);
        assert_eq!(x["c"], IValue::FALSE);
    }

    #[mockalloc::test]
    fn can_nest() {
        let mut x = IObject::new();
        x.insert("a", IValue::NULL);
        x.insert("b", x.clone());
        x.insert("c", IValue::FALSE);
        x.insert("d", x.clone());

        assert_eq!(x.len(), 4);
        assert_eq!(x["a"], IValue::NULL);
        assert_eq!(x["b"].len(), Some(1));
        assert_eq!(x["c"], IValue::FALSE);
        assert_eq!(x["d"].len(), Some(3));
    }

    #[mockalloc::test]
    fn can_remove_and_shrink() {
        let x = vec![
            ("a", IValue::NULL),
            ("b", IValue::TRUE),
            ("c", IValue::FALSE),
        ];
        let mut y: IObject = x.into_iter().collect();
        assert_eq!(y.len(), 3);
        assert_eq!(y.capacity(), 4);

        assert_eq!(y.remove("b"), Some(IValue::TRUE));
        assert_eq!(y.remove("b"), None);
        assert_eq!(y.remove("d"), None);

        assert_eq!(y.len(), 2);
        assert_eq!(y.capacity(), 4);
        assert_eq!(y["a"], IValue::NULL);
        assert_eq!(y["c"], IValue::FALSE);

        y.shrink_to_fit();
        assert_eq!(y.len(), 2);
        assert_eq!(y.capacity(), 2);
        assert_eq!(y["a"], IValue::NULL);
        assert_eq!(y["c"], IValue::FALSE);
    }
}
