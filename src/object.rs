//! Functionality relating to the JSON object type.
//!
//! [`IObject`] is the public *type* for JSON objects. It is a thin, transparent
//! wrapper around an [`IValue`] that is known to be an object. The heap layout
//! and the low-level header machinery live in the [`crate::value::object`]
//! representation module; this module builds the public API (entries, iterators,
//! indexing) on top of that machinery and delegates the value-level operations
//! (clone, drop, hash, equality, formatting) *down* to it.
//!
//! # Safety
//!
//! `IObject` maintains the invariant that its wrapped `IValue` (`self.0`) always
//! has the `Object` tag, which is the precondition the `value::object` header
//! accessors require.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;
use std::ops::{Index, IndexMut};

#[cfg(feature = "indexmap")]
use indexmap::IndexMap;

use crate::string::IString;
use crate::thin::{ThinMut, ThinMutExt, ThinRef};
use crate::value::object::{Header, HeaderMut, HeaderRef, KeyValuePair, ObjectRepr};
use crate::value::IValue;

// Safety: `header` must have capacity for an extra element.
unsafe fn build_entry<'a>(header: ThinMut<'a, Header>, key: IString) -> Entry<'a> {
    match header.split().find_bucket(&key) {
        Err(bucket) => Entry::Vacant(VacantEntry {
            header,
            bucket,
            key,
        }),
        Ok(bucket) => Entry::Occupied(OccupiedEntry { header, bucket }),
    }
}

// Safety: `header` must have capacity for an extra element.
unsafe fn build_entry_or_clone<'a>(header: ThinMut<'a, Header>, key: &IString) -> Entry<'a> {
    match header.split().find_bucket(key) {
        Err(bucket) => Entry::Vacant(VacantEntry {
            header,
            bucket,
            key: key.clone(),
        }),
        Ok(bucket) => Entry::Occupied(OccupiedEntry { header, bucket }),
    }
}

/// A view into an occupied entry in an [`IObject`]. It is part of the [`Entry`] enum.
pub struct OccupiedEntry<'a> {
    header: ThinMut<'a, Header>,
    bucket: usize,
}

impl Debug for OccupiedEntry<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", &self.get())
            .finish()
    }
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
        let split = self.header.reborrow().split_mut();
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
    /// Returns a reference to the key at this entry
    #[must_use]
    pub fn key(&self) -> &IString {
        self.get_key_value().0
    }

    /// Removes and returns the entry as a (key, value) pair.
    pub fn remove_entry(mut self) -> (IString, IValue) {
        // Safety: Bucket is known to be correct
        unsafe {
            self.header
                .reborrow()
                .split_mut()
                .remove_bucket(self.bucket);
            self.header.pop()
        }
    }
    /// Returns a reference to the value in this entry
    #[must_use]
    pub fn get(&self) -> &IValue {
        self.get_key_value().1
    }
    /// Returns a mutable reference to the value in this entry
    pub fn get_mut(&mut self) -> &mut IValue {
        self.get_key_value_mut().1
    }
    /// Converts this into a mutable reference to the value in the entry
    /// with a lifetime bound to the [`IObject`] itself.
    #[must_use]
    pub fn into_mut(self) -> &'a mut IValue {
        self.into_get_key_value_mut().1
    }

    /// Sets the value in this entry, and returns the previous value.
    pub fn insert(&mut self, value: impl Into<IValue>) -> IValue {
        mem::replace(self.get_mut(), value.into())
    }

    /// Removes this entry and returns its value.
    pub fn remove(self) -> IValue {
        self.remove_entry().1
    }
}

/// A view into a vacant entry in an [`IObject`]. It is part of the [`Entry`] enum.
pub struct VacantEntry<'a> {
    header: ThinMut<'a, Header>,
    bucket: usize,
    key: IString,
}

impl Debug for VacantEntry<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .finish()
    }
}

impl<'a> VacantEntry<'a> {
    /// Returns a reference to the key at this entry.
    #[must_use]
    pub fn key(&self) -> &IString {
        &self.key
    }
    /// Takes ownership of the key.
    #[must_use]
    pub fn into_key(self) -> IString {
        self.key
    }
    /// Inserts a value into this entry and returns a mutable reference
    /// to it.
    pub fn insert(mut self, value: impl Into<IValue>) -> &'a mut IValue {
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

/// A view into a single entry in an [`IObject`], which may be either vacant
/// or occupied.
///
/// Obtained using [`IObject::entry`].
#[derive(Debug)]
pub enum Entry<'a> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a>),
    /// A vacant entry.
    Vacant(VacantEntry<'a>),
}

impl<'a> Entry<'a> {
    /// Fills this entry if it's currently vacant, and then
    /// returns a mutable reference to the value at this entry.
    pub fn or_insert(self, default: IValue) -> &'a mut IValue {
        match self {
            Entry::Occupied(occ) => occ.into_mut(),
            Entry::Vacant(vac) => vac.insert(default),
        }
    }

    /// Fills this entry by calling the specified function if it's
    /// currently vacant, and then returns a mutable reference to
    /// the value at this entry.
    pub fn or_insert_with(self, default: impl FnOnce() -> IValue) -> &'a mut IValue {
        match self {
            Entry::Occupied(occ) => occ.into_mut(),
            Entry::Vacant(vac) => vac.insert(default()),
        }
    }

    /// Returns a reference to the key at this entry.
    #[must_use]
    pub fn key(&self) -> &IString {
        match self {
            Entry::Occupied(occ) => occ.key(),
            Entry::Vacant(vac) => vac.key(),
        }
    }

    /// Updates the value in this entry by calling the specified mutation
    /// function if the entry is occupied.
    pub fn and_modify(mut self, f: impl FnOnce(&mut IValue)) -> Self {
        if let Entry::Occupied(occ) = &mut self {
            f(occ.get_mut());
        }
        self
    }
}

/// Iterator over ([`IString`], [`IValue`]) pairs returned from
/// [`IObject::into_iter`]
pub struct IntoIter {
    reversed_object: IObject,
}

impl Debug for IntoIter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter")
            .field("reversed_object", &self.reversed_object)
            .finish()
    }
}

impl Iterator for IntoIter {
    type Item = (IString, IValue);

    fn next(&mut self) -> Option<Self::Item> {
        if self.reversed_object.is_empty() {
            None
        } else {
            Some(unsafe {
                // Safety: Object is not empty
                self.reversed_object.header_mut().pop()
            })
        }
    }
}

impl ExactSizeIterator for IntoIter {
    fn len(&self) -> usize {
        self.reversed_object.len()
    }
}

/// The `IObject` type is similar to a `HashMap<IString, IValue>`. As with the
/// [`IArray`], the length and capacity are stored _inside_ the heap allocation.
/// In addition, `IObject`s preserve the insertion order of their elements, in
/// case that is important in the original JSON.
///
/// Removing from an `IObject` will disrupt the insertion order.
///
/// [`IArray`]: crate::IArray
#[repr(transparent)]
#[derive(Clone)]
pub struct IObject(pub(crate) IValue);

value_subtype_impls!(IObject, into_object, as_object, as_object_mut);

impl IObject {
    /// Constructs a new empty `IObject`. Does not allocate.
    #[must_use]
    pub fn new() -> Self {
        IObject(ObjectRepr::empty())
    }

    /// Constructs a new `IObject` with the specified capacity. At least that many entries
    /// can be added to the object without reallocating.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        IObject(ObjectRepr::with_capacity(cap))
    }

    fn header(&self) -> ThinRef<'_, Header> {
        // Safety: `self.0` is always an object.
        unsafe { ObjectRepr::header(&self.0) }
    }

    // Safety: must not be static
    unsafe fn header_mut(&mut self) -> ThinMut<'_, Header> {
        ObjectRepr::header_mut(&mut self.0)
    }

    /// Returns the capacity of the object. This is the maximum number of entries the object
    /// can hold without reallocating.
    #[must_use]
    pub fn capacity(&self) -> usize {
        // Safety: `self.0` is always an object.
        unsafe { ObjectRepr::capacity(&self.0) }
    }
    /// Returns the number of entries currently stored in the object.
    #[must_use]
    pub fn len(&self) -> usize {
        // Safety: `self.0` is always an object.
        unsafe { ObjectRepr::len(&self.0) }
    }
    /// Returns `true` if the object is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reserves space for at least this many additional entries.
    pub fn reserve(&mut self, additional: usize) {
        // Safety: `self.0` is always an object.
        unsafe { ObjectRepr::reserve(&mut self.0, additional) }
    }

    /// Returns a view of an entry within this object.
    pub fn entry(&mut self, key: impl Into<IString>) -> Entry<'_> {
        self.reserve(1);
        // Safety: cannot be static after reserving space
        unsafe { build_entry(self.header_mut(), key.into()) }
    }
    /// Returns a view of an entry within this object, whilst avoiding
    /// cloning the key if the entry is already occupied.
    pub fn entry_or_clone(&mut self, key: &IString) -> Entry<'_> {
        self.reserve(1);
        // Safety: cannot be static after reserving space
        unsafe { build_entry_or_clone(self.header_mut(), key) }
    }
    /// Returns an iterator over references to the keys in this object.
    pub fn keys(&self) -> impl Iterator<Item = &IString> {
        self.iter().map(|x| x.0)
    }
    /// Returns an iterator over references to the values in this object.
    pub fn values(&self) -> impl Iterator<Item = &IValue> {
        self.iter().map(|x| x.1)
    }
    /// Returns an iterator over (&key, &value) pairs in this object.
    #[must_use]
    pub fn iter(&self) -> Iter<'_> {
        // Safety: `self.0` is always an object.
        Iter(unsafe { ObjectRepr::items(&self.0) }.iter())
    }
    /// Returns an iterator over mutable references to the values in
    /// this object.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut IValue> {
        self.iter_mut().map(|x| x.1)
    }
    /// Returns an iterator over (&key, &mut value) pairs in this object.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        IterMut(
            if self.is_empty() {
                &mut []
            } else {
                // Safety: not static
                unsafe { self.header_mut().split_mut().items }
            }
            .iter_mut(),
        )
    }

    /// Removes all entries from the object. The capacity is unchanged.
    pub fn clear(&mut self) {
        if !self.is_empty() {
            // Safety: not static
            unsafe {
                self.header_mut().clear();
            }
        }
    }
    /// Looks up the specified key in this object and returns a (&key, &value) pair
    /// if found.
    pub fn get_key_value(&self, k: impl ObjectIndex) -> Option<(&IString, &IValue)> {
        k.index_into(self)
    }

    /// Looks up the specified key in this object and returns a (&key, &mut value) pair
    /// if found.
    pub fn get_key_value_mut(&mut self, k: impl ObjectIndex) -> Option<(&IString, &mut IValue)> {
        k.index_into_mut(self)
    }

    /// Looks up the specified key in this object and returns a reference to the
    /// corresponding value if found.
    pub fn get(&self, k: impl ObjectIndex) -> Option<&IValue> {
        self.get_key_value(k).map(|x| x.1)
    }

    /// Looks up the specified key in this object and returns a mutable reference to
    /// the corresponding value if found.
    pub fn get_mut(&mut self, k: impl ObjectIndex) -> Option<&mut IValue> {
        self.get_key_value_mut(k).map(|x| x.1)
    }

    /// Returns `true` if the specified key exists in the object.
    pub fn contains_key(&self, k: impl ObjectIndex) -> bool {
        self.get(k).is_some()
    }

    /// Inserts a new value into this object with the specified key. If a value already
    /// existed at this key, that value is replaced and returend.
    pub fn insert(&mut self, k: impl Into<IString>, v: impl Into<IValue>) -> Option<IValue> {
        match self.entry(k) {
            Entry::Occupied(mut occ) => Some(occ.insert(v)),
            Entry::Vacant(vac) => {
                vac.insert(v);
                None
            }
        }
    }

    /// Removes the entry at the specified key, returning both the key and value if
    /// found.
    pub fn remove_entry(&mut self, k: impl ObjectIndex) -> Option<(IString, IValue)> {
        k.remove(self)
    }

    /// Removes the entry at the specified key, returning the value if found.
    pub fn remove(&mut self, k: impl ObjectIndex) -> Option<IValue> {
        self.remove_entry(k).map(|x| x.1)
    }

    /// Shrinks the memory allocation used by the object such that its
    /// capacity becomes equal to its length.
    pub fn shrink_to_fit(&mut self) {
        // Safety: `self.0` is always an object.
        unsafe { ObjectRepr::shrink_to_fit(&mut self.0) }
    }

    /// Calls the specified function for each entry in the object. Each entry
    /// where the function returns `false` is removed from the object.
    ///
    /// The function also has the ability to modify the values in-place.
    pub fn retain(&mut self, mut f: impl FnMut(&IString, &mut IValue) -> bool) {
        if !self.is_empty() {
            // Safety: not static
            let mut hd = unsafe { self.header_mut() };
            let mut index = 0;
            while index < hd.len {
                let mut split = hd.reborrow().split_mut();

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
}

impl IntoIterator for IObject {
    type Item = (IString, IValue);
    type IntoIter = IntoIter;

    fn into_iter(mut self) -> Self::IntoIter {
        if !self.is_empty() {
            // Safety: not empty, so the object is allocated.
            unsafe {
                let split_header = self.header_mut().split_mut();
                split_header.items.reverse();
            }
        }
        IntoIter {
            reversed_object: self,
        }
    }
}

impl PartialEq for IObject {
    fn eq(&self, other: &Self) -> bool {
        // Delegates through `IValue`'s own `PartialEq`, which dispatches to the
        // object representation.
        self.0 == other.0
    }
}

impl Eq for IObject {}
impl PartialOrd for IObject {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Delegate to the object representation, the single owner of the
        // `a == b => Some(Equal)` invariant (see `ObjectRepr::partial_cmp`).
        self.0.partial_cmp(&other.0)
    }
}

impl Hash for IObject {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Delegates through `IValue`'s own `Hash`, which dispatches to the object
        // representation.
        self.0.hash(state);
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

/// Trait which abstracts over the various string types which can be used
/// to index into an [`IObject`].
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
        if v.is_empty() {
            return None;
        }
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
        if v.is_empty() {
            None
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

    fn index_or_insert(self, v: &mut IObject) -> &mut IValue {
        v.entry_or_clone(self).or_insert(IValue::NULL)
    }

    fn remove(self, v: &mut IObject) -> Option<(IString, IValue)> {
        if v.is_empty() {
            None
        } else {
            // Safety: not static
            let mut hd = unsafe { v.header_mut() };
            let mut split = hd.reborrow().split_mut();
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
        // Delegates through `IValue`'s own `Debug`, which dispatches to the object
        // representation.
        Debug::fmt(&self.0, f)
    }
}

/// Iterator over (&[`IString`], &[`IValue`]) pairs returned from
/// [`IObject::iter`]
#[derive(Debug)]
pub struct Iter<'a>(std::slice::Iter<'a, KeyValuePair>);

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a IString, &'a IValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| (&x.key, &x.value))
    }
}

impl ExactSizeIterator for Iter<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Iterator over (&[`IString`], &mut [`IValue`]) pairs returned from
/// [`IObject::iter_mut`]
#[derive(Debug)]
pub struct IterMut<'a>(std::slice::IterMut<'a, KeyValuePair>);

impl<'a> Iterator for IterMut<'a> {
    type Item = (&'a IString, &'a mut IValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| (&x.key, &mut x.value))
    }
}

impl ExactSizeIterator for IterMut<'_> {
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

#[cfg(feature = "indexmap")]
impl<K: Into<IString>, V: Into<IValue>> From<IndexMap<K, V>> for IObject {
    fn from(other: IndexMap<K, V>) -> Self {
        let mut res = Self::with_capacity(other.len());
        res.extend(other.into_iter().map(|(k, v)| (k.into(), v.into())));
        res
    }
}

/// Converts a [`serde_json::Map`] into an [`IObject`].
///
/// Conversion of numeric values may be lossy if a number is not exactly
/// representable in the destination type. The exact behaviour in that case is
/// not guaranteed to be stable across versions.
impl From<serde_json::Map<String, serde_json::Value>> for IObject {
    fn from(other: serde_json::Map<String, serde_json::Value>) -> Self {
        let mut res = Self::with_capacity(other.len());
        res.extend(other.into_iter().map(|(k, v)| (k, IValue::from(v))));
        res
    }
}

/// Converts an [`IObject`] into a [`serde_json::Map`].
///
/// Conversion of numeric values may be lossy if a number is not exactly
/// representable in the destination type. The exact behaviour in that case is
/// not guaranteed to be stable across versions.
impl From<IObject> for serde_json::Map<String, serde_json::Value> {
    fn from(other: IObject) -> Self {
        other
            .into_iter()
            .map(|(k, v)| (k.as_str().to_owned(), serde_json::Value::from(v)))
            .collect()
    }
}

impl Default for IObject {
    fn default() -> Self {
        Self::new()
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
    fn equal_objects_order_equal_through_ivalue() {
        // Regression: `ObjectRepr::eq` compares objects by value, but
        // `ObjectRepr::partial_cmp` kept the `None` default. That broke the
        // `a == b => a.partial_cmp(b) == Some(Equal)` coherence law for objects
        // compared as values — directly, and nested inside an array, where it
        // would also panic `sort_by(|a, b| a.partial_cmp(b).unwrap())`.
        let make = || {
            let mut o = IObject::new();
            o.insert("k", IValue::TRUE);
            IValue::from(o)
        };

        // Two value-equal objects in distinct allocations, compared as values.
        let a = make();
        let b = make();
        assert_eq!(a, b);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));

        // The same objects nested inside arrays.
        let mut arr_a = crate::array::IArray::new();
        arr_a.push(make());
        let mut arr_b = crate::array::IArray::new();
        arr_b.push(make());
        let (arr_a, arr_b) = (IValue::from(arr_a), IValue::from(arr_b));
        assert_eq!(arr_a, arr_b);
        assert_eq!(arr_a.partial_cmp(&arr_b), Some(Ordering::Equal));
    }

    #[mockalloc::test]
    fn empty_object_is_unallocated() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_of(o: &IObject) -> u64 {
            let mut h = DefaultHasher::new();
            o.hash(&mut h);
            h.finish()
        }

        // An empty object carries no allocation (just the tag). Every read must
        // treat the unallocated form as empty, not dereference it — including the
        // hash-table lookups, which must never probe a zero-capacity table.
        let mut x = IObject::new();
        assert_eq!(x.len(), 0);
        assert_eq!(x.capacity(), 0);
        assert!(x.is_empty());
        assert_eq!(x.get("missing"), None);
        assert_eq!(x.remove("missing"), None);
        assert!(!x.contains_key("missing"));
        assert_eq!(x.iter().count(), 0);
        assert_eq!(format!("{x:?}"), "{}");
        assert_eq!(x.clone().into_iter().count(), 0);

        // Cloning stays empty and equal; an allocated-but-empty object compares and
        // hashes identically to the unallocated one.
        let allocated_empty = IObject::with_capacity(8);
        assert_eq!(x, x.clone());
        assert_eq!(x, allocated_empty);
        assert_eq!(hash_of(&x), hash_of(&allocated_empty));

        // Inserting allocates; removing the last entry returns to length zero.
        x.insert("k", IValue::NULL);
        assert_eq!(x.len(), 1);
        assert_eq!(x.remove("k"), Some(IValue::NULL));
        assert!(x.is_empty());
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
    fn can_convert_serde_json_map() {
        let mut map = serde_json::Map::new();
        map.insert("a".to_owned(), serde_json::Value::Null);
        map.insert("b".to_owned(), serde_json::json!(42));
        map.insert("c".to_owned(), serde_json::json!("hi"));

        let obj = IObject::from(map.clone());
        assert_eq!(obj.len(), 3);
        assert_eq!(obj["a"], IValue::NULL);
        assert_eq!(obj["b"], IValue::from(42));
        assert_eq!(obj["c"], IValue::from("hi"));

        let back: serde_json::Map<String, serde_json::Value> = obj.into();
        assert_eq!(back, map);
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

    // Too slow for miri
    #[cfg(not(miri))]
    #[mockalloc::test]
    fn stress_test() {
        use rand::prelude::*;

        for i in 0..10 {
            // We want our test to be random but for errors to be reproducible
            let mut rng = StdRng::seed_from_u64(i);
            let range = 0..10000;
            let mut ops: Vec<i32> = range.clone().chain(range).collect();
            ops.shuffle(&mut rng);

            let mut x = IObject::new();
            for op in ops {
                let k = IString::intern(&op.to_string());
                if x.contains_key(&k) {
                    x.remove(&k);
                } else {
                    x.insert(k, op);
                }
            }
            assert_eq!(x, IObject::new());
        }
    }
}
