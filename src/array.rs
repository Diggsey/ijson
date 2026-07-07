//! Functionality relating to the JSON array type.
//!
//! [`IArray`] is the public *type* for JSON arrays. It is a thin, transparent
//! wrapper around an [`IValue`] that is known to be an array; the heap layout and
//! every operation on it live in the [`crate::value::array`] representation
//! module. Each method here simply delegates *down* to that module.
//!
//! # Safety
//!
//! `IArray` maintains the invariant that its wrapped `IValue` (`self.0`) always
//! has the `Array` tag, which is exactly the precondition the `value::array`
//! functions require. Every delegation below relies on that invariant.

use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::SliceIndex;

use crate::value::array as repr;
use crate::value::IValue;

/// Iterator over [`IValue`]s returned from [`IArray::into_iter`]
pub struct IntoIter {
    reversed_array: IArray,
}

impl Iterator for IntoIter {
    type Item = IValue;

    fn next(&mut self) -> Option<Self::Item> {
        self.reversed_array.pop()
    }
}

impl ExactSizeIterator for IntoIter {
    fn len(&self) -> usize {
        self.reversed_array.len()
    }
}

impl Debug for IntoIter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter")
            .field("reversed_array", &self.reversed_array)
            .finish()
    }
}

/// The `IArray` type is similar to a `Vec<IValue>`. The primary difference is
/// that the length and capacity are stored _inside_ the heap allocation, so that
/// the `IArray` itself can be a single pointer.
#[repr(transparent)]
#[derive(Clone)]
pub struct IArray(pub(crate) IValue);

value_subtype_impls!(IArray, into_array, as_array, as_array_mut);

impl IArray {
    /// Constructs a new empty `IArray`. Does not allocate.
    #[must_use]
    pub fn new() -> Self {
        IArray(repr::new())
    }

    /// Constructs a new `IArray` with the specified capacity. At least that many items
    /// can be added to the array without reallocating.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        IArray(repr::with_capacity(cap))
    }

    /// Returns the capacity of the array. This is the maximum number of items the array
    /// can hold without reallocating.
    #[must_use]
    pub fn capacity(&self) -> usize {
        // Safety: `self.0` is always an array.
        unsafe { repr::capacity(&self.0) }
    }

    /// Returns the number of items currently stored in the array.
    #[must_use]
    pub fn len(&self) -> usize {
        // Safety: `self.0` is always an array.
        unsafe { repr::len(&self.0) }
    }

    /// Returns `true` if the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows a slice of [`IValue`]s from the array
    #[must_use]
    pub fn as_slice(&self) -> &[IValue] {
        // Safety: `self.0` is always an array.
        unsafe { repr::as_slice(&self.0) }
    }

    /// Borrows a mutable slice of [`IValue`]s from the array
    pub fn as_mut_slice(&mut self) -> &mut [IValue] {
        // Safety: `self.0` is always an array.
        unsafe { repr::as_mut_slice(&mut self.0) }
    }

    /// Reserves space for at least this many additional items.
    pub fn reserve(&mut self, additional: usize) {
        // Safety: `self.0` is always an array.
        unsafe { repr::reserve(&mut self.0, additional) }
    }

    /// Truncates the array by removing items until it is no longer than the specified
    /// length. The capacity is unchanged.
    pub fn truncate(&mut self, len: usize) {
        // Safety: `self.0` is always an array.
        unsafe { repr::truncate(&mut self.0, len) }
    }

    /// Removes all items from the array. The capacity is unchanged.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Inserts a new item into the array at the specified index. Any existing items
    /// on or after this index will be shifted down to accomodate this. For large
    /// arrays, insertions near the front will be slow as it will require shifting
    /// a large number of items.
    pub fn insert(&mut self, index: usize, item: impl Into<IValue>) {
        // Safety: `self.0` is always an array.
        unsafe { repr::insert(&mut self.0, index, item.into()) }
    }

    /// Removes and returns the item at the specified index from the array. Any
    /// items after this index will be shifted back up to close the gap. For large
    /// arrays, removals from near the front will be slow as it will require shifting
    /// a large number of items.
    ///
    /// If the order of the array is unimporant, consider using [`IArray::swap_remove`].
    ///
    /// If the index is outside the array bounds, `None` is returned.
    pub fn remove(&mut self, index: usize) -> Option<IValue> {
        // Safety: `self.0` is always an array.
        unsafe { repr::remove(&mut self.0, index) }
    }

    /// Removes and returns the item at the specified index from the array by
    /// first swapping it with the item currently at the end of the array, and
    /// then popping that last item.
    ///
    /// This can be more efficient than [`IArray::remove`] for large arrays,
    /// but will change the ordering of items within the array.
    ///
    /// If the index is outside the array bounds, `None` is returned.
    pub fn swap_remove(&mut self, index: usize) -> Option<IValue> {
        // Safety: `self.0` is always an array.
        unsafe { repr::swap_remove(&mut self.0, index) }
    }

    /// Pushes a new item onto the back of the array.
    pub fn push(&mut self, item: impl Into<IValue>) {
        // Safety: `self.0` is always an array.
        unsafe { repr::push(&mut self.0, item.into()) }
    }

    /// Pops the last item from the array and returns it. If the array is
    /// empty, `None` is returned.
    pub fn pop(&mut self) -> Option<IValue> {
        // Safety: `self.0` is always an array.
        unsafe { repr::pop(&mut self.0) }
    }

    /// Shrinks the memory allocation used by the array such that its
    /// capacity becomes equal to its length.
    pub fn shrink_to_fit(&mut self) {
        // Safety: `self.0` is always an array.
        unsafe { repr::shrink_to_fit(&mut self.0) }
    }
}

impl IntoIterator for IArray {
    type Item = IValue;
    type IntoIter = IntoIter;

    fn into_iter(mut self) -> Self::IntoIter {
        self.reverse();
        IntoIter {
            reversed_array: self,
        }
    }
}

impl Deref for IArray {
    type Target = [IValue];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for IArray {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Borrow<[IValue]> for IArray {
    fn borrow(&self) -> &[IValue] {
        self.as_slice()
    }
}

impl BorrowMut<[IValue]> for IArray {
    fn borrow_mut(&mut self) -> &mut [IValue] {
        self.as_mut_slice()
    }
}

impl Hash for IArray {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Safety: `self.0` is always an array.
        unsafe { repr::hash(&self.0, state) }
    }
}

impl<U: Into<IValue>> Extend<U> for IArray {
    fn extend<T: IntoIterator<Item = U>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for v in iter {
            self.push(v);
        }
    }
}

impl<U: Into<IValue>> FromIterator<U> for IArray {
    fn from_iter<T: IntoIterator<Item = U>>(iter: T) -> Self {
        let mut res = IArray::new();
        res.extend(iter);
        res
    }
}

impl AsRef<[IValue]> for IArray {
    fn as_ref(&self) -> &[IValue] {
        self.as_slice()
    }
}

impl PartialEq for IArray {
    fn eq(&self, other: &Self) -> bool {
        // Safety: `self.0` and `other.0` are always arrays.
        unsafe { repr::eq(&self.0, &other.0) }
    }
}

impl Eq for IArray {}
impl PartialOrd for IArray {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Safety: `self.0` and `other.0` are always arrays.
        unsafe { repr::cmp(&self.0, &other.0) }
    }
}

impl<I: SliceIndex<[IValue]>> Index<I> for IArray {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<I: SliceIndex<[IValue]>> IndexMut<I> for IArray {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_mut_slice(), index)
    }
}

impl Debug for IArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), f)
    }
}

impl<T: Into<IValue>> From<Vec<T>> for IArray {
    fn from(other: Vec<T>) -> Self {
        let mut res = IArray::with_capacity(other.len());
        res.extend(other.into_iter().map(Into::into));
        res
    }
}

impl<T: Into<IValue> + Clone> From<&[T]> for IArray {
    fn from(other: &[T]) -> Self {
        let mut res = IArray::with_capacity(other.len());
        res.extend(other.iter().cloned().map(Into::into));
        res
    }
}

impl<'a> IntoIterator for &'a IArray {
    type Item = &'a IValue;
    type IntoIter = std::slice::Iter<'a, IValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut IArray {
    type Item = &'a mut IValue;
    type IntoIter = std::slice::IterMut<'a, IValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl Default for IArray {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[mockalloc::test]
    fn can_create() {
        let x = IArray::new();
        let y = IArray::with_capacity(10);

        assert_eq!(x, y);
    }

    #[mockalloc::test]
    fn can_collect() {
        let x = vec![IValue::NULL, IValue::TRUE, IValue::FALSE];
        let y: IArray = x.iter().cloned().collect();

        assert_eq!(x.as_slice(), y.as_slice());
    }

    #[mockalloc::test]
    fn can_push_insert() {
        let mut x = IArray::new();
        x.insert(0, IValue::NULL);
        x.push(IValue::TRUE);
        x.insert(1, IValue::FALSE);

        assert_eq!(x.as_slice(), &[IValue::NULL, IValue::FALSE, IValue::TRUE]);
    }

    #[mockalloc::test]
    fn can_nest() {
        let x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        let y: IArray = vec![
            IValue::NULL,
            x.clone().into(),
            IValue::FALSE,
            x.clone().into(),
        ]
        .into();

        assert_eq!(&y[1], x.as_ref());
    }

    #[mockalloc::test]
    fn can_pop_remove() {
        let mut x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        assert_eq!(x.remove(1), Some(IValue::TRUE));
        assert_eq!(x.pop(), Some(IValue::FALSE));

        assert_eq!(x.as_slice(), &[IValue::NULL]);
    }

    #[mockalloc::test]
    fn can_swap_remove() {
        let mut x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        assert_eq!(x.swap_remove(0), Some(IValue::NULL));

        assert_eq!(x.as_slice(), &[IValue::FALSE, IValue::TRUE]);
    }

    #[mockalloc::test]
    fn can_index() {
        let mut x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        assert_eq!(x[1], IValue::TRUE);
        x[1] = IValue::FALSE;
        assert_eq!(x[1], IValue::FALSE);
    }

    #[mockalloc::test]
    fn can_truncate_and_shrink() {
        let mut x: IArray =
            vec![IValue::NULL, IValue::TRUE, IArray::with_capacity(10).into()].into();
        x.truncate(2);
        assert_eq!(x.len(), 2);
        assert_eq!(x.capacity(), 3);
        x.shrink_to_fit();
        assert_eq!(x.len(), 2);
        assert_eq!(x.capacity(), 2);
    }

    // Too slow for miri
    #[cfg(not(miri))]
    #[mockalloc::test]
    fn stress_test() {
        use rand::prelude::*;

        for i in 0..10 {
            // We want our test to be random but for errors to be reproducible
            let mut rng = StdRng::seed_from_u64(i);
            let mut arr = IArray::new();

            for j in 0..1000 {
                let index = rng.random_range(0..arr.len() + 1);
                if rng.random() {
                    arr.insert(index, j);
                } else {
                    arr.remove(index);
                }
            }
        }
    }
}
