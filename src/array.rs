//! Functionality relating to the JSON array type

use std::alloc::{alloc, dealloc, realloc, Layout, LayoutError};
use std::borrow::{Borrow, BorrowMut};
use std::cmp::{self, Ordering};
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::SliceIndex;

use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};
use crate::{Defrag, DefragAllocator};

use super::value::{IValue, TypeTag};

#[repr(C)]
#[repr(align(4))]
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

static EMPTY_HEADER: Header = Header { len: 0, cap: 0 };

impl IArray {
    fn layout(cap: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<usize>(cap)?)?
            .0
            .pad_to_align())
    }

    fn alloc(cap: usize) -> *mut Header {
        unsafe {
            let ptr = alloc(Self::layout(cap).unwrap()).cast::<Header>();
            ptr.write(Header { len: 0, cap });
            ptr
        }
    }

    fn realloc(ptr: *mut Header, new_cap: usize) -> *mut Header {
        unsafe {
            let old_layout = Self::layout((*ptr).cap).unwrap();
            let new_layout = Self::layout(new_cap).unwrap();
            let ptr = realloc(ptr.cast::<u8>(), old_layout, new_layout.size()).cast::<Header>();
            (*ptr).cap = new_cap;
            ptr
        }
    }

    fn dealloc(ptr: *mut Header) {
        unsafe {
            let layout = Self::layout((*ptr).cap).unwrap();
            dealloc(ptr.cast(), layout);
        }
    }

    /// Constructs a new empty `IArray`. Does not allocate.
    #[must_use]
    pub fn new() -> Self {
        unsafe { IArray(IValue::new_ref(&EMPTY_HEADER, TypeTag::ArrayOrFalse)) }
    }

    /// Constructs a new `IArray` with the specified capacity. At least that many items
    /// can be added to the array without reallocating.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        if cap == 0 {
            Self::new()
        } else {
            IArray(unsafe { IValue::new_ptr(Self::alloc(cap).cast(), TypeTag::ArrayOrFalse) })
        }
    }

    fn header(&self) -> ThinRef<Header> {
        unsafe { ThinRef::new(self.0.ptr().cast()) }
    }

    // Safety: must not be static
    unsafe fn header_mut(&mut self) -> ThinMut<Header> {
        ThinMut::new(self.0.ptr().cast())
    }

    fn is_static(&self) -> bool {
        self.capacity() == 0
    }
    /// Returns the capacity of the array. This is the maximum number of items the array
    /// can hold without reallocating.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.header().cap
    }

    /// Returns the number of items currently stored in the array.
    #[must_use]
    pub fn len(&self) -> usize {
        self.header().len
    }

    /// Returns `true` if the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows a slice of [`IValue`]s from the array
    #[must_use]
    pub fn as_slice(&self) -> &[IValue] {
        self.header().items_slice()
    }

    /// Borrows a mutable slice of [`IValue`]s from the array
    pub fn as_mut_slice(&mut self) -> &mut [IValue] {
        if self.is_static() {
            &mut []
        } else {
            unsafe { self.header_mut().items_slice_mut() }
        }
    }

    fn resize_internal(&mut self, cap: usize) {
        if self.is_static() || cap == 0 {
            *self = Self::with_capacity(cap);
        } else {
            unsafe {
                let new_ptr = Self::realloc(self.0.ptr().cast(), cap);
                self.0.set_ptr(new_ptr.cast());
            }
        }
    }

    /// Reserves space for at least this many additional items.
    pub fn reserve(&mut self, additional: usize) {
        let hd = self.header();
        let current_capacity = hd.cap;
        let desired_capacity = hd.len.checked_add(additional).unwrap();
        if current_capacity >= desired_capacity {
            return;
        }
        self.resize_internal(cmp::max(current_capacity * 2, desired_capacity.max(4)));
    }

    /// Truncates the array by removing items until it is no longer than the specified
    /// length. The capacity is unchanged.
    pub fn truncate(&mut self, len: usize) {
        if self.is_static() {
            return;
        }
        unsafe {
            let mut hd = self.header_mut();
            while hd.len > len {
                hd.pop();
            }
        }
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
        self.reserve(1);

        unsafe {
            // Safety: cannot be static after calling `reserve`
            let mut hd = self.header_mut();
            assert!(index <= hd.len);

            // Safety: We just reserved enough space for at least one extra item
            hd.push(item.into());
            if index < hd.len {
                hd.items_slice_mut()[index..].rotate_right(1);
            }
        }
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
        if index < self.len() {
            // Safety: cannot be static if index <= len
            unsafe {
                let mut hd = self.header_mut();
                hd.reborrow().items_slice_mut()[index..].rotate_left(1);
                hd.pop()
            }
        } else {
            None
        }
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
        if index < self.len() {
            // Safety: cannot be static if index <= len
            unsafe {
                let mut hd = self.header_mut();
                let last_index = hd.len - 1;
                hd.reborrow().items_slice_mut().swap(index, last_index);
                hd.pop()
            }
        } else {
            None
        }
    }

    /// Pushes a new item onto the back of the array.
    pub fn push(&mut self, item: impl Into<IValue>) {
        self.reserve(1);
        // Safety: We just reserved enough space for at least one extra item
        unsafe {
            self.header_mut().push(item.into());
        }
    }

    /// Pops the last item from the array and returns it. If the array is
    /// empty, `None` is returned.
    pub fn pop(&mut self) -> Option<IValue> {
        if self.is_static() {
            None
        } else {
            // Safety: not static
            unsafe { self.header_mut().pop() }
        }
    }

    /// Shrinks the memory allocation used by the array such that its
    /// capacity becomes equal to its length.
    pub fn shrink_to_fit(&mut self) {
        self.resize_internal(self.len());
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        let src = self.header().items_slice();
        let l = src.len();
        let mut res = Self::with_capacity(l);

        if l > 0 {
            unsafe {
                // Safety: we cannot be static if len > 0
                let mut hd = res.header_mut();
                for v in src {
                    // Safety: we reserved enough space at the start
                    hd.push(v.clone());
                }
            }
        }
        res.0
    }
    pub(crate) fn drop_impl(&mut self) {
        self.clear();
        if !self.is_static() {
            unsafe {
                Self::dealloc(self.0.ptr().cast());
                self.0.set_ref(&EMPTY_HEADER);
            }
        }
    }
}

impl<A: DefragAllocator> Defrag<A> for IArray {
    fn defrag(mut self, defrag_allocator: &mut A) -> Self {
        if self.is_static() {
            return self;
        }
        for i in 0..self.len() {
            unsafe {
                let val = self.as_ptr().add(i).read();
                let val = val.defrag(defrag_allocator);
                std::ptr::write(self.as_ptr().add(i) as *mut IValue, val);
            }
        }
        unsafe {
            let new_ptr = defrag_allocator.realloc_ptr(
                self.0.ptr(),
                Self::layout((*self.0.ptr().cast::<Header>()).cap)
                    .expect("layout is expected to return a valid value"),
            );
            self.0.set_ptr(new_ptr.cast());
        }
        self
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
        self.as_slice().hash(state);
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
        if self.0.raw_eq(&other.0) {
            true
        } else {
            self.as_slice() == other.as_slice()
        }
    }
}

impl Eq for IArray {}
impl PartialOrd for IArray {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0.raw_eq(&other.0) {
            Some(Ordering::Equal)
        } else {
            self.as_slice().partial_cmp(other.as_slice())
        }
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
                let index = rng.gen_range(0..arr.len() + 1);
                if rng.gen() {
                    arr.insert(index, j);
                } else {
                    arr.remove(index);
                }
            }
        }
    }
}
