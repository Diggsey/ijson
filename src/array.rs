use std::alloc::{alloc, dealloc, realloc, Layout, LayoutErr};
use std::borrow::{Borrow, BorrowMut};
use std::cmp::{self, Ordering};
use std::hash::Hash;
use std::iter::FromIterator;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

use super::value::{IValue, TypeTag};

#[repr(C)]
#[repr(align(4))]
struct Header {
    len: usize,
    cap: usize,
}

impl Header {
    fn as_ptr(&self) -> *const IValue {
        // Safety: pointers to the end of structs are allowed
        unsafe { (self as *const Header).offset(1) as *const IValue }
    }
    fn as_slice(&self) -> &[IValue] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }
    fn as_mut_slice(&mut self) -> &mut [IValue] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts_mut(self.as_ptr() as *mut _, self.len) }
    }
    fn as_mut_uninit_slice(&self) -> &mut [MaybeUninit<IValue>] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts_mut(self.as_ptr() as *mut _, self.cap) }
    }
    // Safety: Space must already be allocated for the item
    unsafe fn push(&mut self, item: IValue) {
        self.as_mut_uninit_slice()
            .get_unchecked_mut(self.len)
            .as_mut_ptr()
            .write(item);
        self.len += 1;
    }
    fn pop(&mut self) -> Option<IValue> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;

            // Safety: We just checked that an item exists
            unsafe {
                Some(
                    self.as_mut_uninit_slice()
                        .get_unchecked_mut(self.len)
                        .as_mut_ptr()
                        .read(),
                )
            }
        }
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct IArray(IValue);

static EMPTY_HEADER: Header = Header { len: 0, cap: 0 };

impl IArray {
    fn layout(cap: usize) -> Result<Layout, LayoutErr> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<usize>(cap)?)?
            .0
            .pad_to_align())
    }

    fn alloc(cap: usize) -> *mut u8 {
        unsafe {
            let ptr = alloc(Self::layout(cap).unwrap()) as *mut Header;
            (*ptr).len = 0;
            (*ptr).cap = cap;
            ptr as *mut u8
        }
    }

    fn realloc(ptr: *mut u8, new_cap: usize) -> *mut u8 {
        unsafe {
            let old_layout = Self::layout((*(ptr as *const Header)).cap).unwrap();
            let new_layout = Self::layout(new_cap).unwrap();
            let ptr = realloc(ptr as *mut u8, old_layout, new_layout.size());
            (*(ptr as *mut Header)).cap = new_cap;
            ptr
        }
    }

    fn dealloc(ptr: *mut u8) {
        unsafe {
            let layout = Self::layout((*(ptr as *const Header)).cap).unwrap();
            dealloc(ptr, layout);
        }
    }

    pub fn new() -> Self {
        unsafe { IArray(IValue::new_ref(&EMPTY_HEADER, TypeTag::ArrayOrFalse)) }
    }

    pub fn with_capacity(cap: usize) -> Self {
        if cap == 0 {
            Self::new()
        } else {
            IArray(unsafe { IValue::new_ptr(Self::alloc(cap), TypeTag::ArrayOrFalse) })
        }
    }

    fn header(&self) -> &Header {
        unsafe { &*(self.0.ptr() as *const Header) }
    }

    fn header_mut(&mut self) -> &mut Header {
        unsafe { &mut *(self.0.ptr() as *mut Header) }
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
    pub fn as_slice(&self) -> &[IValue] {
        self.header().as_slice()
    }
    pub fn as_mut_slice(&mut self) -> &mut [IValue] {
        self.header_mut().as_mut_slice()
    }
    pub fn reserve(&mut self, additional: usize) {
        let hd = self.header();
        let current_capacity = hd.cap;
        let desired_capacity = hd.len.checked_add(additional).unwrap();
        if current_capacity >= desired_capacity {
            return;
        }
        if self.is_static() {
            *self = Self::with_capacity(desired_capacity);
        } else {
            unsafe {
                let new_ptr = Self::realloc(
                    self.0.ptr(),
                    cmp::max(current_capacity * 2, desired_capacity),
                );
                self.0.set_ptr(new_ptr);
            }
        }
    }
    pub fn push(&mut self, item: IValue) {
        self.reserve(1);
        // Safety: We just reserved enough space for at least one extra item
        unsafe {
            self.header_mut().push(item);
        }
    }
    pub fn pop(&mut self) -> Option<IValue> {
        self.header_mut().pop()
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        let src = self.header().as_slice();
        let mut res = Self::with_capacity(src.len());
        let hd = res.header_mut();
        for v in src {
            // Safety: we reserved enough space at the start
            unsafe {
                hd.push(v.clone());
            }
        }
        res.0
    }
    pub(crate) fn drop_impl(&mut self) {
        if !self.is_static() {
            unsafe {
                Self::dealloc(self.0.ptr());
                self.0.set_ref(&EMPTY_HEADER);
            }
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

impl Extend<IValue> for IArray {
    fn extend<T: IntoIterator<Item = IValue>>(&mut self, iter: T) {
        for v in iter {
            self.push(v);
        }
    }
}

impl FromIterator<IValue> for IArray {
    fn from_iter<T: IntoIterator<Item = IValue>>(iter: T) -> Self {
        let mut res = IArray::new();
        res.extend(iter);
        res
    }
}

impl AsRef<IValue> for IArray {
    fn as_ref(&self) -> &IValue {
        &self.0
    }
}

impl PartialEq for IArray {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for IArray {}
impl Ord for IArray {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
impl PartialOrd for IArray {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
