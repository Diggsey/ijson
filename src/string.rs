use std::alloc::{alloc, dealloc, Layout, LayoutErr};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::ops::Deref;
use std::ptr::{copy_nonoverlapping, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use dashmap::{DashSet, SharedValue};
use lazy_static::lazy_static;

use super::value::{IValue, TypeTag};

#[repr(C)]
#[repr(align(4))]
struct Header {
    len: usize,
    shard_index: usize,
    rc: AtomicUsize,
}

impl Header {
    fn as_ptr(&self) -> *const u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { (self as *const Header).offset(1) as *const u8 }
    }
    fn as_bytes(&self) -> &[u8] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }
    fn as_str(&self) -> &str {
        // Safety: UTF-8 enforced on construction
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }
}

lazy_static! {
    static ref STRING_CACHE: DashSet<WeakIString> = DashSet::new();
}

struct WeakIString {
    ptr: NonNull<Header>,
}

unsafe impl Send for WeakIString {}
unsafe impl Sync for WeakIString {}
impl PartialEq for WeakIString {
    fn eq(&self, other: &Self) -> bool {
        &**self == &**other
    }
}
impl Eq for WeakIString {}
impl Hash for WeakIString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl Deref for WeakIString {
    type Target = str;
    fn deref(&self) -> &str {
        self.borrow()
    }
}

impl Borrow<str> for WeakIString {
    fn borrow(&self) -> &str {
        unsafe { self.ptr.as_ref().as_str() }
    }
}
impl WeakIString {
    fn upgrade(&self) -> IString {
        unsafe {
            self.ptr.as_ref().rc.fetch_add(1, AtomicOrdering::Relaxed);
            IString(IValue::new_ptr(
                self.ptr.as_ptr() as *mut u8,
                TypeTag::StringOrNull,
            ))
        }
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct IString(pub(crate) IValue);

value_subtype_impls!(IString, into_string, as_string, as_string_mut);

static EMPTY_HEADER: Header = Header {
    len: 0,
    shard_index: 0,
    rc: AtomicUsize::new(0),
};

impl IString {
    fn layout(len: usize) -> Result<Layout, LayoutErr> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u8>(len)?)?
            .0
            .pad_to_align())
    }

    fn alloc(s: &str, shard_index: usize) -> *mut Header {
        unsafe {
            let ptr = alloc(Self::layout(s.len()).unwrap()) as *mut Header;
            (*ptr).len = s.len();
            (*ptr).shard_index = shard_index;
            (*ptr).rc = AtomicUsize::new(0);
            copy_nonoverlapping(s.as_ptr(), (*ptr).as_ptr() as *mut u8, s.len());
            ptr
        }
    }

    fn dealloc(ptr: *mut Header) {
        unsafe {
            let layout = Self::layout((*ptr).len).unwrap();
            dealloc(ptr as *mut u8, layout);
        }
    }

    pub fn intern(s: &str) -> Self {
        let cache = &*STRING_CACHE;
        let shard_index = cache.determine_map(s);

        // Safety: `determine_map` should only return valid shard indices
        let shard = unsafe { cache.shards().get_unchecked(shard_index) };
        let mut guard = shard.write();
        if let Some((k, _)) = guard.get_key_value(s) {
            k.upgrade()
        } else {
            let k = unsafe {
                WeakIString {
                    ptr: NonNull::new_unchecked(Self::alloc(s, shard_index)),
                }
            };
            let res = k.upgrade();
            guard.insert(k, SharedValue::new(()));
            res
        }
    }

    fn header(&self) -> &Header {
        unsafe { &*(self.0.ptr() as *const Header) }
    }

    pub fn len(&self) -> usize {
        self.header().len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_str(&self) -> &str {
        self.header().as_str()
    }
    pub fn as_bytes(&self) -> &[u8] {
        self.header().as_bytes()
    }

    pub fn new() -> Self {
        unsafe { IString(IValue::new_ref(&EMPTY_HEADER, TypeTag::StringOrNull)) }
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        if self.is_empty() {
            Self::new().0
        } else {
            self.header().rc.fetch_add(1, AtomicOrdering::Relaxed);
            unsafe { self.0.raw_copy() }
        }
    }
    pub(crate) fn drop_impl(&mut self) {
        if !self.is_empty() {
            let hd = self.header();

            // If the reference count is greater than 1, we can safely decrement it without
            // locking the string cache.
            let mut rc = hd.rc.load(AtomicOrdering::Relaxed);
            while rc > 1 {
                match hd.rc.compare_exchange_weak(
                    rc,
                    rc - 1,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(new_rc) => rc = new_rc,
                }
            }

            // Slow path: we observed a reference count of 1, so we need to lock the string cache
            let cache = &*STRING_CACHE;
            // Safety: the number of shards is fixed
            let shard = unsafe { cache.shards().get_unchecked(hd.shard_index) };
            let mut guard = shard.write();
            if hd.rc.fetch_sub(1, AtomicOrdering::Relaxed) == 1 {
                // Reference count reached zero, free the string
                guard.remove(hd.as_str());
                drop(guard);

                Self::dealloc(hd as *const _ as *mut _);
            }
        }
    }
}

impl Deref for IString {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for IString {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for IString {
    fn from(other: &str) -> Self {
        Self::intern(other)
    }
}

impl From<&mut str> for IString {
    fn from(other: &mut str) -> Self {
        Self::intern(other)
    }
}

impl From<String> for IString {
    fn from(other: String) -> Self {
        Self::intern(other.as_str())
    }
}

impl From<IString> for String {
    fn from(other: IString) -> Self {
        other.as_str().into()
    }
}

impl PartialEq for IString {
    fn eq(&self, other: &Self) -> bool {
        self.0.raw_eq(&other.0)
    }
}

impl Eq for IString {}
impl Ord for IString {
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            Ordering::Equal
        } else {
            self.as_str().cmp(other.as_str())
        }
    }
}
impl PartialOrd for IString {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Hash for IString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.raw_hash(state);
    }
}

impl Debug for IString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_str(), f)
    }
}
