//! Functionality relating to the JSON string type

use std::alloc::{alloc, dealloc, Layout, LayoutError};
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
    rc: AtomicUsize,
    // We use 48 bits for the length and 16 bits for the shard index.
    len_lower: u32,
    len_upper: u16,
    shard_index: u16,
}

impl Header {
    fn len(&self) -> usize {
        (u64::from(self.len_lower) | (u64::from(self.len_upper) << 32)) as usize
    }
    fn shard_index(&self) -> usize {
        self.shard_index as usize
    }
    fn as_ptr(&self) -> *const u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { (self as *const Header).add(1) as *const u8 }
    }
    fn as_bytes(&self) -> &[u8] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
    fn as_str(&self) -> &str {
        // Safety: UTF-8 enforced on construction
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }
}

lazy_static! {
    static ref STRING_CACHE: DashSet<WeakIString> = DashSet::new();
}

// Eagerly initialize the string cache during tests or when the
// `ctor` feature is enabled.
#[cfg(any(test, feature = "ctor"))]
#[ctor::ctor]
fn ctor_init_cache() {
    lazy_static::initialize(&STRING_CACHE);
}

#[doc(hidden)]
pub fn init_cache() {
    lazy_static::initialize(&STRING_CACHE);
}

struct WeakIString {
    ptr: NonNull<Header>,
}

unsafe impl Send for WeakIString {}
unsafe impl Sync for WeakIString {}
impl PartialEq for WeakIString {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for WeakIString {}
impl Hash for WeakIString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state);
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
                self.ptr.as_ptr().cast::<u8>(),
                TypeTag::StringOrNull,
            ))
        }
    }
}

/// The `IString` type is an interned, immutable string, and is where this crate
/// gets its name.
///
/// Cloning an `IString` is cheap, and it can be easily converted from `&str` or
/// `String` types. Comparisons between `IString`s is a simple pointer
/// comparison.
///
/// The memory backing an `IString` is reference counted, so that unlike many
/// string interning libraries, memory is not leaked as new strings are interned.
/// Interning uses `DashSet`, an implementation of a concurrent hash-set, allowing
/// many strings to be interned concurrently without becoming a bottleneck.
///
/// Given the nature of `IString` it is better to intern a string once and reuse
/// it, rather than continually convert from `&str` to `IString`.
#[repr(transparent)]
#[derive(Clone)]
pub struct IString(pub(crate) IValue);

value_subtype_impls!(IString, into_string, as_string, as_string_mut);

static EMPTY_HEADER: Header = Header {
    len_lower: 0,
    len_upper: 0,
    shard_index: 0,
    rc: AtomicUsize::new(0),
};

impl IString {
    fn layout(len: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u8>(len)?)?
            .0
            .pad_to_align())
    }

    fn alloc(s: &str, shard_index: usize) -> *mut Header {
        assert!((s.len() as u64) < (1 << 48));
        assert!(shard_index < (1 << 16));
        unsafe {
            let ptr = alloc(Self::layout(s.len()).unwrap()).cast::<Header>();
            (*ptr).len_lower = s.len() as u32;
            (*ptr).len_upper = ((s.len() as u64) >> 32) as u16;
            (*ptr).shard_index = shard_index as u16;
            (*ptr).rc = AtomicUsize::new(0);
            copy_nonoverlapping(s.as_ptr(), (*ptr).as_ptr() as *mut u8, s.len());
            ptr
        }
    }

    fn dealloc(ptr: *mut Header) {
        unsafe {
            let layout = Self::layout((*ptr).len()).unwrap();
            dealloc(ptr.cast::<u8>(), layout);
        }
    }

    /// Converts a `&str` to an `IString` by interning it in the global string cache.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        if s.is_empty() {
            return Self::new();
        }
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

    /// Returns the length (in bytes) of this string.
    #[must_use]
    pub fn len(&self) -> usize {
        self.header().len()
    }

    /// Returns `true` if this is the empty string "".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Obtains a `&str` from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.header().as_str()
    }

    /// Obtains a byte slice from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.header().as_bytes()
    }

    /// Returns the empty string.
    #[must_use]
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
            let shard = unsafe { cache.shards().get_unchecked(hd.shard_index()) };
            let mut guard = shard.write();
            if hd.rc.fetch_sub(1, AtomicOrdering::Relaxed) == 1 {
                // Reference count reached zero, free the string
                assert!(guard.remove(hd.as_str()).is_some());

                // Shrink the shard if it's mostly empty.
                // The second condition is necessary because `HashMap` sometimes
                // reports a capacity of zero even when it's still backed by an
                // allocation.
                if guard.len() * 3 < guard.capacity() || guard.is_empty() {
                    guard.shrink_to_fit();
                }
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

impl From<&String> for IString {
    fn from(other: &String) -> Self {
        Self::intern(other.as_str())
    }
}

impl From<&mut String> for IString {
    fn from(other: &mut String) -> Self {
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

impl PartialEq<str> for IString {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<IString> for str {
    fn eq(&self, other: &IString) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<String> for IString {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<IString> for String {
    fn eq(&self, other: &IString) -> bool {
        self == other.as_str()
    }
}

impl Default for IString {
    fn default() -> Self {
        Self::new()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[mockalloc::test]
    fn can_intern() {
        let x = IString::intern("foo");
        let y = IString::intern("bar");
        let z = IString::intern("foo");

        assert_eq!(x.as_ptr(), z.as_ptr());
        assert_ne!(x.as_ptr(), y.as_ptr());
        assert_eq!(x.as_str(), "foo");
        assert_eq!(y.as_str(), "bar");
    }

    #[mockalloc::test]
    fn default_interns_string() {
        let x = IString::intern("");
        let y = IString::new();
        let z = IString::intern("foo");

        assert_eq!(x.as_ptr(), y.as_ptr());
        assert_ne!(x.as_ptr(), z.as_ptr());
    }
}
