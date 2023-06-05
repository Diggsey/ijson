//! Functionality relating to the JSON string type

use hashbrown::HashSet;
use std::alloc::{alloc, dealloc, Layout, LayoutError};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::ops::Deref;
use std::ptr::{copy_nonoverlapping, NonNull};

use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};

use super::value::{IValue, TypeTag};

#[repr(C)]
#[repr(align(4))]
struct Header {
    rc: usize,
    // We use 48 bits for the length.
    len_lower: u32,
    len_upper: u16,
}

trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn len(&self) -> usize {
        (u64::from(self.len_lower) | (u64::from(self.len_upper) << 32)) as usize
    }
    fn str_ptr(&self) -> *const u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr().add(1).cast() }
    }
    fn bytes(&self) -> &'a [u8] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts(self.str_ptr(), self.len()) }
    }
    fn str(&self) -> &'a str {
        // Safety: UTF-8 enforced on construction
        unsafe { std::str::from_utf8_unchecked(self.bytes()) }
    }
}

trait HeaderMut<'a>: ThinMutExt<'a, Header> {
    fn str_ptr_mut(mut self) -> *mut u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr_mut().add(1).cast() }
    }
}

impl<'a, T: ThinRefExt<'a, Header>> HeaderRef<'a> for T {}
impl<'a, T: ThinMutExt<'a, Header>> HeaderMut<'a> for T {}

static mut STRING_CACHE: Option<HashSet<WeakIString>> = None;

// Eagerly initialize the string cache during tests or when the
// `ctor` feature is enabled.
#[cfg(any(test, feature = "ctor"))]
#[ctor::ctor]
unsafe fn ctor_init_cache() {
    if STRING_CACHE.is_none() {
        STRING_CACHE = Some(HashSet::new());
    };
}

#[doc(hidden)]
pub fn init_cache() {
    unsafe {
        if STRING_CACHE.is_none() {
            STRING_CACHE = Some(HashSet::new());
        };
    }
}

#[cfg(not(feature = "ctor"))]
unsafe fn get_cache() -> &'static mut HashSet<WeakIString> {
    init_cache();
    STRING_CACHE.as_mut().unwrap()
}

#[cfg(feature = "ctor")]
unsafe fn get_cache() -> &'static mut HashSet<WeakIString> {
    STRING_CACHE.as_mut().unwrap()
}

struct WeakIString {
    ptr: NonNull<Header>,
}

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
        self.header().str()
    }
}
impl WeakIString {
    fn header(&self) -> ThinMut<Header> {
        // Safety: pointer is always valid
        unsafe { ThinMut::new(self.ptr.as_ptr()) }
    }
    fn upgrade(&self) -> IString {
        unsafe {
            self.header().rc += 1;
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
    rc: 0,
};

impl IString {
    fn layout(len: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u8>(len)?)?
            .0
            .pad_to_align())
    }

    fn alloc(s: &str) -> *mut Header {
        assert!((s.len() as u64) < (1 << 48));
        unsafe {
            let ptr = alloc(Self::layout(s.len()).unwrap()).cast::<Header>();
            ptr.write(Header {
                len_lower: s.len() as u32,
                len_upper: ((s.len() as u64) >> 32) as u16,
                rc: 0,
            });
            let hd = ThinMut::new(ptr);
            copy_nonoverlapping(s.as_ptr(), hd.str_ptr_mut(), s.len());
            ptr
        }
    }

    fn dealloc(ptr: *mut Header) {
        unsafe {
            let hd = ThinRef::new(ptr);
            let layout = Self::layout(hd.len()).unwrap();
            dealloc(ptr.cast::<u8>(), layout);
        }
    }

    /// Converts a `&str` to an `IString` by interning it in the global string cache.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        if s.is_empty() {
            return Self::new();
        }

        unsafe {
            let cache = get_cache();

            let k = cache.get_or_insert_with(s, |s| WeakIString {
                ptr: NonNull::new_unchecked(Self::alloc(s)),
            });
            k.upgrade()
        }
    }

    fn header(&self) -> ThinMut<Header> {
        unsafe { ThinMut::new(self.0.ptr().cast()) }
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
        self.header().str()
    }

    /// Obtains a byte slice from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.header().bytes()
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
            self.header().rc += 1;
            unsafe { self.0.raw_copy() }
        }
    }
    pub(crate) fn drop_impl(&mut self) {
        if !self.is_empty() {
            let mut hd = self.header();
            hd.rc -= 1;
            if hd.rc == 0 {
                // Reference count reached zero, free the string
                unsafe {
                    let cache = get_cache();
                    cache.remove(hd.str());

                    // Shrink the cache if it is empty in tests to verify no memory leaks
                    #[cfg(test)]
                    if cache.is_empty() {
                        cache.shrink_to_fit();
                    }
                }
                Self::dealloc(unsafe { self.0.ptr().cast() });
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
