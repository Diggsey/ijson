//! Functionality relating to the JSON string type

use std::alloc::{Layout, LayoutError};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::Deref;
use std::ptr::{copy_nonoverlapping, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use dashmap::{DashSet, SharedValue};
use lazy_static::lazy_static;

use crate::alloc::{alloc_infallible, dealloc_infallible};
use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};

use super::value::{IValue, TypeTag, INLINE_STRING_FLAG};

/// The number of string bytes that fit inline in a pointer-sized [`IValue`].
/// This is 7 on 64-bit platforms and 3 on 32-bit platforms (one byte is used
/// for the tag, inline flag, and length).
pub(crate) const INLINE_CAPACITY: usize = std::mem::size_of::<usize>() - 1;

// The inline string control byte occupies the low byte of the value. It stores
// the `StringOrNull` tag (bits 0-1), the inline flag (bit 2, matching
// `INLINE_STRING_FLAG`), and the length (bits 3-5). The remaining bytes hold up
// to `INLINE_CAPACITY` UTF-8 bytes. `INLINE_STRING_FLAG` is a `usize`, so the
// byte-sized form is asserted equal below.
const INLINE_LEN_SHIFT: u32 = 3;
const INLINE_LEN_MASK: usize = 0b111;

// Memory offsets (within the value) of the control byte and the first character
// byte. The control byte must be the low byte of the integer value (so the tag
// and inline flag land in the low bits), which is offset 0 on little-endian and
// the top byte on big-endian. The characters follow in ascending memory order.
#[cfg(target_endian = "little")]
const INLINE_CONTROL_OFFSET: usize = 0;
#[cfg(target_endian = "little")]
const INLINE_CHAR_OFFSET: usize = 1;
#[cfg(target_endian = "big")]
const INLINE_CONTROL_OFFSET: usize = std::mem::size_of::<usize>() - 1;
#[cfg(target_endian = "big")]
const INLINE_CHAR_OFFSET: usize = 0;

// A heap string header is 8-aligned so that bit 2 of a heap string pointer is
// always clear, letting it double as the inline discriminator.
#[repr(C)]
#[repr(align(8))]
struct Header {
    rc: AtomicUsize,
    // We use 48 bits for the length and 16 bits for the shard index.
    len_lower: u32,
    len_upper: u16,
    shard_index: u16,
}

trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn len(&self) -> usize {
        (u64::from(self.len_lower) | (u64::from(self.len_upper) << 32)) as usize
    }
    fn shard_index(&self) -> usize {
        self.shard_index as usize
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
        self.header().str()
    }
}
impl WeakIString {
    fn header<'a>(&'a self) -> ThinRef<'a, Header> {
        // Safety: pointer is always valid
        unsafe { ThinRef::new(self.ptr) }
    }
    fn upgrade(&self) -> IString {
        unsafe {
            self.ptr.as_ref().rc.fetch_add(1, AtomicOrdering::Relaxed);
            IString(IValue::new_ptr(
                self.ptr.cast::<u8>(),
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
/// Very short strings (up to 7 bytes on 64-bit platforms, 3 bytes on 32-bit) are
/// stored inline within the value itself, so they require no allocation and are
/// never entered into the global cache. Equality and hashing remain correct
/// because a given string always uses exactly one representation.
///
/// Given the nature of `IString` it is better to intern a string once and reuse
/// it, rather than continually convert from `&str` to `IString`.
#[repr(transparent)]
#[derive(Clone)]
pub struct IString(pub(crate) IValue);

value_subtype_impls!(IString, into_string, as_string, as_string_mut);

impl IString {
    // Encodes a string of at most `INLINE_CAPACITY` bytes directly into the
    // pointer-sized value, avoiding any allocation or interning.
    fn new_inline(s: &str) -> Self {
        debug_assert!(s.len() <= INLINE_CAPACITY);

        // Build the payload with the tag bits left clear; `IValue::new_inline`
        // ORs in the `StringOrNull` tag. The control byte carries the inline
        // flag and the length, and the remaining bytes carry the characters.
        let mut bytes = [0u8; std::mem::size_of::<usize>()];
        bytes[INLINE_CONTROL_OFFSET] =
            INLINE_STRING_FLAG as u8 | ((s.len() as u8) << INLINE_LEN_SHIFT);
        bytes[INLINE_CHAR_OFFSET..INLINE_CHAR_OFFSET + s.len()].copy_from_slice(s.as_bytes());

        // Safety: the inline flag keeps the value non-zero, and the payload
        // leaves the tag bits clear.
        unsafe {
            IString(IValue::new_inline(
                TypeTag::StringOrNull,
                usize::from_ne_bytes(bytes),
            ))
        }
    }

    // The byte length of an inline string, read from the control byte.
    fn inline_len(&self) -> usize {
        (self.0.ptr_usize() >> INLINE_LEN_SHIFT) & INLINE_LEN_MASK
    }

    // The UTF-8 bytes of an inline string, borrowed from within `self`.
    fn inline_bytes(&self) -> &[u8] {
        let len = self.inline_len();
        // Safety: `self` is an inline string (checked by callers). Its character
        // bytes live within its own storage at `INLINE_CHAR_OFFSET`, and
        // `len <= INLINE_CAPACITY` fits within the value.
        unsafe {
            let base = (&self.0 as *const IValue).cast::<u8>();
            std::slice::from_raw_parts(base.add(INLINE_CHAR_OFFSET), len)
        }
    }

    fn layout(len: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u8>(len)?)?
            .0
            .pad_to_align())
    }

    fn alloc(s: &str, shard_index: usize) -> NonNull<Header> {
        assert!((s.len() as u64) < (1 << 48));
        assert!(shard_index < (1 << 16));
        unsafe {
            let ptr = alloc_infallible(Self::layout(s.len()).unwrap()).cast::<Header>();
            ptr.write(Header {
                len_lower: s.len() as u32,
                len_upper: ((s.len() as u64) >> 32) as u16,
                shard_index: shard_index as u16,
                rc: AtomicUsize::new(0),
            });
            let hd = ThinMut::new(ptr);
            copy_nonoverlapping(s.as_ptr(), hd.str_ptr_mut(), s.len());
            ptr
        }
    }

    fn dealloc(ptr: NonNull<Header>) {
        unsafe {
            let hd = ThinRef::new(ptr);
            let layout = Self::layout(hd.len()).unwrap();
            dealloc_infallible(ptr.cast::<u8>(), layout);
        }
    }

    /// Converts a `&str` to an `IString`.
    ///
    /// Very short strings (up to 7 bytes on 64-bit platforms, 3 bytes on 32-bit)
    /// are stored inline in the value itself, without allocating or touching the
    /// global cache. Longer strings are interned in the global string cache.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        if s.len() <= INLINE_CAPACITY {
            return Self::new_inline(s);
        }
        let cache = &*STRING_CACHE;
        let shard_index = cache.determine_map(s);

        // Safety: `determine_map` should only return valid shard indices
        let shard = unsafe { cache.shards().get_unchecked(shard_index) };
        let mut guard = shard.write();
        if let Some((k, _)) = guard.get_key_value(s) {
            k.upgrade()
        } else {
            let k = WeakIString {
                ptr: Self::alloc(s, shard_index),
            };
            let res = k.upgrade();
            guard.insert(k, SharedValue::new(()));
            res
        }
    }

    // Safety: must only be called on a heap (interned) string, not an inline one.
    fn header<'a>(&'a self) -> ThinRef<'a, Header> {
        debug_assert!(!self.0.is_inline_string());
        unsafe { ThinRef::new(self.0.ptr().cast()) }
    }

    /// Returns the length (in bytes) of this string.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.0.is_inline_string() {
            self.inline_len()
        } else {
            self.header().len()
        }
    }

    /// Returns `true` if this is the empty string "".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Obtains a `&str` from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // Safety: inline and heap string bytes are both valid UTF-8 by construction.
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// Obtains a byte slice from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        if self.0.is_inline_string() {
            self.inline_bytes()
        } else {
            self.header().bytes()
        }
    }

    /// Returns the empty string.
    #[must_use]
    pub fn new() -> Self {
        Self::new_inline("")
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        // Inline strings are trivially copyable; only heap strings are refcounted.
        if !self.0.is_inline_string() {
            self.header().rc.fetch_add(1, AtomicOrdering::Relaxed);
        }
        unsafe { self.0.raw_copy() }
    }
    pub(crate) fn drop_impl(&mut self) {
        // Inline strings own no allocation, so there is nothing to free.
        if self.0.is_inline_string() {
            return;
        }
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
            assert!(guard.remove(hd.str()).is_some());

            // Shrink the shard if it's mostly empty.
            // The second condition is necessary because `HashMap` sometimes
            // reports a capacity of zero even when it's still backed by an
            // allocation.
            if guard.len() * 3 < guard.capacity() || guard.is_empty() {
                guard.shrink_to_fit();
            }
            drop(guard);

            Self::dealloc(unsafe { self.0.ptr().cast() });
        }
    }
}

impl Deref for IString {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

/// Implement Borrow is incorrect:
/// > In particular Eq, Ord and Hash must be equivalent for borrowed and owned values: x.borrow() == y.borrow()
/// > should give the same result as x == y.
///
/// While Eq and Ord are equivalent, Hash is not, since the hash of an `IString` is the hash of its pointer,
/// while the hash of a `&str` is the hash of its contents. This can lead to surprising behavior when using
/// `IString` as keys in a `HashMap` or `HashSet`, since lookups with `&str` will not find the corresponding
/// `IString` key.
///
/// Only enable this feature as a temporary compatibility measure for libraries that require `Borrow<str>`
/// to be implemented, and be aware of the potential pitfalls when using `IString` as keys in hash-based
/// collections.    
#[cfg(feature = "broken-borrow-impl-compat")]
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

impl Display for IString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.as_str(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A string long enough to always be heap-interned on any platform.
    const LONG_A: &str = "interned_string_a";
    const LONG_B: &str = "interned_string_b";

    #[mockalloc::test]
    fn can_intern() {
        // Long strings share a single interned allocation (pointer identity).
        let x = IString::intern(LONG_A);
        let y = IString::intern(LONG_B);
        let z = IString::intern(LONG_A);

        assert_eq!(x.as_ptr(), z.as_ptr());
        assert_ne!(x.as_ptr(), y.as_ptr());
        assert_eq!(x.as_str(), LONG_A);
        assert_eq!(y.as_str(), LONG_B);
    }

    #[mockalloc::test]
    fn default_interns_string() {
        let x = IString::intern(LONG_A);
        let y = IString::intern(LONG_A);

        assert_eq!(x.as_ptr(), y.as_ptr());
    }

    #[mockalloc::test]
    fn short_strings_are_inline() {
        // Strings up to INLINE_CAPACITY bytes are stored inline: no allocation,
        // and they still compare equal by value.
        let mut cases: Vec<String> = vec![
            String::new(),
            "a".into(),
            "no".into(),
            "yes".into(),
            "é".into(), // 2-byte UTF-8, fits on 32-bit and 64-bit
            "x".repeat(INLINE_CAPACITY),
        ];
        cases.dedup();

        for s in &cases {
            let a = IString::intern(s);
            let b = IString::intern(s);

            assert!(a.0.is_inline_string(), "{:?} should be inline", s);
            assert_eq!(a.as_str(), s.as_str());
            assert_eq!(a.as_bytes(), s.as_bytes());
            assert_eq!(a.len(), s.len());
            assert_eq!(a.is_empty(), s.is_empty());
            assert_eq!(a, b);
            // Value equality holds even though inline strings are not deduplicated
            // to a shared allocation.
            assert_eq!(a, IString::from(s.as_str()));
        }
    }

    #[mockalloc::test]
    fn inline_heap_boundary() {
        // At the capacity boundary, one side is inline and the other is heap.
        let inline = "x".repeat(INLINE_CAPACITY);
        let heap = "x".repeat(INLINE_CAPACITY + 1);

        let a = IString::intern(&inline);
        let b = IString::intern(&heap);

        assert!(a.0.is_inline_string());
        assert!(!b.0.is_inline_string());
        assert_eq!(a.as_str(), inline);
        assert_eq!(b.as_str(), heap);
        assert_ne!(a, b);
        assert!(a < b); // ordering falls back to byte comparison
    }

    // Not a `mockalloc::test`: an all-inline test performs no allocations, which
    // is exactly the point, but `mockalloc` treats zero allocations as an error.
    #[test]
    fn empty_string_is_inline() {
        let a = IString::new();
        let b = IString::intern("");

        assert!(a.0.is_inline_string());
        assert_eq!(a, b);
        assert!(a.is_empty());
        assert_eq!(a.as_str(), "");
        // Empty is a string, not null.
        assert!(!IValue::from(a).is_null());
    }

    #[mockalloc::test]
    fn inline_and_heap_mix_in_object() {
        // Short (inline) and long (heap) keys must both hash and look up correctly.
        let mut obj = crate::IObject::new();
        obj.insert("id", IValue::from(1));
        obj.insert(LONG_A, IValue::from(2));
        obj.insert("no", IValue::from(3));

        assert_eq!(obj["id"], IValue::from(1));
        assert_eq!(obj[LONG_A], IValue::from(2));
        assert_eq!(obj["no"], IValue::from(3));
        assert_eq!(obj.len(), 3);
    }
}
