//! Functionality relating to the JSON string type.
//!
//! `IString` is a *type* that spans two representations — an inline short string
//! (see [`crate::inline::string`]) and a heap interned string (see
//! [`crate::interned`]). The string-specific logic lives as methods on
//! [`IValue`] which pick the representation by tag; `IString` is a thin wrapper
//! that delegates up to them.

use std::cmp::Ordering;
use std::fmt::{self, Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::Deref;

use crate::inline::string as inl;
use crate::interned;
use crate::value::IValue;

#[doc(hidden)]
pub fn init_cache() {
    interned::init_cache();
}

/// String-type methods on [`IValue`], spanning the inline and interned
/// representations. Each assumes the value is a string.
impl IValue {
    pub(crate) fn new_string(s: &str) -> Self {
        if s.len() <= inl::CAPACITY {
            inl::encode(s)
        } else {
            interned::intern(s)
        }
    }

    pub(crate) fn string_len(&self) -> usize {
        if self.is_inline_string() {
            self.inline_string_len()
        } else {
            // Safety: not an inline string, so it is interned.
            unsafe { self.interned_len() }
        }
    }

    pub(crate) fn string_bytes(&self) -> &[u8] {
        if self.is_inline_string() {
            self.inline_string_bytes()
        } else {
            // Safety: not an inline string, so it is interned.
            unsafe { self.interned_bytes() }
        }
    }

    pub(crate) fn string_as_str(&self) -> &str {
        // Safety: inline and interned string bytes are both valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.string_bytes()) }
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
    /// Converts a `&str` to an `IString`.
    ///
    /// Very short strings (up to 7 bytes on 64-bit platforms, 3 bytes on 32-bit)
    /// are stored inline in the value itself, without allocating or touching the
    /// global cache. Longer strings are interned in the global string cache.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        IString(IValue::new_string(s))
    }

    /// Returns the length (in bytes) of this string.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.string_len()
    }

    /// Returns `true` if this is the empty string "".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Obtains a `&str` from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.string_as_str()
    }

    /// Obtains a byte slice from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.string_bytes()
    }

    /// Returns the empty string.
    #[must_use]
    pub fn new() -> Self {
        Self::intern("")
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
impl std::borrow::Borrow<str> for IString {
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
        // Strings up to the inline capacity are stored inline: no allocation,
        // and they still compare equal by value.
        let mut cases: Vec<String> = vec![
            String::new(),
            "a".into(),
            "no".into(),
            "yes".into(),
            "é".into(), // 2-byte UTF-8, fits on 32-bit and 64-bit
            "x".repeat(inl::CAPACITY),
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
        let inline = "x".repeat(inl::CAPACITY);
        let heap = "x".repeat(inl::CAPACITY + 1);

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
