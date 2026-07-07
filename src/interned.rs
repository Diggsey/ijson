//! The heap interned-string representation (tag `String`).
//!
//! Strings too long to store inline are interned in a global, reference-counted
//! cache so that equal strings share one allocation and compare by pointer.
//! Interning uses `DashSet`, a concurrent hash-set, so many strings can be
//! interned at once without contention. The header is 8-aligned so the tag bits
//! stay free.

use std::alloc::{Layout, LayoutError};
use std::borrow::Borrow;
use std::hash::Hash;
use std::ops::Deref;
use std::ptr::{copy_nonoverlapping, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use dashmap::{DashSet, SharedValue};
use lazy_static::lazy_static;

use crate::alloc::{alloc_infallible, dealloc_infallible};
use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};
use crate::value::{IValue, TypeTag};

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

pub(crate) fn init_cache() {
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
    fn upgrade(&self) -> IValue {
        unsafe {
            self.ptr.as_ref().rc.fetch_add(1, AtomicOrdering::Relaxed);
            IValue::new_ptr(self.ptr.cast::<u8>(), TypeTag::String)
        }
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
        let ptr = alloc_infallible(layout(s.len()).unwrap()).cast::<Header>();
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
        let layout = layout(hd.len()).unwrap();
        dealloc_infallible(ptr.cast::<u8>(), layout);
    }
}

/// Interns a string in the global cache, returning a heap string value.
pub(crate) fn intern(s: &str) -> IValue {
    let cache = &*STRING_CACHE;
    let shard_index = cache.determine_map(s);

    // Safety: `determine_map` should only return valid shard indices
    let shard = unsafe { cache.shards().get_unchecked(shard_index) };
    let mut guard = shard.write();
    if let Some((k, _)) = guard.get_key_value(s) {
        k.upgrade()
    } else {
        let k = WeakIString {
            ptr: alloc(s, shard_index),
        };
        let res = k.upgrade();
        guard.insert(k, SharedValue::new(()));
        res
    }
}

impl IValue {
    // Safety: must be a heap (interned) string.
    fn interned_header<'a>(&'a self) -> ThinRef<'a, Header> {
        unsafe { ThinRef::new(self.ptr().cast()) }
    }

    /// The byte length of a heap string. Safety: must be a heap string.
    pub(crate) unsafe fn interned_len(&self) -> usize {
        self.interned_header().len()
    }

    /// The UTF-8 bytes of a heap string. Safety: must be a heap string.
    pub(crate) unsafe fn interned_bytes(&self) -> &[u8] {
        self.interned_header().bytes()
    }

    /// Clones a heap string by bumping its reference count. Safety: must be a
    /// heap string.
    pub(crate) unsafe fn interned_clone(&self) -> IValue {
        self.interned_header()
            .rc
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.raw_copy()
    }

    /// Drops a heap string, freeing it when the reference count reaches zero.
    /// Safety: must be a heap string.
    pub(crate) unsafe fn interned_drop(&mut self) {
        let hd = self.interned_header();

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
        let shard = cache.shards().get_unchecked(hd.shard_index());
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

            dealloc(self.ptr().cast());
        }
    }
}
