//! This crate offers a replacement for `serde-json`'s `Value` type, which is
//! significantly more memory efficient.
//!
//! As a ballpark figure, it will typically use half as much memory as
//! `serde-json` when deserializing a value and the memory footprint of cloning
//! a value is more than 7x smaller.
//!
//! The primary type exposed by this crate is the [`IValue`] type. It is guaranteed
//! to be pointer-sized and has a niche (so `Option<IValue>` is also guaranteed
//! to be pointer-sized).
//!
//! Cargo features:
//!
//! - `ctor`
//!   A global string cache is used when interning strings. This cache is normally
//!   initialized lazily on first use. Enabling the `ctor` feature will cause it
//!   to be eagerly initialized on startup.
//!   There is no performance benefit to this, but it can help avoid false positives
//!   from tools like `mockalloc` which try to detect memory leaks during tests.
#![deny(missing_docs, missing_debug_implementations)]

#[macro_use]
mod macros;

pub mod array;
pub mod number;
pub mod object;

#[cfg(feature = "thread_safe")]
pub mod string;

use std::alloc::Layout;

#[cfg(feature = "thread_safe")]
pub use string::IString;

#[cfg(not(feature = "thread_safe"))]
pub mod unsafe_string;
#[cfg(not(feature = "thread_safe"))]
pub use unsafe_string::IString;

mod thin;
mod value;

pub use array::IArray;
pub use number::INumber;
pub use object::IObject;

pub use value::{
    BoolMut, Destructured, DestructuredMut, DestructuredRef, IValue, ValueIndex, ValueType,
};

mod de;
mod ser;
pub use de::from_value;
pub use ser::to_value;

/// Trait to implement defrag allocator
pub trait DefragAllocator {
    /// Gets a pointer and return an new pointer that points to a copy
    /// of the exact same data. The old pointer should not be used anymore.
    unsafe fn realloc_ptr<T>(&mut self, ptr: *mut T, layout: Layout) -> *mut T;

    /// Allocate memory for defrag
    unsafe fn alloc(&mut self, layout: Layout) -> *mut u8;

    /// Free memory for defrag
    unsafe fn free<T>(&mut self, ptr: *mut T, layout: Layout);
}

/// Trait for object that implements defrag
pub trait Defrag<A: DefragAllocator> {
    /// Defrag implementation
    fn defrag(self, defrag_allocator: &mut A) -> Self;
}
/// Reinitialized the shared strings cache.
/// Any json that still uses a shared string will continue using it.
/// But new strings will be reinitialized instead of reused the old ones.
pub fn reinit_shared_string_cache() {
    unsafe_string::reinit_cache();
}

#[cfg(all(test, not(miri)))]
mod tests {
    use mockalloc::Mockalloc;
    use std::alloc::System;

    #[global_allocator]
    static ALLOCATOR: Mockalloc<System> = Mockalloc(System);
}
