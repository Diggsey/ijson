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
//! - `arbitrary_precision`
//!   Store JSON numbers as their exact decimal value rather than rounding to `f64`. A
//!   short decimal is packed inline; a larger or more precise one spills to a heap
//!   arbitrary-precision decimal. With this on, `"0.1"` is the exact tenth — a *different*
//!   number from the `f64` `0.1` — and a magnitude beyond `f64`'s range is representable.
//!   It also switches deserialization to preserve the exact literal (via serde_json's own
//!   `arbitrary_precision`). Off by default; without it every number is exactly an `f64`.
//!
//! - `ctor`
//!   A global string cache is used when interning strings. This cache is normally
//!   initialized lazily on first use. Enabling the `ctor` feature will cause it
//!   to be eagerly initialized on startup.
//!   There is no performance benefit to this, but it can help avoid false positives
//!   from tools like `mockalloc` which try to detect memory leaks during tests.
//!
//! - `indexmap`
//!   Adds conversions between [`IObject`] and `indexmap`'s `IndexMap`.
//!
//! - `broken-borrow-impl-compat`
//!   Adds `Borrow<str>` for [`IString`], for libraries that require it. Unsound to rely on
//!   for hash-map keys (see the impl's own note); enable only as a temporary measure.
//!
//! - `tracing`
//!   Forwards to `mockalloc`'s `tracing` feature, for allocation tracing under tests.
#![deny(missing_docs, missing_debug_implementations)]

#[macro_use]
mod macros;

mod alloc;
pub mod array;
pub mod number;
pub mod object;
pub mod string;
mod thin;
mod value;

#[cfg(codegen_probes)]
pub use value::codegen_probes;

pub use array::IArray;
pub use number::{INumber, ParseNumberError};
pub use object::IObject;
pub use string::IString;
pub use value::{
    BoolMut, Destructured, DestructuredMut, DestructuredRef, IValue, ValueIndex, ValueType,
};

mod de;
mod ser;
pub use de::from_value;
pub use ser::to_value;

#[cfg(test)]
mod numeric_edge_cases;

#[cfg(all(test, not(miri)))]
mod tests {
    use mockalloc::Mockalloc;
    use std::alloc::System;

    #[global_allocator]
    static ALLOCATOR: Mockalloc<System> = Mockalloc(System);
}
