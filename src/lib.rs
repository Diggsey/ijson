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
#![deny(missing_docs, missing_debug_implementations)]

#[macro_use]
mod macros;

pub mod array;
pub mod number;
pub mod object;
pub mod string;
mod value;

pub use array::IArray;
pub use number::INumber;
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
mod tests {
    use mockalloc::Mockalloc;
    use std::alloc::System;

    #[global_allocator]
    static ALLOCATOR: Mockalloc<System> = Mockalloc(System);
}
