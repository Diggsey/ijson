//! The heap scalar-number representations: an 8-byte payload behind the
//! `NumberI64` / `NumberU64` / `NumberF64` tags.
//!
//! These are three separate representations — one per tag, in [`int`] (`I64Repr`),
//! [`uint`] (`U64Repr`), and [`float`] (`F64Repr`) — not one type that re-inspects
//! the tag: the tag alone determines the kind, so each owns its own decode and
//! construction, storing and reading its payload as its *actual* type. They share
//! only the typed allocation helpers here, which operate on the raw (aligned)
//! allocation pointer; applying and stripping the tag is the caller's (`IValue`'s)
//! responsibility.

mod float;
mod int;
mod uint;

pub(crate) use float::F64Repr;
pub(crate) use int::I64Repr;
pub(crate) use uint::U64Repr;

use std::alloc::Layout;
use std::ptr::NonNull;

use crate::alloc::{alloc_infallible, dealloc_infallible};

/// The heap layout for a scalar payload of type `T` (an `i64`/`u64`/`f64`): `T`'s
/// own size, but forced to at least 8-byte alignment so the low tag bits of the
/// pointer stay free regardless of `T`'s natural alignment.
fn layout<T>() -> Layout {
    Layout::new::<T>()
        .align_to(8)
        .expect("a scalar payload has a valid layout")
}

/// Allocates a heap scalar holding `value`, returning the aligned allocation.
fn alloc<T>(value: T) -> NonNull<u8> {
    // Safety: freshly allocated to `layout::<T>()`, non-null, and aligned for `T`.
    unsafe {
        let ptr = alloc_infallible(layout::<T>()).cast::<T>();
        ptr.as_ptr().write(value);
        ptr.cast()
    }
}

/// Reads the payload as a `T`. Safety: `ptr` must be a live scalar whose 8 bytes
/// are a valid `T` (each representation reads back the type it stored; the raw bits
/// may also be read as `u64`).
///
/// `pub(super)`, not `pub(crate)`: the scalar reps that use it are child modules (which see
/// it regardless), and the only other caller is `IValue::number_repr_key`, a test in the
/// parent `value` module. Its siblings `alloc`/`free`/`layout` are fully private.
pub(super) unsafe fn read<T>(ptr: NonNull<u8>) -> T {
    ptr.cast::<T>().as_ptr().read()
}

/// Frees a scalar allocation of a `T` payload.
/// Safety: `ptr` must be a live scalar allocation of `T`.
unsafe fn free<T>(ptr: NonNull<u8>) {
    dealloc_infallible(ptr, layout::<T>());
}
