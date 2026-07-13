//! The heap `f64` number representation (tag `NumberF64`): the payload is an `f64`
//! together with the shape of the literal it came from.
//!
//! The shape is stored, not assumed. Being *held* as an `f64` and having been *written*
//! as a float are different facts, and only the second is what `has_decimal_point`
//! reports: with `arbitrary_precision`, an integer literal beyond `u64` — say `2^64` —
//! is exactly an `f64` and is kept here, but it is still an integer and must serialize
//! back as one. Assuming the two coincide is what would let an integer silently acquire
//! a decimal point.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::{alloc, free, read};
use crate::number::INumber;
use crate::value::{
    number_cmp, Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, ReprTag, ValueRepr,
    ValueType,
};

/// The payload: the value, and whether the literal it came from was a float.
#[derive(Clone, Copy)]
struct Payload {
    value: f64,
    has_decimal_point: bool,
}

/// The heap `f64` number representation.
pub(crate) struct F64Repr;

impl F64Repr {
    /// Stores an `f64` as a heap scalar. Always succeeds — the heap holds any `f64` —
    /// so it is the total fallback in construction.
    ///
    /// `has_decimal_point` is the *literal's* shape: true for every float, and true for
    /// an `f64` handed in by a Rust `f64` (there is no other reading of one). Only the
    /// arbitrary-precision parser passes false, for an integer literal too large for
    /// `i64`/`u64` that happens to be exactly an `f64`.
    pub(crate) fn store(value: f64, has_decimal_point: bool) -> IValue {
        // Safety: `alloc` returns a fresh, aligned, non-null allocation.
        unsafe {
            IValue::new_ptr(
                ReprTag::NumberF64,
                alloc::<Payload>(Payload {
                    value,
                    has_decimal_point,
                }),
            )
        }
    }

    /// Decodes the payload as a `NumVal`. Safety: `v` must be a live `NumberF64`.
    unsafe fn num_val(v: &IValue) -> NumVal<'static> {
        NumVal::from_f64(read::<Payload>(v.ptr()).value)
    }
}

impl ValueRepr for F64Repr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        Self::num_val(v).hash(state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(Self::num_val(a), b) == Some(Ordering::Equal)
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        number_cmp(Self::num_val(a), b)
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", Self::num_val(v))
    }
    fn destructure(&self, v: IValue) -> Destructured {
        Destructured::Number(INumber(v))
    }
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a> {
        DestructuredRef::Number(v.as_number_unchecked())
    }
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a> {
        DestructuredMut::Number(v.as_number_unchecked_mut())
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        let p = read::<Payload>(v.ptr());
        Self::store(p.value, p.has_decimal_point)
    }
    unsafe fn drop(&self, v: &mut IValue) {
        free::<Payload>(v.ptr());
    }
    unsafe fn num_val<'a>(&self, v: &'a IValue) -> Option<NumVal<'a>> {
        Some(Self::num_val(v))
    }
    /// The literal's shape, not the storage's — see the module docs.
    fn has_decimal_point(&self, v: &IValue) -> bool {
        // Safety: `v` is a `NumberF64` (the tag selected this representation).
        unsafe { read::<Payload>(v.ptr()) }.has_decimal_point
    }
}
