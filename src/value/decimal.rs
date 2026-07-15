//! The heap arbitrary-precision decimal representation (tag `NumberDecimal`).
//!
//! The spill target for a number too large or too precise for every other
//! representation: the exact value `(-1)^negative * magnitude * 10^exp`, with a binary
//! mantissa of as many `u64` limbs as it takes and a fixed-width decimal exponent. It
//! is a single pointer to a heap allocation whose header carries the exponent, sign and
//! limb count, followed by the limbs — the same shape as [`super::array`].
//!
//! Only [`crate::value::canonicalise`] builds one, and only when the value fits no
//! other variant of the numeric model. That is what keeps `NumVal`'s variants disjoint:
//! a number that *could* be an `i64`, a `u64`, an exact `f64` or a small `Decimal` never
//! reaches here, so a decimal value is never equal to a value in another representation
//! — and equality and hashing can be structural.
//!
//! # Presentation vs. value
//!
//! The header also records whether the number was *written* with a decimal point, which
//! is deliberately **not** part of its value: `1e30` and `1000000000000000000000000000000`
//! are the same number and must compare and hash alike, but only the first is a JSON
//! float. `has_decimal_point` reads that flag; `num_val` — which every comparison, hash
//! and conversion goes through — does not see it. (This is why the flag is a header bit
//! rather than a reserved exponent value, as in the inline representation: there, a
//! plain integer always has exponent zero, so one exponent code can carry both. Here
//! canonicalisation folds trailing zeros into the exponent, so an integer's exponent is
//! not zero in general, and the two facts need separate room.)
//!
//! # Safety
//!
//! Every `unsafe fn` here shares one precondition: the `IValue` argument must have the
//! `NumberDecimal` tag. Callers (`IValue`'s trait impls) uphold this via the tag they
//! already matched.

use std::alloc::{Layout, LayoutError};
use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;
use std::ptr::copy_nonoverlapping;

use crate::alloc::{alloc_infallible, dealloc_infallible};
use crate::number::INumber;
use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};

use super::{
    number_cmp, Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, ReprTag, ValueRepr,
    ValueType,
};

#[repr(C)]
#[repr(align(8))]
struct Header {
    /// The exponent of the canonical value. Never a sentinel: it is always the real
    /// exponent, so `num_val` needs no special case.
    exp: i32,
    /// The sign (bit 31), whether the literal had a decimal point (bit 30), and the
    /// number of limbs (bits 0..=29). Packed so the header stays 8 bytes, keeping the
    /// limbs 8-aligned with no padding.
    meta: u32,
}

const NEGATIVE: u32 = 1 << 31;
const DECIMAL_POINT: u32 = 1 << 30;
const LEN_MASK: u32 = DECIMAL_POINT - 1;
/// The most limbs a decimal can have — over 8 GiB of mantissa, so no real number comes
/// close; `alloc` asserts it rather than letting the count silently wrap into the flags.
const MAX_LIMBS: usize = LEN_MASK as usize;

trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn len(&self) -> usize {
        (self.meta & LEN_MASK) as usize
    }
    fn negative(&self) -> bool {
        self.meta & NEGATIVE != 0
    }
    fn has_decimal_point(&self) -> bool {
        self.meta & DECIMAL_POINT != 0
    }
    fn limbs_ptr(&self) -> *const u64 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr().add(1).cast::<u64>() }
    }
    fn limbs(&self) -> &'a [u64] {
        // Safety: the header's limb count is written at construction and never changes
        unsafe { std::slice::from_raw_parts(self.limbs_ptr(), self.len()) }
    }
}

trait HeaderMut<'a>: ThinMutExt<'a, Header> {
    fn limbs_ptr_mut(mut self) -> *mut u64 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr_mut().add(1).cast::<u64>() }
    }
}

impl<'a, T: ThinRefExt<'a, Header>> HeaderRef<'a> for T {}
impl<'a, T: ThinMutExt<'a, Header>> HeaderMut<'a> for T {}

/// The heap arbitrary-precision decimal representation.
pub(crate) struct DecimalRepr;

impl DecimalRepr {
    /// The heap layout for a decimal of `len` limbs: the `Header` followed by the
    /// limbs.
    fn layout(len: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u64>(len)?)?
            .0
            .pad_to_align())
    }

    /// Allocates a decimal and copies `magnitude` into it.
    ///
    /// The caller must pass a *canonical* magnitude — normalised, non-zero, and not
    /// divisible by ten — which is exactly what [`super::canonicalise`] yields; the
    /// uniqueness of that form is what lets `eq`/`hash` be structural.
    pub(crate) fn store(
        negative: bool,
        magnitude: &[u64],
        exp: i32,
        has_decimal_point: bool,
    ) -> IValue {
        assert!(magnitude.len() <= MAX_LIMBS, "decimal mantissa too large");
        debug_assert!(
            !super::bigint::is_zero(magnitude)
                && super::bigint::rem_small(magnitude, super::bigint::TEN) != 0,
            "a stored decimal's magnitude is canonical"
        );
        let meta = magnitude.len() as u32
            | if negative { NEGATIVE } else { 0 }
            | if has_decimal_point { DECIMAL_POINT } else { 0 };

        // Safety: freshly allocated to `layout(len)` and fully written before any read;
        // `limbs_ptr_mut` points at the trailing array we just sized.
        unsafe {
            let ptr = alloc_infallible(Self::layout(magnitude.len()).unwrap()).cast::<Header>();
            ptr.write(Header { exp, meta });
            let hd = ThinMut::new(ptr);
            copy_nonoverlapping(magnitude.as_ptr(), hd.limbs_ptr_mut(), magnitude.len());
            IValue::new_ptr(ReprTag::NumberDecimal, ptr.cast())
        }
    }

    /// Views the header behind a tagged value pointer.
    ///
    /// Safety: `v` must be a live decimal.
    unsafe fn header(v: &IValue) -> ThinRef<'_, Header> {
        ThinRef::new(v.ptr().cast())
    }

    /// Decodes the value. Safety: `v` must be a live decimal.
    unsafe fn num_val(v: &IValue) -> NumVal<'_> {
        let hd = Self::header(v);
        NumVal::from_big(hd.negative(), hd.limbs(), hd.exp)
    }
}

impl ValueRepr for DecimalRepr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    unsafe fn clone(&self, v: &IValue) -> IValue {
        let hd = Self::header(v);
        Self::store(hd.negative(), hd.limbs(), hd.exp, hd.has_decimal_point())
    }
    unsafe fn drop(&self, v: &mut IValue) {
        let layout = Self::layout(Self::header(v).len()).unwrap();
        dealloc_infallible(v.ptr(), layout);
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
    unsafe fn num_val<'a>(&self, v: &'a IValue) -> Option<NumVal<'a>> {
        Some(Self::num_val(v))
    }
    /// Presentation, not value — see the module docs. Read straight from the header, so
    /// it cannot leak into `num_val` and thence into equality.
    fn has_decimal_point(&self, v: &IValue) -> bool {
        // Safety: `v` is a decimal (the tag selected this representation).
        unsafe { Self::header(v) }.has_decimal_point()
    }
    // `to_i64`/`to_u64`/`to_f64`/`to_f64_lossy` use the `ValueRepr` defaults, derived
    // from `num_val`. The first three are always `None`: a value that fit any of them
    // would never have been stored here.
}
