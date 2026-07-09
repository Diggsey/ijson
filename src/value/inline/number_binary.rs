//! The base-2 inline number representation (used *without* `arbitrary_precision`).
//!
//! An inline number is a binary float `mantissa * 2^exp` with `exp` in `-7..=7`,
//! so every inline number is exactly an `f64`. There is no non-`f64` value to
//! represent, so an inline number never reduces to a `NumVal::Decimal`. A float
//! whose (trailing-zero-stripped) binary exponent falls outside the inline range
//! spills to a heap `f64` instead.
//!
//! This is a complete `InlineValue` implementor over the shared bit layout in the
//! parent module; the base-10 counterpart is [`super::number_decimal`].
#![allow(clippy::float_cmp)]

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::super::InlineValue;
use super::{decode, encode, exp_code, fits_mantissa, has_decimal_point, i64_fits_f64};
use super::{integer_decode, EXP_BIAS};
use crate::number::INumber;
use crate::value::{
    num_debug, num_hash, num_to_i64, num_to_u64, number_cmp, Destructured, DestructuredMut,
    DestructuredRef, IValue, NumVal, ValueType,
};

/// Scales `m` by `2^exp` by multiplying or dividing by an exact power of two built
/// from an integer shift. This avoids `f64::powi`, which is *not* guaranteed to be
/// correctly rounded even for a power of two (e.g. some implementations give
/// `2f64.powi(-1) == 0.49999999999999983`). For `exp` in the inline range
/// `-7..=7`, `1 << |exp| <= 128` is an exact `f64`, so the scale is exact and only
/// shifts the binary exponent.
fn scale_pow2(m: f64, exp: i32) -> f64 {
    if exp >= 0 {
        m * (1u64 << exp) as f64
    } else {
        m / (1u64 << (-exp)) as f64
    }
}

/// The exact integer value of `mantissa * 2^exp` if it is an integer that fits
/// `i64`; `None` if it is fractional.
fn to_i64(m: i64, exp: i32) -> Option<i64> {
    if exp >= 0 {
        // `exp <= 7` and `|m| < 2^55`, so `m << exp < 2^62` never overflows `i64`.
        Some(m << exp)
    } else {
        // An exact integer iff the low `k` bits shifted out are zero.
        let k = (-exp) as u32;
        (m & ((1i64 << k) - 1) == 0).then(|| m >> k)
    }
}

/// The exact `f64` value, if exactly representable. `m * 2^exp` is exact whenever
/// the mantissa has at most 53 significant bits — the `2^exp` factor only shifts
/// the binary exponent, which stays well within `f64` range for the inline range.
fn to_f64_exact(bits: usize) -> Option<f64> {
    let (m, exp) = decode(bits);
    i64_fits_f64(m).then(|| scale_pow2(m as f64, exp))
}

/// The (possibly lossy) `f64` value. Exact for every inline binary float (whose
/// mantissa is at most 53 bits); a wider mantissa would round.
fn to_f64_lossy(bits: usize) -> f64 {
    let (m, exp) = decode(bits);
    scale_pow2(m as f64, exp)
}

/// This inline number reduced to a [`NumVal`]. A binary inline float is always an
/// exact `f64`, so it is either an `Int` or a `Float`, never a `Decimal`.
pub(crate) fn num_val(bits: usize) -> NumVal {
    let (m, exp) = decode(bits);
    match to_i64(m, exp) {
        Some(i) => NumVal::Int(i),
        None => NumVal::Float(scale_pow2(m as f64, exp)),
    }
}

/// Encodes a finite `f64` inline as `mantissa * 2^exp`, if it fits: strip the
/// trailing binary zeros from the `f64`'s mantissa into the exponent, then check
/// that the mantissa and the exponent both fit the inline range.
pub(crate) fn encode_f64(value: f64) -> Option<usize> {
    if value == 0.0 {
        // 0.0 / -0.0: an integer zero that carried a decimal point.
        return Some(encode(0, exp_code(0, true)));
    }
    let (frac, e2, neg) = integer_decode(value);
    let tz = frac.trailing_zeros();
    let mag = i128::from(frac >> tz);
    let exp = e2 + tz as i32;
    let m = if neg { -mag } else { mag };
    (fits_mantissa(m) && (-EXP_BIAS..=EXP_BIAS).contains(&exp))
        .then(|| encode(m as i64, exp_code(exp, true)))
}

/// The base-2 inline representation of a JSON number.
pub(crate) struct InlineNumberRepr;
impl InlineValue for InlineNumberRepr {
    fn value_type(&self) -> ValueType {
        ValueType::Number
    }
    fn has_decimal_point(&self, v: &IValue) -> bool {
        has_decimal_point(v.ptr_usize())
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        num_hash(num_val(v.ptr_usize()), state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(a, b) == Ordering::Equal
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        Some(number_cmp(a, b))
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        num_debug(num_val(v.ptr_usize()), f)
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
    unsafe fn to_i64(&self, v: &IValue) -> Option<i64> {
        num_to_i64(num_val(v.ptr_usize()))
    }
    unsafe fn to_u64(&self, v: &IValue) -> Option<u64> {
        num_to_u64(num_val(v.ptr_usize()))
    }
    unsafe fn to_f64(&self, v: &IValue) -> Option<f64> {
        // A binary inline float decodes exactly.
        to_f64_exact(v.ptr_usize())
    }
    unsafe fn to_f64_lossy(&self, v: &IValue) -> Option<f64> {
        Some(to_f64_lossy(v.ptr_usize()))
    }
    // clone/drop use the inline defaults (bit-copy / nothing).
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_magnitudes_never_misencode() {
        // Whenever a value encodes inline it must decode back to exactly the input;
        // values that cannot be represented exactly inline must spill to the heap
        // (`None`) rather than to a wrong value. (Which values fit depends on the
        // base — few do here, since only small dyadic exponents fit — but the
        // round-trip must hold for every value that does.)
        for &x in &[
            0.5_f64,
            1e18,
            1e22,
            1.7014118346046923e38, // 2^127
            f64::MAX,
            1e39,
            6.022e23,
            9.223372036854776e18,
            f64::MIN_POSITIVE,
        ] {
            if let Some(bits) = encode_f64(x) {
                assert_eq!(to_f64_lossy(bits), x, "{:e} misencoded inline", x);
            }
        }
    }

    #[test]
    fn inline_floats_are_always_exact_f64() {
        // The base-2 format holds `mantissa * 2^exp`, which is always an exact
        // `f64`, so a fractional inline number reduces to `Float`, never
        // `Decimal`. `1 * 2^-1 == 0.5`.
        let bits = encode(1, exp_code(-1, true));
        assert_eq!(to_f64_exact(bits), Some(0.5));
        match num_val(bits) {
            NumVal::Float(f) => assert_eq!(f, 0.5),
            _ => panic!("expected Float(0.5)"),
        }
    }
}
