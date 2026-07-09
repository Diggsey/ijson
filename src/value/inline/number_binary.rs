//! The base-2 inline number encoding (used *without* `arbitrary_precision`).
//!
//! An inline number is a binary float `mantissa * 2^exp` with `exp` in `-7..=7`,
//! so every inline number is exactly an `f64`. There is no non-`f64` value to
//! represent, so an inline number never reduces to a `NumVal::Decimal`. A float
//! whose (trailing-zero-stripped) binary exponent falls outside the inline range
//! spills to a heap `f64` instead.
#![allow(clippy::float_cmp)]

use super::{encode, exp_code, fits_mantissa, i64_fits_f64, integer_decode, EXP_BIAS};
use crate::value::NumVal;

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

/// Scales `m` by `2^exp` by multiplying or dividing by an exact power of two
/// built from an integer shift. This avoids `f64::powi`, which is *not* guaranteed
/// to be correctly rounded even for a power of two (e.g. some implementations give
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

/// The exact `f64` value, if exactly representable. `m * 2^exp` is exact whenever
/// the mantissa has at most 53 significant bits — the `2^exp` factor only shifts
/// the binary exponent, which stays well within `f64` range for the inline range.
pub(super) fn to_f64_exact(m: i64, exp: i32) -> Option<f64> {
    i64_fits_f64(m).then(|| scale_pow2(m as f64, exp))
}

/// The (possibly lossy) `f64` value. Exact for every inline binary float (whose
/// mantissa is at most 53 bits); a wider mantissa would round.
pub(super) fn to_f64_lossy(m: i64, exp: i32) -> f64 {
    scale_pow2(m as f64, exp)
}

/// This inline number reduced to a [`NumVal`]. A binary inline float is always an
/// exact `f64`, so it is either an `Int` or a `Float`, never a `Decimal`.
pub(super) fn num_val(m: i64, exp: i32) -> NumVal {
    match to_i64(m, exp) {
        Some(i) => NumVal::Int(i),
        None => NumVal::Float(to_f64_lossy(m, exp)),
    }
}

/// Encodes a finite `f64` inline as `mantissa * 2^exp`, if it fits: strip the
/// trailing binary zeros from the `f64`'s mantissa into the exponent, then check
/// that the mantissa and the exponent both fit the inline range.
pub(super) fn encode_f64(value: f64) -> Option<usize> {
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
