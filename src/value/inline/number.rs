//! The inline decimal number representation.
//!
//! A small number is packed directly into a pointer-sized [`IValue`] with the
//! `Inline` tag and the number sub-family (bit 3 clear):
//!
//!   bits 0-2 : tag (Inline == 0)
//!   bit 3    : 0 (number sub-family)
//!   bits 4-7 : exponent code (see below)
//!   bits 8.. : signed mantissa
//!
//! The value is `mantissa * 10^exp`. The 4-bit exponent code encodes both the
//! power-of-ten exponent and whether the number has a decimal point:
//!
//!   code  0..=6  -> exp -7..=-1  (fraction; decimal point)
//!   code  7      -> exp 0        (integer-valued float "N.0"; decimal point)
//!   code  8..=14 -> exp 1..=7    (integer-valued float in e-notation, trailing
//!                                 zeros factored out; decimal point)
//!   code  15     -> exp 0        (plain integer; no decimal point)
//!
//! Every code except the reserved `15` is simply `exp + 7` ([`EXP_BIAS`]), so the
//! exponent is a single subtraction, and every such value has a decimal point:
//! `has_decimal_point` is exactly `code != 15`.
//!
//! Only a plain integer lacks a decimal point, and only at exponent 0. A plain
//! integer too large for the mantissa is *not* factored into a positive exponent
//! — that would collide with the e-notation float codes and mislabel it as
//! having a decimal point (which drives serialization back to a float). Instead
//! it spills to a heap `i64`/`u64`. Positive exponents are therefore reserved for
//! floats: `1e18` factors to `mantissa 1e11, exp 7` and stays inline, whereas the
//! integer `1000000000000000000` goes to the heap.
//!
//! The reserved code is the *maximum* (`15`), not `0`: the plain-integer path is
//! the only one that emits a zero mantissa (integer zero), so placing it at code
//! `0` would make integer zero the all-zero bit pattern. Keeping it at `15`
//! leaves the all-zero value (mantissa 0, code 0) a non-canonical zero that is
//! never emitted, reserving it as the `NonNull` niche.
#![allow(clippy::float_cmp)]

use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

const EXP_SHIFT: u32 = 4;
const MANTISSA_SHIFT: u32 = 8;
/// Bits available for the signed inline mantissa (56 on 64-bit, 24 on 32-bit).
const MANTISSA_BITS: u32 = usize::BITS - MANTISSA_SHIFT;

/// Every non-reserved code is `exp + EXP_BIAS`, so the exponent is a single
/// subtraction. With `EXP_BIAS == 7`, codes `0..=14` cover exp `-7..=7`.
const EXP_BIAS: i32 = 7;
/// Reserved (maximum) code for a plain integer at exponent 0 with no decimal
/// point. It is the max rather than `0` so that integer zero (mantissa 0) never
/// becomes the all-zero niche pattern; see the module docs. It is also the *only*
/// code without a decimal point.
const INT_EXP0_CODE: usize = 15;

const POW5: [u128; 8] = [1, 5, 25, 125, 625, 3125, 15625, 78125];

fn fits_mantissa(m: i128) -> bool {
    let limit = 1i128 << (MANTISSA_BITS - 1);
    m >= -limit && m < limit
}

/// Maps an exponent (and, at exp 0, a decimal-point flag) to its 4-bit code.
fn exp_code(exp: i32, dot: bool) -> usize {
    if exp == 0 && !dot {
        // The plain integer at exp 0 is the reserved code; every other value is
        // linear in the exponent.
        INT_EXP0_CODE
    } else {
        debug_assert!((-7..=7).contains(&exp), "inline exponent out of range");
        (exp + EXP_BIAS) as usize
    }
}
fn code_exp(code: usize) -> i32 {
    if code == INT_EXP0_CODE {
        0
    } else {
        code as i32 - EXP_BIAS
    }
}
fn code_has_dot(code: usize) -> bool {
    // Every code but the reserved plain-integer code carries a decimal point.
    code != INT_EXP0_CODE
}

// The inline bits for `mantissa * 10^exp` with the given exponent code. The
// `Inline` tag (0) and number sub-family (bit 3) are both zero here, so the low
// bits are clear; canonical values never encode (mantissa 0, code 0), so the
// result is non-zero.
fn encode(mantissa: i64, code: usize) -> usize {
    ((mantissa as usize) << MANTISSA_SHIFT) | (code << EXP_SHIFT)
}

fn mantissa(bits: usize) -> i64 {
    // Arithmetic shift sign-extends the mantissa from the top bits.
    ((bits as isize) >> MANTISSA_SHIFT) as i64
}
fn code(bits: usize) -> usize {
    (bits >> EXP_SHIFT) & 0xf
}

// --- f64 decomposition ------------------------------------------------------

/// Decomposes a finite, non-zero `f64` into `(mantissa, exp2, negative)` such
/// that `value == (-1)^negative * mantissa * 2^exp2`.
fn integer_decode(value: f64) -> (u64, i32, bool) {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    if raw_exp == 0 {
        (frac, -1074, negative)
    } else {
        (frac | 0x0010_0000_0000_0000, raw_exp - 1075, negative)
    }
}

/// If `value` (= `sign * m * 2^e2`) is an exact integer, returns it.
fn f64_as_integer(m: u64, e2: i32, neg: bool) -> Option<i128> {
    let mag: u128 = if e2 >= 0 {
        (u128::from(m)).checked_shl(e2 as u32)?
    } else {
        let sh = (-e2) as u32;
        if sh >= 64 || m & ((1u64 << sh) - 1) != 0 {
            return None;
        }
        u128::from(m >> sh)
    };
    let mag = i128::try_from(mag).ok()?;
    Some(if neg { -mag } else { mag })
}

/// If `value * 10^k` (= `sign * m * 2^e2 * 10^k`) is an exact integer, returns it.
fn f64_scaled_integer(m: u64, e2: i32, neg: bool, k: u32) -> Option<i128> {
    let e = e2 + k as i32;
    let mag: u128 = if e >= 0 {
        u128::from(m)
            .checked_mul(POW5[k as usize])?
            .checked_shl(e as u32)?
    } else {
        let sh = (-e) as u32;
        if sh >= 64 || m & ((1u64 << sh) - 1) != 0 {
            return None;
        }
        u128::from(m >> sh).checked_mul(POW5[k as usize])?
    };
    let mag = i128::try_from(mag).ok()?;
    Some(if neg { -mag } else { mag })
}

// --- Inline encoders (return `None` if the value doesn't fit inline) ---------

/// Encodes a plain integer (no decimal point). Only exponent 0 is available to
/// integers — positive inline exponents are reserved for floats — so a value too
/// large for the mantissa does not fit inline and is stored on the heap instead.
pub(crate) fn encode_int(value: i128) -> Option<usize> {
    fits_mantissa(value).then(|| encode(value as i64, exp_code(0, false)))
}

/// Encodes an integer-valued *float* (`"N.0"`, or e-notation such as `1e18`),
/// factoring out trailing zeros into a positive exponent as needed to fit the
/// mantissa. These carry a decimal point, so they use the `dot` codes.
fn encode_int_float(value: i128) -> Option<usize> {
    let mut m = value;
    let mut exp = 0i32;
    loop {
        if fits_mantissa(m) {
            return Some(encode(m as i64, exp_code(exp, true)));
        }
        if exp >= 7 || m % 10 != 0 {
            return None;
        }
        m /= 10;
        exp += 1;
    }
}

/// Encodes a finite `f64` inline as an exact decimal, if it fits.
pub(crate) fn encode_f64(value: f64) -> Option<usize> {
    if value == 0.0 {
        // 0.0 / -0.0: integer zero that had a decimal point.
        return Some(encode(0, exp_code(0, true)));
    }
    let (m, e2, neg) = integer_decode(value);
    // Integer-valued float: store with the decimal-point flag, factoring trailing
    // zeros into a positive exponent for large magnitudes (e.g. `1e18`).
    if let Some(int) = f64_as_integer(m, e2, neg) {
        return encode_int_float(int);
    }
    // Otherwise fractional: the smallest `k` making `value * 10^k` an exact
    // integer gives the canonical (minimal-fraction) form.
    for k in 1..=7u32 {
        if let Some(d) = f64_scaled_integer(m, e2, neg, k) {
            return if fits_mantissa(d) {
                Some(encode(d as i64, exp_code(-(k as i32), true)))
            } else {
                None
            };
        }
    }
    None
}

// --- Decimal decoders -------------------------------------------------------

/// The exact integer value of `mantissa * 10^exp`, if it is an integer.
fn decimal_to_i128(m: i64, exp: i32) -> Option<i128> {
    if exp >= 0 {
        i128::from(m).checked_mul(10i128.pow(exp as u32))
    } else {
        let div = 10i128.pow((-exp) as u32);
        let m = i128::from(m);
        if m % div == 0 {
            Some(m / div)
        } else {
            None
        }
    }
}

fn decimal_to_f64_lossy(m: i64, exp: i32) -> f64 {
    // The inline exponent range is small, so `10^|exp|` is an exact integer and
    // an exact `f64`; using it (rather than the non-deterministic `powi`) keeps
    // the result deterministic and exact whenever the value is representable.
    if exp >= 0 {
        m as f64 * 10i64.pow(exp as u32) as f64
    } else {
        m as f64 / 10i64.pow((-exp) as u32) as f64
    }
}

/// `true` if `v` is exactly representable as an `f64`.
fn i128_fits_f64(v: i128) -> bool {
    if v == 0 {
        return true;
    }
    let a = v.unsigned_abs();
    128 - a.leading_zeros() - a.trailing_zeros() <= 53
}

/// The exact `f64` value of `mantissa * 10^exp`, if it is exactly representable.
fn decimal_to_f64_exact(m: i64, exp: i32) -> Option<f64> {
    if m == 0 {
        return Some(0.0);
    }
    if exp >= 0 {
        let v = decimal_to_i128(m, exp)?;
        i128_fits_f64(v).then_some(v as f64)
    } else {
        let k = (-exp) as u32;
        let p5 = 5i128.pow(k);
        let mi = i128::from(m);
        if mi % p5 != 0 {
            return None;
        }
        let num = mi / p5; // value == num / 2^k, a dyadic rational
                           // Dividing an exactly-representable integer by a power of two is exact
                           // (it only adjusts the exponent), and avoids the non-deterministic
                           // `powi`.
        i128_fits_f64(num).then_some(num as f64 / (1u64 << k) as f64)
    }
}

// --- High-level per-value operations (on the raw inline bits) ----------------

fn decode(bits: usize) -> (i64, i32) {
    (mantissa(bits), code_exp(code(bits)))
}

/// The exact integer value, if this inline number is an integer.
pub(crate) fn value_i128(bits: usize) -> Option<i128> {
    let (m, exp) = decode(bits);
    decimal_to_i128(m, exp)
}

/// The exact `f64` value, if exactly representable.
pub(crate) fn to_f64_exact(bits: usize) -> Option<f64> {
    let (m, exp) = decode(bits);
    decimal_to_f64_exact(m, exp)
}

/// The (possibly lossy) `f64` value.
pub(crate) fn to_f64_lossy(bits: usize) -> f64 {
    let (m, exp) = decode(bits);
    decimal_to_f64_lossy(m, exp)
}

/// Whether the source of this inline number had a decimal point.
pub(crate) fn has_decimal_point(bits: usize) -> bool {
    code_has_dot(code(bits))
}

/// Hashes an inline number by its numeric value (so `2` and `2.0` agree, and
/// the scheme matches the heap number hash).
pub(crate) fn hash<H: Hasher>(bits: usize, state: &mut H) {
    match value_i128(bits) {
        Some(v) if i64::try_from(v).is_ok() => (v as i64).hash(state),
        Some(v) if u64::try_from(v).is_ok() => (v as u64).hash(state),
        _ => {
            let f = to_f64_lossy(bits);
            (if f == 0.0 { 0 } else { f.to_bits() }).hash(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponent_codes_are_linear_and_niche_safe() {
        // Every value with a decimal point spans exp -7..=7 and is linear in the
        // exponent: `exp = code - EXP_BIAS`. None of them is the reserved code.
        for exp in -7..=7 {
            let c = exp_code(exp, true);
            assert_ne!(c, INT_EXP0_CODE);
            assert_eq!(code_exp(c), exp);
            assert!(code_has_dot(c), "dot value exp {} -> code {}", exp, c);
        }
        // The only code without a decimal point is the reserved plain-integer
        // code, used only at exponent 0 and kept apart from the "N.0" code.
        assert_eq!(exp_code(0, false), INT_EXP0_CODE);
        assert_eq!(code_exp(INT_EXP0_CODE), 0);
        assert!(!code_has_dot(INT_EXP0_CODE));
        assert_ne!(exp_code(0, false), exp_code(0, true));

        // Integer zero (and 0.0) must never be the all-zero bit pattern that is
        // reserved as the `NonNull` niche.
        assert_eq!(encode_int(0), Some(encode(0, INT_EXP0_CODE)));
        assert_ne!(encode_int(0), Some(0));
        assert_ne!(encode_f64(0.0), Some(0));
    }
}
