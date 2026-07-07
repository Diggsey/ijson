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
//! power-of-ten exponent and, for integers, whether the source had a decimal
//! point:
//!
//!   code  0..=6  -> exp -7..=-1  (fractional; always has a decimal point)
//!   code  7      -> exp 0, no decimal point (plain integer)
//!   code  8      -> exp 0, with decimal point ("N.0")
//!   code  9..=15 -> exp 1..=7    (integer; no decimal point)
//!
//! Integers with a decimal point only ever occur at exponent 0: a larger such
//! integer cannot fit the mantissa in fractional form, so it falls back to a
//! heap `f64`. That is why a single pair of exponent-0 codes suffices to carry
//! the decimal-point flag, leaving the layout within 4 exponent bits.
//!
//! The all-zero value (mantissa 0, code 0) is a non-canonical zero and is never
//! emitted, reserving it as the `NonNull` niche.
#![allow(clippy::float_cmp)]

use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

const EXP_SHIFT: u32 = 4;
const MANTISSA_SHIFT: u32 = 8;
/// Bits available for the signed inline mantissa (56 on 64-bit, 24 on 32-bit).
const MANTISSA_BITS: u32 = usize::BITS - MANTISSA_SHIFT;

const POW5: [u128; 8] = [1, 5, 25, 125, 625, 3125, 15625, 78125];

fn fits_mantissa(m: i128) -> bool {
    let limit = 1i128 << (MANTISSA_BITS - 1);
    m >= -limit && m < limit
}

/// Maps an exponent (and, at exp 0, a decimal-point flag) to its 4-bit code.
fn exp_code(exp: i32, dot: bool) -> usize {
    match exp {
        -7..=-1 => (exp + 7) as usize,
        0 => usize::from(dot) + 7,
        1..=7 => (exp + 8) as usize,
        _ => unreachable!("inline exponent out of range"),
    }
}
fn code_exp(code: usize) -> i32 {
    match code {
        0..=6 => code as i32 - 7,
        7 | 8 => 0,
        _ => code as i32 - 8,
    }
}
fn code_has_dot(code: usize) -> bool {
    code <= 6 || code == 8
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

/// Encodes an integer with no decimal point, factoring out trailing zeros as
/// needed to fit the mantissa.
pub(crate) fn encode_int(value: i128) -> Option<usize> {
    let mut m = value;
    let mut exp = 0i32;
    loop {
        if fits_mantissa(m) {
            return Some(encode(m as i64, exp_code(exp, false)));
        }
        if exp >= 7 || m % 10 != 0 {
            return None;
        }
        m /= 10;
        exp += 1;
    }
}

/// Encodes an integer-valued number that had a decimal point (`"N.0"`); these
/// only fit inline at exponent 0.
fn encode_int_dot(value: i128) -> Option<usize> {
    if fits_mantissa(value) {
        Some(encode(value as i64, 8))
    } else {
        None
    }
}

/// Encodes a finite `f64` inline as an exact decimal, if it fits.
pub(crate) fn encode_f64(value: f64) -> Option<usize> {
    if value == 0.0 {
        // 0.0 / -0.0: integer zero that had a decimal point.
        return Some(encode(0, 8));
    }
    let (m, e2, neg) = integer_decode(value);
    // Integer-valued float: store at exp 0 with the decimal-point flag.
    if let Some(int) = f64_as_integer(m, e2, neg) {
        return encode_int_dot(int);
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
