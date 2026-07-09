//! The inline number representation.
//!
//! A small number is packed directly into a pointer-sized [`IValue`] with the
//! `Inline` tag and the number sub-family (bit 3 clear):
//!
//!   bits 0-2 : tag (Inline == 0)
//!   bit 3    : 0 (number sub-family)
//!   bits 4-7 : exponent code (see below)
//!   bits 8.. : signed mantissa
//!
//! The value is `mantissa * BASE^exp`. There are two representations, selected by
//! the `arbitrary_precision` feature and living in [`number_decimal`] (base 10, an
//! exact decimal that can hold non-`f64` values such as `0.1`) and
//! [`number_binary`] (base 2, a binary float, always an exact `f64`). Each is a
//! complete, self-contained `InlineValue` implementor; exactly one is compiled.
//! The bit layout and exponent-code scheme in this module are shared by both.
//!
//! The 4-bit exponent code encodes both the exponent and whether the number has a
//! decimal point:
//!
//!   code  0..=6  -> exp -7..=-1  (fraction; decimal point)
//!   code  7      -> exp 0        (float "N.0"; decimal point)
//!   code  8..=14 -> exp 1..=7    (float with trailing zeros factored out;
//!                                 decimal point)
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
//! floats.
//!
//! The reserved code is the *maximum* (`15`), not `0`: the plain-integer path is
//! the only one that emits a zero mantissa (integer zero), so placing it at code
//! `0` would make integer zero the all-zero bit pattern. Keeping it at `15`
//! leaves the all-zero value (mantissa 0, code 0) a non-canonical zero that is
//! never emitted, reserving it as the `NonNull` niche.

// The two inline number representations. Exactly one is compiled; each is a
// self-contained `InlineValue` implementor (`InlineNumberRepr`) built on the
// shared bit layout below, differing only in the base of `mantissa * BASE^exp`.
#[cfg(feature = "arbitrary_precision")]
#[path = "number_decimal.rs"]
mod base;
#[cfg(not(feature = "arbitrary_precision"))]
#[path = "number_binary.rs"]
mod base;

// The compiled representation and its bit-level constructors/accessor. Everything
// else the representation needs is the shared bit layout it reaches via `super::`.
#[cfg(feature = "arbitrary_precision")]
pub(crate) use base::encode_decimal;
pub(crate) use base::{encode_f64, num_val, InlineNumberRepr};

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

// The inline bits for `mantissa * BASE^exp` with the given exponent code. The
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

fn decode(bits: usize) -> (i64, i32) {
    (mantissa(bits), code_exp(code(bits)))
}

/// Decomposes a finite, non-zero `f64` into `(mantissa, exp2, negative)` such
/// that `value == (-1)^negative * mantissa * 2^exp2`. Shared by both bases'
/// `encode_f64`.
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

/// `true` if the `i64` value `v` is exactly representable as an `f64`.
fn i64_fits_f64(v: i64) -> bool {
    v == 0 || 64 - v.unsigned_abs().leading_zeros() - v.unsigned_abs().trailing_zeros() <= 53
}

/// Encodes a plain integer (no decimal point). Only exponent 0 is available to
/// integers — positive inline exponents are reserved for floats — so a value too
/// large for the mantissa does not fit inline and is stored on the heap instead.
/// Base-independent (an integer is `mantissa * BASE^0` in either base).
pub(crate) fn encode_int(value: i64) -> Option<usize> {
    let limit = 1i64 << (MANTISSA_BITS - 1);
    (value >= -limit && value < limit).then(|| encode(value, exp_code(0, false)))
}

/// Whether the source of this inline number had a decimal point. Base-independent
/// (the dot flag is part of the shared exponent code), so both representations use
/// this directly.
fn has_decimal_point(bits: usize) -> bool {
    code_has_dot(code(bits))
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
