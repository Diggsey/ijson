//! The base-2 inline number representation (used *without* `arbitrary_precision`).
//!
//! A small number is packed directly into a pointer-sized [`IValue`] with the
//! `Inline` tag and the number bit (bit 3) set:
//!
//!   bits 0-2 : tag (Inline == 0)
//!   bit 3    : 1 (the number bit; also keeps the word non-zero)
//!   bits 4-7 : exponent code (see below)
//!   bits 8.. : signed mantissa
//!
//! The value is a binary float `mantissa * 2^exp` with `exp` in `-7..=7`, so every
//! inline number is exactly an `f64`; there is no non-`f64` value to represent, so
//! an inline number never reduces to a `Decimal` [`NumVal`]. A float whose
//! (trailing-zero-stripped) binary exponent falls outside the inline range spills
//! to a heap `f64` instead.
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
//! Every code except the reserved `15` is `exp + 7` ([`EXP_BIAS`]), so the exponent
//! is a single subtraction and every such value has a decimal point. The reserved
//! code is simply the one 4-bit value left over after the dotted exponents. (Integer
//! zero is never the all-zero niche word: the number bit at bit 3 keeps every inline
//! number non-zero.)
//!
//! This module is a complete, independent inline number representation, selected by
//! `arbitrary_precision` being off; the base-10 counterpart is `number_decimal.rs`.
//! They deliberately share no code, so their bit layouts can diverge.
#![allow(clippy::float_cmp)]
// Always compiled so its tests run, but unused as the active representation when
// `arbitrary_precision` is on.
#![cfg_attr(feature = "arbitrary_precision", allow(dead_code))]

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::number::{from_str_with, InlineNumber, InlineNumberError};
use super::InlineValue;
use crate::number::INumber;
use crate::value::{
    number_cmp, Destructured, DestructuredMut, DestructuredRef, IValue, NumVal, ValueType,
};

// --- Bit layout -------------------------------------------------------------

const EXP_SHIFT: u32 = 4;
const MANTISSA_SHIFT: u32 = 8;
/// Bits available for the signed inline mantissa (56 on 64-bit, 24 on 32-bit).
const MANTISSA_BITS: u32 = usize::BITS - MANTISSA_SHIFT;

// A number's fields sit strictly above the number bit (`IS_NUMBER`, bit 3): the
// exponent code at `EXP_SHIFT`, the mantissa at `MANTISSA_SHIFT`. `encode` sets
// `IS_NUMBER` unconditionally, so classification never depends on the payload — but a
// field overlapping bit 3 would still corrupt decoding, so pin the boundary here.
const _: () = assert!(super::IS_NUMBER < (1usize << EXP_SHIFT));

/// Every non-reserved code is `exp + EXP_BIAS`, so the exponent is a single
/// subtraction. With `EXP_BIAS == 7`, codes `0..=14` cover exp `-7..=7`.
const EXP_BIAS: i32 = 7;
/// Reserved (maximum) code for a plain integer at exponent 0 with no decimal point.
/// Codes `0..=14` are the dotted exponents `-7..=7`, so this marker takes the one
/// remaining 4-bit code. It is the *only* code without a decimal point.
const INT_EXP0_CODE: usize = 15;

// --- Pure f64 / integer math (no knowledge of the inline bit layout) --------

/// Decomposes a finite, non-zero `f64` into `(mantissa, exp2, negative)` such that
/// `value == (-1)^negative * mantissa * 2^exp2`.
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

/// The base-2 inline representation of a JSON number.
pub(crate) struct BinaryNumberRepr;

impl BinaryNumberRepr {
    // --- Bit-layout codec ---------------------------------------------------

    fn fits_mantissa(m: i128) -> bool {
        let limit = 1i128 << (MANTISSA_BITS - 1);
        m >= -limit && m < limit
    }

    /// Maps an exponent (and, at exp 0, a decimal-point flag) to its 4-bit code.
    fn exp_code(exp: i32, dot: bool) -> usize {
        if exp == 0 && !dot {
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
        code != INT_EXP0_CODE
    }

    fn encode(mantissa: i64, code: usize) -> usize {
        // `IS_NUMBER` (bit 3) marks the word as a number and keeps it non-zero; the
        // exponent code and mantissa sit above it.
        let bits = super::IS_NUMBER | ((mantissa as usize) << MANTISSA_SHIFT) | (code << EXP_SHIFT);
        debug_assert_eq!(
            bits & super::TAG_MASK,
            0,
            "inline number must leave the tag bits clear"
        );
        bits
    }
    fn mantissa(bits: usize) -> i64 {
        // Arithmetic shift sign-extends the mantissa from the top bits.
        ((bits as isize) >> MANTISSA_SHIFT) as i64
    }
    fn code(bits: usize) -> usize {
        (bits >> EXP_SHIFT) & 0xf
    }
    /// Tells the compiler these bits encode a plain integer — the one exponent code with
    /// no decimal point — without telling it *which* integer.
    ///
    /// Only the codegen probes use this; see `crate::codegen_probes`. It is stated here,
    /// in terms of this representation's own layout, because the two inline number
    /// representations share none.
    ///
    /// # Safety
    ///
    /// `bits` must be a plain inline integer, as [`Self::encode_int`] produces.
    #[cfg(codegen_probes)]
    pub(crate) unsafe fn assume_integer(bits: usize) {
        unsafe { std::hint::assert_unchecked(Self::code(bits) == INT_EXP0_CODE) };
    }

    fn decode(bits: usize) -> (i64, i32) {
        (Self::mantissa(bits), Self::code_exp(Self::code(bits)))
    }

    // --- Decode to a value --------------------------------------------------

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
        let (m, exp) = Self::decode(bits);
        i64_fits_f64(m).then(|| scale_pow2(m as f64, exp))
    }

    /// The (possibly lossy) `f64` value. Exact for every inline binary float (whose
    /// mantissa is at most 53 bits); a wider mantissa would round.
    fn to_f64_lossy(bits: usize) -> f64 {
        let (m, exp) = Self::decode(bits);
        scale_pow2(m as f64, exp)
    }

    /// Whether the source of this inline number had a decimal point.
    fn has_decimal_point(bits: usize) -> bool {
        Self::code_has_dot(Self::code(bits))
    }

    /// Decodes inline bits to a [`NumVal`]. A binary inline float is always an exact
    /// `f64`, so it reduces to an integer (`Int`/`UInt`) or a `Float`, never a `Decimal`.
    fn num_val(bits: usize) -> NumVal<'static> {
        let (m, exp) = Self::decode(bits);
        match Self::to_i64(m, exp) {
            Some(i) => NumVal::from_i64(i),
            None => NumVal::from_f64(scale_pow2(m as f64, exp)),
        }
    }
}

impl InlineNumber for BinaryNumberRepr {
    /// Only exponent 0 is available to integers — positive inline exponents are
    /// reserved for floats — so a value too large for the mantissa does not fit
    /// inline and goes to the heap instead.
    fn encode_int(value: i64) -> Option<usize> {
        let limit = 1i64 << (MANTISSA_BITS - 1);
        (value >= -limit && value < limit).then(|| Self::encode(value, Self::exp_code(0, false)))
    }

    /// Encodes `mantissa * 2^exp`: strip the trailing binary zeros from the `f64`'s
    /// mantissa into the exponent, then check that the mantissa and the exponent
    /// both fit the inline range.
    fn encode_f64(value: f64) -> Option<usize> {
        if value == 0.0 {
            // 0.0 / -0.0: an integer zero that carried a decimal point.
            return Some(Self::encode(0, Self::exp_code(0, true)));
        }
        let (frac, e2, neg) = integer_decode(value);
        let tz = frac.trailing_zeros();
        let mag = i128::from(frac >> tz);
        let exp = e2 + tz as i32;
        let m = if neg { -mag } else { mag };
        (Self::fits_mantissa(m) && (-EXP_BIAS..=EXP_BIAS).contains(&exp))
            .then(|| Self::encode(m as i64, Self::exp_code(exp, true)))
    }

    /// A float is stored as its nearest finite `f64` when that fits inline.
    fn from_str(s: &str) -> Result<usize, InlineNumberError> {
        from_str_with(s, Self::encode_int, |s| {
            s.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .and_then(Self::encode_f64)
        })
    }
}

impl InlineValue for BinaryNumberRepr {
    fn value_type(&self, _v: &IValue) -> ValueType {
        ValueType::Number
    }
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        Self::num_val(v.usize_()).hash(state);
    }
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        number_cmp(Self::num_val(a.usize_()), b) == Some(Ordering::Equal)
    }
    unsafe fn partial_cmp(&self, a: &IValue, b: &IValue) -> Option<Ordering> {
        number_cmp(Self::num_val(a.usize_()), b)
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", Self::num_val(v.usize_()))
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
    // clone/drop use the inline defaults (bit-copy / nothing); to_i64/to_u64/as_bytes
    // use the `InlineValue` defaults (derived from `num_val`, or `None`).
    unsafe fn num_val<'a>(&self, v: &'a IValue) -> Option<NumVal<'a>> {
        Some(Self::num_val(v.usize_()))
    }
    fn has_decimal_point(&self, v: &IValue) -> bool {
        Self::has_decimal_point(v.usize_())
    }
    unsafe fn to_f64(&self, v: &IValue) -> Option<f64> {
        // A binary inline float decodes exactly.
        Self::to_f64_exact(v.usize_())
    }
    unsafe fn to_f64_lossy(&self, v: &IValue) -> Option<f64> {
        Some(Self::to_f64_lossy(v.usize_()))
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
            let c = BinaryNumberRepr::exp_code(exp, true);
            assert_ne!(c, INT_EXP0_CODE);
            assert_eq!(BinaryNumberRepr::code_exp(c), exp);
            assert!(
                BinaryNumberRepr::code_has_dot(c),
                "dot value exp {} -> code {}",
                exp,
                c
            );
        }
        assert_eq!(BinaryNumberRepr::exp_code(0, false), INT_EXP0_CODE);
        assert_eq!(BinaryNumberRepr::code_exp(INT_EXP0_CODE), 0);
        assert!(!BinaryNumberRepr::code_has_dot(INT_EXP0_CODE));
        assert_ne!(
            BinaryNumberRepr::exp_code(0, false),
            BinaryNumberRepr::exp_code(0, true)
        );

        // Integer zero (and 0.0) are non-zero because every number sets `IS_NUMBER`,
        // so they are never the all-zero niche.
        assert_eq!(
            BinaryNumberRepr::encode_int(0),
            Some(BinaryNumberRepr::encode(0, INT_EXP0_CODE))
        );
        assert_ne!(BinaryNumberRepr::encode_int(0), Some(0));
        assert_ne!(BinaryNumberRepr::encode_f64(0.0), Some(0));
    }

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
            if let Some(bits) = BinaryNumberRepr::encode_f64(x) {
                assert_eq!(
                    BinaryNumberRepr::to_f64_lossy(bits),
                    x,
                    "{:e} misencoded inline",
                    x
                );
            }
        }
    }

    #[test]
    fn inline_floats_are_always_exact_f64() {
        // The base-2 format holds `mantissa * 2^exp`, which is always an exact
        // `f64`, so a fractional inline number is exactly an `f64` (`to_f64` is
        // `Some`), never a non-`f64` `Decimal`. `1 * 2^-1 == 0.5`.
        let bits = BinaryNumberRepr::encode(1, BinaryNumberRepr::exp_code(-1, true));
        assert_eq!(BinaryNumberRepr::to_f64_exact(bits), Some(0.5));
        let nv = BinaryNumberRepr::num_val(bits);
        assert_eq!(nv.to_i64(), None); // a fraction, not an integer
        assert_eq!(nv.to_f64(), Some(0.5)); // but exactly an f64
    }
}
