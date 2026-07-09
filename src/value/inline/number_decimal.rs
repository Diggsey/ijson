//! The base-10 inline number representation (used *with* `arbitrary_precision`).
//!
//! An inline number is an exact decimal `mantissa * 10^exp` with `exp` in
//! `-7..=7`. Unlike the binary encoding it can represent values that are not
//! exact `f64`s — e.g. the fraction `0.1 == 1 * 10^-1` — which reduce to a
//! `NumVal::Decimal` holding the exact value. Larger or more precise decimals
//! spill to the heap arbitrary-precision representation.
//!
//! This is a complete `InlineValue` implementor over the shared bit layout in the
//! parent module; the base-2 counterpart is [`super::number_binary`].
#![allow(clippy::float_cmp)]

use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::{self, Formatter};
use std::hash::Hasher;

use super::super::InlineValue;
use super::{decode, encode, exp_code, fits_mantissa, has_decimal_point, i64_fits_f64};
use super::{integer_decode, EXP_BIAS};
use crate::number::INumber;
use crate::value::{
    decimal_to_f64_lossy, num_debug, num_hash, num_to_i64, num_to_u64, number_cmp, Destructured,
    DestructuredMut, DestructuredRef, IValue, NumVal, ValueType,
};

const POW5: [u128; 8] = [1, 5, 25, 125, 625, 3125, 15625, 78125];

/// `x << n`, or `None` if the shift would overflow a `u128`. `u128::checked_shl`
/// only rejects shift *amounts* `>= 128`; it silently drops the high bits when
/// the *value* overflows, so it cannot be used to detect a too-large product.
fn shl_checked(x: u128, n: u32) -> Option<u128> {
    if x == 0 {
        Some(0)
    } else if n <= x.leading_zeros() {
        Some(x << n)
    } else {
        None
    }
}

/// `true` if the (larger) `i128` value `v` is exactly representable as an `f64`.
fn i128_fits_f64(v: i128) -> bool {
    v == 0 || 128 - v.unsigned_abs().leading_zeros() - v.unsigned_abs().trailing_zeros() <= 53
}

/// If `value` (= `sign * m * 2^e2`) is an exact integer, returns it.
fn f64_as_integer(m: u64, e2: i32, neg: bool) -> Option<i128> {
    let mag: u128 = if e2 >= 0 {
        shl_checked(u128::from(m), e2 as u32)?
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
        shl_checked(u128::from(m).checked_mul(POW5[k as usize])?, e as u32)?
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

/// The exact integer value of `mantissa * 10^exp` if it is an integer that fits
/// `i64`; `None` if it is fractional or out of `i64` range.
fn decimal_to_i64(m: i64, exp: i32) -> Option<i64> {
    if exp >= 0 {
        m.checked_mul(10i64.pow(exp as u32))
    } else {
        let div = 10i64.pow((-exp) as u32);
        (m % div == 0).then_some(m / div)
    }
}

/// The exact integer value of `mantissa * 10^exp`, if it is an integer. Used for
/// the f64-exactness check, where the value can exceed `i64` (a large
/// integer-valued float, up to ~`3.6e23`).
fn decimal_to_i128(m: i64, exp: i32) -> Option<i128> {
    // `exp` is non-negative here (fractional values divide instead); the product
    // can exceed `i64` but always fits `i128`.
    i128::from(m).checked_mul(10i128.pow(exp as u32))
}

/// The exact `f64` value of `mantissa * 10^exp`, if it is exactly representable.
fn decimal_to_f64_exact(m: i64, exp: i32) -> Option<f64> {
    if m == 0 {
        return Some(0.0);
    }
    if exp >= 0 {
        // The value can exceed `i64` here (a large integer-valued float).
        let v = decimal_to_i128(m, exp)?;
        i128_fits_f64(v).then_some(v as f64)
    } else {
        // Fractional: `value == num / 2^k` after dividing out `5^k`. Everything
        // fits `i64` (`m` is the mantissa, `5^k <= 5^7`), and dividing an exact
        // integer by a power of two is itself exact.
        let k = (-exp) as u32;
        let p5 = 5i64.pow(k);
        if m % p5 != 0 {
            return None;
        }
        let num = m / p5;
        i64_fits_f64(num).then_some(num as f64 / (1u64 << k) as f64)
    }
}

/// This inline decimal reduced to a [`NumVal`]. An integer that fits `i64` becomes
/// `Int`; a value that is exactly an `f64` becomes `Float`; anything else — an
/// exact `mantissa * 10^exp` that is neither (e.g. `0.1`) — becomes the exact
/// `Decimal`. `num_val` covers the whole representable domain, not just what
/// today's constructors produce.
pub(crate) fn num_val(bits: usize) -> NumVal {
    let (m, exp) = decode(bits);
    if let Some(i) = decimal_to_i64(m, exp) {
        NumVal::Int(i)
    } else if let Some(f) = decimal_to_f64_exact(m, exp) {
        NumVal::Float(f)
    } else {
        NumVal::Decimal { mantissa: m, exp }
    }
}

/// The exact `f64` value, if exactly representable.
fn to_f64_exact(bits: usize) -> Option<f64> {
    let (m, exp) = decode(bits);
    decimal_to_f64_exact(m, exp)
}

/// The (possibly lossy) `f64` value. Every inline decimal a constructor can
/// produce is exactly an `f64`, so decode it exactly; only a non-`f64` inline
/// decimal falls back to the correctly-rounded conversion.
fn to_f64_lossy(bits: usize) -> f64 {
    let (m, exp) = decode(bits);
    decimal_to_f64_exact(m, exp).unwrap_or_else(|| decimal_to_f64_lossy(m, exp))
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

/// Encodes an exact decimal `mantissa * 10^exp` (written with a decimal point)
/// inline, or `None` if it does not fit the inline representation. Unlike
/// [`encode_f64`], the value need not be an exact `f64` — this is how e.g. `0.1`
/// (parsed from a string) is stored as the exact `1 * 10^-1`.
///
/// The result is canonical: bit-for-bit identical to [`encode_f64`] for a value
/// that is an exact `f64`, so the two constructors never disagree.
pub(crate) fn encode_decimal(mantissa: i128, exp: i32) -> Option<usize> {
    if mantissa == 0 {
        // 0.0 / -0.0 with a decimal point.
        return Some(encode(0, exp_code(0, true)));
    }
    // Strip trailing zeros to reach the canonical (minimal-mantissa) form.
    let mut m = mantissa;
    let mut e = exp;
    while m % 10 == 0 {
        m /= 10;
        e += 1;
    }
    if e >= 0 {
        // An integer written as a float: the integer-float encoder keeps the
        // largest mantissa that fits (minimal exponent), matching `encode_f64`.
        encode_int_float(m.checked_mul(10i128.checked_pow(e as u32)?)?)
    } else {
        // A fraction; `m * 10^e` with `m` free of trailing zeros is canonical.
        (fits_mantissa(m) && e >= -EXP_BIAS).then(|| encode(m as i64, exp_code(e, true)))
    }
}

/// The base-10 inline representation of a JSON number.
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
        // The inline decimal decodes exactly itself.
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
        // Regression: `u128::checked_shl` silently drops overflow bits, so
        // `encode_f64(2^127)` once wrongly produced an inline zero. Whenever a
        // value encodes inline it must decode back to exactly the input; values
        // that cannot be represented exactly inline must spill to the heap
        // (`None`) rather than to a wrong value.
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
    fn num_val_is_total_over_the_inline_domain() {
        // The base-10 format can hold any `mantissa * 10^exp`, including exact
        // decimals that are not exact `f64`s — e.g. `0.1 == 1 * 10^-1`. Today's
        // constructors reach one via the string parser, but `num_val` decodes the
        // representation and must handle the whole domain, holding the exact value
        // rather than panicking or rounding.
        let bits = encode(1, exp_code(-1, true)); // 1 * 10^-1 == 0.1
        assert!(
            to_f64_exact(bits).is_none(),
            "0.1 is not exactly representable as f64"
        );
        match num_val(bits) {
            NumVal::Decimal { mantissa, exp } => assert_eq!((mantissa, exp), (1, -1)),
            _ => panic!("expected an exact Decimal"),
        }
    }
}
