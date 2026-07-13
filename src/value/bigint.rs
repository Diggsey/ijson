//! Arbitrary-precision magnitude arithmetic — only the operations the heap decimal
//! representation and [`NumVal`](super::NumVal) need, not a general bignum library.
//!
//! A magnitude is a little-endian slice of `u64` limbs, kept *normalised*: the most
//! significant limb is non-zero, and zero is the empty slice. Every function here
//! preserves that, so a magnitude has exactly one representation and comparison is a
//! limb-wise scan. The sign and the decimal exponent live in the representation's
//! header — never here, so this module is pure unsigned integer arithmetic.
//!
//! The small-operand routines (`mul_small`, `div_small`, ...) take a `u64` and use a
//! `u128` accumulator, which is why they need no general multi-limb multiply or divide:
//! everything the decimal representation does is scaling by 10 (or 2 and 5, its prime
//! factors) and dividing by them again to canonicalise.
//!
//! Always compiled — and unit-tested — so the arithmetic is exercised in both feature
//! configurations, even though only `arbitrary_precision` builds a value from it.
#![cfg_attr(not(feature = "arbitrary_precision"), allow(dead_code))]

use std::cmp::Ordering;

/// The largest power of ten that fits a `u64` (`10^19 < 2^64`), so a magnitude can be
/// scaled by, and rendered in, 19 decimal digits at a time.
const POW10_CHUNK: u64 = 10_000_000_000_000_000_000;
const POW10_CHUNK_EXP: u64 = 19;

/// The largest power of five that fits a `u64` (`5^27 < 2^63`).
const POW5_CHUNK: u64 = 7_450_580_596_923_828_125;
const POW5_CHUNK_EXP: u64 = 27;

/// Drops leading zero limbs, restoring the invariant that the top limb is non-zero
/// (and that zero is the empty slice).
pub(crate) fn trim(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

/// `true` if the magnitude is zero. Normalised, so zero is exactly the empty slice.
pub(crate) fn is_zero(limbs: &[u64]) -> bool {
    limbs.is_empty()
}

/// `limbs *= m`, growing by at most one limb.
pub(crate) fn mul_small(limbs: &mut Vec<u64>, m: u64) {
    if m == 0 {
        limbs.clear();
        return;
    }
    let mut carry: u64 = 0;
    for limb in limbs.iter_mut() {
        let wide = u128::from(*limb) * u128::from(m) + u128::from(carry);
        *limb = wide as u64;
        carry = (wide >> 64) as u64;
    }
    if carry != 0 {
        limbs.push(carry);
    }
}

/// `limbs += a`.
pub(crate) fn add_small(limbs: &mut Vec<u64>, a: u64) {
    let mut carry = a;
    for limb in limbs.iter_mut() {
        if carry == 0 {
            return;
        }
        let (sum, overflowed) = limb.overflowing_add(carry);
        *limb = sum;
        carry = u64::from(overflowed);
    }
    if carry != 0 {
        limbs.push(carry);
    }
}

/// `limbs /= d`, returning the remainder. `d` must be non-zero.
pub(crate) fn div_small(limbs: &mut Vec<u64>, d: u64) -> u64 {
    debug_assert!(d != 0, "division by zero");
    let mut rem: u64 = 0;
    // Most significant limb first: each step divides `rem:limb` (a 128-bit value) by
    // `d`, leaving the quotient limb in place and carrying the remainder down.
    for limb in limbs.iter_mut().rev() {
        let wide = (u128::from(rem) << 64) | u128::from(*limb);
        *limb = (wide / u128::from(d)) as u64;
        rem = (wide % u128::from(d)) as u64;
    }
    trim(limbs);
    rem
}

/// The remainder of `limbs / d`, leaving `limbs` untouched. `d` must be non-zero.
pub(crate) fn rem_small(limbs: &[u64], d: u64) -> u64 {
    debug_assert!(d != 0, "division by zero");
    let mut rem: u64 = 0;
    for &limb in limbs.iter().rev() {
        let wide = (u128::from(rem) << 64) | u128::from(limb);
        rem = (wide % u128::from(d)) as u64;
    }
    rem
}

/// The magnitude denoted by a run of ASCII decimal digits (`b'0'..=b'9'`).
pub(crate) fn from_decimal_digits(digits: &[u8]) -> Vec<u64> {
    let mut limbs = Vec::new();
    for &digit in digits {
        debug_assert!(digit.is_ascii_digit(), "not a decimal digit");
        mul_small(&mut limbs, 10);
        add_small(&mut limbs, u64::from(digit - b'0'));
    }
    trim(&mut limbs);
    limbs
}

/// Orders two normalised magnitudes.
pub(crate) fn cmp(a: &[u64], b: &[u64]) -> Ordering {
    // Normalised, so the longer magnitude is the larger one.
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// The number of significant bits; zero has none.
pub(crate) fn bit_len(limbs: &[u64]) -> u64 {
    match limbs.last() {
        None => 0,
        Some(&top) => (limbs.len() as u64 - 1) * 64 + (64 - u64::from(top.leading_zeros())),
    }
}

/// The number of trailing zero bits; zero has none (by convention — callers check
/// [`is_zero`] first).
pub(crate) fn trailing_zeros(limbs: &[u64]) -> u64 {
    for (index, &limb) in limbs.iter().enumerate() {
        if limb != 0 {
            return index as u64 * 64 + u64::from(limb.trailing_zeros());
        }
    }
    0
}

/// The number of *significant* bits: the width of the magnitude once its factors of two
/// are divided out. This is the precision the value actually needs, so an `f64` — whose
/// mantissa is 53 bits — can hold it exactly only if this is at most 53.
pub(crate) fn significant_bits(limbs: &[u64]) -> u64 {
    bit_len(limbs) - trailing_zeros(limbs)
}

/// The magnitude of a `u64`.
pub(crate) fn from_u64(x: u64) -> Vec<u64> {
    if x == 0 {
        Vec::new()
    } else {
        vec![x]
    }
}

/// `limbs <<= n`.
pub(crate) fn shl(limbs: &mut Vec<u64>, n: u64) {
    if is_zero(limbs) || n == 0 {
        return;
    }
    let bit_shift = (n % 64) as u32;
    if bit_shift != 0 {
        let mut carry: u64 = 0;
        for limb in limbs.iter_mut() {
            let carried_out = *limb >> (64 - bit_shift);
            *limb = (*limb << bit_shift) | carry;
            carry = carried_out;
        }
        if carry != 0 {
            limbs.push(carry);
        }
    }
    // Whole limbs of shift are just zero limbs spliced in below the least significant
    // one; `n / 64` is bounded by the allocation this would need, so the cast is safe.
    let limb_shift = (n / 64) as usize;
    if limb_shift != 0 {
        limbs.splice(0..0, std::iter::repeat_n(0, limb_shift));
    }
}

/// `limbs *= 5^n`.
pub(crate) fn mul_pow5(limbs: &mut Vec<u64>, mut n: u64) {
    while n >= POW5_CHUNK_EXP {
        mul_small(limbs, POW5_CHUNK);
        n -= POW5_CHUNK_EXP;
    }
    if n != 0 {
        mul_small(limbs, 5u64.pow(n as u32));
    }
}

/// `limbs *= 10^n`, via `10^n == 5^n * 2^n` — so this needs only the small multiply
/// and a shift, no multi-limb multiplication.
pub(crate) fn mul_pow10(limbs: &mut Vec<u64>, n: u64) {
    mul_pow5(limbs, n);
    shl(limbs, n);
}

/// The magnitude's decimal digits (ASCII, no leading zeros; `"0"` for zero) — the
/// inverse of [`from_decimal_digits`].
pub(crate) fn to_decimal_digits(limbs: &[u64]) -> Vec<u8> {
    if is_zero(limbs) {
        return vec![b'0'];
    }
    // Peel off 19 digits at a time (the most a `u64` remainder can hold), least
    // significant chunk first.
    let mut rest = limbs.to_vec();
    let mut chunks = Vec::new();
    while !is_zero(&rest) {
        chunks.push(div_small(&mut rest, POW10_CHUNK));
    }
    // The leading chunk has no leading zeros; every chunk below it is a full 19 digits,
    // zeros included.
    let mut digits = chunks.pop().expect("non-zero").to_string().into_bytes();
    while let Some(chunk) = chunks.pop() {
        digits.extend_from_slice(
            format!("{:0width$}", chunk, width = POW10_CHUNK_EXP as usize).as_bytes(),
        );
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A magnitude is always normalised, so every operation must leave the top limb
    /// non-zero — otherwise `cmp` (which compares lengths first) would be wrong.
    fn assert_normalised(limbs: &[u64]) {
        assert_ne!(limbs.last(), Some(&0), "leading zero limb: {:?}", limbs);
    }

    /// Builds a magnitude from a decimal string, for readability in the tests below.
    fn mag(decimal: &str) -> Vec<u64> {
        let limbs = from_decimal_digits(decimal.as_bytes());
        assert_normalised(&limbs);
        limbs
    }

    /// Renders a magnitude back to decimal, for readability in the tests below.
    fn to_decimal(limbs: &[u64]) -> String {
        String::from_utf8(to_decimal_digits(limbs)).unwrap()
    }

    #[test]
    fn decimal_digits_round_trip() {
        // Values either side of a limb boundary, so the multi-limb carries are covered.
        for value in [
            "0",
            "1",
            "9",
            "18446744073709551615",                    // u64::MAX
            "18446744073709551616",                    // u64::MAX + 1: the first two-limb magnitude
            "340282366920938463463374607431768211455", // u128::MAX
            "340282366920938463463374607431768211456", // u128::MAX + 1: three limbs
            "123456789012345678901234567890123456789012345678901234567890",
        ] {
            let limbs = mag(value);
            assert_eq!(to_decimal(&limbs), value, "round trip of {}", value);
        }
    }

    #[test]
    fn zero_is_the_empty_magnitude() {
        // Normalisation makes zero unique, which is what lets `cmp` compare lengths.
        assert!(is_zero(&mag("0")));
        assert!(is_zero(&mag("0000")));
        assert_eq!(mag("0"), Vec::<u64>::new());
        assert_eq!(bit_len(&mag("0")), 0);
    }

    #[test]
    fn mul_and_div_by_small_are_inverse() {
        let mut limbs = mag("123456789012345678901234567890");
        mul_small(&mut limbs, 10);
        assert_normalised(&limbs);
        assert_eq!(to_decimal(&limbs), "1234567890123456789012345678900");

        assert_eq!(div_small(&mut limbs, 10), 0);
        assert_normalised(&limbs);
        assert_eq!(to_decimal(&limbs), "123456789012345678901234567890");
    }

    #[test]
    fn div_small_reports_the_remainder() {
        let mut limbs = mag("18446744073709551617"); // u64::MAX + 2
        assert_eq!(div_small(&mut limbs, 10), 7);
        assert_eq!(to_decimal(&limbs), "1844674407370955161");

        // `rem_small` agrees, without consuming the magnitude.
        let limbs = mag("18446744073709551617");
        assert_eq!(rem_small(&limbs, 10), 7);
        assert_eq!(to_decimal(&limbs), "18446744073709551617");
    }

    #[test]
    fn multiplying_by_zero_normalises_to_zero() {
        let mut limbs = mag("123456789012345678901234567890");
        mul_small(&mut limbs, 0);
        assert!(is_zero(&limbs));
        assert_normalised(&limbs);
    }

    #[test]
    fn add_small_carries_across_limbs() {
        // Adding 1 to u64::MAX must carry into a second limb.
        let mut limbs = mag("18446744073709551615");
        add_small(&mut limbs, 1);
        assert_normalised(&limbs);
        assert_eq!(to_decimal(&limbs), "18446744073709551616");
        assert_eq!(limbs, vec![0, 1]);

        // ...and adding to zero must allocate the first limb.
        let mut limbs = mag("0");
        add_small(&mut limbs, 7);
        assert_eq!(limbs, vec![7]);
    }

    #[test]
    fn cmp_orders_by_magnitude() {
        assert_eq!(cmp(&mag("0"), &mag("0")), Ordering::Equal);
        assert_eq!(cmp(&mag("0"), &mag("1")), Ordering::Less);
        // Different limb counts: the longer magnitude wins outright.
        assert_eq!(
            cmp(&mag("18446744073709551616"), &mag("18446744073709551615")),
            Ordering::Greater
        );
        // Same limb count: the most significant limb decides.
        assert_eq!(
            cmp(
                &mag("340282366920938463463374607431768211455"),
                &mag("340282366920938463463374607431768211454")
            ),
            Ordering::Greater
        );
        assert_eq!(
            cmp(&mag("123456789012345678901"), &mag("123456789012345678901")),
            Ordering::Equal
        );
    }

    #[test]
    fn bit_len_brackets_the_magnitude() {
        assert_eq!(bit_len(&mag("0")), 0);
        assert_eq!(bit_len(&mag("1")), 1);
        assert_eq!(bit_len(&mag("255")), 8);
        assert_eq!(bit_len(&mag("18446744073709551615")), 64); // u64::MAX
        assert_eq!(bit_len(&mag("18446744073709551616")), 65); // u64::MAX + 1
    }

    #[test]
    fn significant_bits_ignores_factors_of_two() {
        // The precision a value actually needs: what decides whether an `f64`, with its
        // 53-bit mantissa, can hold it exactly. A power of two needs one bit however
        // large it is, which is why `2^100` is an exact `f64` and `10^30` is not.
        assert_eq!(significant_bits(&mag("1")), 1);
        assert_eq!(significant_bits(&mag("2")), 1);
        assert_eq!(significant_bits(&mag("3")), 2);
        assert_eq!(significant_bits(&mag("18446744073709551616")), 1); // 2^64
        assert_eq!(
            significant_bits(&mag("1267650600228229401496703205376")), // 2^100
            1
        );
        // 10^30 == 2^30 * 5^30, and 5^30 needs 70 bits — too many for an `f64`.
        let mut ten_pow_30 = mag("1");
        mul_pow10(&mut ten_pow_30, 30);
        assert_eq!(significant_bits(&ten_pow_30), 70);

        // 2^64 + 1 is odd, so nothing can be divided out.
        assert_eq!(significant_bits(&mag("18446744073709551617")), 65);
    }

    #[test]
    fn from_u64_normalises_zero() {
        assert_eq!(from_u64(0), Vec::<u64>::new());
        assert_eq!(from_u64(7), vec![7]);
        assert_eq!(from_u64(u64::MAX), vec![u64::MAX]);
    }

    #[test]
    fn shl_shifts_across_limb_boundaries() {
        // A pure bit shift, a pure limb shift, and one that is both.
        let mut limbs = mag("1");
        shl(&mut limbs, 3);
        assert_eq!(limbs, vec![8]);

        let mut limbs = mag("1");
        shl(&mut limbs, 64);
        assert_eq!(limbs, vec![0, 1]);
        assert_eq!(to_decimal(&limbs), "18446744073709551616"); // 2^64

        let mut limbs = mag("1");
        shl(&mut limbs, 65);
        assert_eq!(limbs, vec![0, 2]);

        // The carry out of the top limb must grow the magnitude.
        let mut limbs = mag("18446744073709551615"); // u64::MAX
        shl(&mut limbs, 1);
        assert_normalised(&limbs);
        assert_eq!(to_decimal(&limbs), "36893488147419103230");

        // Shifting zero stays zero (and must not splice in zero limbs, which would
        // break normalisation).
        let mut limbs = mag("0");
        shl(&mut limbs, 130);
        assert!(is_zero(&limbs));
        assert_normalised(&limbs);
    }

    #[test]
    fn mul_pow10_matches_repeated_multiplication() {
        // Across the 19-digit chunk boundary, so both the chunked and remainder paths
        // of `mul_pow5` run, and against a naive oracle.
        for n in [0u64, 1, 5, 18, 19, 20, 27, 28, 60] {
            let mut scaled = mag("123456789");
            mul_pow10(&mut scaled, n);
            assert_normalised(&scaled);

            let mut oracle = mag("123456789");
            for _ in 0..n {
                mul_small(&mut oracle, 10);
            }
            assert_eq!(scaled, oracle, "123456789 * 10^{}", n);
        }

        // 10^n on its own, checked against the decimal rendering.
        let mut power = mag("1");
        mul_pow10(&mut power, 30);
        assert_eq!(to_decimal(&power), format!("1{}", "0".repeat(30)));
    }

    #[test]
    fn to_decimal_digits_pads_the_lower_chunks() {
        // The leading 19-digit chunk is unpadded, but every chunk below it keeps its
        // internal zeros — dropping them would silently divide the value by a power of
        // ten.
        assert_eq!(to_decimal_digits(&mag("0")), b"0");
        assert_eq!(to_decimal_digits(&mag("1")), b"1");

        // 10^19: one leading digit, then a chunk that is *all* zeros.
        let mut power = mag("1");
        mul_pow10(&mut power, 19);
        assert_eq!(to_decimal(&power), "10000000000000000000");

        // A value whose low chunk has a leading zero of its own.
        assert_eq!(
            to_decimal(&mag("10000000000000000001")), // 10^19 + 1
            "10000000000000000001"
        );
    }
}
