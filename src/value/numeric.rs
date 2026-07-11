//! The numeric value model shared by every number representation.
//!
//! [`NumVal`] is a number reduced to a form suitable for *exact* comparison,
//! hashing, and conversion, independent of how it is stored. Each number
//! representation decodes its storage to a `NumVal` (via `ValueRepr::num_val`); the
//! methods on `NumVal` then implement every numeric value operation once, shared by
//! all representations. The free helpers below are the scalar arithmetic those
//! methods are built from.

use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hasher;

// A number reduced to a canonical form suitable for exact numeric comparison,
// hashing, and conversion. Construction — via `from_i64`/`from_u64`/`from_f64`/
// `from_decimal` — normalises every value into a single variant, so equal numbers
// are always the same variant:
//
//   - `Int`     : an integer in `i64` range.
//   - `UInt`    : an integer in `(i64::MAX, u64::MAX]`.
//   - `Float`   : an exact `f64` that is *not* an integer in `i64`/`u64` range (a
//                 fraction, or an integer-valued float beyond `u64`); never `±0.0`.
//   - `Decimal` : the residual — an exact `mantissa * 10^exp` that is neither an
//                 integer nor an exact `f64` (e.g. the fraction `0.1`), stored with
//                 trailing zeros removed. Only the base-10 inline representation
//                 (`arbitrary_precision`) produces one; `exp` is in `-7..=7`.
//
// Because the variants are disjoint, equality and hashing reduce to the variant and
// its fields, and the accessors mostly select a single variant. `Decimal` and its
// arithmetic are always compiled and unit-tested, even when the active
// representation never produces one.
#[derive(Clone, Copy)]
pub(crate) struct NumVal(Repr);

#[derive(Clone, Copy)]
enum Repr {
    Int(i64),
    UInt(u64),
    Float(f64),
    Decimal { mantissa: i64, exp: i32 },
}

impl NumVal {
    /// An `i64`.
    pub(crate) fn from_i64(x: i64) -> NumVal {
        NumVal(Repr::Int(x))
    }

    /// A `u64` — stored as `Int` when it also fits `i64`.
    pub(crate) fn from_u64(x: u64) -> NumVal {
        match i64::try_from(x) {
            Ok(i) => NumVal(Repr::Int(i)),
            Err(_) => NumVal(Repr::UInt(x)),
        }
    }

    /// A finite `f64` — reduced to `Int`/`UInt` when it is exactly an integer in
    /// range (so `1e18` and the integer `10^18` become the same value).
    pub(crate) fn from_f64(x: f64) -> NumVal {
        if x.fract() == 0.0 {
            if (-I64_RANGE..I64_RANGE).contains(&x) {
                return NumVal(Repr::Int(x as i64));
            }
            if (0.0..U64_RANGE).contains(&x) {
                return NumVal(Repr::UInt(x as u64));
            }
        }
        NumVal(Repr::Float(x))
    }

    /// An exact `mantissa * 10^exp` — reduced to `Int`/`UInt` when it is an integer
    /// in range and to `Float` when it is exactly an `f64`; otherwise a canonical
    /// `Decimal` with trailing zeros removed.
    pub(crate) fn from_decimal(mantissa: i64, exp: i32) -> NumVal {
        if let Some(v) = decimal_int_value(mantissa, exp) {
            if let Ok(i) = i64::try_from(v) {
                return NumVal(Repr::Int(i));
            }
            if let Ok(u) = u64::try_from(v) {
                return NumVal(Repr::UInt(u));
            }
            // A larger integer-valued number; it may still be an exact `f64` below.
        }
        if let Some(f) = decimal_to_f64_exact(mantissa, exp) {
            return NumVal(Repr::Float(f));
        }
        let (mantissa, exp) = canonical_decimal(mantissa, exp);
        NumVal(Repr::Decimal { mantissa, exp })
    }

    /// The exact `i64` value, if it is one. Only `Int` holds an `i64`-range integer.
    pub(crate) fn to_i64(self) -> Option<i64> {
        match self.0 {
            Repr::Int(x) => Some(x),
            _ => None,
        }
    }

    /// The exact `u64` value, if it is a non-negative integer in range.
    pub(crate) fn to_u64(self) -> Option<u64> {
        match self.0 {
            Repr::Int(x) => u64::try_from(x).ok(),
            Repr::UInt(x) => Some(x),
            _ => None,
        }
    }

    /// The exact `f64` value, if it is exactly representable.
    pub(crate) fn to_f64(self) -> Option<f64> {
        match self.0 {
            Repr::Int(x) => can_represent_as_f64(x.unsigned_abs()).then_some(x as f64),
            Repr::UInt(x) => can_represent_as_f64(x).then_some(x as f64),
            Repr::Float(x) => Some(x),
            // A `Decimal` is, by construction, not exactly an `f64`.
            Repr::Decimal { .. } => None,
        }
    }

    /// The (possibly lossy) `f64` value.
    pub(crate) fn to_f64_lossy(self) -> f64 {
        match self.0 {
            Repr::Int(x) => x as f64,
            Repr::UInt(x) => x as f64,
            Repr::Float(x) => x,
            Repr::Decimal { mantissa, exp } => decimal_to_f64_lossy(mantissa, exp),
        }
    }

    /// Hashes by value. The variants are disjoint (equal numbers share one), and each
    /// is canonical, so each hashes its own fields directly.
    pub(crate) fn hash(self, state: &mut dyn Hasher) {
        match self.0 {
            Repr::Int(x) => state.write_i64(x),
            Repr::UInt(x) => state.write_u64(x),
            // `Float` never holds `±0.0` or an integer in `u64` range (those reduce to
            // `Int`/`UInt`), so its bit pattern is unique per value.
            Repr::Float(x) => state.write_u64(x.to_bits()),
            Repr::Decimal { mantissa, exp } => {
                state.write_i64(mantissa);
                state.write_i32(exp);
            }
        }
    }

    /// The exact total order over two numbers, across every variant.
    pub(crate) fn cmp(self, other: NumVal) -> Ordering {
        use Repr::{Decimal, Float, Int, UInt};
        match (self.0, other.0) {
            (Int(x), Int(y)) => x.cmp(&y),
            (UInt(x), UInt(y)) => x.cmp(&y),
            (Int(x), UInt(y)) => {
                if x < 0 {
                    Ordering::Less
                } else {
                    (x as u64).cmp(&y)
                }
            }
            (UInt(x), Int(y)) => {
                if y < 0 {
                    Ordering::Greater
                } else {
                    x.cmp(&(y as u64))
                }
            }
            (Int(x), Float(y)) => cmp_i64_f64(x, y),
            (Float(x), Int(y)) => cmp_i64_f64(y, x).reverse(),
            (UInt(x), Float(y)) => cmp_u64_f64(x, y),
            (Float(x), UInt(y)) => cmp_u64_f64(y, x).reverse(),
            (Float(x), Float(y)) => x.partial_cmp(&y).unwrap(),
            (
                Decimal { mantissa, exp },
                Decimal {
                    mantissa: m2,
                    exp: e2,
                },
            ) => cmp_decimal_decimal(mantissa, exp, m2, e2),
            (Decimal { mantissa, exp }, Int(y)) => cmp_decimal_int(mantissa, exp, i128::from(y)),
            (Int(x), Decimal { mantissa, exp }) => {
                cmp_decimal_int(mantissa, exp, i128::from(x)).reverse()
            }
            (Decimal { mantissa, exp }, UInt(y)) => cmp_decimal_int(mantissa, exp, i128::from(y)),
            (UInt(x), Decimal { mantissa, exp }) => {
                cmp_decimal_int(mantissa, exp, i128::from(x)).reverse()
            }
            // A `Decimal` (an exact non-`f64` value) and a `Float` are compared
            // exactly: `0.1` (decimal) and `0.1_f64` are different numbers.
            (Decimal { mantissa, exp }, Float(y)) => cmp_decimal_f64(mantissa, exp, y),
            (Float(x), Decimal { mantissa, exp }) => cmp_decimal_f64(mantissa, exp, x).reverse(),
        }
    }
}

/// Formats a number the way `serde_json` would (integer if it is one).
impl Debug for NumVal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(x) = self.to_i64() {
            Debug::fmt(&x, f)
        } else if let Some(x) = self.to_u64() {
            Debug::fmt(&x, f)
        } else {
            Debug::fmt(&self.to_f64_lossy(), f)
        }
    }
}

// `2^63` and `2^64`: one past `i64::MAX`/`u64::MAX`, and (negated) exactly `i64::MIN`.
// The maxima are *not* representable as `f64` — they round up to these powers of two
// — and no `f64` lies between a maximum and the next power of two. So these bound the
// `f64`s that convert to the integer type exactly: a whole `x` fits `i64` iff
// `-I64_RANGE <= x < I64_RANGE`, and `u64` iff `0 <= x < U64_RANGE`. (Using the
// powers of two directly, rather than `i64::MAX as f64` etc., keeps that reasoning
// explicit and matches the comparison helpers below.)
const I64_RANGE: f64 = 9_223_372_036_854_775_808.0;
const U64_RANGE: f64 = 18_446_744_073_709_551_616.0;

fn can_represent_as_f64(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 11
}

// `a == trunc(b)` already; the fractional part of `b` breaks the tie.
fn cmp_by_fraction(b: f64, bt: f64) -> Ordering {
    if b == bt {
        Ordering::Equal
    } else if b > bt {
        Ordering::Less // b has a positive fractional part, so b > a
    } else {
        Ordering::Greater
    }
}

/// Compares an `i64` to a finite float exactly.
fn cmp_i64_f64(a: i64, b: f64) -> Ordering {
    if b >= I64_RANGE {
        return Ordering::Less; // b >= 2^63 > i64::MAX >= a
    }
    if b < -I64_RANGE {
        return Ordering::Greater; // b < -2^63 == i64::MIN <= a
    }
    let bt = b.trunc(); // now in [-2^63, 2^63), so `bt as i64` is exact
    match a.cmp(&(bt as i64)) {
        Ordering::Equal => cmp_by_fraction(b, bt),
        ord => ord,
    }
}

/// Compares a `u64` to a finite float exactly.
fn cmp_u64_f64(a: u64, b: f64) -> Ordering {
    if b < 0.0 {
        return Ordering::Greater; // a >= 0 > b
    }
    if b >= U64_RANGE {
        return Ordering::Less; // b >= 2^64 > u64::MAX >= a
    }
    let bt = b.trunc(); // now in [0, 2^64), so `bt as u64` is exact
    match a.cmp(&(bt as u64)) {
        Ordering::Equal => cmp_by_fraction(b, bt),
        ord => ord,
    }
}

// --- Exact `Decimal` arithmetic ---------------------------------------------
// A `Decimal { mantissa, exp }` is the exact value `mantissa * 10^exp` with
// `exp` in the inline range `-7..=7`, so `10^|exp|` fits an `i64` and the scaled
// products below fit an `i128`.

/// The exact integer value of `mantissa * 10^exp`, if it is an integer.
fn decimal_int_value(mantissa: i64, exp: i32) -> Option<i128> {
    if exp >= 0 {
        Some(i128::from(mantissa) * 10i128.pow(exp as u32))
    } else {
        let div = 10i128.pow((-exp) as u32);
        (i128::from(mantissa) % div == 0).then_some(i128::from(mantissa) / div)
    }
}

/// The exact `f64` value of `mantissa * 10^exp`, if it is exactly representable.
/// Used both by [`NumVal::from_decimal`] to classify a decimal and by the base-10
/// inline representation to decode one that is an `f64`.
pub(crate) fn decimal_to_f64_exact(mantissa: i64, exp: i32) -> Option<f64> {
    if mantissa == 0 {
        return Some(0.0);
    }
    if exp >= 0 {
        // A (possibly large) integer; exact iff it fits the `f64` mantissa.
        let v = decimal_int_value(mantissa, exp)?;
        i128_fits_f64(v).then_some(v as f64)
    } else {
        // `value == num / 2^k` after dividing out `5^k` from `10^k = 2^k * 5^k`;
        // exact iff `num` fits `f64` (dividing an exact integer by a power of two is
        // itself exact). Everything fits `i64` for the inline exponent range.
        let k = (-exp) as u32;
        let p5 = 5i64.pow(k);
        if mantissa % p5 != 0 {
            return None;
        }
        let num = mantissa / p5;
        i64_fits_f64(num).then_some(num as f64 / (1u64 << k) as f64)
    }
}

/// `true` if the `i64` value is exactly representable as an `f64`.
fn i64_fits_f64(v: i64) -> bool {
    v == 0 || 64 - v.unsigned_abs().leading_zeros() - v.unsigned_abs().trailing_zeros() <= 53
}

/// `true` if the (larger) `i128` value is exactly representable as an `f64`.
fn i128_fits_f64(v: i128) -> bool {
    v == 0 || 128 - v.unsigned_abs().leading_zeros() - v.unsigned_abs().trailing_zeros() <= 53
}

/// The nearest `f64` to `mantissa * 10^exp`, correctly rounded — even for a
/// mantissa above `2^53` (reachable via a non-`f64` `Decimal`) — and without
/// allocating.
pub(crate) fn decimal_to_f64_lossy(mantissa: i64, exp: i32) -> f64 {
    if exp >= 0 {
        // An exact integer; the `i128 -> f64` cast rounds correctly.
        return (i128::from(mantissa) * 10i128.pow(exp as u32)) as f64;
    }
    let m_abs = mantissa.unsigned_abs();
    let denom = 10u128.pow((-exp) as u32);
    let v = if m_abs < (1 << 53) {
        // `m_abs` and `10^k` are both exact `f64`s, so one division rounds
        // correctly.
        m_abs as f64 / denom as f64
    } else {
        // `m_abs >= 2^53` means `|value| >= 2^29`, so scaling by `2^64` leaves
        // ample guard bits: integer-divide, fold the remainder into a sticky bit,
        // then let the correctly-rounded `u128 -> f64` cast finish the rounding.
        let scaled = u128::from(m_abs) << 64;
        let mut q = scaled / denom;
        if scaled % denom != 0 {
            q |= 1;
        }
        q as f64 / 18_446_744_073_709_551_616.0 // 2^64
    };
    if mantissa < 0 {
        -v
    } else {
        v
    }
}

/// Compares `mantissa * 10^exp` to the integer `n`, exactly.
fn cmp_decimal_int(mantissa: i64, exp: i32, n: i128) -> Ordering {
    if exp >= 0 {
        (i128::from(mantissa) * 10i128.pow(exp as u32)).cmp(&n)
    } else {
        // `mantissa / 10^k` vs `n` ⟺ `mantissa` vs `n * 10^k`.
        i128::from(mantissa).cmp(&(n * 10i128.pow((-exp) as u32)))
    }
}

/// Compares two exact decimals.
fn cmp_decimal_decimal(m1: i64, e1: i32, m2: i64, e2: i32) -> Ordering {
    let de = e1 - e2;
    if de >= 0 {
        (i128::from(m1) * 10i128.pow(de as u32)).cmp(&i128::from(m2))
    } else {
        i128::from(m1).cmp(&(i128::from(m2) * 10i128.pow((-de) as u32)))
    }
}

/// `mantissa * 10^exp` with trailing decimal zeros removed, so equal decimals
/// share one form (used for hashing).
fn canonical_decimal(mut mantissa: i64, mut exp: i32) -> (i64, i32) {
    if mantissa == 0 {
        return (0, 0);
    }
    while mantissa % 10 == 0 {
        mantissa /= 10;
        exp += 1;
    }
    (mantissa, exp)
}

/// `x << n`, or `None` when the value (not just the shift amount) would overflow.
fn shl_u128(x: u128, n: u32) -> Option<u128> {
    if x == 0 {
        Some(0)
    } else if n <= x.leading_zeros() {
        Some(x << n)
    } else {
        None
    }
}

/// Decomposes a finite, positive `f64` into `(frac, exp2)` with `v == frac * 2^exp2`.
fn f64_frac_exp(v: f64) -> (u64, i32) {
    let bits = v.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    if raw_exp == 0 {
        (frac, -1074) // subnormal
    } else {
        (frac | 0x0010_0000_0000_0000, raw_exp - 1075)
    }
}

/// Compares a `u128` to a finite, non-negative float exactly.
fn cmp_u128_f64(a: u128, b: f64) -> Ordering {
    const U128_RANGE: f64 = 340_282_366_920_938_463_463_374_607_431_768_211_456.0; // 2^128
    if b >= U128_RANGE {
        return Ordering::Less; // b >= 2^128 > a
    }
    let bt = b.trunc(); // now in [0, 2^128), so `bt as u128` is exact
    match a.cmp(&(bt as u128)) {
        Ordering::Equal => cmp_by_fraction(b, bt),
        ord => ord,
    }
}

/// Compares `m_abs * 10^exp` to a finite, positive float `v`, exactly.
fn cmp_decimal_magnitude(m_abs: u64, exp: i32, v: f64) -> Ordering {
    if exp >= 0 {
        // `m_abs * 10^exp` is an integer (it fits `u128` for the inline range).
        cmp_u128_f64(u128::from(m_abs) * 10u128.pow(exp as u32), v)
    } else {
        // `|d| = m_abs / 10^k` vs `v = frac * 2^fe`. Clearing `10^k = 2^k * 5^k`:
        // compare `m_abs` to `frac * 5^k * 2^(fe + k)`.
        let k = (-exp) as u32;
        let (frac, fe) = f64_frac_exp(v);
        let p = u128::from(frac) * 5u128.pow(k); // < 2^70
        let s = fe + k as i32;
        if s >= 0 {
            match shl_u128(p, s as u32) {
                Some(rhs) => u128::from(m_abs).cmp(&rhs),
                None => Ordering::Less, // rhs overflows u128 -> |d| < v
            }
        } else {
            match shl_u128(u128::from(m_abs), (-s) as u32) {
                Some(lhs) => lhs.cmp(&p),
                None => Ordering::Greater, // lhs overflows u128 -> |d| > v
            }
        }
    }
}

/// Compares `mantissa * 10^exp` to a finite float exactly. A `Decimal` is, by
/// construction, never exactly an `f64`, so this never returns `Equal`.
fn cmp_decimal_f64(mantissa: i64, exp: i32, v: f64) -> Ordering {
    let d_neg = mantissa < 0;
    if v == 0.0 {
        return if d_neg {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if d_neg != (v < 0.0) {
        return if d_neg {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let ord = cmp_decimal_magnitude(mantissa.unsigned_abs(), exp, v.abs());
    if d_neg {
        ord.reverse()
    } else {
        ord
    }
}

// These exercise the `Decimal` variant and its arithmetic, which are always
// compiled (though only the base-10 inline representation produces one), so they
// run in either configuration.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(nv: NumVal) -> u64 {
        let mut h = DefaultHasher::new();
        nv.hash(&mut h);
        h.finish()
    }

    #[test]
    fn from_f64_normalises_integers_at_the_i64_u64_boundaries() {
        // 2^63 (== i64::MAX + 1) and 2^64 (== u64::MAX + 1) are exact f64s; the
        // integer maxima themselves are not representable.
        let two63 = 9_223_372_036_854_775_808.0_f64;
        let two64 = 18_446_744_073_709_551_616.0_f64;

        // The largest i64-valued f64 is 2^63 - 1024; it reduces to `Int`. 2^63 itself
        // exceeds i64::MAX, so it spills to `UInt`.
        assert_eq!(
            NumVal::from_f64(two63 - 1024.0).to_i64(),
            Some(i64::MAX - 1023)
        );
        assert_eq!(NumVal::from_f64(two63).to_i64(), None);
        assert_eq!(NumVal::from_f64(two63).to_u64(), Some(1 << 63));

        // i64::MIN is exactly -2^63 and stays `Int` (the lower bound is inclusive).
        assert_eq!(NumVal::from_f64(-two63).to_i64(), Some(i64::MIN));

        // The largest u64-valued f64 is 2^64 - 2048; it reduces to `UInt`. 2^64 itself
        // exceeds u64::MAX, so it stays `Float`.
        assert_eq!(
            NumVal::from_f64(two64 - 2048.0).to_u64(),
            Some(u64::MAX - 2047)
        );
        assert_eq!(NumVal::from_f64(two64).to_u64(), None);
        assert_eq!(NumVal::from_f64(two64).to_f64(), Some(two64));
    }

    #[test]
    fn decimal_extracts_integers_but_not_fractions() {
        // 0.1 is a fraction: not an integer, not an exact f64 — stays a `Decimal`.
        let tenth = NumVal::from_decimal(1, -1);
        assert_eq!(tenth.to_i64(), None);
        assert_eq!(tenth.to_u64(), None);
        assert_eq!(tenth.to_f64(), None);
        assert_eq!(tenth.to_f64_lossy(), 0.1);

        // 20 * 10^-1 == 2 is an integer-valued decimal: normalises to `Int`.
        let two = NumVal::from_decimal(20, -1);
        assert_eq!(two.to_i64(), Some(2));
        assert_eq!(two.to_u64(), Some(2));
    }

    #[test]
    fn decimal_compares_exactly_against_decimals_and_integers() {
        let tenth = NumVal::from_decimal(1, -1); // 0.1
        let three_tenths = NumVal::from_decimal(3, -1); // 0.3
        let tenth_scaled = NumVal::from_decimal(10, -2); // 0.10 == 0.1
        assert_eq!(tenth.cmp(three_tenths), Ordering::Less);
        assert_eq!(tenth.cmp(tenth_scaled), Ordering::Equal);
        assert_eq!(tenth.cmp(NumVal::from_i64(0)), Ordering::Greater);
        assert_eq!(tenth.cmp(NumVal::from_i64(1)), Ordering::Less);

        // An integer-valued decimal orders exactly with the equal integer.
        let two = NumVal::from_decimal(20, -1);
        assert_eq!(two.cmp(NumVal::from_i64(2)), Ordering::Equal);
        assert_eq!(two.cmp(NumVal::from_u64(2)), Ordering::Equal);
        assert_eq!(NumVal::from_i64(2).cmp(two), Ordering::Equal);
    }

    #[test]
    fn decimal_hash_stays_consistent_with_equality() {
        // An integer-valued decimal hashes like the equal integer.
        let two_dec = NumVal::from_decimal(20, -1);
        assert_eq!(hash_of(two_dec), hash_of(NumVal::from_i64(2)));
        assert_eq!(hash_of(two_dec), hash_of(NumVal::from_u64(2)));

        // Equal fractions hash alike (canonical form), regardless of how the
        // mantissa/exponent are written.
        let a = NumVal::from_decimal(1, -1);
        let b = NumVal::from_decimal(10, -2);
        assert_eq!(hash_of(a), hash_of(b));

        // The exact decimal 0.1 and the f64 0.1 are *different* numbers, so they
        // are unequal — and hashing them differently is therefore allowed.
        assert_ne!(a.cmp(NumVal::from_f64(0.1)), Ordering::Equal);
        assert_eq!(a.cmp(NumVal::from_f64(0.1)), Ordering::Less);
    }

    #[test]
    fn decimal_to_f64_lossy_is_correctly_rounded() {
        // The result must match the correctly-rounded std parser, including for a
        // mantissa above 2^53 (which a naive `m as f64 / 10^k` would double-round).
        // The string parse here is only the test oracle, not the implementation.
        for &(m, e) in &[
            (1_i64, -1),              // 0.1
            (3, -1),                  // 0.3
            (9492881567496375, -1),   // 949288156749637.5 (exact f64)
            (12345678901234567, -1),  // 1234567890123456.7 (> 2^53 mantissa)
            (99999999999999999, -7),  // 9999999999.9999999
            (-12345678901234567, -3), // negative
            (36028797018963967, -1),  // (2^55 - 1) * 10^-1
            (36028797018963967, 7),   // exp >= 0 path
            (7, -7),                  // 7e-7 (tiny, small mantissa)
        ] {
            let oracle: f64 = format!("{}e{}", m, e).parse().unwrap();
            assert_eq!(decimal_to_f64_lossy(m, e), oracle, "{} e {}", m, e);
        }
    }
}
