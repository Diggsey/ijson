//! The numeric value model shared by every number representation.
//!
//! [`NumVal`] is a number reduced to a form suitable for *exact* comparison,
//! hashing, and conversion, independent of how it is stored. Each number
//! representation decodes its storage to a `NumVal` (via `ValueRepr::num_val`); the
//! methods on `NumVal` then implement every numeric value operation once, shared by
//! all representations. The free helpers below are the scalar arithmetic those
//! methods are built from.
//!
//! The `Big` variant and everything that serves it (`canonicalise`, `render_decimal`,
//! the arbitrary-precision comparison) are always compiled — and unit-tested, so both
//! feature configurations exercise them — but only `arbitrary_precision` ever *builds*
//! such a value, so without it they are dead outside the tests.
#![cfg_attr(not(feature = "arbitrary_precision"), allow(dead_code))]

use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hasher;

use super::bigint;

// A number reduced to a canonical form suitable for exact numeric comparison,
// hashing, and conversion. Construction — via `from_i64`/`from_u64`/`from_f64`/
// `from_decimal`/`canonicalise` — normalises every value into a single variant, so
// equal numbers are always the same variant:
//
//   - `Int`     : an integer in `i64` range.
//   - `UInt`    : an integer in `(i64::MAX, u64::MAX]`.
//   - `Float`   : an exact `f64` that is *not* an integer in `i64`/`u64` range (a
//                 fraction, or an integer-valued float beyond `u64`); never `±0.0`.
//   - `Decimal` : an exact `mantissa * 10^exp` that is neither an integer nor an exact
//                 `f64` (e.g. the fraction `0.1`), small enough for the fixed-width
//                 arithmetic below (see `fits_decimal`). Only the base-10 inline
//                 representation (`arbitrary_precision`) produces one.
//   - `Big`     : the residual — the same thing with an arbitrary-precision mantissa,
//                 borrowed from the heap decimal representation that stores it.
//
// Because the variants are disjoint, equality and hashing reduce to the variant and
// its fields, and the accessors mostly select a single variant. `Decimal`/`Big` and
// their arithmetic are always compiled and unit-tested, even when the active
// representation never produces one.
//
// The lifetime is `Big`'s: its mantissa is borrowed from the value it was decoded
// from, which is why `NumVal` is a *view* rather than an owned number. It stays `Copy`.
#[derive(Clone, Copy)]
pub(crate) struct NumVal<'a>(Repr<'a>);

#[derive(Clone, Copy)]
enum Repr<'a> {
    Int(i64),
    UInt(u64),
    Float(f64),
    Decimal { mantissa: i64, exp: i32 },
    Big(BigDec<'a>),
}

/// An exact `(-1)^negative * magnitude * 10^exp` with an arbitrary-precision mantissa.
///
/// The magnitude is normalised (see [`bigint`]), never zero, and never divisible by
/// ten — which makes the form *unique*, so two `Big`s are equal exactly when their
/// fields are, and hashing them is structural.
#[derive(Clone, Copy)]
pub(crate) struct BigDec<'a> {
    negative: bool,
    magnitude: &'a [u64],
    exp: i32,
}

/// The owned counterpart, used only to give a small number a `BigDec` view for the
/// arbitrary-precision comparison path.
struct OwnedBigDec {
    negative: bool,
    magnitude: Vec<u64>,
    exp: i32,
}

impl OwnedBigDec {
    fn as_ref(&self) -> BigDec<'_> {
        BigDec {
            negative: self.negative,
            magnitude: &self.magnitude,
            exp: self.exp,
        }
    }
}

/// The canonical form of a parsed decimal literal, from [`canonicalise`] — the single
/// place that decides how a number entering the library is stored.
///
/// It always carries the exact `(-1)^negative * magnitude * 10^exp`, because the heap
/// decimal representation may have to store it even when it *would* fit something
/// narrower (a float literal whose value happens to be an integer still has to record
/// its decimal point, and only the decimal representation has room for that). `small`
/// says which fixed-width variant it reduces to, so the caller can pick the cheapest
/// representation that preserves the literal.
pub(crate) struct Canonical {
    pub(crate) negative: bool,
    pub(crate) magnitude: Vec<u64>,
    pub(crate) exp: i32,
    /// The fixed-width variant this value reduces to, or `None` if it needs arbitrary
    /// precision.
    pub(crate) small: Option<NumVal<'static>>,
}

impl<'a> NumVal<'a> {
    /// An `i64`.
    pub(crate) fn from_i64(x: i64) -> NumVal<'static> {
        NumVal(Repr::Int(x))
    }

    /// A `u64` — stored as `Int` when it also fits `i64`.
    pub(crate) fn from_u64(x: u64) -> NumVal<'static> {
        match i64::try_from(x) {
            Ok(i) => NumVal(Repr::Int(i)),
            Err(_) => NumVal(Repr::UInt(x)),
        }
    }

    /// A finite `f64` — reduced to `Int`/`UInt` when it is exactly an integer in
    /// range (so `1e18` and the integer `10^18` become the same value).
    pub(crate) fn from_f64(x: f64) -> NumVal<'static> {
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
    ///
    /// `(mantissa, exp)` must lie in the `Decimal` domain once canonical (see
    /// [`fits_decimal`]); the base-10 inline representation guarantees it, and
    /// [`canonicalise`] routes everything else to `Big`.
    pub(crate) fn from_decimal(mantissa: i64, exp: i32) -> NumVal<'static> {
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
        debug_assert!(
            fits_decimal(mantissa, i64::from(exp)),
            "{}e{} is outside the `Decimal` domain, so its exact arithmetic may overflow",
            mantissa,
            exp
        );
        NumVal(Repr::Decimal { mantissa, exp })
    }

    /// A canonical decimal held by the heap representation.
    ///
    /// This *reduces*, exactly as [`canonicalise`] does — it does not simply wrap the
    /// fields as a `Big`. The heap decimal representation stores some values a narrower
    /// one could also hold (`1.2345678901234567891e19` is an integer, but only the
    /// decimal representation can record that it was written as a float), and the same
    /// number reached through two representations has to decode to the *same* variant,
    /// or it would compare and hash as two different numbers.
    ///
    /// The one reduction it does not attempt is the exact-`f64` case, which needs the
    /// decimal digits: it cannot miss one, because an exact `f64` is never stored here
    /// (an `f64` representation holds those, and records the decimal point besides).
    pub(crate) fn from_big(negative: bool, magnitude: &'a [u64], exp: i32) -> NumVal<'a> {
        debug_assert!(
            !bigint::is_zero(magnitude) && bigint::rem_small(magnitude, 10) != 0,
            "a stored decimal's magnitude is canonical: non-zero and not divisible by ten"
        );
        if let Some(nv) = reduce_fixed(negative, magnitude, exp) {
            return nv;
        }
        NumVal(Repr::Big(BigDec {
            negative,
            magnitude,
            exp,
        }))
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
            // Neither a `Decimal` nor a `Big` is exactly an `f64`: both constructors
            // reduce such a value to `Float` instead.
            Repr::Decimal { .. } | Repr::Big(_) => None,
        }
    }

    /// The (possibly lossy) `f64` value.
    pub(crate) fn to_f64_lossy(self) -> f64 {
        match self.0 {
            Repr::Int(x) => x as f64,
            Repr::UInt(x) => x as f64,
            Repr::Float(x) => x,
            Repr::Decimal { mantissa, exp } => decimal_to_f64_lossy(mantissa, exp),
            Repr::Big(b) => b.to_f64_lossy(),
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
            // A `Big`'s form is unique (see `BigDec`), so its fields hash directly.
            Repr::Big(b) => {
                state.write_u8(u8::from(b.negative));
                state.write_i32(b.exp);
                for &limb in b.magnitude {
                    state.write_u64(limb);
                }
            }
        }
    }

    /// The exact total order over two numbers, across every variant.
    pub(crate) fn cmp(self, other: NumVal<'_>) -> Ordering {
        use Repr::{Big, Decimal, Float, Int, UInt};
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
            // Once a `Big` is involved, the fixed-width arithmetic above cannot hold
            // the operands: give the *other* side a `BigDec` view — every variant is
            // exactly a finite decimal — and compare in arbitrary precision. This is
            // the only path that allocates, and no all-small comparison reaches it.
            (Big(x), Big(y)) => x.cmp(y),
            (Big(x), _) => {
                let y = other.to_owned_big();
                x.cmp(y.as_ref())
            }
            (_, Big(y)) => {
                let x = self.to_owned_big();
                x.as_ref().cmp(y)
            }
        }
    }

    /// The exact value as JSON text, or `None` when `serde` can already carry it
    /// exactly — an integer in `i64`/`u64` range, or an exact `f64`.
    ///
    /// A `Decimal` or a `Big` is by construction *neither*, so serializing it through
    /// `f64` would silently change it (`1e-400` would become `0.0`, and a 25-digit
    /// fraction would round). Those are exactly the numbers that have to be written from
    /// their own digits.
    ///
    /// `has_decimal_point` is the literal's shape, not part of the value: it decides
    /// only whether the text must read back as a JSON float (see [`render_decimal`]).
    pub(crate) fn exact_json(self, has_decimal_point: bool) -> Option<String> {
        match self.0 {
            // `serialize_i64`/`serialize_u64` write an integer exactly, in integer
            // syntax — right for an integer literal.
            Repr::Int(_) | Repr::UInt(_) if !has_decimal_point => None,
            // `serialize_f64` writes an exact `f64` exactly, in float syntax — right for
            // a float literal.
            Repr::Float(_) if has_decimal_point => None,
            // Everything else is a number `serde` cannot write without changing it. Two
            // cases beyond the exact decimals, both where the value's shape and the
            // literal's disagree, and every primitive gets one of them wrong:
            //
            //   - an integer literal beyond `u64` that is exactly an `f64` (`1e20`) —
            //     float syntax would make it a float;
            //   - a float literal whose value is an integer but not an exact `f64`
            //     (`1.2345678901234567891e19`) — integer syntax would make it an
            //     integer, and an `f64` would round it.
            //
            // Both are written from their digits, in the shape the literal had.
            _ => {
                let exact = self.to_owned_big();
                Some(render_decimal(
                    exact.negative,
                    &bigint::to_decimal_digits(&exact.magnitude),
                    exact.exp,
                    has_decimal_point,
                ))
            }
        }
    }

    /// This number as an exact big decimal. Every variant is exactly a finite decimal:
    /// an integer is one at exponent zero, and a binary float `frac * 2^e` is one
    /// because `2^-k == 5^k * 10^-k`.
    fn to_owned_big(self) -> OwnedBigDec {
        let (negative, magnitude, exp) = match self.0 {
            Repr::Int(x) => (x < 0, bigint::from_u64(x.unsigned_abs()), 0),
            Repr::UInt(x) => (false, bigint::from_u64(x), 0),
            Repr::Decimal { mantissa, exp } => {
                (mantissa < 0, bigint::from_u64(mantissa.unsigned_abs()), exp)
            }
            Repr::Float(x) => {
                let (frac, exp2) = f64_frac_exp(x.abs());
                let mut magnitude = bigint::from_u64(frac);
                let exp = if exp2 >= 0 {
                    bigint::shl(&mut magnitude, exp2 as u64);
                    0
                } else {
                    // `frac / 2^k == frac * 5^k / 10^k`, so a power of two in the
                    // denominator is exactly a decimal, at exponent `-k == exp2` — with
                    // `k` up to 1074 for a subnormal, which is why the mantissa grows.
                    bigint::mul_pow5(&mut magnitude, (-exp2) as u64);
                    exp2
                };
                (x < 0.0, magnitude, exp)
            }
            Repr::Big(b) => (b.negative, b.magnitude.to_vec(), b.exp),
        };
        OwnedBigDec {
            negative,
            magnitude,
            exp,
        }
    }
}

impl BigDec<'_> {
    /// The exact total order over two arbitrary-precision decimals.
    fn cmp(self, other: BigDec<'_>) -> Ordering {
        // A `Big` is never zero, but the *other* operand's view can be — comparing an
        // arbitrary-precision number against plain `0` is an ordinary thing to do — so
        // zero is settled before the sign, and never by it. (Only `Int(0)` is zero, and
        // its view is non-negative, so there is no `-0` to disagree with `0`.)
        match (
            bigint::is_zero(self.magnitude),
            bigint::is_zero(other.magnitude),
        ) {
            (true, true) => return Ordering::Equal,
            (true, false) if other.negative => return Ordering::Greater,
            (true, false) => return Ordering::Less,
            (false, true) if self.negative => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {}
        }
        match (self.negative, other.negative) {
            (false, true) => return Ordering::Greater,
            (true, false) => return Ordering::Less,
            _ => {}
        }
        let ord = cmp_magnitude(self.magnitude, self.exp, other.magnitude, other.exp);
        if self.negative {
            ord.reverse()
        } else {
            ord
        }
    }

    /// The nearest `f64`, correctly rounded.
    ///
    /// Rendering the exact digits and handing them to the standard library's decimal
    /// parser — which is correctly rounded — is both simpler and easier to trust than
    /// a bespoke bignum-to-binary rounding, and this is the arbitrary-precision slow
    /// path: no small number reaches it. Out-of-range values parse to `±inf`/`0.0`,
    /// which is the correct rounding.
    fn to_f64_lossy(self) -> f64 {
        self.to_exponential()
            .parse()
            .expect("`to_exponential` emits a well-formed decimal literal")
    }

    /// The exact value in JSON e-notation (`-123e-4`).
    fn to_exponential(self) -> String {
        let digits = bigint::to_decimal_digits(self.magnitude);
        let mut s = String::with_capacity(digits.len() + 13);
        if self.negative {
            s.push('-');
        }
        // Safety: `to_decimal_digits` emits ASCII digits.
        s.push_str(std::str::from_utf8(&digits).expect("ASCII digits"));
        s.push('e');
        s.push_str(&self.exp.to_string());
        s
    }
}

/// Formats a number the way `serde_json` would (integer if it is one).
impl Debug for NumVal<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(x) = self.to_i64() {
            Debug::fmt(&x, f)
        } else if let Some(x) = self.to_u64() {
            Debug::fmt(&x, f)
        } else if let Repr::Big(b) = self.0 {
            // A `Big` has more precision than an `f64` can show, by construction.
            f.write_str(&b.to_exponential())
        } else {
            Debug::fmt(&self.to_f64_lossy(), f)
        }
    }
}

/// Writes the exact value `(-1)^negative * digits * 10^exp` as JSON text.
///
/// `digits` is the canonical magnitude — non-zero, no leading zeros. Two properties are
/// load-bearing, because this is what a number is *serialized* as, and re-parsing it has
/// to give back the number we started with:
///
///  - **The value is exact.** Never routed through an `f64`, which is the whole point:
///    the numbers that reach here are precisely the ones an `f64` would change.
///  - **The JSON shape is preserved.** A float must read back as a float, so when
///    `has_decimal_point` the text always carries a `.` or an `e` — including for a
///    float whose value happens to be a whole number (`1.2345678901234567891e19`),
///    which would otherwise come back as an integer. An integer, conversely, is never
///    written in e-notation, which JSON would read as a float.
fn render_decimal(negative: bool, digits: &[u8], exp: i32, has_decimal_point: bool) -> String {
    debug_assert!(!digits.is_empty() && digits[0] != b'0');
    let digits = std::str::from_utf8(digits).expect("ASCII digits");
    let n = digits.len() as i64;
    // Where the decimal point falls, counted in digits from the start of `digits`.
    let point = n + i64::from(exp);

    let mut s = String::new();
    if negative {
        s.push('-');
    }

    // An integer is written out in full, however many zeros that takes. It cannot run
    // away: an integer literal has no exponent of its own, so `exp` is only what
    // canonicalisation folded *out* of the digits, and writing them back costs exactly
    // what the literal already spent.
    if !has_decimal_point {
        debug_assert!(exp >= 0, "an integer has a non-negative exponent");
        s.push_str(digits);
        s.extend(std::iter::repeat_n('0', exp as usize));
        return s;
    }

    // A float, positionally while that stays compact — and in e-notation otherwise, so
    // an extreme exponent costs a few characters rather than gigabytes of zeros.
    const MAX_LEADING_ZEROS: i64 = 6;
    const MAX_TRAILING_ZEROS: i64 = 20;
    if point <= -MAX_LEADING_ZEROS || point > n + MAX_TRAILING_ZEROS {
        // `d.ddd e±X`, the leading digit before the point.
        s.push_str(&digits[..1]);
        if n > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push_str(&(point - 1).to_string());
    } else if point <= 0 {
        s.push_str("0.");
        s.extend(std::iter::repeat_n('0', (-point) as usize));
        s.push_str(digits);
    } else if point >= n {
        s.push_str(digits);
        s.extend(std::iter::repeat_n('0', (point - n) as usize));
        // The value is a whole number, but the literal was a float and must stay one.
        s.push_str(".0");
    } else {
        let (whole, fraction) = digits.split_at(point as usize);
        s.push_str(whole);
        s.push('.');
        s.push_str(fraction);
    }
    s
}

/// The canonical form of the exact value `(-1)^negative * digits * 10^exp`, where
/// `digits` is a run of ASCII decimal digits.
///
/// This is the *single* place that decides which `NumVal` variant a parsed number
/// reduces to. Together with [`NumVal::from_big`], which repeats the same reduction on
/// the way back out of the heap representation, it is what keeps the variants disjoint
/// across representations — and hence `hash` consistent with `cmp`.
///
/// `exp + digits.len()` must not overflow an `i32`; the parser enforces it.
pub(crate) fn canonicalise(negative: bool, digits: &[u8], exp: i32) -> Canonical {
    let mut magnitude = bigint::from_decimal_digits(digits);
    if bigint::is_zero(&magnitude) {
        return Canonical {
            negative: false,
            magnitude,
            exp: 0,
            small: Some(NumVal::from_i64(0)),
        };
    }

    // Divide out every factor of ten, which makes `magnitude * 10^exp` the *unique*
    // decimal form of this value — the invariant `Big` compares and hashes on. Each
    // step consumes a digit, so `exp` can grow by at most `digits.len()`.
    let mut exp = i64::from(exp);
    while bigint::rem_small(&magnitude, 10) == 0 {
        bigint::div_small(&mut magnitude, 10);
        exp += 1;
    }
    let exp = i32::try_from(exp).expect("the parser bounds `exp + digits.len()` to an `i32`");

    let small = reduce_fixed(negative, &magnitude, exp);

    Canonical {
        negative,
        magnitude,
        exp,
        small,
    }
}

/// The fixed-width variant the canonical `(-1)^negative * magnitude * 10^exp` reduces
/// to, or `None` if it genuinely needs arbitrary precision.
///
/// This is the *complete* reduction, and the only one. [`canonicalise`] uses it when a
/// number enters the library, and [`NumVal::from_big`] runs it again on every decode of
/// a stored decimal — so a number reduces to the same variant no matter which
/// representation happens to hold it. That has to hold across the *integer* and *float*
/// representations alike: `2^64` is exactly an `f64`, and is held as one when it is
/// written `1.8446744073709552e19` but as a decimal when it is written out as an
/// integer (only that has room to record the absent decimal point). If the two decoded
/// differently they would be two different numbers.
fn reduce_fixed(negative: bool, magnitude: &[u64], exp: i32) -> Option<NumVal<'static>> {
    // An integer in `u64` range. (With `exp` negative the value is not an integer at
    // all: the canonical magnitude is not divisible by ten, so dividing it by a power
    // of ten cannot come out whole.)
    if exp >= 0 {
        if let Some(n) = integer_value(magnitude, exp) {
            if !negative {
                return Some(NumVal::from_u64(n));
            }
            // `-n` is exactly an `i64` while `n <= 2^63`, whose negation is `i64::MIN`.
            return i64::try_from(-i128::from(n)).ok().map(NumVal::from_i64);
        }
    }

    // Small enough for the fixed-width `Decimal` arithmetic: hand it to `from_decimal`,
    // which owns the `Int`/`UInt`/`Float`/`Decimal` split within that domain — including
    // its own exact-`f64` test, so `1e20` comes back a `Float` here.
    if let [mantissa] = *magnitude {
        let signed = if negative {
            i64::try_from(-i128::from(mantissa)).ok()
        } else {
            i64::try_from(mantissa).ok()
        };
        if let Some(m) = signed.filter(|&m| fits_decimal(m, i64::from(exp))) {
            return Some(NumVal::from_decimal(m, exp));
        }
    }

    // Too big for that domain, but still possibly an exact `f64` — `2^64` and `2^100`
    // both are, on one significant bit apiece.
    exact_f64(negative, magnitude, exp).map(NumVal::from_f64)
}

/// The exact `f64` value of `(-1)^negative * magnitude * 10^exp`, if it is exactly one.
///
/// Structured as a cheap rejection followed by an expensive decision, because
/// [`reduce_fixed`] runs on every comparison and hash of a stored decimal and the answer
/// is nearly always "no". The decision itself is the correctly-rounded conversion
/// checked back against the exact value — the same trick as elsewhere, and far easier to
/// trust than a bespoke bignum-to-binary rounding with its subnormal edge cases.
fn exact_f64(negative: bool, magnitude: &[u64], exp: i32) -> Option<f64> {
    if exp >= 0 {
        // `value == magnitude * 5^exp * 2^exp`. Multiplying by the *odd* `5^exp` leaves
        // the trailing zeros alone and only lengthens the number, so the significant
        // bits can only grow: the magnitude's own must already fit an `f64`'s 53. And
        // `5^exp` adds more than two bits per power, so past 22 nothing can fit.
        if exp > 22 || bigint::significant_bits(magnitude) > 53 {
            return None;
        }
    } else {
        // `value == magnitude / (2^k * 5^k)` is exact only if `5^k` divides the
        // magnitude — which, for a canonical magnitude (never divisible by ten), is
        // already unusual. Testing as many powers of five as fit a `u64` rejects all but
        // a few in one pass.
        let k = (-i64::from(exp)).min(POW5_PROBE_EXP) as u32;
        if bigint::rem_small(magnitude, 5u64.pow(k)) != 0 {
            return None;
        }
    }

    let big = BigDec {
        negative,
        magnitude,
        exp,
    };
    let f = big.to_f64_lossy();
    (f.is_finite() && NumVal(Repr::Big(big)).cmp(NumVal::from_f64(f)) == Ordering::Equal)
        .then_some(f)
}

/// The largest power of five that fits a `u64`, used to probe divisibility above.
const POW5_PROBE_EXP: i64 = 27;

/// The exact value of `magnitude * 10^exp` if it fits a `u64`.
fn integer_value(magnitude: &[u64], exp: i32) -> Option<u64> {
    debug_assert!(exp >= 0);
    // A magnitude of more than one limb already exceeds `u64::MAX`, and scaling by a
    // non-negative power of ten only makes it larger — so the scaling never has to be
    // performed to rule it out, and an enormous exponent costs nothing here. What is
    // left is a single limb, where `checked_pow`/`checked_mul` decide it exactly.
    let mantissa = match *magnitude {
        [] => return Some(0),
        [m] => m,
        _ => return None,
    };
    mantissa.checked_mul(10u64.checked_pow(u32::try_from(exp).ok()?)?)
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
// A `Decimal { mantissa, exp }` is the exact value `mantissa * 10^exp`, restricted to
// the domain `fits_decimal` defines so that the fixed-width arithmetic below cannot
// overflow.

/// The least exponent a `Decimal` may have. The base-10 inline representation's is
/// `-7`, and canonicalisation (which only divides out factors of ten) can never lower
/// it, so this covers everything that representation produces.
const DECIMAL_MIN_EXP: i64 = -7;

/// The greatest *decimal magnitude* — `digits(mantissa) + exp`, the position of the
/// leading digit — a `Decimal` may have. Canonicalisation preserves this sum (dropping
/// a trailing zero costs a digit and gains an exponent), and the inline representation
/// bounds it at `17 + 7 == 24` (a 56-bit signed mantissa at exponent 7), so 26 covers
/// it with headroom.
const DECIMAL_MAX_MAGNITUDE: i64 = 26;

/// Whether a *canonical* `mantissa * 10^exp` belongs to the `Decimal` variant.
///
/// The single boundary between `Decimal` and `Big`: [`NumVal::from_decimal`] asserts
/// its canonical output satisfies this, and [`canonicalise`] sends anything that does
/// not to `Big`. Because both consult one predicate they cannot drift apart, so no
/// number can reach both variants — which is what makes the variants disjoint, and
/// hashing them structurally sound.
///
/// It also bounds every `i128` product in this section. With a magnitude of at most
/// `10^26` and an exponent of at least `-7`, the largest is the rescaling in
/// [`cmp_decimal_decimal`], at `10^(26 + 7) == 10^33` — comfortably inside `i128::MAX`
/// (about `1.7 * 10^38`).
fn fits_decimal(mantissa: i64, exp: i64) -> bool {
    exp >= DECIMAL_MIN_EXP
        && i64::from(decimal_digits(mantissa.unsigned_abs())) + exp <= DECIMAL_MAX_MAGNITUDE
}

/// The number of decimal digits in a `u64` (zero has one).
fn decimal_digits(mut x: u64) -> u32 {
    let mut digits = 1;
    while x >= 10 {
        x /= 10;
        digits += 1;
    }
    digits
}

// --- Exact arbitrary-precision comparison -----------------------------------

/// Brackets the number of decimal digits in a non-zero magnitude, from its bit length
/// alone — no division. `2^(bits-1) <= v < 2^bits`, so `log10(v)` lies between
/// `(bits-1) * log10(2)` and `bits * log10(2)`; a digit of slack either way absorbs any
/// rounding in the `f64` logarithm.
fn decimal_digit_bounds(magnitude: &[u64]) -> (i64, i64) {
    debug_assert!(!bigint::is_zero(magnitude));
    let bits = bigint::bit_len(magnitude);
    let lo = ((bits - 1) as f64 * std::f64::consts::LOG10_2).floor() as i64;
    let hi = (bits as f64 * std::f64::consts::LOG10_2).floor() as i64 + 2;
    (lo, hi)
}

/// Compares `a * 10^ea` to `b * 10^eb` exactly, for non-zero magnitudes.
fn cmp_magnitude(a: &[u64], ea: i32, b: &[u64], eb: i32) -> Ordering {
    // The decimal magnitude — where the leading digit sits — almost always settles it
    // without touching the digits, and does so however far apart the exponents are.
    // (`i64`, because a magnitude's digit count can exceed an `i32`.)
    let (a_lo, a_hi) = decimal_digit_bounds(a);
    let (b_lo, b_hi) = decimal_digit_bounds(b);
    let (ea, eb) = (i64::from(ea), i64::from(eb));
    if a_hi + ea < b_lo + eb {
        return Ordering::Less;
    }
    if a_lo + ea > b_hi + eb {
        return Ordering::Greater;
    }

    // Too close to call: scale both to a common exponent and compare the integers.
    // Reaching here means the brackets overlap, so the exponents differ by no more than
    // the difference in digit counts (plus the slack) — the rescaling is bounded by the
    // operands' own size and cannot blow up, however extreme the exponents themselves.
    let common = ea.min(eb);
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    bigint::mul_pow10(&mut sa, (ea - common) as u64);
    bigint::mul_pow10(&mut sb, (eb - common) as u64);
    bigint::cmp(&sa, &sb)
}

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

// These exercise the `Decimal` and `Big` variants and their arithmetic, which are
// always compiled (though only `arbitrary_precision` builds a value from them), so
// they run in either configuration.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(nv: NumVal<'_>) -> u64 {
        let mut h = DefaultHasher::new();
        nv.hash(&mut h);
        h.finish()
    }

    /// A decimal literal, as the parser hands it over: `(-1)^negative * digits * 10^exp`.
    type Lit = (bool, &'static str, i32);

    /// Canonicalises a literal, keeping any arbitrary-precision mantissa alive in
    /// `storage` — exactly what the heap representation will do for a `Canonical::Big`.
    fn num_val<'a>(
        storage: &'a mut Option<OwnedBigDec>,
        (negative, digits, exp): Lit,
    ) -> NumVal<'a> {
        let c = canonicalise(negative, digits.as_bytes(), exp);
        if let Some(nv) = c.small {
            return nv;
        }
        let owned = storage.insert(OwnedBigDec {
            negative: c.negative,
            magnitude: c.magnitude,
            exp: c.exp,
        });
        NumVal::from_big(owned.negative, &owned.magnitude, owned.exp)
    }

    /// The literal's `NumVal`, which must not need arbitrary precision.
    fn small(lit: Lit) -> NumVal<'static> {
        canonicalise(lit.0, lit.1.as_bytes(), lit.2)
            .small
            .unwrap_or_else(|| panic!("{}e{} unexpectedly needs arbitrary precision", lit.1, lit.2))
    }

    /// Asserts the literal *does* need arbitrary precision — i.e. that it is genuinely
    /// beyond every fixed-width variant, which is what keeps them disjoint.
    fn assert_big(lit: Lit) {
        assert!(
            canonicalise(lit.0, lit.1.as_bytes(), lit.2).small.is_none(),
            "{}e{} should need arbitrary precision",
            lit.1,
            lit.2
        );
    }

    fn cmp_lit(a: Lit, b: Lit) -> Ordering {
        let (mut sa, mut sb) = (None, None);
        num_val(&mut sa, a).cmp(num_val(&mut sb, b))
    }

    fn hash_lit(lit: Lit) -> u64 {
        let mut storage = None;
        hash_of(num_val(&mut storage, lit))
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

    // --- Arbitrary precision ------------------------------------------------

    #[test]
    fn canonicalise_reduces_to_the_smallest_variant() {
        // Zero, however it is written.
        assert_eq!(small((false, "0", 0)).to_i64(), Some(0));
        assert_eq!(small((false, "0000", 5)).to_i64(), Some(0));
        assert_eq!(small((true, "0", -3)).to_i64(), Some(0));

        // Integers, including ones whose trailing zeros the exponent absorbs and then
        // gives back — the canonical form must not change the value.
        assert_eq!(small((false, "123", 0)).to_i64(), Some(123));
        assert_eq!(small((true, "123", 0)).to_i64(), Some(-123));
        assert_eq!(small((false, "12300", -2)).to_i64(), Some(123));
        assert_eq!(small((false, "123", 2)).to_i64(), Some(12300));

        // The `i64`/`u64` boundaries. `-2^63` is `i64::MIN`; `+2^63` spills to `UInt`;
        // `u64::MAX` has no trailing zeros, so it stays a full 20-digit mantissa.
        assert_eq!(
            small((true, "9223372036854775808", 0)).to_i64(),
            Some(i64::MIN)
        );
        assert_eq!(small((false, "9223372036854775808", 0)).to_i64(), None);
        assert_eq!(
            small((false, "9223372036854775808", 0)).to_u64(),
            Some(1 << 63)
        );
        assert_eq!(
            small((false, "18446744073709551615", 0)).to_u64(),
            Some(u64::MAX)
        );

        // Past `u64`, but 2^64 is an exact `f64`, so it reduces to `Float` — not `Big`.
        assert_eq!(
            small((false, "18446744073709551616", 0)).to_f64(),
            Some(18_446_744_073_709_551_616.0)
        );

        // An exact `f64` fraction whose exponent is *below* the `Decimal` domain
        // (`-8 < -7`): the arbitrary-precision path still has to spot it, or the same
        // number could end up in two variants.
        assert_eq!(small((false, "390625", -8)).to_f64(), Some(0.003_906_25)); // 2^-8

        // 0.1 is neither an integer nor an exact `f64`: a `Decimal`.
        let tenth = small((false, "1", -1));
        assert_eq!(tenth.to_i64(), None);
        assert_eq!(tenth.to_f64(), None);
        assert_eq!(tenth.to_f64_lossy(), 0.1);
    }

    #[test]
    fn canonicalise_spills_only_what_no_fixed_width_variant_holds() {
        // A big integer: past `u64`, and far too many significant bits for an `f64`.
        assert_big((false, "123456789012345678901234567890123", 0));
        // The same value negated — `Big` is not just an unsigned overflow escape.
        assert_big((true, "123456789012345678901234567890123", 0));
        // More precision than an `f64` has, in a fraction.
        assert_big((false, "1234567890123456789012345", -25));
        // A small mantissa, but an exponent no fixed-width variant covers.
        assert_big((false, "1", -30));
        assert_big((false, "7", 40));
        // Extreme exponents must not blow up (the magnitude short-circuit handles them).
        assert_big((false, "3", 1_000_000));
        assert_big((false, "3", -1_000_000));
    }

    #[test]
    fn big_orders_exactly_against_every_other_variant() {
        let big = (false, "123456789012345678901234567890123", 0); // ~1.2e32
        assert_eq!(cmp_lit(big, (false, "1", 0)), Ordering::Greater);
        assert_eq!(cmp_lit((false, "1", 0), big), Ordering::Less);
        assert_eq!(cmp_lit(big, (true, "1", 0)), Ordering::Greater);
        // Against a `Float` on either side of it.
        assert_eq!(cmp_lit(big, (false, "1", 33)), Ordering::Less); // 1e33
        assert_eq!(cmp_lit(big, (false, "1", 32)), Ordering::Greater); // 1e32
                                                                       // Against a `Decimal`.
        assert_eq!(cmp_lit(big, (false, "1", -1)), Ordering::Greater);

        // Sign decides first, whatever the magnitudes: a vast negative is below a
        // vanishing positive.
        assert_eq!(
            cmp_lit((true, "9", 1_000_000), (false, "1", -1_000_000)),
            Ordering::Less
        );

        // Two `Big`s: the exponent, then the digits.
        let a = (false, "1234567890123456789012345", -25); // 0.1234...
        let b = (false, "1234567890123456789012346", -25); // one ulp of the last digit up
        assert_eq!(cmp_lit(a, b), Ordering::Less);
        assert_eq!(cmp_lit(b, a), Ordering::Greater);
        assert_eq!(cmp_lit(a, a), Ordering::Equal);
        // Negated, the order reverses.
        assert_eq!(
            cmp_lit((true, a.1, a.2), (true, b.1, b.2)),
            Ordering::Greater
        );
    }

    #[test]
    fn big_is_never_equal_to_a_fixed_width_variant() {
        // The disjointness invariant, stated as behaviour: a `Big` is by construction
        // not an integer and not an exact `f64`, so it cannot compare equal to one.
        // (If `canonicalise` ever failed to reduce a reducible value, this would be the
        // first thing to break — and it would take hashing down with it.)
        for &lit in &[
            (false, "1234567890123456789012345", -25),
            (false, "123456789012345678901234567890123", 0),
            (false, "1", -30),
        ] {
            let (mut sa, mut sb) = (None, None);
            let a = num_val(&mut sa, lit);
            assert_eq!(a.to_i64(), None);
            assert_eq!(a.to_u64(), None);
            assert_eq!(a.to_f64(), None);

            // The nearest `f64` is close, but never exactly equal.
            let nearest = num_val(&mut sb, lit).to_f64_lossy();
            assert_ne!(a.cmp(NumVal::from_f64(nearest)), Ordering::Equal);
        }
    }

    #[test]
    fn equal_numbers_hash_alike_however_they_are_written() {
        // Hash/eq coherence is the invariant that the variants being disjoint buys us,
        // and the one a wrong `canonicalise` would silently break. Each group below is
        // one number, written several ways — every member must compare `Equal` to every
        // other and hash to the same value.
        let groups: &[&[Lit]] = &[
            &[(false, "0", 0), (false, "000", 3), (true, "0", -3)],
            &[
                (false, "123", 0),
                (false, "1230", -1),
                (false, "123000", -3),
            ],
            &[(false, "1", -1), (false, "10", -2), (false, "100000", -6)],
            // Past `u64`, an exact `f64`.
            &[
                (false, "18446744073709551616", 0),
                (false, "18446744073709551616000", -3),
            ],
            // Arbitrary precision, so the `Big` path must agree with itself.
            &[
                (false, "1234567890123456789012345", -25),
                (false, "12345678901234567890123450", -26),
                (false, "1234567890123456789012345000", -28),
            ],
            &[
                (false, "123456789012345678901234567890123", 0),
                (false, "123456789012345678901234567890123000", -3),
                (false, "12345678901234567890123456789012300", -2),
            ],
        ];
        for group in groups {
            for &a in *group {
                for &b in *group {
                    assert_eq!(
                        cmp_lit(a, b),
                        Ordering::Equal,
                        "{}e{} != {}e{}",
                        a.1,
                        a.2,
                        b.1,
                        b.2
                    );
                    assert_eq!(
                        hash_lit(a),
                        hash_lit(b),
                        "{}e{} and {}e{} are equal but hash differently",
                        a.1,
                        a.2,
                        b.1,
                        b.2
                    );
                }
            }
        }
    }

    #[test]
    fn big_orders_as_a_total_order() {
        // Strictly ascending, and spanning every variant — so each pair also crosses a
        // variant boundary. `cmp` must agree in both directions, which is what
        // `sort_by` relies on: an ordering that is merely "usually right" panics there
        // rather than misbehaving quietly.
        let ascending: &[Lit] = &[
            (true, "1", 0),                                  // -1
            (false, "0", 0),                                 // 0
            (false, "3", -1_000_000),                        // vanishingly small, but positive
            (false, "1", -30),                               // Big (exponent below `Decimal`)
            (false, "1", -1),                                // 0.1 -> Decimal
            (false, "1234567890123456789012345", -25),       // ~0.1234... -> Big
            (false, "5", -1),                                // 0.5 -> Float
            (false, "1", 0),                                 // 1 -> Int
            (false, "18446744073709551616", 0),              // 2^64 -> Float
            (false, "123456789012345678901234567890123", 0), // ~1.2e32 -> Big
            (false, "3", 1_000_000),                         // enormous
        ];
        for (i, &a) in ascending.iter().enumerate() {
            for &b in &ascending[i + 1..] {
                assert_eq!(
                    cmp_lit(a, b),
                    Ordering::Less,
                    "{}e{} should be less than {}e{}",
                    a.1,
                    a.2,
                    b.1,
                    b.2
                );
                assert_eq!(
                    cmp_lit(b, a),
                    Ordering::Greater,
                    "{}e{} vs {}e{} is not antisymmetric",
                    a.1,
                    a.2,
                    b.1,
                    b.2
                );
            }
        }
    }

    #[test]
    fn big_to_f64_lossy_is_correctly_rounded() {
        // The standard library's parser is the oracle *and* the implementation here, so
        // what this really pins down is the exact-digit rendering that feeds it: a
        // mis-rendered mantissa (a dropped internal zero, an unpadded chunk) would show
        // up as a wrong value.
        for &(digits, exp) in &[
            ("1234567890123456789012345", -25),
            ("123456789012345678901234567890123", 0),
            // A magnitude that straddles the 19-digit rendering chunk, with a zero
            // exactly on the boundary.
            ("10000000000000000001", -1),
            ("99999999999999999999999999999999", -16),
            ("1", -30),
            ("7", 40),
        ] {
            let mut storage = None;
            let nv = num_val(&mut storage, (false, digits, exp));
            let oracle: f64 = format!("{}e{}", digits, exp).parse().unwrap();
            assert_eq!(nv.to_f64_lossy(), oracle, "{}e{}", digits, exp);

            let mut storage = None;
            let nv = num_val(&mut storage, (true, digits, exp));
            assert_eq!(nv.to_f64_lossy(), -oracle, "-{}e{}", digits, exp);
        }
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
