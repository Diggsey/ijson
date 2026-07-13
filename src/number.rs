//! Functionality relating to the JSON number type.
//!
//! [`INumber`] is the public *type* for JSON numbers. It is a thin, transparent
//! wrapper around an [`IValue`] that is known to be a number; all of the actual
//! logic (construction, conversions, comparison, hashing) lives on `IValue` as
//! its `new_*`/`number_*` methods and is shared with `IValue` itself. The number
//! can be stored either inline or as a heap scalar, but that choice is entirely
//! hidden behind this type.
#![allow(clippy::float_cmp)]

use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::str::FromStr;

use crate::value::IValue;

/// The `INumber` type represents a JSON number. It is decoupled from any specific
/// representation, and internally uses several. There is no way to determine the
/// internal representation: instead the caller is expected to convert the number
/// using one of the fallible `to_xxx` functions and handle the cases where the
/// number does not convert to the desired type.
///
/// Special floating point values (eg. NaN, Infinity, etc.) cannot be stored within
/// an `INumber`.
///
/// Whilst `INumber` does not consider `2.0` and `2` to be different numbers (ie.
/// they will compare equal) it does allow you to distinguish them using the
/// method `INumber::has_decimal_point()`. That said, calling `to_i32` on
/// `2.0` will succeed with the value `2`.
///
/// Small numbers — integers and short decimals alike — are stored inline without
/// a heap allocation. Larger integers (`i64`/`u64`) and floating point values
/// that are not short decimals are stored behind a pointer.
#[repr(transparent)]
#[derive(Clone)]
pub struct INumber(pub(crate) IValue);

value_subtype_impls!(INumber, into_number, as_number, as_number_mut);

impl INumber {
    /// Returns the number zero (without a decimal point). Does not allocate.
    #[must_use]
    pub fn zero() -> Self {
        INumber(IValue::new_i64(0))
    }
    /// Returns the number one (without a decimal point). Does not allocate.
    #[must_use]
    pub fn one() -> Self {
        INumber(IValue::new_i64(1))
    }

    /// Converts this number to an i64 if it can be represented exactly.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }
    /// Converts this number to a u64 if it can be represented exactly.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
    /// Converts this number to an f64 if it can be represented exactly.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
    /// Converts this number to an f32 if it can be represented exactly.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }
    /// Converts this number to an i32 if it can be represented exactly.
    #[must_use]
    pub fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }
    /// Converts this number to a u32 if it can be represented exactly.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }
    /// Converts this number to an isize if it can be represented exactly.
    #[must_use]
    pub fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }
    /// Converts this number to a usize if it can be represented exactly.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }
    /// Converts this number to an f64, potentially losing precision in the process.
    #[must_use]
    pub fn to_f64_lossy(&self) -> f64 {
        // Always a number, so the representation always yields a value.
        self.0.to_f64_lossy().unwrap()
    }
    /// Converts this number to an f32, potentially losing precision in the process.
    #[must_use]
    pub fn to_f32_lossy(&self) -> f32 {
        // Always a number, so the representation always yields a value.
        self.0.to_f32_lossy().unwrap()
    }

    /// This allows distinguishing between `1.0` and `1` in the original JSON.
    /// Numeric operations will otherwise treat these two values as equivalent.
    #[must_use]
    pub fn has_decimal_point(&self) -> bool {
        self.0.has_decimal_point()
    }
}

impl Hash for INumber {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl From<u64> for INumber {
    fn from(v: u64) -> Self {
        INumber(IValue::new_u64(v))
    }
}
impl From<u32> for INumber {
    fn from(v: u32) -> Self {
        INumber(IValue::new_u64(u64::from(v)))
    }
}
impl From<u16> for INumber {
    fn from(v: u16) -> Self {
        INumber(IValue::new_u64(u64::from(v)))
    }
}
impl From<u8> for INumber {
    fn from(v: u8) -> Self {
        INumber(IValue::new_u64(u64::from(v)))
    }
}
impl From<usize> for INumber {
    fn from(v: usize) -> Self {
        INumber(IValue::new_u64(v as u64))
    }
}

impl From<i64> for INumber {
    fn from(v: i64) -> Self {
        INumber(IValue::new_i64(v))
    }
}
impl From<i32> for INumber {
    fn from(v: i32) -> Self {
        INumber(IValue::new_i64(i64::from(v)))
    }
}
impl From<i16> for INumber {
    fn from(v: i16) -> Self {
        INumber(IValue::new_i64(i64::from(v)))
    }
}
impl From<i8> for INumber {
    fn from(v: i8) -> Self {
        INumber(IValue::new_i64(i64::from(v)))
    }
}
impl From<isize> for INumber {
    fn from(v: isize) -> Self {
        INumber(IValue::new_i64(v as i64))
    }
}

impl TryFrom<f64> for INumber {
    type Error = ();
    fn try_from(v: f64) -> Result<Self, ()> {
        // `new_f64` rejects non-finite input, so finiteness is enforced there.
        IValue::new_f64(v).map(INumber).ok_or(())
    }
}

impl TryFrom<f32> for INumber {
    type Error = ();
    fn try_from(v: f32) -> Result<Self, ()> {
        IValue::new_f64(f64::from(v)).map(INumber).ok_or(())
    }
}

/// The error returned when a string cannot be parsed as an [`INumber`]: either it
/// is not a valid JSON number, or its magnitude is beyond the representable range
/// (a float that would be infinite).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseNumberError(());

impl fmt::Display for ParseNumberError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("invalid JSON number")
    }
}

impl std::error::Error for ParseNumberError {}

/// Parses a JSON number from its textual form, exactly as the JSON grammar
/// defines it (see <https://www.json.org/>).
///
/// A float-shaped number is parsed to the nearest `f64` (as deserializing through
/// `serde_json` would), while a plain in-range integer keeps its exact integer
/// representation (no decimal point).
///
/// With the `arbitrary_precision` feature this instead preserves the *exact*
/// decimal value where it fits: `"0.1"` becomes the exact `1 * 10^-1` rather than
/// the `f64` approximation — and thus a *different* value from the `f64` `0.1` —
/// falling back to the nearest `f64` only when the value is too large or too
/// precise to hold exactly.
///
/// # Examples
///
/// ```
/// # use ijson::INumber;
/// let n: INumber = "1.5".parse().unwrap();
/// assert_eq!(n.to_f64(), Some(1.5));
/// assert!(n.has_decimal_point());
///
/// let n: INumber = "-42".parse().unwrap();
/// assert_eq!(n.to_i64(), Some(-42));
/// assert!(!n.has_decimal_point());
///
/// // Invalid JSON numbers are rejected (leading zero, bare '+', trailing '.').
/// assert!("01".parse::<INumber>().is_err());
/// assert!("+1".parse::<INumber>().is_err());
/// assert!("1.".parse::<INumber>().is_err());
/// ```
impl FromStr for INumber {
    type Err = ParseNumberError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use crate::value::inline::{
            InlineNumber, InlineNumberError, InlineNumberRepr, NumberShape,
        };

        // A valid JSON float is always accepted by `f64::from_str`; only its
        // magnitude can be out of range (parsing to an infinity). `new_f64` rejects
        // the infinity, so the `INumber` stays finite.
        let float_from = |s: &str| {
            s.parse::<f64>()
                .ok()
                .and_then(IValue::new_f64)
                .ok_or(ParseNumberError(()))
        };

        // What a number that does not fit inline falls back to, once the shape-specific
        // fast paths below have declined it.
        //
        // With `arbitrary_precision` it is stored exactly, in whichever representation
        // can hold it (see `IValue::new_decimal`) — that is the whole point of the
        // feature, and the reason the exponent is bounded rather than the value.
        // Without it, there is nowhere exact to put such a number, so it rounds to the
        // nearest `f64` as it always has.
        #[cfg(feature = "arbitrary_precision")]
        let spill = |s: &str, shape: &NumberShape| -> Result<IValue, ParseNumberError> {
            use crate::value::inline::parse_json_number;

            // `from_str` already validated the grammar, so this re-parse cannot fail —
            // it only takes the literal apart. An exponent too extreme for the decimal
            // representation falls back to an `f64`, which is the infinity or zero it
            // would round to in any case.
            let parsed = parse_json_number(s).expect("`from_str` already validated the grammar");
            match parsed.significand() {
                Some((digits, exp)) => Ok(IValue::new_decimal(
                    parsed.negative,
                    &digits,
                    exp,
                    matches!(shape, NumberShape::Float),
                )),
                None => float_from(s),
            }
        };
        #[cfg(not(feature = "arbitrary_precision"))]
        let spill = |s: &str, _shape: &NumberShape| float_from(s);

        // Ask the active inline representation to store it directly; only if that
        // spills (a valid number too large for the inline form) do we fall back to
        // a heap representation appropriate to its shape.
        let value = match InlineNumberRepr::from_str(s) {
            // Safety: `from_str` returns valid inline-number bits.
            Ok(bits) => unsafe { IValue::new_inline_number(bits) },
            Err(InlineNumberError::Invalid) => return Err(ParseNumberError(())),
            Err(InlineNumberError::Spill(shape @ NumberShape::Integer)) => {
                // A plain integer in range needs nothing cleverer than a heap scalar.
                if let Ok(v) = s.parse::<i64>() {
                    IValue::new_i64(v)
                } else if let Ok(v) = s.parse::<u64>() {
                    IValue::new_u64(v)
                } else {
                    spill(s, &shape)?
                }
            }
            Err(InlineNumberError::Spill(shape @ NumberShape::Float)) => spill(s, &shape)?,
        };
        Ok(INumber(value))
    }
}

/// Converts a [`serde_json::Number`] into an [`INumber`].
///
/// Conversion may be lossy if the number is not exactly representable as an
/// `INumber`. The exact behaviour in that case (e.g. clamping an out-of-range
/// magnitude) is not guaranteed to be stable across versions.
impl From<serde_json::Number> for INumber {
    fn from(n: serde_json::Number) -> Self {
        if let Some(v) = n.as_u64() {
            INumber::from(v)
        } else if let Some(v) = n.as_i64() {
            INumber::from(v)
        } else {
            // A serde_json number is always representable as an f64, so this
            // cannot return `None`; if it does, an invariant broke.
            let v = n
                .as_f64()
                .expect("a serde_json number is always an integer or float");
            // Standard JSON numbers are finite. Only the `arbitrary_precision`
            // feature can parse a magnitude beyond f64's range (an infinity);
            // clamp it so the result stays a finite, representable number and
            // `try_from` cannot fail.
            INumber::try_from(v.clamp(f64::MIN, f64::MAX)).expect("a clamped f64 is always finite")
        }
    }
}

/// Converts an [`INumber`] into a [`serde_json::Number`].
///
/// Conversion may be lossy if the number is not exactly representable as a
/// `serde_json::Number`. The exact behaviour in that case (e.g. rounding) is
/// not guaranteed to be stable across versions.
impl From<INumber> for serde_json::Number {
    fn from(n: INumber) -> Self {
        if let Some(v) = n.to_u64() {
            serde_json::Number::from(v)
        } else if let Some(v) = n.to_i64() {
            serde_json::Number::from(v)
        } else {
            // Not an integer, so it is stored as an f64. An `INumber` is always
            // finite, so `from_f64` cannot fail; a failure here would mean the
            // `INumber` invariant was violated.
            serde_json::Number::from_f64(n.to_f64_lossy()).expect("an INumber is always finite")
        }
    }
}

impl PartialEq for INumber {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for INumber {}
impl Ord for INumber {
    fn cmp(&self, other: &Self) -> Ordering {
        // Two numbers always compare (via the representation's `partial_cmp`).
        self.0.partial_cmp(&other.0).unwrap()
    }
}
impl PartialOrd for INumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Debug for INumber {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Default for INumber {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn can_create() {
        let x = INumber::zero();
        let y: INumber = (0.0).try_into().unwrap();

        assert_eq!(x, y);
        assert!(!x.has_decimal_point());
        assert!(y.has_decimal_point());
        assert_eq!(x.to_i32(), Some(0));
        assert_eq!(y.to_i32(), Some(0));
    }

    #[test]
    fn stores_small_integers_inline() {
        for v in [0i64, 1, -1, 42, -42, 1000, -1000, 1_000_000] {
            let n = INumber::from(v);
            assert!(n.0.is_inline(), "{} should be inline", v);
            assert_eq!(n.to_i64(), Some(v));
            assert!(!n.has_decimal_point());
        }
    }

    #[test]
    fn stores_short_decimals_inline() {
        for (v, s) in [
            (0.5f64, "0.5"),
            (0.25, "0.25"),
            (63.5, "63.5"),
            (2.0, "2.0"),
        ] {
            let n = INumber::try_from(v).unwrap();
            assert!(n.0.is_inline(), "{} should be inline", s);
            assert_eq!(n.to_f64(), Some(v), "{}", s);
            assert!(n.has_decimal_point(), "{}", s);
        }
    }

    #[test]
    fn integer_and_float_compare_equal() {
        let i = INumber::from(2);
        let f = INumber::try_from(2.0).unwrap();
        assert_eq!(i, f);
        assert!(!i.has_decimal_point());
        assert!(f.has_decimal_point());
    }

    #[mockalloc::test]
    fn integer_boundaries_roundtrip() {
        for v in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX] {
            assert_eq!(INumber::from(v).to_i64(), Some(v), "{}", v);
        }
        for v in [0u64, 1, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX] {
            assert_eq!(INumber::from(v).to_u64(), Some(v), "{}", v);
        }
        // A large round *integer* that exceeds the mantissa does not fit inline:
        // positive inline exponents are reserved for floats, so it spills to the
        // heap (but still round-trips). The threshold differs by pointer width.
        let big_round = if usize::BITS == 64 {
            10i64.pow(18)
        } else {
            10i64.pow(8)
        };
        for v in [big_round, -big_round] {
            let n = INumber::from(v);
            assert!(!n.0.is_inline(), "{} (integer) should be on the heap", v);
            assert_eq!(n.to_i64(), Some(v));
            assert!(!n.has_decimal_point());

            // The same magnitude as an e-notation *float* factors into a positive
            // inline exponent with the base-10 encoding; the base-2 encoding has
            // no such small exponent, so it spills to the heap. Either way it
            // round-trips and keeps its decimal point.
            let f = INumber::try_from(v as f64).unwrap();
            #[cfg(feature = "arbitrary_precision")]
            assert!(f.0.is_inline(), "{} (float) should factor inline", v);
            assert_eq!(f.to_i64(), Some(v));
            assert!(f.has_decimal_point());
        }
        // Assorted large integers round-trip regardless of representation.
        for v in [
            10i64.pow(15),
            10i64.pow(18),
            i64::MAX,
            9_999_999_999_999_937,
        ] {
            assert_eq!(INumber::from(v).to_i64(), Some(v), "{}", v);
        }
    }

    #[test]
    fn negative_short_decimals() {
        for v in [-0.5f64, -2.5, -63.5, -0.125] {
            let n = INumber::try_from(v).unwrap();
            assert_eq!(n.to_f64(), Some(v));
            assert!(n.has_decimal_point());
            assert_eq!(-n.to_f64_lossy(), -v);
        }
    }

    #[mockalloc::test]
    fn large_values_use_heap() {
        let big = INumber::from(u64::MAX);
        assert!(!big.0.is_inline());
        assert_eq!(big.to_u64(), Some(u64::MAX));

        let pi = INumber::try_from(std::f64::consts::PI).unwrap();
        assert!(!pi.0.is_inline());
        assert_eq!(pi.to_f64(), Some(std::f64::consts::PI));
        assert!(pi.has_decimal_point());
    }

    #[test]
    fn ordering() {
        let mut v = [
            INumber::from(-5),
            INumber::try_from(2.5).unwrap(),
            INumber::from(2),
            INumber::from(u64::MAX),
            INumber::try_from(-0.5).unwrap(),
            INumber::from(0),
        ];
        v.sort();
        let f: Vec<f64> = v.iter().map(INumber::to_f64_lossy).collect();
        assert_eq!(f, [-5.0, -0.5, 0.0, 2.0, 2.5, u64::MAX as f64]);
    }

    #[mockalloc::test]
    fn ordering_across_representations() {
        // Spans inline decimals, inline integers, and heap i64/u64/f64.
        let mut v = [
            INumber::try_from(std::f64::consts::PI).unwrap(),
            INumber::from(3),
            INumber::from(u64::MAX),
            INumber::from(i64::MIN),
            INumber::try_from(2.999_999_999).unwrap(),
            INumber::from(0),
        ];
        v.sort();
        let got: Vec<f64> = v.iter().map(INumber::to_f64_lossy).collect();
        assert!(got.windows(2).all(|w| w[0] <= w[1]), "{:?}", got);
        assert_eq!(got[0], i64::MIN as f64);
        assert_eq!(*got.last().unwrap(), u64::MAX as f64);
    }

    #[test]
    fn large_integer_and_enotation_float_serialize_distinctly() {
        // A plain large integer carries no decimal point and serializes back as
        // an integer, even though it now lives on the heap.
        let int: IValue = serde_json::from_str("1000000000000000000").unwrap();
        assert!(!int.as_number().unwrap().has_decimal_point());
        assert_eq!(serde_json::to_string(&int).unwrap(), "1000000000000000000");

        // The same magnitude written in e-notation is a float: it factors into a
        // positive inline exponent, still reports a decimal point, and serializes
        // back as a float.
        let float: IValue = serde_json::from_str("1e18").unwrap();
        assert!(float.as_number().unwrap().has_decimal_point());
        let s = serde_json::to_string(&float).unwrap();
        assert!(
            s.contains('.') || s.contains('e') || s.contains('E'),
            "expected a float rendering, got {}",
            s
        );

        // Regardless of representation they are numerically equal.
        assert_eq!(int, float);
    }

    #[test]
    fn parses_valid_integers() {
        for (s, v) in [
            ("0", 0i64),
            ("-0", 0),
            ("42", 42),
            ("-42", -42),
            ("9223372036854775807", i64::MAX),
            ("-9223372036854775808", i64::MIN),
        ] {
            let n: INumber = s.parse().unwrap();
            assert_eq!(n.to_i64(), Some(v), "{}", s);
            assert!(!n.has_decimal_point(), "{} should have no decimal point", s);
        }

        // Above `i64::MAX` but within `u64`.
        let n: INumber = "18446744073709551615".parse().unwrap();
        assert_eq!(n.to_u64(), Some(u64::MAX));
        assert!(!n.has_decimal_point());
    }

    #[test]
    fn parses_valid_floats() {
        for (s, v) in [
            ("0.0", 0.0f64),
            ("-0.0", -0.0),
            ("1.5", 1.5),
            ("0.5", 0.5),
            ("-3.25", -3.25),
            ("1e3", 1000.0),
            ("1E3", 1000.0),
            ("1e+3", 1000.0),
            ("1e-3", 0.001),
            ("1.5e2", 150.0),
        ] {
            let n: INumber = s.parse().unwrap();
            assert_eq!(n.to_f64_lossy(), v, "{}", s);
            assert!(n.has_decimal_point(), "{} should have a decimal point", s);
        }
    }

    /// A number is one number, whichever representation happens to hold it.
    ///
    /// An integer past `u64` can only be stored as a decimal — the sole integer-capable
    /// representation with room for one — even when its value is exactly an `f64`. The
    /// *same* value arriving as a Rust `f64` (`From<f64>`, or serde's `visit_f64`) is
    /// stored as an `f64`. Nothing about the number differs, so decoding them must not:
    /// they have to compare equal and hash alike, or a `HashMap` keyed on them would
    /// hold the same number twice.
    ///
    /// This is why reducing to a `NumVal` has to be *complete*, exact-`f64` case
    /// included, rather than stopping at "the decimal representation holds it, so it
    /// must need arbitrary precision".
    #[test]
    #[cfg(feature = "arbitrary_precision")]
    fn a_value_is_the_same_number_in_either_representation() {
        fn hash_of(n: &INumber) -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            n.hash(&mut h);
            h.finish()
        }

        for (written, exactly_an_f64) in [
            ("18446744073709551616", 18_446_744_073_709_551_616.0_f64), // 2^64
            ("100000000000000000000", 1e20),
            ("10000000000000000000000", 1e22),
            // 2^100: a single significant bit, so exactly an `f64` despite its size.
            (
                "1267650600228229401496703205376",
                1_267_650_600_228_229_401_496_703_205_376.0_f64,
            ),
        ] {
            // Stored as a decimal (an integer literal beyond `u64`)...
            let as_integer: INumber = written.parse().unwrap();
            // ...and as an `f64` (the same value, handed straight in as one).
            let as_float = INumber::try_from(exactly_an_f64).unwrap();

            assert_eq!(as_integer, as_float, "{} != its own f64", written);
            assert_eq!(
                hash_of(&as_integer),
                hash_of(&as_float),
                "{} and its f64 are equal but hash differently",
                written
            );
            assert_eq!(as_integer.to_f64(), Some(exactly_an_f64), "{}", written);

            // ...and each still remembers how it was written.
            assert!(!as_integer.has_decimal_point(), "{}", written);
            assert!(as_float.has_decimal_point(), "{}", written);
            assert_eq!(serde_json::to_string(&as_integer).unwrap(), written);
        }

        // An integer past `u64` that is *not* an exact `f64` stays arbitrary-precision,
        // and still equals the same value written as a float.
        let big: INumber = "123456789012345678901234567890".parse().unwrap();
        let same: INumber = "1.2345678901234567890123456789e29".parse().unwrap();
        assert_eq!(big, same);
        assert_eq!(hash_of(&big), hash_of(&same));
        assert_eq!(big.to_f64(), None);
        assert!(!big.has_decimal_point() && same.has_decimal_point());
    }

    #[test]
    fn a_plain_integer_beyond_u64_stays_an_integer() {
        // 1e20 is exactly an `f64`, so it is *stored* as one either way. What differs is
        // whether that is allowed to change what it is.
        let n: INumber = "100000000000000000000".parse().unwrap(); // 1e20
        assert_eq!(n.to_f64_lossy(), 1e20);

        // Without exact decimals, an integer past `u64` has nowhere to go but a float,
        // and is reported as one (as serde_json does).
        #[cfg(not(feature = "arbitrary_precision"))]
        {
            assert!(n.has_decimal_point());
        }

        // With them, being held as an `f64` is an implementation detail: it is still an
        // integer literal, so it has no decimal point and serializes back as written.
        // (An integer that is *not* exactly an `f64` is held exactly, and likewise.)
        #[cfg(feature = "arbitrary_precision")]
        {
            assert!(!n.has_decimal_point());
            assert_eq!(serde_json::to_string(&n).unwrap(), "100000000000000000000");

            let odd: INumber = "123456789012345678901234567890".parse().unwrap();
            assert!(!odd.has_decimal_point());
            assert_eq!(odd.to_i64(), None);
            assert_eq!(
                serde_json::to_string(&odd).unwrap(),
                "123456789012345678901234567890"
            );

            // The same number written as a float is the same *value*, but keeps its
            // float shape — the two must compare and hash alike all the same.
            let as_float: INumber = "1.2345678901234567890123456789e29".parse().unwrap();
            assert_eq!(odd, as_float);
            assert!(as_float.has_decimal_point());
        }
    }

    #[test]
    fn rejects_invalid_json_numbers() {
        for s in [
            "",
            " ",
            "1 ",
            " 1",
            "+1",
            "01",
            "-01",
            "00",
            "1.",
            ".5",
            "1.e2",
            "1e",
            "1e+",
            "1e-",
            "1..2",
            "1.2.3",
            "abc",
            "NaN",
            "Infinity",
            "-Infinity",
            "0x1f",
            "1_000",
            "1,000",
            "--1",
            "1-",
            "e5",
            ".",
            "-",
            "+",
            "0.",
            "1.0.",
            "0xa",
        ] {
            assert!(s.parse::<INumber>().is_err(), "{:?} should be rejected", s);
        }
    }

    #[test]
    #[cfg(not(feature = "arbitrary_precision"))]
    fn rejects_out_of_range_magnitude() {
        // Without exact decimals there is nowhere to put a JSON float whose magnitude
        // overflows `f64`, so it is not representable.
        assert!("1e400".parse::<INumber>().is_err());
        assert!("-1e400".parse::<INumber>().is_err());
    }

    #[test]
    #[cfg(feature = "arbitrary_precision")]
    fn holds_magnitudes_beyond_f64() {
        // ...but an exact decimal has room for them, so they are ordinary numbers: the
        // exponent, not the `f64` range, is the limit. They are still *finite* — only
        // their `f64` projection overflows — so ordering and equality stay exact.
        let big: INumber = "1e400".parse().unwrap();
        assert!(big.has_decimal_point());
        assert_eq!(big.to_f64(), None);
        assert_eq!(big, "10e399".parse::<INumber>().unwrap());
        assert!(big > "9.9e399".parse::<INumber>().unwrap());
        assert!(big > INumber::try_from(f64::MAX).unwrap());
        assert!("-1e400".parse::<INumber>().unwrap() < big);

        // And they round-trip, which serializing through an `f64` (an infinity, written
        // as `null`) could not do.
        assert_eq!(serde_json::to_string(&big).unwrap(), "1e400");
        assert_eq!(
            serde_json::to_string(&"1e-400".parse::<INumber>().unwrap()).unwrap(),
            "1e-400"
        );
    }

    #[test]
    fn try_from_rejects_non_finite() {
        // Finiteness is enforced at the single `new_f64` boundary, so every
        // construction path that reaches it rejects NaN/Infinity rather than storing
        // a number that would break `INumber`'s finite-only invariant.
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(INumber::try_from(v).is_err(), "{} should be rejected", v);
        }
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(INumber::try_from(v).is_err(), "{} should be rejected", v);
        }
        // A finite value still constructs.
        assert!(INumber::try_from(1.5_f64).is_ok());
    }

    #[test]
    fn round_trips_through_serialization() {
        // For integers and exact-f64 floats, from_str and serde_json agree.
        for s in ["0", "-42", "1.5", "1e3", "18446744073709551615"] {
            let n: INumber = s.parse().unwrap();
            let via_serde: INumber = serde_json::from_str(s).unwrap();
            assert_eq!(n, via_serde, "{}", s);
            assert_eq!(
                n.has_decimal_point(),
                via_serde.has_decimal_point(),
                "{}",
                s
            );
        }
        // Every parse survives its own serialize+reparse, including exact decimals
        // (like 0.001) that serde_json would instead round to an f64.
        for s in ["0", "-42", "1.5", "1e3", "0.001", "0.1", "3.5e-4"] {
            let n: INumber = s.parse().unwrap();
            let out = serde_json::to_string(&n).unwrap();
            let back: INumber = out.parse().unwrap();
            assert_eq!(n, back, "{} -> {}", s, out);
        }
    }

    #[cfg(feature = "arbitrary_precision")]
    #[test]
    fn from_str_preserves_exact_decimals() {
        // "0.1" is stored as the exact decimal 1 * 10^-1, a *different* value from
        // the f64 0.1 (which is 0.1000000000000000055...).
        let d: INumber = "0.1".parse().unwrap();
        let f = INumber::try_from(0.1_f64).unwrap();
        assert!(d.has_decimal_point());
        assert_ne!(d, f, "exact 0.1 must differ from the f64 0.1");
        assert!(d < f, "0.1 (exact) < 0.1_f64");
        assert_eq!(d.to_f64(), None, "0.1 is not exactly an f64");
        assert_eq!(d.to_f64_lossy(), 0.1_f64, "nearest f64");
        assert_eq!(serde_json::to_string(&d).unwrap(), "0.1");

        // "0.10", "1e-1" etc. are the same value and compare equal to "0.1".
        for s in ["0.10", "1e-1", "0.100000"] {
            assert_eq!(s.parse::<INumber>().unwrap(), d, "{}", s);
        }

        // An exact-f64 decimal, by contrast, equals its f64.
        let half: INumber = "0.5".parse().unwrap();
        assert_eq!(half, INumber::try_from(0.5_f64).unwrap());
        assert_eq!(half.to_f64(), Some(0.5));

        // Too many fractional digits for the *inline* decimal, but the heap decimal
        // holds it exactly — so, like `0.1`, it is a different number from the `f64`
        // nearest it, and says so rather than quietly rounding.
        let pi: INumber = "3.141592653589793".parse().unwrap();
        assert_ne!(pi, INumber::try_from(std::f64::consts::PI).unwrap());
        assert_eq!(pi.to_f64(), None);
        assert_eq!(pi.to_f64_lossy(), std::f64::consts::PI);
        assert_eq!(serde_json::to_string(&pi).unwrap(), "3.141592653589793");
    }
}
