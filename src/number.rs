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
        if v.is_finite() {
            Ok(INumber(IValue::new_f64(v)))
        } else {
            Err(())
        }
    }
}

impl TryFrom<f32> for INumber {
    type Error = ();
    fn try_from(v: f32) -> Result<Self, ()> {
        if v.is_finite() {
            Ok(INumber(IValue::new_f64(f64::from(v))))
        } else {
            Err(())
        }
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

/// Whether a valid JSON number is written as a plain integer or with a fraction
/// or exponent (i.e. as a float).
enum NumberShape {
    Integer,
    Float,
}

/// Validates `s` against the JSON number grammar (see <https://www.json.org/>),
/// reporting whether it is integer- or float-shaped, or `None` if it is not a
/// valid JSON number.
///
/// ```text
/// number   = integer fraction exponent
/// integer  = digit | onenine digits | '-' digit | '-' onenine digits
/// fraction = "" | '.' digits
/// exponent = "" | ('E' | 'e') sign digits
/// sign     = "" | '+' | '-'
/// digits   = digit | digit digits
/// ```
fn classify_json_number(s: &str) -> Option<NumberShape> {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;

    // Optional minus sign (a leading '+' is not permitted).
    if i < n && b[i] == b'-' {
        i += 1;
    }

    // Integer part: a single '0', or a '1'..='9' followed by more digits. This
    // rejects leading zeros such as "01".
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(&c) if c.is_ascii_digit() => {
            i += 1;
            while i < n && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        _ => return None,
    }

    let mut is_float = false;

    // Fraction: a '.' must be followed by at least one digit.
    if i < n && b[i] == b'.' {
        is_float = true;
        i += 1;
        if !matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            return None;
        }
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
    }

    // Exponent: 'e'/'E', an optional sign, then at least one digit.
    if i < n && (b[i] == b'e' || b[i] == b'E') {
        is_float = true;
        i += 1;
        if i < n && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        if !matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            return None;
        }
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
    }

    // Reject any trailing characters (and surrounding whitespace).
    if i != n {
        return None;
    }

    Some(if is_float {
        NumberShape::Float
    } else {
        NumberShape::Integer
    })
}

/// Parses a JSON number from its textual form, exactly as the JSON grammar
/// defines it (see <https://www.json.org/>).
///
/// Unlike deserializing through `serde_json` (which rounds to the nearest `f64`),
/// this preserves the *exact* decimal value: a number written with a fraction or
/// exponent is stored as the exact decimal it denotes when that fits inline, so
/// `"0.1"` is the exact `1 * 10^-1` rather than the `f64` approximation — and thus
/// a *different* value from the `f64` `0.1`. A plain in-range integer keeps its
/// integer representation (no decimal point); anything too large or too precise to
/// hold exactly falls back to the nearest `f64`.
///
/// # Examples
///
/// ```
/// # use ijson::INumber;
/// use std::convert::TryFrom;
///
/// let n: INumber = "1.5".parse().unwrap();
/// assert_eq!(n.to_f64(), Some(1.5));
/// assert!(n.has_decimal_point());
///
/// let n: INumber = "-42".parse().unwrap();
/// assert_eq!(n.to_i64(), Some(-42));
/// assert!(!n.has_decimal_point());
///
/// // The exact decimal `0.1` is kept, so it differs from the f64 `0.1`.
/// let d: INumber = "0.1".parse().unwrap();
/// assert_ne!(d, INumber::try_from(0.1_f64).unwrap());
/// assert_eq!(d.to_f64(), None); // not exactly an f64
///
/// // Invalid JSON numbers are rejected (leading zero, bare '+', trailing '.').
/// assert!("01".parse::<INumber>().is_err());
/// assert!("+1".parse::<INumber>().is_err());
/// assert!("1.".parse::<INumber>().is_err());
/// ```
impl FromStr for INumber {
    type Err = ParseNumberError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let float_from = |s: &str| -> Result<IValue, ParseNumberError> {
            // A valid JSON float is always accepted by `f64::from_str`; only its
            // magnitude can be out of range (parsing to an infinity), which we
            // reject so the `INumber` stays finite.
            match s.parse::<f64>() {
                Ok(v) if v.is_finite() => Ok(IValue::new_f64(v)),
                _ => Err(ParseNumberError(())),
            }
        };

        let value = match classify_json_number(s).ok_or(ParseNumberError(()))? {
            NumberShape::Integer => {
                if let Ok(v) = s.parse::<i64>() {
                    IValue::new_i64(v)
                } else if let Ok(v) = s.parse::<u64>() {
                    IValue::new_u64(v)
                } else {
                    // An integer beyond `u64`'s range is stored as a float.
                    float_from(s)?
                }
            }
            // Store the exact decimal inline when it fits; otherwise (too many
            // digits, or the exponent out of the inline range) fall back to f64.
            NumberShape::Float => {
                match parse_decimal(s).and_then(|(m, e)| IValue::new_decimal(m, e)) {
                    Some(v) => v,
                    None => float_from(s)?,
                }
            }
        };
        Ok(INumber(value))
    }
}

/// Extracts the exact decimal value `mantissa * 10^exp` from a validated JSON
/// *float*-shaped string. Returns `None` if the significant digits or exponent
/// overflow (too many to hold exactly), so the caller falls back to `f64`.
fn parse_decimal(s: &str) -> Option<(i128, i32)> {
    let b = s.as_bytes();
    let mut i = 0;
    let neg = b[0] == b'-';
    if neg {
        i += 1;
    }
    let mut mantissa: i128 = 0;
    let mut frac_digits: i32 = 0;
    let mut seen_point = false;
    let mut explicit_exp: i32 = 0;
    while i < b.len() {
        match b[i] {
            c @ b'0'..=b'9' => {
                mantissa = mantissa
                    .checked_mul(10)?
                    .checked_add(i128::from(c - b'0'))?;
                if seen_point {
                    frac_digits += 1;
                }
                i += 1;
            }
            b'.' => {
                seen_point = true;
                i += 1;
            }
            b'e' | b'E' => {
                i += 1;
                let exp_neg = b[i] == b'-';
                if b[i] == b'+' || b[i] == b'-' {
                    i += 1;
                }
                let mut e: i32 = 0;
                while i < b.len() {
                    e = e.checked_mul(10)?.checked_add(i32::from(b[i] - b'0'))?;
                    i += 1;
                }
                explicit_exp = if exp_neg { -e } else { e };
                break;
            }
            // A validated float contains no other bytes.
            _ => return None,
        }
    }
    let exp = explicit_exp.checked_sub(frac_digits)?;
    Some((if neg { -mantissa } else { mantissa }, exp))
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
        // heap (but still round-trips). The same magnitude as an e-notation
        // *float* does factor into a positive inline exponent. The threshold
        // differs by pointer width.
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

            let f = INumber::try_from(v as f64).unwrap();
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

    #[test]
    fn a_plain_integer_beyond_u64_becomes_a_float() {
        // Not exactly representable, so it is stored as a float (like serde_json).
        let n: INumber = "100000000000000000000".parse().unwrap(); // 1e20
        assert_eq!(n.to_f64_lossy(), 1e20);
        assert!(n.has_decimal_point());
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
    fn rejects_out_of_range_magnitude() {
        // A finite JSON float whose magnitude overflows f64 is not representable.
        assert!("1e400".parse::<INumber>().is_err());
        assert!("-1e400".parse::<INumber>().is_err());
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

        // Too many fractional digits to hold exactly -> nearest f64 (matches the
        // f64 constructor).
        let pi: INumber = "3.141592653589793".parse().unwrap();
        assert_eq!(pi, INumber::try_from(std::f64::consts::PI).unwrap());
    }
}
