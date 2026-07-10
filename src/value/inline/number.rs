//! Machinery shared by the two inline number representations.
//!
//! The bit layout and encoding live entirely in each representation
//! ([`super::number_binary`] / [`super::number_decimal`]), which share no encoding
//! code so they can diverge. What they *do* share is this: the [`InlineNumber`]
//! trait they both implement, and the JSON-number grammar validation their
//! `from_str` uses.

use std::convert::TryFrom;

use crate::value::{IValue, NumVal};

/// Whether a valid JSON number is written as a plain integer or with a fraction or
/// exponent (i.e. as a float). Determines the heap fallback when a number does not
/// fit inline.
pub(crate) enum NumberShape {
    Integer,
    Float,
}

/// Why a string could not be stored as an inline number.
pub(crate) enum InlineNumberError {
    /// Not a valid JSON number.
    Invalid,
    /// A valid JSON number whose value does not fit the inline form; the caller
    /// falls back to a heap representation appropriate to its shape.
    Spill(NumberShape),
}

/// The interface both inline number representations implement: encoding a value or
/// a JSON-number string into inline bits, and decoding bits back to a [`NumVal`].
/// The methods are associated functions (no `self`) because they operate on raw
/// values and bits, not on a representation instance.
///
/// This is distinct from [`super::InlineValue`], the dynamically dispatched
/// per-type behaviour [`super::InlineRepr`] delegates to; this is the statically
/// resolved construction/decoding of the active number representation.
pub(crate) trait InlineNumber {
    /// Encodes a plain integer (no decimal point) inline, or `None` if it does not
    /// fit.
    fn encode_int(value: i64) -> Option<usize>;
    /// Encodes a finite `f64` inline, or `None` if it does not fit.
    fn encode_f64(value: f64) -> Option<usize>;
    /// Decodes inline bits to a [`NumVal`] for the shared numeric utilities.
    fn num_val(bits: usize) -> NumVal;
    /// Parses a JSON number string directly into inline bits — this is where each
    /// representation applies its own scheme (an exact decimal for base 10, the
    /// nearest binary float for base 2). Returns [`InlineNumberError::Invalid`] if
    /// `s` is not a JSON number, or [`InlineNumberError::Spill`] if it is a valid
    /// number that does not fit inline, so the caller can store it on the heap.
    fn from_str(s: &str) -> Result<usize, InlineNumberError>;

    /// Constructs an inline `IValue` from an `i64`, or `None` if it does not fit
    /// inline (so the caller falls back to a heap scalar).
    fn from_i64(value: i64) -> Option<IValue> {
        Self::encode_int(value).map(IValue::new_inline_number)
    }
    /// Constructs an inline `IValue` from a `u64`, or `None` if it does not fit —
    /// the inline form only holds the signed range.
    fn from_u64(value: u64) -> Option<IValue> {
        i64::try_from(value).ok().and_then(Self::from_i64)
    }
    /// Constructs an inline `IValue` from a finite `f64`, or `None` if it does not
    /// fit inline.
    fn from_f64(value: f64) -> Option<IValue> {
        Self::encode_f64(value).map(IValue::new_inline_number)
    }
}

/// Validates `s` against the JSON number grammar (see <https://www.json.org/>),
/// reporting whether it is integer- or float-shaped, or `None` if it is not a
/// valid JSON number. Shared by both representations' [`InlineNumber::from_str`].
///
/// ```text
/// number   = integer fraction exponent
/// integer  = digit | onenine digits | '-' digit | '-' onenine digits
/// fraction = "" | '.' digits
/// exponent = "" | ('E' | 'e') sign digits
/// sign     = "" | '+' | '-'
/// digits   = digit | digit digits
/// ```
pub(crate) fn classify_json_number(s: &str) -> Option<NumberShape> {
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

/// The shared body of [`InlineNumber::from_str`]: validate the JSON-number grammar,
/// then encode inline or report a spill. Integers encode identically in both
/// representations (a plain integer token), so only the *float* encoder differs —
/// each representation passes its own (`encode_float` returns the inline bits, or
/// `None` to spill a valid float to the heap).
pub(crate) fn from_str_with(
    s: &str,
    encode_int: impl FnOnce(i64) -> Option<usize>,
    encode_float: impl FnOnce(&str) -> Option<usize>,
) -> Result<usize, InlineNumberError> {
    match classify_json_number(s).ok_or(InlineNumberError::Invalid)? {
        // A validated integer that fits `i64` may still exceed the inline mantissa;
        // one beyond `i64` never fits inline. Either way it spills to the heap.
        NumberShape::Integer => s
            .parse::<i64>()
            .ok()
            .and_then(encode_int)
            .ok_or(InlineNumberError::Spill(NumberShape::Integer)),
        NumberShape::Float => encode_float(s).ok_or(InlineNumberError::Spill(NumberShape::Float)),
    }
}
