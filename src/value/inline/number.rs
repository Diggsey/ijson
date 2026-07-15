//! Machinery shared by the two inline number representations.
//!
//! The bit layout and encoding live entirely in each representation
//! ([`super::number_binary`] / [`super::number_decimal`]), which share no encoding
//! code so they can diverge. What they *do* share is this: the [`InlineNumber`]
//! trait they both implement, and the JSON-number grammar validation their
//! `from_str` uses.

use std::convert::TryFrom;

use crate::value::IValue;

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
/// a JSON-number string into inline bits. The methods are associated functions (no
/// `self`) because they operate on raw values and bits, not on a representation
/// instance. (Decoding bits back to a [`NumVal`] is the job of each representation's
/// `InlineValue`/`ValueRepr` impl, not this trait.)
///
/// This is distinct from [`super::InlineValue`], the dynamically dispatched
/// per-type behaviour [`super::InlineRepr`] delegates to; this is the statically
/// resolved construction of the active number representation.
pub(crate) trait InlineNumber {
    /// Encodes a plain integer (no decimal point) inline, or `None` if it does not
    /// fit.
    fn encode_int(value: i64) -> Option<usize>;
    /// Encodes a finite `f64` inline, or `None` if it does not fit.
    fn encode_f64(value: f64) -> Option<usize>;
    /// Parses a JSON number string directly into inline bits — this is where each
    /// representation applies its own scheme (an exact decimal for base 10, the
    /// nearest binary float for base 2). Returns [`InlineNumberError::Invalid`] if
    /// `s` is not a JSON number, or [`InlineNumberError::Spill`] if it is a valid
    /// number that does not fit inline, so the caller can store it on the heap.
    fn from_str(s: &str) -> Result<usize, InlineNumberError>;

    /// Constructs an inline `IValue` from an `i64`, or `None` if it does not fit
    /// inline (so the caller falls back to a heap scalar).
    fn from_i64(value: i64) -> Option<IValue> {
        // Safety: `encode_int` returns valid inline-number bits.
        Self::encode_int(value).map(|bits| unsafe { IValue::new_inline_number(bits) })
    }
    /// Constructs an inline `IValue` from a `u64`, or `None` if it does not fit —
    /// the inline form only holds the signed range.
    fn from_u64(value: u64) -> Option<IValue> {
        i64::try_from(value).ok().and_then(Self::from_i64)
    }
    /// Constructs an inline `IValue` from a finite `f64`, or `None` if it does not
    /// fit inline.
    fn from_f64(value: f64) -> Option<IValue> {
        // Safety: `encode_f64` returns valid inline-number bits.
        Self::encode_f64(value).map(|bits| unsafe { IValue::new_inline_number(bits) })
    }
}

/// A validated JSON number literal, taken apart.
///
/// The digits of the integer and fraction parts, read as one run, are the significand;
/// the value is `(-1)^negative * <int_digits><frac_digits> * 10^(written_exp -
/// frac_digits.len())`. Both slices borrow the literal, so validating costs no
/// allocation — only the arbitrary-precision path, which actually needs the digits,
/// pays for them.
///
/// Only the arbitrary-precision path reads the parts; without it, `from_str_with` uses
/// nothing but the shape.
#[cfg_attr(not(feature = "arbitrary_precision"), allow(dead_code))]
pub(crate) struct JsonNumber<'a> {
    pub(crate) negative: bool,
    pub(crate) int_digits: &'a [u8],
    pub(crate) frac_digits: &'a [u8],
    /// The exponent as written (zero if the literal had none), saturated — a literal
    /// long enough to overflow this would have to be gigabytes of exponent digits.
    pub(crate) written_exp: i64,
    pub(crate) shape: NumberShape,
}

#[cfg_attr(not(feature = "arbitrary_precision"), allow(dead_code))]
impl JsonNumber<'_> {
    /// The significand's digits and the exponent of the value they scale: the exact
    /// value is `(-1)^negative * digits * 10^exp`. Allocates, so only the
    /// arbitrary-precision path calls it.
    ///
    /// `None` if the exponent is too extreme to represent — see [`MAX_EXP`].
    pub(crate) fn significand(&self) -> Option<(Vec<u8>, i32)> {
        // Moving the fraction's digits into the significand shifts the exponent down by
        // one per digit, which is the only place the written exponent can grow in
        // magnitude; `canonicalise` may then raise it again by at most the digit count,
        // so bounding the result at `MAX_EXP` leaves it room to do so within an `i32`.
        //
        // Saturating: `written_exp` is itself already saturated (an absurd literal like
        // `1e-99999999999999999999` reaches `i64::MIN`), and subtracting the fraction
        // length from that would overflow. The result is rejected by the `MAX_EXP` bound
        // regardless — saturation only keeps it a well-defined, still-out-of-range number.
        let exp = self
            .written_exp
            .saturating_sub(self.frac_digits.len() as i64);
        if exp.unsigned_abs() > MAX_EXP {
            return None;
        }
        let mut digits = Vec::with_capacity(self.int_digits.len() + self.frac_digits.len());
        digits.extend_from_slice(self.int_digits);
        digits.extend_from_slice(self.frac_digits);
        Some((digits, exp as i32))
    }
}

/// The largest exponent magnitude an arbitrary-precision decimal may have, leaving
/// `canonicalise` room to raise it by the number of digits without overflowing an
/// `i32`. Numbers beyond it are stored as an `f64` (an infinity or a zero, as they
/// would be anyway), exactly as they are without `arbitrary_precision`.
#[cfg_attr(not(feature = "arbitrary_precision"), allow(dead_code))]
pub(crate) const MAX_EXP: u64 = 1 << 30;

/// Validates `s` against the JSON number grammar (see <https://www.json.org/>) and
/// takes it apart, or `None` if it is not a valid JSON number. Shared by both inline
/// representations' [`InlineNumber::from_str`] (which need only the shape) and, with
/// `arbitrary_precision`, by the heap decimal representation (which needs the digits).
/// One parser, so the grammar has one home.
///
/// ```text
/// number   = integer fraction exponent
/// integer  = digit | onenine digits | '-' digit | '-' onenine digits
/// fraction = "" | '.' digits
/// exponent = "" | ('E' | 'e') sign digits
/// sign     = "" | '+' | '-'
/// digits   = digit | digit digits
/// ```
pub(crate) fn parse_json_number(s: &str) -> Option<JsonNumber<'_>> {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;

    // Optional minus sign (a leading '+' is not permitted).
    let negative = i < n && b[i] == b'-';
    if negative {
        i += 1;
    }

    // Integer part: a single '0', or a '1'..='9' followed by more digits. This
    // rejects leading zeros such as "01".
    let int_start = i;
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
    let int_digits = &b[int_start..i];

    let mut is_float = false;
    let mut frac_digits: &[u8] = &[];

    // Fraction: a '.' must be followed by at least one digit.
    if i < n && b[i] == b'.' {
        is_float = true;
        i += 1;
        if !matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            return None;
        }
        let frac_start = i;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        frac_digits = &b[frac_start..i];
    }

    // Exponent: 'e'/'E', an optional sign, then at least one digit.
    let mut written_exp: i64 = 0;
    if i < n && (b[i] == b'e' || b[i] == b'E') {
        is_float = true;
        i += 1;
        let exp_negative = i < n && b[i] == b'-';
        if i < n && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        if !matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            return None;
        }
        while i < n && b[i].is_ascii_digit() {
            // Saturating, so an absurd exponent is a huge magnitude rather than a wrap
            // to a small one. `significand` rejects anything past `MAX_EXP` anyway.
            written_exp = written_exp
                .saturating_mul(10)
                .saturating_add(i64::from(b[i] - b'0'));
            i += 1;
        }
        if exp_negative {
            written_exp = -written_exp;
        }
    }

    // Reject any trailing characters (and surrounding whitespace).
    if i != n {
        return None;
    }

    Some(JsonNumber {
        negative,
        int_digits,
        frac_digits,
        written_exp,
        shape: if is_float {
            NumberShape::Float
        } else {
            NumberShape::Integer
        },
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
    match parse_json_number(s)
        .ok_or(InlineNumberError::Invalid)?
        .shape
    {
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
