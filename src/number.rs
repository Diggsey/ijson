//! Functionality relating to the JSON number type
#![allow(clippy::float_cmp)]

use std::alloc::Layout;
use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;

use crate::alloc::{alloc_infallible, dealloc_infallible};

use super::value::{IValue, TypeTag};

// ===========================================================================
// Inline decimal layout (tag `Inline`, number sub-family, i.e. bit 3 clear)
//
//   bits 0-2 : tag (Inline == 0)
//   bit 3    : 0 (number sub-family)
//   bits 4-7 : exponent code (see below)
//   bits 8.. : signed mantissa
//
// The value is `mantissa * 10^exp`. The 4-bit exponent code encodes both the
// power-of-ten exponent and, for integers, whether the source had a decimal
// point:
//
//   code  0..=6  -> exp -7..=-1  (fractional; always has a decimal point)
//   code  7      -> exp 0, no decimal point (plain integer)
//   code  8      -> exp 0, with decimal point ("N.0")
//   code  9..=15 -> exp 1..=7    (integer; no decimal point)
//
// Integers with a decimal point only ever occur at exponent 0: a larger such
// integer cannot fit the mantissa in fractional form, so it falls back to a
// heap `f64`. That is why a single pair of exponent-0 codes suffices to carry
// the decimal-point flag, leaving the layout within 4 exponent bits.
//
// The all-zero value (mantissa 0, code 0) is a non-canonical zero and is never
// emitted, reserving it as the `NonNull` niche.
// ===========================================================================

const EXP_SHIFT: u32 = 4;
const MANTISSA_SHIFT: u32 = 8;
/// Bits available for the signed inline mantissa (56 on 64-bit, 24 on 32-bit).
const MANTISSA_BITS: u32 = usize::BITS - MANTISSA_SHIFT;

const POW5: [u128; 8] = [1, 5, 25, 125, 625, 3125, 15625, 78125];

fn fits_mantissa(m: i128) -> bool {
    let limit = 1i128 << (MANTISSA_BITS - 1);
    m >= -limit && m < limit
}

/// Maps an exponent (and, at exp 0, a decimal-point flag) to its 4-bit code.
fn exp_code(exp: i32, dot: bool) -> usize {
    match exp {
        -7..=-1 => (exp + 7) as usize,
        0 => usize::from(dot) + 7,
        1..=7 => (exp + 8) as usize,
        _ => unreachable!("inline exponent out of range"),
    }
}
fn code_exp(code: usize) -> i32 {
    match code {
        0..=6 => code as i32 - 7,
        7 | 8 => 0,
        _ => code as i32 - 8,
    }
}
fn code_has_dot(code: usize) -> bool {
    code <= 6 || code == 8
}

fn encode(mantissa: i64, code: usize) -> IValue {
    let payload = ((mantissa as usize) << MANTISSA_SHIFT) | (code << EXP_SHIFT);
    // Safety: tag Inline (0) with the number sub-family (bit 3 clear); the low 3
    // bits are clear and canonical values never encode (mantissa 0, code 0), so
    // the result is non-zero.
    unsafe { IValue::new_inline(TypeTag::Inline, payload) }
}

fn inline_mantissa(v: &IValue) -> i64 {
    // Arithmetic shift sign-extends the mantissa from the top bits.
    ((v.ptr_usize() as isize) >> MANTISSA_SHIFT) as i64
}
fn inline_code(v: &IValue) -> usize {
    (v.ptr_usize() >> EXP_SHIFT) & 0xf
}

// --- f64 decomposition ------------------------------------------------------

/// Decomposes a finite, non-zero `f64` into `(mantissa, exp2, negative)` such
/// that `value == (-1)^negative * mantissa * 2^exp2`.
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

/// If `value` (= `sign * m * 2^e2`) is an exact integer, returns it.
fn f64_as_integer(m: u64, e2: i32, neg: bool) -> Option<i128> {
    let mag: u128 = if e2 >= 0 {
        (u128::from(m)).checked_shl(e2 as u32)?
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
        u128::from(m)
            .checked_mul(POW5[k as usize])?
            .checked_shl(e as u32)?
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

// --- Inline encoders (return `None` if the value doesn't fit inline) ---------

/// Encodes an integer with no decimal point, factoring out trailing zeros as
/// needed to fit the mantissa.
fn encode_int(value: i128) -> Option<IValue> {
    let mut m = value;
    let mut exp = 0i32;
    loop {
        if fits_mantissa(m) {
            return Some(encode(m as i64, exp_code(exp, false)));
        }
        if exp >= 7 || m % 10 != 0 {
            return None;
        }
        m /= 10;
        exp += 1;
    }
}

/// Encodes an integer-valued number that had a decimal point (`"N.0"`); these
/// only fit inline at exponent 0.
fn encode_int_dot(value: i128) -> Option<IValue> {
    if fits_mantissa(value) {
        Some(encode(value as i64, 8))
    } else {
        None
    }
}

/// Encodes a finite `f64` inline as an exact decimal, if it fits.
fn encode_f64(value: f64) -> Option<IValue> {
    if value == 0.0 {
        // 0.0 / -0.0: integer zero that had a decimal point.
        return Some(encode(0, 8));
    }
    let (m, e2, neg) = integer_decode(value);
    // Integer-valued float: store at exp 0 with the decimal-point flag.
    if let Some(int) = f64_as_integer(m, e2, neg) {
        return encode_int_dot(int);
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

// --- Heap payloads ----------------------------------------------------------

fn heap_layout() -> Layout {
    // An 8-byte payload, 8-aligned so the tag bits stay free.
    Layout::from_size_align(8, 8).unwrap()
}

fn new_heap(tag: TypeTag, bits: u64) -> IValue {
    // Safety: freshly allocated, 8-aligned, non-null.
    unsafe {
        let ptr = alloc_infallible(heap_layout()).cast::<u64>();
        ptr.as_ptr().write(bits);
        IValue::new_ptr(ptr.cast(), tag)
    }
}

// Safety: must be a heap number (tag `NumberI64`/`NumberU64`/`NumberF64`).
unsafe fn heap_bits(v: &IValue) -> u64 {
    v.ptr().cast::<u64>().as_ptr().read()
}

// --- Decimal decoders -------------------------------------------------------

/// The exact integer value of `mantissa * 10^exp`, if it is an integer.
fn decimal_to_i128(m: i64, exp: i32) -> Option<i128> {
    if exp >= 0 {
        i128::from(m).checked_mul(10i128.pow(exp as u32))
    } else {
        let div = 10i128.pow((-exp) as u32);
        let m = i128::from(m);
        if m % div == 0 {
            Some(m / div)
        } else {
            None
        }
    }
}

fn decimal_to_f64_lossy(m: i64, exp: i32) -> f64 {
    // The inline exponent range is small, so `10^|exp|` is an exact integer and
    // an exact `f64`; using it (rather than the non-deterministic `powi`) keeps
    // the result deterministic and exact whenever the value is representable.
    if exp >= 0 {
        m as f64 * 10i64.pow(exp as u32) as f64
    } else {
        m as f64 / 10i64.pow((-exp) as u32) as f64
    }
}

/// `true` if `v` is exactly representable as an `f64`.
fn i128_fits_f64(v: i128) -> bool {
    if v == 0 {
        return true;
    }
    let a = v.unsigned_abs();
    128 - a.leading_zeros() - a.trailing_zeros() <= 53
}

/// The exact `f64` value of `mantissa * 10^exp`, if it is exactly representable.
fn decimal_to_f64_exact(m: i64, exp: i32) -> Option<f64> {
    if m == 0 {
        return Some(0.0);
    }
    if exp >= 0 {
        let v = decimal_to_i128(m, exp)?;
        i128_fits_f64(v).then_some(v as f64)
    } else {
        let k = (-exp) as u32;
        let p5 = 5i128.pow(k);
        let mi = i128::from(m);
        if mi % p5 != 0 {
            return None;
        }
        let num = mi / p5; // value == num / 2^k, a dyadic rational
                           // Dividing an exactly-representable integer by a power of two is exact
                           // (it only adjusts the exponent), and avoids the non-deterministic
                           // `powi`.
        i128_fits_f64(num).then_some(num as f64 / (1u64 << k) as f64)
    }
}

// --- Numeric comparison -----------------------------------------------------

fn can_represent_as_f64(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 11
}

/// Compares an exact integer to a finite float exactly.
fn cmp_int_f64(a: i128, b: f64) -> Ordering {
    const LIMIT: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0; // 2^127
    if b >= LIMIT {
        return Ordering::Less;
    }
    if b <= -LIMIT {
        return Ordering::Greater;
    }
    let bt = b.trunc();
    match a.cmp(&(bt as i128)) {
        Ordering::Equal => {
            if b == bt {
                Ordering::Equal
            } else if b > bt {
                Ordering::Less // b has a positive fractional part, so b > a
            } else {
                Ordering::Greater
            }
        }
        ord => ord,
    }
}

/// A number reduced to a form suitable for exact numeric comparison. Integer
/// values (including integer-valued floats within range) become `Int`; genuinely
/// fractional values become `Float`.
enum NumVal {
    Int(i128),
    Float(f64),
}

fn cmp_num(a: &NumVal, b: &NumVal) -> Ordering {
    match (a, b) {
        (NumVal::Int(x), NumVal::Int(y)) => x.cmp(y),
        (NumVal::Int(x), NumVal::Float(y)) => cmp_int_f64(*x, *y),
        (NumVal::Float(x), NumVal::Int(y)) => cmp_int_f64(*y, *x).reverse(),
        (NumVal::Float(x), NumVal::Float(y)) => x.partial_cmp(y).unwrap(),
    }
}

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
        Self::new_i64(0)
    }
    /// Returns the number one (without a decimal point). Does not allocate.
    #[must_use]
    pub fn one() -> Self {
        Self::new_i64(1)
    }

    fn new_i64(value: i64) -> Self {
        INumber(
            encode_int(i128::from(value))
                .unwrap_or_else(|| new_heap(TypeTag::NumberI64, value as u64)),
        )
    }

    fn new_u64(value: u64) -> Self {
        if let Ok(v) = i64::try_from(value) {
            Self::new_i64(v)
        } else {
            // Too large for `i64`; still prefer inline if it factors, else heap `u64`.
            INumber(
                encode_int(i128::from(value))
                    .unwrap_or_else(|| new_heap(TypeTag::NumberU64, value)),
            )
        }
    }

    fn new_f64(value: f64) -> Self {
        INumber(encode_f64(value).unwrap_or_else(|| new_heap(TypeTag::NumberF64, value.to_bits())))
    }

    fn num_val(&self) -> NumVal {
        if self.0.is_inline_number() {
            let m = inline_mantissa(&self.0);
            let exp = code_exp(inline_code(&self.0));
            match decimal_to_i128(m, exp) {
                Some(i) => NumVal::Int(i),
                None => NumVal::Float(decimal_to_f64_lossy(m, exp)),
            }
        } else {
            // Safety: not inline, so it is a heap number.
            unsafe {
                match self.0.type_tag() {
                    TypeTag::NumberI64 => NumVal::Int(i128::from(heap_bits(&self.0) as i64)),
                    TypeTag::NumberU64 => NumVal::Int(i128::from(heap_bits(&self.0))),
                    _ => NumVal::Float(f64::from_bits(heap_bits(&self.0))),
                }
            }
        }
    }

    /// Converts this number to an i64 if it can be represented exactly.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        match self.num_val() {
            NumVal::Int(v) => i64::try_from(v).ok(),
            NumVal::Float(v) => {
                if v.fract() == 0.0 && v >= i64::MIN as f64 && v < i64::MAX as f64 {
                    Some(v as i64)
                } else {
                    None
                }
            }
        }
    }
    /// Converts this number to a u64 if it can be represented exactly.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match self.num_val() {
            NumVal::Int(v) => u64::try_from(v).ok(),
            NumVal::Float(v) => {
                if v.fract() == 0.0 && v >= 0.0 && v < u64::MAX as f64 {
                    Some(v as u64)
                } else {
                    None
                }
            }
        }
    }
    /// Converts this number to an f64 if it can be represented exactly.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        if self.0.is_inline_number() {
            decimal_to_f64_exact(inline_mantissa(&self.0), code_exp(inline_code(&self.0)))
        } else {
            // Safety: not inline, so it is a heap number.
            unsafe {
                match self.0.type_tag() {
                    TypeTag::NumberI64 => {
                        let v = heap_bits(&self.0) as i64;
                        can_represent_as_f64(v.unsigned_abs()).then_some(v as f64)
                    }
                    TypeTag::NumberU64 => {
                        let v = heap_bits(&self.0);
                        can_represent_as_f64(v).then_some(v as f64)
                    }
                    _ => Some(f64::from_bits(heap_bits(&self.0))),
                }
            }
        }
    }
    /// Converts this number to an f32 if it can be represented exactly.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        // A value is exactly an f32 only if it is exactly an f64.
        let v = self.to_f64()?;
        let u = v as f32;
        (f64::from(u) == v).then_some(u)
    }
    /// Converts this number to an i32 if it can be represented exactly.
    #[must_use]
    pub fn to_i32(&self) -> Option<i32> {
        self.to_i64().and_then(|x| x.try_into().ok())
    }
    /// Converts this number to a u32 if it can be represented exactly.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        self.to_u64().and_then(|x| x.try_into().ok())
    }
    /// Converts this number to an isize if it can be represented exactly.
    #[must_use]
    pub fn to_isize(&self) -> Option<isize> {
        self.to_i64().and_then(|x| x.try_into().ok())
    }
    /// Converts this number to a usize if it can be represented exactly.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        self.to_u64().and_then(|x| x.try_into().ok())
    }
    /// Converts this number to an f64, potentially losing precision in the process.
    #[must_use]
    pub fn to_f64_lossy(&self) -> f64 {
        if self.0.is_inline_number() {
            decimal_to_f64_lossy(inline_mantissa(&self.0), code_exp(inline_code(&self.0)))
        } else {
            // Safety: not inline, so it is a heap number.
            unsafe {
                match self.0.type_tag() {
                    TypeTag::NumberI64 => heap_bits(&self.0) as i64 as f64,
                    TypeTag::NumberU64 => heap_bits(&self.0) as f64,
                    _ => f64::from_bits(heap_bits(&self.0)),
                }
            }
        }
    }
    /// Converts this number to an f32, potentially losing precision in the process.
    #[must_use]
    pub fn to_f32_lossy(&self) -> f32 {
        self.to_f64_lossy() as f32
    }

    /// This allows distinguishing between `1.0` and `1` in the original JSON.
    /// Numeric operations will otherwise treat these two values as equivalent.
    #[must_use]
    pub fn has_decimal_point(&self) -> bool {
        if self.0.is_inline_number() {
            code_has_dot(inline_code(&self.0))
        } else {
            self.0.type_tag() == TypeTag::NumberF64
        }
    }

    // Only reached for heap numbers: `IValue::clone` copies inline values
    // directly. Copies the payload into a fresh allocation.
    pub(crate) fn clone_impl(&self) -> IValue {
        debug_assert!(!self.0.is_inline_number());
        // Safety: heap number payload.
        unsafe { new_heap(self.0.type_tag(), heap_bits(&self.0)) }
    }
    // Only reached for heap numbers: `IValue::drop` leaves inline values alone.
    pub(crate) fn drop_impl(&mut self) {
        debug_assert!(!self.0.is_inline_number());
        // Safety: heap number; free its payload.
        unsafe { dealloc_infallible(self.0.ptr(), heap_layout()) }
    }
}

impl Hash for INumber {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if let Some(v) = self.to_i64() {
            v.hash(state);
        } else if let Some(v) = self.to_u64() {
            v.hash(state);
        } else {
            let v = self.to_f64_lossy();
            let bits = if v == 0.0 { 0 } else { v.to_bits() };
            bits.hash(state);
        }
    }
}

impl From<u64> for INumber {
    fn from(v: u64) -> Self {
        Self::new_u64(v)
    }
}
impl From<u32> for INumber {
    fn from(v: u32) -> Self {
        Self::new_u64(u64::from(v))
    }
}
impl From<u16> for INumber {
    fn from(v: u16) -> Self {
        Self::new_u64(u64::from(v))
    }
}
impl From<u8> for INumber {
    fn from(v: u8) -> Self {
        Self::new_u64(u64::from(v))
    }
}
impl From<usize> for INumber {
    fn from(v: usize) -> Self {
        Self::new_u64(v as u64)
    }
}

impl From<i64> for INumber {
    fn from(v: i64) -> Self {
        Self::new_i64(v)
    }
}
impl From<i32> for INumber {
    fn from(v: i32) -> Self {
        Self::new_i64(i64::from(v))
    }
}
impl From<i16> for INumber {
    fn from(v: i16) -> Self {
        Self::new_i64(i64::from(v))
    }
}
impl From<i8> for INumber {
    fn from(v: i8) -> Self {
        Self::new_i64(i64::from(v))
    }
}
impl From<isize> for INumber {
    fn from(v: isize) -> Self {
        Self::new_i64(v as i64)
    }
}

impl TryFrom<f64> for INumber {
    type Error = ();
    fn try_from(v: f64) -> Result<Self, ()> {
        if v.is_finite() {
            Ok(Self::new_f64(v))
        } else {
            Err(())
        }
    }
}

impl TryFrom<f32> for INumber {
    type Error = ();
    fn try_from(v: f32) -> Result<Self, ()> {
        if v.is_finite() {
            Ok(Self::new_f64(f64::from(v)))
        } else {
            Err(())
        }
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
        if self.0.raw_eq(&other.0) {
            Ordering::Equal
        } else {
            cmp_num(&self.num_val(), &other.num_val())
        }
    }
}
impl PartialOrd for INumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Debug for INumber {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(v) = self.to_i64() {
            Debug::fmt(&v, f)
        } else if let Some(v) = self.to_u64() {
            Debug::fmt(&v, f)
        } else {
            Debug::fmt(&self.to_f64_lossy(), f)
        }
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
            assert!(n.0.is_inline_number(), "{} should be inline", v);
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
            assert!(n.0.is_inline_number(), "{} should be inline", s);
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
        // Large round integers factor into the inline exponent.
        for v in [
            10i64.pow(16),
            10i64.pow(17),
            10i64.pow(18),
            -(10i64.pow(18)),
        ] {
            let n = INumber::from(v);
            assert!(n.0.is_inline_number(), "{} should factor inline", v);
            assert_eq!(n.to_i64(), Some(v));
        }
        // A large non-round integer cannot factor and uses the heap.
        let prime = 9_999_999_999_999_937i64; // > 2^53, not divisible by 10
        assert_eq!(INumber::from(prime).to_i64(), Some(prime));
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

    #[mockalloc::test]
    fn large_values_use_heap() {
        let big = INumber::from(u64::MAX);
        assert!(!big.0.is_inline_number());
        assert_eq!(big.to_u64(), Some(u64::MAX));

        let pi = INumber::try_from(std::f64::consts::PI).unwrap();
        assert!(!pi.0.is_inline_number());
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
}
