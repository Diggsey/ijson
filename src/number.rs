//! Functionality relating to the JSON number type.
//!
//! `INumber` is a *type* that spans several representations — the inline decimal
//! (see [`crate::inline::number`]) and the heap scalars `i64`/`u64`/`f64` (see
//! [`crate::scalar`]). The number-specific logic lives as free functions here
//! which pick the representation by tag; both the root value type and the thin
//! `INumber` wrapper delegate to them.
#![allow(clippy::float_cmp)]

use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};

use crate::inline::number as inl;
use crate::scalar;
use crate::value::{IValue, TypeTag};

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

// Number-type operations spanning the inline and scalar representations. These
// are free functions on `&IValue` (not methods on `IValue`); the root value type
// delegates to them, and `INumber` delegates to them too. Each assumes the value
// is a number.

pub(crate) fn new_i64(value: i64) -> IValue {
    match inl::encode_int(i128::from(value)) {
        // Safety: `encode_int` returns valid inline bits; the scalar allocation
        // is aligned and non-null.
        Some(bits) => unsafe { IValue::new_inline(TypeTag::Inline, bits) },
        None => unsafe { IValue::new_ptr(scalar::alloc(value as u64), TypeTag::NumberI64) },
    }
}

pub(crate) fn new_u64(value: u64) -> IValue {
    if let Ok(v) = i64::try_from(value) {
        new_i64(v)
    } else {
        // Too large for `i64`; still prefer inline if it factors, else heap `u64`.
        match inl::encode_int(i128::from(value)) {
            Some(bits) => unsafe { IValue::new_inline(TypeTag::Inline, bits) },
            None => unsafe { IValue::new_ptr(scalar::alloc(value), TypeTag::NumberU64) },
        }
    }
}

pub(crate) fn new_f64(value: f64) -> IValue {
    match inl::encode_f64(value) {
        Some(bits) => unsafe { IValue::new_inline(TypeTag::Inline, bits) },
        None => unsafe { IValue::new_ptr(scalar::alloc(value.to_bits()), TypeTag::NumberF64) },
    }
}

// Safety: reads the heap scalar payload; only called for heap number tags.
unsafe fn scalar_bits(v: &IValue) -> u64 {
    scalar::read(v.ptr())
}

fn num_val(v: &IValue) -> NumVal {
    if v.is_inline() {
        let bits = v.ptr_usize();
        match inl::value_i128(bits) {
            Some(i) => NumVal::Int(i),
            None => NumVal::Float(inl::to_f64_lossy(bits)),
        }
    } else {
        // Safety: not inline, so it is a heap scalar number.
        unsafe {
            match v.type_tag() {
                TypeTag::NumberI64 => NumVal::Int(i128::from(scalar_bits(v) as i64)),
                TypeTag::NumberU64 => NumVal::Int(i128::from(scalar_bits(v))),
                _ => NumVal::Float(f64::from_bits(scalar_bits(v))),
            }
        }
    }
}

pub(crate) fn to_i64(v: &IValue) -> Option<i64> {
    match num_val(v) {
        NumVal::Int(x) => i64::try_from(x).ok(),
        NumVal::Float(x) => {
            (x.fract() == 0.0 && x >= i64::MIN as f64 && x < i64::MAX as f64).then_some(x as i64)
        }
    }
}

pub(crate) fn to_u64(v: &IValue) -> Option<u64> {
    match num_val(v) {
        NumVal::Int(x) => u64::try_from(x).ok(),
        NumVal::Float(x) => {
            (x.fract() == 0.0 && x >= 0.0 && x < u64::MAX as f64).then_some(x as u64)
        }
    }
}

pub(crate) fn to_f64(v: &IValue) -> Option<f64> {
    if v.is_inline() {
        inl::to_f64_exact(v.ptr_usize())
    } else {
        // Safety: not inline, so it is a heap scalar number.
        unsafe {
            match v.type_tag() {
                TypeTag::NumberI64 => {
                    let x = scalar_bits(v) as i64;
                    can_represent_as_f64(x.unsigned_abs()).then_some(x as f64)
                }
                TypeTag::NumberU64 => {
                    let x = scalar_bits(v);
                    can_represent_as_f64(x).then_some(x as f64)
                }
                _ => Some(f64::from_bits(scalar_bits(v))),
            }
        }
    }
}

pub(crate) fn to_f64_lossy(v: &IValue) -> f64 {
    if v.is_inline() {
        inl::to_f64_lossy(v.ptr_usize())
    } else {
        // Safety: not inline, so it is a heap scalar number.
        unsafe {
            match v.type_tag() {
                TypeTag::NumberI64 => scalar_bits(v) as i64 as f64,
                TypeTag::NumberU64 => scalar_bits(v) as f64,
                _ => f64::from_bits(scalar_bits(v)),
            }
        }
    }
}

pub(crate) fn has_decimal_point(v: &IValue) -> bool {
    if v.is_inline() {
        inl::has_decimal_point(v.ptr_usize())
    } else {
        v.type_tag() == TypeTag::NumberF64
    }
}

pub(crate) fn to_i32(v: &IValue) -> Option<i32> {
    to_i64(v).and_then(|x| x.try_into().ok())
}
pub(crate) fn to_u32(v: &IValue) -> Option<u32> {
    to_u64(v).and_then(|x| x.try_into().ok())
}
pub(crate) fn to_isize(v: &IValue) -> Option<isize> {
    to_i64(v).and_then(|x| x.try_into().ok())
}
pub(crate) fn to_usize(v: &IValue) -> Option<usize> {
    to_u64(v).and_then(|x| x.try_into().ok())
}
pub(crate) fn to_f32(v: &IValue) -> Option<f32> {
    // A value is exactly an f32 only if it is exactly an f64.
    let x = to_f64(v)?;
    let u = x as f32;
    (f64::from(u) == x).then_some(u)
}
pub(crate) fn to_f32_lossy(v: &IValue) -> f32 {
    to_f64_lossy(v) as f32
}

pub(crate) fn hash<H: Hasher>(v: &IValue, state: &mut H) {
    if let Some(x) = to_i64(v) {
        x.hash(state);
    } else if let Some(x) = to_u64(v) {
        x.hash(state);
    } else {
        let f = to_f64_lossy(v);
        (if f == 0.0 { 0 } else { f.to_bits() }).hash(state);
    }
}

pub(crate) fn cmp(a: &IValue, b: &IValue) -> Ordering {
    if a.raw_eq(b) {
        Ordering::Equal
    } else {
        cmp_num(&num_val(a), &num_val(b))
    }
}

pub(crate) fn debug(v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
    if let Some(x) = to_i64(v) {
        Debug::fmt(&x, f)
    } else if let Some(x) = to_u64(v) {
        Debug::fmt(&x, f)
    } else {
        Debug::fmt(&to_f64_lossy(v), f)
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
        INumber(new_i64(0))
    }
    /// Returns the number one (without a decimal point). Does not allocate.
    #[must_use]
    pub fn one() -> Self {
        INumber(new_i64(1))
    }

    /// Converts this number to an i64 if it can be represented exactly.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        to_i64(&self.0)
    }
    /// Converts this number to a u64 if it can be represented exactly.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        to_u64(&self.0)
    }
    /// Converts this number to an f64 if it can be represented exactly.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        to_f64(&self.0)
    }
    /// Converts this number to an f32 if it can be represented exactly.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        to_f32(&self.0)
    }
    /// Converts this number to an i32 if it can be represented exactly.
    #[must_use]
    pub fn to_i32(&self) -> Option<i32> {
        to_i32(&self.0)
    }
    /// Converts this number to a u32 if it can be represented exactly.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        to_u32(&self.0)
    }
    /// Converts this number to an isize if it can be represented exactly.
    #[must_use]
    pub fn to_isize(&self) -> Option<isize> {
        to_isize(&self.0)
    }
    /// Converts this number to a usize if it can be represented exactly.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        to_usize(&self.0)
    }
    /// Converts this number to an f64, potentially losing precision in the process.
    #[must_use]
    pub fn to_f64_lossy(&self) -> f64 {
        to_f64_lossy(&self.0)
    }
    /// Converts this number to an f32, potentially losing precision in the process.
    #[must_use]
    pub fn to_f32_lossy(&self) -> f32 {
        to_f32_lossy(&self.0)
    }

    /// This allows distinguishing between `1.0` and `1` in the original JSON.
    /// Numeric operations will otherwise treat these two values as equivalent.
    #[must_use]
    pub fn has_decimal_point(&self) -> bool {
        has_decimal_point(&self.0)
    }
}

impl Hash for INumber {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash(&self.0, state);
    }
}

impl From<u64> for INumber {
    fn from(v: u64) -> Self {
        INumber(new_u64(v))
    }
}
impl From<u32> for INumber {
    fn from(v: u32) -> Self {
        INumber(new_u64(u64::from(v)))
    }
}
impl From<u16> for INumber {
    fn from(v: u16) -> Self {
        INumber(new_u64(u64::from(v)))
    }
}
impl From<u8> for INumber {
    fn from(v: u8) -> Self {
        INumber(new_u64(u64::from(v)))
    }
}
impl From<usize> for INumber {
    fn from(v: usize) -> Self {
        INumber(new_u64(v as u64))
    }
}

impl From<i64> for INumber {
    fn from(v: i64) -> Self {
        INumber(new_i64(v))
    }
}
impl From<i32> for INumber {
    fn from(v: i32) -> Self {
        INumber(new_i64(i64::from(v)))
    }
}
impl From<i16> for INumber {
    fn from(v: i16) -> Self {
        INumber(new_i64(i64::from(v)))
    }
}
impl From<i8> for INumber {
    fn from(v: i8) -> Self {
        INumber(new_i64(i64::from(v)))
    }
}
impl From<isize> for INumber {
    fn from(v: isize) -> Self {
        INumber(new_i64(v as i64))
    }
}

impl TryFrom<f64> for INumber {
    type Error = ();
    fn try_from(v: f64) -> Result<Self, ()> {
        if v.is_finite() {
            Ok(INumber(new_f64(v)))
        } else {
            Err(())
        }
    }
}

impl TryFrom<f32> for INumber {
    type Error = ();
    fn try_from(v: f32) -> Result<Self, ()> {
        if v.is_finite() {
            Ok(INumber(new_f64(f64::from(v))))
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
        cmp(&self.0, &other.0)
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
        // A large round integer that exceeds the mantissa still factors into the
        // inline exponent (the threshold differs by pointer width).
        let big_round = if usize::BITS == 64 {
            10i64.pow(18)
        } else {
            10i64.pow(8)
        };
        for v in [big_round, -big_round] {
            let n = INumber::from(v);
            assert!(n.0.is_inline(), "{} should factor inline", v);
            assert_eq!(n.to_i64(), Some(v));
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
}
