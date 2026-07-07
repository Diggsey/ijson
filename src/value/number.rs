//! Number-type logic spanning the inline and heap-scalar representations.
//!
//! A JSON number is not itself a representation: it is stored either as an
//! inline decimal (see [`super::inline::number`]) or as a heap scalar
//! `i64`/`u64`/`f64` (see [`super::scalar`]). This module holds the logic that
//! spans those representations — construction, exact/lossy conversions,
//! comparison and hashing — as free functions on `&IValue`. `IValue`'s own
//! trait impls delegate here, and the public [`crate::INumber`] wrapper does too.
#![allow(clippy::float_cmp)]

use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};

use super::inline::number as inl;
use super::scalar;
use super::{IValue, TypeTag};

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
