use std::alloc::{alloc, dealloc, Layout, LayoutErr};
use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;

use super::value::{IValue, TypeTag};

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NumberType {
    Static,
    I24,
    I64,
    U64,
    F64,
}

#[repr(C)]
#[repr(align(4))]
struct Header {
    type_: NumberType,
    short: u8,
    static_: i16,
}

fn can_represent_as_f64(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 11
}

fn can_represent_as_f32(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 40
}

fn cmp_i64_to_f64(a: i64, b: f64) -> Ordering {
    if a < 0 {
        cmp_u64_to_f64(a.wrapping_neg() as u64, -b).reverse()
    } else {
        cmp_u64_to_f64(a as u64, b)
    }
}

fn cmp_u64_to_f64(a: u64, b: f64) -> Ordering {
    if can_represent_as_f64(a) {
        // If we can represent as an f64, we can just cast and compare
        (a as f64).partial_cmp(&b).unwrap()
    } else if b <= (0x20000000000000u64 as f64) {
        // If the floating point number is less than all non-representable
        // integers, and our integer is non-representable, then we know
        // the integer is greater.
        Ordering::Greater
    } else if b >= u64::MAX as f64 {
        // If the floating point number is larger than the largest u64, then
        // the integer is smaller.
        Ordering::Less
    } else {
        // The remaining floating point values can be losslessly converted to u64.
        a.cmp(&(b as u64))
    }
}

impl Header {
    fn as_i24_unchecked(&self) -> i32 {
        ((self.static_ as i32) << 8) | (self.short as i32)
    }
    unsafe fn as_i64_unchecked(&self) -> &i64 {
        &*(self as *const _ as *const i64).offset(1)
    }
    unsafe fn as_u64_unchecked(&self) -> &u64 {
        &*(self as *const _ as *const u64).offset(1)
    }
    unsafe fn as_f64_unchecked(&self) -> &f64 {
        &*(self as *const _ as *const f64).offset(1)
    }
    unsafe fn as_i64_unchecked_mut(&mut self) -> &mut i64 {
        &mut *(self as *mut _ as *mut i64).offset(1)
    }
    unsafe fn as_u64_unchecked_mut(&mut self) -> &mut u64 {
        &mut *(self as *mut _ as *mut u64).offset(1)
    }
    unsafe fn as_f64_unchecked_mut(&mut self) -> &mut f64 {
        &mut *(self as *mut _ as *mut f64).offset(1)
    }
    fn to_i64(&self) -> Option<i64> {
        // Safety: We only call methods appropriate for the type
        unsafe {
            match self.type_ {
                NumberType::Static => Some(self.static_ as i64),
                NumberType::I24 => Some(self.as_i24_unchecked() as i64),
                NumberType::I64 => Some(*self.as_i64_unchecked()),
                NumberType::U64 => {
                    let v = *self.as_u64_unchecked();
                    if v <= i64::MAX as u64 {
                        Some(v as i64)
                    } else {
                        None
                    }
                }
                NumberType::F64 => {
                    let v = *self.as_f64_unchecked();
                    if v.fract() == 0.0 {
                        if v > i64::MIN as f64 && v < i64::MAX as f64 {
                            return Some(v as i64);
                        }
                    }
                    None
                }
            }
        }
    }
    fn to_u64(&self) -> Option<u64> {
        // Safety: We only call methods appropriate for the type
        unsafe {
            match self.type_ {
                NumberType::Static => {
                    if self.static_ >= 0 {
                        Some(self.static_ as u64)
                    } else {
                        None
                    }
                }
                NumberType::I24 => {
                    let v = self.as_i24_unchecked();
                    if v >= 0 {
                        Some(v as u64)
                    } else {
                        None
                    }
                }
                NumberType::I64 => {
                    let v = *self.as_i64_unchecked();
                    if v >= 0 {
                        Some(v as u64)
                    } else {
                        None
                    }
                }
                NumberType::U64 => Some(*self.as_u64_unchecked()),
                NumberType::F64 => {
                    let v = *self.as_f64_unchecked();
                    if v.fract() == 0.0 {
                        if v > 0.0 && v < u64::MAX as f64 {
                            return Some(v as u64);
                        }
                    }
                    None
                }
            }
        }
    }
    fn to_f64(&self) -> Option<f64> {
        // Safety: We only call methods appropriate for the type
        unsafe {
            match self.type_ {
                NumberType::Static => Some(self.static_ as f64),
                NumberType::I24 => Some(self.as_i24_unchecked() as f64),
                NumberType::I64 => {
                    let v = *self.as_i64_unchecked();
                    let can_represent = if v < 0 {
                        can_represent_as_f64(v.wrapping_neg() as u64)
                    } else {
                        can_represent_as_f64(v as u64)
                    };
                    if can_represent {
                        Some(v as f64)
                    } else {
                        None
                    }
                }
                NumberType::U64 => {
                    let v = *self.as_u64_unchecked();
                    if can_represent_as_f64(v) {
                        Some(v as f64)
                    } else {
                        None
                    }
                }
                NumberType::F64 => Some(*self.as_f64_unchecked()),
            }
        }
    }
    fn to_f32(&self) -> Option<f32> {
        // Safety: We only call methods appropriate for the type
        unsafe {
            match self.type_ {
                NumberType::Static => Some(self.static_ as f32),
                NumberType::I24 => Some(self.as_i24_unchecked() as f32),
                NumberType::I64 => {
                    let v = *self.as_i64_unchecked();
                    let can_represent = if v < 0 {
                        can_represent_as_f32(v.wrapping_neg() as u64)
                    } else {
                        can_represent_as_f32(v as u64)
                    };
                    if can_represent {
                        Some(v as f32)
                    } else {
                        None
                    }
                }
                NumberType::U64 => {
                    let v = *self.as_u64_unchecked();
                    if can_represent_as_f32(v) {
                        Some(v as f32)
                    } else {
                        None
                    }
                }
                NumberType::F64 => {
                    let v = *self.as_f64_unchecked();
                    let u = v as f32;
                    if v == (u as f64) {
                        Some(u)
                    } else {
                        None
                    }
                }
            }
        }
    }
    fn has_decimal_point(&self) -> bool {
        match self.type_ {
            NumberType::Static | NumberType::I24 | NumberType::I64 | NumberType::U64 => false,
            NumberType::F64 => true,
        }
    }
    fn to_f64_lossy(&self) -> f64 {
        unsafe {
            match self.type_ {
                NumberType::Static => self.static_ as f64,
                NumberType::I24 => self.as_i24_unchecked() as f64,
                NumberType::I64 => *self.as_i64_unchecked() as f64,
                NumberType::U64 => *self.as_u64_unchecked() as f64,
                NumberType::F64 => *self.as_f64_unchecked(),
            }
        }
    }
    fn cmp(&self, other: &Header) -> Ordering {
        // Fast path
        if self.type_ == other.type_ {
            // Safety: We only call methods for the appropriate type
            unsafe {
                match self.type_ {
                    NumberType::Static => self.static_.cmp(&other.static_),
                    NumberType::I24 => self.as_i24_unchecked().cmp(&other.as_i24_unchecked()),
                    NumberType::I64 => self.as_i64_unchecked().cmp(other.as_i64_unchecked()),
                    NumberType::U64 => self.as_u64_unchecked().cmp(other.as_u64_unchecked()),
                    NumberType::F64 => self
                        .as_f64_unchecked()
                        .partial_cmp(other.as_f64_unchecked())
                        .unwrap(),
                }
            }
        } else {
            // Safety: We only call methods for the appropriate type
            unsafe {
                match (self.type_, other.type_) {
                    (NumberType::U64, NumberType::F64) => {
                        cmp_u64_to_f64(*self.as_u64_unchecked(), *other.as_f64_unchecked())
                    }
                    (NumberType::F64, NumberType::U64) => {
                        cmp_u64_to_f64(*other.as_u64_unchecked(), *self.as_f64_unchecked())
                            .reverse()
                    }
                    (NumberType::I64, NumberType::F64) => {
                        cmp_i64_to_f64(*self.as_i64_unchecked(), *other.as_f64_unchecked())
                    }
                    (NumberType::F64, NumberType::I64) => {
                        cmp_i64_to_f64(*other.as_i64_unchecked(), *self.as_f64_unchecked())
                            .reverse()
                    }
                    (_, NumberType::F64) => self
                        .to_f64()
                        .unwrap()
                        .partial_cmp(other.as_f64_unchecked())
                        .unwrap(),
                    (NumberType::F64, _) => other
                        .to_f64()
                        .unwrap()
                        .partial_cmp(self.as_f64_unchecked())
                        .unwrap()
                        .reverse(),
                    (NumberType::U64, _) => Ordering::Greater,
                    (_, NumberType::U64) => Ordering::Less,
                    _ => (self.to_i64().cmp(&other.to_i64())),
                }
            }
        }
    }
}

macro_rules! define_static_numbers {
    (@recurse $from:ident ($($offset:expr,)*) ()) => {
        [$(Header {
            type_: NumberType::Static,
            short: 0,
            static_: $from + ($offset),
        }),*]
    };
    (@recurse $from:ident ($($offset:expr,)*) ($u:literal $($v:literal)*)) => {
        define_static_numbers!(@recurse $from ($($offset,)* $($offset | (1 << $u),)*) ($($v)*))
    };
    ($from:ident $($v:literal)*) => {
        define_static_numbers!(@recurse $from (0,) ($($v)*))
    };
}

// We want to cover the range -128..256 with static numbers so that arrays of i8 and u8 can be
// stored reasonably efficiently. In practice, we end up covering -128..384.
const STATIC_LOWER: i16 = -128;
const STATIC_LEN: usize = 512;
const STATIC_UPPER: i16 = STATIC_LOWER + STATIC_LEN as i16;
static STATIC_NUMBERS: [Header; STATIC_LEN] =
    define_static_numbers!(STATIC_LOWER 0 1 2 3 4 5 6 7 8);

#[repr(transparent)]
#[derive(Clone)]
pub struct INumber(pub(crate) IValue);

value_subtype_impls!(INumber, into_number, as_number, as_number_mut);

impl INumber {
    fn layout(type_: NumberType) -> Result<Layout, LayoutErr> {
        let mut res = Layout::new::<Header>();
        match type_ {
            NumberType::Static => unreachable!(),
            NumberType::I24 => {}
            NumberType::I64 => res = res.extend(Layout::new::<i64>())?.0.pad_to_align(),
            NumberType::U64 => res = res.extend(Layout::new::<u64>())?.0.pad_to_align(),
            NumberType::F64 => res = res.extend(Layout::new::<f64>())?.0.pad_to_align(),
        }
        Ok(res)
    }

    fn alloc(type_: NumberType) -> *mut Header {
        unsafe {
            let ptr = alloc(Self::layout(type_).unwrap()) as *mut Header;
            (*ptr).type_ = type_;
            (*ptr).static_ = 0;
            (*ptr).short = 0;
            ptr
        }
    }

    fn dealloc(ptr: *mut Header) {
        unsafe {
            let layout = Self::layout((*ptr).type_).unwrap();
            dealloc(ptr as *mut u8, layout);
        }
    }

    pub fn zero() -> Self {
        // Safety: 0 is in the static range
        unsafe { Self::new_static(0) }
    }
    pub fn one() -> Self {
        // Safety: 1 is in the static range
        unsafe { Self::new_static(1) }
    }
    // Safety: Value must be in the range STATIC_LOWER..STATIC_UPPER
    unsafe fn new_static(value: i16) -> Self {
        INumber(IValue::new_ref(
            &STATIC_NUMBERS[(value - STATIC_LOWER) as usize],
            TypeTag::Number,
        ))
    }
    fn new_ptr(type_: NumberType) -> Self {
        unsafe {
            INumber(IValue::new_ptr(
                Self::alloc(type_) as *mut u8,
                TypeTag::Number,
            ))
        }
    }
    fn header(&self) -> &Header {
        unsafe { &*(self.0.ptr() as *const Header) }
    }

    fn header_mut(&mut self) -> &mut Header {
        unsafe { &mut *(self.0.ptr() as *mut Header) }
    }

    fn is_static(&self) -> bool {
        self.header().type_ == NumberType::Static
    }

    // Value must fit in an i24
    fn new_short(value: i32) -> Self {
        if value >= STATIC_LOWER as i32 && value < STATIC_UPPER as i32 {
            // Safety: We checked the value is in the static range
            unsafe { Self::new_static(value as i16) }
        } else {
            let lo_bits = value as u8;
            let hi_bits = (value >> 8) as i16;
            let mut res = Self::new_ptr(NumberType::I24);
            let hd = res.header_mut();
            hd.short = lo_bits;
            hd.static_ = hi_bits;
            res
        }
    }

    fn new_i64(value: i64) -> Self {
        if value as u64 & 0xFFFFFFFFFF000000 == 0 {
            Self::new_short(value as i32)
        } else {
            let mut res = Self::new_ptr(NumberType::I64);
            // Safety: We know this is an i64 because we just created it
            unsafe {
                *res.header_mut().as_i64_unchecked_mut() = value;
            }
            res
        }
    }

    fn new_u64(value: u64) -> Self {
        if value <= i64::MAX as u64 {
            Self::new_i64(value as i64)
        } else {
            let mut res = Self::new_ptr(NumberType::U64);
            // Safety: We know this is an i64 because we just created it
            unsafe {
                *res.header_mut().as_u64_unchecked_mut() = value;
            }
            res
        }
    }

    fn new_f64(value: f64) -> Self {
        let mut res = Self::new_ptr(NumberType::F64);
        // Safety: We know this is an i64 because we just created it
        unsafe {
            *res.header_mut().as_f64_unchecked_mut() = value;
        }
        res
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        let hd = self.header();
        // Safety: We only call methods appropriate for the matched type
        unsafe {
            match hd.type_ {
                NumberType::Static => self.0.raw_copy(),
                NumberType::I24 => Self::new_short(hd.as_i24_unchecked()).0,
                NumberType::I64 => Self::new_i64(*hd.as_i64_unchecked()).0,
                NumberType::U64 => Self::new_u64(*hd.as_u64_unchecked()).0,
                NumberType::F64 => Self::new_f64(*hd.as_f64_unchecked()).0,
            }
        }
    }
    pub(crate) fn drop_impl(&mut self) {
        if !self.is_static() {
            unsafe {
                Self::dealloc(self.header_mut() as *mut _);
                self.0.set_ref(&STATIC_NUMBERS[0]);
            }
        }
    }

    /// Converts this number to an i64 if it can be represented exactly
    pub fn to_i64(&self) -> Option<i64> {
        self.header().to_i64()
    }
    /// Converts this number to an f64 if it can be represented exactly
    pub fn to_u64(&self) -> Option<u64> {
        self.header().to_u64()
    }
    /// Converts this number to an f64 if it can be represented exactly
    pub fn to_f64(&self) -> Option<f64> {
        self.header().to_f64()
    }
    /// Converts this number to an f32 if it can be represented exactly
    pub fn to_f32(&self) -> Option<f32> {
        self.header().to_f32()
    }
    /// Converts this number to an i32 if it can be represented exactly
    pub fn to_i32(&self) -> Option<i32> {
        self.header().to_i64().and_then(|x| x.try_into().ok())
    }
    pub fn to_f64_lossy(&self) -> f64 {
        self.header().to_f64_lossy()
    }
    pub fn to_f32_lossy(&self) -> f32 {
        self.to_f64_lossy() as f32
    }

    /// This allows distinguishing between `1.0` and `1` in the original JSON.
    /// Numeric operations will otherwise treat these two values as equivalent.
    pub fn has_decimal_point(&self) -> bool {
        self.header().has_decimal_point()
    }
}

impl Hash for INumber {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let hd = self.header();
        if let Some(v) = hd.to_i64() {
            v.hash(state);
        } else if let Some(v) = hd.to_u64() {
            v.hash(state);
        } else if let Some(v) = hd.to_f64() {
            let bits = if v == 0.0 {
                0 // this accounts for +0.0 and -0.0
            } else {
                v.to_bits()
            };
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
        Self::new_u64(v as u64)
    }
}
impl From<u16> for INumber {
    fn from(v: u16) -> Self {
        Self::new_short(v as i32)
    }
}
impl From<u8> for INumber {
    fn from(v: u8) -> Self {
        // Safety: All u8s are in the static range
        unsafe { Self::new_static(v as i16) }
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
        Self::new_i64(v as i64)
    }
}
impl From<i16> for INumber {
    fn from(v: i16) -> Self {
        Self::new_short(v as i32)
    }
}
impl From<i8> for INumber {
    fn from(v: i8) -> Self {
        // Safety: All i8s are in the static range
        unsafe { Self::new_static(v as i16) }
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
            Ok(Self::new_f64(v as f64))
        } else {
            Err(())
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
            self.header().cmp(other.header())
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
        } else if let Some(v) = self.to_f64() {
            Debug::fmt(&v, f)
        } else {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[mockalloc::test]
    fn can_create() {
        let x = INumber::zero();
        let y: INumber = (0.0).try_into().unwrap();

        assert_eq!(x, y);
        assert!(!x.has_decimal_point());
        assert!(y.has_decimal_point());
        assert_eq!(x.to_i32(), Some(0));
        assert_eq!(y.to_i32(), Some(0));
        assert!(INumber::try_from(f32::INFINITY).is_err());
        assert!(INumber::try_from(f64::INFINITY).is_err());
        assert!(INumber::try_from(f32::NEG_INFINITY).is_err());
        assert!(INumber::try_from(f64::NEG_INFINITY).is_err());
        assert!(INumber::try_from(f32::NAN).is_err());
        assert!(INumber::try_from(f64::NAN).is_err());
    }

    #[mockalloc::test]
    fn can_store_various_numbers() {
        let x: INumber = 256.into();
        assert_eq!(x.to_i64(), Some(256));
        assert_eq!(x.to_u64(), Some(256));
        assert_eq!(x.to_f64(), Some(256.0));

        let x: INumber = 0x1000000.into();
        assert_eq!(x.to_i64(), Some(0x1000000));
        assert_eq!(x.to_u64(), Some(0x1000000));
        assert_eq!(x.to_f64(), Some(16777216.0));

        let x: INumber = i64::MIN.into();
        assert_eq!(x.to_i64(), Some(i64::MIN));
        assert_eq!(x.to_u64(), None);
        assert_eq!(x.to_f64(), Some(-9223372036854775808.0));

        let x: INumber = i64::MAX.into();
        assert_eq!(x.to_i64(), Some(i64::MAX));
        assert_eq!(x.to_u64(), Some(i64::MAX as u64));
        assert_eq!(x.to_f64(), None);

        let x: INumber = u64::MAX.into();
        assert_eq!(x.to_i64(), None);
        assert_eq!(x.to_u64(), Some(u64::MAX));
        assert_eq!(x.to_f64(), None);
    }

    #[mockalloc::test]
    fn can_compare_various_numbers() {
        assert!(INumber::from(1) < INumber::try_from(1.5).unwrap());
        assert!(INumber::from(2) > INumber::try_from(1.5).unwrap());
        assert!(INumber::from(-2) < INumber::try_from(1.5).unwrap());
        assert!(INumber::from(-2) < INumber::try_from(-1.5).unwrap());
        assert!(INumber::from(-2) == INumber::try_from(-2.0).unwrap());
        assert!(INumber::try_from(-1.5).unwrap() > INumber::from(-2));
        assert!(INumber::try_from(1e30).unwrap() > INumber::from(u64::MAX));
        assert!(INumber::try_from(1e30).unwrap() > INumber::from(i64::MAX));
        assert!(INumber::try_from(-1e30).unwrap() < INumber::from(i64::MIN));
        assert!(INumber::try_from(-1e30).unwrap() < INumber::from(i64::MIN));
        assert!(INumber::try_from(99999999000.0).unwrap() < INumber::from(99999999001u64));
    }
}
