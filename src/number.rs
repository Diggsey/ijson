use std::alloc::{alloc, dealloc, Layout, LayoutErr};
use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
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
    static_: i8,
    short: u16,
}

fn can_represent_as_f64(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 11
}

fn can_represent_as_f32(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 40
}

fn cmp_i64_to_f64(a: i64, b: f64) -> Ordering {
    if a < 0 {
        cmp_u64_to_f64((-a) as u64, -b).reverse()
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
        ((self.static_ as i32) << 16) | (self.short as i32)
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
                        can_represent_as_f64((-v) as u64)
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
                        can_represent_as_f32((-v) as u64)
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
    fn is_integer(&self) -> bool {
        match self.type_ {
            NumberType::Static | NumberType::I24 | NumberType::I64 | NumberType::U64 => true,
            NumberType::F64 => false,
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
    ($($v:expr)*) => {
        [$(Header {
            type_: NumberType::Static,
            static_: ($v as u8) as i8,
            short: 0,
        }),*]
    };
}

static STATIC_NUMBERS: [Header; 256] = define_static_numbers!(
    0x00 0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0a 0x0b 0x0c 0x0d 0x0e 0x0f
    0x10 0x11 0x12 0x13 0x14 0x15 0x16 0x17 0x18 0x19 0x1a 0x1b 0x1c 0x1d 0x1e 0x1f
    0x20 0x21 0x22 0x23 0x24 0x25 0x26 0x27 0x28 0x29 0x2a 0x2b 0x2c 0x2d 0x2e 0x2f
    0x30 0x31 0x32 0x33 0x34 0x35 0x36 0x37 0x38 0x39 0x3a 0x3b 0x3c 0x3d 0x3e 0x3f
    0x40 0x41 0x42 0x43 0x44 0x45 0x46 0x47 0x48 0x49 0x4a 0x4b 0x4c 0x4d 0x4e 0x4f
    0x50 0x51 0x52 0x53 0x54 0x55 0x56 0x57 0x58 0x59 0x5a 0x5b 0x5c 0x5d 0x5e 0x5f
    0x60 0x61 0x62 0x63 0x64 0x65 0x66 0x67 0x68 0x69 0x6a 0x6b 0x6c 0x6d 0x6e 0x6f
    0x70 0x71 0x72 0x73 0x74 0x75 0x76 0x77 0x78 0x79 0x7a 0x7b 0x7c 0x7d 0x7e 0x7f
    0x80 0x81 0x82 0x83 0x84 0x85 0x86 0x87 0x88 0x89 0x8a 0x8b 0x8c 0x8d 0x8e 0x8f
    0x90 0x91 0x92 0x93 0x94 0x95 0x96 0x97 0x98 0x99 0x9a 0x9b 0x9c 0x9d 0x9e 0x9f
    0xa0 0xa1 0xa2 0xa3 0xa4 0xa5 0xa6 0xa7 0xa8 0xa9 0xaa 0xab 0xac 0xad 0xae 0xaf
    0xb0 0xb1 0xb2 0xb3 0xb4 0xb5 0xb6 0xb7 0xb8 0xb9 0xba 0xbb 0xbc 0xbd 0xbe 0xbf
    0xc0 0xc1 0xc2 0xc3 0xc4 0xc5 0xc6 0xc7 0xc8 0xc9 0xca 0xcb 0xcc 0xcd 0xce 0xcf
    0xd0 0xd1 0xd2 0xd3 0xd4 0xd5 0xd6 0xd7 0xd8 0xd9 0xda 0xdb 0xdc 0xdd 0xde 0xdf
    0xe0 0xe1 0xe2 0xe3 0xe4 0xe5 0xe6 0xe7 0xe8 0xe9 0xea 0xeb 0xec 0xed 0xee 0xef
    0xf0 0xf1 0xf2 0xf3 0xf4 0xf5 0xf6 0xf7 0xf8 0xf9 0xfa 0xfb 0xfc 0xfd 0xfe 0xff
);

#[repr(transparent)]
#[derive(Clone)]
pub struct INumber(IValue);

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

    pub fn new() -> Self {
        Self::new_static(0)
    }
    fn new_static(value: i8) -> Self {
        unsafe {
            INumber(IValue::new_ref(
                &STATIC_NUMBERS[value as u8 as usize],
                TypeTag::Number,
            ))
        }
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
        if value >= i8::MIN as i32 && value <= i8::MAX as i32 {
            Self::new_static(value as i8)
        } else {
            let lo_bits = value as u32 as u16;
            let hi_bits = (value >> 16) as i8;
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

    /// This allows distinguishing between `1.0` and `1` in the original JSON.
    /// Numeric operations will otherwise treat these two values as equivalent.
    pub fn is_integer(&self) -> bool {
        self.header().is_integer()
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
        Self::new_short(v as i32)
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
        Self::new_static(v)
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
        self.header().cmp(other.header())
    }
}
impl PartialOrd for INumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl AsRef<IValue> for INumber {
    fn as_ref(&self) -> &IValue {
        &self.0
    }
}
