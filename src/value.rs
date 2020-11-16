use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::convert::TryFrom;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::hint::unreachable_unchecked;
use std::mem;
use std::ops::{Deref, Index, IndexMut};
use std::ptr::NonNull;

use super::array::IArray;
use super::number::INumber;
use super::object::IObject;
use super::string::IString;

#[repr(transparent)]
pub struct IValue {
    ptr: NonNull<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Destructured {
    Null,
    Bool(bool),
    Number(INumber),
    String(IString),
    Array(IArray),
    Object(IObject),
}

impl Destructured {
    pub fn as_ref(&self) -> DestructuredRef {
        use DestructuredRef::*;
        match self {
            Self::Null => Null,
            Self::Bool(b) => Bool(*b),
            Self::Number(v) => Number(v),
            Self::String(v) => String(v),
            Self::Array(v) => Array(v),
            Self::Object(v) => Object(v),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DestructuredRef<'a> {
    Null,
    Bool(bool),
    Number(&'a INumber),
    String(&'a IString),
    Array(&'a IArray),
    Object(&'a IObject),
}

#[derive(Debug)]
pub enum DestructuredMut<'a> {
    Null,
    Bool(BoolMut<'a>),
    Number(&'a mut INumber),
    String(&'a mut IString),
    Array(&'a mut IArray),
    Object(&'a mut IObject),
}

#[derive(Debug)]
pub struct BoolMut<'a>(&'a mut IValue);

impl<'a> BoolMut<'a> {
    pub fn set(&mut self, value: bool) {
        *self.0 = value.into();
    }
    pub fn get(&self) -> bool {
        self.0.is_true()
    }
}

impl<'a> Deref for BoolMut<'a> {
    type Target = bool;
    fn deref(&self) -> &bool {
        if self.get() {
            &true
        } else {
            &false
        }
    }
}

pub(crate) const ALIGNMENT: usize = 4;

#[repr(usize)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypeTag {
    Number = 0,
    StringOrNull = 1,
    ArrayOrFalse = 2,
    ObjectOrTrue = 3,
}

impl From<usize> for TypeTag {
    fn from(other: usize) -> Self {
        // Safety: `% ALIGNMENT` can only return valid variants
        unsafe { mem::transmute(other % ALIGNMENT) }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueType {
    // Stored inline
    Null,
    Bool,

    // Stored behind pointer
    Number,
    String,
    Array,
    Object,
}

unsafe impl Send for IValue {}
unsafe impl Sync for IValue {}

impl IValue {
    // Safety: Tag must not be `Number`
    const unsafe fn new_inline(tag: TypeTag) -> Self {
        Self {
            ptr: NonNull::new_unchecked(tag as usize as *mut u8),
        }
    }
    // Safety: Pointer must be non-null and aligned to at least ALIGNMENT
    pub(crate) unsafe fn new_ptr(p: *mut u8, tag: TypeTag) -> Self {
        Self {
            ptr: NonNull::new_unchecked(p.offset(tag as isize)),
        }
    }
    // Safety: Reference must be aligned to at least ALIGNMENT
    pub(crate) unsafe fn new_ref<T>(r: &T, tag: TypeTag) -> Self {
        Self::new_ptr(r as *const _ as *mut u8, tag)
    }
    pub const NULL: Self = unsafe { Self::new_inline(TypeTag::StringOrNull) };
    pub const FALSE: Self = unsafe { Self::new_inline(TypeTag::ArrayOrFalse) };
    pub const TRUE: Self = unsafe { Self::new_inline(TypeTag::ObjectOrTrue) };

    pub(crate) fn ptr_usize(&self) -> usize {
        self.ptr.as_ptr() as usize
    }
    // Safety: Must only be called on non-inline types
    pub(crate) unsafe fn ptr(&self) -> *mut u8 {
        self.ptr
            .as_ptr()
            .wrapping_offset(-((self.ptr_usize() % ALIGNMENT) as isize))
    }
    // Safety: Pointer must be non-null and aligned to at least ALIGNMENT
    pub(crate) unsafe fn set_ptr(&mut self, ptr: *mut u8) {
        let tag = self.type_tag();
        self.ptr = NonNull::new_unchecked(ptr.offset(tag as isize));
    }
    // Safety: Reference must be aligned to at least ALIGNMENT
    pub(crate) unsafe fn set_ref<T>(&mut self, r: &T) {
        self.set_ptr(r as *const T as *mut u8)
    }
    pub(crate) unsafe fn raw_copy(&self) -> Self {
        Self { ptr: self.ptr }
    }
    pub(crate) fn raw_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
    pub(crate) fn raw_hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state)
    }
    fn is_ptr(&self) -> bool {
        self.ptr_usize() >= ALIGNMENT
    }
    fn type_tag(&self) -> TypeTag {
        self.ptr_usize().into()
    }

    pub fn type_(&self) -> ValueType {
        match (self.type_tag(), self.is_ptr()) {
            // Pointers
            (TypeTag::Number, true) => ValueType::Number,
            (TypeTag::StringOrNull, true) => ValueType::String,
            (TypeTag::ArrayOrFalse, true) => ValueType::Array,
            (TypeTag::ObjectOrTrue, true) => ValueType::Object,

            // Non-pointers
            (TypeTag::StringOrNull, false) => ValueType::Null,
            (TypeTag::ArrayOrFalse, false) => ValueType::Bool,
            (TypeTag::ObjectOrTrue, false) => ValueType::Bool,

            // Safety: due to invariants on IValue
            _ => unsafe { unreachable_unchecked() },
        }
    }

    pub fn destructure(self) -> Destructured {
        match self.type_() {
            ValueType::Null => Destructured::Null,
            ValueType::Bool => Destructured::Bool(self.is_true()),
            ValueType::Number => Destructured::Number(INumber(self)),
            ValueType::String => Destructured::String(IString(self)),
            ValueType::Array => Destructured::Array(IArray(self)),
            ValueType::Object => Destructured::Object(IObject(self)),
        }
    }

    pub fn destructure_ref(&self) -> DestructuredRef {
        // Safety: we check the type
        unsafe {
            match self.type_() {
                ValueType::Null => DestructuredRef::Null,
                ValueType::Bool => DestructuredRef::Bool(self.is_true()),
                ValueType::Number => DestructuredRef::Number(self.as_number_unchecked()),
                ValueType::String => DestructuredRef::String(self.as_string_unchecked()),
                ValueType::Array => DestructuredRef::Array(self.as_array_unchecked()),
                ValueType::Object => DestructuredRef::Object(self.as_object_unchecked()),
            }
        }
    }

    pub fn destructure_mut(&mut self) -> DestructuredMut {
        // Safety: we check the type
        unsafe {
            match self.type_() {
                ValueType::Null => DestructuredMut::Null,
                ValueType::Bool => DestructuredMut::Bool(BoolMut(self)),
                ValueType::Number => DestructuredMut::Number(self.as_number_unchecked_mut()),
                ValueType::String => DestructuredMut::String(self.as_string_unchecked_mut()),
                ValueType::Array => DestructuredMut::Array(self.as_array_unchecked_mut()),
                ValueType::Object => DestructuredMut::Object(self.as_object_unchecked_mut()),
            }
        }
    }

    pub fn get(&self, index: impl ValueIndex) -> Option<&IValue> {
        index.index_into(self)
    }
    pub fn get_mut(&mut self, index: impl ValueIndex) -> Option<&mut IValue> {
        index.index_into_mut(self)
    }
    pub fn remove(&mut self, index: impl ValueIndex) -> Option<IValue> {
        index.remove(self)
    }
    pub fn take(&mut self) -> IValue {
        mem::replace(self, IValue::NULL)
    }

    pub fn len(&self) -> Option<usize> {
        match self.type_() {
            // Safety: checked type
            ValueType::Array => Some(unsafe { self.as_array_unchecked().len() }),
            // Safety: checked type
            ValueType::Object => Some(unsafe { self.as_object_unchecked().len() }),
            _ => None,
        }
    }

    // # Null methods
    pub fn is_null(&self) -> bool {
        self.ptr == Self::NULL.ptr
    }

    // # Bool methods
    pub fn is_bool(&self) -> bool {
        self.ptr == Self::TRUE.ptr || self.ptr == Self::FALSE.ptr
    }

    pub fn is_true(&self) -> bool {
        self.ptr == Self::TRUE.ptr
    }

    pub fn is_false(&self) -> bool {
        self.ptr == Self::FALSE.ptr
    }

    pub fn to_bool(&self) -> Option<bool> {
        if self.is_bool() {
            Some(self.is_true())
        } else {
            None
        }
    }

    // # Number methods
    pub fn is_number(&self) -> bool {
        self.type_tag() == TypeTag::Number
    }

    // Safety: Must be a string
    unsafe fn as_number_unchecked(&self) -> &INumber {
        mem::transmute(self)
    }

    // Safety: Must be a string
    unsafe fn as_number_unchecked_mut(&mut self) -> &mut INumber {
        mem::transmute(self)
    }

    pub fn as_number(&self) -> Option<&INumber> {
        if self.is_number() {
            // Safety: INumber is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_number_unchecked() })
        } else {
            None
        }
    }

    pub fn as_number_mut(&mut self) -> Option<&mut INumber> {
        if self.is_number() {
            // Safety: INumber is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_number_unchecked_mut() })
        } else {
            None
        }
    }

    pub fn into_number(self) -> Result<INumber, IValue> {
        if self.is_number() {
            Ok(INumber(self))
        } else {
            Err(self)
        }
    }

    /// Converts this value to an i64 if it is a number that can be represented exactly
    pub fn to_i64(&self) -> Option<i64> {
        self.as_number()?.to_i64()
    }
    /// Converts this value to a u64 if it is a number that can be represented exactly
    pub fn to_u64(&self) -> Option<u64> {
        self.as_number()?.to_u64()
    }
    /// Converts this value to an f64 if it is a number that can be represented exactly
    pub fn to_f64(&self) -> Option<f64> {
        self.as_number()?.to_f64()
    }
    /// Converts this value to an f32 if it is a number that can be represented exactly
    pub fn to_f32(&self) -> Option<f32> {
        self.as_number()?.to_f32()
    }
    /// Converts this value to an i32 if it is a number that can be represented exactly
    pub fn to_i32(&self) -> Option<i32> {
        self.as_number()?.to_i32()
    }
    pub fn to_f64_lossy(&self) -> Option<f64> {
        Some(self.as_number()?.to_f64_lossy())
    }
    pub fn to_f32_lossy(&self) -> Option<f32> {
        Some(self.as_number()?.to_f32_lossy())
    }

    // # String methods
    pub fn is_string(&self) -> bool {
        self.type_tag() == TypeTag::StringOrNull && self.is_ptr()
    }

    // Safety: Must be a string
    unsafe fn as_string_unchecked(&self) -> &IString {
        mem::transmute(self)
    }

    // Safety: Must be a string
    unsafe fn as_string_unchecked_mut(&mut self) -> &mut IString {
        mem::transmute(self)
    }

    pub fn as_string(&self) -> Option<&IString> {
        if self.is_string() {
            // Safety: IString is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_string_unchecked() })
        } else {
            None
        }
    }

    pub fn as_string_mut(&mut self) -> Option<&mut IString> {
        if self.is_string() {
            // Safety: IString is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_string_unchecked_mut() })
        } else {
            None
        }
    }

    pub fn into_string(self) -> Result<IString, IValue> {
        if self.is_string() {
            Ok(IString(self))
        } else {
            Err(self)
        }
    }

    // # Array methods
    pub fn is_array(&self) -> bool {
        self.type_tag() == TypeTag::ArrayOrFalse && self.is_ptr()
    }

    // Safety: Must be an array
    unsafe fn as_array_unchecked(&self) -> &IArray {
        mem::transmute(self)
    }

    // Safety: Must be an array
    unsafe fn as_array_unchecked_mut(&mut self) -> &mut IArray {
        mem::transmute(self)
    }

    pub fn as_array(&self) -> Option<&IArray> {
        if self.is_array() {
            // Safety: IArray is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_array_unchecked() })
        } else {
            None
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut IArray> {
        if self.is_array() {
            // Safety: IArray is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_array_unchecked_mut() })
        } else {
            None
        }
    }

    pub fn into_array(self) -> Result<IArray, IValue> {
        if self.is_array() {
            Ok(IArray(self))
        } else {
            Err(self)
        }
    }

    // # Object methods
    pub fn is_object(&self) -> bool {
        self.type_tag() == TypeTag::ObjectOrTrue && self.is_ptr()
    }

    // Safety: Must be an array
    unsafe fn as_object_unchecked(&self) -> &IObject {
        mem::transmute(self)
    }

    // Safety: Must be an array
    unsafe fn as_object_unchecked_mut(&mut self) -> &mut IObject {
        mem::transmute(self)
    }

    pub fn as_object(&self) -> Option<&IObject> {
        if self.is_object() {
            // Safety: IObject is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_object_unchecked() })
        } else {
            None
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut IObject> {
        if self.is_object() {
            // Safety: IObject is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_object_unchecked_mut() })
        } else {
            None
        }
    }

    pub fn into_object(self) -> Result<IObject, IValue> {
        if self.is_number() {
            Ok(IObject(self))
        } else {
            Err(self)
        }
    }
}

impl Clone for IValue {
    fn clone(&self) -> Self {
        match self.type_() {
            // Inline types can be trivially copied
            ValueType::Null | ValueType::Bool => Self { ptr: self.ptr },
            // Safety: We checked the type
            ValueType::Array => unsafe { self.as_array_unchecked() }.clone_impl(),
            ValueType::Object => unsafe { self.as_object_unchecked() }.clone_impl(),
            ValueType::String => unsafe { self.as_string_unchecked() }.clone_impl(),
            ValueType::Number => unsafe { self.as_number_unchecked() }.clone_impl(),
        }
    }
}

impl Drop for IValue {
    fn drop(&mut self) {
        match self.type_() {
            // Inline types can be trivially dropped
            ValueType::Null | ValueType::Bool => {}
            // Safety: We checked the type
            ValueType::Array => unsafe { self.as_array_unchecked_mut() }.drop_impl(),
            ValueType::Object => unsafe { self.as_object_unchecked_mut() }.drop_impl(),
            ValueType::String => unsafe { self.as_string_unchecked_mut() }.drop_impl(),
            ValueType::Number => unsafe { self.as_number_unchecked_mut() }.drop_impl(),
        }
    }
}

impl Hash for IValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.type_() {
            // Inline and interned types can be trivially hashed
            ValueType::Null | ValueType::Bool | ValueType::String => self.ptr.hash(state),
            // Safety: We checked the type
            ValueType::Array => unsafe { self.as_array_unchecked() }.hash(state),
            // Safety: We checked the type
            ValueType::Object => unsafe { self.as_object_unchecked() }.hash(state),
            // Safety: We checked the type
            ValueType::Number => unsafe { self.as_number_unchecked() }.hash(state),
        }
    }
}

impl PartialEq for IValue {
    fn eq(&self, other: &Self) -> bool {
        let (t1, t2) = (self.type_(), other.type_());
        if t1 == t2 {
            // Safety: Only methods for the appropriate type are called
            unsafe {
                match t1 {
                    // Inline and interned types can be trivially compared
                    ValueType::Null | ValueType::Bool | ValueType::String => self.ptr == other.ptr,
                    ValueType::Number => self.as_number_unchecked() == other.as_number_unchecked(),
                    ValueType::Array => self.as_array_unchecked() == other.as_array_unchecked(),
                    ValueType::Object => self.as_object_unchecked() == other.as_object_unchecked(),
                }
            }
        } else {
            false
        }
    }
}

impl Eq for IValue {}
impl PartialOrd for IValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (t1, t2) = (self.type_(), other.type_());
        if t1 == t2 {
            // Safety: Only methods for the appropriate type are called
            unsafe {
                match t1 {
                    // Inline and interned types can be trivially compared
                    ValueType::Null => Some(Ordering::Equal),
                    ValueType::Bool => self.is_true().partial_cmp(&other.is_true()),
                    ValueType::String => self
                        .as_string_unchecked()
                        .partial_cmp(other.as_string_unchecked()),
                    ValueType::Number => self
                        .as_number_unchecked()
                        .partial_cmp(other.as_number_unchecked()),
                    ValueType::Array => self
                        .as_array_unchecked()
                        .partial_cmp(other.as_array_unchecked()),
                    ValueType::Object => return None,
                }
            }
        } else {
            t1.partial_cmp(&t2)
        }
    }
}

mod private {
    #[doc(hidden)]
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for &str {}
    impl Sealed for &super::IString {}
    impl<T: Sealed> Sealed for &T {}
}

pub trait ValueIndex: private::Sealed + Copy {
    #[doc(hidden)]
    fn index_into(self, v: &IValue) -> Option<&IValue>;

    #[doc(hidden)]
    fn index_into_mut(self, v: &mut IValue) -> Option<&mut IValue>;

    #[doc(hidden)]
    fn index_or_insert(self, v: &mut IValue) -> &mut IValue;

    #[doc(hidden)]
    fn remove(self, v: &mut IValue) -> Option<IValue>;
}

impl ValueIndex for usize {
    fn index_into(self, v: &IValue) -> Option<&IValue> {
        v.as_array().unwrap().get(self)
    }

    fn index_into_mut(self, v: &mut IValue) -> Option<&mut IValue> {
        v.as_array_mut().unwrap().get_mut(self)
    }

    fn index_or_insert(self, v: &mut IValue) -> &mut IValue {
        self.index_into_mut(v).unwrap()
    }

    fn remove(self, v: &mut IValue) -> Option<IValue> {
        v.as_array_mut().unwrap().remove(self)
    }
}

impl ValueIndex for &str {
    fn index_into(self, v: &IValue) -> Option<&IValue> {
        v.as_object().unwrap().get(&IString::intern(self))
    }

    fn index_into_mut(self, v: &mut IValue) -> Option<&mut IValue> {
        v.as_object_mut().unwrap().get_mut(&IString::intern(self))
    }

    fn index_or_insert(self, v: &mut IValue) -> &mut IValue {
        &mut v.as_object_mut().unwrap()[self]
    }

    fn remove(self, v: &mut IValue) -> Option<IValue> {
        v.as_object_mut().unwrap().remove(self)
    }
}

impl ValueIndex for &IString {
    fn index_into(self, v: &IValue) -> Option<&IValue> {
        v.as_object().unwrap().get(self)
    }

    fn index_into_mut(self, v: &mut IValue) -> Option<&mut IValue> {
        v.as_object_mut().unwrap().get_mut(self)
    }

    fn index_or_insert(self, v: &mut IValue) -> &mut IValue {
        &mut v.as_object_mut().unwrap()[self]
    }

    fn remove(self, v: &mut IValue) -> Option<IValue> {
        v.as_object_mut().unwrap().remove(self)
    }
}

impl<T: ValueIndex> ValueIndex for &T {
    fn index_into(self, v: &IValue) -> Option<&IValue> {
        (*self).index_into(v)
    }

    fn index_into_mut(self, v: &mut IValue) -> Option<&mut IValue> {
        (*self).index_into_mut(v)
    }

    fn index_or_insert(self, v: &mut IValue) -> &mut IValue {
        (*self).index_or_insert(v)
    }

    fn remove(self, v: &mut IValue) -> Option<IValue> {
        (*self).remove(v)
    }
}

impl<I: ValueIndex> Index<I> for IValue {
    type Output = IValue;

    #[inline]
    fn index(&self, index: I) -> &IValue {
        index.index_into(self).unwrap()
    }
}

impl<I: ValueIndex> IndexMut<I> for IValue {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut IValue {
        index.index_or_insert(self)
    }
}

impl Debug for IValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        unsafe {
            match self.type_() {
                // Inline and interned types can be trivially hashed
                ValueType::Null => f.write_str("null"),
                ValueType::Bool => Debug::fmt(&self.is_true(), f),
                // Safety: We checked the type
                ValueType::String => Debug::fmt(self.as_string_unchecked(), f),
                // Safety: We checked the type
                ValueType::Array => Debug::fmt(self.as_array_unchecked(), f),
                // Safety: We checked the type
                ValueType::Object => Debug::fmt(self.as_object_unchecked(), f),
                // Safety: We checked the type
                ValueType::Number => Debug::fmt(self.as_number_unchecked(), f),
            }
        }
    }
}

impl<T: Into<IValue>> From<Option<T>> for IValue {
    fn from(other: Option<T>) -> Self {
        if let Some(v) = other {
            v.into()
        } else {
            Self::NULL
        }
    }
}

impl From<bool> for IValue {
    fn from(other: bool) -> Self {
        if other {
            Self::TRUE
        } else {
            Self::FALSE
        }
    }
}

typed_conversions! {
    INumber: i8, u8, i16, u16, i32, u32, i64, u64, isize, usize;
    IString: String, &str, &mut str;
    IArray:
        Vec<T> where (T: Into<IValue>),
        &[T] where (T: Into<IValue> + Clone);
    IObject:
        HashMap<K, V> where (K: Into<IString>, V: Into<IValue>),
        BTreeMap<K, V> where (K: Into<IString>, V: Into<IValue>);
}

impl From<f32> for IValue {
    fn from(v: f32) -> Self {
        INumber::try_from(v).map(Into::into).unwrap_or(IValue::NULL)
    }
}

impl From<f64> for IValue {
    fn from(v: f64) -> Self {
        INumber::try_from(v).map(Into::into).unwrap_or(IValue::NULL)
    }
}
