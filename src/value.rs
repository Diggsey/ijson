use std::cmp::Ordering;
use std::hash::Hash;
use std::hint::unreachable_unchecked;
use std::mem;
use std::ptr::NonNull;

use super::array::IArray;
use super::number::INumber;
use super::string::IString;

#[repr(transparent)]
pub struct IValue {
    ptr: NonNull<u8>,
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
    pub(crate) const unsafe fn new_ptr(p: *mut u8, tag: TypeTag) -> Self {
        Self {
            ptr: NonNull::new_unchecked(p.offset(tag as isize)),
        }
    }
    // Safety: Reference must be aligned to at least ALIGNMENT
    pub(crate) const unsafe fn new_ref<T>(r: &T, tag: TypeTag) -> Self {
        Self::new_ptr(r as *const _ as *mut u8, tag)
    }
    pub const NULL: Self = unsafe { Self::new_inline(TypeTag::StringOrNull) };
    pub const FALSE: Self = unsafe { Self::new_inline(TypeTag::ArrayOrFalse) };
    pub const TRUE: Self = unsafe { Self::new_inline(TypeTag::ObjectOrTrue) };

    fn ptr_usize(&self) -> usize {
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

    pub fn is_null(&self) -> bool {
        self.ptr == Self::NULL.ptr
    }

    pub fn is_bool(&self) -> bool {
        self.ptr == Self::TRUE.ptr || self.ptr == Self::FALSE.ptr
    }

    pub fn is_number(&self) -> bool {
        self.type_tag() == TypeTag::Number
    }

    pub fn is_string(&self) -> bool {
        self.type_tag() == TypeTag::StringOrNull && self.is_ptr()
    }

    pub fn is_array(&self) -> bool {
        self.type_tag() == TypeTag::ArrayOrFalse && self.is_ptr()
    }

    pub fn is_object(&self) -> bool {
        self.type_tag() == TypeTag::ObjectOrTrue && self.is_ptr()
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
}

impl Clone for IValue {
    fn clone(&self) -> Self {
        match self.type_() {
            // Inline types can be trivially copied
            ValueType::Null | ValueType::Bool => Self { ptr: self.ptr },
            // Safety: We checked the type
            ValueType::Array => unsafe { self.as_array_unchecked() }.clone_impl(),
            ValueType::String => unsafe { self.as_string_unchecked() }.clone_impl(),
            ValueType::Number => unsafe { self.as_number_unchecked() }.clone_impl(),
            _ => unimplemented!(),
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
            ValueType::String => unsafe { self.as_string_unchecked_mut() }.drop_impl(),
            ValueType::Number => unsafe { self.as_number_unchecked_mut() }.drop_impl(),
            _ => unimplemented!(),
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
            ValueType::Number => unsafe { self.as_number_unchecked() }.hash(state),
            _ => unimplemented!(),
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
                    ValueType::Object => unimplemented!(),
                }
            }
        } else {
            false
        }
    }
}

impl Eq for IValue {}
impl Ord for IValue {
    fn cmp(&self, other: &Self) -> Ordering {
        let (t1, t2) = (self.type_(), other.type_());
        if t1 == t2 {
            // Safety: Only methods for the appropriate type are called
            unsafe {
                match t1 {
                    // Inline and interned types can be trivially compared
                    ValueType::Null => Ordering::Equal,
                    ValueType::Bool => self.is_true().cmp(&other.is_true()),
                    ValueType::String => {
                        self.as_string_unchecked().cmp(other.as_string_unchecked())
                    }
                    ValueType::Number => {
                        self.as_number_unchecked().cmp(other.as_number_unchecked())
                    }
                    ValueType::Array => self.as_array_unchecked().cmp(other.as_array_unchecked()),
                    ValueType::Object => unimplemented!(),
                }
            }
        } else {
            t1.cmp(&t2)
        }
    }
}
impl PartialOrd for IValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
