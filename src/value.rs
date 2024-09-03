use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::convert::TryFrom;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::hint::unreachable_unchecked;
use std::mem;
use std::ops::{Deref, Index, IndexMut};
use std::ptr::NonNull;

use crate::{Defrag, DefragAllocator};

use super::array::IArray;
use super::number::INumber;
use super::object::IObject;

#[cfg(feature = "thread_safe")]
use super::string::IString;
#[cfg(not(feature = "thread_safe"))]
use super::unsafe_string::IString;

/// Stores an arbitrary JSON value.
///
/// Compared to [`serde_json::Value`] this type is a struct rather than an enum, as
/// this is necessary to achieve the important size reductions. This means that
/// you cannot directly `match` on an `IValue` to determine its type.
///
/// Instead, an `IValue` offers several ways to get at the inner type:
///
/// - Destructuring using `IValue::destructure[{_ref,_mut}]()`
///
///   These methods return wrapper enums which you _can_ directly match on, so
///   these methods are the most direct replacement for matching on a `Value`.
///
/// - Borrowing using `IValue::as_{array,object,string,number}[_mut]()`
///
///   These methods return an `Option` of the corresponding reference if the
///   type matches the one expected. These methods exist for the variants
///   which are not `Copy`.
///
/// - Converting using `IValue::into_{array,object,string,number}()`
///
///   These methods return a `Result` of the corresponding type (or the
///   original `IValue` if the type is not the one expected). These methods
///   also exist for the variants which are not `Copy`.
///
/// - Getting using `IValue::to_{bool,{i,u,f}{32,64}}[_lossy]}()`
///
///   These methods return an `Option` of the corresponding type. These
///   methods exist for types where the return value would be `Copy`.
///
/// You can also check the type of the inner value without specifically
/// accessing it using one of these methods:
///
/// - Checking using `IValue::is_{null,bool,number,string,array,object,true,false}()`
///
///   These methods exist for all types.
///
/// - Getting the type with [`IValue::type_`]
///
///   This method returns the [`ValueType`] enum, which has a variant for each of the
///   six JSON types.
#[repr(transparent)]
pub struct IValue {
    ptr: NonNull<u8>,
}

/// Enum returned by [`IValue::destructure`] to allow matching on the type of
/// an owned [`IValue`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Destructured {
    /// Null.
    Null,
    /// Boolean.
    Bool(bool),
    /// Number.
    Number(INumber),
    /// String.
    String(IString),
    /// Array.
    Array(IArray),
    /// Object.
    Object(IObject),
}

impl Destructured {
    /// Convert to the borrowed form of thie enum.
    #[must_use]
    pub fn as_ref(&self) -> DestructuredRef {
        use DestructuredRef::{Array, Bool, Null, Number, Object, String};
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

/// Enum returned by [`IValue::destructure_ref`] to allow matching on the type of
/// a reference to an [`IValue`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DestructuredRef<'a> {
    /// Null.
    Null,
    /// Boolean.
    /// [`IValue`]s do not directly contain booleans, so the value is returned
    /// directly instead of as a reference.
    Bool(bool),
    /// Number.
    Number(&'a INumber),
    /// String.
    String(&'a IString),
    /// Array.
    Array(&'a IArray),
    /// Object.
    Object(&'a IObject),
}

/// Enum returned by [`IValue::destructure_mut`] to allow matching on the type of
/// a mutable reference to an [`IValue`].
#[derive(Debug)]
pub enum DestructuredMut<'a> {
    /// Null.
    Null,
    /// Boolean.
    /// [`IValue`]s do not directly contain booleans, so this variant contains
    /// a proxy type which allows getting and setting the original [`IValue`]
    /// as a `bool`.
    Bool(BoolMut<'a>),
    /// Number.
    Number(&'a mut INumber),
    /// String.
    String(&'a mut IString),
    /// Array.
    Array(&'a mut IArray),
    /// Object.
    Object(&'a mut IObject),
}

/// A proxy type which imitates a `&mut bool`.
#[derive(Debug)]
pub struct BoolMut<'a>(&'a mut IValue);

impl<'a> BoolMut<'a> {
    /// Set the [`IValue`] referenced by this proxy type to either
    /// `true` or `false`.
    pub fn set(&mut self, value: bool) {
        *self.0 = value.into();
    }
    /// Get the boolean value stored in the [`IValue`] from which
    /// this proxy was obtained.
    #[must_use]
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

/// Enum which distinguishes the six JSON types.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueType {
    // Stored inline
    /// Null.
    Null,
    /// Boolean.
    Bool,

    // Stored behind pointer
    /// Number.
    Number,
    /// String.
    String,
    /// Array.
    Array,
    /// Object.
    Object,
}

unsafe impl Send for IValue {}
unsafe impl Sync for IValue {}

impl<A: DefragAllocator> Defrag<A> for IValue {
    fn defrag(self, defrag_allocator: &mut A) -> Self {
        match self.destructure() {
            Destructured::Null => IValue::NULL,
            Destructured::Bool(val) => {
                if val {
                    IValue::TRUE
                } else {
                    IValue::FALSE
                }
            }
            Destructured::Array(array) => array.defrag(defrag_allocator).0,
            Destructured::Object(obj) => obj.defrag(defrag_allocator).0,
            Destructured::String(s) => s.defrag(defrag_allocator).0,
            Destructured::Number(n) => n.defrag(defrag_allocator).0,
        }
    }
}

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
            ptr: NonNull::new_unchecked(p.add(tag as usize)),
        }
    }
    // Safety: Reference must be aligned to at least ALIGNMENT
    pub(crate) unsafe fn new_ref<T>(r: &T, tag: TypeTag) -> Self {
        Self::new_ptr(r as *const _ as *mut u8, tag)
    }

    /// JSON `null`.
    pub const NULL: Self = unsafe { Self::new_inline(TypeTag::StringOrNull) };
    /// JSON `false`.
    pub const FALSE: Self = unsafe { Self::new_inline(TypeTag::ArrayOrFalse) };
    /// JSON `true`.
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
        self.ptr = NonNull::new_unchecked(ptr.add(tag as usize));
    }
    // Safety: Reference must be aligned to at least ALIGNMENT
    pub(crate) unsafe fn set_ref<T>(&mut self, r: &T) {
        self.set_ptr(r as *const T as *mut u8);
    }
    pub(crate) unsafe fn raw_copy(&self) -> Self {
        Self { ptr: self.ptr }
    }
    pub(crate) fn raw_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
    fn is_ptr(&self) -> bool {
        self.ptr_usize() >= ALIGNMENT
    }
    fn type_tag(&self) -> TypeTag {
        self.ptr_usize().into()
    }

    /// Returns the type of this value.
    #[must_use]
    pub fn type_(&self) -> ValueType {
        match (self.type_tag(), self.is_ptr()) {
            // Pointers
            (TypeTag::Number, true) => ValueType::Number,
            (TypeTag::StringOrNull, true) => ValueType::String,
            (TypeTag::ArrayOrFalse, true) => ValueType::Array,
            (TypeTag::ObjectOrTrue, true) => ValueType::Object,

            // Non-pointers
            (TypeTag::StringOrNull, false) => ValueType::Null,
            (TypeTag::ArrayOrFalse, false) | (TypeTag::ObjectOrTrue, false) => ValueType::Bool,

            // Safety: due to invariants on IValue
            _ => unsafe { unreachable_unchecked() },
        }
    }

    /// Destructures this value into an enum which can be `match`ed on.
    #[must_use]
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

    /// Destructures a reference to this value into an enum which can be `match`ed on.
    #[must_use]
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

    /// Destructures a mutable reference to this value into an enum which can be `match`ed on.
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

    /// Indexes into this value with a number or string.
    /// Panics if the value is not an array or object.
    /// Panics if attempting to index an array with a string.
    /// Panics if attempting to index an object with a number.
    /// Returns `None` if the index type is correct, but there is
    /// no value at this index.
    pub fn get(&self, index: impl ValueIndex) -> Option<&IValue> {
        index.index_into(self)
    }

    /// Mutably indexes into this value with a number or string.
    /// Panics if the value is not an array or object.
    /// Panics if attempting to index an array with a string.
    /// Panics if attempting to index an object with a number.
    /// Returns `None` if the index type is correct, but there is
    /// no value at this index.
    pub fn get_mut(&mut self, index: impl ValueIndex) -> Option<&mut IValue> {
        index.index_into_mut(self)
    }

    /// Removes a value at the specified numberic or string index.
    /// Panics if this is not an array or object.
    /// Panics if attempting to index an array with a string.
    /// Panics if attempting to index an object with a number.
    /// Returns `None` if the index type is correct, but there is
    /// no value at this index.
    pub fn remove(&mut self, index: impl ValueIndex) -> Option<IValue> {
        index.remove(self)
    }

    /// Takes this value, replacing it with [`IValue::NULL`].
    pub fn take(&mut self) -> IValue {
        mem::replace(self, IValue::NULL)
    }

    /// Returns the length of this value if it is an array or object.
    /// Returns `None` for other types.
    #[must_use]
    pub fn len(&self) -> Option<usize> {
        match self.type_() {
            // Safety: checked type
            ValueType::Array => Some(unsafe { self.as_array_unchecked().len() }),
            // Safety: checked type
            ValueType::Object => Some(unsafe { self.as_object_unchecked().len() }),
            _ => None,
        }
    }

    /// Returns whether this value is empty if it is an array or object.
    /// Returns `None` for other types.
    #[must_use]
    pub fn is_empty(&self) -> Option<bool> {
        match self.type_() {
            // Safety: checked type
            ValueType::Array => Some(unsafe { self.as_array_unchecked().is_empty() }),
            // Safety: checked type
            ValueType::Object => Some(unsafe { self.as_object_unchecked().is_empty() }),
            _ => None,
        }
    }

    // # Null methods
    /// Returns `true` if this is the `null` value.
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.ptr == Self::NULL.ptr
    }

    // # Bool methods
    /// Returns `true` if this is a boolean.
    #[must_use]
    pub fn is_bool(&self) -> bool {
        self.ptr == Self::TRUE.ptr || self.ptr == Self::FALSE.ptr
    }

    /// Returns `true` if this is the `true` value.
    #[must_use]
    pub fn is_true(&self) -> bool {
        self.ptr == Self::TRUE.ptr
    }

    /// Returns `true` if this is the `false` value.
    #[must_use]
    pub fn is_false(&self) -> bool {
        self.ptr == Self::FALSE.ptr
    }

    /// Converts this value to a `bool`.
    /// Returns `None` if it's not a boolean.
    #[must_use]
    pub fn to_bool(&self) -> Option<bool> {
        if self.is_bool() {
            Some(self.is_true())
        } else {
            None
        }
    }

    // # Number methods
    /// Returns `true` if this is a number.
    #[must_use]
    pub fn is_number(&self) -> bool {
        self.type_tag() == TypeTag::Number
    }

    unsafe fn unchecked_cast_ref<T>(&self) -> &T {
        &*(self as *const Self).cast::<T>()
    }

    unsafe fn unchecked_cast_mut<T>(&mut self) -> &mut T {
        &mut *(self as *mut Self).cast::<T>()
    }

    // Safety: Must be a string
    unsafe fn as_number_unchecked(&self) -> &INumber {
        self.unchecked_cast_ref()
    }

    // Safety: Must be a string
    unsafe fn as_number_unchecked_mut(&mut self) -> &mut INumber {
        self.unchecked_cast_mut()
    }

    /// Gets a reference to this value as an [`INumber`].
    /// Returns `None` if it's not a number.
    #[must_use]
    pub fn as_number(&self) -> Option<&INumber> {
        if self.is_number() {
            // Safety: INumber is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_number_unchecked() })
        } else {
            None
        }
    }

    /// Gets a mutable reference to this value as an [`INumber`].
    /// Returns `None` if it's not a number.
    pub fn as_number_mut(&mut self) -> Option<&mut INumber> {
        if self.is_number() {
            // Safety: INumber is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_number_unchecked_mut() })
        } else {
            None
        }
    }

    /// Converts this value to an [`INumber`].
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if it's not a number.
    pub fn into_number(self) -> Result<INumber, IValue> {
        if self.is_number() {
            Ok(INumber(self))
        } else {
            Err(self)
        }
    }

    /// Converts this value to an i64 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        self.as_number()?.to_i64()
    }
    /// Converts this value to a u64 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        self.as_number()?.to_u64()
    }
    /// Converts this value to an f64 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        self.as_number()?.to_f64()
    }
    /// Converts this value to an f32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        self.as_number()?.to_f32()
    }
    /// Converts this value to an i32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_i32(&self) -> Option<i32> {
        self.as_number()?.to_i32()
    }
    /// Converts this value to a u32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        self.as_number()?.to_u32()
    }
    /// Converts this value to an isize if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_isize(&self) -> Option<isize> {
        self.as_number()?.to_isize()
    }
    /// Converts this value to a usize if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        self.as_number()?.to_usize()
    }
    /// Converts this value to an f64 if it is a number, potentially losing precision
    /// in the process.
    #[must_use]
    pub fn to_f64_lossy(&self) -> Option<f64> {
        Some(self.as_number()?.to_f64_lossy())
    }
    /// Converts this value to an f32 if it is a number, potentially losing precision
    /// in the process.
    #[must_use]
    pub fn to_f32_lossy(&self) -> Option<f32> {
        Some(self.as_number()?.to_f32_lossy())
    }

    // # String methods
    /// Returns `true` if this is a string.
    #[must_use]
    pub fn is_string(&self) -> bool {
        self.type_tag() == TypeTag::StringOrNull && self.is_ptr()
    }

    // Safety: Must be a string
    unsafe fn as_string_unchecked(&self) -> &IString {
        self.unchecked_cast_ref()
    }

    // Safety: Must be a string
    unsafe fn as_string_unchecked_mut(&mut self) -> &mut IString {
        self.unchecked_cast_mut()
    }

    /// Gets a reference to this value as an [`IString`].
    /// Returns `None` if it's not a string.
    #[must_use]
    pub fn as_string(&self) -> Option<&IString> {
        if self.is_string() {
            // Safety: IString is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_string_unchecked() })
        } else {
            None
        }
    }

    /// Gets a mutable reference to this value as an [`IString`].
    /// Returns `None` if it's not a string.
    pub fn as_string_mut(&mut self) -> Option<&mut IString> {
        if self.is_string() {
            // Safety: IString is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_string_unchecked_mut() })
        } else {
            None
        }
    }

    /// Converts this value to an [`IString`].
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if it's not a string.
    pub fn into_string(self) -> Result<IString, IValue> {
        if self.is_string() {
            Ok(IString(self))
        } else {
            Err(self)
        }
    }

    // # Array methods
    /// Returns `true` if this is an array.
    #[must_use]
    pub fn is_array(&self) -> bool {
        self.type_tag() == TypeTag::ArrayOrFalse && self.is_ptr()
    }

    // Safety: Must be an array
    unsafe fn as_array_unchecked(&self) -> &IArray {
        self.unchecked_cast_ref()
    }

    // Safety: Must be an array
    unsafe fn as_array_unchecked_mut(&mut self) -> &mut IArray {
        self.unchecked_cast_mut()
    }

    /// Gets a reference to this value as an [`IArray`].
    /// Returns `None` if it's not an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&IArray> {
        if self.is_array() {
            // Safety: IArray is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_array_unchecked() })
        } else {
            None
        }
    }

    /// Gets a mutable reference to this value as an [`IArray`].
    /// Returns `None` if it's not an array.
    pub fn as_array_mut(&mut self) -> Option<&mut IArray> {
        if self.is_array() {
            // Safety: IArray is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_array_unchecked_mut() })
        } else {
            None
        }
    }

    /// Converts this value to an [`IArray`].
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if it's not an array.
    pub fn into_array(self) -> Result<IArray, IValue> {
        if self.is_array() {
            Ok(IArray(self))
        } else {
            Err(self)
        }
    }

    // # Object methods
    /// Returns `true` if this is an object.
    #[must_use]
    pub fn is_object(&self) -> bool {
        self.type_tag() == TypeTag::ObjectOrTrue && self.is_ptr()
    }

    // Safety: Must be an array
    unsafe fn as_object_unchecked(&self) -> &IObject {
        self.unchecked_cast_ref()
    }

    // Safety: Must be an array
    unsafe fn as_object_unchecked_mut(&mut self) -> &mut IObject {
        self.unchecked_cast_mut()
    }

    /// Gets a reference to this value as an [`IObject`].
    /// Returns `None` if it's not an object.
    #[must_use]
    pub fn as_object(&self) -> Option<&IObject> {
        if self.is_object() {
            // Safety: IObject is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_object_unchecked() })
        } else {
            None
        }
    }

    /// Gets a mutable reference to this value as an [`IObject`].
    /// Returns `None` if it's not an object.
    pub fn as_object_mut(&mut self) -> Option<&mut IObject> {
        if self.is_object() {
            // Safety: IObject is a `#[repr(transparent)]` wrapper around IValue
            Some(unsafe { self.as_object_unchecked_mut() })
        } else {
            None
        }
    }

    /// Converts this value to an [`IObject`].
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if it's not an object.
    pub fn into_object(self) -> Result<IObject, IValue> {
        if self.is_object() {
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
                    ValueType::Null | ValueType::Bool => self.ptr == other.ptr,
                    ValueType::String => self.as_string_unchecked() == other.as_string_unchecked(),
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
                    ValueType::Object => None,
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

/// Trait which abstracts over the various number and string types
/// which can be used to index into an [`IValue`].
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
    IString: String, &String, &mut String, &str, &mut str;
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

impl Default for IValue {
    fn default() -> Self {
        Self::NULL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[mockalloc::test]
    fn can_use_literal() {
        let x: IValue = ijson!({
            "foo": "bar",
            "x": [],
            "y": ["hi", "there", 1, 2, null, false, true, 63.5],
            "z": [false, {
                "a": null
            }, {}]
        });
        let y: IValue = serde_json::from_str(
            r#"{
                "foo": "bar",
                "x": [],
                "y": ["hi", "there", 1, 2, null, false, true, 63.5],
                "z": [false, {
                    "a": null
                }, {}]
            }"#,
        )
        .unwrap();
        assert_eq!(x, y);
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_null() {
        let x: IValue = IValue::NULL;
        assert!(x.is_null());
        assert_eq!(x.type_(), ValueType::Null);
        assert!(matches!(x.clone().destructure(), Destructured::Null));
        assert!(matches!(x.clone().destructure_ref(), DestructuredRef::Null));
        assert!(matches!(x.clone().destructure_mut(), DestructuredMut::Null));
    }

    #[test]
    fn test_bool() {
        for v in [true, false].iter().copied() {
            let mut x = IValue::from(v);
            assert!(x.is_bool());
            assert_eq!(x.type_(), ValueType::Bool);
            assert_eq!(x.to_bool(), Some(v));
            assert!(matches!(x.clone().destructure(), Destructured::Bool(u) if u == v));
            assert!(matches!(x.clone().destructure_ref(), DestructuredRef::Bool(u) if u == v));
            assert!(
                matches!(x.clone().destructure_mut(), DestructuredMut::Bool(u) if u.get() == v)
            );

            if let DestructuredMut::Bool(mut b) = x.destructure_mut() {
                b.set(!v);
            }

            assert_eq!(x.to_bool(), Some(!v));
        }
    }

    #[mockalloc::test]
    fn test_number() {
        for v in 300..400 {
            let mut x = IValue::from(v);
            assert!(x.is_number());
            assert_eq!(x.type_(), ValueType::Number);
            assert_eq!(x.to_i32(), Some(v));
            assert_eq!(x.to_u32(), Some(v as u32));
            assert_eq!(x.to_i64(), Some(i64::from(v)));
            assert_eq!(x.to_u64(), Some(v as u64));
            assert_eq!(x.to_isize(), Some(v as isize));
            assert_eq!(x.to_usize(), Some(v as usize));
            assert_eq!(x.as_number(), Some(&v.into()));
            assert_eq!(x.as_number_mut(), Some(&mut v.into()));
            assert!(matches!(x.clone().destructure(), Destructured::Number(u) if u == v.into()));
            assert!(
                matches!(x.clone().destructure_ref(), DestructuredRef::Number(u) if *u == v.into())
            );
            assert!(
                matches!(x.clone().destructure_mut(), DestructuredMut::Number(u) if *u == v.into())
            );
        }
    }

    #[mockalloc::test]
    fn test_string() {
        for v in 0..10 {
            let s = v.to_string();
            let mut x = IValue::from(&s);
            assert!(x.is_string());
            assert_eq!(x.type_(), ValueType::String);
            assert_eq!(x.as_string(), Some(&IString::intern(&s)));
            assert_eq!(x.as_string_mut(), Some(&mut IString::intern(&s)));
            assert!(matches!(x.clone().destructure(), Destructured::String(u) if u == s));
            assert!(matches!(x.clone().destructure_ref(), DestructuredRef::String(u) if *u == s));
            assert!(matches!(x.clone().destructure_mut(), DestructuredMut::String(u) if *u == s));
        }
    }

    #[mockalloc::test]
    fn test_array() {
        for v in 0..10 {
            let mut a: IArray = (0..v).collect();
            let mut x = IValue::from(a.clone());
            assert!(x.is_array());
            assert_eq!(x.type_(), ValueType::Array);
            assert_eq!(x.as_array(), Some(&a));
            assert_eq!(x.as_array_mut(), Some(&mut a));
            assert!(matches!(x.clone().destructure(), Destructured::Array(u) if u == a));
            assert!(matches!(x.clone().destructure_ref(), DestructuredRef::Array(u) if *u == a));
            assert!(matches!(x.clone().destructure_mut(), DestructuredMut::Array(u) if *u == a));
        }
    }

    #[mockalloc::test]
    fn test_object() {
        for v in 0..10 {
            let mut o: IObject = (0..v).map(|i| (i.to_string(), i)).collect();
            let mut x = IValue::from(o.clone());
            assert!(x.is_object());
            assert_eq!(x.type_(), ValueType::Object);
            assert_eq!(x.as_object(), Some(&o));
            assert_eq!(x.as_object_mut(), Some(&mut o));
            assert!(matches!(x.clone().destructure(), Destructured::Object(u) if u == o));
            assert!(matches!(x.clone().destructure_ref(), DestructuredRef::Object(u) if *u == o));
            assert!(matches!(x.clone().destructure_mut(), DestructuredMut::Object(u) if *u == o));
        }
    }

    #[mockalloc::test]
    fn test_into_object_for_object() {
        let o: IObject = (0..10).map(|i| (i.to_string(), i)).collect();
        let x = IValue::from(o.clone());

        assert_eq!(x.into_object(), Ok(o));
    }
}

#[cfg(test)]
mod tests_defrag {
    use std::alloc::{alloc, dealloc, Layout};

    use super::*;

    struct DummyDefragAllocator;

    impl DefragAllocator for DummyDefragAllocator {
        unsafe fn realloc_ptr<T>(&mut self, ptr: *mut T, layout: Layout) -> *mut T {
            let new_ptr = self.alloc(layout).cast::<T>();
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), new_ptr.cast::<u8>(), layout.size());
            self.free(ptr, layout);
            new_ptr
        }

        /// Allocate memory for defrag
        unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
            alloc(layout)
        }

        /// Free memory for defrag
        unsafe fn free<T>(&mut self, ptr: *mut T, layout: Layout) {
            dealloc(ptr as *mut u8, layout);
        }
    }

    fn test_defrag_generic(val: IValue) {
        let defrag_val = val.clone();
        crate::reinit_shared_string_cache();
        let defrag_val = defrag_val.defrag(&mut DummyDefragAllocator);
        assert_eq!(val, defrag_val);
    }

    #[test]
    fn test_defrag_null() {
        test_defrag_generic(ijson!(null));
    }

    #[test]
    fn test_defrag_bool() {
        test_defrag_generic(ijson!(true));
        test_defrag_generic(ijson!(false));
    }

    #[test]
    fn test_defrag_number() {
        test_defrag_generic(ijson!(1));
        test_defrag_generic(ijson!(1000000000));
        test_defrag_generic(ijson!(-1000000000));
        test_defrag_generic(ijson!(1.11111111));
        test_defrag_generic(ijson!(-1.11111111));
    }

    #[test]
    fn test_defrag_string() {
        test_defrag_generic(ijson!("test"));
        test_defrag_generic(ijson!(""));
    }

    #[test]
    fn test_defrag_array() {
        test_defrag_generic(ijson!([1, 2, "bar"]));
    }

    #[test]
    fn test_defrag_array_of_numbers() {
        test_defrag_generic(ijson!([1, 2, 3]));
    }

    #[test]
    fn test_defrag_empty_array_of_numbers() {
        test_defrag_generic(ijson!([]));
    }

    #[test]
    fn test_defrag_object() {
        test_defrag_generic(ijson!({"foo": "bar"}));
    }

    #[test]
    fn test_defrag_empty_object() {
        test_defrag_generic(ijson!({}));
    }

    #[test]
    fn test_defrag_complex() {
        test_defrag_generic(ijson!([
            {"foo": "bar"}, 1, 2, 3, null, true, false, {"test":[1,2,null, true, false, 3, {"foo":[1, "bar"]}]}
        ]));
    }
}
