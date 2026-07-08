// The number and string dispatch below compares floats for exact equality on
// purpose (a number that round-trips is bit-for-bit equal); that is correct
// here, so silence the lint for the whole module.
#![allow(clippy::float_cmp)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;
use std::ops::{Deref, Index, IndexMut};
use std::ptr::NonNull;

#[cfg(feature = "indexmap")]
use indexmap::IndexMap;

// The value module owns `IValue` and its representations. Each representation is
// a submodule: the two array/object representations plus the `inline`, `scalar`
// and `interned` storage forms. `number` and `string` are *not* representations
// — each is a JSON type that spans two of those representations (inline decimal
// vs heap scalar; inline short string vs interned string). That per-type
// dispatch therefore lives on `IValue` itself, as the `new_*`/`number_*`/
// `string_*` methods below, which pick a representation as early as possible and
// then defer to it. The public wrapper types (`IArray`, `INumber`, `IObject`,
// `IString`) live in the top-level modules and delegate down to those methods.
pub(crate) mod array;
pub(crate) mod inline;
pub(crate) mod interned;
pub(crate) mod object;
pub(crate) mod scalar;

use crate::array::IArray;
use crate::number::INumber;
use crate::object::IObject;
use crate::string::IString;

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
    pub fn as_ref<'a>(&'a self) -> DestructuredRef<'a> {
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

impl BoolMut<'_> {
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

impl Deref for BoolMut<'_> {
    type Target = bool;
    fn deref(&self) -> &bool {
        if self.get() {
            &true
        } else {
            &false
        }
    }
}

const ALIGNMENT: usize = 8;

// All heap allocations pointed to by an `IValue` are aligned to `ALIGNMENT`, so
// the low 3 bits of the pointer are free to hold the `TypeTag`. Every non-inline
// tag therefore corresponds to a pointer; the `Inline` tag (0) instead stores
// the whole value inline. The inline family's bit layout, flags, and constant
// bit patterns live in the `inline` module.

#[repr(usize)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TypeTag {
    /// A value stored entirely inline (null, bool, small number, short string).
    Inline = 0,
    /// Pointer to a heap `i64` payload.
    NumberI64 = 1,
    /// Pointer to a heap `u64` payload.
    NumberU64 = 2,
    /// Pointer to a heap `f64` payload.
    NumberF64 = 3,
    /// Reserved for a future arbitrary-precision number representation.
    #[allow(dead_code)]
    NumberReserved = 4,
    /// Pointer to an interned string header.
    String = 5,
    /// Pointer to an array header.
    Array = 6,
    /// Pointer to an object header.
    Object = 7,
}

impl From<usize> for TypeTag {
    fn from(other: usize) -> Self {
        // Safety: `% ALIGNMENT` (== 8) can only return valid variants 0..=7
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

impl IValue {
    // Safety: `payload` must leave the low 3 tag bits clear (so it does not
    // corrupt the tag when ORed in) and, together with the tag, must not be
    // all-zero (reserved as the niche). Used to build inline values; `tag` is
    // normally `Inline`, with the payload carrying the sub-family and data.
    const unsafe fn new_inline(tag: TypeTag, payload: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked((tag as usize | payload) as *mut u8),
        }
    }
    // Safety: Pointer must be non-null and aligned to at least ALIGNMENT
    unsafe fn new_ptr(p: NonNull<u8>, tag: TypeTag) -> Self {
        Self {
            ptr: p.add(tag as usize),
        }
    }
    // Safety: Reference must be aligned to at least ALIGNMENT
    unsafe fn new_ref<T>(r: &T, tag: TypeTag) -> Self {
        Self::new_ptr(NonNull::from_ref(r).cast(), tag)
    }

    /// JSON `null`.
    pub const NULL: Self = unsafe { Self::new_inline(TypeTag::Inline, inline::NULL) };
    /// JSON `false`.
    pub const FALSE: Self = unsafe { Self::new_inline(TypeTag::Inline, inline::FALSE) };
    /// JSON `true`.
    pub const TRUE: Self = unsafe { Self::new_inline(TypeTag::Inline, inline::TRUE) };

    fn ptr_usize(&self) -> usize {
        self.ptr.as_ptr() as usize
    }
    // Safety: Must only be called on non-inline types
    unsafe fn ptr(&self) -> NonNull<u8> {
        self.ptr.offset(-((self.ptr_usize() % ALIGNMENT) as isize))
    }
    // Safety: Pointer must be non-null and aligned to at least ALIGNMENT
    unsafe fn set_ptr(&mut self, ptr: NonNull<u8>) {
        let tag = self.type_tag();
        self.ptr = ptr.add(tag as usize);
    }
    // Safety: Reference must be aligned to at least ALIGNMENT
    unsafe fn set_ref<T>(&mut self, r: &T) {
        self.set_ptr(NonNull::from_ref(r).cast());
    }
    unsafe fn raw_copy(&self) -> Self {
        Self { ptr: self.ptr }
    }
    pub(crate) fn raw_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
    pub(crate) fn raw_hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
    fn type_tag(&self) -> TypeTag {
        self.ptr_usize().into()
    }

    /// Whether this value is stored inline (tag `Inline`) rather than behind a
    /// pointer. What *kind* of inline value it is remains the `inline` module's
    /// concern.
    pub(crate) fn is_inline(&self) -> bool {
        self.type_tag() == TypeTag::Inline
    }

    /// Returns the type of this value.
    #[must_use]
    pub fn type_(&self) -> ValueType {
        match self.type_tag() {
            // Heap number pointers
            TypeTag::NumberI64
            | TypeTag::NumberU64
            | TypeTag::NumberF64
            | TypeTag::NumberReserved => ValueType::Number,
            TypeTag::String => ValueType::String,
            TypeTag::Array => ValueType::Array,
            TypeTag::Object => ValueType::Object,
            TypeTag::Inline => inline::value_type(self.ptr_usize()),
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
    pub fn destructure_ref<'a>(&'a self) -> DestructuredRef<'a> {
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
    pub fn destructure_mut<'a>(&'a mut self) -> DestructuredMut<'a> {
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
        self.ptr_usize() == inline::NULL
    }

    // # Bool methods
    /// Returns `true` if this is a boolean.
    #[must_use]
    pub fn is_bool(&self) -> bool {
        let bits = self.ptr_usize();
        bits == inline::TRUE || bits == inline::FALSE
    }

    /// Returns `true` if this is the `true` value.
    #[must_use]
    pub fn is_true(&self) -> bool {
        self.ptr_usize() == inline::TRUE
    }

    /// Returns `true` if this is the `false` value.
    #[must_use]
    pub fn is_false(&self) -> bool {
        self.ptr_usize() == inline::FALSE
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
        self.type_() == ValueType::Number
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
        self.is_number().then(|| self.number_to_i64()).flatten()
    }
    /// Converts this value to a u64 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        self.is_number().then(|| self.number_to_u64()).flatten()
    }
    /// Converts this value to an f64 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        self.is_number().then(|| self.number_to_f64()).flatten()
    }
    /// Converts this value to an f32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        self.is_number().then(|| self.number_to_f32()).flatten()
    }
    /// Converts this value to an i32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_i32(&self) -> Option<i32> {
        self.is_number().then(|| self.number_to_i32()).flatten()
    }
    /// Converts this value to a u32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        self.is_number().then(|| self.number_to_u32()).flatten()
    }
    /// Converts this value to an isize if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_isize(&self) -> Option<isize> {
        self.is_number().then(|| self.number_to_isize()).flatten()
    }
    /// Converts this value to a usize if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        self.is_number().then(|| self.number_to_usize()).flatten()
    }
    /// Converts this value to an f64 if it is a number, potentially losing precision
    /// in the process.
    #[must_use]
    pub fn to_f64_lossy(&self) -> Option<f64> {
        self.is_number().then(|| self.number_to_f64_lossy())
    }
    /// Converts this value to an f32 if it is a number, potentially losing precision
    /// in the process.
    #[must_use]
    pub fn to_f32_lossy(&self) -> Option<f32> {
        self.is_number().then(|| self.number_to_f32_lossy())
    }

    // # String methods
    /// Returns `true` if this is a string.
    #[must_use]
    pub fn is_string(&self) -> bool {
        self.type_() == ValueType::String
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
        self.type_tag() == TypeTag::Array
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
        self.type_tag() == TypeTag::Object
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

// A number reduced to a form suitable for exact numeric comparison. Values that
// fit `i64` become `Int`; heap `u64`s above `i64::MAX` become `UInt`; everything
// else — fractions and integer-valued floats too large for `i64`, all exactly
// representable as `f64` — becomes `Float`. This spans the full numeric range
// without needing `i128`.
enum NumVal {
    Int(i64),
    UInt(u64),
    Float(f64),
}

fn can_represent_as_f64(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 11
}

// `a == trunc(b)` already; the fractional part of `b` breaks the tie.
fn cmp_by_fraction(b: f64, bt: f64) -> Ordering {
    if b == bt {
        Ordering::Equal
    } else if b > bt {
        Ordering::Less // b has a positive fractional part, so b > a
    } else {
        Ordering::Greater
    }
}

/// Compares an `i64` to a finite float exactly.
fn cmp_i64_f64(a: i64, b: f64) -> Ordering {
    const I64_RANGE: f64 = 9_223_372_036_854_775_808.0; // 2^63
    if b >= I64_RANGE {
        return Ordering::Less; // b >= 2^63 > i64::MAX >= a
    }
    if b < -I64_RANGE {
        return Ordering::Greater; // b < -2^63 == i64::MIN <= a
    }
    let bt = b.trunc(); // now in [-2^63, 2^63), so `bt as i64` is exact
    match a.cmp(&(bt as i64)) {
        Ordering::Equal => cmp_by_fraction(b, bt),
        ord => ord,
    }
}

/// Compares a `u64` to a finite float exactly.
fn cmp_u64_f64(a: u64, b: f64) -> Ordering {
    const U64_RANGE: f64 = 18_446_744_073_709_551_616.0; // 2^64
    if b < 0.0 {
        return Ordering::Greater; // a >= 0 > b
    }
    if b >= U64_RANGE {
        return Ordering::Less; // b >= 2^64 > u64::MAX >= a
    }
    let bt = b.trunc(); // now in [0, 2^64), so `bt as u64` is exact
    match a.cmp(&(bt as u64)) {
        Ordering::Equal => cmp_by_fraction(b, bt),
        ord => ord,
    }
}

fn cmp_num(a: &NumVal, b: &NumVal) -> Ordering {
    use NumVal::{Float, Int, UInt};
    match (a, b) {
        (Int(x), Int(y)) => x.cmp(y),
        (UInt(x), UInt(y)) => x.cmp(y),
        (Int(x), UInt(y)) => {
            if *x < 0 {
                Ordering::Less
            } else {
                (*x as u64).cmp(y)
            }
        }
        (UInt(x), Int(y)) => {
            if *y < 0 {
                Ordering::Greater
            } else {
                x.cmp(&(*y as u64))
            }
        }
        (Int(x), Float(y)) => cmp_i64_f64(*x, *y),
        (Float(x), Int(y)) => cmp_i64_f64(*y, *x).reverse(),
        (UInt(x), Float(y)) => cmp_u64_f64(*x, *y),
        (Float(x), UInt(y)) => cmp_u64_f64(*y, *x).reverse(),
        (Float(x), Float(y)) => x.partial_cmp(y).unwrap(),
    }
}

// Number-type dispatch. A JSON number is stored either as an inline decimal
// (`inline::number`) or a heap scalar (`scalar`). Construction asks the inline
// representation to encode the value and only falls back to the heap when it
// cannot; the accessors dispatch on the tag and defer to the owning
// representation immediately.
impl IValue {
    pub(crate) fn new_i64(value: i64) -> Self {
        match inline::number::encode_int(value) {
            // Safety: `encode_int` returns valid inline bits; the scalar
            // allocation is aligned and non-null.
            Some(bits) => unsafe { Self::new_inline(TypeTag::Inline, bits) },
            None => unsafe { Self::new_ptr(scalar::alloc(value as u64), TypeTag::NumberI64) },
        }
    }

    pub(crate) fn new_u64(value: u64) -> Self {
        match i64::try_from(value) {
            // Fits `i64`: canonicalise through the signed path.
            Ok(v) => Self::new_i64(v),
            // Anything above `i64::MAX` far exceeds the inline mantissa, so it can
            // only be stored as a heap `u64`.
            // Safety: the scalar allocation is aligned and non-null.
            Err(_) => unsafe { Self::new_ptr(scalar::alloc(value), TypeTag::NumberU64) },
        }
    }

    pub(crate) fn new_f64(value: f64) -> Self {
        match inline::number::encode_f64(value) {
            Some(bits) => unsafe { Self::new_inline(TypeTag::Inline, bits) },
            None => unsafe { Self::new_ptr(scalar::alloc(value.to_bits()), TypeTag::NumberF64) },
        }
    }

    // Safety: reads the heap scalar payload; only valid for heap number tags.
    unsafe fn scalar_bits(&self) -> u64 {
        scalar::read(self.ptr())
    }

    fn number_num_val(&self) -> NumVal {
        if self.is_inline() {
            let bits = self.ptr_usize();
            match inline::number::value_i64(bits) {
                Some(i) => NumVal::Int(i),
                // Fractional, or an integer-valued float too large for `i64` —
                // both are exactly representable as `f64`.
                None => NumVal::Float(
                    inline::number::to_f64_exact(bits)
                        .expect("a non-i64 inline number is exactly representable as f64"),
                ),
            }
        } else {
            // Safety: not inline, so it is a heap scalar number.
            unsafe {
                match self.type_tag() {
                    TypeTag::NumberI64 => NumVal::Int(self.scalar_bits() as i64),
                    TypeTag::NumberU64 => NumVal::UInt(self.scalar_bits()),
                    _ => NumVal::Float(f64::from_bits(self.scalar_bits())),
                }
            }
        }
    }

    pub(crate) fn number_to_i64(&self) -> Option<i64> {
        match self.number_num_val() {
            NumVal::Int(x) => Some(x),
            NumVal::UInt(x) => i64::try_from(x).ok(),
            NumVal::Float(x) => (x.fract() == 0.0 && x >= i64::MIN as f64 && x < i64::MAX as f64)
                .then_some(x as i64),
        }
    }

    pub(crate) fn number_to_u64(&self) -> Option<u64> {
        match self.number_num_val() {
            NumVal::Int(x) => u64::try_from(x).ok(),
            NumVal::UInt(x) => Some(x),
            NumVal::Float(x) => {
                (x.fract() == 0.0 && x >= 0.0 && x < u64::MAX as f64).then_some(x as u64)
            }
        }
    }

    pub(crate) fn number_to_f64(&self) -> Option<f64> {
        if self.is_inline() {
            inline::number::to_f64_exact(self.ptr_usize())
        } else {
            // Safety: not inline, so it is a heap scalar number.
            unsafe {
                match self.type_tag() {
                    TypeTag::NumberI64 => {
                        let x = self.scalar_bits() as i64;
                        can_represent_as_f64(x.unsigned_abs()).then_some(x as f64)
                    }
                    TypeTag::NumberU64 => {
                        let x = self.scalar_bits();
                        can_represent_as_f64(x).then_some(x as f64)
                    }
                    _ => Some(f64::from_bits(self.scalar_bits())),
                }
            }
        }
    }

    pub(crate) fn number_to_f64_lossy(&self) -> f64 {
        if self.is_inline() {
            inline::number::to_f64_lossy(self.ptr_usize())
        } else {
            // Safety: not inline, so it is a heap scalar number.
            unsafe {
                match self.type_tag() {
                    TypeTag::NumberI64 => self.scalar_bits() as i64 as f64,
                    TypeTag::NumberU64 => self.scalar_bits() as f64,
                    _ => f64::from_bits(self.scalar_bits()),
                }
            }
        }
    }

    pub(crate) fn number_has_decimal_point(&self) -> bool {
        if self.is_inline() {
            inline::number::has_decimal_point(self.ptr_usize())
        } else {
            self.type_tag() == TypeTag::NumberF64
        }
    }

    pub(crate) fn number_to_i32(&self) -> Option<i32> {
        self.number_to_i64().and_then(|x| x.try_into().ok())
    }
    pub(crate) fn number_to_u32(&self) -> Option<u32> {
        self.number_to_u64().and_then(|x| x.try_into().ok())
    }
    pub(crate) fn number_to_isize(&self) -> Option<isize> {
        self.number_to_i64().and_then(|x| x.try_into().ok())
    }
    pub(crate) fn number_to_usize(&self) -> Option<usize> {
        self.number_to_u64().and_then(|x| x.try_into().ok())
    }
    pub(crate) fn number_to_f32(&self) -> Option<f32> {
        // A value is exactly an f32 only if it is exactly an f64.
        let x = self.number_to_f64()?;
        let u = x as f32;
        (f64::from(u) == x).then_some(u)
    }
    pub(crate) fn number_to_f32_lossy(&self) -> f32 {
        self.number_to_f64_lossy() as f32
    }

    pub(crate) fn number_hash<H: Hasher>(&self, state: &mut H) {
        if let Some(x) = self.number_to_i64() {
            x.hash(state);
        } else if let Some(x) = self.number_to_u64() {
            x.hash(state);
        } else {
            let f = self.number_to_f64_lossy();
            (if f == 0.0 { 0 } else { f.to_bits() }).hash(state);
        }
    }

    pub(crate) fn number_cmp(&self, other: &Self) -> Ordering {
        if self.raw_eq(other) {
            Ordering::Equal
        } else {
            cmp_num(&self.number_num_val(), &other.number_num_val())
        }
    }

    pub(crate) fn number_debug(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(x) = self.number_to_i64() {
            Debug::fmt(&x, f)
        } else if let Some(x) = self.number_to_u64() {
            Debug::fmt(&x, f)
        } else {
            Debug::fmt(&self.number_to_f64_lossy(), f)
        }
    }
}

// String-type dispatch. A JSON string is stored either inline (`inline::string`)
// or interned (`interned`). Construction asks the inline representation to encode
// the string and falls back to interning only when it does not fit; the accessors
// dispatch on the tag and defer to the owning representation immediately.
impl IValue {
    pub(crate) fn new_string(s: &str) -> Self {
        match inline::string::try_encode(s) {
            // Safety: `try_encode` returns valid inline-string bits.
            Some(bits) => unsafe { Self::new_inline(TypeTag::Inline, bits) },
            // Safety: `intern` returns a live, aligned interned header pointer.
            None => unsafe { Self::new_ptr(interned::intern(s), TypeTag::String) },
        }
    }

    pub(crate) fn string_len(&self) -> usize {
        if self.is_inline() {
            inline::string::len(self.ptr_usize())
        } else {
            // Safety: not an inline string, so it is interned.
            unsafe { interned::len(self.ptr()) }
        }
    }

    pub(crate) fn string_bytes(&self) -> &[u8] {
        if self.is_inline() {
            // Safety: an inline string keeps its bytes within `self`'s storage.
            unsafe { inline::string::bytes(NonNull::from(self).cast(), self.ptr_usize()) }
        } else {
            // Safety: not an inline string, so it is interned.
            unsafe { interned::bytes(self.ptr()) }
        }
    }

    pub(crate) fn string_as_str(&self) -> &str {
        // Safety: inline and interned string bytes are both valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.string_bytes()) }
    }

    pub(crate) fn string_cmp(&self, other: &Self) -> Ordering {
        if self.raw_eq(other) {
            Ordering::Equal
        } else {
            self.string_as_str().cmp(other.string_as_str())
        }
    }

    pub(crate) fn string_debug(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.string_as_str(), f)
    }
}

#[cfg(test)]
impl IValue {
    /// Test-only key identifying the exact internal representation of a *number*:
    /// its tag together with the inline bits (for inline values) or the 8-byte
    /// heap scalar payload. Two numbers with equal keys are stored bit-for-bit
    /// identically. Only meaningful when called on a number.
    pub(crate) fn number_repr_key(&self) -> (u8, u64) {
        let tag = self.type_tag() as u8;
        if self.is_inline() {
            (tag, self.ptr_usize() as u64)
        } else {
            // Safety: a heap number stores its payload as the 8-byte scalar at
            // `ptr()`; only called on numbers.
            (tag, unsafe { scalar::read(self.ptr()) })
        }
    }
}

impl Clone for IValue {
    fn clone(&self) -> Self {
        // Dispatch on the raw representation tag rather than the semantic
        // `type_()`. Inline values own no heap storage, so they are copied
        // bit-for-bit; only the pointer representations delegate to their module.
        match self.type_tag() {
            TypeTag::Inline => Self { ptr: self.ptr },
            // Safety: the tag identifies the representation
            TypeTag::NumberI64
            | TypeTag::NumberU64
            | TypeTag::NumberF64
            | TypeTag::NumberReserved => unsafe {
                Self::new_ptr(scalar::alloc(scalar::read(self.ptr())), self.type_tag())
            },
            TypeTag::String => unsafe {
                interned::bump_rc(self.ptr());
                self.raw_copy()
            },
            TypeTag::Array => unsafe { array::clone(self) },
            TypeTag::Object => unsafe { object::clone(self) },
        }
    }
}

impl Drop for IValue {
    fn drop(&mut self) {
        // Dispatch on the raw representation tag. Inline values own no heap
        // storage and need no cleanup; only the pointer representations free
        // their allocation. Deliberately avoids `type_()` (a semantic query) so
        // teardown can never re-enter type classification.
        match self.type_tag() {
            TypeTag::Inline => {}
            // Safety: the tag identifies the representation
            TypeTag::NumberI64
            | TypeTag::NumberU64
            | TypeTag::NumberF64
            | TypeTag::NumberReserved => unsafe { scalar::free(self.ptr()) },
            TypeTag::String => unsafe { interned::release(self.ptr()) },
            TypeTag::Array => unsafe { array::drop(self) },
            TypeTag::Object => unsafe { object::drop(self) },
        }
    }
}

impl Hash for IValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.type_tag() {
            // Inline numbers hash by numeric value (via the same code path as heap
            // numbers, so `2` and `2.0` agree); inline strings and the `null`/bool
            // constants have a canonical bit pattern and hash by it.
            TypeTag::Inline => {
                if inline::is_number(self.ptr_usize()) {
                    self.number_hash(state);
                } else {
                    self.ptr_usize().hash(state);
                }
            }
            TypeTag::NumberI64
            | TypeTag::NumberU64
            | TypeTag::NumberF64
            | TypeTag::NumberReserved => self.number_hash(state),
            // Interned strings hash by their canonical pointer
            TypeTag::String => self.ptr.hash(state),
            TypeTag::Array => unsafe { array::hash(self, state) },
            TypeTag::Object => unsafe { object::hash(self, state) },
        }
    }
}

impl PartialEq for IValue {
    fn eq(&self, other: &Self) -> bool {
        let t = self.type_();
        if t != other.type_() {
            return false;
        }
        // Safety: both values are of type `t`; only that type's op is called
        unsafe {
            match t {
                // `null`, booleans and (canonical) strings compare by bits
                ValueType::Null | ValueType::Bool | ValueType::String => self.raw_eq(other),
                ValueType::Number => self.number_cmp(other) == Ordering::Equal,
                ValueType::Array => array::eq(self, other),
                ValueType::Object => object::eq(self, other),
            }
        }
    }
}

impl Eq for IValue {}
impl PartialOrd for IValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (t1, t2) = (self.type_(), other.type_());
        if t1 == t2 {
            // Safety: both values are of type `t1`; only that type's op is called
            unsafe {
                match t1 {
                    ValueType::Null => Some(Ordering::Equal),
                    ValueType::Bool => self.is_true().partial_cmp(&other.is_true()),
                    ValueType::String => Some(self.string_cmp(other)),
                    ValueType::Number => Some(self.number_cmp(other)),
                    ValueType::Array => array::cmp(self, other),
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
        // Debug is a semantic (display) operation, so it dispatches on the JSON
        // type; `type_()` delegates inline classification to the `inline` module.
        // Debug is a semantic (display) operation, so it dispatches on the JSON
        // type and delegates to the owning module.
        // Safety: each arm only borrows the value as the type it just checked.
        unsafe {
            match self.type_() {
                ValueType::Null => f.write_str("null"),
                ValueType::Bool => Debug::fmt(&self.is_true(), f),
                ValueType::Number => self.number_debug(f),
                ValueType::String => self.string_debug(f),
                ValueType::Array => array::debug(self, f),
                ValueType::Object => object::debug(self, f),
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

#[cfg(feature = "indexmap")]
typed_conversions! {
    IObject:
        IndexMap<K, V> where (K: Into<IString>, V: Into<IValue>);
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

/// Collects an iterator of values into an array [`IValue`].
impl<T: Into<IValue>> FromIterator<T> for IValue {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        IArray::from_iter(iter).into()
    }
}

/// Collects an iterator of key-value pairs into an object [`IValue`].
impl<K: Into<IString>, V: Into<IValue>> FromIterator<(K, V)> for IValue {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        IObject::from_iter(iter).into()
    }
}

/// Converts a [`serde_json::Value`] into an [`IValue`].
///
/// Conversion of numeric values may be lossy if the number is not exactly
/// representable in the destination type. The exact behaviour in that case
/// (e.g. rounding, or clamping an out-of-range magnitude) is not guaranteed
/// to be stable across versions.
impl From<serde_json::Value> for IValue {
    fn from(other: serde_json::Value) -> Self {
        match other {
            serde_json::Value::Null => IValue::NULL,
            serde_json::Value::Bool(b) => b.into(),
            serde_json::Value::Number(n) => INumber::from(n).into(),
            serde_json::Value::String(s) => s.into(),
            serde_json::Value::Array(a) => a.into_iter().collect(),
            serde_json::Value::Object(o) => IObject::from(o).into(),
        }
    }
}

/// Converts an [`IValue`] into a [`serde_json::Value`].
///
/// Conversion of numeric values may be lossy if the number is not exactly
/// representable in the destination type. The exact behaviour in that case
/// (e.g. rounding, or clamping an out-of-range magnitude) is not guaranteed
/// to be stable across versions.
impl From<IValue> for serde_json::Value {
    fn from(other: IValue) -> Self {
        match other.destructure() {
            Destructured::Null => serde_json::Value::Null,
            Destructured::Bool(b) => serde_json::Value::Bool(b),
            Destructured::Number(n) => serde_json::Value::Number(n.into()),
            Destructured::String(s) => serde_json::Value::String(s.as_str().to_owned()),
            Destructured::Array(a) => {
                serde_json::Value::Array(a.into_iter().map(Into::into).collect())
            }
            Destructured::Object(o) => serde_json::Value::Object(o.into()),
        }
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

    // Not a `mockalloc::test`: numbers in this range are now stored inline and
    // perform no allocation, which `mockalloc` treats as an error.
    #[test]
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

    #[mockalloc::test]
    fn test_from_iter_array() {
        let x: IValue = (0..5).collect();
        let y: IValue = ijson!([0, 1, 2, 3, 4]);
        assert_eq!(x, y);

        let empty: IValue = std::iter::empty::<i32>().collect();
        assert_eq!(empty, ijson!([]));
    }

    #[mockalloc::test]
    fn test_from_iter_object() {
        let x: IValue = (0..3).map(|i| (i.to_string(), i)).collect();
        let y: IValue = ijson!({"0": 0, "1": 1, "2": 2});
        assert_eq!(x, y);

        let empty: IValue = std::iter::empty::<(String, i32)>().collect();
        assert_eq!(empty, ijson!({}));
    }

    #[mockalloc::test]
    fn test_serde_json_roundtrip() {
        let json = serde_json::json!({
            "null": null,
            "bool": true,
            "int": 42,
            "neg": -17,
            "big": 18446744073709551615u64,
            "float": 63.5,
            "string": "hello",
            "array": [1, 2, 3, "four", false, null],
            "object": {"nested": [1.5, {"deep": true}]}
        });

        let ivalue: IValue = json.clone().into();
        let back: serde_json::Value = ivalue.clone().into();
        assert_eq!(json, back);

        // Also check consistency with the serde-based conversion.
        let via_serde: IValue = crate::to_value(&json).unwrap();
        assert_eq!(ivalue, via_serde);
    }
}
