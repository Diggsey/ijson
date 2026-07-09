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

// The value module owns `IValue` and its representations. Each heap
// representation implements the `ValueRepr` trait in its own submodule: `array`,
// `object`, `scalar` (heap number) and `interned` (heap string). The whole
// inline family shares a single `ValueRepr` impl, `inline::InlineRepr`, which
// decodes the family bits and dispatches to an inline sub-representation
// (`inline::number`/`string`/`constant`) via the inline-only `InlineValue`
// trait. `IValue` finds a value's representation once, via `repr()`, and every
// operation delegates down to it; the per-`NumVal`/`&str` logic that both number
// (or both string) representations share is factored into the standalone
// `num_*`/`string_*` utility functions below, never into a representation that
// reaches back up.
//
// A JSON *number* or *string* spans two representations, so `new_*` construction
// picks one as early as possible, and the one place that has to resolve an
// operand of unknown representation — comparing two numbers or two strings — does
// so through `num_val_of`/`string_as_str`. The public wrapper types (`IArray`,
// `INumber`, `IObject`, `IString`) live in the top-level modules and delegate
// down through `IValue`.
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
        self.repr().value_type(self)
    }

    /// Destructures this value into an enum which can be `match`ed on.
    #[must_use]
    pub fn destructure(self) -> Destructured {
        self.repr().destructure(self)
    }

    /// Destructures a reference to this value into an enum which can be `match`ed on.
    #[must_use]
    pub fn destructure_ref<'a>(&'a self) -> DestructuredRef<'a> {
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().destructure_ref(self) }
    }

    /// Destructures a mutable reference to this value into an enum which can be `match`ed on.
    pub fn destructure_mut<'a>(&'a mut self) -> DestructuredMut<'a> {
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().destructure_mut(self) }
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
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().len(self) }
    }

    /// Returns whether this value is empty if it is an array or object.
    /// Returns `None` for other types.
    #[must_use]
    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|len| len == 0)
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
        self.repr().to_bool(self)
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
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().to_i64(self) }
    }
    /// Converts this value to a u64 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().to_u64(self) }
    }
    /// Converts this value to an f64 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().to_f64(self) }
    }
    /// Converts this value to an f32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        // A value is exactly an f32 only if it is exactly an f64.
        self.to_f64().and_then(|x| {
            let u = x as f32;
            (f64::from(u) == x).then_some(u)
        })
    }
    /// Converts this value to an i32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_i32(&self) -> Option<i32> {
        self.to_i64().and_then(|x| x.try_into().ok())
    }
    /// Converts this value to a u32 if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        self.to_u64().and_then(|x| x.try_into().ok())
    }
    /// Converts this value to an isize if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_isize(&self) -> Option<isize> {
        self.to_i64().and_then(|x| x.try_into().ok())
    }
    /// Converts this value to a usize if it is a number that can be represented exactly.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        self.to_u64().and_then(|x| x.try_into().ok())
    }
    /// Converts this value to an f64 if it is a number, potentially losing precision
    /// in the process.
    #[must_use]
    pub fn to_f64_lossy(&self) -> Option<f64> {
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().to_f64_lossy(self) }
    }
    /// Converts this value to an f32 if it is a number, potentially losing precision
    /// in the process.
    #[must_use]
    pub fn to_f32_lossy(&self) -> Option<f32> {
        self.to_f64_lossy().map(|x| x as f32)
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
// fit `i64` become `Int`; `u64`s above `i64::MAX` become `UInt`; a value that is
// exactly an `f64` (a fraction or a large integer-valued float) becomes `Float`.
//
// `Decimal` is the residual: an exact `mantissa * 10^exp` that is *not* an `i64`
// and *not* exactly an `f64` — for example the fraction `0.1`. The inline decimal
// format can represent such values, so `NumVal` must be able to hold them exactly
// rather than rounding to the nearest `f64`. (No constructor produces one today —
// they all go through `encode_f64`/`encode_int` — but the type spans the whole
// representable domain, not the current constructor set.) `exp` is always in the
// inline range `-7..=7`.
//
// Each number *representation* reduces its own storage to a `NumVal` (see
// `inline::number::num_val` and `scalar::num_val`); the standalone `num_*`
// functions below then implement every value operation once, shared by both.
#[derive(Clone, Copy)]
pub(crate) enum NumVal {
    Int(i64),
    UInt(u64),
    Float(f64),
    Decimal { mantissa: i64, exp: i32 },
}

pub(crate) fn can_represent_as_f64(x: u64) -> bool {
    x.leading_zeros() + x.trailing_zeros() >= 11
}

/// The exact `i64` value, if it is an integer in range.
pub(crate) fn num_to_i64(nv: NumVal) -> Option<i64> {
    match nv {
        NumVal::Int(x) => Some(x),
        NumVal::UInt(x) => i64::try_from(x).ok(),
        NumVal::Float(x) => {
            (x.fract() == 0.0 && x >= i64::MIN as f64 && x < i64::MAX as f64).then_some(x as i64)
        }
        NumVal::Decimal { mantissa, exp } => {
            decimal_int_value(mantissa, exp).and_then(|v| i64::try_from(v).ok())
        }
    }
}

/// The exact `u64` value, if it is a non-negative integer in range.
pub(crate) fn num_to_u64(nv: NumVal) -> Option<u64> {
    match nv {
        NumVal::Int(x) => u64::try_from(x).ok(),
        NumVal::UInt(x) => Some(x),
        NumVal::Float(x) => {
            (x.fract() == 0.0 && x >= 0.0 && x < u64::MAX as f64).then_some(x as u64)
        }
        NumVal::Decimal { mantissa, exp } => {
            decimal_int_value(mantissa, exp).and_then(|v| u64::try_from(v).ok())
        }
    }
}

/// The exact `f64` value, if it is exactly representable. (The inline
/// representation has its own decimal-exact path; this covers the heap scalar.)
pub(crate) fn num_to_f64(nv: NumVal) -> Option<f64> {
    match nv {
        NumVal::Int(x) => can_represent_as_f64(x.unsigned_abs()).then_some(x as f64),
        NumVal::UInt(x) => can_represent_as_f64(x).then_some(x as f64),
        NumVal::Float(x) => Some(x),
        // A `Decimal` is, by construction, not exactly an `f64`.
        NumVal::Decimal { .. } => None,
    }
}

/// The (possibly lossy) `f64` value.
pub(crate) fn num_to_f64_lossy(nv: NumVal) -> f64 {
    match nv {
        NumVal::Int(x) => x as f64,
        NumVal::UInt(x) => x as f64,
        NumVal::Float(x) => x,
        NumVal::Decimal { mantissa, exp } => decimal_to_f64_lossy(mantissa, exp),
    }
}

/// Hashes a number by its numeric value, so the inline and heap forms of a value
/// (e.g. the float `1e18` and the integer `10^18`, which compare equal) agree.
pub(crate) fn num_hash(nv: NumVal, state: &mut dyn Hasher) {
    if let Some(x) = num_to_i64(nv) {
        state.write_i64(x);
    } else if let Some(x) = num_to_u64(nv) {
        state.write_u64(x);
    } else if let NumVal::Decimal { mantissa, exp } = nv {
        // A non-integer decimal is never equal to an integer or an `f64`, so its
        // hash only has to agree with equal decimals: hash the canonical form.
        let (m, e) = canonical_decimal(mantissa, exp);
        state.write_i64(m);
        state.write_i32(e);
    } else {
        let f = num_to_f64_lossy(nv);
        state.write_u64(if f == 0.0 { 0 } else { f.to_bits() });
    }
}

/// Formats a number the way `serde_json` would (integer if it is one).
pub(crate) fn num_debug(nv: NumVal, f: &mut Formatter<'_>) -> fmt::Result {
    if let Some(x) = num_to_i64(nv) {
        Debug::fmt(&x, f)
    } else if let Some(x) = num_to_u64(nv) {
        Debug::fmt(&x, f)
    } else {
        Debug::fmt(&num_to_f64_lossy(nv), f)
    }
}

// Reduces a number of *either* representation to a `NumVal`. Numbers span two
// representations, so the binary comparison below has to resolve each operand's
// representation; this is the one place that dispatch remains.
fn num_val_of(v: &IValue) -> NumVal {
    if v.is_inline() {
        inline::number::num_val(v.ptr_usize())
    } else {
        // Safety: a non-inline number is a heap scalar.
        unsafe { scalar::num_val(v) }
    }
}

/// Compares two numbers exactly, regardless of how each is represented.
pub(crate) fn number_cmp(a: &IValue, b: &IValue) -> Ordering {
    if a.raw_eq(b) {
        Ordering::Equal
    } else {
        cmp_num(&num_val_of(a), &num_val_of(b))
    }
}

/// Compares two strings, regardless of how each is represented.
pub(crate) fn string_cmp(a: &IValue, b: &IValue) -> Ordering {
    if a.raw_eq(b) {
        Ordering::Equal
    } else {
        a.string_as_str().cmp(b.string_as_str())
    }
}

/// Formats a string of either representation.
pub(crate) fn string_debug(v: &IValue, f: &mut Formatter<'_>) -> fmt::Result {
    Debug::fmt(v.string_as_str(), f)
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

// --- Exact `Decimal` arithmetic ---------------------------------------------
// A `Decimal { mantissa, exp }` is the exact value `mantissa * 10^exp` with
// `exp` in the inline range `-7..=7`, so `10^|exp|` fits an `i64` and the scaled
// products below fit an `i128`.

/// The exact integer value of `mantissa * 10^exp`, if it is an integer.
fn decimal_int_value(mantissa: i64, exp: i32) -> Option<i128> {
    if exp >= 0 {
        Some(i128::from(mantissa) * 10i128.pow(exp as u32))
    } else {
        let div = 10i128.pow((-exp) as u32);
        (i128::from(mantissa) % div == 0).then_some(i128::from(mantissa) / div)
    }
}

/// The nearest `f64` to `mantissa * 10^exp`, correctly rounded even for a mantissa
/// above `2^53`: an integer via the `i128 -> f64` cast, a fraction via the
/// correctly-rounded `f64` string parser (avoiding a double rounding).
fn decimal_to_f64_lossy(mantissa: i64, exp: i32) -> f64 {
    if exp >= 0 {
        (i128::from(mantissa) * 10i128.pow(exp as u32)) as f64
    } else {
        format!("{}e{}", mantissa, exp).parse().unwrap()
    }
}

/// Compares `mantissa * 10^exp` to the integer `n`, exactly.
fn cmp_decimal_int(mantissa: i64, exp: i32, n: i128) -> Ordering {
    if exp >= 0 {
        (i128::from(mantissa) * 10i128.pow(exp as u32)).cmp(&n)
    } else {
        // `mantissa / 10^k` vs `n` ⟺ `mantissa` vs `n * 10^k`.
        i128::from(mantissa).cmp(&(n * 10i128.pow((-exp) as u32)))
    }
}

/// Compares two exact decimals.
fn cmp_decimal_decimal(m1: i64, e1: i32, m2: i64, e2: i32) -> Ordering {
    let de = e1 - e2;
    if de >= 0 {
        (i128::from(m1) * 10i128.pow(de as u32)).cmp(&i128::from(m2))
    } else {
        i128::from(m1).cmp(&(i128::from(m2) * 10i128.pow((-de) as u32)))
    }
}

/// `mantissa * 10^exp` with trailing decimal zeros removed, so equal decimals
/// share one form (used for hashing).
fn canonical_decimal(mut mantissa: i64, mut exp: i32) -> (i64, i32) {
    if mantissa == 0 {
        return (0, 0);
    }
    while mantissa % 10 == 0 {
        mantissa /= 10;
        exp += 1;
    }
    (mantissa, exp)
}

/// `x << n`, or `None` when the value (not just the shift amount) would overflow.
fn shl_u128(x: u128, n: u32) -> Option<u128> {
    if x == 0 {
        Some(0)
    } else if n <= x.leading_zeros() {
        Some(x << n)
    } else {
        None
    }
}

/// Decomposes a finite, positive `f64` into `(frac, exp2)` with `v == frac * 2^exp2`.
fn f64_frac_exp(v: f64) -> (u64, i32) {
    let bits = v.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    if raw_exp == 0 {
        (frac, -1074) // subnormal
    } else {
        (frac | 0x0010_0000_0000_0000, raw_exp - 1075)
    }
}

/// Compares a `u128` to a finite, non-negative float exactly.
fn cmp_u128_f64(a: u128, b: f64) -> Ordering {
    const U128_RANGE: f64 = 340_282_366_920_938_463_463_374_607_431_768_211_456.0; // 2^128
    if b >= U128_RANGE {
        return Ordering::Less; // b >= 2^128 > a
    }
    let bt = b.trunc(); // now in [0, 2^128), so `bt as u128` is exact
    match a.cmp(&(bt as u128)) {
        Ordering::Equal => cmp_by_fraction(b, bt),
        ord => ord,
    }
}

/// Compares `m_abs * 10^exp` to a finite, positive float `v`, exactly.
fn cmp_decimal_magnitude(m_abs: u64, exp: i32, v: f64) -> Ordering {
    if exp >= 0 {
        // `m_abs * 10^exp` is an integer (it fits `u128` for the inline range).
        cmp_u128_f64(u128::from(m_abs) * 10u128.pow(exp as u32), v)
    } else {
        // `|d| = m_abs / 10^k` vs `v = frac * 2^fe`. Clearing `10^k = 2^k * 5^k`:
        // compare `m_abs` to `frac * 5^k * 2^(fe + k)`.
        let k = (-exp) as u32;
        let (frac, fe) = f64_frac_exp(v);
        let p = u128::from(frac) * 5u128.pow(k); // < 2^70
        let s = fe + k as i32;
        if s >= 0 {
            match shl_u128(p, s as u32) {
                Some(rhs) => u128::from(m_abs).cmp(&rhs),
                None => Ordering::Less, // rhs overflows u128 -> |d| < v
            }
        } else {
            match shl_u128(u128::from(m_abs), (-s) as u32) {
                Some(lhs) => lhs.cmp(&p),
                None => Ordering::Greater, // lhs overflows u128 -> |d| > v
            }
        }
    }
}

/// Compares `mantissa * 10^exp` to a finite float exactly. A `Decimal` is, by
/// construction, never exactly an `f64`, so this never returns `Equal`.
fn cmp_decimal_f64(mantissa: i64, exp: i32, v: f64) -> Ordering {
    let d_neg = mantissa < 0;
    if v == 0.0 {
        return if d_neg {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if d_neg != (v < 0.0) {
        return if d_neg {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let ord = cmp_decimal_magnitude(mantissa.unsigned_abs(), exp, v.abs());
    if d_neg {
        ord.reverse()
    } else {
        ord
    }
}

fn cmp_num(a: &NumVal, b: &NumVal) -> Ordering {
    use NumVal::{Decimal, Float, Int, UInt};
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
        (
            Decimal { mantissa, exp },
            Decimal {
                mantissa: m2,
                exp: e2,
            },
        ) => cmp_decimal_decimal(*mantissa, *exp, *m2, *e2),
        (Decimal { mantissa, exp }, Int(y)) => cmp_decimal_int(*mantissa, *exp, i128::from(*y)),
        (Int(x), Decimal { mantissa, exp }) => {
            cmp_decimal_int(*mantissa, *exp, i128::from(*x)).reverse()
        }
        (Decimal { mantissa, exp }, UInt(y)) => cmp_decimal_int(*mantissa, *exp, i128::from(*y)),
        (UInt(x), Decimal { mantissa, exp }) => {
            cmp_decimal_int(*mantissa, *exp, i128::from(*x)).reverse()
        }
        // A `Decimal` (an exact non-`f64` value) and a `Float` are compared
        // exactly: `0.1` (decimal) and `0.1_f64` are different numbers.
        (Decimal { mantissa, exp }, Float(y)) => cmp_decimal_f64(*mantissa, *exp, *y),
        (Float(x), Decimal { mantissa, exp }) => cmp_decimal_f64(*mantissa, *exp, *x).reverse(),
    }
}

#[cfg(test)]
mod num_val_tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(nv: NumVal) -> u64 {
        let mut h = DefaultHasher::new();
        num_hash(nv, &mut h);
        h.finish()
    }

    #[test]
    fn decimal_extracts_integers_but_not_fractions() {
        // 0.1 is a fraction: not an integer, not an exact f64.
        let tenth = NumVal::Decimal {
            mantissa: 1,
            exp: -1,
        };
        assert_eq!(num_to_i64(tenth), None);
        assert_eq!(num_to_u64(tenth), None);
        assert_eq!(num_to_f64(tenth), None);
        assert_eq!(num_to_f64_lossy(tenth), 0.1);

        // 20 * 10^-1 == 2 is an integer-valued decimal.
        let two = NumVal::Decimal {
            mantissa: 20,
            exp: -1,
        };
        assert_eq!(num_to_i64(two), Some(2));
        assert_eq!(num_to_u64(two), Some(2));
    }

    #[test]
    fn decimal_compares_exactly_against_decimals_and_integers() {
        let tenth = NumVal::Decimal {
            mantissa: 1,
            exp: -1,
        }; // 0.1
        let three_tenths = NumVal::Decimal {
            mantissa: 3,
            exp: -1,
        }; // 0.3
        let tenth_scaled = NumVal::Decimal {
            mantissa: 10,
            exp: -2,
        }; // 0.10 == 0.1
        assert_eq!(cmp_num(&tenth, &three_tenths), Ordering::Less);
        assert_eq!(cmp_num(&tenth, &tenth_scaled), Ordering::Equal);
        assert_eq!(cmp_num(&tenth, &NumVal::Int(0)), Ordering::Greater);
        assert_eq!(cmp_num(&tenth, &NumVal::Int(1)), Ordering::Less);

        // An integer-valued decimal orders exactly with the equal integer.
        let two = NumVal::Decimal {
            mantissa: 20,
            exp: -1,
        };
        assert_eq!(cmp_num(&two, &NumVal::Int(2)), Ordering::Equal);
        assert_eq!(cmp_num(&two, &NumVal::UInt(2)), Ordering::Equal);
        assert_eq!(cmp_num(&NumVal::Int(2), &two), Ordering::Equal);
    }

    #[test]
    fn decimal_hash_stays_consistent_with_equality() {
        // An integer-valued decimal hashes like the equal integer.
        let two_dec = NumVal::Decimal {
            mantissa: 20,
            exp: -1,
        };
        assert_eq!(hash_of(two_dec), hash_of(NumVal::Int(2)));
        assert_eq!(hash_of(two_dec), hash_of(NumVal::UInt(2)));

        // Equal fractions hash alike (canonical form), regardless of how the
        // mantissa/exponent are written.
        let a = NumVal::Decimal {
            mantissa: 1,
            exp: -1,
        };
        let b = NumVal::Decimal {
            mantissa: 10,
            exp: -2,
        };
        assert_eq!(hash_of(a), hash_of(b));

        // The exact decimal 0.1 and the f64 0.1 are *different* numbers, so they
        // are unequal — and hashing them differently is therefore allowed.
        assert_ne!(cmp_num(&a, &NumVal::Float(0.1)), Ordering::Equal);
        assert_eq!(cmp_num(&a, &NumVal::Float(0.1)), Ordering::Less);
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

    /// Constructs the exact decimal `mantissa * 10^exp` (as written, with a
    /// decimal point) if it fits the inline representation; `None` otherwise, so
    /// the caller can fall back to an `f64`. This is how a decimal that is *not*
    /// an exact `f64` (e.g. `0.1`) is stored without losing precision.
    pub(crate) fn new_decimal(mantissa: i128, exp: i32) -> Option<Self> {
        // Safety: `encode_decimal` returns valid inline bits.
        inline::number::encode_decimal(mantissa, exp)
            .map(|bits| unsafe { Self::new_inline(TypeTag::Inline, bits) })
    }

    /// Whether this number was written with a decimal point (`1.0` vs `1`).
    /// Delegates down to the representation; `false` for non-numbers.
    pub(crate) fn has_decimal_point(&self) -> bool {
        self.repr().has_decimal_point(self)
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

    pub(crate) fn string_bytes(&self) -> &[u8] {
        // Safety: `repr()` selects this value's own representation; `as_bytes` is
        // `Some` for both string representations (and this is only called on a
        // string).
        unsafe { self.repr().as_bytes(self).expect("not a string") }
    }

    pub(crate) fn string_as_str(&self) -> &str {
        // Safety: inline and interned string bytes are both valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.string_bytes()) }
    }

    pub(crate) fn string_len(&self) -> usize {
        self.string_bytes().len()
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

/// The operations each value *representation* provides, along with defaults so a
/// representation only overrides what it supports. Every operation is
/// fundamentally per-representation: cloning an inline value is a bit-copy, a
/// scalar allocates, an interned string bumps a refcount, and so on. `IValue`'s
/// public methods and its `Clone`/`Drop`/`PartialEq`/`PartialOrd`/`Debug` impls
/// find the representation once, via [`IValue::repr`], and delegate to a trait
/// method — the delegation only ever goes downward.
///
/// # Safety
///
/// Every method's `IValue` arguments must belong to this representation. For the
/// binary `eq`/`partial_cmp`, the first argument is this representation and the
/// second is guaranteed by the caller to be the same JSON *type* (possibly a
/// different representation of it — e.g. an inline vs heap number).
pub(crate) trait ValueRepr {
    /// The JSON type this representation stores. Takes `v` because a single
    /// representation may cover several types (the inline family), decoding `v`
    /// to tell them apart; representations that cover one type ignore it.
    fn value_type(&self, v: &IValue) -> ValueType;

    /// Clone the value. Default: copy the pointer word — correct for the inline
    /// representations, which own no heap storage. Heap reps override.
    unsafe fn clone(&self, v: &IValue) -> IValue {
        IValue { ptr: v.ptr }
    }
    /// Release the value's storage. Default: nothing (inline). Heap reps override.
    unsafe fn drop(&self, _v: &mut IValue) {}
    /// Hash by value. Default: the canonical pointer word — correct for the inline
    /// constants and both string representations (equal values share it). Numbers
    /// hash by their numeric value (so the inline and heap forms of a value agree),
    /// and collections recurse into their elements' representations.
    ///
    /// `hash` uses `&mut dyn Hasher` because a trait-object method cannot be
    /// generic; `IValue: Hash` erases the concrete hasher once, at the top.
    unsafe fn hash(&self, v: &IValue, state: &mut dyn Hasher) {
        state.write_usize(v.ptr_usize());
    }
    /// Equality within a type. Default: canonical bits — correct for the constants
    /// and strings. Numbers and collections override.
    unsafe fn eq(&self, a: &IValue, b: &IValue) -> bool {
        a.raw_eq(b)
    }
    /// Ordering within a type. Default: unordered (only `Object` keeps it).
    unsafe fn partial_cmp(&self, _a: &IValue, _b: &IValue) -> Option<Ordering> {
        None
    }
    unsafe fn debug(&self, v: &IValue, f: &mut Formatter<'_>) -> fmt::Result;

    /// Wrap this value in the owned destructuring enum.
    fn destructure(&self, v: IValue) -> Destructured;
    /// Wrap a reference to this value in the borrowed destructuring enum.
    unsafe fn destructure_ref<'a>(&self, v: &'a IValue) -> DestructuredRef<'a>;
    /// Wrap a mutable reference to this value in the mutable destructuring enum.
    unsafe fn destructure_mut<'a>(&self, v: &'a mut IValue) -> DestructuredMut<'a>;

    // Value accessors: the default is "this representation is not that kind of
    // value", so each representation only overrides what it actually supports.
    fn to_bool(&self, _v: &IValue) -> Option<bool> {
        None
    }
    unsafe fn to_i64(&self, _v: &IValue) -> Option<i64> {
        None
    }
    unsafe fn to_u64(&self, _v: &IValue) -> Option<u64> {
        None
    }
    unsafe fn to_f64(&self, _v: &IValue) -> Option<f64> {
        None
    }
    unsafe fn to_f64_lossy(&self, _v: &IValue) -> Option<f64> {
        None
    }
    /// Length of a collection; `None` for everything else.
    unsafe fn len(&self, _v: &IValue) -> Option<usize> {
        None
    }
    /// The UTF-8 bytes of a string; `None` for every non-string representation.
    /// The two string representations override this with their own byte access.
    unsafe fn as_bytes<'a>(&self, _v: &'a IValue) -> Option<&'a [u8]> {
        None
    }
    /// Whether a number was written with a decimal point (`1.0` vs `1`); `false`
    /// for every non-number representation. The two number representations
    /// override this.
    fn has_decimal_point(&self, _v: &IValue) -> bool {
        false
    }
}

impl IValue {
    /// The representation this value is stored in — the single dispatch point that
    /// every method delegates through. Returning a `&'static dyn` built from a
    /// match of concrete zero-sized markers lets the optimizer devirtualize each
    /// arm back to a direct call once this is inlined.
    #[inline]
    fn repr(&self) -> &'static dyn ValueRepr {
        match self.type_tag() {
            // One representation covers the whole inline family; it decodes the
            // family bits to dispatch further (see `inline::InlineRepr`).
            TypeTag::Inline => &inline::InlineRepr,
            TypeTag::NumberI64
            | TypeTag::NumberU64
            | TypeTag::NumberF64
            | TypeTag::NumberReserved => &scalar::ScalarRepr,
            TypeTag::String => &interned::InternedRepr,
            TypeTag::Array => &array::ArrayRepr,
            TypeTag::Object => &object::ObjectRepr,
        }
    }
}

impl Clone for IValue {
    fn clone(&self) -> Self {
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().clone(self) }
    }
}

impl Drop for IValue {
    fn drop(&mut self) {
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().drop(self) }
    }
}

impl Hash for IValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Erase the concrete hasher once, then delegate down to this value's
        // representation — like every other operation.
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().hash(self, state) }
    }
}

impl PartialEq for IValue {
    fn eq(&self, other: &Self) -> bool {
        // Different JSON types are never equal. Within a type, the representation
        // handles any cross-representation comparison (e.g. inline vs heap number).
        // Safety: both operands share a type.
        self.type_() == other.type_() && unsafe { self.repr().eq(self, other) }
    }
}

impl Eq for IValue {}
impl PartialOrd for IValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (t1, t2) = (self.type_(), other.type_());
        if t1 == t2 {
            // Safety: both operands share a type.
            unsafe { self.repr().partial_cmp(self, other) }
        } else {
            // Different types are ordered by the `ValueType` enum.
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
        // Safety: `repr()` selects this value's own representation.
        unsafe { self.repr().debug(self, f) }
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
