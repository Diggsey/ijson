use std::convert::TryFrom;
use std::fmt::{self, Formatter};
use std::slice;

use serde::de::{
    DeserializeSeed, EnumAccess, Error as SError, Expected, IntoDeserializer, MapAccess, SeqAccess,
    Unexpected, VariantAccess, Visitor,
};
use serde::{forward_to_deserialize_any, Deserialize, Deserializer};
use serde_json::error::Error;

use super::array::IArray;
use super::number::INumber;
use super::object::IObject;
use super::string::IString;
use super::value::{DestructuredRef, IValue};

impl<'de> Deserialize<'de> for IValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor)
    }
}

impl<'de> Deserialize<'de> for INumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NumberVisitor)
    }
}

impl<'de> Deserialize<'de> for IString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(StringVisitor)
    }
}

impl<'de> Deserialize<'de> for IArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ArrayVisitor)
    }
}

impl<'de> Deserialize<'de> for IObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ObjectVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = IValue;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("any valid JSON value")
    }

    #[inline]
    fn visit_bool<E: SError>(self, value: bool) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_i64<E: SError>(self, value: i64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_u64<E: SError>(self, value: u64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_f64<E: SError>(self, value: f64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_str<E: SError>(self, value: &str) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_string<E: SError>(self, value: String) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_none<E: SError>(self) -> Result<IValue, E> {
        Ok(IValue::NULL)
    }

    #[inline]
    fn visit_some<D>(self, deserializer: D) -> Result<IValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer)
    }

    #[inline]
    fn visit_unit<E: SError>(self) -> Result<IValue, E> {
        Ok(IValue::NULL)
    }

    #[inline]
    fn visit_seq<V>(self, visitor: V) -> Result<IValue, V::Error>
    where
        V: SeqAccess<'de>,
    {
        ArrayVisitor.visit_seq(visitor).map(Into::into)
    }

    #[cfg(not(feature = "arbitrary_precision"))]
    fn visit_map<V>(self, visitor: V) -> Result<IValue, V::Error>
    where
        V: MapAccess<'de>,
    {
        ObjectVisitor.visit_map(visitor).map(Into::into)
    }

    // With `arbitrary_precision` a number can arrive here disguised as a one-entry map (see
    // `NUMBER_TOKEN`), so peek the first key: the token means it is really a number, anything
    // else is a genuine object key and the object is finished from there.
    #[cfg(feature = "arbitrary_precision")]
    fn visit_map<V>(self, mut map: V) -> Result<IValue, V::Error>
    where
        V: MapAccess<'de>,
    {
        match map.next_key_seed(MapKeyClassifier)? {
            Some(MapKey::Number) => number_from_map_value(&mut map).map(IValue::from),
            Some(MapKey::Key(first_key)) => build_object(map, Some(first_key)).map(Into::into),
            None => Ok(IObject::with_capacity(0).into()),
        }
    }
}

struct NumberVisitor;

impl<'de> Visitor<'de> for NumberVisitor {
    type Value = INumber;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON number")
    }

    #[inline]
    fn visit_i64<E: SError>(self, value: i64) -> Result<INumber, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_u64<E: SError>(self, value: u64) -> Result<INumber, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_f64<E: SError>(self, value: f64) -> Result<INumber, E> {
        INumber::try_from(value).map_err(|_| E::invalid_value(Unexpected::Float(value), &self))
    }

    // With `arbitrary_precision`, serde_json delivers a float or an out-of-`u64` integer as
    // a `{ NUMBER_TOKEN: "<literal>" }` map rather than through `visit_f64`. Re-parse the
    // literal exactly. (Integers in range still come through `visit_u64`/`visit_i64` above.)
    #[cfg(feature = "arbitrary_precision")]
    fn visit_map<V>(self, mut map: V) -> Result<INumber, V::Error>
    where
        V: MapAccess<'de>,
    {
        if map.next_key::<NumberToken>()?.is_none() {
            return Err(V::Error::invalid_type(Unexpected::Map, &self));
        }
        number_from_map_value(&mut map)
    }
}

struct StringVisitor;

impl Visitor<'_> for StringVisitor {
    type Value = IString;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON string")
    }

    #[inline]
    fn visit_str<E: SError>(self, value: &str) -> Result<IString, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_string<E: SError>(self, value: String) -> Result<Self::Value, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_bytes<E: SError>(self, value: &[u8]) -> Result<Self::Value, E> {
        match std::str::from_utf8(value) {
            Ok(s) => Ok(s.into()),
            Err(_) => Err(SError::invalid_value(Unexpected::Bytes(value), &self)),
        }
    }

    #[inline]
    fn visit_byte_buf<E: SError>(self, value: Vec<u8>) -> Result<Self::Value, E> {
        match String::from_utf8(value) {
            Ok(s) => Ok(s.into()),
            Err(e) => Err(SError::invalid_value(
                Unexpected::Bytes(&e.into_bytes()),
                &self,
            )),
        }
    }
}

struct ArrayVisitor;

impl<'de> Visitor<'de> for ArrayVisitor {
    type Value = IArray;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON array")
    }

    #[inline]
    fn visit_seq<V>(self, mut visitor: V) -> Result<IArray, V::Error>
    where
        V: SeqAccess<'de>,
    {
        let mut arr = IArray::with_capacity(visitor.size_hint().unwrap_or(0));
        while let Some(v) = visitor.next_element::<IValue>()? {
            arr.push(v);
        }
        Ok(arr)
    }
}

struct ObjectVisitor;

impl<'de> Visitor<'de> for ObjectVisitor {
    type Value = IObject;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON object")
    }

    fn visit_map<V>(self, visitor: V) -> Result<IObject, V::Error>
    where
        V: MapAccess<'de>,
    {
        build_object(visitor, None)
    }
}

/// Builds an object from the remaining map entries, optionally with one key already peeled
/// off (`first_key`). The peel-first form is what lets `ValueVisitor` look at the first key
/// to tell a real object from `arbitrary_precision`'s number-in-a-map and still finish the
/// object if it was one; `ObjectVisitor` just passes `None`.
fn build_object<'de, V>(mut map: V, first_key: Option<IString>) -> Result<IObject, V::Error>
where
    V: MapAccess<'de>,
{
    let mut obj =
        IObject::with_capacity(map.size_hint().unwrap_or(0) + usize::from(first_key.is_some()));
    if let Some(key) = first_key {
        let value = map.next_value::<IValue>()?;
        obj.insert(key, value);
    }
    while let Some((k, v)) = map.next_entry::<IString, IValue>()? {
        obj.insert(k, v);
    }
    Ok(obj)
}

// `arbitrary_precision` deserialization goes through one parser. serde_json, with its own
// `arbitrary_precision` on, does not round a number to `f64` on the way in — it hands over
// the raw literal as a single-entry map, `{ NUMBER_TOKEN: "<digits>" }`. Intercepting that
// map and re-parsing the literal with `INumber::from_str` (the same parser `str::parse`
// uses) is what keeps a value exact through a round trip: without it, `from_str("0.1")` and
// `from_str::<serde_json>("0.1")` would be different numbers, and a magnitude beyond `f64`
// would be unreadable. Integers in `i64`/`u64` range still arrive through `visit_u64`/`i64`.
#[cfg(feature = "arbitrary_precision")]
pub(crate) const NUMBER_TOKEN: &str = "$serde_json::private::Number";

/// Parses the value half of a number map (its key already consumed) with `INumber`'s own
/// parser — the single boundary all number text passes through.
#[cfg(feature = "arbitrary_precision")]
fn number_from_map_value<'de, V>(map: &mut V) -> Result<INumber, V::Error>
where
    V: MapAccess<'de>,
{
    let literal = map.next_value::<String>()?;
    literal
        .parse::<INumber>()
        .map_err(|_| V::Error::custom(format_args!("invalid JSON number {:?}", literal)))
}

/// A map key classified as either `arbitrary_precision`'s number token or a genuine object
/// key. Peeking it is how [`ValueVisitor`] tells the two maps apart.
#[cfg(feature = "arbitrary_precision")]
enum MapKey {
    Number,
    Key(IString),
}

#[cfg(feature = "arbitrary_precision")]
struct MapKeyClassifier;

#[cfg(feature = "arbitrary_precision")]
impl<'de> DeserializeSeed<'de> for MapKeyClassifier {
    type Value = MapKey;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<MapKey, D::Error> {
        struct KeyVisitor;
        impl Visitor<'_> for KeyVisitor {
            type Value = MapKey;
            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("a JSON object key")
            }
            fn visit_str<E: SError>(self, s: &str) -> Result<MapKey, E> {
                Ok(if s == NUMBER_TOKEN {
                    MapKey::Number
                } else {
                    MapKey::Key(IString::intern(s))
                })
            }
        }
        deserializer.deserialize_str(KeyVisitor)
    }
}

/// A map key that must be exactly the number token — nothing else can be a valid number.
#[cfg(feature = "arbitrary_precision")]
struct NumberToken;

#[cfg(feature = "arbitrary_precision")]
impl<'de> Deserialize<'de> for NumberToken {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl Visitor<'_> for V {
            type Value = NumberToken;
            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("the arbitrary-precision number token")
            }
            fn visit_str<E: SError>(self, s: &str) -> Result<NumberToken, E> {
                if s == NUMBER_TOKEN {
                    Ok(NumberToken)
                } else {
                    Err(E::custom("expected a JSON number"))
                }
            }
        }
        deserializer.deserialize_identifier(V)
    }
}

macro_rules! deserialize_number {
    ($method:ident) => {
        fn $method<V>(self, visitor: V) -> Result<V::Value, Error>
        where
            V: Visitor<'de>,
        {
            if let Some(v) = self.as_number() {
                v.deserialize_any(visitor)
            } else {
                Err(self.invalid_type(&visitor))
            }
        }
    };
}

impl<'de> Deserializer<'de> for &'de IValue {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::Null => visitor.visit_unit(),
            DestructuredRef::Bool(v) => visitor.visit_bool(v),
            DestructuredRef::Number(v) => v.deserialize_any(visitor),
            DestructuredRef::String(v) => v.deserialize_any(visitor),
            DestructuredRef::Array(v) => v.deserialize_any(visitor),
            DestructuredRef::Object(v) => v.deserialize_any(visitor),
        }
    }

    deserialize_number!(deserialize_i8);
    deserialize_number!(deserialize_i16);
    deserialize_number!(deserialize_i32);
    deserialize_number!(deserialize_i64);
    deserialize_number!(deserialize_u8);
    deserialize_number!(deserialize_u16);
    deserialize_number!(deserialize_u32);
    deserialize_number!(deserialize_u64);
    deserialize_number!(deserialize_f32);
    deserialize_number!(deserialize_f64);

    #[inline]
    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self.is_null() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    #[inline]
    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::String(v) => v.deserialize_enum(name, variants, visitor),
            DestructuredRef::Object(v) => v.deserialize_enum(name, variants, visitor),
            other => Err(SError::invalid_type(other.unexpected(), &"string or map")),
        }
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.to_bool() {
            visitor.visit_bool(v)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.as_string() {
            v.deserialize_str(visitor)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::String(v) => v.deserialize_bytes(visitor),
            DestructuredRef::Array(v) => v.deserialize_bytes(visitor),
            other => Err(other.invalid_type(&visitor)),
        }
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_bytes(visitor)
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self.is_null() {
            visitor.visit_unit()
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.as_array() {
            v.deserialize_seq(visitor)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.as_object() {
            v.deserialize_map(visitor)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::Array(v) => v.deserialize_struct(name, fields, visitor),
            DestructuredRef::Object(v) => v.deserialize_struct(name, fields, visitor),
            other => Err(other.invalid_type(&visitor)),
        }
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_unit()
    }
}

impl<'de> Deserializer<'de> for &'de INumber {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        // `to_i64`/`to_u64`/`to_f64` are the *exact* accessors, and so are partial: a
        // number need be neither an integer in range nor an exact `f64` (`0.1` is
        // exactly a decimal; an integer beyond `u64` fits nothing). Serde has no
        // arbitrary-precision visitor method, so anything the visitor cannot take
        // exactly is delivered as the nearest `f64` — which is total, and is already
        // what every inexact number gets.
        if !self.has_decimal_point() {
            if let Some(v) = self.to_i64() {
                return visitor.visit_i64(v);
            }
            if let Some(v) = self.to_u64() {
                return visitor.visit_u64(v);
            }
        }
        visitor.visit_f64(self.to_f64_lossy())
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

impl<'de> Deserializer<'de> for &'de IString {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.as_str())
    }

    fn deserialize_enum<V>(
        self,
        _name: &str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_enum(EnumDeserializer {
            variant: self,
            value: None,
        })
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct identifier ignored_any
    }
}

impl<'de> Deserializer<'de> for &'de IArray {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let len = self.len();
        let mut deserializer = ArrayAccess::new(self);
        let seq = visitor.visit_seq(&mut deserializer)?;
        let remaining = deserializer.iter.len();
        if remaining == 0 {
            Ok(seq)
        } else {
            Err(SError::invalid_length(len, &"fewer elements in array"))
        }
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

impl<'de> Deserializer<'de> for &'de IObject {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let len = self.len();
        let mut deserializer = ObjectAccess::new(self);
        let seq = visitor.visit_map(&mut deserializer)?;
        let remaining = deserializer.iter.len();
        if remaining == 0 {
            Ok(seq)
        } else {
            Err(SError::invalid_length(len, &"fewer elements in object"))
        }
    }

    #[inline]
    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let mut iter = self.iter();
        let (variant, value) = iter
            .next()
            .ok_or_else(|| SError::invalid_value(Unexpected::Map, &"object with a single key"))?;
        // enums are encoded in json as maps with a single key:value pair
        if iter.next().is_some() {
            return Err(SError::invalid_value(
                Unexpected::Map,
                &"object with a single key",
            ));
        }
        visitor.visit_enum(EnumDeserializer {
            variant,
            value: Some(value),
        })
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct identifier ignored_any
    }
}

trait MaybeUnexpected<'de>: Sized {
    fn invalid_type<E>(self, exp: &dyn Expected) -> E
    where
        E: SError,
    {
        SError::invalid_type(self.unexpected(), exp)
    }

    fn unexpected(self) -> Unexpected<'de>;
}

impl<'de> MaybeUnexpected<'de> for &'de IValue {
    fn unexpected(self) -> Unexpected<'de> {
        self.destructure_ref().unexpected()
    }
}

impl<'de> MaybeUnexpected<'de> for DestructuredRef<'de> {
    fn unexpected(self) -> Unexpected<'de> {
        match self {
            Self::Null => Unexpected::Unit,
            Self::Bool(b) => Unexpected::Bool(b),
            Self::Number(v) => v.unexpected(),
            Self::String(v) => v.unexpected(),
            Self::Array(v) => v.unexpected(),
            Self::Object(v) => v.unexpected(),
        }
    }
}

impl<'de> MaybeUnexpected<'de> for &'de INumber {
    fn unexpected(self) -> Unexpected<'de> {
        // Partial accessors, exactly as in `deserialize_any` above — and the same
        // fallback, so the number serde reports in an error message is the one it would
        // have been given.
        if !self.has_decimal_point() {
            if let Some(v) = self.to_i64() {
                return Unexpected::Signed(v);
            }
            if let Some(v) = self.to_u64() {
                return Unexpected::Unsigned(v);
            }
        }
        Unexpected::Float(self.to_f64_lossy())
    }
}

impl<'de> MaybeUnexpected<'de> for &'de IString {
    fn unexpected(self) -> Unexpected<'de> {
        Unexpected::Str(self.as_str())
    }
}

impl<'de> MaybeUnexpected<'de> for &'de IArray {
    fn unexpected(self) -> Unexpected<'de> {
        Unexpected::Seq
    }
}

impl<'de> MaybeUnexpected<'de> for &'de IObject {
    fn unexpected(self) -> Unexpected<'de> {
        Unexpected::Map
    }
}

struct EnumDeserializer<'de> {
    variant: &'de IString,
    value: Option<&'de IValue>,
}

impl<'de> EnumAccess<'de> for EnumDeserializer<'de> {
    type Error = Error;
    type Variant = VariantDeserializer<'de>;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant), Error>
    where
        V: DeserializeSeed<'de>,
    {
        let variant = self.variant.into_deserializer();
        let visitor = VariantDeserializer { value: self.value };
        seed.deserialize(variant).map(|v| (v, visitor))
    }
}

impl<'de> IntoDeserializer<'de, Error> for &'de IString {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self::Deserializer {
        self
    }
}

struct VariantDeserializer<'de> {
    value: Option<&'de IValue>,
}

impl<'de> VariantAccess<'de> for VariantDeserializer<'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        if let Some(value) = self.value {
            Deserialize::deserialize(value)
        } else {
            Ok(())
        }
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if let Some(value) = self.value {
            seed.deserialize(value)
        } else {
            Err(SError::invalid_type(
                Unexpected::UnitVariant,
                &"newtype variant",
            ))
        }
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value.map(IValue::destructure_ref) {
            Some(DestructuredRef::Array(v)) => v.deserialize_any(visitor),
            Some(other) => Err(SError::invalid_type(other.unexpected(), &"tuple variant")),
            None => Err(SError::invalid_type(
                Unexpected::UnitVariant,
                &"tuple variant",
            )),
        }
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value.map(IValue::destructure_ref) {
            Some(DestructuredRef::Object(v)) => v.deserialize_any(visitor),
            Some(other) => Err(SError::invalid_type(other.unexpected(), &"struct variant")),
            None => Err(SError::invalid_type(
                Unexpected::UnitVariant,
                &"struct variant",
            )),
        }
    }
}

struct ArrayAccess<'de> {
    iter: slice::Iter<'de, IValue>,
}

impl<'de> ArrayAccess<'de> {
    fn new(slice: &'de [IValue]) -> Self {
        ArrayAccess { iter: slice.iter() }
    }
}

impl<'de> SeqAccess<'de> for ArrayAccess<'de> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.iter.next() {
            Some(value) => seed.deserialize(value).map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        match self.iter.size_hint() {
            (lower, Some(upper)) if lower == upper => Some(upper),
            _ => None,
        }
    }
}

struct ObjectAccess<'de> {
    iter: <&'de IObject as IntoIterator>::IntoIter,
    value: Option<&'de IValue>,
}

impl<'de> ObjectAccess<'de> {
    fn new(obj: &'de IObject) -> Self {
        ObjectAccess {
            iter: obj.into_iter(),
            value: None,
        }
    }
}

impl<'de> MapAccess<'de> for ObjectAccess<'de> {
    type Error = Error;

    fn next_key_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if let Some((key, value)) = self.iter.next() {
            self.value = Some(value);
            seed.deserialize(key).map(Some)
        } else {
            Ok(None)
        }
    }

    fn next_value_seed<T>(&mut self, seed: T) -> Result<T::Value, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if let Some(value) = self.value.take() {
            seed.deserialize(value)
        } else {
            Err(SError::custom("value is missing"))
        }
    }

    fn size_hint(&self) -> Option<usize> {
        match self.iter.size_hint() {
            (lower, Some(upper)) if lower == upper => Some(upper),
            _ => None,
        }
    }
}

/// Converts an [`IValue`] to an arbitrary type using that type's [`serde::Deserialize`]
/// implementation.
///
/// # Errors
///
/// Will return `Error` if `value` fails to deserialize.
pub fn from_value<'de, T>(value: &'de IValue) -> Result<T, Error>
where
    T: Deserialize<'de>,
{
    T::deserialize(value)
}

#[cfg(test)]
mod tests {
    use crate::{INumber, IValue};

    // The literals whose deserialization behaviour differs by feature. In every case the
    // point is that the *route in* does not change the value: `str::parse` and
    // `serde_json::from_str` are two doors to the same parser, so they must agree, and
    // whatever we serialize must read back unchanged.
    const CASES: &[&str] = &[
        "0",
        "-0",
        "42",
        "-42",
        "0.1",
        "0.5",
        "3.14159",
        "1.0",
        "1e3",
        "-2.5e-4",
        "0.0",
        "-0.0",
        // An exact `f64` whose *shortest* decimal (`441044444333116.1`) differs from its
        // exact value: `serialize_f64` would emit the shortest, which an exact reparse reads
        // as a different number. Must serialize its exact value to round-trip. (Fuzzer.)
        "441044444333116.125",
        // Beyond `i64`/`u64` — a big integer, kept exact under `arbitrary_precision`.
        "123456789012345678901234567890",
        "-123456789012345678901234567890",
        // More precision than an `f64` holds.
        "0.12345678901234567890123456789",
        // Beyond `f64`'s range entirely.
        "1e400",
        "-1e400",
        "1e-400",
    ];

    /// The single most important invariant this whole path exists to hold: text becomes the
    /// same number whichever door it comes through. `serde_json::from_str` and `str::parse`
    /// share one parser now, so they cannot disagree — in *either* feature configuration
    /// (without `arbitrary_precision` both round to `f64`, and both reject an out-of-range
    /// magnitude; with it, both keep the exact value).
    #[test]
    fn the_two_parsers_agree() {
        for &s in CASES {
            let via_parse = s.parse::<INumber>();
            let via_serde = serde_json::from_str::<INumber>(s);
            match (via_parse, via_serde) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a, b, "{:?}: parse and serde gave different numbers", s);

                    // Value always agrees; shape agrees too, save one long-standing
                    // exception that has nothing to do with this feature — the JSON token
                    // `-0`. `INumber::from_str` reads it by the grammar, as the *integer*
                    // `0` (no decimal point); `serde_json` reads it as the float `-0.0`.
                    // They still denote the same number, so this is presentation, not value.
                    if s != "-0" {
                        assert_eq!(
                            a.has_decimal_point(),
                            b.has_decimal_point(),
                            "{:?}: parse and serde disagree on integer/float shape",
                            s
                        );
                    }
                }
                (Err(_), Err(_)) => {}
                (a, b) => panic!(
                    "{:?}: one parser accepted and the other rejected (parse ok={}, serde ok={})",
                    s,
                    a.is_ok(),
                    b.is_ok()
                ),
            }
        }
    }

    /// Anything we can hold, we can round-trip through JSON text and through `serde_json`
    /// itself — same value, same integer/float shape. This is what would break if the write
    /// side (`Serialize`) and the read side (`Deserialize`) ever stopped sharing a parser:
    /// we could emit JSON we could not read back, or read it back as a different number.
    #[test]
    fn round_trips_through_serde_json() {
        for &s in CASES {
            let Ok(n) = s.parse::<INumber>() else {
                // Rejected here must mean rejected by serde too (checked above); nothing to
                // round-trip.
                continue;
            };
            let text = serde_json::to_string(&n).unwrap();
            let back = serde_json::from_str::<INumber>(&text).unwrap_or_else(|e| {
                panic!(
                    "{:?} serialized to {:?}, which serde cannot read: {}",
                    s, text, e
                )
            });
            assert_eq!(back, n, "{:?} -> {:?} -> a different number", s, text);
            assert_eq!(
                back.has_decimal_point(),
                n.has_decimal_point(),
                "{:?} -> {:?}: shape",
                s,
                text
            );

            // The same, wrapped in a container, so the object/array visitors are exercised
            // too — the number-in-a-map interception must not disturb real object parsing.
            let doc = format!("{{\"k\":[{}]}}", text);
            let v: IValue = serde_json::from_str(&doc).unwrap();
            let inner = &v.as_object().unwrap()["k"].as_array().unwrap()[0];
            assert_eq!(inner.as_number().unwrap(), &n, "{:?} inside a container", s);
        }
    }

    /// A genuine object whose key happens to look nothing like the number token still
    /// deserializes as an object — and one *couldn't* collide, but check a normal one works.
    #[test]
    fn objects_still_deserialize() {
        let v: IValue = serde_json::from_str(r#"{"a":1,"b":[2,3],"c":{"d":0.5}}"#).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 3);
        assert_eq!(obj["a"].to_i64(), Some(1));
        assert_eq!(obj["b"].as_array().unwrap().len(), 2);
        assert_eq!(obj["c"].as_object().unwrap()["d"].to_f64_lossy(), Some(0.5));
    }
}
