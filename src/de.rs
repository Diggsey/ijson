use std::convert::TryFrom;
use std::fmt::{self, Formatter};

use serde::de::{Error, MapAccess, SeqAccess, Unexpected, Visitor};
use serde::{Deserialize, Deserializer};

use super::array::IArray;
use super::number::INumber;
use super::object::IObject;
use super::string::IString;
use super::value::IValue;

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
    fn visit_bool<E: Error>(self, value: bool) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_i64<E: Error>(self, value: i64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_u64<E: Error>(self, value: u64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_f64<E: Error>(self, value: f64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_str<E: Error>(self, value: &str) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_string<E: Error>(self, value: String) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_none<E: Error>(self) -> Result<IValue, E> {
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
    fn visit_unit<E: Error>(self) -> Result<IValue, E> {
        Ok(IValue::NULL)
    }

    #[inline]
    fn visit_seq<V>(self, visitor: V) -> Result<IValue, V::Error>
    where
        V: SeqAccess<'de>,
    {
        ArrayVisitor.visit_seq(visitor).map(Into::into)
    }

    fn visit_map<V>(self, visitor: V) -> Result<IValue, V::Error>
    where
        V: MapAccess<'de>,
    {
        ObjectVisitor.visit_map(visitor).map(Into::into)
    }
}

struct NumberVisitor;

impl<'de> Visitor<'de> for NumberVisitor {
    type Value = INumber;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON number")
    }

    #[inline]
    fn visit_i64<E: Error>(self, value: i64) -> Result<INumber, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_u64<E: Error>(self, value: u64) -> Result<INumber, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_f64<E: Error>(self, value: f64) -> Result<INumber, E> {
        INumber::try_from(value).map_err(|_| Error::invalid_value(Unexpected::Float(value), &self))
    }
}

struct StringVisitor;

impl<'de> Visitor<'de> for StringVisitor {
    type Value = IString;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON string")
    }

    #[inline]
    fn visit_str<E: Error>(self, value: &str) -> Result<IString, E> {
        Ok(value.into())
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

    fn visit_map<V>(self, mut visitor: V) -> Result<IObject, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut obj = IObject::with_capacity(visitor.size_hint().unwrap_or(0));
        while let Some((k, v)) = visitor.next_entry::<IString, IValue>()? {
            obj.insert(k, v);
        }
        Ok(obj)
    }
}
