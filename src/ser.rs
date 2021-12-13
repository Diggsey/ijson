use serde::ser::{
    Error as _, Impossible, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant,
};
use serde::{Serialize, Serializer};
use serde_json::error::Error;

use super::array::IArray;
use super::number::INumber;
use super::object::IObject;
use super::string::IString;
use super::value::{DestructuredRef, IValue};

impl Serialize for IValue {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.destructure_ref() {
            DestructuredRef::Null => serializer.serialize_unit(),
            DestructuredRef::Bool(b) => serializer.serialize_bool(b),
            DestructuredRef::Number(n) => n.serialize(serializer),
            DestructuredRef::String(s) => s.serialize(serializer),
            DestructuredRef::Array(v) => v.serialize(serializer),
            DestructuredRef::Object(o) => o.serialize(serializer),
        }
    }
}

impl Serialize for INumber {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.has_decimal_point() {
            serializer.serialize_f64(self.to_f64().unwrap())
        } else if let Some(v) = self.to_i64() {
            serializer.serialize_i64(v)
        } else {
            serializer.serialize_u64(self.to_u64().unwrap())
        }
    }
}

impl Serialize for IString {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl Serialize for IArray {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_seq(Some(self.len()))?;
        for v in self {
            s.serialize_element(v)?;
        }
        s.end()
    }
}

impl Serialize for IObject {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_map(Some(self.len()))?;
        for (k, v) in self {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

pub struct ValueSerializer;

impl Serializer for ValueSerializer {
    type Ok = IValue;
    type Error = Error;

    type SerializeSeq = SerializeArray;
    type SerializeTuple = SerializeArray;
    type SerializeTupleStruct = SerializeArray;
    type SerializeTupleVariant = SerializeArrayVariant;
    type SerializeMap = SerializeObject;
    type SerializeStruct = SerializeObject;
    type SerializeStructVariant = SerializeObjectVariant;

    #[inline]
    fn serialize_bool(self, value: bool) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_i8(self, value: i8) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_i16(self, value: i16) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_i32(self, value: i32) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    fn serialize_i64(self, value: i64) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_u8(self, value: u8) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_u16(self, value: u16) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_u32(self, value: u32) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_u64(self, value: u64) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_f32(self, value: f32) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_f64(self, value: f64) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    #[inline]
    fn serialize_char(self, value: char) -> Result<IValue, Self::Error> {
        let mut buffer = [0_u8; 4];
        Ok(value.encode_utf8(&mut buffer).into())
    }

    #[inline]
    fn serialize_str(self, value: &str) -> Result<IValue, Self::Error> {
        Ok(value.into())
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<IValue, Self::Error> {
        let array: IArray = value.iter().copied().collect();
        Ok(array.into())
    }

    #[inline]
    fn serialize_unit(self) -> Result<IValue, Self::Error> {
        Ok(IValue::NULL)
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<IValue, Self::Error> {
        self.serialize_unit()
    }

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<IValue, Self::Error> {
        self.serialize_str(variant)
    }

    #[inline]
    fn serialize_newtype_struct<T>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<IValue, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<IValue, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        let mut obj = IObject::new();
        obj.insert(variant, value.serialize(self)?);
        Ok(obj.into())
    }

    #[inline]
    fn serialize_none(self) -> Result<IValue, Self::Error> {
        self.serialize_unit()
    }

    #[inline]
    fn serialize_some<T>(self, value: &T) -> Result<IValue, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Ok(SerializeArray {
            array: IArray::with_capacity(len.unwrap_or(0)),
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Ok(SerializeArrayVariant {
            name: variant.into(),
            array: IArray::with_capacity(len),
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Ok(SerializeObject {
            object: IObject::with_capacity(len.unwrap_or(0)),
            next_key: None,
        })
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.serialize_map(Some(len))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Ok(SerializeObjectVariant {
            name: variant.into(),
            object: IObject::with_capacity(len),
        })
    }
}

pub struct SerializeArray {
    array: IArray,
}

pub struct SerializeArrayVariant {
    name: IString,
    array: IArray,
}

pub struct SerializeObject {
    object: IObject,
    next_key: Option<IString>,
}

pub struct SerializeObjectVariant {
    name: IString,
    object: IObject,
}

impl SerializeSeq for SerializeArray {
    type Ok = IValue;
    type Error = Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.array.push(value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<IValue, Self::Error> {
        Ok(self.array.into())
    }
}

impl SerializeTuple for SerializeArray {
    type Ok = IValue;
    type Error = Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<IValue, Self::Error> {
        SerializeSeq::end(self)
    }
}

impl SerializeTupleStruct for SerializeArray {
    type Ok = IValue;
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<IValue, Self::Error> {
        SerializeSeq::end(self)
    }
}

impl SerializeTupleVariant for SerializeArrayVariant {
    type Ok = IValue;
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.array.push(value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<IValue, Self::Error> {
        let mut object = IObject::new();
        object.insert(self.name, self.array);

        Ok(object.into())
    }
}

impl SerializeMap for SerializeObject {
    type Ok = IValue;
    type Error = Error;

    fn serialize_key<T>(&mut self, key: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.next_key = Some(key.serialize(ObjectKeySerializer)?);
        Ok(())
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        // Panic because this indicates a bug in the program rather than an
        // expected failure.
        let key = self
            .next_key
            .take()
            .expect("serialize_value called before serialize_key");
        self.object.insert(key, value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<IValue, Self::Error> {
        Ok(self.object.into())
    }
}

struct ObjectKeySerializer;

fn key_must_be_a_string() -> Error {
    Error::custom("Object key must be a string")
}

impl Serializer for ObjectKeySerializer {
    type Ok = IString;
    type Error = Error;

    type SerializeSeq = Impossible<IString, Error>;
    type SerializeTuple = Impossible<IString, Error>;
    type SerializeTupleStruct = Impossible<IString, Error>;
    type SerializeTupleVariant = Impossible<IString, Error>;
    type SerializeMap = Impossible<IString, Error>;
    type SerializeStruct = Impossible<IString, Error>;
    type SerializeStructVariant = Impossible<IString, Error>;

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<IString, Self::Error> {
        Ok(variant.into())
    }

    #[inline]
    fn serialize_newtype_struct<T>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<IString, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_bool(self, _value: bool) -> Result<IString, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_i8(self, value: i8) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_i16(self, value: i16) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_i32(self, value: i32) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_i64(self, value: i64) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_u8(self, value: u8) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_u16(self, value: u16) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_u32(self, value: u32) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_u64(self, value: u64) -> Result<IString, Self::Error> {
        Ok(value.to_string().into())
    }

    fn serialize_f32(self, _value: f32) -> Result<IString, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_f64(self, _value: f64) -> Result<IString, Self::Error> {
        Err(key_must_be_a_string())
    }

    #[inline]
    fn serialize_char(self, value: char) -> Result<IString, Self::Error> {
        let mut buffer = [0_u8; 4];
        Ok(value.encode_utf8(&mut buffer).into())
    }

    #[inline]
    fn serialize_str(self, value: &str) -> Result<IString, Self::Error> {
        Ok(value.into())
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<IString, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_unit(self) -> Result<IString, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<IString, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<IString, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        Err(key_must_be_a_string())
    }

    fn serialize_none(self) -> Result<IString, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_some<T>(self, _value: &T) -> Result<IString, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        Err(key_must_be_a_string())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err(key_must_be_a_string())
    }
}

impl SerializeStruct for SerializeObject {
    type Ok = IValue;
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        SerializeMap::serialize_entry(self, key, value)
    }

    fn end(self) -> Result<IValue, Self::Error> {
        SerializeMap::end(self)
    }
}

impl SerializeStructVariant for SerializeObjectVariant {
    type Ok = IValue;
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.object.insert(key, value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<IValue, Self::Error> {
        let mut object = IObject::new();
        object.insert(self.name, self.object);
        Ok(object.into())
    }
}

/// Converts an arbitrary type to an [`IValue`] using that type's [`serde::Serialize`]
/// implementation.
/// # Errors
/// 
/// Will return `Error` if `value` fails to serialize.
pub fn to_value<T>(value: T) -> Result<IValue, Error>
where
    T: Serialize,
{
    value.serialize(ValueSerializer)
}
