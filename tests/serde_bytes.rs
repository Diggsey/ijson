//! Covers the bytes paths of the serde bridge, which no JSON input reaches (JSON has no byte
//! type): `ValueSerializer::serialize_bytes` (bytes serialize as an array of byte values), the
//! `deserialize_bytes` / `deserialize_byte_buf` dispatch on `&IValue`, the
//! `ObjectKeySerializer` bytes-key rejection, and `StringVisitor`'s `visit_bytes` /
//! `visit_byte_buf` — the only way an `IString` is built from bytes, reached here with
//! byte-oriented deserializers.

use std::collections::BTreeMap;

use serde::de::value::BytesDeserializer;
use serde::de::{Deserializer, Visitor};
use serde::{forward_to_deserialize_any, Deserialize};

use ijson::{ijson, IString, IValue};

#[test]
fn bytes_serialize_as_an_array_and_round_trip() {
    // serde_bytes serializes through `serialize_bytes`, which ijson stores as an array of the
    // byte values.
    let value = ijson::to_value(serde_bytes::ByteBuf::from(vec![1u8, 2, 255])).unwrap();
    let arr = value.as_array().expect("bytes serialize as an array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[2], IValue::from(255));

    // `deserialize_byte_buf` reads them straight back (array -> bytes).
    let back: serde_bytes::ByteBuf = ijson::from_value(&value).unwrap();
    assert_eq!(back.as_ref(), [1u8, 2, 255]);
}

#[test]
fn byte_buf_from_string_and_type_error() {
    // A string deserialized as bytes yields its UTF-8 bytes.
    let back: serde_bytes::ByteBuf = ijson::from_value(&IValue::from("abc")).unwrap();
    assert_eq!(back.as_ref(), b"abc");

    // Anything that is neither a string nor an array is a type error.
    assert!(ijson::from_value::<serde_bytes::ByteBuf>(&IValue::from(5)).is_err());
    assert!(ijson::from_value::<serde_bytes::ByteBuf>(&ijson!({ "a": 1 })).is_err());
}

#[test]
fn bytes_map_key_is_rejected() {
    // A byte-string key has no JSON object-key form: `ObjectKeySerializer::serialize_bytes`
    // must reject it.
    let map = BTreeMap::from([(serde_bytes::ByteBuf::from(vec![1u8, 2]), 0)]);
    assert!(ijson::to_value(map).is_err());
}

/// A minimal deserializer that hands its payload to the visitor as an *owned* byte buffer, so
/// `StringVisitor::visit_byte_buf` (which no `serde_json` or `IValue` deserializer reaches) is
/// exercised.
struct ByteBufDeserializer(Vec<u8>);

impl<'de> Deserializer<'de> for ByteBufDeserializer {
    type Error = serde::de::value::Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_byte_buf(self.0)
    }
    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

#[test]
fn istring_from_bytes_visitors() {
    // `visit_bytes`: a borrowed byte slice that is valid UTF-8 becomes the string.
    let de = BytesDeserializer::<serde::de::value::Error>::new(b"hello");
    assert_eq!(IString::deserialize(de).unwrap().as_str(), "hello");

    // `visit_byte_buf`: an owned buffer, likewise.
    let s = IString::deserialize(ByteBufDeserializer(b"world".to_vec())).unwrap();
    assert_eq!(s.as_str(), "world");

    // Invalid UTF-8 is rejected on both paths.
    let bad = BytesDeserializer::<serde::de::value::Error>::new(&[0xff, 0xfe]);
    assert!(IString::deserialize(bad).is_err());
    assert!(IString::deserialize(ByteBufDeserializer(vec![0xff, 0xfe])).is_err());
}
