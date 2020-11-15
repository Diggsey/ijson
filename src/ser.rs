use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

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
