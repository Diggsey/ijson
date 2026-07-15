//! Exercises the serde bridge on `IValue` and its subtypes directly (as opposed to via a
//! derived user type, which `serde_derive.rs` covers): serializing an `IValue` to JSON text,
//! deserializing each JSON type back into `IValue`/subtypes, using an `IValue` *as* a
//! deserializer (`from_value`) for every type, and the enum / type-mismatch error paths.

use std::collections::BTreeMap;

use ijson::{ijson, IArray, INumber, IObject, IString, IValue};

/// A document covering all six JSON types, serialized through `serde_json`. This drives every
/// arm of `Serialize for IValue` plus the `Serialize` impls of `IString`/`IArray`/`IObject`/
/// `INumber`, and round-trips back to an equal value.
#[test]
fn ivalue_serializes_to_json_text() {
    let v = ijson!({
        "null": null,
        "bool": true,
        "int": 42,
        "float": 1.5,
        "str": "hello",
        "arr": [1, "two", false, null],
        "obj": { "nested": 7 }
    });
    let text = serde_json::to_string(&v).unwrap();
    let back: IValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v, back);

    // The subtypes serialize directly too.
    assert_eq!(
        serde_json::to_string(&IString::intern("x")).unwrap(),
        r#""x""#
    );
    let arr: IArray = vec![IValue::from(1), IValue::from(2)].into();
    assert_eq!(serde_json::to_string(&arr).unwrap(), "[1,2]");
    let mut obj = IObject::new();
    obj.insert("k", IValue::from(1));
    assert_eq!(serde_json::to_string(&obj).unwrap(), r#"{"k":1}"#);
    assert_eq!(serde_json::to_string(&INumber::from(-5)).unwrap(), "-5");
}

/// Each subtype's `Deserialize` impl, including the escaped-string path that forces serde_json
/// to hand over an owned `String` (`visit_string`), and the `Visitor::expecting` message each
/// produces on a type mismatch.
#[test]
fn subtypes_deserialize_from_json_text() {
    let s: IString = serde_json::from_str(r#""a\tb""#).unwrap(); // escape -> visit_string
    assert_eq!(s.as_str(), "a\tb");
    assert_eq!(
        serde_json::from_str::<INumber>("42").unwrap(),
        INumber::from(42)
    );
    assert_eq!(serde_json::from_str::<IArray>("[1,2,3]").unwrap().len(), 3);
    let obj: IObject = serde_json::from_str(r#"{"a":1}"#).unwrap();
    assert_eq!(obj["a"], IValue::from(1));

    // A type mismatch triggers each visitor's `expecting`.
    assert!(serde_json::from_str::<IString>("123").is_err());
    assert!(serde_json::from_str::<INumber>(r#""x""#).is_err());
    assert!(serde_json::from_str::<IArray>("123").is_err());
    assert!(serde_json::from_str::<IObject>("123").is_err());
}

/// Using an `IValue` *as* a serde `Deserializer` (`from_value`) for every JSON type: drives the
/// `deserialize_any` arm of each, `ArrayAccess`/`ObjectAccess`, and the number sub-deserializer.
#[test]
fn ivalue_as_deserializer_for_every_type() {
    let cases = vec![
        IValue::NULL,
        IValue::from(true),
        IValue::from(42),
        IValue::from(1.5),
        IValue::from("s"),
        ijson!([1, 2]),
        ijson!({ "a": 1 }),
    ];
    for v in cases {
        let back: serde_json::Value = ijson::from_value(&v).unwrap();
        // The value survives the conversion into serde_json's model and back.
        assert_eq!(v, IValue::from(back));
    }

    // Typed extraction, not just serde_json::Value.
    assert_eq!(ijson::from_value::<i64>(&IValue::from(7)).unwrap(), 7);
    assert_eq!(
        ijson::from_value::<String>(&IValue::from("hi")).unwrap(),
        "hi"
    );
    assert_eq!(
        ijson::from_value::<Vec<i32>>(&ijson!([1, 2, 3])).unwrap(),
        vec![1, 2, 3]
    );
    let map: BTreeMap<String, i32> = ijson::from_value(&ijson!({ "a": 1, "b": 2 })).unwrap();
    assert_eq!(map["b"], 2);
}

#[derive(Debug, PartialEq, serde::Deserialize)]
enum Shape {
    Unit,
    Newtype(i32),
    Tuple(i32, i32),
    Struct { x: i32 },
}

/// The enum-deserialization error surface: `deserialize_enum` needs a string or single-key
/// object, and each `VariantDeserializer` arm rejects a payload of the wrong shape (or a
/// missing one).
#[test]
fn enum_deserialize_error_paths() {
    // A number is neither a string nor a single-key object.
    assert!(ijson::from_value::<Shape>(&IValue::from(5)).is_err());

    // A single-key object is a data variant; the payload shape must match the variant.
    assert!(ijson::from_value::<Shape>(&ijson!({ "Tuple": 5 })).is_err()); // tuple needs array
    assert!(ijson::from_value::<Shape>(&ijson!({ "Struct": [1, 2] })).is_err()); // struct needs object

    // More than one key is not a valid enum encoding.
    assert!(ijson::from_value::<Shape>(&ijson!({ "Unit": null, "extra": 1 })).is_err());

    // A bare string is a unit-variant encoding: reading it as a data variant fails (there is
    // no payload), while reading it as the unit variant succeeds.
    assert!(ijson::from_value::<Shape>(&IValue::from("Newtype")).is_err());
    assert!(ijson::from_value::<Shape>(&IValue::from("Tuple")).is_err());
    assert!(ijson::from_value::<Shape>(&IValue::from("Struct")).is_err());
    assert_eq!(
        ijson::from_value::<Shape>(&IValue::from("Unit")).unwrap(),
        Shape::Unit
    );
    // A unit variant encoded as a single-key object also works (the value is consumed as `()`).
    assert_eq!(
        ijson::from_value::<Shape>(&ijson!({ "Unit": null })).unwrap(),
        Shape::Unit
    );

    // Every data variant round-trips from its correct encoding.
    assert_eq!(
        ijson::from_value::<Shape>(&ijson!({ "Newtype": 9 })).unwrap(),
        Shape::Newtype(9)
    );
    assert_eq!(
        ijson::from_value::<Shape>(&ijson!({ "Tuple": [1, 2] })).unwrap(),
        Shape::Tuple(1, 2)
    );
    assert_eq!(
        ijson::from_value::<Shape>(&ijson!({ "Struct": { "x": 3 } })).unwrap(),
        Shape::Struct { x: 3 }
    );
}

/// Type-mismatch errors report the value they *found*, driving each `MaybeUnexpected`
/// (`unexpected`) arm — the `Unexpected` value serde puts in the error message.
#[test]
fn type_mismatch_reports_the_found_type() {
    // Number wanted, but the value is of each other type in turn.
    assert!(ijson::from_value::<i32>(&IValue::NULL).is_err()); // Unexpected::Unit
    assert!(ijson::from_value::<i32>(&IValue::from(true)).is_err()); // Unexpected::Bool
    assert!(ijson::from_value::<i32>(&IValue::from("s")).is_err()); // Unexpected::Str
    assert!(ijson::from_value::<i32>(&ijson!([1])).is_err()); // Unexpected::Seq
    assert!(ijson::from_value::<i32>(&ijson!({ "a": 1 })).is_err()); // Unexpected::Map

    // String wanted, number found: Unexpected::Signed / Unsigned / Float.
    assert!(ijson::from_value::<String>(&IValue::from(-1)).is_err());
    assert!(ijson::from_value::<String>(&IValue::from(1u64 << 63)).is_err());
    assert!(ijson::from_value::<String>(&IValue::from(1.5)).is_err());

    // A map requested from a scalar takes the `deserialize_map` mismatch branch, and a unit
    // requested from a non-null takes the `deserialize_unit` one.
    assert!(ijson::from_value::<BTreeMap<String, i32>>(&IValue::from(5)).is_err());
    assert!(ijson::from_value::<()>(&IValue::from(5)).is_err());
}
