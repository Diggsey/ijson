//! Round-trips `#[derive(Serialize, Deserialize)]` types through `ijson::to_value` and
//! `ijson::from_value`.
//!
//! These two functions are a public API (`ijson::to_value` / `ijson::from_value`), but the
//! rest of the suite only ever drives the *dynamic* serde path: serializing a
//! `serde_json::Value` or deserializing back into one both go through `serialize_map` /
//! `deserialize_any` and nothing else. The typed methods — `serialize_struct`, the four
//! `serialize_*_variant`s, `ObjectKeySerializer`, and on the read side `deserialize_struct` /
//! `deserialize_enum` / the typed `deserialize_i8..f64` / `EnumDeserializer` / `ArrayAccess` /
//! `ObjectAccess` — only fire when a *derived* type drives the (de)serializer. A `derive` is
//! the only thing that exercises them, so that is what this test is for.
//!
//! Every value here is chosen to be exactly representable in *both* number features (integers
//! and halving decimals like `0.5`/`0.25`), so a round trip is exact whether an inline number
//! is a binary `f64` (default) or an exact decimal (`arbitrary_precision`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A unit struct: exercises `serialize_unit_struct` / `deserialize_unit_struct`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Unit;

/// A newtype struct: exercises `serialize_newtype_struct` / `deserialize_newtype_struct`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Wrapper(u64);

/// A tuple struct: exercises `serialize_tuple_struct` and the seq read path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Pair(i8, u16);

/// A nested named-field struct, so `serialize_struct` / `deserialize_struct` recurse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Inner {
    flag: bool,
    name: String,
    ch: char,
}

/// One enum with all four variant shapes, so every `serialize_*_variant` and the matching
/// `VariantDeserializer` arm is driven by round-tripping a value of each.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Shape {
    Unit,
    Newtype(i32),
    Tuple(i32, String),
    Struct { x: f64, y: f64 },
}

/// A field of every scalar width, plus options, tuples, sequences, a string-keyed map and the
/// nested types above — one value that touches the whole typed serde surface at once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Everything {
    i8_: i8,
    i16_: i16,
    i32_: i32,
    i64_: i64,
    u8_: u8,
    u16_: u16,
    u32_: u32,
    u64_: u64,
    f32_: f32,
    f64_: f64,
    b: bool,
    s: String,
    c: char,
    opt_some: Option<i32>,
    opt_none: Option<i32>,
    unit: Unit,
    newtype: Wrapper,
    tuple_struct: Pair,
    tuple: (i32, String, bool),
    seq: Vec<i32>,
    nested: Inner,
    variants: Vec<Shape>,
    map: BTreeMap<String, i32>,
}

fn sample() -> Everything {
    Everything {
        i8_: -8,
        i16_: -1600,
        i32_: -320_000,
        i64_: -6_400_000_000,
        u8_: 8,
        u16_: 1600,
        u32_: 320_000,
        u64_: 6_400_000_000,
        f32_: 1.5,
        f64_: -2.25,
        b: true,
        s: "hello".to_owned(),
        c: 'λ',
        opt_some: Some(7),
        opt_none: None,
        unit: Unit,
        newtype: Wrapper(42),
        tuple_struct: Pair(-3, 9),
        tuple: (1, "two".to_owned(), false),
        seq: vec![1, 2, 3, 4],
        nested: Inner {
            flag: false,
            name: "inner".to_owned(),
            ch: 'x',
        },
        variants: vec![
            Shape::Unit,
            Shape::Newtype(-5),
            Shape::Tuple(6, "t".to_owned()),
            Shape::Struct { x: 0.5, y: -0.25 },
        ],
        map: BTreeMap::from([("a".to_owned(), 1), ("b".to_owned(), 2)]),
    }
}

/// The core invariant: a derived value survives `to_value` → `IValue` → `from_value` byte for
/// byte. This alone drives nearly every typed method on both `ValueSerializer` and the
/// `Deserializer for &IValue`.
#[test]
fn round_trips_a_derived_value() {
    let original = sample();
    let value = ijson::to_value(&original).expect("serialize into IValue");
    let restored: Everything = ijson::from_value(&value).expect("deserialize from IValue");
    assert_eq!(original, restored);
}

/// Each enum variant shape on its own, so a failure names which one — the aggregate test above
/// would only say "the whole struct differs".
#[test]
fn round_trips_each_enum_variant() {
    for variant in [
        Shape::Unit,
        Shape::Newtype(-5),
        Shape::Tuple(6, "t".to_owned()),
        Shape::Struct { x: 0.5, y: -0.25 },
    ] {
        let value = ijson::to_value(&variant).expect("serialize variant");
        let restored: Shape = ijson::from_value(&value).expect("deserialize variant");
        assert_eq!(variant, restored);
    }
}

/// A unit enum variant serializes as a bare string; a data variant as a single-key object.
/// Pin that shape, since the deserializer's `deserialize_enum` tells the two apart by it.
#[test]
fn enum_encoding_shape() {
    let unit = ijson::to_value(Shape::Unit).unwrap();
    assert_eq!(unit.as_string().map(|s| s.as_str()), Some("Unit"));

    let newtype = ijson::to_value(Shape::Newtype(9)).unwrap();
    let obj = newtype.as_object().expect("data variant is an object");
    assert_eq!(obj.len(), 1);
    assert_eq!(obj["Newtype"].to_i64(), Some(9));
}

/// Integer and char map keys go through `ObjectKeySerializer`'s numeric/char arms, which turn
/// them into string keys. (Only the write side: serde's own integer-key *reading* parses the
/// string back, which this crate's `IString` deserializer does not do, so this is a one-way
/// check of the key serializer, not a round trip.)
#[test]
fn non_string_map_keys_serialize_as_strings() {
    let int_keyed: BTreeMap<u32, i32> = BTreeMap::from([(1, 10), (2, 20)]);
    let value = ijson::to_value(&int_keyed).unwrap();
    let obj = value.as_object().expect("map serializes as object");
    assert_eq!(obj["1"].to_i64(), Some(10));
    assert_eq!(obj["2"].to_i64(), Some(20));

    let char_keyed: BTreeMap<char, i32> = BTreeMap::from([('a', 1)]);
    let value = ijson::to_value(&char_keyed).unwrap();
    assert_eq!(value.as_object().unwrap()["a"].to_i64(), Some(1));
}

/// A key that is neither a string nor an integer has no representation as a JSON object key,
/// and the key serializer must reject it rather than invent one.
#[test]
fn unrepresentable_map_key_is_rejected() {
    let bad: BTreeMap<bool, i32> = BTreeMap::from([(true, 1)]);
    assert!(ijson::to_value(&bad).is_err());
}

/// Unknown object fields are skipped via `deserialize_ignored_any`, which nothing else here
/// reaches: deserialize a struct from an object carrying an extra key.
#[test]
fn extra_object_fields_are_ignored() {
    let mut obj = ijson::to_value(&Inner {
        flag: true,
        name: "n".to_owned(),
        ch: 'z',
    })
    .unwrap();
    obj.as_object_mut()
        .unwrap()
        .insert("unexpected", ijson::IValue::from(123));

    let restored: Inner = ijson::from_value(&obj).expect("extra field ignored");
    assert_eq!(restored.name, "n");
}

/// The error side: asking for a type the value cannot supply must fail with a type error, not
/// a wrong value. Each case drives a different `MaybeUnexpected::unexpected` arm and the
/// `else` branch of a typed `deserialize_*`.
#[test]
fn type_mismatches_are_errors() {
    let s = ijson::IValue::from("text");
    let n = ijson::IValue::from(5);
    let arr: ijson::IValue = ijson::IArray::new().into();

    assert!(ijson::from_value::<u32>(&s).is_err()); // number from string
    assert!(ijson::from_value::<String>(&n).is_err()); // string from number
    assert!(ijson::from_value::<bool>(&n).is_err()); // bool from number
    assert!(ijson::from_value::<Vec<i32>>(&n).is_err()); // seq from number
    assert!(ijson::from_value::<Inner>(&arr).is_err()); // struct from array (wrong length)
    assert!(ijson::from_value::<Inner>(&n).is_err()); // struct from a scalar
    assert!(ijson::from_value::<Shape>(&n).is_err()); // enum from number
}

/// A unit-only enum and a newtype struct, used only as map keys below.
#[derive(Serialize, PartialEq, Eq, PartialOrd, Ord)]
enum UnitKey {
    A,
    B,
}

#[derive(Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct IdKey(u32);

/// The map-key shapes that legitimately serialize to a string: integers of assorted widths, a
/// unit enum variant (its name), and a newtype struct (its inner value). These drive the
/// `ObjectKeySerializer` arms that a plain string key does not.
#[test]
fn serializable_map_key_shapes() {
    // Integer keys of several widths stringify. (One per signedness/size class — the arms are
    // identical per width, so a representative sample is the point, not an exhaustive one.)
    let signed = ijson::to_value(BTreeMap::from([(-8i8, 1), (8, 2)])).unwrap();
    assert_eq!(signed.as_object().unwrap()["-8"].to_i64(), Some(1));
    let wide = ijson::to_value(BTreeMap::from([(1i64 << 40, 1)])).unwrap();
    assert_eq!(wide.as_object().unwrap()["1099511627776"].to_i64(), Some(1));
    let unsigned = ijson::to_value(BTreeMap::from([(200u8, 1)])).unwrap();
    assert_eq!(unsigned.as_object().unwrap()["200"].to_i64(), Some(1));

    // A unit enum variant used as a key serializes to the variant name.
    let by_variant = ijson::to_value(BTreeMap::from([(UnitKey::A, 1), (UnitKey::B, 2)])).unwrap();
    let o = by_variant.as_object().unwrap();
    assert_eq!(o["A"].to_i64(), Some(1));
    assert_eq!(o["B"].to_i64(), Some(2));

    // A newtype struct key forwards to its inner value.
    let by_id = ijson::to_value(BTreeMap::from([(IdKey(7), 1)])).unwrap();
    assert_eq!(by_id.as_object().unwrap()["7"].to_i64(), Some(1));
}

/// Keys that have no string form must be rejected, across the shapes a key can take. (One
/// representative per shape; the remaining `ObjectKeySerializer` error arms are the same
/// one-line rejection for key types that cannot occur in practice.)
#[test]
fn non_scalar_map_keys_are_rejected() {
    assert!(ijson::to_value(BTreeMap::from([((1, 2), 0)])).is_err()); // tuple key
    assert!(ijson::to_value(BTreeMap::from([(vec![1, 2], 0)])).is_err()); // seq key
    assert!(ijson::to_value(BTreeMap::from([((), 0)])).is_err()); // unit key
    assert!(ijson::to_value(BTreeMap::from([(Some(1), 0)])).is_err()); // option Some key
    assert!(ijson::to_value(BTreeMap::from([(None::<i32>, 0)])).is_err()); // option None key
}
