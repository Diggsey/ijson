//! Fuzzes the typed `to_value` / `from_value` path: an arbitrary
//! `#[derive(Serialize, Deserialize)]` value must survive a round trip through
//! `IValue` unchanged.
//!
//! The `json` target only ever drives the *dynamic* serde path — `IValue`'s own
//! `Serialize`/`Deserialize`, through `serde_json`. This one drives the *typed* machinery
//! instead: `ValueSerializer`, `ObjectKeySerializer`, and the `Deserializer for &IValue` with
//! its enum/variant/map/seq access, which only a derived type reaches. The `Arbitrary` derive
//! lets the fuzzer build the value directly, so every field kind and every enum-variant shape
//! is explored without first passing through JSON text.
//!
//! The payload's types are chosen so the round trip is *exact*:
//!
//!   - no floats — a non-finite `f64` is not representable in an `IValue` and would not
//!     survive the trip; finite floats are already fuzzed by the `number_*` targets;
//!   - only string-keyed maps — ijson serializes an integer map key to its string form but
//!     does not parse it back out, so an integer-keyed map would not round-trip;
//!   - map *values* are integers, and every single-key object is an enum variant, so the
//!     `arbitrary_precision` number-token interception (which needs a one-key object whose
//!     value is a *string* that parses as a number) can never misfire here — the target is
//!     correct under both feature configurations.
#![no_main]

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use serde::{Deserialize, Serialize};

/// All four enum-variant encodings: a bare string, and single-key objects wrapping a value,
/// an array, and an object respectively.
#[derive(Arbitrary, Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Variant {
    Unit,
    Newtype(i32),
    Tuple(i8, bool),
    Struct { a: u16, b: String },
}

#[derive(Arbitrary, Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Inner {
    n: i64,
    text: String,
    flag: bool,
}

/// A field of every integer width, plus bool/char/string, an option, a tuple, a sequence, a
/// nested struct, a sequence of every enum shape, and a string-keyed map — one value that
/// touches the whole typed serde surface.
#[derive(Arbitrary, Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Payload {
    i8_: i8,
    i16_: i16,
    i32_: i32,
    i64_: i64,
    u8_: u8,
    u16_: u16,
    u32_: u32,
    u64_: u64,
    b: bool,
    c: char,
    s: String,
    opt: Option<i32>,
    tuple: (i32, bool),
    seq: Vec<i16>,
    nested: Inner,
    variants: Vec<Variant>,
    map: BTreeMap<String, i32>,
}

fuzz_target!(|input: Payload| {
    // Serializing any value we can build must succeed...
    let value = ijson::to_value(&input).expect("to_value on a derived value");
    // ...and deserializing it straight back must recover exactly the same value.
    let back: Payload = ijson::from_value(&value).expect("from_value back into the same type");
    assert_eq!(input, back, "round-trip through IValue changed the value");
});
