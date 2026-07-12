//! Fuzzes `IValue` over arbitrary JSON documents — the whole value tree, objects
//! and arrays included, not just the numbers the other targets cover.
//!
//! For every input ijson deserializes as JSON it checks the value-level contracts
//! hold over the entire tree:
//!
//!   - `clone` produces an equal value that hashes the same;
//!   - ordering is reflexive and coherent with `==` (`v.partial_cmp(&v_clone) ==
//!     Some(Equal)`) — the coherence that must hold even for an object nested inside
//!     an array, a class of bug that is otherwise easy to miss;
//!   - a serialize/deserialize round-trip recovers an equal value (and hash);
//!   - and, whenever `serde_json` also accepts the input, ijson round-trips it to the
//!     same `serde_json::Value` as `serde_json` itself — cross-checking the
//!     (de)serializers against the reference implementation.
#![no_main]

use ijson::IValue;
use libfuzzer_sys::fuzz_target;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hash(v: &IValue) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

fuzz_target!(|data: &str| {
    let ours: IValue = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Cloning yields an equal value in a fresh allocation that hashes identically,
    // and orders `Equal` against the original. The distinct-allocation `Equal` is
    // exactly the `PartialOrd`/`PartialEq` coherence that objects (and arrays of
    // objects) have to satisfy.
    let clone = ours.clone();
    assert!(ours == clone, "value != its own clone: {:?}", data);
    assert_eq!(hash(&ours), hash(&clone), "clone hashes differently: {:?}", data);
    assert_eq!(
        ours.partial_cmp(&clone),
        Some(Ordering::Equal),
        "clone does not compare `Equal`: {:?}",
        data
    );

    // Serialize -> deserialize round-trips to an equal value.
    let text = serde_json::to_string(&ours).expect("serialize IValue");
    let reparsed: IValue = serde_json::from_str(&text).expect("reparse ijson output");
    assert!(ours == reparsed, "round-trip changed value: {:?} -> {}", data, text);
    assert_eq!(hash(&ours), hash(&reparsed), "round-trip changed hash: {:?}", data);

    // When `serde_json` also accepts the input, ijson must agree with it on the value.
    // Comparing `serde_json::Value` to `serde_json::Value` (ijson's output reparsed
    // vs. serde_json's own parse) avoids comparing numbers across two different models
    // and the f64-overflow boundary where the two float parsers can disagree.
    if let Ok(theirs) = serde_json::from_str::<serde_json::Value>(data) {
        let ours_as_serde: serde_json::Value =
            serde_json::from_str(&text).expect("serde_json rejects ijson's own output");
        assert_eq!(ours_as_serde, theirs, "ijson and serde_json disagree on {:?}", data);
    }
});
