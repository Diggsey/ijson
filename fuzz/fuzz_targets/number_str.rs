//! Fuzzes `INumber`'s `FromStr` parser over arbitrary strings.
//!
//! Every string that parses must be a number that round-trips through
//! serialization, agrees with `serde_json`'s acceptance (away from the f64
//! overflow boundary, where the two float parsers can disagree), and — when it is
//! an in-range integer — equals the directly-constructed integer.
#![no_main]

use ijson::{INumber, IValue};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    if let Ok(n) = data.parse::<INumber>() {
        let v = IValue::from(n);
        assert!(v.is_number(), "{:?} parsed but is not a number", data);

        // `INumber`'s parser accepts a strict subset of what `serde_json` accepts
        // as a JSON number (it additionally rejects surrounding whitespace), so
        // `serde_json` must accept anything it does. The one exception is right at
        // the f64 overflow boundary, where `std`'s and `serde_json`'s float
        // parsers can disagree on finiteness — so this is checked only for values
        // comfortably within range.
        if v.to_f64_lossy().map_or(false, |x| x.abs() < 1e308) {
            let via_serde: serde_json::Value = serde_json::from_str(data)
                .expect("from_str accepted a string serde_json rejects");
            assert!(via_serde.is_number(), "{:?} is not a serde number", data);
        }

        // Serializing and reparsing through the same parser reproduces the value.
        let out = serde_json::to_string(&v).expect("serialize");
        let re: INumber = out.parse().expect("reparse");
        assert_eq!(IValue::from(re), v, "{:?} round-trip via {}", data, out);

        // An in-range integer parses to exactly that integer.
        if let Ok(i) = data.parse::<i64>() {
            assert_eq!(v, IValue::from(i), "{:?} integer value", data);
        } else if let Ok(u) = data.parse::<u64>() {
            assert_eq!(v, IValue::from(u), "{:?} integer value", data);
        }
    }
});
