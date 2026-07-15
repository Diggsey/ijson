//! Fuzzes `f64` construction over the whole bit space (so NaN, infinities and
//! subnormals are all reachable).
//!
//! Every finite `f64` must be storable and round-trip *exactly* through the
//! `to_f64`/`to_f64_lossy` accessors, and any successful integer conversion must
//! recover the same value. Every non-finite `f64` must be rejected.
#![no_main]

use ijson::{INumber, IValue};
use libfuzzer_sys::fuzz_target;
use std::convert::TryFrom;

fuzz_target!(|bits: u64| {
    let x = f64::from_bits(bits);
    if x.is_finite() {
        let n = INumber::try_from(x).expect("a finite f64 is representable");
        // Exact round-trip. `0.0 == -0.0`, so canonicalizing -0.0 to +0.0 still
        // satisfies this.
        assert_eq!(n.to_f64(), Some(x), "{:e} to_f64", x);
        assert_eq!(n.to_f64_lossy(), x, "{:e} to_f64_lossy", x);
        // Any integer conversion that succeeds recovers the same numeric value.
        if let Some(i) = n.to_i64() {
            assert_eq!(i as f64, x, "{:e} to_i64 value", x);
        }
        if let Some(u) = n.to_u64() {
            assert_eq!(u as f64, x, "{:e} to_u64 value", x);
        }
    } else {
        assert!(INumber::try_from(x).is_err(), "{:e} accepted", x);
        assert!(IValue::from(x).is_null(), "{:e} not null", x);
    }
});
