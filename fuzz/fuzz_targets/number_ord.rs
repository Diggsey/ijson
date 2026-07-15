//! Fuzzes the comparison and hashing of numbers reached through *different*
//! constructors (inline/heap integers, inline/heap floats, and the string
//! parser), which is exactly where cross-representation bugs hide.
//!
//! Over every pair and triple drawn from an arbitrary set of numbers it checks
//! the ordering is a consistent total order (antisymmetry, transitivity, and
//! agreement between `==` and `cmp == Equal`) and that the `Hash`/`Eq` contract
//! holds (equal values hash equally).
#![no_main]

use arbitrary::Arbitrary;
use ijson::{INumber, IValue};
use libfuzzer_sys::fuzz_target;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A number reached through one of ijson's constructors. Letting the fuzzer pick
/// both the constructor and the value stresses every representation and the
/// cross-representation comparisons between them.
#[derive(Arbitrary, Debug)]
enum NumSource {
    I64(i64),
    U64(u64),
    /// Raw bits reinterpreted as an `f64`.
    F64(u64),
    /// An arbitrary string parsed through `INumber`'s `FromStr`.
    Str(String),
}

fn build(src: &NumSource) -> Option<IValue> {
    match src {
        NumSource::I64(x) => Some(IValue::from(*x)),
        NumSource::U64(x) => Some(IValue::from(*x)),
        NumSource::F64(bits) => {
            let f = f64::from_bits(*bits);
            f.is_finite().then(|| IValue::from(f))
        }
        NumSource::Str(s) => s.parse::<INumber>().ok().map(IValue::from),
    }
}

fn hash(v: &IValue) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

fuzz_target!(|srcs: (NumSource, NumSource, NumSource)| {
    let nums: Vec<IValue> = [&srcs.0, &srcs.1, &srcs.2]
        .iter()
        .filter_map(|s| build(s))
        .collect();

    for a in &nums {
        assert!(a.is_number());
        for b in &nums {
            let ab = a.partial_cmp(b).expect("two numbers always order");
            let ba = b.partial_cmp(a).expect("two numbers always order");
            // Antisymmetry.
            assert_eq!(ab, ba.reverse(), "antisymmetry: {:?} vs {:?}", a, b);
            // `==` agrees with `cmp == Equal`.
            assert_eq!(a == b, ab == Ordering::Equal, "eq/cmp: {:?} vs {:?}", a, b);
            // Hash/Eq contract: equal values hash equally.
            if a == b {
                assert_eq!(hash(a), hash(b), "equal but unequal hash: {:?} vs {:?}", a, b);
            }
        }
    }

    // Transitivity of the ordering over the (up to three) values.
    for a in &nums {
        for b in &nums {
            for c in &nums {
                if a <= b && b <= c {
                    assert!(a <= c, "transitivity: {:?} <= {:?} <= {:?}", a, b, c);
                }
            }
        }
    }
});
