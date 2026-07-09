//! Shared edge-case inputs for exhaustive testing of the numeric code paths.
//!
//! Blind fuzzing tends to miss the values that actually matter, because the
//! interesting behaviour lives at the *discontinuities* in how a number is
//! represented. This module enumerates those discontinuities and provides one
//! list per input flavour (`i64`, `u64`, `f64`, and raw JSON strings) covering
//! each of them. A number is stored as `mantissa * 10^exp`:
//!
//! - **Inline mantissa fit** — an integer fits inline only if `|value| < 2^55`
//!   (64-bit) or `< 2^23` (32-bit); otherwise it spills to a heap scalar.
//! - **f64 exact-representability** — an integer is exactly an `f64` iff it has
//!   at most 53 significant bits (the `2^53` cliff); this gates `to_f64`.
//! - **Integer type limits** — `i8..i64`, `u8..u64`, `i32`/`u32`, and the
//!   `i64`/`u64` seam at `2^63` (`i64::MAX + 1`).
//! - **float→int range** — `to_i64` accepts `x < i64::MAX as f64` (`== 2^63`)
//!   and `to_u64` accepts `x < u64::MAX as f64` (`== 2^64`).
//! - **Inline decimal fractions** — dyadic short decimals at `exp -1..=-7`; the
//!   deepest exactly-representable one is `2^-7 = 0.0078125`.
//! - **Positive-exponent factoring** — integer-valued *floats* factor trailing
//!   zeros into `exp 1..=7` (so `1e18` stays inline), while plain integers do
//!   not (they spill to the heap).
//! - **int-vs-float distinction** — `has_decimal_point` (which drives
//!   serialization) separates `1e18` from `1000000000000000000`.
//! - **Special floats** — `0.0`/`-0.0`, NaN/±Inf, subnormals, `MIN`/`MAX`, and
//!   `2^127` (the exact-comparison limit in `cmp_int_f64`).
//! - **Big-int overflow** — an integer literal beyond `u64::MAX` is reparsed by
//!   `serde_json` as an `f64`, so it becomes a float.
#![allow(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    clippy::approx_constant
)]

use std::f64;

/// `i64` inputs. Covers both mantissa-fit boundaries (`2^23`, `2^55`), the
/// `2^53` f64 cliff, every signed-integer type limit, powers of ten (which the
/// factoring logic is sensitive to), and assorted "awkward" magnitudes with no
/// trailing zeros.
pub(crate) fn i64_cases() -> Vec<i64> {
    let mut v = vec![
        // Tiny values and sign boundaries.
        0,
        1,
        -1,
        2,
        -2,
        5,
        -5,
        9,
        10,
        -10,
        11,
        -11,
        99,
        100,
        -100,
        101,
        127,
        128,
        -128,
        -129,
        255,
        256,
        1000,
        -1000,
        1_000_000,
        -1_000_000,
        // i32 / u32 limits (relevant to `to_i32`/`to_u32`).
        i32::MAX as i64,
        i32::MIN as i64,
        i32::MAX as i64 + 1,
        i32::MIN as i64 - 1,
        u32::MAX as i64,
        u32::MAX as i64 + 1,
        1i64 << 31,
        -(1i64 << 31),
        // The 2^53 f64 cliff: `to_f64` must switch from exact to `None` here.
        (1i64 << 53) - 1,
        1i64 << 53,
        (1i64 << 53) + 1,
        (1i64 << 53) + 2,
        -((1i64 << 53) + 1),
        1i64 << 52,
        1i64 << 54,
        // Inline mantissa fit at 2^23 (the 32-bit boundary).
        (1i64 << 23) - 1,
        1i64 << 23,
        (1i64 << 23) + 1,
        -(1i64 << 23),
        -(1i64 << 23) - 1,
        // Inline mantissa fit at 2^55 (the 64-bit boundary).
        (1i64 << 55) - 1,
        1i64 << 55,
        (1i64 << 55) + 1,
        -(1i64 << 55),
        -(1i64 << 55) - 1,
        -(1i64 << 55) + 1,
        // i64 limits.
        i64::MIN,
        i64::MIN + 1,
        i64::MAX,
        i64::MAX - 1,
        // Large magnitudes with no trailing zeros (must not factor, and are not
        // f64-exact) — several are prime.
        9_007_199_254_740_881,
        9_999_999_999_999_937,
        9_223_372_036_854_775_783,
        6_148_914_691_236_517_205, // 0x5555_5555_5555_5555
        -6_148_914_691_236_517_205,
        // Round magnitudes above the mantissa (previously factored as integers,
        // now spill to the heap).
        36_000_000_000_000_000,
        -36_000_000_000_000_000,
    ];
    // Powers of ten, positive and negative (10^18 < i64::MAX < 10^19).
    let mut p: i64 = 1;
    for _ in 0..=18 {
        v.push(p);
        v.push(-p);
        p = p.saturating_mul(10);
    }
    v
}

/// `u64` inputs. Covers the `i64`/`u64` seam at `2^63`, the top of the `u64`
/// range, the shared mantissa/f64 boundaries, and powers of ten up to `10^19`.
pub(crate) fn u64_cases() -> Vec<u64> {
    let mut v = vec![
        0,
        1,
        2,
        u32::MAX as u64,
        u32::MAX as u64 + 1,
        // 2^53 f64 cliff.
        (1u64 << 53) - 1,
        1u64 << 53,
        (1u64 << 53) + 1,
        // Inline mantissa fit at 2^55.
        (1u64 << 55) - 1,
        1u64 << 55,
        (1u64 << 55) + 1,
        // The i64/u64 seam: i64::MAX, then the first value that needs `u64`.
        i64::MAX as u64,
        i64::MAX as u64 + 1, // == 2^63
        (1u64 << 63) - 1,
        1u64 << 63,
        (1u64 << 63) + 1,
        // Top of the u64 range.
        u64::MAX,
        u64::MAX - 1,
        18_446_744_073_709_551_557, // largest u64 prime
        12_297_829_382_473_034_410, // 0xAAAA_AAAA_AAAA_AAAA
        6_148_914_691_236_517_205,  // 0x5555_5555_5555_5555
    ];
    // Powers of ten up to 10^19 (10^20 > u64::MAX).
    let mut p: u64 = 1;
    for _ in 0..=19 {
        v.push(p);
        p = p.saturating_mul(10);
    }
    v
}

/// Finite `f64` inputs. Covers exact dyadic short decimals (inline), decimals
/// that merely *look* short but are not exactly representable (heap), the
/// integer-valued floats that factor into positive exponents, the float→int
/// conversion thresholds at `2^63`/`2^64`, and the extremes of the `f64` range.
pub(crate) fn f64_cases() -> Vec<f64> {
    let mut v = vec![
        // Zero (and negative zero, which canonicalises to +0.0).
        0.0,
        -0.0,
        // Small integers as floats ("N.0").
        1.0,
        -1.0,
        2.0,
        -2.0,
        10.0,
        100.0,
        -100.0,
        // Exact dyadic short decimals (inline, exp -1..=-7).
        0.5,
        -0.5,
        0.25,
        0.75,
        0.125,
        0.375,
        0.0625,
        0.1875,
        2.5,
        63.5,
        -63.5,
        0.0078125,  // 2^-7, the deepest inline fraction
        -0.0078125, // 2^-7, negative
        0.00390625, // 2^-8, just too deep -> heap
        // "Short-looking" decimals that are not exactly representable (heap).
        0.1,
        0.2,
        0.3,
        0.7,
        1.1,
        3.14,
        0.30000000000000004, // 0.1 + 0.2
        // Irrational-ish constants (heap f64).
        f64::consts::PI,
        f64::consts::E,
        f64::consts::TAU,
        f64::consts::SQRT_2,
        -f64::consts::PI,
        // Integer-valued floats spanning the factoring range (some inline via a
        // positive exponent, the largest ones spill to the heap).
        1e7,
        1e8,
        1e15,
        1e16,
        1e17,
        1e18,
        1e19,
        1e20,
        1e21,
        1e22, // largest power of ten that is exactly an f64
        1e23, // not exactly an f64 -> heap
        -1e18,
        // Powers of two as floats (all exact integers).
        (1u64 << 52) as f64,
        (1u64 << 53) as f64,
        (1u64 << 54) as f64,
        (1u64 << 55) as f64,
        // f32 exactness cliff at 2^24 (relevant to `to_f32`).
        16777216.0, // 2^24
        16777217.0, // 2^24 + 1
        16777218.0,
        // float -> int conversion thresholds.
        9223372036854774784.0,  // 2^63 - 1024, largest f64 int below i64::MAX
        9223372036854775808.0,  // 2^63 == i64::MAX as f64 (not convertible to i64)
        18446744073709549568.0, // 2^64 - 2048, largest f64 int below u64::MAX
        18446744073709551616.0, // 2^64 == u64::MAX as f64 (not convertible to u64)
        // The exact-comparison limit in `cmp_int_f64`.
        1.7014118346046923e38, // 2^127
        // Extremes of the f64 range.
        f64::MAX,
        f64::MIN, // == -f64::MAX
        f64::MIN_POSITIVE,
        f64::from_bits(1), // smallest positive subnormal (~5e-324)
        -f64::from_bits(1),
        f64::EPSILON,
    ];
    // Negatives of the integer-valued and extreme floats, for symmetry.
    let extra: Vec<f64> = [1e15, 1e16, 1e17, 1e19, 1e20, 1e22, 1e23, f64::MAX]
        .iter()
        .map(|x| -x)
        .collect();
    v.extend(extra);
    v
}

/// Non-finite `f64` inputs, which cannot be stored in an `INumber` and must be
/// rejected by `try_from` (and turned into `null` by `IValue::from`).
pub(crate) fn f64_nonfinite_cases() -> Vec<f64> {
    vec![f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
}

/// Raw JSON number strings to deserialize via `serde_json`. Covers the
/// integer/float syntactic distinction, the `i64`/`u64`/overflow parsing seams,
/// e-notation in every form, trailing-zero decimals, and both the shallowest and
/// deepest inline fractions.
pub(crate) fn json_number_cases() -> Vec<&'static str> {
    vec![
        // Plain integers, including both signs of zero.
        "0",
        "-0",
        "1",
        "-1",
        "10",
        "-10",
        "100",
        "255",
        "256",
        "1000000000000000000",
        "-1000000000000000000",
        // Integer parsing seams: i64::MAX, i64::MAX+1 (u64), u64::MAX, overflow.
        "9223372036854775807",
        "9223372036854775808",
        "18446744073709551615",
        "18446744073709551616",  // u64::MAX + 1 -> f64
        "-9223372036854775808",  // i64::MIN
        "-9223372036854775809",  // i64::MIN - 1 -> f64
        "100000000000000000000", // 1e20 written out -> f64
        "-100000000000000000000",
        // Mantissa-fit boundaries as integers.
        "8388608",           // 2^23
        "8388607",           // 2^23 - 1
        "36028797018963968", // 2^55
        "36028797018963967", // 2^55 - 1
        "9007199254740992",  // 2^53
        "9007199254740993",  // 2^53 + 1 (exact integer, not f64-exact)
        // No-trailing-zero large integers.
        "12345678901234567",
        "9999999999999937",
        // Floats with an explicit decimal point.
        "0.0",
        "-0.0",
        "1.0",
        "2.0",
        "1.5",
        "-1.5",
        "0.5",
        "0.25",
        "0.1",
        "3.141592653589793",
        "0.0078125",  // 2^-7 (inline)
        "0.00390625", // 2^-8 (heap)
        "0.30000000000000004",
        // Trailing-zero decimals (re-serialise to a shorter form but stay equal).
        "1.50",
        "2.000",
        "100.00",
        "9007199254740992.0", // 2^53 as a float
        // e-notation in every spelling.
        "1e0",
        "0e0",
        "1e1",
        "2e2",
        "1e18", // same magnitude as the integer "1000000000000000000"
        "1E18",
        "1e+18",
        "1.5e3",
        "1.5E-3",
        "1e-7",
        "1e7",  // == 10000000.0, a float despite being integer-valued
        "1e22", // factors inline
        "1e23", // spills to the heap
        // Extremes of the f64 range (all finite and parseable).
        "1e308",
        "1e-308",
        "5e-324",                 // smallest positive subnormal
        "1e-400",                 // underflows to 0.0
        "1.7976931348623157e308", // f64::MAX
    ]
}

/// Number strings for `INumber`'s own [`FromStr`](std::str::FromStr) parser,
/// mirroring the other lists: the curated JSON strings above plus every `i64`,
/// `u64` and `f64` boundary rendered as text. The parser must accept them all and
/// agree with both direct construction and `serde_json`.
pub(crate) fn string_number_cases() -> Vec<String> {
    let mut v: Vec<String> = json_number_cases()
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    v.extend(i64_cases().iter().map(|x| x.to_string()));
    v.extend(u64_cases().iter().map(|x| x.to_string()));
    // Render floats via `serde_json` so each keeps float syntax (a decimal point
    // or exponent), exactly as it would appear in JSON.
    v.extend(
        f64_cases()
            .iter()
            .map(|x| serde_json::to_string(x).expect("a finite f64 serialises")),
    );
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IValue;
    use std::collections::hash_map::DefaultHasher;
    use std::convert::TryFrom;
    use std::hash::{Hash, Hasher};

    // An integer is exactly representable as an f64 iff it has at most 53
    // significant bits, i.e. `leading_zeros + trailing_zeros >= 11` over 64 bits.
    fn f64_exact_u64(a: u64) -> bool {
        a.leading_zeros() + a.trailing_zeros() >= 11
    }

    fn hash_of(v: &IValue) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    fn json(s: &str) -> IValue {
        serde_json::from_str(s).unwrap_or_else(|e| panic!("parse {:?}: {}", s, e))
    }

    // Parses a number string through `INumber`'s own `FromStr` parser.
    fn inum(s: &str) -> IValue {
        IValue::from(
            s.parse::<crate::INumber>()
                .unwrap_or_else(|e| panic!("parse {:?}: {}", s, e)),
        )
    }

    // Every number from the edge-case lists, as `IValue`s — including the string
    // list parsed through `INumber::from_str`. Deliberately contains duplicates
    // (the same value reached via different types/representations/parsers), which
    // is exactly what the equality/hash/order invariants must tolerate.
    fn number_pool() -> Vec<IValue> {
        let mut pool = Vec::new();
        pool.extend(i64_cases().into_iter().map(IValue::from));
        pool.extend(u64_cases().into_iter().map(IValue::from));
        pool.extend(f64_cases().into_iter().map(IValue::from));
        pool.extend(string_number_cases().iter().map(|s| inum(s)));
        pool
    }

    #[test]
    fn i64_inputs_are_consistent() {
        for &x in &i64_cases() {
            let v = IValue::from(x);
            assert!(v.is_number(), "{} not a number", x);
            // Round-trips exactly and is an integer (no decimal point).
            assert_eq!(v.to_i64(), Some(x), "{} to_i64", x);
            assert!(!v.as_number().unwrap().has_decimal_point(), "{} dot", x);
            // `to_u64` succeeds exactly when the value is non-negative.
            assert_eq!(v.to_u64(), u64::try_from(x).ok(), "{} to_u64", x);
            // `to_f64` is `Some` exactly when the value is f64-exact.
            assert_eq!(
                v.to_f64().is_some(),
                f64_exact_u64(x.unsigned_abs()),
                "{} to_f64 exactness",
                x
            );
            // Narrowing conversions agree with the standard library.
            assert_eq!(v.to_i32(), i32::try_from(x).ok(), "{} to_i32", x);
            assert_eq!(v.to_u32(), u32::try_from(x).ok(), "{} to_u32", x);
            // serde round-trip preserves the value.
            let s = serde_json::to_string(&v).unwrap();
            let back: IValue = serde_json::from_str(&s).unwrap();
            assert_eq!(v, back, "{} serde round-trip ({})", x, s);
        }
    }

    #[test]
    fn u64_inputs_are_consistent() {
        for &x in &u64_cases() {
            let v = IValue::from(x);
            assert!(v.is_number(), "{} not a number", x);
            assert_eq!(v.to_u64(), Some(x), "{} to_u64", x);
            assert!(!v.as_number().unwrap().has_decimal_point(), "{} dot", x);
            assert_eq!(v.to_i64(), i64::try_from(x).ok(), "{} to_i64", x);
            assert_eq!(v.to_f64().is_some(), f64_exact_u64(x), "{} to_f64", x);
            let s = serde_json::to_string(&v).unwrap();
            let back: IValue = serde_json::from_str(&s).unwrap();
            assert_eq!(v, back, "{} serde round-trip ({})", x, s);
        }
    }

    #[test]
    fn f64_inputs_are_consistent() {
        for &x in &f64_cases() {
            let v = IValue::from(x);
            assert!(v.is_number(), "{} not a number", x);
            // Every f64 keeps a decimal point, and reconstructs exactly (lossy
            // conversion is only lossy for values we cannot store, and those
            // still store their own bits). `0.0 == -0.0`, so this holds for -0.0.
            assert!(v.as_number().unwrap().has_decimal_point(), "{} dot", x);
            assert_eq!(v.to_f64_lossy(), Some(x), "{} to_f64_lossy", x);
            // When integer conversion succeeds the recovered integer equals x.
            if let Some(i) = v.to_i64() {
                assert_eq!(i as f64, x, "{} to_i64 value", x);
            }
            if let Some(u) = v.to_u64() {
                assert_eq!(u as f64, x, "{} to_u64 value", x);
            }
            // serde_json's default float *parser* is not exact-round-trip for all
            // magnitudes (its precise parser is behind the `float_roundtrip`
            // feature). Require only that ijson loses no more than serde_json's own
            // f64 pipeline does. (ijson canonicalises -0.0 to +0.0; `0.0 == -0.0`.)
            let out = serde_json::to_string(&v).unwrap();
            let back: IValue = serde_json::from_str(&out).unwrap();
            let baseline: f64 = serde_json::from_str(&serde_json::to_string(&x).unwrap()).unwrap();
            assert_eq!(
                back.to_f64_lossy(),
                Some(baseline),
                "{} serde round-trip",
                x
            );
        }
    }

    #[test]
    fn round_trips_through_inumber_exactly() {
        // Converting a number into an `INumber` and back out with the matching
        // accessor must return exactly the original value.
        for &x in &i64_cases() {
            assert_eq!(crate::INumber::from(x).to_i64(), Some(x), "i64 {}", x);
        }
        for &x in &u64_cases() {
            assert_eq!(crate::INumber::from(x).to_u64(), Some(x), "u64 {}", x);
        }
        for &x in &f64_cases() {
            let n = crate::INumber::try_from(x).unwrap();
            // The *exact* accessor round-trips (not merely the lossy one). `0.0 ==
            // -0.0`, so the canonicalisation of -0.0 to +0.0 still satisfies this.
            assert_eq!(n.to_f64(), Some(x), "f64 {} exact", x);
            assert_eq!(n.to_f64_lossy(), x, "f64 {} lossy", x);
        }
    }

    #[test]
    fn nonfinite_f64_inputs_are_rejected() {
        for &x in &f64_nonfinite_cases() {
            assert!(crate::INumber::try_from(x).is_err(), "{} accepted", x);
            assert!(IValue::from(x).is_null(), "{} not null", x);
        }
    }

    #[test]
    fn json_inputs_are_consistent() {
        for &s in &json_number_cases() {
            let v: IValue =
                serde_json::from_str(s).unwrap_or_else(|e| panic!("parse {:?}: {}", s, e));
            assert!(v.is_number(), "{:?} not a number", s);

            // A number has a decimal point iff serde stored it as a float: when
            // written with `.`/`e`/`E`, when it is a bare integer too large for
            // `i64`/`u64` (reparsed as a float), or the token "-0" (which
            // serde_json parses as -0.0 to preserve the sign).
            let syntactic_float = s.bytes().any(|b| matches!(b, b'.' | b'e' | b'E'));
            let fits_int = s.parse::<i64>().is_ok() || s.parse::<u64>().is_ok();
            let expect_dot = syntactic_float || !fits_int || s == "-0";
            assert_eq!(
                v.as_number().unwrap().has_decimal_point(),
                expect_dot,
                "{:?} decimal-point",
                s
            );

            // ijson's number deserialization agrees with serde_json's own
            // `Value` -> `IValue` path (both use serde_json's parser, so this is
            // exact regardless of that parser's float precision).
            let via_value: serde_json::Value = serde_json::from_str(s).unwrap();
            assert_eq!(v, IValue::from(via_value), "{:?} vs serde Value", s);

            // Serialising then reparsing is consistent between ijson and serde
            // (both share the same parser, so any float imprecision is identical).
            let out = serde_json::to_string(&v).unwrap();
            let back: IValue = serde_json::from_str(&out).unwrap();
            let back_value: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(back, IValue::from(back_value), "{:?} reparse agreement", s);
        }
    }

    #[test]
    fn string_inputs_are_consistent() {
        for s in &string_number_cases() {
            let s = s.as_str();
            let v = inum(s);
            assert!(v.is_number(), "{:?} not a number", s);

            // A number has a decimal point iff it is written as a float: with a
            // fraction/exponent, or as a bare integer too large for `i64`/`u64`
            // (stored as a float). Unlike serde_json — which parses "-0" as -0.0
            // to keep the sign — the string parser treats the integer token "-0"
            // as the integer `0`, faithfully to the JSON grammar (so no dot).
            let syntactic_float = s.bytes().any(|b| matches!(b, b'.' | b'e' | b'E'));
            let fits_int = s.parse::<i64>().is_ok() || s.parse::<u64>().is_ok();
            let expect_dot = syntactic_float || !fits_int;
            let dot = v.as_number().unwrap().has_decimal_point();
            assert_eq!(dot, expect_dot, "{:?} decimal-point", s);

            if !expect_dot {
                // An exact integer: both parsers agree on the value.
                let js = json(s);
                assert_eq!(v, js, "{:?} vs serde_json", s);
                // When serde also stored it as an integer they land on identical
                // bits. (serde stores the token "-0" as the float -0.0 — a
                // different decimal-point class — so that one is skipped.)
                if !js.as_number().unwrap().has_decimal_point() {
                    assert_eq!(v.number_repr_key(), js.number_repr_key(), "{:?} repr", s);
                }
            } else {
                // Stored as a float. serde_json's default parser can round large
                // magnitudes differently than `std`'s `f64::from_str` (its precise
                // parser is behind the `float_roundtrip` feature), so the value is
                // checked against a direct `std` `f64` parse rather than serde.
                assert_eq!(v, IValue::from(s.parse::<f64>().unwrap()), "{:?} value", s);
            }

            // Serialising and reparsing (through the same parser) is consistent.
            let out = serde_json::to_string(&v).unwrap();
            assert_eq!(inum(&out), v, "{:?} round-trip ({})", s, out);
        }
    }

    #[test]
    fn string_parsing_matches_direct_construction() {
        // Rendering each numeric edge case to text and parsing it back through
        // `INumber::from_str` reproduces the directly-constructed number.
        for &x in &i64_cases() {
            let v = inum(&x.to_string());
            assert_eq!(v, IValue::from(x), "i64 {}", x);
            assert!(!v.as_number().unwrap().has_decimal_point(), "i64 {} dot", x);
        }
        for &x in &u64_cases() {
            let v = inum(&x.to_string());
            assert_eq!(v, IValue::from(x), "u64 {}", x);
            assert!(!v.as_number().unwrap().has_decimal_point(), "u64 {} dot", x);
        }
        for &x in &f64_cases() {
            let v = inum(&serde_json::to_string(&x).unwrap());
            assert_eq!(v, IValue::from(x), "f64 {}", x);
            assert!(v.as_number().unwrap().has_decimal_point(), "f64 {} dot", x);
        }
    }

    #[test]
    fn equal_magnitudes_agree_across_representations() {
        // A value written as an integer and as an e-notation float must compare
        // equal and hash equal, even though one is inline and the other heap and
        // they disagree on `has_decimal_point`.
        let pairs = [
            ("100", "1e2"),
            ("10000000", "1e7"),
            ("1000000000000000000", "1e18"),
        ];
        for (int_str, float_str) in pairs {
            let int = json(int_str);
            let float = json(float_str);
            assert_eq!(int, float, "{} == {}", int_str, float_str);
            assert!(!int.as_number().unwrap().has_decimal_point());
            assert!(float.as_number().unwrap().has_decimal_point());
            assert_eq!(
                hash_of(&int),
                hash_of(&float),
                "{} hash {}",
                int_str,
                float_str
            );
        }
    }

    #[test]
    fn normalisation_makes_equal_values_indistinguishable() {
        // Each group is the SAME mathematical number reached through different
        // types and representations. Within a group every value must compare
        // equal and hash equal (the crux the caller called out: -0.0, +0.0 and
        // integer 0 all normalise together, as do 1 and 1.0); representatives of
        // different groups must differ.
        let groups: Vec<Vec<IValue>> = vec![
            // Zero: both signed zeros and integer zero.
            vec![
                IValue::from(0_i64),
                IValue::from(0_u64),
                IValue::from(0.0_f64),
                IValue::from(-0.0_f64),
                json("0"),
                json("-0"),
                json("0.0"),
                json("-0.0"),
                json("0e0"),
            ],
            // One.
            vec![
                IValue::from(1_i64),
                IValue::from(1_u64),
                IValue::from(1.0_f64),
                json("1"),
                json("1.0"),
                json("1e0"),
            ],
            // Negative one.
            vec![
                IValue::from(-1_i64),
                IValue::from(-1.0_f64),
                json("-1"),
                json("-1.0"),
            ],
            // A hundred, integer and float spellings.
            vec![
                IValue::from(100_i64),
                IValue::from(1e2_f64),
                json("100"),
                json("1e2"),
                json("100.0"),
            ],
            // 2^55: exactly representable as i64, u64 and f64.
            vec![
                IValue::from(1_i64 << 55),
                IValue::from(1_u64 << 55),
                IValue::from((1_u64 << 55) as f64),
            ],
            // 10^18: heap integer vs inline factored e-notation float.
            vec![
                IValue::from(1_000_000_000_000_000_000_i64),
                IValue::from(1e18_f64),
                json("1000000000000000000"),
                json("1e18"),
            ],
            // 2^63: u64 vs float (both heap), straddling the i64/u64 seam.
            vec![
                IValue::from(1_u64 << 63),
                IValue::from(9223372036854775808.0_f64),
                json("9223372036854775808"),
            ],
            // A dyadic fraction.
            vec![IValue::from(0.5_f64), json("0.5"), json("5e-1")],
        ];

        for g in &groups {
            for a in g {
                for b in g {
                    assert_eq!(a, b, "same group not equal: {:?} vs {:?}", a, b);
                    assert_eq!(
                        hash_of(a),
                        hash_of(b),
                        "same group hashes differ: {:?} vs {:?}",
                        a,
                        b
                    );
                }
            }
        }
        for (i, g1) in groups.iter().enumerate() {
            for (j, g2) in groups.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        g1[0], g2[0],
                        "different groups equal: {:?} vs {:?}",
                        g1[0], g2[0]
                    );
                }
            }
        }
    }

    #[test]
    fn sorting_matches_reference_order() {
        // A strictly-increasing sequence spanning representations and the
        // precision-trap adjacencies (a large integer just above the float it
        // rounds to, and the i64/u64/float seams).
        let ordered: Vec<IValue> = vec![
            IValue::from(f64::MIN), // -f64::MAX
            IValue::from(-1e300_f64),
            IValue::from(i64::MIN),
            IValue::from(-1e18_f64),
            IValue::from(-(1_i64 << 55)),
            IValue::from(-1_000_000_i64),
            IValue::from(-2.5_f64),
            IValue::from(-1_i64),
            IValue::from(-0.5_f64),
            IValue::from(-0.0078125_f64),
            IValue::from(0_i64),
            IValue::from(0.0078125_f64),
            IValue::from(0.5_f64),
            IValue::from(1_i64),
            IValue::from(1.5_f64),
            IValue::from(2_i64),
            IValue::from(100_i64),
            IValue::from(9007199254740992.0_f64), // 2^53 as a float
            IValue::from(9007199254740993_i64),   // 2^53 + 1 (> the float above)
            IValue::from(1e17_f64),
            IValue::from(1e18_f64),
            IValue::from(i64::MAX),    // 2^63 - 1
            IValue::from(1_u64 << 63), // 2^63
            IValue::from(u64::MAX),    // 2^64 - 1
            IValue::from(1e20_f64),    // > u64::MAX
            IValue::from(1e300_f64),
            IValue::from(f64::MAX),
        ];

        // Strictly increasing, so `cmp` never conflates two distinct values.
        for w in ordered.windows(2) {
            assert!(
                w[0] < w[1],
                "not strictly increasing: {:?} !< {:?}",
                w[0],
                w[1]
            );
        }

        // Sorting a shuffled copy recovers exactly this order.
        let mut shuffled = ordered.clone();
        shuffled.reverse();
        shuffled.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(shuffled, ordered);
    }

    #[test]
    fn representation_is_canonical() {
        // Canonicalisation: the same number, reached different ways, must have the
        // identical internal representation. The *only* allowed difference is the
        // decimal point — an integer and the float form of the same value differ
        // (and larger values differ further in inline-vs-heap storage). Within one
        // decimal-point class the representation must be unique.

        // Integers reached via `i64`, `u64` and JSON all land on the same bits.
        for &x in &i64_cases() {
            let base = IValue::from(x).number_repr_key();
            assert_eq!(
                json(&x.to_string()).number_repr_key(),
                base,
                "int {} JSON",
                x
            );
            if x >= 0 {
                let via_u64 = IValue::from(x as u64).number_repr_key();
                assert_eq!(via_u64, base, "int {} via u64", x);
            }
        }
        for &x in &u64_cases() {
            let base = IValue::from(x).number_repr_key();
            assert_eq!(
                json(&x.to_string()).number_repr_key(),
                base,
                "uint {} JSON",
                x
            );
            if let Ok(i) = i64::try_from(x) {
                assert_eq!(
                    IValue::from(i).number_repr_key(),
                    base,
                    "uint {} via i64",
                    x
                );
            }
        }

        // Grouped equal magnitudes: within a decimal-point class the bits match
        // exactly. In particular -0.0, +0.0 and integer 0 collapse as expected
        // (the two signed zeros are literally the same bits).
        let groups: Vec<Vec<IValue>> = vec![
            vec![
                IValue::from(0_i64),
                IValue::from(0_u64),
                IValue::from(0.0_f64),
                IValue::from(-0.0_f64),
                json("0"),
                json("-0"),
                json("0.0"),
                json("-0.0"),
                json("0e0"),
            ],
            vec![
                IValue::from(1_i64),
                IValue::from(1.0_f64),
                json("1"),
                json("1.0"),
                json("1e0"),
            ],
            vec![
                IValue::from(100_i64),
                IValue::from(1e2_f64),
                json("100"),
                json("1e2"),
                json("100.0"),
            ],
            vec![
                IValue::from(1_000_000_000_000_000_000_i64),
                IValue::from(1e18_f64),
                json("1000000000000000000"),
                json("1e18"),
            ],
            vec![IValue::from(0.5_f64), json("0.5"), json("5e-1")],
        ];
        for g in &groups {
            for a in g {
                for b in g {
                    if a.as_number().unwrap().has_decimal_point()
                        == b.as_number().unwrap().has_decimal_point()
                    {
                        assert_eq!(
                            a.number_repr_key(),
                            b.number_repr_key(),
                            "same value & decimal-point, different representation: {:?} vs {:?}",
                            a,
                            b
                        );
                    }
                }
            }
        }
    }

    // Exhaustive pairwise checks over the whole pool are O(n^2); skip them under
    // Miri (which only needs to see the unsafe paths, covered by the tests above).
    #[cfg(not(miri))]
    #[test]
    fn equal_values_share_hash_and_representation() {
        let pool = number_pool();
        for a in &pool {
            for b in &pool {
                if a != b {
                    continue;
                }
                // The Hash/Eq contract: equal values must hash equal. This is the
                // cross-check between `inline::number::hash` and `number_hash`.
                assert_eq!(
                    hash_of(a),
                    hash_of(b),
                    "equal values hash differently: {:?} vs {:?}",
                    a,
                    b
                );
                // Canonicalisation: equal values with the same decimal point must
                // be bit-for-bit identical, no matter how they were constructed.
                if a.as_number().unwrap().has_decimal_point()
                    == b.as_number().unwrap().has_decimal_point()
                {
                    assert_eq!(
                        a.number_repr_key(),
                        b.number_repr_key(),
                        "equal values, same decimal-point, different representation: {:?} vs {:?}",
                        a,
                        b
                    );
                }
            }
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn comparison_is_a_consistent_total_order() {
        use std::cmp::Ordering;
        let pool = number_pool();
        for a in &pool {
            for b in &pool {
                let ab = a
                    .partial_cmp(b)
                    .unwrap_or_else(|| panic!("no ordering: {:?} vs {:?}", a, b));
                let ba = b
                    .partial_cmp(a)
                    .unwrap_or_else(|| panic!("no ordering: {:?} vs {:?}", b, a));
                // Antisymmetry and agreement between `==` and `cmp == Equal`.
                assert_eq!(ab, ba.reverse(), "antisymmetry: {:?} vs {:?}", a, b);
                assert_eq!(
                    a == b,
                    ab == Ordering::Equal,
                    "eq/cmp disagree: {:?} vs {:?}",
                    a,
                    b
                );
            }
        }
    }
}
