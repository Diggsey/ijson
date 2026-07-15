# Changelog

## 0.1.7

- Add `FromIterator<T: Into<IValue>>` for `IValue` (collects into an array) and `FromIterator<(K: Into<IString>, V: Into<IValue>)>` for `IValue` (collects into an object), mirroring `serde_json::Value`.
- Add `From<serde_json::Value> for IValue` and `From<IValue> for serde_json::Value` for smoother interoperability with `serde_json`.
- Add `From<serde_json::Map<String, serde_json::Value>> for IObject` and the reverse, matching the existing `HashMap`/`BTreeMap`/`IndexMap` conversions.
- Store small values — `null`, booleans, small numbers, and short strings — inline within the pointer-sized `IValue`, behind a widened 3-bit tag scheme. Small numbers are held as a `mantissa * base^exp` value (a base-2 binary float by default, so every inline number is exactly an `f64`), so most JSON numbers (integers, timestamps/ids, and short fractions such as `0.5` and `63.5`) require no allocation or pointer indirection; only larger integers and out-of-range floats fall back to an 8-byte heap payload. Note: `IString::as_str().as_ptr()` is no longer stable across equal short strings, since they are no longer deduplicated to a shared allocation (value equality via `==` is unaffected).
- Add an opt-in `arbitrary_precision` feature that stores numbers as their exact decimal value (`mantissa * 10^exp` inline, or a heap arbitrary-precision decimal) instead of rounding to `f64`. With it enabled, values such as `0.1` are kept exactly, and integers and decimals beyond `f64`'s range and precision become representable; deserialization routes through the same parser as `str::parse`, so a value round-trips exactly.

## 0.1.6

- **Breaking:** Remove `Borrow<str>` impl for `IString` by default. The impl violates the `Borrow` contract because `IString` hashes by pointer, not by contents, causing silent lookup failures in `HashMap`/`HashSet` when using `&str` keys. A `broken-borrow-impl-compat` feature flag is available as a temporary compatibility measure.

## 0.1.5

- Fix potential undefined behavior on allocation failure.
- Update dependencies.

## 0.1.4

- Add optional `indexmap` feature to enable conversion between `IObject` and `IndexMap`.
- Fix unsound offset calculation in internal `Header` layout.
- Fix `is_object` typo in `into_object`.
- Upgrade `dashmap` to 5.4.

## 0.1.3

- Add missing string de-serialization paths.
- Update dependencies.

## 0.1.2

- Fix incorrect interning of empty string.

## 0.1.1

- Fix bounds on short integer storage.
- Add CI and coverage measurement.

## 0.1.0

- Initial release: memory-efficient `IValue` replacement for `serde_json::Value`.
- Interned strings (`IString`) and numbers (`INumber`).
- Robin-hood hash table based `IObject`.
- `IArray` type.
- Full serde serialization/deserialization support.
- `From`/`Into` conversions for common types.
