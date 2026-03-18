# Changelog

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
