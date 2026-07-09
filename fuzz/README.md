# ijson numeric fuzz targets

Coverage-guided [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) targets
that check the numeric code against the same invariants the exhaustive tests in
[`src/numeric_edge_cases.rs`](../src/numeric_edge_cases.rs) enforce — but over
adversarial, machine-generated inputs instead of a curated list.

| Target | What it drives | Invariants |
| --- | --- | --- |
| `number_str` | `INumber::from_str` over arbitrary strings | parses ⟹ is a number; accepted ⟹ `serde_json` accepts (away from the f64 overflow boundary); serialize+reparse round-trips; an in-range integer equals the direct construction |
| `number_value` | `f64` construction over the whole bit space | every finite `f64` round-trips *exactly* through `to_f64`/`to_f64_lossy`; integer conversions recover the value; non-finite `f64`s are rejected |
| `number_ord` | comparison/hashing of numbers built through *different* constructors | a consistent total order (antisymmetry, transitivity, `==` ⇔ `cmp == Equal`) and the `Hash`/`Eq` contract, across representations |

`number_value` found a real bug on first run: `to_f64_lossy` decoded an inline
decimal whose scaled mantissa exceeds `2^53` by rounding the mantissa before
scaling (`949288156749637.5` came back as `.6`). Fixed, with the value added to
`f64_cases`.

## Running

```sh
cargo +nightly fuzz run number_value
cargo +nightly fuzz run number_str
cargo +nightly fuzz run number_ord
```

Bound a run with libFuzzer flags, e.g. `-- -runs=5000000` or `-- -max_total_time=60`.

### Windows

Coverage-guided fuzzing works with the default AddressSanitizer build; it just
needs the ASan runtime available. Per the
[cargo-fuzz book](https://rust-fuzz.github.io/book/cargo-fuzz/windows/setup.html),
install the **"C++ AddressSanitizer"** component (alongside the MSVC v143 build
tools) via the Visual Studio Installer, then run from a "Developer PowerShell for
VS" (or otherwise ensure the MSVC `bin` directory is on `PATH`) so link.exe finds
`clang_rt.asan_dynamic_runtime_thunk-x86_64.lib` and the ASan DLL loads at
runtime.

If you only have a standalone LLVM install, its `clang_rt` is compatible
(Microsoft's ASan *is* LLVM's compiler-rt); point the link and run paths at it:

```pwsh
$rt = "C:\Program Files\LLVM\lib\clang\<version>\lib\windows"
$env:LIB  = "$rt;$env:LIB"    # so link.exe finds clang_rt.asan_*.lib
$env:PATH = "$rt;$env:PATH"   # so clang_rt.asan_dynamic-x86_64.dll loads
cargo +nightly fuzz run number_value
```
