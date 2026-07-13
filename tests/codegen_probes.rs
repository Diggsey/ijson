//! Codegen tests: what the basic operations actually compile to.
//!
//! `codegen.rs` checks one coarse property across the whole library — that nothing
//! dispatches through a vtable. These check a handful of individual operations, closely:
//!
//!   - **Constructing** a small value is a compile-time constant. A short string, a small
//!     number, a bool and `null` live *in the pointer word*, so the constructor is pure
//!     arithmetic; for a known input it should fold to a single `ret` of the encoded word.
//!     That word is asserted exactly, which pins the bit layout from the outside — the
//!     encoding is otherwise only ever asserted by code sharing the very constants it is
//!     testing, so it could not catch a layout that quietly agreed with itself.
//!
//!   - **Reading** a value stays on the fast path. The accessors take a value whose
//!     representation is not known at compile time, so they cannot fold away; what they
//!     *can* do is stay free of the costs that would make them slow — a vtable, an
//!     allocation, a panic path, or a call out of line where the work should be inline.
//!
//! # Reading the constants
//!
//! An inline word is `[ payload | flags | tag ]`, lowest bits first:
//!
//! ```text
//!   bits 0-2  tag        (`Inline` == 0)
//!   bit  3    IS_NUMBER
//!   bit  4    IS_STRING  (when IS_NUMBER is clear)
//!   bits 5-7  payload    (a constant's discriminant, or an inline string's length)
//!   bits 8..  the string's bytes, or a number's mantissa
//! ```
//!
//! # Fragility
//!
//! These are deliberately strict, so they *will* fail if the optimizer changes what it can
//! fold or inline — that is the point. A regression here means an operation that was free
//! stopped being free, and nothing else in the suite would notice: the behaviour stays
//! identical, only the generated code is worse. If a failure shows a different constant,
//! decode it against the table above before assuming the compiler is at fault; it more
//! likely means the layout moved.
//!
//! Fat LTO is required, because these operations are not `#[inline]` and so export no MIR:
//! without it the probe crate can only *call* them, and there is nothing to fold or look
//! inside.

mod common;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The operations under test. `#[no_mangle]`, so each survives LTO as an exported root and
/// appears in the IR under a name we can find.
///
/// The Rust ABI, deliberately: an `extern "C"` function cannot unwind, so rustc wraps its
/// body in an abort-on-unwind shim and leaves the real work out of line — which would hide
/// the very code these tests exist to look at.
const PROBES: &str = r#"
use ijson::{INumber, IString, IValue, ValueType};

// Constructing, from an input known at compile time.
#[no_mangle] pub fn probe_null() -> IValue { IValue::NULL }
#[no_mangle] pub fn probe_true() -> IValue { IValue::from(true) }
#[no_mangle] pub fn probe_false() -> IValue { IValue::from(false) }
#[no_mangle] pub fn probe_empty_string() -> IString { IString::from("") }
#[no_mangle] pub fn probe_inline_string() -> IString { IString::from("abc") }
#[no_mangle] pub fn probe_zero() -> INumber { INumber::from(0i64) }
#[no_mangle] pub fn probe_small_int() -> INumber { INumber::from(42i64) }
#[no_mangle] pub fn probe_negative_int() -> INumber { INumber::from(-1i64) }
#[no_mangle] pub fn probe_half() -> IValue { IValue::from(0.5f64) }

// Reading, from a value whose representation is not.
#[no_mangle] pub fn probe_to_i64(n: &INumber) -> Option<i64> { n.to_i64() }
#[no_mangle] pub fn probe_to_u64(n: &INumber) -> Option<u64> { n.to_u64() }
#[no_mangle] pub fn probe_to_f64(n: &INumber) -> Option<f64> { n.to_f64() }
#[no_mangle] pub fn probe_to_f64_lossy(n: &INumber) -> f64 { n.to_f64_lossy() }
#[no_mangle] pub fn probe_has_decimal_point(n: &INumber) -> bool { n.has_decimal_point() }
#[no_mangle] pub fn probe_is_number(v: &IValue) -> bool { v.is_number() }
#[no_mangle] pub fn probe_type(v: &IValue) -> ValueType { v.type_() }
"#;

// --- Constructing -----------------------------------------------------------

/// The word each construction must fold to, and how it decomposes. The IR prints an `i64`
/// constant signed, which is why the negative mantissa comes out negative.
// Every number is written as `mantissa << 8 | exponent code << 4 | IS_NUMBER`, so the
// fields line up and can be compared by eye. Zero's mantissa is written out as such rather
// than dropped, which is the point clippy objects to.
#[allow(clippy::identity_op)]
fn constants() -> Vec<(&'static str, i64, &'static str)> {
    // Integers encode identically under both inline number representations — a plain
    // integer is a mantissa at the reserved exponent code either way — so only the *float*
    // differs, which is the one number below that is feature-dependent.
    let half = if cfg!(feature = "arbitrary_precision") {
        // Base 10: `5 * 10^-1`. mantissa 5 << 8 | code (-1 + bias 7) << 4 | IS_NUMBER.
        (5 << 8) | (6 << 4) | 8
    } else {
        // Base 2: `1 * 2^-1`. The same shape, a different mantissa and exponent for the
        // same value — which is the whole difference between the two representations.
        (1 << 8) | (6 << 4) | 8
    };

    vec![
        (
            "probe_null",
            1 << 5,
            "Null: discriminant 1 in the payload bits",
        ),
        ("probe_true", 3 << 5, "True: discriminant 3"),
        ("probe_false", 2 << 5, "False: discriminant 2"),
        (
            "probe_empty_string",
            1 << 4,
            "IS_STRING alone: length 0, no bytes — and still non-zero, so the empty string \
             is not the reserved niche",
        ),
        (
            "probe_inline_string",
            // 0x63'62'61'70: the bytes 'c','b','a' above the control byte
            // `IS_STRING | (3 << 5)`, little-endian.
            (i64::from(b'c') << 24)
                | (i64::from(b'b') << 16)
                | (i64::from(b'a') << 8)
                | (1 << 4)
                | (3 << 5),
            "\"abc\": control byte 0x70, then the three bytes",
        ),
        (
            "probe_zero",
            (0 << 8) | (15 << 4) | 8,
            "integer 0: a zero mantissa at the reserved exponent code — and *not* the \
             all-zero niche, because IS_NUMBER is set",
        ),
        (
            "probe_small_int",
            (42 << 8) | (15 << 4) | 8,
            "integer 42: mantissa 42 at the reserved exponent code",
        ),
        (
            "probe_negative_int",
            (-1 << 8) | (15 << 4) | 8,
            "integer -1: the mantissa is signed, so the top bits are all ones",
        ),
        ("probe_half", half, "0.5, in the active inline number base"),
    ]
}

#[test]
#[cfg_attr(miri, ignore = "shells out to a compiler, which Miri cannot run")]
// The constants are the 64-bit little-endian encoding. The layout is the same shape
// elsewhere, but the mantissa and the inline-string capacity are narrower, so the words
// differ and there is nothing to gain from re-deriving them here.
#[cfg_attr(
    not(all(target_pointer_width = "64", target_endian = "little")),
    ignore = "the pinned constants are the 64-bit little-endian encoding"
)]
fn constructing_a_small_value_folds_to_a_constant() {
    let ir = probe_ir();

    let mut failures = Vec::new();
    for (name, word, meaning) in constants() {
        let Some(body) = common::body_of(ir, name) else {
            failures.push(format!("{}: not found in the emitted IR", name));
            continue;
        };

        // Exactly one instruction, returning exactly this word: no call, no branch, no
        // allocation — the construction happened at compile time.
        let want = format!("ret ptr inttoptr (i64 {} to ptr)", word);
        if body != [want.as_str()] {
            failures.push(format!(
                "{} ({})\n    expected: {}\n    got:      {}",
                name,
                meaning,
                want,
                body.join("\n              ")
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "constructing a small value no longer folds to its constant:\n\n{}\n\n\
         Each of these should compile to a single `ret` of the encoded inline word. A \
         *different constant* most likely means the inline bit layout changed — decode it \
         against the table in this file's module docs. Anything other than a lone `ret` \
         means the constructor stopped folding, so building a small value now costs real \
         work at run time.",
        failures.join("\n\n")
    );
}

// --- Reading ----------------------------------------------------------------

/// An accessor, and the *only* calls its generated code is allowed to make.
///
/// An allow-list rather than a list of forbidden things: a fast path should call nothing
/// at all, and the exceptions are few enough to name. Anything else — a panic, an
/// allocation, or work left out of line that should have been inlined — is a regression,
/// and naming what is allowed means a *new* kind of cost cannot slip through unlisted.
struct FastPath {
    name: &'static str,
    /// Matched as substrings of the mangled symbol, so they survive both manglings.
    allowed: &'static [&'static str],
    /// Why each allowance is there — quoted back when the operation trips over it.
    because: &'static str,
}

fn fast_paths() -> Vec<FastPath> {
    // `INumber::to_f64_lossy` unwraps the `Option` that `IValue` returns, because an
    // `INumber` is always a number. The unwrap asserts the type's own invariant rather than
    // handling a case that can arise, and it is cold — better than the alternative, a
    // default value that would silently paper over the bug.
    const UNWRAP: &str = "unwrap_failed";

    // The numeric model. A conversion may call into it: the value work lives there, and
    // with `arbitrary_precision` an arm of it reduces a bignum — real work, rightly out of
    // line. Whether the *small* arms are inlined into the caller is LLVM's cost-model call
    // and differs by platform, so it is not pinned here; what is pinned is that nothing
    // else is called, and that none of it panics, allocates, or goes through a vtable.
    const NUMERIC: &str = "numeric";

    let dispatch_only = |name| FastPath {
        name,
        allowed: &[],
        because: "asking a value's type is a switch on the tag and nothing more",
    };
    let converts = |name| FastPath {
        name,
        allowed: &[NUMERIC],
        because: "a conversion may call into the numeric model, and nothing else",
    };

    vec![
        dispatch_only("probe_is_number"),
        dispatch_only("probe_type"),
        dispatch_only("probe_has_decimal_point"),
        converts("probe_to_i64"),
        converts("probe_to_u64"),
        converts("probe_to_f64"),
        FastPath {
            name: "probe_to_f64_lossy",
            allowed: &[NUMERIC, UNWRAP],
            because: "a conversion may call into the numeric model; and                       `INumber::to_f64_lossy` unwraps, asserting that an `INumber` really                       is a number",
        },
    ]
}

/// What a call it should not be making costs it.
fn cost_of(symbol: &str) -> &'static str {
    // An `unreachable` *instruction* is fine, and expected: an exhaustive switch over the
    // tag ends in `default.unreachable`, which generates nothing. A *call* into the panic
    // machinery is not — it drags in the formatting `Arguments` and a cold block, in an
    // operation that cannot actually fail.
    if symbol.contains("panic") || symbol.contains("unwrap_failed") {
        "panics"
    } else if symbol.contains("__rust_alloc") || symbol.contains("__rust_realloc") {
        "allocates"
    } else {
        "is not inlined — the work should be straight-line code here"
    }
}

#[test]
#[cfg_attr(miri, ignore = "shells out to a compiler, which Miri cannot run")]
fn reading_a_value_stays_on_the_fast_path() {
    let ir = probe_ir();

    let mut failures = Vec::new();
    for probe in fast_paths() {
        let Some(body) = common::body_of(ir, probe.name) else {
            failures.push(format!("{}: not found in the emitted IR", probe.name));
            continue;
        };
        assert!(!body.is_empty(), "{}: empty body", probe.name);
        let called = common::called_symbols(&body);

        // Never, for any accessor: the dispatch resolved to a vtable. `codegen.rs` asserts
        // this across the library; here it is asserted of the operations that matter most.
        if body.iter().any(|line| common::is_indirect_call(line)) {
            failures.push(format!(
                "{}: calls through a function pointer — the representation dispatch is no \
                 longer devirtualized",
                probe.name
            ));
        }

        for symbol in called {
            if probe.allowed.iter().any(|allowed| symbol.contains(allowed)) {
                continue;
            }
            failures.push(format!(
                "{}: {} — calls `{}`\n      (all it may call: {})",
                probe.name,
                cost_of(symbol),
                symbol,
                probe.because
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "reading a value no longer stays on the fast path:\n\n  {}\n\n\
         These run in the innermost loop of anything that walks a document, and every cost \
         above is invisible from behaviour alone — the results stay correct, they just stop \
         being cheap.\n\n\
         If an operation has started to panic, look for an `unreachable!()` on a state that \
         cannot occur: it is indeed unreachable, but the compiler does not know that, so it \
         emits the panic and its `Arguments` into every caller. State the invariant with a \
         `debug_assert!` over a total fallback instead — checked where checks are \
         affordable, and generating nothing where they are not.",
        failures.join("\n  "),
    );
}

// --- Building the probes ----------------------------------------------------

/// The probe crate's IR, built once however many tests ask for it.
fn probe_ir() -> &'static str {
    static IR: OnceLock<String> = OnceLock::new();
    IR.get_or_init(emit_probe_ir)
}

/// Writes a crate of probes with a path dependency on this one, builds it with fat LTO,
/// and returns the emitted LLVM IR.
fn emit_probe_ir() -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("codegen-probes");
    std::fs::create_dir_all(dir.join("src")).expect("create the probe crate");

    // A single-quoted TOML string, so a Windows path's backslashes are not escapes. A
    // `cdylib` because its `#[no_mangle]` exports are LTO roots: a `lib` would have its
    // probes internalized and dropped, and a `bin` keeps only what `main` reaches.
    let manifest = format!(
        "[package]\nname = \"probes\"\nversion = \"0.0.0\"\nedition = \"2018\"\n\n\
         [lib]\ncrate-type = [\"cdylib\"]\n\n\
         [dependencies]\nijson = {{ path = '{}'{} }}\n\n\
         [profile.release]\nlto = \"fat\"\ncodegen-units = 1\n",
        env!("CARGO_MANIFEST_DIR"),
        if cfg!(feature = "arbitrary_precision") {
            ", features = [\"arbitrary_precision\"]"
        } else {
            ""
        },
    );
    write_if_changed(&dir.join("Cargo.toml"), &manifest);
    write_if_changed(&dir.join("src/lib.rs"), PROBES);

    let status = common::nested_cargo()
        .current_dir(&dir)
        .args([
            "rustc",
            "--release",
            "--",
            "--emit=llvm-ir",
            "-Cdebuginfo=0",
        ])
        .status()
        .expect("failed to run `cargo rustc` on the probe crate");
    assert!(status.success(), "building the probe crate failed");

    // `codegen-units = 1` means a single module, but the file is named for the crate, and
    // that naming is not something to hard-code across platforms.
    let deps = dir.join("target/release/deps");
    let ir_file = std::fs::read_dir(&deps)
        .expect("read the probe crate's deps directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "ll")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("probes"))
        })
        // A stale `.ll` from an earlier build can linger; take the freshest.
        .max_by_key(|path| path.metadata().and_then(|meta| meta.modified()).ok())
        .unwrap_or_else(|| panic!("no `probes*.ll` was emitted in {:?}", deps));

    let ir = std::fs::read_to_string(&ir_file).expect("read the probe crate's LLVM IR");

    // If the nested build was instrumented anyway, every probe begins by bumping a
    // profiling counter: nothing folds, nothing inlines, and the IR is not the library's.
    // Say that, rather than reporting the instrumentation as a codegen regression.
    assert!(
        !common::is_instrumented(&ir),
        "the probe crate was built with coverage instrumentation, so its IR is not the \
         library's. Something is still feeding flags into the nested build:\n\n{}",
        common::flag_environment()
    );
    ir
}

/// Leaves the file alone when the contents already match, so `cargo` does not rebuild the
/// probe crate (and re-run LTO) on every invocation.
fn write_if_changed(path: &Path, contents: &str) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(contents) {
        std::fs::write(path, contents).unwrap_or_else(|e| panic!("write {:?}: {}", path, e));
    }
}
