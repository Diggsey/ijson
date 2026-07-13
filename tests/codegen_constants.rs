//! Codegen test: constructing a small value is a compile-time constant.
//!
//! The whole point of the inline representations is that a short string, a small number,
//! a bool and `null` live *in the pointer word* — no allocation, no branching, nothing to
//! free. When the value is known at compile time that should collapse all the way down:
//! the constructor is pure arithmetic on constants, so the optimizer ought to fold it to
//! a single `ret` of the encoded word.
//!
//! This asserts exactly that, and asserts the *exact word*. It is the counterpart to
//! `codegen.rs`, which checks a coarse property (nothing dispatches through a vtable)
//! across the whole library; this one checks a handful of operations down to the
//! instruction. Between them: the dispatch stays static, and the cheap things stay free.
//!
//! Pinning the constants also pins the bit layout, from the outside. The encoding is
//! otherwise only ever asserted by code that shares the constants it is testing — here it
//! is read back out of the compiler and decoded against the layout as documented, so a
//! change to (say) `PAYLOAD_SHIFT` cannot quietly agree with itself.
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
//! This is deliberately strict, so it *will* fail if the optimizer changes what it can
//! fold — that is the point; a regression here means constructing a small value stopped
//! being free. If a failure shows a different constant, decode it against the table above
//! before assuming the compiler is at fault: it more likely means the layout moved.
//!
//! Requires fat LTO, because the constructors are not `#[inline]` and so export no MIR:
//! without it the probe crate can only *call* them, and there is nothing to fold.

mod common;

use std::path::{Path, PathBuf};

/// The operations whose codegen is pinned. Each is `extern "C"` and `#[no_mangle]`, so it
/// survives LTO as an exported root and appears in the IR under a name we can find.
const PROBES: &str = r#"
#![allow(improper_ctypes_definitions)]
use ijson::{INumber, IString, IValue};

#[no_mangle] pub extern "C" fn probe_null() -> IValue { IValue::NULL }
#[no_mangle] pub extern "C" fn probe_true() -> IValue { IValue::from(true) }
#[no_mangle] pub extern "C" fn probe_false() -> IValue { IValue::from(false) }
#[no_mangle] pub extern "C" fn probe_empty_string() -> IString { IString::from("") }
#[no_mangle] pub extern "C" fn probe_inline_string() -> IString { IString::from("abc") }
#[no_mangle] pub extern "C" fn probe_zero() -> INumber { INumber::from(0i64) }
#[no_mangle] pub extern "C" fn probe_small_int() -> INumber { INumber::from(42i64) }
#[no_mangle] pub extern "C" fn probe_negative_int() -> INumber { INumber::from(-1i64) }
#[no_mangle] pub extern "C" fn probe_half() -> IValue { IValue::from(0.5f64) }
"#;

/// The word each probe must fold to, and how that word decomposes. The IR prints an `i64`
/// constant signed, which is why the negative mantissa comes out negative.
// Every number below is written as `mantissa << 8 | exponent code << 4 | IS_NUMBER`, so
// the fields line up and can be compared by eye. Zero's mantissa is written out as such
// rather than dropped, which is the point clippy objects to.
#[allow(clippy::identity_op)]
fn expected() -> Vec<(&'static str, i64, &'static str)> {
    // Integers encode identically under both inline number representations — a plain
    // integer is a mantissa at the reserved exponent code either way — so only the
    // *float* differs, which is the one number below that is feature-dependent.
    let half = if cfg!(feature = "arbitrary_precision") {
        // Base 10: `5 * 10^-1`. mantissa 5 << 8 | code (-1 + bias 7) << 4 | IS_NUMBER.
        (5 << 8) | (6 << 4) | 8
    } else {
        // Base 2: `1 * 2^-1`. Same shape, a different mantissa and exponent for the same
        // value — which is the whole difference between the two representations.
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
            "IS_STRING alone: length 0, no bytes — and still non-zero, so the empty \
             string is not the reserved niche",
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

    // `codegen-units = 1` means a single module, but the file is named for the crate and
    // that naming is not something to hard-code across platforms.
    let deps = dir.join("target/release/deps");
    let ir = std::fs::read_dir(&deps)
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

    std::fs::read_to_string(&ir).expect("read the probe crate's LLVM IR")
}

/// Leaves the file alone when the contents already match, so `cargo` does not rebuild the
/// probe crate (and re-run LTO) on every invocation.
fn write_if_changed(path: &Path, contents: &str) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(contents) {
        std::fs::write(path, contents).unwrap_or_else(|e| panic!("write {:?}: {}", path, e));
    }
}

/// The instruction lines in the body of `@name`, or `None` if it is not defined.
fn body_of<'a>(ir: &'a str, name: &str) -> Option<Vec<&'a str>> {
    let header = format!("@{}(", name);
    let mut lines = ir
        .lines()
        .skip_while(|l| !(l.starts_with("define") && l.contains(&header)))
        .skip(1);

    let mut body = Vec::new();
    for line in &mut lines {
        let line = line.trim();
        if line == "}" {
            return Some(body);
        }
        // Skip block labels and comments; keep the instructions.
        if line.is_empty() || line.starts_with(';') || line.ends_with(':') {
            continue;
        }
        body.push(line);
    }
    None
}

#[test]
#[cfg_attr(miri, ignore = "shells out to a compiler, which Miri cannot run")]
// The constants below are the 64-bit little-endian encoding. The layout is the same
// shape elsewhere, but the mantissa and inline-string capacity are narrower, so the words
// differ and there is nothing to gain from re-deriving them here.
#[cfg_attr(
    not(all(target_pointer_width = "64", target_endian = "little")),
    ignore = "the pinned constants are the 64-bit little-endian encoding"
)]
fn constructing_a_small_value_folds_to_a_constant() {
    let ir = emit_probe_ir();

    // If the nested build was instrumented anyway, every probe begins by bumping a
    // profiling counter and nothing folds. That is not a regression in the library, so
    // say so plainly rather than reporting the instrumentation as one.
    assert!(
        !common::is_instrumented(&ir),
        "the probe crate was built with coverage instrumentation, so its IR is not the          library's — every function starts by bumping a profiling counter and nothing          folds. Something is still feeding flags into the nested build:

{}",
        common::flag_environment()
    );

    let mut failures = Vec::new();
    for (name, word, meaning) in expected() {
        let Some(body) = body_of(&ir, name) else {
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
