//! Shared by the two codegen tests, both of which work by asking the compiler to emit
//! the IR for a *fresh* build and reading it back.

use std::process::Command;

/// A `cargo` invocation insulated from the build that is running it.
///
/// The IR these tests read has to be the library's own. Anything the outer build injects
/// through the environment ends up in it — most sharply under `cargo llvm-cov` (the
/// coverage job), whose flags carry `-C instrument-coverage`: that puts a profiling
/// counter at the top of every function, including the ones expected to have folded away
/// to a constant, so the tests would be reading instrumented code and reporting the
/// instrumentation as a regression.
///
/// `RUSTFLAGS` is *set empty* rather than removed, because emptying it also shadows any
/// `build.rustflags`/`target.*.rustflags` from a config file, which removing it would
/// not. `CARGO_ENCODED_RUSTFLAGS` outranks it, so that one has to go.
pub fn nested_cargo() -> Command {
    let mut cargo = Command::new(env!("CARGO"));
    cargo
        .env("RUSTFLAGS", "")
        .env("RUSTDOCFLAGS", "")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTDOCFLAGS")
        .env_remove("CARGO_BUILD_RUSTFLAGS")
        .env_remove("CARGO_BUILD_RUSTDOCFLAGS")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("LLVM_PROFILE_FILE");

    // `target.<triple>.rustflags` also has an environment form, and the triple is not
    // something to guess at.
    for (key, _) in std::env::vars() {
        if key.starts_with("CARGO_TARGET_") && key.ends_with("_RUSTFLAGS") {
            cargo.env_remove(key);
        }
    }
    cargo
}

/// Every environment variable that could be feeding flags into the nested build, as this
/// process sees them. Reported when a codegen test fails, because the likeliest reason for
/// IR that does not look like the library's is that something upstream is rewriting it —
/// and that is otherwise invisible from a CI log.
pub fn flag_environment() -> String {
    let mut found: Vec<String> = std::env::vars()
        .filter(|(k, _)| {
            k.contains("RUSTFLAGS")
                || k.contains("RUSTDOCFLAGS")
                || k.contains("PROFILE")
                || k.contains("COV")
                || k == "RUSTC"
                || k.contains("WRAPPER")
        })
        .map(|(k, v)| format!("  {}={}", k, v))
        .collect();
    found.sort();
    if found.is_empty() {
        "  (nothing set)".to_owned()
    } else {
        found.join("\n")
    }
}

/// Whether the IR carries coverage instrumentation, which means the flags above got
/// through anyway: every function then starts by bumping a profiling counter, so nothing
/// folds and the IR is not the library's.
pub fn is_instrumented(ir: &str) -> bool {
    ir.contains("__profc")
}
