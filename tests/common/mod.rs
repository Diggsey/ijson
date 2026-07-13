//! Shared by the two codegen tests, both of which work by asking the compiler to emit
//! the IR for a *fresh* build and reading it back.
//!
//! Each integration test compiles this module separately, so anything only one of them
//! uses is dead code in the other.
#![allow(dead_code)]

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

/// The callee of an LLVM IR call instruction: `@name` for a direct call, `%reg` for one
/// through a function pointer. `None` if the line is not a call.
///
/// `call` and `invoke` are matched as whole *tokens*, not as the substring `" call "` — a
/// line may or may not have been trimmed of its indentation, and `call fastcc void @f(..)`
/// at the start of a line would otherwise be missed entirely. (It was, which made an
/// earlier version of this quietly blind to every call it was supposed to find.)
///
/// The callee is the last token before the argument list, so cutting the line at its first
/// `(` and taking the final token finds it — for `tail call`, `invoke`, and a plain `call`
/// alike. Rust's mangled names never contain a literal parenthesis.
pub fn call_target(line: &str) -> Option<&str> {
    let head = &line[..line.find('(')?];
    let mut tokens = head.split_whitespace();
    tokens
        .any(|token| token == "call" || token == "invoke")
        .then(|| tokens.last())?
}

/// Whether the line calls through a function *pointer* — i.e. a vtable.
pub fn is_indirect_call(line: &str) -> bool {
    call_target(line).is_some_and(|callee| callee.starts_with('%'))
}

/// The instruction lines in the body of `@name`, or `None` if it is not defined. Labels,
/// comments and blank lines are dropped; what is left is what the function *does*.
pub fn body_of<'a>(ir: &'a str, name: &str) -> Option<Vec<&'a str>> {
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
        if line.is_empty() || line.starts_with(';') || line.ends_with(':') {
            continue;
        }
        body.push(line);
    }
    None
}

/// The names of the functions a body calls, ignoring LLVM's own intrinsics (`llvm.*` is
/// not a call in any meaningful sense — `llvm.trunc.f64` is an instruction) and the
/// anonymous constants a call's arguments may point at.
pub fn called_symbols<'a>(body: &[&'a str]) -> Vec<&'a str> {
    let mut names: Vec<&str> = body
        .iter()
        .filter_map(|line| call_target(line))
        .filter_map(|callee| callee.strip_prefix('@'))
        .map(|name| name.trim_matches('"'))
        .filter(|name| !name.starts_with("llvm.") && !name.starts_with("anon."))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}
