//! Shared by the two codegen tests, both of which work by asking the compiler to emit
//! the IR for a *fresh* build and reading it back.

use std::process::Command;

/// A `cargo` invocation insulated from the build that is running it.
///
/// The IR these tests read has to be the library's own. Anything the outer build injects
/// through the environment would end up in it — most sharply under `cargo llvm-cov`,
/// whose `RUSTFLAGS` carry `-C instrument-coverage`: that puts a profiling counter at the
/// top of every function, including the ones expected to have folded away to a constant,
/// so the tests would be reading instrumented code and reporting the instrumentation as a
/// regression.
pub fn nested_cargo() -> Command {
    let mut cargo = Command::new(env!("CARGO"));
    cargo
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTDOCFLAGS")
        .env_remove("CARGO_ENCODED_RUSTDOCFLAGS")
        .env_remove("LLVM_PROFILE_FILE");
    cargo
}
