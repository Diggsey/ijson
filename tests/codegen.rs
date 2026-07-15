//! Codegen test: the value dispatch must stay devirtualized.
//!
//! `IValue` routes every operation through `ReprTag::with`, which hands each match
//! arm's *concrete* representation to a closure. Because the coercion to `&dyn
//! ValueRepr` therefore happens at a per-arm call site, its vtable is a compile-time
//! constant that constant-folding resolves into a direct call.
//!
//! Returning a single `&dyn` from the match instead — as an earlier `repr()` did —
//! merges every arm's vtable into a phi the optimizer cannot see through, and each
//! operation silently pays an indirect vtable call. Nothing else in the test suite
//! would notice: the behaviour is identical, only the codegen is worse. Hence this
//! test.
//!
//! It builds the library to LLVM IR and asserts that no ijson function calls through a
//! function pointer. There is exactly one deliberate exception: the `hash` impls take
//! `&mut dyn Hasher` (a trait-object method cannot be generic, so `IValue: Hash` erases
//! the concrete hasher once, at the top), so their `Hasher` calls really are dynamic.
//!
//! Being a codegen test, this is inherently sensitive to the compiler: it asserts only
//! the coarse property (no indirect calls), not any particular instruction sequence.

mod common;

use std::path::PathBuf;

/// The `hash` impls are the only ijson functions allowed to call through a function
/// pointer: they dispatch on `&mut dyn Hasher`, by design.
///
/// Matched across both symbol manglings, because they differ in how they join the trait
/// to the method and the compiler picks between them by target and version — the legacy
/// scheme writes `..ValueRepr$GT$4hash..`, and `v0` writes `..9ValueRepr4hash`. Keying
/// on either spelling alone silently stops matching on the other, which does not weaken
/// the test into passing — it turns every legitimate `hash` into a reported offender.
fn erases_a_hasher(mangled: &str) -> bool {
    // `4hash` is the length-prefixed method name in `v0`, and also appears in the legacy
    // scheme (whose own suffix is `17h<hex>`, not `4hash`), so it does not over-match.
    (mangled.contains("ValueRepr") || mangled.contains("InlineValue")) && mangled.contains("4hash")
}

/// Builds the library in release with LLVM IR emitted, and returns the IR text.
fn emit_llvm_ir() -> String {
    // A target directory of our own, so the nested build does not contend with the
    // `cargo test` invocation that is running us.
    let target_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("codegen-ir");
    let status = common::nested_cargo()
        .args(["rustc", "--lib", "--release", "--target-dir"])
        .arg(&target_dir)
        .args(["--", "--emit=llvm-ir", "-Cdebuginfo=0"])
        .status()
        .expect("failed to run `cargo rustc`");
    assert!(status.success(), "`cargo rustc --emit=llvm-ir` failed");

    let ir_file = std::fs::read_dir(target_dir.join("release/deps"))
        .expect("read the build's deps directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "ll")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("ijson-"))
        })
        // Stale `.ll` files from earlier builds linger; take the freshest.
        .max_by_key(|path| path.metadata().and_then(|meta| meta.modified()).ok())
        .expect("no `ijson-*.ll` was emitted");

    std::fs::read_to_string(&ir_file).expect("read the emitted LLVM IR")
}

#[test]
#[cfg_attr(miri, ignore = "shells out to a compiler, which Miri cannot run")]
fn value_dispatch_is_devirtualized() {
    let ir = emit_llvm_ir();

    let mut function = String::new();
    let mut indirect = 0usize;
    let mut ijson_functions = 0usize;
    let mut saw_the_clone_dispatch = false;
    let mut saw_a_hasher_erasure = false;
    let mut offenders: Vec<String> = Vec::new();

    for line in ir.lines() {
        if line.starts_with("define") {
            // `define <attrs> @"<mangled>"(<args>) {`
            function = line
                .split_once('@')
                .map(|(_, tail)| tail.trim_start_matches('"'))
                .and_then(|tail| tail.split('(').next())
                .unwrap_or_default()
                .trim_end_matches('"')
                .to_owned();
            indirect = 0;
        } else if line == "}" {
            if function.contains("ijson") {
                ijson_functions += 1;
                saw_the_clone_dispatch |= function.contains("IValue") && function.contains("clone");
                saw_a_hasher_erasure |= erases_a_hasher(&function);
                if indirect > 0 && !erases_a_hasher(&function) {
                    offenders.push(format!("  {indirect} indirect call(s) in {function}"));
                }
            }
            function.clear();
        } else if !function.is_empty() && common::is_indirect_call(line) {
            indirect += 1;
        }
    }

    // Guard against a vacuous pass: if the emitted IR or the name mangling ever changes
    // shape, the scan above could quietly match nothing and assert success over an
    // empty set.
    assert!(
        ijson_functions > 50,
        "only {} ijson functions found in the emitted IR — the scan is not seeing the \
         library, so this test is not checking anything",
        ijson_functions
    );
    assert!(
        saw_the_clone_dispatch,
        "did not find `IValue`'s `Clone` impl among the {} ijson functions scanned — \
         the name matching is stale, so this test is not checking the dispatch",
        ijson_functions
    );

    // If the exception matcher goes stale (the symbol mangling differs by target and
    // compiler version), every legitimate `hash` becomes a reported offender. Say so
    // here, rather than leaving the reader to infer it from a list of false positives.
    assert!(
        saw_a_hasher_erasure,
        "did not recognise a single `hash` impl among the {} ijson functions scanned — \
         `erases_a_hasher` no longer matches the symbol mangling, so every legitimate \
         `hash` would be reported as an offender below",
        ijson_functions
    );

    assert!(
        offenders.is_empty(),
        "the value dispatch is no longer devirtualized: these ijson functions call \
         through a function pointer (i.e. a vtable), which should only happen in the \
         `hash` impls that erase `&mut dyn Hasher`.\n\n{}\n\nIf a `&dyn ValueRepr` was \
         reintroduced (e.g. a `repr()` returning one from a `match`), every arm's \
         vtable merges into a phi the optimizer cannot fold, and each operation pays an \
         indirect call. Dispatch by handing the *concrete* representation to a closure \
         (`ReprTag::with`) instead.",
        offenders.join("\n")
    );
}
