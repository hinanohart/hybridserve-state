#![no_main]

//! Coverage-guided fuzz target for the `.hss` parser.
//!
//! The parser must never panic on adversarial input — it may only return
//! `Ok` or `Err`. Run with a nightly toolchain and `cargo-fuzz`:
//!
//! ```bash
//! cargo install cargo-fuzz
//! cargo +nightly fuzz run parse
//! ```
//!
//! CI does not run this (it needs nightly + LLVM sanitizers); the workspace
//! `cargo test -p hybridserve_state_core` instead runs an in-process fuzz smoke
//! (`parse_never_panics_on_random_input`).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = hybridserve_state_core::parse(data);
});
