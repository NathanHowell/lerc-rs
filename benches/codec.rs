//! Entry point for the Criterion wall-clock benchmarks in `codec_impl.rs`.
//!
//! The real implementation depends on `criterion` and `lerc-cpp-ref`, which
//! are only available as dev-dependencies for non-wasm32 targets (see
//! Cargo.toml). `cargo build/check --target wasm32-unknown-unknown
//! --all-targets` still needs this crate to produce a valid binary, so on
//! wasm32 we compile an empty stub instead of the real benchmarks.

#[cfg(not(target_arch = "wasm32"))]
include!("impl/codec.rs");

#[cfg(target_arch = "wasm32")]
fn main() {}
