//! Entry point for the iai-callgrind benchmarks in `iai_codec_impl.rs`.
//!
//! The real implementation depends on `iai-callgrind`, which is only
//! available as a dev-dependency for non-wasm32 targets (see Cargo.toml).
//! `cargo build/check --target wasm32-unknown-unknown --all-targets` still
//! needs this crate to produce a valid binary, so on wasm32 we compile an
//! empty stub instead of the real benchmarks.

#[cfg(not(target_arch = "wasm32"))]
include!("impl/iai_codec.rs");

#[cfg(target_arch = "wasm32")]
fn main() {}
