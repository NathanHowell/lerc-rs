# Fuzzing

[cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets for the decoder
and the encoder. They need a nightly toolchain:

```sh
cargo install cargo-fuzz
cargo +nightly fuzz run decode fuzz/corpus/decode fuzz/seeds/decode   # from the repository root
cargo +nightly fuzz run roundtrip
```

| Target | Checks |
| --- | --- |
| `decode` | Arbitrary bytes never panic, overflow, hang or allocate without bound in `decode_info` or `decode`. |
| `roundtrip` | Small arbitrary images encode, and decode within the requested tolerance. |

The first byte of a `decode` input is a mode: when bit 0 is set, the Fletcher-32
checksum of a Lerc2 blob is recomputed after the fuzzer's mutations, so inputs
reach the tile, mask and Huffman parsers instead of stopping at the checksum.
`seeds/decode` holds small valid blobs with that bit set. Name `corpus/decode`
(ignored by git) first: libFuzzer writes the inputs it finds into the first
directory it is given. A Lerc1 seed helps too:
`esri-lerc/testData/world.lerc1`, if the submodule is checked out.

`decode` is run with `-rss_limit_mb=2048 -max_len=70000` in CI. The target does
not decode blobs whose header claims more than 2^24 samples: a few header bytes
can describe a huge constant image, and decoding it only measures the machine.

A crash is a decoder bug. Add the input as a test in `tests/malformed.rs`
before fixing it.
