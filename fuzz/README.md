# Fuzzing

Eight libFuzzer targets cover every parser that touches untrusted bytes:

| target | covers |
| --- | --- |
| `inflate` | DEFLATE / zlib decoding, stream statistics |
| `png` | PNG and APNG decoding, plus a re-encode round trip |
| `jpeg_decode` | every JPEG variant (baseline, progressive, arithmetic, 12-bit, lossless, hierarchical, CMYK) to pixels |
| `jpeg_coefs` | the lossless-recompression path: coefficient parsing, requantize, both entropy writers |
| `webp` | VP8, VP8L, alpha, animated WebP, plus a re-encode round trip |
| `gif` | GIF and animated GIF |
| `avif` | AVIF container, image sequences, and the AV1 decoder behind them |
| `av1_obu` | a bare AV1 OBU stream, skipping the container |

Run one locally (needs nightly and `cargo install cargo-fuzz`):

    cargo +nightly fuzz run png fuzz/seeds/png -- -max_total_time=60

with a seed directory such as `fixtures/`. A crash reproduces with `cargo +nightly fuzz run <target> <crash-file>`.

`oss-fuzz/` holds the project files for continuous fuzzing on Google's OSS-Fuzz.

`.clusterfuzzlite/` and the `cflite_*.yml` workflows run the same targets in GitHub Actions: ten minutes on pull
requests that touch a codec, a long batch run twice a day, and a weekly corpus prune.
