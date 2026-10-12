# rust-compress

A fast command-line image compressor for **PNG** and **JPEG**. The compression engine is written from scratch in Rust — DEFLATE, PNG and JPEG codecs, Huffman coding, DCT, color quantization — with no codec crates. Only [`clap`](https://crates.io/crates/clap) (argument parsing) and [`rayon`](https://crates.io/crates/rayon) (parallelism) are dependencies.

```
rust-compress photo.jpg                 # -> photo.min.jpg next to the original
rust-compress ./images -r -o ./out      # whole tree, mirrored into ./out
rust-compress shot.png --lossy          # palette quantization: usually 50-70% smaller
rust-compress *.png -l 9 --in-place     # squeeze harder (slow), only replace if smaller
rust-compress logo.png -f jpeg -q 85    # convert while compressing
rust-compress photo.jpg --target-ssim 0.95   # lowest quality that still scores SSIM >= 0.95
```

## What it does

**JPEG** — works in the DCT domain, so most files are never decoded to pixels:
- Sources at or below the target quality (`-q`, default 80) are *transcoded losslessly*: same pixels, better entropy coding. The progressive scan layout is searched (72 candidate layouts, costed from symbol statistics) and compared against baseline; the smallest wins.
- Higher-quality sources are re-quantized directly on the coefficients (never to a finer table than the source had) with **trellis quantization**: a per-block dynamic program that trades distortion against the bit cost of each run/size symbol, using Huffman code lengths measured on a first pass (`--no-trellis` to skip).
- `--target-ssim S` replaces the guess-a-quality step: it bisects for the lowest quality whose reconstruction still scores at least SSIM `S` against the original (mean 8×8-window SSIM over all channels). The needed quality varies enormously between images — at 0.95 the sample files needed anywhere from q17 to q85 — so this saves bytes on easy images and protects hard ones.
- Pixels are only touched when something must change: chroma subsampling (4:4:4 → 4:2:0 below q90, or `--subsampling`), EXIF orientation (baked into the pixels because EXIF is stripped), or PNG → JPEG conversion.

**PNG** — lossless by default, bit-exact pixels:
- Reduces the representation: drops unused alpha, RGB → gray, ≤256 colors → palette, 1/2/4-bit depths, 16 → 8-bit when lossless.
- Fully transparent pixels get their invisible color rewritten to whatever compresses best (zeroed for palettes, copied from the left/above neighbor for filters); `--keep-hidden-pixels` preserves the exact bytes. Palette order is chosen among four orderings by trial compression.
- Chooses the scanline filter strategy by trial-compressing a sample of the image (the usual sum-of-abs heuristic is often wrong for screenshots/graphics).
- Own DEFLATE: hash-chain LZ77 with lazy matching (levels 1-6, compressed in parallel segments) and Zopfli-style optimal parsing (levels 7-9: shortest path over all matches with an iteratively refined cost model, adaptive block splitting, and direct tests of the distances earlier chunks found productive — e.g. the row above — which hash chains miss in flat regions).
- `--lossy` quantizes to a ≤256-color palette: exact color histogram, median cut, k-means refinement, and edge-aware Floyd-Steinberg dithering (`--colors N`, `--dither 0..1`).

**Safety** — an output is never larger than its input (the original bytes are kept instead), files are written atomically, existing outputs are not overwritten without `--force`, ICC profiles and PNG color chunks are kept (`--strip-icc` to drop them), animated PNGs are refused rather than flattened, and decoders are hardened against truncated/corrupt/hostile input.

## Usage

```
rust-compress [OPTIONS] <INPUT>...

  -r, --recursive         descend into subdirectories
  -o, --output <PATH>     output file (one input file) or directory (mirrors the input tree)
      --in-place          replace originals, only when the result is smaller
  -q, --quality <1-100>   JPEG quality ceiling (default 80)
  -f, --format <FMT>      keep | png | jpeg (default keep)
      --subsampling <S>   auto | 444 | 420
      --lossy             PNG: allow palette quantization
      --colors <2-256>    with --lossy (default 256)
      --dither <0.0-1.0>  with --lossy (default 0.5)
  -l, --level <0-9>       PNG effort (default 7; 1-6 fast, 7-9 optimal parsing, 9 slowest)
      --no-trellis        JPEG: skip rate-distortion quantization (faster, larger)
      --target-ssim <S>   JPEG: smallest quality with SSIM >= S (0-1; 0.95 small, 0.97 good, 0.99 near-lossless)
      --keep-hidden-pixels  PNG: keep exact color of fully transparent pixels
      --strip-icc         drop ICC profiles / PNG color chunks
      --background <HEX>  color for flattening transparency (default ffffff)
  -j, --jobs <N>          worker threads (default: all cores)
      --dry-run           report savings without writing
      --force             overwrite existing outputs
      --quiet             only errors and the summary line
```

Without `-o`/`--in-place`, `photo.jpg` becomes `photo.min.jpg`; files already named `*.min.*` are skipped when walking directories. The exit status is non-zero if any file failed; the rest of the batch still runs.

## Head-to-head with established tools

Measured on five real JPEGs and three PNG screenshots (mozjpeg 4.1.5, oxipng 10.2.1 with `-t 1`, pngquant 3.0.3, zopflipng from upstream git). Timings are single-threaded **CPU time** on a loaded 2-core laptop, so treat them as indicative; sizes are exact.

| Task | rust-compress vs. the established tool |
|---|---|
| Lossy JPEG (`-q 60/70/80`) | **At parity with mozjpeg at equal PSNR**: mean +1.4% / −0.4% / +0.6% size (per file −10% … +7%) |
| Lossy JPEG speed | 51-246 ms vs `cjpeg` 64-410 ms (cjpeg excludes decoding; with trellis) |
| Lossless JPEG (`-q 100`) | +0.6% vs `jpegtran -optimize -progressive` (range +0.1% … +1.0%), **faster** (55-206 ms vs 86-236 ms) |
| Lossless PNG, default `-l 7` | +0.2% … +2.3% vs `oxipng -o2` (+1.7%, +0.2%, +2.3% on the three files); 0.85-1.6 s CPU vs 0.6-1.3 s |
| Lossless PNG, `-l 9` | +1.3% … +3.0% vs `zopflipng` (15 iterations); 4.5-8 s, slower than `oxipng -o4` |
| Lossless PNG, `-l 6` | ~12% larger than `oxipng -o2`, but 4-6× less CPU |
| Lossy PNG (`--lossy`) | **Better than pngquant**: 0.4-1.0 dB higher PSNR at −4.3%, −4.5%, −4.0% size; 1.5-1.9× its CPU time |
| PNG with hidden transparent pixels | tools that keep invisible RGB (oxipng by default) can be orders of magnitude larger: 580 KB → 4 KB on a noisy-under-alpha test image |

So it matches the established tools on JPEG, beats pngquant on lossy PNG, and is a couple of percent behind oxipng/zopflipng on lossless PNG. Against weaker baselines it is far ahead (e.g. 14-28% smaller than PIL's `optimize=True` PNGs).

What it offers beyond ratio: no C dependencies, one binary for both formats, a coefficient-domain JPEG pipeline, and parallel batch processing.

## Build

```
cargo build --release        # binary: target/release/rust-compress
cargo test                   # unit + integration tests (cross-checked against the `image` crate)
cargo clippy --all-targets -- -D warnings
```

Requires a stable Rust toolchain (edition 2024); `rust-toolchain.toml` pins `stable` with clippy/rustfmt. No C compiler or system libraries are needed.

## Layout

```
src/util/       bit I/O, CRC-32 / Adler-32, canonical Huffman (shared by DEFLATE and JPEG)
src/deflate/    inflate, LZ77 matcher, block encoder, optimal parser (level 9)
src/png/        decoder (all color types/depths, Adam7), filters, encoder (reduction + strategy search), quantizer
src/jpeg/       decoder (baseline + progressive), DCT, baseline/progressive encoders, trellis quantization, quality search
src/metrics.rs  SSIM
src/pipeline.rs per-file policy: read → decode/transcode → never-grow check → atomic write
examples/       png_probe, deflate_probe, stream_probe, jpeg_probe: developer tools for tuning ratio and speed
tests/          codec round trips, reference-decoder cross-checks, CLI end-to-end
```

## Limitations

- Formats: PNG and JPEG only (no WebP/AVIF/GIF). JPEG: 8-bit Huffman, 1 or 3 components (no CMYK, arithmetic coding or 12-bit). Animated PNG is refused.
- EXIF/XMP/comments are stripped (orientation is applied to the pixels; ICC is kept).
- No resizing yet.
