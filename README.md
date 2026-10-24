# rust-compress

A fast command-line image compressor for **PNG**, **JPEG**, **WebP** and **AVIF** (GIF files can be converted). The compression engine is written from scratch in Rust — DEFLATE, PNG, JPEG, GIF, WebP (lossless and lossy) and AVIF (AV1) codecs, Huffman coding, DCT, color quantization — with no codec crates. Only [`clap`](https://crates.io/crates/clap) (argument parsing) and [`rayon`](https://crates.io/crates/rayon) (parallelism) are dependencies.

## Install

```
curl -fsSL https://raw.githubusercontent.com/coder-11125/rust-compress/main/install.sh | sh
```

Installs the `rc` binary for Linux or macOS (x86_64 or arm64) from the latest GitHub release. It goes to `/usr/local/bin` when that directory is writable, otherwise to `~/.local/bin`. To pin a release or choose a directory:

```
curl -fsSL https://raw.githubusercontent.com/coder-11125/rust-compress/main/install.sh | RC_VERSION=v0.1.0 RC_INSTALL_DIR="$HOME/bin" sh
```

Windows users can download the `.zip` asset from the [releases page](https://github.com/coder-11125/rust-compress/releases). To build from source instead, see [Build](#build).

## Quick start

```
rc photo.jpg                 # -> photo.min.jpg next to the original
rc ./images -r -o ./out      # whole tree, mirrored into ./out
rc shot.png --lossy          # palette quantization: usually 50-70% smaller
rc *.png -l 9 --in-place     # squeeze harder (slow), only replace if smaller
rc logo.png -f jpeg -q 85    # convert while compressing
rc photo.jpg --target-ssim 0.95   # lowest quality that still scores SSIM >= 0.95
rc shot.png -f webp          # lossless WebP: ~20% smaller than the optimised PNG
rc big.jpg --resize 1920x    # shrink to fit 1920 px wide (never enlarges)
rc photo.jpg --keep-metadata # keep EXIF / XMP / comments
cat in.png | rc - > out.png  # stdin to stdout
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
- Chooses the scanline filter strategy by trial-compressing a sample of the image with a real LZ77 level (the usual sum-of-abs heuristic is often wrong for screenshots/graphics). Candidates: the five fixed filters, per-row min-sum, entropy and *distinct-bigram* heuristics, and at `-l 9` a **brute-force search** that picks each row's filter by actually compressing it together with the preceding rows (skipped above 48 MB).
- Own DEFLATE: hash-chain LZ77 with lazy matching (levels 1-6, compressed in parallel segments) and Zopfli-style optimal parsing (levels 7-9). The optimal parser finds matches with a **binary-tree match finder** (every match length at its nearest distance, no chain-length cliff), then runs a shortest path over them with an iteratively refined cost model, and adaptive block splitting. The refinement is started from two seeds — the longest-match parse and a parse priced from the data's own byte histogram (which avoids a slow drift on noisy data, where a greedy seed is full of coincidental short matches) — and the cheaper result wins.
- `--lossy` quantizes to a ≤256-color palette: exact color histogram, median cut, k-means refinement, and edge-aware Floyd-Steinberg dithering (`--colors N`, `--dither 0..1`).

**WebP (lossy)** — a from-scratch VP8 key-frame encoder and decoder (`-f webp --lossy -q N`):
- The decoder is bit-exact with libwebp (planes and RGB output, including the "fancy" chroma upsampling, loop filters, segments, multiple partitions and `ALPH`).
- The encoder does rate-distortion mode decisions (16x16, 4x4 and chroma intra modes), quantizes with deadzone biasing, tunes the token probabilities from first-pass statistics, and searches the loop-filter level. Its streams decode to the identical picture in libwebp. At the same `-q` on twelve real images it lands within a few percent of libwebp's size at equal or slightly higher PSNR.
- Transparency is stored losslessly in an `ALPH` chunk (best of four prediction filters, coded with the VP8L encoder); hidden pixels are smoothed to be cheap. Lossy WebP inputs are only re-encoded with `--lossy`; otherwise they are left untouched.

**AVIF** — a from-scratch AV1 still-picture decoder and intra encoder inside a HEIF container (`-f avif -q N`):
- The decoder implements the AV1 intra-frame toolset: all partition types, intra modes (directional with edge filtering and upsampling, smooth, Paeth, filter-intra, chroma-from-luma, palette), every transform size and type, segmentation, delta-q/lf, quantizer matrices, the deblocking filter, CDEF, loop restoration (Wiener and self-guided), super-resolution and film grain, in 8/10/12 bits and 4:0:0/4:2:0/4:2:2/4:4:4. Output matches dav1d bit for bit on streams from libaom, SVT-AV1, rav1e and ffmpeg, and libavif/Pillow pixel for pixel (±1 from chroma upsampling). Containers: single images, grids, alpha auxiliary images (also premultiplied), `clap`/`irot`/`imir`, `colr` nclx and ICC, EXIF and XMP. Inter prediction is not implemented: for an animated (`avis`) file only key-frame-only sequences decode. **Intra block copy** (`allow_intrabc`, with its displacement vector prediction, variable transform trees and inter transform sets) is decoded bit for bit like dav1d. Bit depths above 8 and 4:2:2 are implemented but have had less cross-checking than 8-bit 4:2:0 / 4:4:4.
- The encoder is lossy only. It searches partitions, intra modes with angle deltas, chroma-from-luma and transform sizes per superblock by rate-distortion cost using the real adaptive entropy-coder state, then picks the deblocking level and CDEF strengths by measured error, tries **intra block copy** for pictures with repeated content (screenshots, text, UI: a hash of every 8x8 window finds exact duplicates, and the copy variant is kept only if it wins in rate-distortion terms, since block copy frames cannot use the loop filters), and chooses a loop-restoration filter (Wiener or self-guided) per unit. Restoration parameters live inside the tile data, so the tiles are written a second time from the decisions already made. Output is 8-bit or 10-bit (`--bit-depth auto|8|10`; auto keeps 16-bit sources at 10 bits) 4:2:0 (4:4:4 at `-q 90` and up, or `--subsampling 444`) or monochrome for gray images; transparency is a monochrome auxiliary image. ICC, EXIF (with `--keep-metadata`) and XMP are stored. Streams decode identically in dav1d, libaom and Pillow.
- AVIF inputs are left untouched unless `--lossy` is given, since re-encoding only adds generation loss.
- The encoder is considerably slower than the JPEG/WebP ones and about the size of libaom (`cpu-used 4`) at equal RGB PSNR: on eight photos and screenshots, −6.5% on average, from −26% to +4.3% per image, both decoded through the same dav1d/ffmpeg path; palette, filter-intra and inter tools are not implemented in the encoder.

**WebP (lossless)** — full VP8L encoder and decoder:
- Encoder tries a palette transform (bundling up to 8 pixels per symbol) or subtract-green + per-tile predictor + cross-color decorrelation, ranks the pipelines with a cheap estimate and fully encodes the best ones.
- LZ77 with the format's 2-D distance codes, an optimal (cost-model driven) parse, a colour cache sized by trial, and **entropy-clustered prefix-code groups**: image tiles with similar statistics share one set of codes.
- 8-bit only: 16-bit inputs are reduced to 8-bit (the report note says so). Images are limited to 16384×16384 by the format. Lossy WebP (VP8) is decoded for conversions or an explicit `--lossy` re-encode; when the format is kept it is left untouched, since recompressing it would only add generation loss. Animated WebP is treated the same way.

**Other formats**
- GIF files (still images: palettes, interlacing, transparency) convert to PNG, JPEG or WebP with `-f`. Animated GIFs convert to PNG or JPEG only as a refusal with a clear message; with `-f avif` they become animated AVIF. There is no GIF output, so directory runs only pick up `.gif` files when `-f` names a target.
- CMYK and YCCK JPEGs are transcoded losslessly in the DCT domain (identical coefficients, Adobe colour flag kept); converting or resizing them goes through RGB.
- Animated PNGs are passed through unchanged instead of failing the batch, unless `-f avif` asks for an animated AVIF.
- **Animations** — animated GIF, APNG, animated WebP and animated AVIF inputs convert to an animated AVIF (`-f avif`; with `--lossy` for AVIF inputs): full-canvas frames with the files' disposal and blend rules, per-frame durations and looping, alpha as a second track, every frame a key frame (no inter prediction yet), frames encoded in parallel. Output decodes in libavif and dav1d. Animated AVIF files whose frames use inter prediction (most files from libavif or ffmpeg) cannot be read yet. A finite repeat count is played forever.

**Metadata** — only color information survives by default (ICC profile, PNG color chunks). `--keep-metadata` also keeps EXIF, XMP, JPEG comments / Photoshop blocks and PNG text, time and pixel-density chunks. EXIF and XMP carry across PNG ↔ JPEG ↔ WebP conversions; when an orientation is baked into the pixels the EXIF orientation is reset, its recorded dimensions are corrected and a now-stale thumbnail is dropped.

**Resizing** — `--resize WxH | Wx | xH | P%` shrinks to fit (aspect ratio kept, never enlarges) with a Lanczos-3 filter on premultiplied alpha, for 8- and 16-bit images. Output matches Pillow's Lanczos to within ~0.1 level on average. Because the result must differ from the original, the never-grow rule does not apply.

**Pipes** — `-` as the input reads one image from standard input; `-o -` (the default for stdin) writes to standard output, with all reporting moved to standard error.

**Safety** — an output is never larger than its input (the original bytes are kept instead), files are written atomically, existing outputs are not overwritten without `--force`, ICC profiles and PNG color chunks are kept (`--strip-icc` to drop them), animations are never flattened, and decoders are hardened against truncated/corrupt/hostile input (see *Testing*).

## Usage

```
rc [OPTIONS] <INPUT>...

  -r, --recursive         descend into subdirectories
  -o, --output <PATH>     output file (one input file) or directory (mirrors the input tree)
      --in-place          replace originals, only when the result is smaller
  -q, --quality <1-100>   JPEG quality ceiling (default 80)
  -f, --format <FMT>      keep | png | jpeg | webp | avif (default keep)
      --subsampling <S>   auto | 444 | 420
      --bit-depth <D>     AVIF output: auto | 8 | 10 (auto: 10 for 16-bit sources)
      --lossy             PNG palette quantization, lossy WebP with -f webp, or re-encode an AVIF input (-q = quality)
      --colors <2-256>    with --lossy (default 256)
      --dither <0.0-1.0>  with --lossy (default 0.5)
  -l, --level <0-9>       PNG / WebP effort (default 7; PNG: 1-6 fast, 7-9 optimal parsing, 9 slowest)
      --no-trellis        JPEG: skip rate-distortion quantization (faster, larger)
      --target-ssim <S>   JPEG: smallest quality with SSIM >= S (0-1; 0.95 small, 0.97 good, 0.99 near-lossless)
      --keep-hidden-pixels  PNG: keep exact color of fully transparent pixels
      --strip-icc         drop ICC profiles / PNG color chunks
      --keep-metadata     keep EXIF, XMP, comments, PNG text/time/density
      --resize <SIZE>     shrink to fit: WxH, Wx, xH or P% (never enlarges)
      --background <HEX>  color for flattening transparency (default ffffff)
  -j, --jobs <N>          worker threads (default: all cores)
      --dry-run           report savings without writing
      --force             overwrite existing outputs
      --quiet             only errors and the summary line
```

`<INPUT>` may be `-` (standard input) and `-o -` writes standard output. Without `-o`/`--in-place`, `photo.jpg` becomes `photo.min.jpg`; files already named `*.min.*` are skipped when walking directories. The exit status is non-zero if any file failed; the rest of the batch still runs.

## Head-to-head with established tools

JPEG and lossy-PNG rows: five real JPEGs and three PNG screenshots (mozjpeg 4.1.5, oxipng 10.2.1 with `-t 1`, pngquant 3.0.3). Timings are single-threaded **CPU time** on a loaded 2-core laptop, so treat them as indicative; sizes are exact.

Lossless-PNG and WebP rows: 12 PNGs (screenshots, diagrams, UI art, a 16-bit image, a gradient, a 6 MB RGBA illustration) against oxipng 9.1.1 (as packaged in `pyoxipng`), compared with `--keep-hidden-pixels` so neither side gets credit for rewriting invisible pixels. Sizes are exact; times are wall-clock on 4 cores for the whole process.

| Task | rust-compress vs. the established tool |
|---|---|
| Lossy JPEG (`-q 60/70/80`) | **At parity with mozjpeg at equal PSNR**: mean +1.4% / −0.4% / +0.6% size (per file −10% … +7%) |
| Lossy JPEG speed | 51-246 ms vs `cjpeg` 64-410 ms (cjpeg excludes decoding; with trellis) |
| Lossless JPEG (`-q 100`) | +0.6% vs `jpegtran -optimize -progressive` (range +0.1% … +1.0%), **faster** (55-206 ms vs 86-236 ms) |
| Lossless PNG, default `-l 7` | **Equal or smaller than `oxipng -o2` on 8 of 12 files**, worst +1.5%, best −19.2%; 12-image total −1.5%. Wall time 0.09-1.3 s per file vs 0.04-2.0 s for `-o2` (faster on the larger files) |
| Lossless PNG, `-l 9` | **Equal or smaller than `oxipng -o4` on 11 of 12 files**, worst +0.7%, best −19.0%; total −1.6%. Wall time 0.3-6.9 s per file vs 0.1-3.4 s for `-o4` (slower on the largest image: 6.9 s vs 3.4 s) |
| Lossless PNG, `-l 6` | +11% vs `oxipng -o2` in total (worst +16.6%), at a fraction of the CPU: the fast levels use hash-chain LZ77 only |
| Lossless WebP (`-f webp`, `-l 7`) | **+6.4% in total vs libwebp at its maximum effort** (`method=6, quality=100`, via Pillow): within +0.4% … +14.5% on 11 of 12 images, one image at +31%. Roughly **2-10× faster** in wall time than libwebp's max effort. About **22% smaller than our optimised PNG** of the same pixels (excluding two 16-bit images that WebP stores at 8-bit) |
| Lossy PNG (`--lossy`) | **Better than pngquant**: 0.4-1.0 dB higher PSNR at −4.3%, −4.5%, −4.0% size; 1.5-1.9× its CPU time |
| PNG with hidden transparent pixels | tools that keep invisible RGB (oxipng by default) can be orders of magnitude larger: 580 KB → 4 KB on a noisy-under-alpha test image |

So it matches the established tools on JPEG, beats pngquant on lossy PNG, and matches or beats oxipng on lossless PNG at `-l 7`/`-l 9`. Against weaker baselines it is far ahead (e.g. 14-28% smaller than PIL's `optimize=True` PNGs).

What it offers beyond ratio: no C dependencies, one binary for all of these formats, a coefficient-domain JPEG pipeline, pipe support, and parallel batch processing.

## Testing

`cargo test` runs the unit tests and integration suites, which cross-check the decoders against the `image` crate (and, for WebP and CMYK, against files made by libwebp and libjpeg-turbo) and verify encoder output with independent decoders. `tests/hostile.rs` feeds every decoder and encoder path truncated, bit-flipped and header-corrupted variants of every fixture and requires no panic, no hang and (for WebP) an exact re-encode round trip. This is deterministic mutation testing, not coverage-guided fuzzing. For a deeper sweep: `HOSTILE_ROUNDS=20000 cargo test --release --test hostile`.

## Build

```
cargo build --release        # binary: target/release/rc
cargo test                   # unit + integration tests (cross-checked against the `image` crate)
cargo clippy --all-targets -- -D warnings
```

Requires a stable Rust toolchain (edition 2024); `rust-toolchain.toml` pins `stable` with clippy/rustfmt. No C compiler or system libraries are needed.

## Layout

```
src/util/       bit I/O, CRC-32 / Adler-32, canonical Huffman (shared by DEFLATE, JPEG and WebP)
src/deflate/    inflate, LZ77 matchers (hash chains; binary tree for levels 7-9), block encoder, optimal parser (levels 7-9)
src/png/        decoder (all color types/depths, Adam7), filters, encoder (reduction + strategy search), quantizer
src/jpeg/       decoder (baseline + progressive), DCT, baseline/progressive encoders, trellis quantization, quality search
src/avif/       HEIF/ISOBMFF container, AV1 decoder (entropy, intra prediction, transforms, loop filters, CDEF, restoration, film grain) and intra encoder
src/webp/       RIFF container, VP8L decoder and encoder (transforms, LZ77 + optimal parse, tile-group clustering)
src/gif.rs      GIF decoder (still images)
src/exif.rs     EXIF editing for kept metadata (orientation, dimensions, thumbnail)
src/resize.rs   Lanczos-3 downscaling
src/metrics.rs  SSIM
src/pipeline.rs per-file policy: read → decode/transcode → never-grow check → atomic write
examples/       png_probe, deflate_probe, stream_probe, jpeg_probe: developer tools for tuning ratio and speed
tests/          codec round trips, reference-decoder cross-checks, hostile input, CLI end-to-end
```

## Limitations

- Formats: PNG, JPEG, WebP (lossless and lossy) and AVIF in and out; GIF in. Animated output only as AVIF. JPEG: 8-bit Huffman, 1, 3 or 4 components (no arithmetic coding or 12-bit).
- WebP output is 8-bit, and AVIF output is 8-bit unless the source is 16-bit or `--bit-depth 10` is given.
- CMYK JPEGs are never requantized (`-q` does not apply to them); YCCK decoding is covered by unit tests but has no real-file fixture.
- EXIF/XMP/comments are dropped unless `--keep-metadata` is given. An EXIF block too large for a JPEG segment (over 64 KB) cannot be carried into a JPEG.
- Resizing only shrinks and filters in gamma-encoded space.
