# rust-compress

A fast command-line image compressor for **PNG**, **JPEG**, **WebP**, **AVIF** and **GIF**. The compression engine is written from scratch in Rust — DEFLATE, PNG, JPEG, GIF, WebP (lossless and lossy) and AVIF (AV1) codecs, Huffman coding, DCT, color quantization — with no codec crates. Only [`clap`](https://crates.io/crates/clap) (argument parsing) and [`rayon`](https://crates.io/crates/rayon) (parallelism) are dependencies.

## Why rc (TL;DR)

**One binary, five image formats, no codec libraries.** Every codec (DEFLATE, PNG, JPEG, GIF, WebP, AV1/AVIF) is written from scratch in Rust, so there is nothing to install besides `rc` itself: no libjpeg, libpng, libwebp or libaom, no C compiler to build it, and nothing to patch when a C library has a security bug. It beats or matches the established single-purpose tools where it matters most, and is honest about where it still trails.

**What you get**

- **A top-tier PNG optimizer.** Lossless PNG with a Zopfli-style optimal parser, a binary-tree match finder and brute-force per-row filter search: smaller than `oxipng -o2` on 11 of 14 files and within about 2% of `oxipng -o max -Z`/zopflipng at a twentieth of their time; 6-9% smaller than optipng, advpng and pngcrush. Invisible pixels under transparency are rewritten for size (580 KB → 4 KB on a noisy-under-alpha test image where tools that keep the hidden RGB stay huge).
- **JPEG at parity with mozjpeg, perceptually.** Tuned against SSIMULACRA2, not just PSNR: on average 0.8-1.2% *smaller* than mozjpeg at equal perceptual quality (see the numbers below). Existing JPEGs are processed in the DCT domain, so most are never decoded to pixels, and a source already at or below the target quality is transcoded losslessly (same pixels, better entropy coding; +0.4% vs mozjpeg's `jpegtran`).
- **Formats and corners others skip.** Arithmetic-coded JPEG (about 15% smaller than the original on photographs, though browsers cannot open it, so it is opt-in), 12-bit, lossless and hierarchical JPEG, CMYK/YCCK, 16-bit PNG, APNG, animated GIF/WebP/AVIF in any direction, and an AV1 decoder bit-exact with dav1d.
- **Aims at a quality, not a number.** `--target-ssim`, `--target-ssimulacra2`, `--target-psnr`, `--target-ms-ssim` and `--target-size` search for the cheapest setting that meets your goal, per image.
- **Safe by construction.** The output is never larger than the input when the format is kept, files are written atomically, existing outputs are not overwritten without `--force`, metadata is handled deliberately (ICC kept, EXIF/XMP opt-in), and every decoder and encoder path is mutation-tested with truncated, bit-flipped and corrupt input on every test run (more than 440 tests; deterministic mutation testing, not coverage-guided fuzzing).
- **Built for batches and pipelines.** Parallel over all cores, directory trees with include/exclude globs and name templates, `--in-place`, `--dry-run`, JSON reports, stdin/stdout pipes, memory limits, and `rc update` with checksum verification.

**Benchmark at a glance** (details, caveats and the methodology in [Head-to-head](#head-to-head-with-established-tools))

| Area | Against | Result |
|---|---|---|
| Lossless PNG, `-l 9` | `oxipng -o2` / `-o4` | **−1.2% / −0.6%** total size; smaller on 11 of 14 files |
| Lossless PNG, default | `oxipng -o2` | Same size, 1.8× its CPU |
| Lossless PNG, `-l 9` | zopflipng, `oxipng -o max -Z` | Within ~2%, **~20× faster** |
| Lossy PNG (`--lossy`) | pngquant | **4% smaller** at similar quality (pngquant + oxipng is 6% smaller still) |
| Lossy JPEG | mozjpeg 4.0.3, equal SSIMULACRA2 | **−0.8% to −1.2%** mean at scores 55-85 (worst file +3.5%, on a 10-image corpus) |
| Lossless JPEG | mozjpeg `jpegtran` | +0.4% (beats libjpeg-turbo's `jpegtran` and jpegoptim) |
| WebP lossless | `cwebp -lossless` | +2% at ~10× the CPU (default effort is now 5: same size as 7 at about half the time, see the WebP note) |
| WebP lossy | `cwebp -m 6` | +6-12% at ~3× the CPU (since tuned: about 1% smaller on smooth wallpapers, unproven on photographs, see the WebP note) |
| AVIF | libaom speed 6 / speed 4 | +3-4% / +9-12%, ~30× slower than speed 6 |
| GIF kept as GIF | gifsicle | −0.9% vs −4.5%; `--lossy` −5% vs `--lossy=80` −32% |
| GIF → WebP | gif2webp | Lossless −23% vs +10% (better); lossy ~4% larger |

**Where it is weaker, plainly:** AVIF is behind the dedicated encoders (and lossy WebP and GIF `--lossy` were behind in the benchmark run behind the table; both have since been reworked, see the WebP and GIF notes under *Head-to-head*), and the AVIF encoder is far slower than libaom's fast presets. The PNG, WebP, AVIF and GIF rows come from the project's benchmark run and were not re-run when the JPEG encoder was retuned; the JPEG row was re-measured with `tools/bench_jpeg.py` on a small corpus of smooth desktop images, so re-run it on your own photos (it takes any folder via `--src`).

## Install

```
curl -fsSL https://raw.githubusercontent.com/coder-11125/rust-compress/main/install.sh | sh
```

Installs the `rc` binary for Linux or macOS (x86_64 or arm64) from the latest GitHub release. It goes to `/usr/local/bin` when that directory is writable, otherwise to `~/.local/bin`. To pin a release or choose a directory:

```
curl -fsSL https://raw.githubusercontent.com/coder-11125/rust-compress/main/install.sh | RC_VERSION=v0.1.0 RC_INSTALL_DIR="$HOME/bin" sh
```

Windows users can download the `.zip` asset from the [releases page](https://github.com/coder-11125/rust-compress/releases). To build from source instead, see [Build](#build).

### Updating

```
rc update            # shows "Updating rc 0.1.0 -> 0.1.1", downloads, verifies and replaces the binary
rc update --check    # only report whether a newer release exists
rc update --to v0.1.0   # install a specific tag (older ones too); --force reinstalls the current one
```

The newest release is looked up on GitHub, the archive's SHA-256 is checked against the published `.sha256` file, and the new binary must run (`--version`) before it replaces the old one, which is swapped atomically. It needs `curl` and `tar` on `PATH` (both ship with macOS, Linux and Windows 10+). If the binary lives somewhere you cannot write (such as `/usr/local/bin`), re-run with `sudo`. A file literally named `update` has to be passed as `./update`.

## Quick start

```
rc photo.jpg                 # -> photo.min.jpg next to the original
rc ./images -r -o ./out      # whole tree, mirrored into ./out
rc shot.png --lossy          # palette quantization: usually 50-70% smaller
rc *.png -l 9 --in-place     # squeeze harder (slow), only replace if smaller
rc logo.png -f jpeg -q 85    # convert while compressing
rc photo.jpg --target-ssim 0.95   # lowest quality that still scores SSIM >= 0.95
rc photo.jpg --target-ssimulacra2 80   # lowest quality that still scores SSIMULACRA 2 >= 80 (perceptual)
rc shot.png -f webp          # lossless WebP: ~20% smaller than the optimised PNG
rc big.jpg --resize 1920x    # shrink to fit 1920 px wide (never enlarges)
rc photo.jpg --keep-metadata # keep EXIF / XMP / comments
cat in.png | rc - > out.png  # stdin to stdout
```

## What it does

**JPEG** — works in the DCT domain, so most files are never decoded to pixels:
- Sources at or below the target quality (`-q`, default 80) are *transcoded losslessly*: same pixels, better entropy coding. The progressive scan layout is searched (144 candidate layouts for colour images, 24 for gray: DC and AC successive-approximation depths and AC band splits at coefficients 2, 5 and 8, costed from symbol statistics) and compared against baseline; the smallest wins. A split at coefficient 2, which gives the lowest frequencies their own Huffman table, is what mozjpeg's default scan script uses and cut the luma scan's bytes by about 12% on one smooth test image.
- Higher-quality sources are re-quantized directly on the coefficients (never to a finer table than the source had) with **trellis quantization**: a per-block dynamic program that trades distortion against the bit cost of each run/size symbol, using Huffman code lengths measured on a first pass (`--no-trellis` to skip).
- **Perceptually tuned quality scale.** `-q` maps to a flatter base quantization table than the JPEG standard's Annex K tables (the same one mozjpeg writes by default, for luma and chroma alike, clamped to 255 so output stays baseline-compatible), and the trellis weights each coefficient's error by `(median step / step)`, so errors in finely quantized low frequencies count more, with a trellis strength (κ = 0.02) tuned for SSIMULACRA2 and not for MSE. The tuning target was size at equal **SSIMULACRA2** against mozjpeg 4.0.3 (`tools/bench_jpeg.py`): with the Annex K tables the encoder was 15-19% larger than mozjpeg at equal SSIMULACRA2, the table change took that to 3-5%, and a trellis twice as strong as the old default ended up a further 5-6 percentage points behind. The trade-off is **PSNR**: at the same `-q` the picture scores lower in PSNR than with the Annex K tables (about 1.4 dB at q95 on the test fixture), so a q value no longer gives the same PSNR as libjpeg's. A source whose quantization tables are already at or coarser than the target is still transcoded losslessly.
- `--target-ssim S` replaces the guess-a-quality step: it bisects for the lowest quality whose reconstruction still scores at least SSIM `S` against the original (mean 8×8-window SSIM over all channels). The needed quality varies enormously between images — at 0.95 the sample files needed anywhere from q17 to q85 — so this saves bytes on easy images and protects hard ones.
- Pixels are only touched when something must change: chroma subsampling (4:4:4 → 4:2:0 below q90, or `--subsampling 444|422|420`), EXIF orientation (baked into the pixels because EXIF is stripped), or PNG → JPEG conversion.
- **Arithmetic coding** (SOF9 sequential, SOF10 progressive, with restart intervals and `DAC` conditioning) is read and written. A source that is arithmetic-coded stays arithmetic-coded (`--arithmetic auto`, the default), so the result opens wherever the input did; `--arithmetic on` converts any JPEG, `off` converts back to Huffman for compatibility. The coefficients are untouched, so the picture is identical. On five 5120×2880 photographs (55 MB of JPEG) the arithmetic files are 15.1% smaller than the originals and 6.0% smaller (2.0-8.4%) than this tool's own optimized-Huffman output, at about the same encoding CPU time. **The catch:** browsers and many programs cannot open arithmetic-coded JPEG at all (libjpeg and libjpeg-turbo can), which is why it is never switched on by itself for Huffman sources.
- **12-bit** JPEG (SOF1/SOF2 with 12-bit samples; Huffman or arithmetic; gray, colour, CMYK and YCCK) is decoded. There is no DCT-domain path for it, so it is read into a 16-bit image: `-f png` keeps the full precision, `-f jpeg` and the other 8-bit formats reduce it to 8 bits (the report says so).
- **Lossless JPEG** (SOF3 Huffman, and SOF11 arithmetic; 2 to 16 bits per sample; gray, colour and CMYK; all seven predictors, point transform, restart intervals, subsampled components) is decoded exactly, and a lossless source stays lossless: the samples are recoded without changing one, trying all seven predictors in parallel and costing each with real optimized Huffman tables. On nine lossless files made by libjpeg-turbo from photographs that is 4.2% smaller in total (0% to 15.4%; the large gains are sources written with a poor predictor), and libjpeg-turbo decodes every recoded file to the original samples. `-f png` keeps up to 16 bits; `--lossy` (or `--resize`) re-encodes as an ordinary lossy JPEG. Lossless JPEG is common in scientific, medical and raw-camera data. Arithmetic-coded lossless (SOF11) follows Annex H of the standard (a 25-context model over the left and upper differences) and is checked against the JPEG committee's reference libjpeg (24 files: 2 to 16 bits, gray, RGB and CMYK, with and without restart intervals, all decoded identically); libjpeg-turbo is built without it, and `--arithmetic on` is the only way this tool writes it.
- **Hierarchical JPEG** (the `DHP`/`EXP` frame sequences of Annex J: a small first frame, then differential frames that code the difference to the upsampled reconstruction so far) is decoded: DCT frames (SOF0-2, SOF9-10) followed by differential DCT (SOF5/6, SOF13/14) and differential lossless (SOF7/15) frames, or an entirely lossless pyramid, at 8 to 16 bits, with the standard's `(Ra + Rb) / 2` truncating bilinear upsampling. The result is flattened to an ordinary picture, so `-f png` and the other targets work; kept as a JPEG it is left unchanged unless `--lossy` asks for an ordinary lossy re-encode. It is the least-travelled corner of JPEG and almost nothing reads it (browsers and libjpeg-turbo do not), so it is checked against the JPEG committee's reference software (Richter's libjpeg, through `pylibjpeg-libjpeg`) instead: more than a hundred streams from this crate's pyramid writer (`rc_jpeg::hierarchical::write_hierarchical`, which builds each differential frame against the decoder's own reconstruction) decode identically there. Lossless pyramids, Huffman and arithmetic, gray and colour, expanding in either or both directions or refining at the same size, reproduce their source exactly in both decoders; DCT ones agree to the IDCT's rounding (1-2 levels). That comparison found three real defects, all fixed: the two-dimensional upsampling is one truncating average of four neighbours, not two truncating steps (T.81's "horizontal, then vertical" admits both readings); reconstructions are kept unclamped between frames, so a lossless frame can correct a DCT frame's overshoot at a hard edge; and each frame is dequantized with the tables in force when it started, not the last ones defined. Two differences remain by design: the reference accepts a `DHP` only at 8 or 12 bits although T.81 allows 2-16 for lossless (this decoder reads 16-bit hierarchical files, which only it can check), and it insists the largest frame matches the `DHP` size (this decoder decodes what is there).

**PNG** — lossless by default, bit-exact pixels:
- Reduces the representation: drops unused alpha, RGB → gray, ≤256 colors → palette, 1/2/4-bit depths, 16 → 8-bit when lossless.
- Fully transparent pixels get their invisible color rewritten to whatever compresses best (zeroed for palettes, copied from the left/above neighbor for filters); `--keep-hidden-pixels` preserves the exact bytes. Palette order is chosen among four orderings by trial compression.
- Alpha that is only fully transparent or opaque can drop the alpha channel entirely: gray/RGB pixels plus a `tRNS` colour key for the transparent value (the hidden colours are still rewritten, so `--keep-hidden-pixels` keeps the alpha channel).
- Chooses the scanline filter strategy by trial-compressing a sample of the image with a real LZ77 level (the usual sum-of-abs heuristic is often wrong for screenshots/graphics). Candidates: the five fixed filters, per-row min-sum, entropy and *distinct-bigram* heuristics, and at `-l 9` a **brute-force search** that picks each row's filter by actually compressing it together with the preceding rows (skipped above 48 MB).
- Own DEFLATE: hash-chain LZ77 with lazy matching (levels 1-6, compressed in parallel segments) and Zopfli-style optimal parsing (levels 7-9). The optimal parser finds matches with a **binary-tree match finder** (every match length at its nearest distance, no chain-length cliff), then runs a shortest path over them with an iteratively refined cost model, and adaptive block splitting. The refinement is started from two seeds — the longest-match parse and a parse priced from the data's own byte histogram (which avoids a slow drift on noisy data, where a greedy seed is full of coincidental short matches) — and the cheaper result wins.
- `--lossy` quantizes to a ≤256-color palette: exact color histogram, median cut, k-means refinement, and edge-aware Floyd-Steinberg dithering (`--colors N`, `--dither 0..1`).

**WebP (lossy)** — a from-scratch VP8 key-frame encoder and decoder (`-f webp --lossy -q N`):
- The decoder is bit-exact with libwebp (planes and RGB output, including the "fancy" chroma upsampling, loop filters, segments, multiple partitions and `ALPH`).
- The encoder does rate-distortion mode decisions (16x16, 4x4 and chroma intra modes), quantizes with deadzone biasing, tunes the token probabilities from first-pass statistics, and searches the loop-filter level. Its streams decode to the identical picture in libwebp. In the project's earlier head-to-head benchmark it was 6-12% larger than `cwebp -m 6` at about 3x the CPU. It has since been tuned for SSIMULACRA2 rather than squared error (a loop filter at half the MSE-optimal strength with header sharpness 6, cheaper chroma decisions): on 10 smooth desktop wallpapers that took it from about 5% larger than libwebp method 6 to about 1% smaller at equal SSIMULACRA2 65-75 (see the WebP note under *Head-to-head*). That is a metric result on smooth images, not yet checked on real photographs, where a weaker loop filter may behave differently.
- Transparency is stored losslessly in an `ALPH` chunk (best of four prediction filters, coded with the VP8L encoder); hidden pixels are smoothed to be cheap. Lossy WebP inputs are only re-encoded with `--lossy`; otherwise they are left untouched.

**AVIF** — a from-scratch AV1 still-picture decoder and intra encoder inside a HEIF container (`-f avif -q N`):
- The decoder implements the AV1 intra-frame toolset: all partition types, intra modes (directional with edge filtering and upsampling, smooth, Paeth, filter-intra, chroma-from-luma, palette), every transform size and type, segmentation, delta-q/lf, quantizer matrices, the deblocking filter, CDEF, loop restoration (Wiener and self-guided), super-resolution and film grain, in 8/10/12 bits and 4:0:0/4:2:0/4:2:2/4:4:4. Output matches dav1d bit for bit on streams from libaom, SVT-AV1, rav1e and ffmpeg, and libavif/Pillow pixel for pixel (±1 from chroma upsampling). Containers: single images, grids, alpha auxiliary images (also premultiplied), `clap`/`irot`/`imir`, `colr` nclx and ICC, EXIF and XMP. **Inter prediction** — reference slots with CDF and segmentation-map carry-over, single and compound references (average, distance, wedge, difference-weighted), inter-intra, OBMC, local and global warped motion, temporal motion vectors, scaled references and `show_existing_frame` — decodes libaom streams (all tools on, 8/10-bit, 4:0:0/4:2:0/4:4:4, tiles, odd sizes) bit for bit like dav1d; scaled references, 4:2:2 and film grain on inter frames are implemented but not cross-checked; animated AVIFs with inter frames (libavif, ffmpeg) decode too. **Intra block copy** (`allow_intrabc`, with its displacement vector prediction, variable transform trees and inter transform sets) is decoded bit for bit like dav1d. Bit depths above 8 and 4:2:2 are implemented but have had less cross-checking than 8-bit 4:2:0 / 4:4:4.
- The encoder is lossy only. It searches partitions, intra modes with angle deltas, chroma-from-luma and transform sizes per superblock by rate-distortion cost using the real adaptive entropy-coder state, then picks the deblocking level and CDEF strengths by measured error, tries **intra block copy** for pictures with repeated content (screenshots, text, UI: a hash of every 8x8 window finds exact duplicates, and the copy variant is kept only if it wins in rate-distortion terms, since block copy frames cannot use the loop filters), tries **palette mode** (exact or clustered per-block colour lists with a colour index map) for pictures with many flat few-colour blocks, keeping it only when it wins in rate-distortion terms, and chooses a loop-restoration filter (Wiener or self-guided) per unit. Restoration parameters live inside the tile data, so the tiles are written a second time from the decisions already made. Output is 8-bit or 10-bit (`--bit-depth auto|8|10`; auto keeps 16-bit sources at 10 bits) 4:2:0 (4:4:4 at `-q 90` and up, or `--subsampling 444`) or monochrome for gray images; transparency is a monochrome auxiliary image. ICC, EXIF (with `--keep-metadata`) and XMP are stored. Streams decode identically in dav1d, libaom and Pillow.
- AVIF inputs are left untouched unless `--lossy` is given, since re-encoding only adds generation loss.
- The encoder is considerably slower than the JPEG/WebP ones. An earlier PSNR-only comparison against libaom (`cpu-used 4`) on eight photos and screenshots found it −6.5% smaller on average (from −26% to +4.3% per image, both decoded through the same dav1d/ffmpeg path), but the current head-to-head benchmark (see below) puts it 3-4% larger than libaom speed 6 and 9-12% larger than speed 4, and about 30x slower than speed 6; treat that table as the current figure. Palette, filter-intra and compound/OBMC/warp tools are not implemented in the encoder. Inter frames (one reference, block matching with quarter-sample vectors) are used for animations.

**WebP (lossless)** — full VP8L encoder and decoder:
- Encoder tries a palette transform (bundling up to 8 pixels per symbol) or subtract-green + per-tile predictor + cross-color decorrelation, ranks the pipelines with a cheap estimate and fully encodes the best ones.
- **Effort levels** (`-l`, default **5** for lossless WebP): level 5 was as small as 7 (slightly smaller on three wallpapers) for about half the time; 6 is larger than both and about as slow as 7, so there is no reason to use it; 7-9 are the slow, thorough levels; 0-4 are a fast tier that fits the transforms on sampled pixels and uses a cheaper parse (on three wallpapers, level 4 was 0.75% larger than 7 for 6-8x less wall time). The fast levels try fewer candidate pipelines, so graphics and screenshots can come out noticeably larger at them than at 5.
- LZ77 with the format's 2-D distance codes, an optimal (cost-model driven) parse, a colour cache sized by trial, and **entropy-clustered prefix-code groups**: image tiles with similar statistics share one set of codes.
- 8-bit only: 16-bit inputs are reduced to 8-bit (the report note says so). Images are limited to 16384×16384 by the format. Lossy WebP (VP8) is decoded for conversions or an explicit `--lossy` re-encode; when the format is kept it is left untouched, since recompressing it would only add generation loss. Animated WebP is treated the same way: left untouched unless `--lossy` asks for a re-encode.

**Quality targets** — `--target-ssim`, `--target-ssimulacra2`, `--target-psnr`, `--target-ms-ssim` and `--target-size` bisect the cheapest quality (or palette size for PNG) that still meets the goal, and the report note shows what was chosen and the score it reached. JPEG takes `--target-ssim` and `--target-ssimulacra2`; lossy WebP, AVIF and `--lossy` PNG take all of them.

**SSIMULACRA 2** (`--target-ssimulacra2 SCORE`) is a perceptual metric by Jon Sneyers (Cloudinary), implemented here from scratch in `core/src/ssimulacra2.rs`. It compares the images in the XYB colour space of JPEG XL at six scales, looking at structural similarity and at how much edge energy the distortion adds (ringing, blocking) or removes (blur). 100 means identical, about 90 is visually indistinguishable, 70 good, 50 clearly degraded, and the score is not clamped, so severe damage goes negative. Use it when you want the *look* to stay constant across very different images, which SSIM does not capture well. Notes:
- It costs time: about 35 G instructions for a 14.7-megapixel pair (roughly a second on four idle cores), once per search step, against a fraction of that for SSIM. Images must be at least 8x8 pixels; alpha is composited onto mid-grey.
- **How closely it matches the reference.** It was checked against the py-ssimulacra2 Python port (the only implementation available here, not the official libjxl tool): 54 synthetic pairs from 8x8 to 513x385 differ by at most 7e-6 points, and 18 pairs of 1280-pixel photographs by at most 1e-6. The port blurs with a truncated Gaussian and, through an accidental transposition in its code, treats the left and right image edges as zero and the top and bottom as mirrored; `rc` reproduces that, so its scores can differ from the official tool, which blurs with a recursive filter. The two border rules differ by up to about 5 points on a 640-pixel image at low quality and by much less on large images, because only pixels near the edges are affected. This has not been measured against the official tool.
- **JPEG is scored on rc's own decode.** Another decoder rounds slightly differently (libjpeg-turbo differed on 7-15% of pixels by 1-2 levels on one test photograph), and SSIMULACRA 2 is sensitive enough at high quality that this moved the score of the same file by 1.2-1.9 points around targets of 75-90 (lossy WebP, whose decoder is bit-exact with libwebp, agreed to three decimals). So a JPEG written for `--target-ssimulacra2 80` may score a point or two lower in other viewers.

**Other formats**
- **GIF** reads and writes still and animated files (palettes, interlacing, transparency, disposal). A GIF kept as a GIF is re-coded without changing a pixel when every frame fits a palette: frames that share one palette use a single global table, while frames whose palettes differ get a local color table each, still with each frame stored as the rectangle that changed and unchanged pixels transparent; the original is kept when that is not smaller. A local table holds only the colors of the rectangle its frame actually draws, so a frame may use all 256 colors across its canvas and still be coded exactly when what it changes is small, and its LZW codes get narrower. A GIF whose changed rectangle needs more colors than fit (leaving room for the transparency/unchanged entry) would have to be re-quantized, so it is left unchanged unless `--lossy` (or `--resize`) is given. **Lossy GIF** (`--gif-lossy N`, 0-200 on gifsicle's `--lossy` scale; `--lossy` alone means 80): a pixel within the budget of what the previous frame already shows is left unchanged, and while the LZW coder runs a pixel may take a nearby palette color (or "unchanged") when that lets it extend a dictionary match. Every pixel stays within a color distance (RGB vector length) of 0.4 x N of its source color, a substitution is what the next frame is compared with so errors do not accumulate, and level 0 is exact. It applies to animations that are coded exactly otherwise, with a shared palette or with per-frame ones. Converting *to* GIF (`-f gif`, from any format) quantizes to `--colors` entries (default 256) with one palette for the whole animation, no dithering for animations (the noise would change from frame to frame) and `--dither` for stills; GIF transparency is on/off, so pixels below 50% alpha become transparent.
- CMYK and YCCK JPEGs stay four-channel and are re-quantized in the DCT domain like any other JPEG (never to a finer table than the source had, so a source at or below `-q` is transcoded losslessly with identical coefficients; the Adobe colour flag is kept). All four channels of plain CMYK use the luma table, since each is an ink amount; in YCCK, Y and K use it and the two colour-difference channels use the chroma table. Converting or resizing them goes through RGB.
- Animated PNGs kept as PNG are passed through unchanged instead of failing the batch; other animated inputs become an animated PNG with `-f png` (APNG: lossless RGB/RGBA full-canvas frames, `--lossy` quantizes them, `--keep-hidden-pixels` preserves invisible colours).
- **Animations** — animated GIF, APNG, animated WebP and animated AVIF inputs convert to an animated **PNG/APNG**, **AVIF**, **WebP** or **GIF** (`-f png|avif|webp|gif`; `--lossy` re-encodes an animated WebP or AVIF input). Frames keep the files' disposal and blend rules, per-frame durations and looping (a finite repeat count is played forever). **PNG/APNG**: every frame is stored as a complete canvas, so there are no dirty rectangles yet. **AVIF**: alpha as a second track, the first frame a key frame and the others inter frames predicted from the previous output (a key frame instead where that is smaller, e.g. at scene changes); on camera-pan style content 6–8 times smaller than key frames at the same PSNR; decodes in libavif and dav1d. **WebP**: lossless by default, lossy VP8 frames with `--lossy`; after the first frame only the changed rectangle is stored, and each lossless frame is tried both replacing the canvas and blended over it with the unchanged pixels transparent, keeping the smaller (lossy frames are measured against the canvas as a decoder reconstructs it, so errors do not build up). Decodes in libwebp (Pillow), pixel for pixel in the lossless case. **GIF**: see above.

**Metadata** — only color information survives by default (ICC profile, PNG color chunks). `--keep-metadata` also keeps EXIF, XMP, JPEG comments / Photoshop blocks and PNG text, time and pixel-density chunks. EXIF and XMP carry across PNG ↔ JPEG ↔ WebP conversions; when an orientation is baked into the pixels the EXIF orientation is reset, its recorded dimensions are corrected and a now-stale thumbnail is dropped.

**Resizing** — `--resize WxH | Wx | xH | P%` shrinks to fit (aspect ratio kept, never enlarges) with a Lanczos-3 filter on premultiplied alpha, for 8- and 16-bit images. The filter averages in linear light (the sRGB curve is undone before and applied after), so a fine black-and-white pattern comes out as the mid-gray the eye sees (188) rather than 128; this assumes the sRGB transfer curve. `--resize-encoded` filters the stored values instead, as most tools do: it is a little faster and matches Pillow's Lanczos to within ~0.1 level on average (the default differs from Pillow by about 2 levels on a photo-like image). Because the result must differ from the original, the never-grow rule does not apply.

**Pipes** — `-` as the input reads one image from standard input; `-o -` (the default for stdin) writes to standard output, with all reporting moved to standard error.

**Safety** — when a file keeps its format, the output is never larger than its input (the original bytes are kept instead; converting to another format or resizing can make a file bigger, and the report then says "larger"), files are written atomically, existing outputs are not overwritten without `--force`, ICC profiles and PNG color chunks are kept (`--strip-icc` to drop them), animations are never flattened, and decoders are hardened against truncated/corrupt/hostile input (see *Testing*).

## Usage

```
rc [OPTIONS] <INPUT>...

  -r, --recursive         descend into subdirectories
      --include <GLOB>    only process directory-walk files matching GLOB (repeatable)
      --exclude <GLOB>    skip directory-walk files matching GLOB (repeatable)
      --min-suffix <S>    default output suffix `<name>.<S>.<ext>` (default min)
      --name-template <T> output path template when no -o/--in-place ({root},{dir},{rel},{rel_stem},{stem},{name},{ext},{suffix})
  -o, --output <PATH>     output file (one input file) or directory (mirrors the input tree)
      --in-place          replace originals, only when the result is smaller (keeps permissions/mtime)
  -q, --quality <1-100>   JPEG quality ceiling (default 80)
  -f, --format <FMT>      keep | png | jpeg | webp | avif | gif (default keep)
      --subsampling <S>   auto | 444 | 422 | 420
      --arithmetic <A>    JPEG output coding: auto (keep the source's) | on | off
      --bit-depth <D>     AVIF output: auto | 8 | 10 (auto: 10 for 16-bit sources)
      --lossy             PNG palette quantization, lossy WebP with -f webp (animations too), re-encode an AVIF or animated WebP input, a lossless JPEG as lossy JPEG, re-quantize a GIF that cannot be represented exactly, or code a GIF lossily (level 80, see --gif-lossy) (-q = quality)
      --colors <2-256>    with --lossy (default 256)
      --dither <0.0-1.0>  with --lossy (default 0.5)
      --gif-lossy <0-200> GIF output: lossiness on gifsicle's --lossy scale (default 0; 80 with --lossy)
  -l, --level <0-9>       PNG / WebP / AVIF effort (default 7; lossless WebP 5; PNG: 1-6 fast, 7-9 optimal parsing, 9 slowest)
      --no-trellis        JPEG: skip rate-distortion quantization (faster, larger)
      --target-ssim <S>   smallest quality with SSIM >= S (0-1; 0.95 small, 0.97 good, 0.99 near-lossless)
      --target-psnr <DB>  lossy WebP/AVIF and --lossy PNG: smallest quality with PSNR >= DB
      --target-ms-ssim <S>  lossy WebP/AVIF and --lossy PNG: smallest quality with MS-SSIM >= S
      --target-ssimulacra2 <SCORE>  JPEG, lossy WebP/AVIF and --lossy PNG: smallest quality with SSIMULACRA 2 >= SCORE (0-100)
      --target-size <B>   lossy WebP/AVIF and --lossy PNG: highest quality whose output fits B bytes
      --keep-hidden-pixels  PNG: keep exact color of fully transparent pixels
      --strip-icc         drop ICC profiles / PNG color chunks
      --keep-metadata     keep EXIF, XMP, comments, PNG text/time/density
      --resize <SIZE>     shrink to fit: WxH, Wx, xH or P% (never enlarges)
      --resize-encoded    with --resize: filter the stored values instead of linear light
      --background <HEX>  color for flattening transparency (default ffffff)
      --max-pixels <N>    reject images with more than N pixels (default and maximum 150000000)
      --max-memory <MIB>  soft limit on the memory images processed at once may need (default 4096); the rest wait
  -j, --jobs <N>          worker threads, 1-1024 (default: all cores)
      --dry-run           report savings without writing
      --force             overwrite existing outputs
      --quiet             only errors and the summary line
      --json              print the batch report as one JSON object
```

`<INPUT>` may be `-` (standard input) and `-o -` writes standard output. Without `-o`/`--in-place`, `photo.jpg` becomes `photo.min.jpg` (`--min-suffix` changes `min`, `--name-template` replaces the whole path pattern); files already ending in `.<suffix>.` are skipped when walking directories, and `--include`/`--exclude` filter directory walks by file name or path relative to each input directory (`*`/`?` stay within a component, `**` crosses directories). `--json` prints per-file results plus the summary as JSON (to stderr when the image itself goes to stdout). The exit status is non-zero if any file failed; the rest of the batch still runs.

## Head-to-head with established tools

Results of the project's current benchmark against the established tools; **PNG, WebP, AVIF and GIF rows are carried over from that benchmark run and were not re-run when the JPEG encoder changed.** Only the *Lossy JPEG* row was re-measured with `tools/bench_jpeg.py` (rc vs mozjpeg 4.0.3 through `jpeglib`, sizes interpolated at equal SSIMULACRA2 over a sweep of quality settings, ICC stripped on our side so profile bytes do not count against us; 10 macOS desktop-wallpaper images downscaled to 1280 px, which are smoother than typical photographs, so treat it as indicative and re-run it on your own images with `--src`). Sizes are exact; timings are indicative CPU times and vary with the machine.

| Area | Result | Verdict |
|---|---|---|
| Lossless PNG, `-l 9` | −1.2% total vs `oxipng -o2` (smaller on 11 of 14 files), −0.6% vs `-o4`; within about 2% of `oxipng -o max -Z` and zopflipng, which are 20× slower; beats optipng/advpng/pngcrush by 6-9% | **Best area** |
| Lossless PNG, default | Same size as `oxipng -o2`, 1.8× its CPU | Parity |
| Lossy PNG (`--lossy`) | 4% smaller than pngquant at similar quality; pngquant + oxipng is 6% smaller still | Competitive |
| Lossy JPEG | **−0.8% / −1.1% / −1.2% / −1.0% mean size vs mozjpeg at equal SSIMULACRA2 55 / 65 / 75 / 85** (worst single file +3.5% / +3.1% / +1.7% / −0.2%; the 85 row has only 2 images). Before the perceptual tuning this was +16-19% at 55-75. Lower PSNR than libjpeg's tables at the same `-q` (see *What it does*) | Parity on this small corpus (worst files ~3% behind at low quality) |
| Lossless JPEG (`-q 100`) | +0.4% vs mozjpeg's jpegtran (up to +3.5% on one file); beats libjpeg-turbo jpegtran and jpegoptim | Good |
| WebP lossless | +2% vs `cwebp -lossless` at about 10× the CPU | Behind |
| WebP lossy | +6-12% vs `cwebp -m 6`, about 3× the CPU | Behind |
| AVIF | +3-4% vs libaom speed 6 (+9-12% vs speed 4), about 30× slower than speed 6 | Behind, slow |
| GIF (kept as GIF) | −0.9% vs gifsicle −4.5%; `--lossy` gives −5% vs `gifsicle --lossy=80` −32% | Behind |
| GIF → WebP | Lossless: −23% vs gif2webp +10% (better). Lossy: about 4% larger | Mixed |

Two measurements from earlier runs that the table above does not repeat: `-l 6` PNG is about 11% larger than `oxipng -o2` in total at a fraction of the CPU (the fast levels use hash-chain LZ77 only), and PNGs with hidden transparent pixels can be orders of magnitude smaller than what tools that keep the invisible RGB (oxipng by default) produce: 580 KB → 4 KB on a noisy-under-alpha test image.

**GIF, re-measured after the table above was produced.** The GIF encoder changed after that benchmark run (compact per-frame palettes and lossy coding), so the two GIF rows are from before. With `tools/bench_gif.py` against gifsicle 1.96 (`-O3 --lossy`) on **four synthetic animations that I generated** (a camera pan over a photo, a sprite over a photo, a typing UI and an ffmpeg test pattern; quality is the mean SSIMULACRA2 of 8 sampled frames against the input GIF): lossless re-coding of a GIF whose frames each carry their own palette went from "left unchanged" to −49% (211 KB → 108 KB, pixel-exact, +0.7% vs gifsicle `-O3`); at equal quality rc was 10-32% smaller on the pan, 34-80% smaller on the sprite, 2.7% smaller on the UI animation and 0.7-2.9% larger on the test pattern; lossless output is still about 4-6% above gifsicle `-O3` on the UI animation and the test pattern. The big wins come from animations whose frames are dithered independently, where near-identical pixels become "unchanged"; four synthetic files is a very small sample, and the real-world corpus behind the table has not been re-run.

**WebP, re-measured after the table above was produced.** The WebP encoders changed after that benchmark run, so the two WebP rows are from before. With `tools/bench_webp.py` against libwebp method 6 (through Pillow, not a standalone `cwebp`) on desktop wallpapers, which are smooth images rather than typical photographs or screenshots:
- *Lossless* (3 images at 1280 px, all pixel-exact): effort 5 produced 3,007,018 bytes against 3,007,810 at effort 7 (-0.03%) in about half the time (5-10 s against 9-17 s per file); effort 6 was +0.17% and as slow as 7; effort 4 was +0.75% at 1.4-2.9 s per file. Against libwebp method 6, levels 5 and 7 were +3.3% in total and level 4 +4.1%. Wall-clock times come from a busy machine, so read them as indicative. This is not comparable with the +2% in the table, which used `cwebp -lossless` at its default speed.
- *Lossy* (10 images, size at equal SSIMULACRA2, positive means `rc` is larger): at score 65 the mean went from +4.8% to -0.8% (worst file +11.7% to +6.5%) and at score 75 from +5.1% to -1.4% (worst file +22.9% to +9.5%). The 55 and 85 rows have only 1-5 images that both encoders reach and say nothing. The gain comes mostly from a weaker loop filter, a choice SSIMULACRA2 rewards because it penalizes blur, so it needs checking on real photographs and by eye. A rate-distortion trellis and token-probability tuning, which helped JPEG, were not tried.

So PNG is the strong area, JPEG is at parity with mozjpeg on perceptual quality (on the small test corpus above), and AVIF is still behind and much slower. Lossy and lossless WebP and GIF have improved since the benchmark run (see the notes above). Against weaker baselines it is far ahead (e.g. 14-28% smaller than PIL's `optimize=True` PNGs).

What it offers beyond ratio: no C dependencies, one binary for all of these formats, a coefficient-domain JPEG pipeline, pipe support, and parallel batch processing.

## Testing

`cargo test` runs the unit tests and integration suites, which cross-check the decoders against the `image` crate (and, for WebP and CMYK/YCCK, against files made by libwebp and libjpeg-turbo, or decoded by it) and verify encoder output with independent decoders. `cli/tests/hostile.rs` feeds every decoder and encoder path truncated, bit-flipped and header-corrupted variants of every fixture and requires no panic, no hang and (for WebP) an exact re-encode round trip. This is deterministic mutation testing, not coverage-guided fuzzing. For a deeper sweep: `HOSTILE_ROUNDS=20000 cargo test --release -p rust-compress --test hostile`. The SSIMULACRA 2 implementation is checked against the py-ssimulacra2 reference on 54 synthetic image pairs (`fixtures/ssim2`, regenerated deterministically by `tools/ssim2_vectors.py`; `cli/tests/ssimulacra2.rs` requires a match within 1e-4) and against real photographs with `tools/ssim2_parity.py`, which also reports how far apart the two implementations are.

### Benchmarking

One script per format in `tools/`, sharing `tools/benchlib.py`. Each compares `rc` with an established encoder on images you provide, on Linux, macOS and Windows:

| Script | Compares `rc` against | Measures |
|---|---|---|
| `bench_png.py` | oxipng through `pyoxipng` (`--ref-levels 2,4`; 7 = `-o max -Z`), or any command | sizes and time per file with a pixel-exactness check; `--lossy` compares palette sizes at equal SSIMULACRA2 (needs `--ref-cmd`, e.g. pngquant, which no pip package bundles) |
| `bench_jpeg.py` | mozjpeg 4.0.3 through `jpeglib`, or any command | size at equal SSIMULACRA2 |
| `bench_webp.py` | libwebp method 6 through Pillow, or any command | size at equal SSIMULACRA2; `--lossless` compares bytes (both pixel-exact) |
| `bench_avif.py` | libavif through Pillow (`--ref-speed`, default 6), or any command | size at equal SSIMULACRA2 |
| `bench_gif.py` | gifsicle `-O3 --lossy` through `gifsicle-bin`, or any command | size at equal mean-frame SSIMULACRA2 over a lossiness sweep |

```
python3 -m venv env
env/bin/pip install numpy pillow ssimulacra2 jpeglib pyoxipng gifsicle-bin   # install what you need
cargo build --release -p rust-compress
env/bin/python tools/bench_jpeg.py /tmp/bench --src ~/Pictures/photos [--src another/dir] [--limit 0] [--size 0]
env/bin/python tools/bench_png.py  /tmp/bench --src ~/Pictures/screenshots --ref-levels 2,4 --level 9
```

- **Inputs:** `--src` takes files or directories (repeatable, or `RC_BENCH_SRC` with `os.pathsep`-separated paths); `--limit N` caps the count; `--size PX` downscales the longer side (0 keeps full size). The quality benchmarks re-save inputs as lossless PNG first, so use pristine photographs; `bench_png.py` and `bench_gif.py` use your files as they are.
- **Reference encoder:** every script takes `--ref-cmd` with `{in}`, `{out}` and the quality placeholder (`{q}`, `{c}` for PNG palette size, `{l}` for GIF lossiness) to compare against any command line, e.g. `--ref-cmd 'cwebp -q {q} -m 6 {in} -o {out}'`, and `--ref-qs` when the reference's quality scale differs from `rc`'s. Reference results are cached per command.
- **Sweep:** `--qs` (rc's settings) and `--targets` (SSIMULACRA2 scores to compare at); `--rc PATH` picks the binary; `--rc-arg` passes extra options to `rc`.
- **Output:** each file's size difference at every score, then the mean and the worst file; positive means `rc` is larger. A file counts in a score row only if both sweeps reach that score. Outputs are decoded with Pillow for scoring, so your Pillow must read the format (AVIF needs Pillow 11.3+ or `pillow-avif-plugin`).

The *Lossy JPEG* row and the GIF note above come from these scripts; the PNG, WebP and AVIF rows come from the project's earlier benchmark run. A three-image check of lossy WebP against libwebp method 6 with `bench_webp.py` gave +3% to +9% mean size (worst file +22%), in line with the table, but that sample is far too small to quote.

## Build

```
cargo build --release        # binary: target/release/rc
cargo test --workspace        # unit + integration tests (cross-checked against the `image` crate)
cargo clippy --workspace --all-targets -- -D warnings
```

Requires a stable Rust toolchain (edition 2024); `rust-toolchain.toml` pins `stable` with clippy/rustfmt. No C compiler or system libraries are needed.

## Layout

```
tools/      `crosscheck.py` (decoder cross-checks), `ssim2_vectors.py` / `ssim2_parity.py` (SSIMULACRA 2 fixtures and parity against the Python reference) and the benchmarks `bench_png.py`, `bench_jpeg.py`, `bench_webp.py`, `bench_avif.py`, `bench_gif.py` (shared code in `benchlib.py`)
core/       shared types and utilities (rc-core): image and animation types, errors, bit I/O, CRC-32 / Adler-32, canonical Huffman, EXIF editing, Lanczos-3 downscaling, SSIM/PSNR/MS-SSIM/SSIMULACRA 2, color quantizer
deflate/    inflate, LZ77 matchers (hash chains; binary tree for levels 7-9), block encoder, optimal parser (levels 7-9)
png/        decoder (all color types/depths, Adam7, APNG), filters, encoder (reduction + strategy search, APNG writer)
jpeg/       decoder (baseline, progressive, arithmetic, lossless, hierarchical; 2-16 bit), DCT, baseline/progressive/arithmetic/lossless encoders, a hierarchical pyramid writer, trellis quantization, quality search
webp/       RIFF container, VP8L decoder and encoder (transforms, LZ77 + optimal parse, tile-group clustering), VP8 decoder and key-frame encoder, animated WebP writer
gif/        GIF decoder and encoder (still and animated: shared palette, frame differencing, LZW)
avif/       HEIF/ISOBMFF container, AV1 decoder (entropy, intra and inter prediction, transforms, loop filters, CDEF, restoration, film grain) and encoder
cli/        the `rc` binary: argument parsing, input collection, per-file policy (read → decode/transcode → never-grow check → atomic write)
testutil/   fixture loading and reference decoding shared by the test suites
fixtures/   sample files for the tests
```

Each codec is its own crate (`rc-png`, `rc-jpeg`, `rc-webp`, `rc-gif`, `rc-avif`, `rc-deflate`) on top of `rc-core`, with its decoder and encoder in separate modules and its own `tests/` and `examples/` (developer tools for tuning ratio and speed); only `cli` depends on all of them. The AV1 tile syntax is written once and shared by the AVIF decoder and encoder, so those two stay together inside `avif/`.

## Limitations

- Formats: PNG, JPEG, WebP (lossless and lossy), AVIF and GIF in and out. Animations are written as PNG/APNG, AVIF, WebP or GIF (APNG frames are complete canvases, not dirty rectangles). JPEG: 1, 3 or 4 components, Huffman or arithmetic coding, 8 or 12 bits, plus lossless (SOF3 and SOF11, 2-16 bits) and hierarchical; 12-bit input is written back at 8 bits, hierarchical input is only read, and JPEG 2000, JPEG-LS and JPEG XL (different standards) are not supported.
- WebP output is always 8-bit: the WebP format has no higher bit depth. AVIF output is 8-bit unless the source is 16-bit or `--bit-depth 10` is given.
- GIF output is at most 256 colors and its transparency is on/off, both limits of the GIF format; animated GIFs share one palette and are not dithered. A GIF that needs a separate palette per frame is not re-coded losslessly (it is left unchanged, or re-quantized with `--lossy`).
- EXIF/XMP/comments are dropped unless `--keep-metadata` is given. An EXIF block over 64 KB cannot be carried into a JPEG: the format allows one 64 KB segment for it.
- Resizing only shrinks, and its linear-light filtering assumes the sRGB transfer curve (images tagged with a different curve are treated as sRGB; `--resize-encoded` skips the assumption).
- What the cross-checks cover (`tools/crosscheck.py` re-runs them; it needs `pip install numpy pillow imagecodecs pylibjpeg-libjpeg jpeglib` and Chrome for the browser part): arithmetic JPEG (120 files written by IJG libjpeg 7, 8 and 9 and libjpeg-turbo 2.1, sequential and progressive, plus our writer's output, all coefficient-identical in libjpeg and libjpeg-turbo); 12-bit including CMYK (libjpeg-turbo's 12-bit encoder and decoder, within 1-3 levels of 4095 from float versus integer IDCT rounding); lossless JPEG (libjpeg-turbo's encoder and decoder, 12 files and 119 recoded ones, exact); arithmetic lossless and hierarchical (the reference libjpeg, above); linear-light resizing (numpy plus Pillow's float Lanczos, mean difference 0.000); animated GIF and WebP and animated AVIF (Chrome 150's WebCodecs decoder and Pillow, frame by frame: pixel-identical with matching frame durations, including transparency and disposal); and lossy animated WebP against libwebp on panning footage, where this encoder is 5-11% smaller *and* 0.2-1.6 dB higher in PSNR at every quality tried.
- Still unchecked: Safari and Firefox; the Windows `rc update` executable swap (its test is written and the code type-checks and lints for Windows, but it has only run on Unix); lossless JPEG with subsampled components and 16-bit hierarchical JPEG (only our own writer produces them); 12-bit arithmetic-coded JPEG.
- Browsers: Chrome decodes our GIF, animated WebP, animated AVIF and lossless JPEG output, and refuses arithmetic-coded, 12-bit and hierarchical JPEG, as expected (this is why `--arithmetic` stays off for Huffman sources). Chrome skips the smoothing libjpeg applies when it doubles subsampled chroma, so a 4:2:0 JPEG with hard colour edges looks slightly different there than in libjpeg-based viewers; that is its choice, not a property of the file.
- The YCCK test files were built with our own writer (`jpeg/examples/make_ycck.rs`, since no tool here writes YCCK) and checked against libjpeg's decode; a file exported by Photoshop would be a better fixture.
